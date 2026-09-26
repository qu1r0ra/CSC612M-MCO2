import json
import os
import struct
import subprocess
from pathlib import Path

import numpy as np
import pytest

from mco2_oracle import (
    _round_codes_fp32,
    compress_record_fp32,
    decode_record,
    philox_words,
    reference_fp64,
    scale_fp32,
)

ROOT = Path(__file__).resolve().parents[1]
BINARY = ROOT / "build" / ("mco2.exe" if os.name == "nt" else "mco2")
HEADER = struct.Struct("<4sBBHQf")


def _run(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(BINARY), *arguments],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def _write_f32(path: Path, values: np.ndarray) -> None:
    path.write_bytes(np.asarray(values, dtype="<f4").tobytes())


def _compress(
    tmp_path: Path,
    values: np.ndarray,
    *,
    seed: int = 1,
    extra: tuple[str, ...] = (),
) -> bytes:
    input_path = tmp_path / "input.f32"
    output_path = tmp_path / "record.msq"
    _write_f32(input_path, values)
    result = _run(
        "compress",
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        "--seed",
        str(seed),
        *extra,
    )
    assert result.returncode == 0, result.stderr
    return output_path.read_bytes()


def test_compress_matches_prescribed_scale_and_word_record(tmp_path):
    values = np.asarray([-2, -1, 0, 1, 2], dtype=np.float32)
    words = np.asarray([0, 0, 0, 0xFFFFFFFF, 0], dtype="<u4")
    input_path = tmp_path / "input.f32"
    words_path = tmp_path / "words.u32"
    output_path = tmp_path / "record.msq"
    _write_f32(input_path, values)
    words_path.write_bytes(words.tobytes())

    result = _run(
        "compress",
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        "--seed",
        "7",
        "--scale",
        "2",
        "--words",
        str(words_path),
    )

    assert result.returncode == 0, result.stderr
    expected = bytes.fromhex(
        "4d 53 51 31 01 08 00 00 05 00 00 00 00 00 00 00 00 00 00 40 00 3f 7f be fe"
    )
    assert output_path.read_bytes() == expected
    assert output_path.read_bytes() == compress_record_fp32(values, scale=2.0, words=words)


def test_seeded_cli_matches_independent_oracle_and_is_deterministic(tmp_path):
    values = np.random.default_rng(612).normal(size=513).astype(np.float32)
    extra = ("--tensor-id", "7", "--invocation-id", "9")

    first = _compress(tmp_path, values, seed=0x123456789ABCDEF, extra=extra)
    second = _compress(tmp_path, values, seed=0x123456789ABCDEF, extra=extra)

    assert first == second
    assert first == compress_record_fp32(
        values,
        seed=0x123456789ABCDEF,
        tensor_id=7,
        invocation_id=9,
    )


def test_layer_two_scale_and_reconstruction_are_within_documented_fp32_bound(tmp_path):
    values = np.random.default_rng(2026).normal(size=513).astype(np.float32)
    record = _compress(
        tmp_path,
        values,
        seed=0xBADC0FFEE,
        extra=("--tensor-id", "17", "--invocation-id", "23"),
    )
    input_path = tmp_path / "record.msq"
    output_path = tmp_path / "decoded.f32"
    result = _run("decompress", "--input", str(input_path), "--output", str(output_path))

    assert result.returncode == 0, result.stderr
    bits = HEADER.unpack_from(record)
    assert bits[:4] == (b"MSQ1", 1, 8, 0)
    fp32_scale = bits[5]
    expected_fp32_scale = float(scale_fp32(values))
    fp64_scale, fp64_decoded = reference_fp64(
        values,
        seed=0xBADC0FFEE,
        tensor_id=17,
        invocation_id=23,
    )

    block_count = (len(values) + 255) // 256
    operations = 12 + (block_count - 1).bit_length()
    unit_roundoff = 2.0**-24
    epsilon = operations * unit_roundoff / (1 - operations * unit_roundoff)
    epsilon += 4 * len(values) * 2.0**-149 + 2.0**-149 / (2 * fp64_scale)
    assert abs(fp32_scale - fp64_scale) <= fp64_scale * epsilon
    assert fp32_scale == expected_fp32_scale

    decoded = np.frombuffer(output_path.read_bytes(), dtype="<f4").astype(np.float64)
    output_bound = fp64_scale * (2 / 127 + 2 * epsilon / (1 - epsilon) + 5 * unit_roundoff)
    assert np.max(np.abs(decoded - fp64_decoded)) <= output_bound


@pytest.mark.parametrize(
    "values", [np.asarray([], dtype=np.float32), np.zeros(5, dtype=np.float32)]
)
def test_empty_and_all_zero_compress_use_zero_scale(tmp_path, values):
    record = _compress(tmp_path, values)

    assert HEADER.unpack_from(record)[5] == 0.0
    assert record[HEADER.size :] == bytes([127]) * len(values)
    np.testing.assert_array_equal(decode_record(record), values)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("magic", "bad record magic"),
        ("version", "unsupported record version"),
        ("bit-width", "unsupported bit width"),
        ("reserved", "reserved header bytes must be zero"),
        ("count", "element count is too large"),
        ("payload", "payload length does not match element count"),
    ],
)
def test_decompress_rejects_each_minimum_header_error(tmp_path, mutation, message):
    record = bytearray(
        compress_record_fp32(np.asarray([1.0], dtype=np.float32), scale=1.0, words=[0])
    )
    if mutation == "magic":
        record[0] ^= 1
    elif mutation == "version":
        record[4] = 2
    elif mutation == "bit-width":
        record[5] = 2
    elif mutation == "reserved":
        record[6] = 1
    elif mutation == "count":
        record[8:16] = bytes([0xFF]) * 8
    elif mutation == "payload":
        record[8:16] = struct.pack("<Q", 2)

    input_path = tmp_path / "malformed.msq"
    output_path = tmp_path / "decoded.f32"
    input_path.write_bytes(record)
    result = _run("decompress", "--input", str(input_path), "--output", str(output_path))

    assert result.returncode != 0
    assert message in result.stderr


def test_compress_matches_prescribed_scale_and_word_record_4bit(tmp_path):
    values = np.asarray([-2, -1, 0, 1, 2], dtype=np.float32)
    words = np.asarray([0, 0, 0, 0xFFFFFFFF, 0], dtype="<u4")
    input_path = tmp_path / "input.f32"
    words_path = tmp_path / "words.u32"
    output_path = tmp_path / "record.msq"
    _write_f32(input_path, values)
    words_path.write_bytes(words.tobytes())

    result = _run(
        "compress",
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        "--seed",
        "7",
        "--bits",
        "4",
        "--scale",
        "2",
        "--words",
        str(words_path),
    )

    assert result.returncode == 0, result.stderr
    expected = bytes.fromhex("4d 53 51 31 01 04 00 00 05 00 00 00 00 00 00 00 00 00 00 40 30 a7 0e")
    assert output_path.read_bytes() == expected
    assert output_path.read_bytes() == compress_record_fp32(values, bits=4, scale=2.0, words=words)

    decoded_path = tmp_path / "decoded.f32"
    dec_res = _run("decompress", "--input", str(output_path), "--output", str(decoded_path))
    assert dec_res.returncode == 0, dec_res.stderr
    decoded = np.frombuffer(decoded_path.read_bytes(), dtype="<f4")
    assert len(decoded) == 5
    assert decoded[0] == -2.0
    assert decoded[2] == 0.0
    assert decoded[4] == 2.0


def test_layer_two_4bit_scale_and_reconstruction_within_bound(tmp_path):
    values = np.random.default_rng(2026).normal(size=513).astype(np.float32)
    record = _compress(
        tmp_path,
        values,
        seed=0xBADC0FFEE,
        extra=("--bits", "4", "--tensor-id", "17", "--invocation-id", "23"),
    )
    input_path = tmp_path / "record.msq"
    output_path = tmp_path / "decoded.f32"
    result = _run("decompress", "--input", str(input_path), "--output", str(output_path))

    assert result.returncode == 0, result.stderr
    bits = HEADER.unpack_from(record)
    assert bits[:4] == (b"MSQ1", 1, 4, 0)
    fp32_scale = bits[5]
    fp64_scale, fp64_decoded = reference_fp64(
        values,
        bits=4,
        seed=0xBADC0FFEE,
        tensor_id=17,
        invocation_id=23,
    )

    block_count = (len(values) + 255) // 256
    operations = 12 + (block_count - 1).bit_length()
    unit_roundoff = 2.0**-24
    epsilon = operations * unit_roundoff / (1 - operations * unit_roundoff)
    epsilon += 4 * len(values) * 2.0**-149 + 2.0**-149 / (2 * fp64_scale)
    assert abs(fp32_scale - fp64_scale) <= fp64_scale * epsilon

    decoded = np.frombuffer(output_path.read_bytes(), dtype="<f4").astype(np.float64)
    output_bound = fp64_scale * (2.0 / 7.0 + 2.0 * epsilon / (1.0 - epsilon) + 5.0 * unit_roundoff)
    assert np.max(np.abs(decoded - fp64_decoded)) <= output_bound


@pytest.mark.parametrize(
    ("case", "mutation_fn", "expected_message"),
    [
        (
            "out-of-range 8-bit code",
            lambda rec: rec.__setitem__(HEADER.size, 255),
            "record contains an invalid code",
        ),
        (
            "out-of-range 4-bit code",
            lambda rec: (rec.__setitem__(5, 4), rec.__setitem__(HEADER.size, 0x0F)),
            "record contains an invalid code",
        ),
        (
            "nonzero padding nibble",
            lambda rec: (rec.__setitem__(5, 4), rec.__setitem__(HEADER.size, 0x17)),
            "unused padding nibble must be zero",
        ),
        (
            "nonfinite scale",
            lambda rec: rec.__setitem__(slice(16, 20), struct.pack("<f", float("nan"))),
            "scale must be finite and non-negative",
        ),
        (
            "negative scale",
            lambda rec: rec.__setitem__(slice(16, 20), struct.pack("<f", -1.0)),
            "scale must be finite and non-negative",
        ),
        (
            "zero-scale with non-center 8-bit code",
            lambda rec: (
                rec.__setitem__(slice(16, 20), struct.pack("<f", 0.0)),
                rec.__setitem__(HEADER.size, 0),
            ),
            "zero-scale records must contain only the center code",
        ),
        (
            "zero-scale with non-center 4-bit code",
            lambda rec: (
                rec.__setitem__(5, 4),
                rec.__setitem__(slice(16, 20), struct.pack("<f", 0.0)),
                rec.__setitem__(HEADER.size, 0x00),
            ),
            "zero-scale records must contain only the center code",
        ),
    ],
)
def test_decompress_rejects_extended_decoder_errors(tmp_path, case, mutation_fn, expected_message):
    record = bytearray(
        compress_record_fp32(np.asarray([1.0], dtype=np.float32), scale=1.0, words=[0])
    )
    mutation_fn(record)
    input_path = tmp_path / "malformed.msq"
    output_path = tmp_path / "decoded.f32"
    input_path.write_bytes(record)
    result = _run("decompress", "--input", str(input_path), "--output", str(output_path))
    assert result.returncode != 0
    assert expected_message in result.stderr


def test_compress_rejects_l2_norm_overflow_and_scale_is_never_saturated(tmp_path):
    input_path = tmp_path / "overflow.f32"
    output_path = tmp_path / "record.msq"
    _write_f32(input_path, np.asarray([3.0e38, 3.0e38], dtype=np.float32))
    result = _run(
        "compress",
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        "--seed",
        "1",
    )
    assert result.returncode != 0
    assert "scale overflow" in result.stderr.lower()
    assert not output_path.exists()


def test_edge_cases_signed_zero_and_large_magnitudes(tmp_path):
    # Signed zero
    sz = np.asarray([-0.0, 0.0], dtype=np.float32)
    for bits in ("4", "8"):
        rec = _compress(tmp_path, sz, extra=("--bits", bits))
        input_path = tmp_path / "record.msq"
        output_path = tmp_path / "decoded.f32"
        input_path.write_bytes(rec)
        result = _run("decompress", "--input", str(input_path), "--output", str(output_path))
        assert result.returncode == 0
        decoded = np.frombuffer(output_path.read_bytes(), dtype="<f4")
        assert not np.signbit(decoded[0])
        assert not np.signbit(decoded[1])
        assert decoded[0] == 0.0
        assert decoded[1] == 0.0

    # Large magnitude rescaling without overflow
    large = np.asarray([1.0e20, -1.0e20, 1.0e20], dtype=np.float32)
    for bits in ("4", "8"):
        rec = _compress(tmp_path, large, extra=("--bits", bits))
        input_path = tmp_path / "record.msq"
        output_path = tmp_path / "decoded.f32"
        input_path.write_bytes(rec)
        result = _run("decompress", "--input", str(input_path), "--output", str(output_path))
        assert result.returncode == 0
        decoded = np.frombuffer(output_path.read_bytes(), dtype="<f4")
        assert np.all(np.isfinite(decoded))


def test_edge_cases_saturation_and_odd_lengths(tmp_path):
    # Saturation boundary
    sat = np.asarray([-2.0, 2.0], dtype=np.float32)
    rec4 = _compress(tmp_path, sat, extra=("--bits", "4", "--scale", "2.0"))
    payload4 = rec4[HEADER.size :]
    assert len(payload4) == 1
    assert (payload4[0] & 0x0F) == 0  # -2.0 clamped/mapped to 0
    assert ((payload4[0] >> 4) & 0x0F) == 14  # +2.0 clamped/mapped to 14 (2*s)

    # Odd length 4-bit payload zeroes high nibble
    odd = np.asarray([2.0], dtype=np.float32)
    rec_odd = _compress(tmp_path, odd, extra=("--bits", "4", "--scale", "2.0"))
    payload_odd = rec_odd[HEADER.size :]
    assert len(payload_odd) == 1
    assert (payload_odd[0] & 0x0F) == 14
    assert ((payload_odd[0] >> 4) & 0x0F) == 0  # Unused padding nibble is 0

    # Non-multiple length (5, 257)
    for n in (5, 257):
        vals = np.linspace(-1.0, 1.0, n, dtype=np.float32)
        rec = _compress(tmp_path, vals, extra=("--bits", "4"))
        input_path = tmp_path / "record.msq"
        output_path = tmp_path / "decoded.f32"
        input_path.write_bytes(rec)
        result = _run("decompress", "--input", str(input_path), "--output", str(output_path))
        assert result.returncode == 0
        decoded = np.frombuffer(output_path.read_bytes(), dtype="<f4")
        assert len(decoded) == n


def test_layer_three_course_subset_unbiasedness():
    n = 1024
    t = 4096
    scale = np.float32(2.0)

    # Fixed 1024-element vector with zeros, negatives, exact points, and varied fractions
    x = np.empty(n, dtype=np.float32)
    x[0] = 0.0
    x[1] = -0.0
    x[2] = 2.0
    x[3] = -2.0
    idx = 4
    for k in range(1, 7):
        x[idx] = 2.0 * (k / 7.0)
        x[idx + 1] = -2.0 * (k / 7.0)
        idx += 2
    for k in range(1, 127):
        x[idx] = 2.0 * (k / 127.0)
        x[idx + 1] = -2.0 * (k / 127.0)
        idx += 2
    x[idx:] = np.linspace(-1.95, 1.95, n - idx, dtype=np.float32)

    words = np.empty((t, n), dtype=np.uint32)
    for seed in range(t):
        words[seed] = philox_words(n, seed)

    for bits, s in [(4, 7), (8, 127)]:
        scaled = np.minimum(np.abs(x) / scale * s, float(s))
        l = np.floor(scaled).astype(np.int32)
        p = scaled - l.astype(np.float32)

        decoded_sum = np.zeros(n, dtype=np.float64)
        for seed in range(t):
            codes = _round_codes_fp32(x, scale, words[seed], signed_limit=s)
            signed = codes.astype(np.int16) - s
            dec = (signed.astype(np.float32) / np.float32(s)) * scale
            decoded_sum += dec

        mean_dec = decoded_sum / t
        diff = np.abs(mean_dec - x)

        zero_p = (p == 0) | (p == 1)
        assert np.all(diff[zero_p] == 0.0), f"Exact equality failed for bits={bits}"

        bound = 5.0 * (float(scale) / s) * np.sqrt(p * (1.0 - p) / t)
        assert np.all(diff <= bound + 1e-6), f"5-sigma bound failed for bits={bits}"


def test_compress_rejects_nonfinite_input_and_prescribed_nonzero_empty_scale(tmp_path):
    input_path = tmp_path / "input.f32"
    output_path = tmp_path / "record.msq"
    _write_f32(input_path, np.asarray([np.nan], dtype=np.float32))
    nonfinite = _run(
        "compress",
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        "--seed",
        "1",
    )

    assert nonfinite.returncode != 0
    assert "non-finite" in nonfinite.stderr

    _write_f32(input_path, np.asarray([], dtype=np.float32))
    empty_nonzero_scale = _run(
        "compress",
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        "--seed",
        "1",
        "--scale",
        "1",
    )
    assert empty_nonzero_scale.returncode != 0
    assert "zero scale" in empty_nonzero_scale.stderr


def test_compress_rejects_id_overflow_and_nonzero_scale_for_all_zero_input(tmp_path):
    input_path = tmp_path / "input.f32"
    output_path = tmp_path / "record.msq"
    _write_f32(input_path, np.zeros(2, dtype=np.float32))

    id_overflow = _run(
        "compress",
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        "--seed",
        "1",
        "--tensor-id",
        str(1 << 32),
    )
    assert id_overflow.returncode != 0
    assert "identifiers must fit in uint32" in id_overflow.stderr

    nonzero_scale = _run(
        "compress",
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        "--seed",
        "1",
        "--scale",
        "1",
    )
    assert nonzero_scale.returncode != 0
    assert "zero scale" in nonzero_scale.stderr


@pytest.mark.parametrize("bits", [4, 8])
def test_cpu_bench_reports_raw_samples_and_base_record(tmp_path, bits: int):
    values = np.linspace(-1.5, 1.5, 17, dtype=np.float32)
    input_path = tmp_path / "bench-input.f32"
    bench_record_path = tmp_path / "bench-record.msq"
    _write_f32(input_path, values)

    result = _run(
        "bench",
        "--input",
        str(input_path),
        "--record-output",
        str(bench_record_path),
        "--seed",
        "812",
        "--bits",
        str(bits),
        "--tensor-id",
        "17",
        "--invocation-id",
        "23",
        "--warmup",
        "2",
        "--reps",
        "3",
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["configuration"] == {
        "backend": "cpu",
        "bits": bits,
        "count": len(values),
        "seed": 812,
        "tensor_id": 17,
        "invocation_id": 23,
        "warmup": 2,
        "reps": 3,
        "repetition_invocation_ids": [23, 24, 25],
        "warmup_invocation_ids": [26, 27],
        "boundary": "host-host",
        "transfer_policy": "none",
        "block_size": 256,
        "grid_size": 0,
        "prescribed_scale": None,
    }
    assert len(payload["samples_ms"]) == 3
    assert all(sample >= 0 for sample in payload["samples_ms"])
    assert "k1_ms" not in payload
    assert payload["header_bytes"] == HEADER.size
    assert payload["payload_bytes"] == (len(values) + (bits == 4)) // (2 if bits == 4 else 1)

    bench_record = bench_record_path.read_bytes()
    compress_record = _compress(
        tmp_path,
        values,
        seed=812,
        extra=(
            "--bits",
            str(bits),
            "--tensor-id",
            "17",
            "--invocation-id",
            "23",
        ),
    )
    assert bench_record == compress_record


@pytest.mark.parametrize(
    "extra",
    [
        ("--boundary", "resident"),
        ("--boundary", "resident-graph"),
        ("--boundary", "host-origin"),
        ("--transfer-policy", "pinned"),
        ("--transfer-policy", "pageable"),
        ("--block-size", "128"),
        ("--grid-size", "3"),
    ],
)
def test_cpu_bench_rejects_cuda_only_options(tmp_path, extra):
    input_path = tmp_path / "input.f32"
    _write_f32(input_path, np.ones(3, dtype=np.float32))
    result = _run("bench", "--input", str(input_path), "--seed", "1", *extra)

    assert result.returncode != 0
    assert "invalid argument" in result.stderr
    assert result.stdout == ""


def test_bench_rejects_invocation_range_before_emitting_samples(tmp_path):
    input_path = tmp_path / "input.f32"
    _write_f32(input_path, np.ones(3, dtype=np.float32))
    result = _run(
        "bench",
        "--input",
        str(input_path),
        "--seed",
        "1",
        "--invocation-id",
        str(0xFFFFFFFF),
        "--warmup",
        "0",
        "--reps",
        "2",
    )

    assert result.returncode != 0
    assert "identifiers must fit in uint32" in result.stderr
    assert result.stdout == ""


def test_cpu_bench_rejects_nonfinite_input_before_emitting_samples(tmp_path):
    input_path = tmp_path / "input.f32"
    _write_f32(input_path, np.asarray([np.nan], dtype=np.float32))
    result = _run(
        "bench",
        "--input",
        str(input_path),
        "--seed",
        "1",
        "--warmup",
        "0",
        "--reps",
        "1",
    )

    assert result.returncode != 0
    assert "non-finite" in result.stderr
    assert result.stdout == ""


def _expect(
    tmp_path: Path, values: np.ndarray, *extra: str
) -> tuple[subprocess.CompletedProcess[str], Path]:
    input_path = tmp_path / "expect-input.f32"
    output_path = tmp_path / "sums.f64"
    _write_f32(input_path, values)
    result = _run("expect", "--input", str(input_path), "--output", str(output_path), *extra)
    return result, output_path


@pytest.mark.parametrize("bits", ["4", "8"])
def test_expect_sums_match_decoded_compress_records(tmp_path, bits):
    values = np.random.default_rng(7).normal(size=33).astype(np.float32)
    seeds, seed_start = 5, 3
    result, output_path = _expect(
        tmp_path, values, "--seeds", str(seeds), "--seed-start", str(seed_start), "--bits", bits
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["seeds"] == seeds
    assert report["count"] == values.size

    decoded = np.stack(
        [
            decode_record(_compress(tmp_path, values, seed=k, extra=("--bits", bits))).astype(
                np.float64
            )
            for k in range(seed_start, seed_start + seeds)
        ]
    )
    raw = np.frombuffer(output_path.read_bytes(), dtype="<f8")
    assert raw.size == 2 * values.size
    np.testing.assert_array_equal(raw[: values.size], decoded.sum(axis=0))
    np.testing.assert_array_equal(raw[values.size :], (decoded * decoded).sum(axis=0))


@pytest.mark.parametrize(
    "extra",
    [
        ("--seeds", "0"),
        ("--seeds", "2", "--bits", "6"),
        ("--seeds", "2", "--backend", "gpu"),
        ("--bits", "8"),
    ],
)
def test_expect_rejects_invalid_arguments(tmp_path, extra):
    result, output_path = _expect(tmp_path, np.asarray([1.0, -0.5], dtype=np.float32), *extra)
    assert result.returncode != 0
    assert result.stdout == ""
    assert not output_path.exists()
