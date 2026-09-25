import os
import struct
import subprocess
from pathlib import Path

import numpy as np
import pytest

from mco2_oracle import compress_record_fp32, decode_record, reference_fp64, scale_fp32

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
        record[5] = 4
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
