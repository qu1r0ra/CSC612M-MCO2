import json
import os
import struct
import subprocess
from pathlib import Path

import numpy as np
import pytest

from mco2_oracle import compress_record_fp32, reference_fp64, scale_fp32

ROOT = Path(__file__).resolve().parents[1]
BINARY = ROOT / "build" / ("mco2.exe" if os.name == "nt" else "mco2")
HEADER = struct.Struct("<4sBBHQf")

pytestmark = pytest.mark.skipif(
    os.environ.get("MCO2_TEST_CUDA") != "1",
    reason="run with `just test-cuda` on a CUDA device",
)


def _run(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(BINARY), *arguments],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def _write_values(path: Path, values: np.ndarray) -> None:
    path.write_bytes(np.asarray(values, dtype="<f4").tobytes())


def _compress(
    tmp_path: Path,
    values: np.ndarray,
    *,
    backend: str,
    seed: int = 1,
    extra: tuple[str, ...] = (),
) -> tuple[subprocess.CompletedProcess[str], bytes | None]:
    input_path = tmp_path / "input.f32"
    output_path = tmp_path / f"record-{backend}.msq"
    _write_values(input_path, values)
    result = _run(
        "compress",
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        "--seed",
        str(seed),
        "--backend",
        backend,
        *extra,
    )
    return result, output_path.read_bytes() if output_path.exists() else None


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("count", [0, 1, 3, 4, 5, 255, 256, 257, 1025, 65539])
@pytest.mark.parametrize(
    "seed,tensor_id,invocation_id",
    [
        (1, 0, 0),
        (1, 17, 0),
        (1, 0, 23),
        (0x123456789ABCDEF, 17, 23),
        (0xFEDCBA9876543210, 9, 0xFFFFFFFF),
    ],
)
def test_cuda_record_is_byte_identical_to_cpu(
    tmp_path, bits: int, count: int, seed: int, tensor_id: int, invocation_id: int
):
    values = np.random.default_rng(count + tensor_id).normal(size=count).astype(np.float32)
    options = (
        "--bits",
        str(bits),
        "--tensor-id",
        str(tensor_id),
        "--invocation-id",
        str(invocation_id),
    )

    cpu_result, cpu_record = _compress(tmp_path, values, backend="cpu", seed=seed, extra=options)
    cuda_result, cuda_record = _compress(tmp_path, values, backend="cuda", seed=seed, extra=options)

    assert cpu_result.returncode == 0, cpu_result.stderr
    assert cuda_result.returncode == 0, cuda_result.stderr
    assert cuda_record == cpu_record
    assert cuda_record == compress_record_fp32(
        values, bits=bits, seed=seed, tensor_id=tensor_id, invocation_id=invocation_id
    )


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("count", [5, 257, 1025])
def test_cuda_prescribed_scale_and_words_match_oracle(tmp_path, bits: int, count: int):
    values = np.linspace(-2.0, 2.0, count, dtype=np.float32)
    words = np.random.default_rng(42).integers(0, 2**32, size=count, dtype=np.uint32)
    words_path = tmp_path / "words.u32"
    words_path.write_bytes(words.astype("<u4").tobytes())

    result, record = _compress(
        tmp_path,
        values,
        backend="cuda",
        seed=7,
        extra=("--bits", str(bits), "--scale", "2.0", "--words", str(words_path)),
    )

    assert result.returncode == 0, result.stderr
    assert record == compress_record_fp32(values, bits=bits, scale=2.0, words=words)


@pytest.mark.parametrize("bits", [4, 8])
def test_cuda_empty_and_all_zero_records_match_cpu(tmp_path, bits: int):
    for values in (np.asarray([], dtype=np.float32), np.zeros(257, dtype=np.float32)):
        cpu_result, cpu_record = _compress(
            tmp_path, values, backend="cpu", extra=("--bits", str(bits))
        )
        cuda_result, cuda_record = _compress(
            tmp_path, values, backend="cuda", extra=("--bits", str(bits))
        )

        assert cpu_result.returncode == 0, cpu_result.stderr
        assert cuda_result.returncode == 0, cuda_result.stderr
        assert cuda_record == cpu_record
        assert cuda_record is not None
        assert HEADER.unpack_from(cuda_record)[5] == 0.0
        center_code = 7 if bits == 4 else 127
        if bits == 8:
            assert cuda_record[HEADER.size :] == bytes([center_code]) * len(values)
        else:
            if len(values) == 0:
                assert len(cuda_record[HEADER.size :]) == 0
            else:
                pairs = len(values) // 2
                expected_payload = bytes([center_code | (center_code << 4)]) * pairs
                if len(values) % 2 == 1:
                    expected_payload += bytes([center_code])
                assert cuda_record[HEADER.size :] == expected_payload


@pytest.mark.parametrize("bits,s", [(4, 7), (8, 127)])
def test_cuda_layer_two_decode_satisfies_fp64_bound(tmp_path, bits: int, s: int):
    values = np.random.default_rng(2026).normal(size=4097).astype(np.float32)
    seed, tensor_id, invocation_id = 0xBADC0FFEE, 17, 23
    result, record = _compress(
        tmp_path,
        values,
        backend="cuda",
        seed=seed,
        extra=(
            "--bits",
            str(bits),
            "--tensor-id",
            str(tensor_id),
            "--invocation-id",
            str(invocation_id),
        ),
    )

    assert result.returncode == 0, result.stderr
    assert record is not None
    fp32_scale = HEADER.unpack_from(record)[5]
    assert fp32_scale == float(scale_fp32(values))
    fp64_scale, fp64_decoded = reference_fp64(
        values,
        bits=bits,
        seed=seed,
        tensor_id=tensor_id,
        invocation_id=invocation_id,
    )

    block_count = (len(values) + 255) // 256
    operations = 12 + (block_count - 1).bit_length()
    unit_roundoff = 2.0**-24
    epsilon = operations * unit_roundoff / (1 - operations * unit_roundoff)
    epsilon += 4 * len(values) * 2.0**-149 + 2.0**-149 / (2 * fp64_scale)
    assert abs(fp32_scale - fp64_scale) <= fp64_scale * epsilon

    record_path = tmp_path / "record-cuda.msq"
    decoded_path = tmp_path / "decoded.f32"
    record_path.write_bytes(record)
    decoded_result = _run("decompress", "--input", str(record_path), "--output", str(decoded_path))
    assert decoded_result.returncode == 0, decoded_result.stderr
    decoded = np.frombuffer(decoded_path.read_bytes(), dtype="<f4").astype(np.float64)
    output_bound = fp64_scale * (2 / s + 2 * epsilon / (1 - epsilon) + 5 * unit_roundoff)
    assert np.max(np.abs(decoded - fp64_decoded)) <= output_bound


def test_cuda_k3_4bit_packing_structure_and_odd_length(tmp_path):
    # Odd length: 1 element
    odd1 = np.asarray([2.0], dtype=np.float32)
    result, record = _compress(
        tmp_path, odd1, backend="cuda", extra=("--bits", "4", "--scale", "2.0")
    )
    assert result.returncode == 0, result.stderr
    assert record is not None
    payload = record[HEADER.size :]
    assert len(payload) == 1
    # Lower element is in low nibble, high nibble is zeroed
    assert (payload[0] >> 4) == 0

    # 3 elements with prescribed words
    odd3 = np.asarray([2.0, -2.0, 1.0], dtype=np.float32)
    words = np.asarray([0, 0, 0], dtype="<u4")
    words_path = tmp_path / "words.u32"
    words_path.write_bytes(words.tobytes())
    result3, record3 = _compress(
        tmp_path,
        odd3,
        backend="cuda",
        extra=("--bits", "4", "--scale", "2.0", "--words", str(words_path)),
    )
    assert result3.returncode == 0, result3.stderr
    assert record3 is not None
    payload3 = record3[HEADER.size :]
    assert len(payload3) == 2  # (3 + 1) // 2
    # Last byte must have high nibble 0
    assert (payload3[1] >> 4) == 0


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize(
    "block_size,grid_size",
    [
        (32, 1),  # grid smaller than element count
        (64, 2),  # small grid
        (128, 7),
        (256, 13),
        (512, 1),
        (1024, 3),
        (256, 100),
    ],
)
def test_cuda_launch_geometry_independence(tmp_path, bits: int, block_size: int, grid_size: int):
    values = np.random.default_rng(42).normal(size=1025).astype(np.float32)
    seed = 12345
    base_res, base_rec = _compress(
        tmp_path, values, backend="cuda", seed=seed, extra=("--bits", str(bits))
    )
    geom_res, geom_rec = _compress(
        tmp_path,
        values,
        backend="cuda",
        seed=seed,
        extra=(
            "--bits",
            str(bits),
            "--block-size",
            str(block_size),
            "--grid-size",
            str(grid_size),
        ),
    )
    cpu_res, cpu_rec = _compress(
        tmp_path, values, backend="cpu", seed=seed, extra=("--bits", str(bits))
    )

    assert base_res.returncode == 0, base_res.stderr
    assert geom_res.returncode == 0, geom_res.stderr
    assert cpu_res.returncode == 0, cpu_res.stderr
    assert geom_rec == base_rec
    assert geom_rec == cpu_rec


@pytest.mark.parametrize("grid_size", [1, 2, 7, 13, 64, 256, 1024])
def test_cuda_k1_grid_size_independence(tmp_path, grid_size: int):
    values = np.random.default_rng(777).normal(size=4097).astype(np.float32)
    seed = 999
    base_res, base_rec = _compress(tmp_path, values, backend="cuda", seed=seed)
    grid_res, grid_rec = _compress(
        tmp_path,
        values,
        backend="cuda",
        seed=seed,
        extra=("--grid-size", str(grid_size)),
    )
    assert base_res.returncode == 0, base_res.stderr
    assert grid_res.returncode == 0, grid_res.stderr
    assert base_rec is not None and grid_rec is not None
    # Compare scales in header bit for bit
    base_scale = HEADER.unpack_from(base_rec)[5]
    grid_scale = HEADER.unpack_from(grid_rec)[5]
    assert base_scale == grid_scale
    assert base_rec == grid_rec


@pytest.mark.parametrize("bits", [4, 8])
def test_cuda_determinism_repeated_runs(tmp_path, bits: int):
    values = np.random.default_rng(888).normal(size=513).astype(np.float32)
    seed = 42
    extra = ("--bits", str(bits), "--tensor-id", "5", "--invocation-id", "11")
    records = []
    for _ in range(3):
        res, rec = _compress(tmp_path, values, backend="cuda", seed=seed, extra=extra)
        assert res.returncode == 0, res.stderr
        records.append(rec)
    assert records[0] == records[1] == records[2]


@pytest.mark.parametrize("bits", [4, 8])
def test_cuda_edge_cases_match_cpu(tmp_path, bits: int):
    # Signed zero
    sz = np.asarray([-0.0, 0.0], dtype=np.float32)
    cpu_res, cpu_rec = _compress(tmp_path, sz, backend="cpu", extra=("--bits", str(bits)))
    cuda_res, cuda_rec = _compress(tmp_path, sz, backend="cuda", extra=("--bits", str(bits)))
    assert cpu_res.returncode == 0 and cuda_res.returncode == 0
    assert cuda_rec == cpu_rec
    # Verify decode to +0.0
    rec_path = tmp_path / "sz.msq"
    dec_path = tmp_path / "sz.f32"
    rec_path.write_bytes(cuda_rec)
    dec_res = _run("decompress", "--input", str(rec_path), "--output", str(dec_path))
    assert dec_res.returncode == 0
    dec = np.frombuffer(dec_path.read_bytes(), dtype="<f4")
    assert np.all(dec == 0.0)
    assert not any(np.signbit(dec))

    # Largest representable magnitudes without overflow
    large = np.asarray([1.0e20, -1.0e20, 1.0e20], dtype=np.float32)
    cpu_res, cpu_rec = _compress(tmp_path, large, backend="cpu", extra=("--bits", str(bits)))
    cuda_res, cuda_rec = _compress(tmp_path, large, backend="cuda", extra=("--bits", str(bits)))
    assert cpu_res.returncode == 0 and cuda_res.returncode == 0
    assert cuda_rec == cpu_rec

    # Saturation boundary
    sat = np.asarray([-2.0, 2.0], dtype=np.float32)
    cpu_res, cpu_rec = _compress(
        tmp_path, sat, backend="cpu", extra=("--bits", str(bits), "--scale", "2.0")
    )
    cuda_res, cuda_rec = _compress(
        tmp_path, sat, backend="cuda", extra=("--bits", str(bits), "--scale", "2.0")
    )
    assert cpu_res.returncode == 0 and cuda_res.returncode == 0
    assert cuda_rec == cpu_rec

    # Odd lengths
    for n in (1, 3, 5, 257):
        vals = np.linspace(-1.0, 1.0, n, dtype=np.float32)
        cpu_res, cpu_rec = _compress(tmp_path, vals, backend="cpu", extra=("--bits", str(bits)))
        cuda_res, cuda_rec = _compress(tmp_path, vals, backend="cuda", extra=("--bits", str(bits)))
        assert cpu_res.returncode == 0 and cuda_res.returncode == 0
        assert cuda_rec == cpu_rec


@pytest.mark.parametrize(
    "values,extra,message",
    [
        (np.asarray([np.nan], dtype=np.float32), (), "non-finite"),
        (np.asarray([3.0e38, 3.0e38], dtype=np.float32), (), "scale overflow"),
        (
            np.asarray([1.0], dtype=np.float32),
            ("--tensor-id", str(2**32)),
            "identifiers",
        ),
        (
            np.asarray([1.0], dtype=np.float32),
            ("--invocation-id", str(2**32)),
            "identifiers",
        ),
        (np.asarray([1.0], dtype=np.float32), ("--scale", "0"), "scale"),
        (
            np.asarray([1.0], dtype=np.float32),
            ("--block-size", "0"),
            "invalid argument",
        ),
        (
            np.asarray([1.0], dtype=np.float32),
            ("--block-size", "1025"),
            "invalid argument",
        ),
        (
            np.asarray([1.0], dtype=np.float32),
            ("--grid-size", "0"),
            "invalid argument",
        ),
        (
            np.asarray([1.0], dtype=np.float32),
            ("--grid-size", "65536"),
            "invalid argument",
        ),
    ],
)
def test_cuda_rejects_invalid_input_and_scale(tmp_path, values, extra, message):
    cuda_result, cuda_record = _compress(tmp_path, values, backend="cuda", extra=extra)
    assert cuda_result.returncode != 0
    assert message in cuda_result.stderr.lower()
    assert cuda_record is None


def test_cuda_geometry_options_rejected_on_cpu(tmp_path):
    values = np.asarray([1.0], dtype=np.float32)
    cpu_result, _ = _compress(tmp_path, values, backend="cpu", extra=("--block-size", "128"))
    assert cpu_result.returncode != 0
    assert "invalid argument" in cpu_result.stderr.lower()


def test_cuda_rejects_nonzero_scale_for_empty_input(tmp_path):
    cpu_result, cpu_record = _compress(
        tmp_path,
        np.asarray([], dtype=np.float32),
        backend="cpu",
        extra=("--scale", "1"),
    )
    cuda_result, cuda_record = _compress(
        tmp_path,
        np.asarray([], dtype=np.float32),
        backend="cuda",
        extra=("--scale", "1"),
    )
    assert cpu_result.returncode != 0
    assert cuda_result.returncode != 0
    assert cuda_result.stderr == cpu_result.stderr
    assert "scale" in cuda_result.stderr.lower()
    assert cpu_record is None
    assert cuda_record is None


@pytest.mark.parametrize("bits", [4, 8])
def test_cuda_timings_are_one_json_stderr_line(tmp_path, bits: int):
    values = np.random.default_rng(93).normal(size=1 << 16).astype(np.float32)
    result, record = _compress(
        tmp_path, values, backend="cuda", extra=("--bits", str(bits), "--timings")
    )

    assert result.returncode == 0, result.stderr
    assert record is not None
    lines = result.stderr.splitlines()
    assert len(lines) == 1
    timings = json.loads(lines[0])
    assert set(timings) == {"k1_ms", "k2_ms", "k3_ms", "h2d_ms", "d2h_ms"}
    assert all(isinstance(value, (int, float)) and value >= 0 for value in timings.values())
