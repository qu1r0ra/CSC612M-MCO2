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
    tmp_path, count: int, seed: int, tensor_id: int, invocation_id: int
):
    values = np.random.default_rng(count + tensor_id).normal(size=count).astype(np.float32)
    options = ("--tensor-id", str(tensor_id), "--invocation-id", str(invocation_id))

    cpu_result, cpu_record = _compress(tmp_path, values, backend="cpu", seed=seed, extra=options)
    cuda_result, cuda_record = _compress(tmp_path, values, backend="cuda", seed=seed, extra=options)

    assert cpu_result.returncode == 0, cpu_result.stderr
    assert cuda_result.returncode == 0, cuda_result.stderr
    assert cuda_record == cpu_record
    assert cuda_record == compress_record_fp32(
        values, seed=seed, tensor_id=tensor_id, invocation_id=invocation_id
    )


@pytest.mark.parametrize("count", [5, 257, 1025])
def test_cuda_prescribed_scale_and_words_match_oracle(tmp_path, count: int):
    values = np.linspace(-2.0, 2.0, count, dtype=np.float32)
    words = np.random.default_rng(42).integers(0, 2**32, size=count, dtype=np.uint32)
    words_path = tmp_path / "words.u32"
    words_path.write_bytes(words.astype("<u4").tobytes())

    result, record = _compress(
        tmp_path,
        values,
        backend="cuda",
        seed=7,
        extra=("--scale", "2.0", "--words", str(words_path)),
    )

    assert result.returncode == 0, result.stderr
    assert record == compress_record_fp32(values, scale=2.0, words=words)


def test_cuda_empty_and_all_zero_records_match_cpu(tmp_path):
    for values in (np.asarray([], dtype=np.float32), np.zeros(257, dtype=np.float32)):
        cpu_result, cpu_record = _compress(tmp_path, values, backend="cpu")
        cuda_result, cuda_record = _compress(tmp_path, values, backend="cuda")

        assert cpu_result.returncode == 0, cpu_result.stderr
        assert cuda_result.returncode == 0, cuda_result.stderr
        assert cuda_record == cpu_record
        assert cuda_record is not None
        assert HEADER.unpack_from(cuda_record)[5] == 0.0
        assert cuda_record[HEADER.size :] == bytes([127]) * len(values)


def test_cuda_layer_two_decode_satisfies_fp64_bound(tmp_path):
    values = np.random.default_rng(2026).normal(size=4097).astype(np.float32)
    seed, tensor_id, invocation_id = 0xBADC0FFEE, 17, 23
    result, record = _compress(
        tmp_path,
        values,
        backend="cuda",
        seed=seed,
        extra=("--tensor-id", str(tensor_id), "--invocation-id", str(invocation_id)),
    )

    assert result.returncode == 0, result.stderr
    assert record is not None
    fp32_scale = HEADER.unpack_from(record)[5]
    assert fp32_scale == float(scale_fp32(values))
    fp64_scale, fp64_decoded = reference_fp64(
        values,
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
    output_bound = fp64_scale * (2 / 127 + 2 * epsilon / (1 - epsilon) + 5 * unit_roundoff)
    assert np.max(np.abs(decoded - fp64_decoded)) <= output_bound


@pytest.mark.parametrize(
    "values,extra,message",
    [
        (np.asarray([np.nan], dtype=np.float32), (), "non-finite"),
        (np.asarray([3.0e38, 3.0e38], dtype=np.float32), (), "scale overflow"),
        (np.asarray([1.0], dtype=np.float32), ("--tensor-id", str(2**32)), "identifiers"),
        (np.asarray([1.0], dtype=np.float32), ("--invocation-id", str(2**32)), "identifiers"),
        (np.asarray([1.0], dtype=np.float32), ("--scale", "0"), "scale"),
    ],
)
def test_cuda_rejects_invalid_input_and_scale(tmp_path, values, extra, message):
    cpu_result, cpu_record = _compress(tmp_path, values, backend="cpu", extra=extra)
    cuda_result, cuda_record = _compress(tmp_path, values, backend="cuda", extra=extra)
    assert cpu_result.returncode != 0
    assert cuda_result.returncode != 0
    assert cuda_result.stderr == cpu_result.stderr
    assert message in cuda_result.stderr.lower()
    assert cpu_record is None
    assert cuda_record is None


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


def test_cuda_timings_are_one_json_stderr_line(tmp_path):
    values = np.random.default_rng(93).normal(size=1 << 16).astype(np.float32)
    result, record = _compress(tmp_path, values, backend="cuda", extra=("--timings",))

    assert result.returncode == 0, result.stderr
    assert record is not None
    lines = result.stderr.splitlines()
    assert len(lines) == 1
    timings = json.loads(lines[0])
    assert set(timings) == {"k1_ms", "k2_ms", "k3_ms", "h2d_ms", "d2h_ms"}
    assert all(isinstance(value, (int, float)) and value >= 0 for value in timings.values())
