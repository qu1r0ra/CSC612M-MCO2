"""Correctness gate: byte identity across paths and the FP64 bound layer."""

from __future__ import annotations

import struct
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from stoquant.design import BenchPath, build_paths
from stoquant.oracle import HEADER_STRUCT, decode_record, reference_fp64
from stoquant.runner import DEFAULT_COMPRESSION_SEED, k1_args


def verify_correctness(
    binary: Path,
    input_path: Path,
    count: int,
    bits: int,
    *,
    backends: Sequence[str] = ("cpu", "cuda"),
    paths: Sequence[BenchPath] | None = None,
    seed: int = DEFAULT_COMPRESSION_SEED,
    tensor_id: int = 0,
    invocation_id: int = 0,
    tmp_dir: Path,
    force_fail: bool = False,
    k1: str = "reference",
) -> tuple[bool, dict[str, Any]]:
    """Gating check before timing each case.

    Checks:
    1. CPU comparator bench record == CPU compress record.
    2. If CUDA backend enabled: CPU record == CUDA compress record bit-for-bit.
    3. Every other selected path's bench record == its backend's compress record.
    4. Decoded record satisfies Layer 2 mathematical bounds against FP64 oracle.
    """
    if paths is None:
        paths = build_paths(backends)
    if force_fail:
        return False, {
            "status": "failed",
            "byte_identical_to_compress": False,
            "cpu_cuda_byte_identical": False,
            "layer2_scale_within_bound": False,
            "layer2_reconstruction_within_bound": False,
            "error_message": "Forced failure for testing verification gate",
        }

    include_cuda = "cuda" in backends
    cpu_comp_path = tmp_dir / f"cpu_comp_b{bits}_n{count}.msq"
    cuda_comp_path = tmp_dir / f"cuda_comp_b{bits}_n{count}.msq"
    cpu_bench_path = tmp_dir / f"cpu_bench_b{bits}_n{count}.msq"

    # 1. CPU compress
    res_cpu_comp = subprocess.run(
        [
            str(binary),
            "compress",
            "--input",
            str(input_path),
            "--output",
            str(cpu_comp_path),
            "--seed",
            str(seed),
            "--bits",
            str(bits),
            "--tensor-id",
            str(tensor_id),
            "--invocation-id",
            str(invocation_id),
            "--backend",
            "cpu",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if res_cpu_comp.returncode != 0:
        return False, {
            "status": "failed",
            "error_message": f"CPU compress failed: {res_cpu_comp.stderr.strip()}",
        }

    # 2. CPU bench --record-output
    res_cpu_bench = subprocess.run(
        [
            str(binary),
            "bench",
            "--input",
            str(input_path),
            "--record-output",
            str(cpu_bench_path),
            "--seed",
            str(seed),
            "--bits",
            str(bits),
            "--tensor-id",
            str(tensor_id),
            "--invocation-id",
            str(invocation_id),
            "--backend",
            "cpu",
            "--warmup",
            "0",
            "--reps",
            "1",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if res_cpu_bench.returncode != 0:
        return False, {
            "status": "failed",
            "error_message": f"CPU bench record failed: {res_cpu_bench.stderr.strip()}",
        }

    cpu_comp_bytes = cpu_comp_path.read_bytes()
    cpu_bench_bytes = cpu_bench_path.read_bytes()

    if cpu_bench_bytes != cpu_comp_bytes:
        return False, {
            "status": "failed",
            "byte_identical_to_compress": False,
            "cpu_cuda_byte_identical": False,
            "error_message": "CPU bench record is not byte-identical to CPU compress record.",
        }

    if "cpu-avx2" in backends:
        avx2_comp_path = tmp_dir / f"cpu-avx2_comp_b{bits}_n{count}.msq"
        res_avx2_comp = subprocess.run(
            [
                str(binary),
                "compress",
                "--input",
                str(input_path),
                "--output",
                str(avx2_comp_path),
                "--seed",
                str(seed),
                "--bits",
                str(bits),
                "--tensor-id",
                str(tensor_id),
                "--invocation-id",
                str(invocation_id),
                "--backend",
                "cpu-avx2",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if res_avx2_comp.returncode != 0:
            return False, {
                "status": "failed",
                "error_message": f"cpu-avx2 compress failed: {res_avx2_comp.stderr.strip()}",
            }
        if avx2_comp_path.read_bytes() != cpu_comp_bytes:
            return False, {
                "status": "failed",
                "cpu_avx2_byte_identical": False,
                "error_message": "cpu-avx2 compress record is not byte-identical to CPU.",
            }

    cuda_comp_bytes: bytes | None = None
    if include_cuda:
        res_cuda_comp = subprocess.run(
            [
                str(binary),
                "compress",
                "--input",
                str(input_path),
                "--output",
                str(cuda_comp_path),
                "--seed",
                str(seed),
                "--bits",
                str(bits),
                "--tensor-id",
                str(tensor_id),
                "--invocation-id",
                str(invocation_id),
                "--backend",
                "cuda",
                *k1_args("cuda", k1),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if res_cuda_comp.returncode != 0:
            return False, {
                "status": "failed",
                "error_message": f"CUDA compress failed: {res_cuda_comp.stderr.strip()}",
            }
        cuda_comp_bytes = cuda_comp_path.read_bytes()

    bench_paths: dict[BenchPath, Path] = {
        path: tmp_dir / f"bench_{path.label}_b{bits}_n{count}.msq"
        for path in paths
        if path.boundary != "comparator"
    }
    for path, record_path in bench_paths.items():
        res_bench = subprocess.run(
            [
                str(binary),
                "bench",
                "--input",
                str(input_path),
                "--record-output",
                str(record_path),
                "--seed",
                str(seed),
                "--bits",
                str(bits),
                "--tensor-id",
                str(tensor_id),
                "--invocation-id",
                str(invocation_id),
                "--backend",
                path.backend,
                *k1_args(path.backend, k1),
                *path.extra_args,
                "--warmup",
                "0",
                "--reps",
                "1",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if res_bench.returncode != 0:
            name = "CUDA " + path.key if path.backend == "cuda" else path.label
            return False, {
                "status": "failed",
                "error_message": f"{name} bench record failed: {res_bench.stderr.strip()}",
            }

    # Each path must reproduce its own backend's compress record; the AVX2
    # comparator's is the scalar one, checked above.
    byte_identical_to_compress = all(
        record_path.read_bytes() == (cuda_comp_bytes if path.backend == "cuda" else cpu_comp_bytes)
        for path, record_path in bench_paths.items()
    )
    cpu_cuda_byte_identical = cuda_comp_bytes is None or cpu_comp_bytes == cuda_comp_bytes

    if not byte_identical_to_compress:
        return False, {
            "status": "failed",
            "byte_identical_to_compress": False,
            "cpu_cuda_byte_identical": cpu_cuda_byte_identical,
            "error_message": "A bench record output is not byte-identical to compress output.",
        }

    if not cpu_cuda_byte_identical:
        return False, {
            "status": "failed",
            "byte_identical_to_compress": True,
            "cpu_cuda_byte_identical": False,
            "error_message": "CPU and CUDA records are not byte-identical.",
        }

    # Layer 2 decode validation against FP64 oracle
    input_bytes = input_path.read_bytes()
    values = np.frombuffer(input_bytes, dtype="<f4")

    try:
        decoded = decode_record(cpu_bench_bytes)
    except (ValueError, TypeError, struct.error) as exc:
        return False, {
            "status": "failed",
            "byte_identical_to_compress": True,
            "cpu_cuda_byte_identical": True,
            "layer2_scale_within_bound": False,
            "layer2_reconstruction_within_bound": False,
            "error_message": f"Record decoding failed: {exc}",
        }

    fp64_scale, fp64_decoded = reference_fp64(
        values,
        bits=bits,
        seed=seed,
        tensor_id=tensor_id,
        invocation_id=invocation_id,
    )

    header = HEADER_STRUCT.unpack_from(cpu_bench_bytes)
    fp32_scale = header[5]

    block_count = (count + 255) // 256
    operations = 12 + (block_count - 1).bit_length()
    unit_roundoff = 2.0**-24
    epsilon = operations * unit_roundoff / (1.0 - operations * unit_roundoff)
    epsilon += 4 * count * 2.0**-149 + 2.0**-149 / (2.0 * fp64_scale)

    scale_diff = abs(fp32_scale - fp64_scale)
    scale_bound = fp64_scale * epsilon
    scale_ok = scale_diff <= scale_bound

    s = 7 if bits == 4 else 127
    reconstruction_bound = fp64_scale * (
        2.0 / s + 2.0 * epsilon / (1.0 - epsilon) + 5.0 * unit_roundoff
    )
    max_recon_error = float(np.max(np.abs(decoded.astype(np.float64) - fp64_decoded)))
    reconstruction_ok = max_recon_error <= reconstruction_bound

    if not scale_ok or not reconstruction_ok:
        return False, {
            "status": "failed",
            "byte_identical_to_compress": True,
            "cpu_cuda_byte_identical": True,
            "layer2_scale_within_bound": scale_ok,
            "layer2_reconstruction_within_bound": reconstruction_ok,
            "scale_error": scale_diff,
            "scale_bound": scale_bound,
            "reconstruction_error": max_recon_error,
            "reconstruction_bound": reconstruction_bound,
            "error_message": (
                f"Layer 2 bound check failed: scale_ok={scale_ok} "
                f"({scale_diff} <= {scale_bound}), "
                f"reconstruction_ok={reconstruction_ok} "
                f"({max_recon_error} <= {reconstruction_bound})"
            ),
        }

    return True, {
        "status": "passed",
        "byte_identical_to_compress": True,
        "cpu_cuda_byte_identical": True,
        "layer2_scale_within_bound": True,
        "layer2_reconstruction_within_bound": True,
        "scale_error": scale_diff,
        "scale_bound": scale_bound,
        "reconstruction_error": max_recon_error,
        "reconstruction_bound": reconstruction_bound,
        "error_message": None,
    }
