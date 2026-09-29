"""Correctness gate: byte identity across paths and the FP64 bound layer."""

from __future__ import annotations

import struct
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from stoquant.design import BenchPath, build_paths
from stoquant.oracle import HEADER_STRUCT, decode_record, fp64_error_bounds, reference_fp64
from stoquant.runner import DEFAULT_COMPRESSION_SEED, compress_args, k1_args


class CorrectnessFailure(Exception):
    """A gate phase failed; `details` is the correctness record for the case."""

    def __init__(self, details: dict[str, Any]) -> None:
        super().__init__(details.get("error_message"))
        self.details = details


@dataclass(frozen=True)
class RecordRun:
    """The binary, input and record parameters every process of one gate shares."""

    binary: Path
    input_path: Path
    tmp_dir: Path
    count: int
    bits: int
    seed: int
    tensor_id: int
    invocation_id: int
    k1: str

    def record_path(self, name: str) -> Path:
        return self.tmp_dir / f"{name}_b{self.bits}_n{self.count}.msq"

    def compress(self, backend: str, output: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            compress_args(
                self.binary,
                self.input_path,
                output,
                seed=self.seed,
                bits=self.bits,
                tensor_id=self.tensor_id,
                invocation_id=self.invocation_id,
                backend=backend,
                k1=self.k1,
            ),
            capture_output=True,
            text=True,
            check=False,
        )

    def bench(
        self, backend: str, output: Path, extra_args: Sequence[str] = ()
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                str(self.binary),
                "bench",
                "--input",
                str(self.input_path),
                "--record-output",
                str(output),
                "--seed",
                str(self.seed),
                "--bits",
                str(self.bits),
                "--tensor-id",
                str(self.tensor_id),
                "--invocation-id",
                str(self.invocation_id),
                "--backend",
                backend,
                *k1_args(backend, self.k1),
                *extra_args,
                "--warmup",
                "0",
                "--reps",
                "1",
            ],
            capture_output=True,
            text=True,
            check=False,
        )


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

    run = RecordRun(
        binary=binary,
        input_path=input_path,
        tmp_dir=tmp_dir,
        count=count,
        bits=bits,
        seed=seed,
        tensor_id=tensor_id,
        invocation_id=invocation_id,
        k1=k1,
    )
    try:
        cpu_comp_bytes, cpu_bench_bytes = check_cpu_reference(run)
        if "cpu-avx2" in backends:
            check_avx2_comparator(run, cpu_comp_bytes)
        cuda_comp_bytes = compress_cuda_reference(run) if "cuda" in backends else None
        bench_records = record_other_paths(run, paths)
        check_cross_path_identity(bench_records, cpu_comp_bytes, cuda_comp_bytes)
        return True, check_fp64_bounds(run, cpu_bench_bytes)
    except CorrectnessFailure as failure:
        return False, failure.details


def check_cpu_reference(run: RecordRun) -> tuple[bytes, bytes]:
    """CPU compress and CPU bench records, which must be byte-identical."""
    cpu_comp_path = run.record_path("cpu_comp")
    cpu_bench_path = run.record_path("cpu_bench")

    res_cpu_comp = run.compress("cpu", cpu_comp_path)
    if res_cpu_comp.returncode != 0:
        raise CorrectnessFailure(
            {
                "status": "failed",
                "error_message": f"CPU compress failed: {res_cpu_comp.stderr.strip()}",
            }
        )

    res_cpu_bench = run.bench("cpu", cpu_bench_path)
    if res_cpu_bench.returncode != 0:
        raise CorrectnessFailure(
            {
                "status": "failed",
                "error_message": f"CPU bench record failed: {res_cpu_bench.stderr.strip()}",
            }
        )

    cpu_comp_bytes = cpu_comp_path.read_bytes()
    cpu_bench_bytes = cpu_bench_path.read_bytes()

    if cpu_bench_bytes != cpu_comp_bytes:
        raise CorrectnessFailure(
            {
                "status": "failed",
                "byte_identical_to_compress": False,
                "cpu_cuda_byte_identical": False,
                "error_message": "CPU bench record is not byte-identical to CPU compress record.",
            }
        )
    return cpu_comp_bytes, cpu_bench_bytes


def check_avx2_comparator(run: RecordRun, cpu_comp_bytes: bytes) -> None:
    """The AVX2 comparator's compress record must match the scalar CPU one."""
    avx2_comp_path = run.record_path("cpu-avx2_comp")
    res_avx2_comp = run.compress("cpu-avx2", avx2_comp_path)
    if res_avx2_comp.returncode != 0:
        raise CorrectnessFailure(
            {
                "status": "failed",
                "error_message": f"cpu-avx2 compress failed: {res_avx2_comp.stderr.strip()}",
            }
        )
    if avx2_comp_path.read_bytes() != cpu_comp_bytes:
        raise CorrectnessFailure(
            {
                "status": "failed",
                "cpu_avx2_byte_identical": False,
                "error_message": "cpu-avx2 compress record is not byte-identical to CPU.",
            }
        )


def compress_cuda_reference(run: RecordRun) -> bytes:
    """The CUDA compress record, which CUDA bench paths must reproduce."""
    cuda_comp_path = run.record_path("cuda_comp")
    res_cuda_comp = run.compress("cuda", cuda_comp_path)
    if res_cuda_comp.returncode != 0:
        raise CorrectnessFailure(
            {
                "status": "failed",
                "error_message": f"CUDA compress failed: {res_cuda_comp.stderr.strip()}",
            }
        )
    return cuda_comp_path.read_bytes()


def record_other_paths(run: RecordRun, paths: Sequence[BenchPath]) -> dict[BenchPath, Path]:
    """One bench record per selected path except the comparator."""
    bench_paths: dict[BenchPath, Path] = {
        path: run.record_path(f"bench_{path.label}")
        for path in paths
        if path.boundary != "comparator"
    }
    for path, record_path in bench_paths.items():
        res_bench = run.bench(path.backend, record_path, path.extra_args)
        if res_bench.returncode != 0:
            name = "CUDA " + path.key if path.backend == "cuda" else path.label
            raise CorrectnessFailure(
                {
                    "status": "failed",
                    "error_message": f"{name} bench record failed: {res_bench.stderr.strip()}",
                }
            )
    return bench_paths


def check_cross_path_identity(
    bench_paths: dict[BenchPath, Path], cpu_comp_bytes: bytes, cuda_comp_bytes: bytes | None
) -> None:
    """Bench records match their backend's compress record, and CPU matches CUDA."""
    # Each path must reproduce its own backend's compress record; the AVX2
    # comparator's is the scalar one, checked above.
    byte_identical_to_compress = all(
        record_path.read_bytes() == (cuda_comp_bytes if path.backend == "cuda" else cpu_comp_bytes)
        for path, record_path in bench_paths.items()
    )
    cpu_cuda_byte_identical = cuda_comp_bytes is None or cpu_comp_bytes == cuda_comp_bytes

    if not byte_identical_to_compress:
        raise CorrectnessFailure(
            {
                "status": "failed",
                "byte_identical_to_compress": False,
                "cpu_cuda_byte_identical": cpu_cuda_byte_identical,
                "error_message": "A bench record output is not byte-identical to compress output.",
            }
        )

    if not cpu_cuda_byte_identical:
        raise CorrectnessFailure(
            {
                "status": "failed",
                "byte_identical_to_compress": True,
                "cpu_cuda_byte_identical": False,
                "error_message": "CPU and CUDA records are not byte-identical.",
            }
        )


def check_fp64_bounds(run: RecordRun, cpu_bench_bytes: bytes) -> dict[str, Any]:
    """Layer 2 decode validation against the FP64 oracle; returns the passing record."""
    count, bits = run.count, run.bits
    input_bytes = run.input_path.read_bytes()
    values = np.frombuffer(input_bytes, dtype="<f4")

    try:
        decoded = decode_record(cpu_bench_bytes)
    except (ValueError, TypeError, struct.error) as exc:
        raise CorrectnessFailure(
            {
                "status": "failed",
                "byte_identical_to_compress": True,
                "cpu_cuda_byte_identical": True,
                "layer2_scale_within_bound": False,
                "layer2_reconstruction_within_bound": False,
                "error_message": f"Record decoding failed: {exc}",
            }
        ) from exc

    fp64_scale, fp64_decoded = reference_fp64(
        values,
        bits=bits,
        seed=run.seed,
        tensor_id=run.tensor_id,
        invocation_id=run.invocation_id,
    )

    header = HEADER_STRUCT.unpack_from(cpu_bench_bytes)
    fp32_scale = header[5]

    try:
        scale_bound, reconstruction_bound = fp64_error_bounds(count, bits, fp64_scale)
    except ValueError as exc:
        raise CorrectnessFailure(
            {
                "status": "failed",
                "byte_identical_to_compress": True,
                "cpu_cuda_byte_identical": True,
                "layer2_scale_within_bound": False,
                "layer2_reconstruction_within_bound": False,
                "error_message": f"Layer 2 bound check failed: {exc}",
            }
        ) from exc

    scale_diff = abs(fp32_scale - fp64_scale)
    scale_ok = scale_diff <= scale_bound
    max_recon_error = float(np.max(np.abs(decoded.astype(np.float64) - fp64_decoded)))
    reconstruction_ok = max_recon_error <= reconstruction_bound

    if not scale_ok or not reconstruction_ok:
        raise CorrectnessFailure(
            {
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
        )

    return {
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
