"""Bench-process execution and GPU and in-process warm-up."""

from __future__ import annotations

import json
import math
import subprocess
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

DEFAULT_WARMUPS = 10
DEFAULT_REPS = 30
DEFAULT_COMPRESSION_SEED = 42

# The GPU idles at low clocks; a sustained resident workload before the first
# timed case brings it to its working clocks.
DEFAULT_GPU_WARMUP_SECONDS = 20.0
DEFAULT_CASE_WARMUP_SECONDS = 3.0
GPU_WARMUP_REPS = 100
# Sweep protocol revision 2: every timed process of every path first runs untimed
# repetitions for at least this long, in the same process. Warm-up in separate
# processes does not carry over to the timed one (issue #26 diagnosis).
DEFAULT_IN_PROCESS_WARMUP_SECONDS = 1.0
WARMUP_PROBE_REPS = 3


def k1_args(backend: str, k1: str) -> list[str]:
    return ["--k1", k1] if backend == "cuda" else []


def compress_args(
    binary: Path,
    input_path: Path,
    output: Path,
    *,
    seed: int,
    bits: int,
    tensor_id: int,
    invocation_id: int,
    backend: str,
    k1: str = "reference",
) -> list[str]:
    return [
        str(binary),
        "compress",
        "--input",
        str(input_path),
        "--output",
        str(output),
        "--seed",
        str(seed),
        "--bits",
        str(bits),
        "--tensor-id",
        str(tensor_id),
        "--invocation-id",
        str(invocation_id),
        "--backend",
        backend,
        *k1_args(backend, k1),
    ]


def warm_up_gpu(
    binary: Path, input_path: Path, seconds: float, k1: str = "reference"
) -> dict[str, Any]:
    """Run the resident CUDA path untimed until `seconds` have passed."""
    start = time.monotonic()
    processes = 0
    while time.monotonic() - start < seconds:
        payload, error = run_bench_process(
            binary,
            input_path,
            bits=8,
            backend="cuda",
            k1=k1,
            extra_args=["--boundary", "resident"],
            seed=DEFAULT_COMPRESSION_SEED,
            warmups=0,
            reps=GPU_WARMUP_REPS,
        )
        if payload is None:
            raise RuntimeError(f"GPU warm-up failed: {error}")
        processes += 1
    return {
        "seconds_requested": seconds,
        "seconds_elapsed": time.monotonic() - start,
        "workload": f"cuda resident, 8-bit, {input_path.stem}, {GPU_WARMUP_REPS} reps per process",
        "processes": processes,
    }


def run_bench_process(
    binary: Path,
    input_path: Path,
    *,
    bits: int,
    backend: str,
    extra_args: Sequence[str],
    seed: int,
    warmups: int,
    reps: int,
    k1: str = "reference",
) -> tuple[dict[str, Any] | None, str | None]:
    bench_cmd = [
        str(binary),
        "bench",
        "--input",
        str(input_path),
        "--seed",
        str(seed),
        "--bits",
        str(bits),
        "--tensor-id",
        "0",
        "--invocation-id",
        "0",
        "--backend",
        backend,
        *k1_args(backend, k1),
        "--warmup",
        str(warmups),
        "--reps",
        str(reps),
        *extra_args,
    ]
    proc = subprocess.run(bench_cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        return None, f"stoquant bench failed: {proc.stderr.strip()}"
    return json.loads(proc.stdout), None


def in_process_warmups(minimum: int, seconds: float, rep_ms: float) -> int:
    """Warm-up count that covers `seconds` at `rep_ms` per repetition, never below `minimum`."""
    if seconds <= 0 or rep_ms <= 0:
        return minimum
    return max(minimum, math.ceil(seconds * 1000.0 / rep_ms))
