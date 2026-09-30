"""Bench-process execution and GPU and in-process warm-up."""

from __future__ import annotations

import json
import math
import subprocess
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from stoquant.design import CUDA_RESIDENT

DEFAULT_WARMUPS = 10
DEFAULT_REPS = 30
DEFAULT_COMPRESSION_SEED = 42
DEFAULT_BENCH_TIMEOUT_SECONDS = 300

# The GPU idles at low clocks; a sustained resident workload before the first
# timed case brings it to its working clocks.
DEFAULT_GPU_WARMUP_SECONDS = 20.0
DEFAULT_CASE_WARMUP_SECONDS = 3.0
GPU_WARMUP_REPS = 100
# Sweep protocol revision 2: every timed process of every path first runs untimed
# repetitions for at least this long, in the same process. Warm-up in separate
# processes does not carry over to the timed one (issue #26 diagnosis).
DEFAULT_IN_PROCESS_WARMUP_SECONDS = 1.0


class BenchProcess(Protocol):
    """Execute one native command without changing its arguments or environment."""

    def run(
        self, argv: Sequence[str], *, creationflags: int = 0
    ) -> subprocess.CompletedProcess[str]: ...


@dataclass(frozen=True)
class RealBenchProcess:
    timeout_seconds: float = DEFAULT_BENCH_TIMEOUT_SECONDS

    def run(
        self, argv: Sequence[str], *, creationflags: int = 0
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            list(argv),
            capture_output=True,
            text=True,
            check=False,
            timeout=self.timeout_seconds,
            creationflags=creationflags,
        )


@dataclass
class FakeBenchProcess:
    """Replay process results through the same public interface in tests."""

    results: list[subprocess.CompletedProcess[str]]
    commands: list[list[str]] = field(default_factory=list)

    def run(
        self, argv: Sequence[str], *, creationflags: int = 0
    ) -> subprocess.CompletedProcess[str]:
        self.commands.append(list(argv))
        if not self.results:
            raise AssertionError(f"No fake bench result for {list(argv)!r}")
        return self.results.pop(0)


REAL_BENCH_PROCESS = RealBenchProcess()


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
    """The `compress` command line; `--k1` is passed only for the CUDA backend."""
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
    binary: Path,
    input_path: Path,
    seconds: float,
    k1: str = "reference",
    process: BenchProcess = REAL_BENCH_PROCESS,
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
            extra_args=["--boundary", CUDA_RESIDENT.boundary],
            seed=DEFAULT_COMPRESSION_SEED,
            warmups=0,
            reps=GPU_WARMUP_REPS,
            process=process,
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
    creationflags: int = 0,
    process: BenchProcess = REAL_BENCH_PROCESS,
) -> tuple[dict[str, Any] | None, str | None]:
    if any(arg == "--k1" or arg.startswith("--k1=") for arg in extra_args):
        raise ValueError("Pass the K1 variant through k1, not extra_args")
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
    try:
        proc = process.run(bench_cmd, creationflags=creationflags)
    except subprocess.TimeoutExpired:
        return None, "stoquant bench timed out"
    except OSError as exc:
        return None, f"stoquant bench could not start: {exc}"
    if proc.returncode != 0:
        return None, f"stoquant bench failed: {proc.stderr.strip()}"
    try:
        payload = json.loads(proc.stdout)
    except (TypeError, ValueError):
        return None, "stoquant bench returned invalid JSON"
    if not isinstance(payload, dict):
        return None, "stoquant bench returned non-object JSON"
    return payload, None


def in_process_warmups(minimum: int, seconds: float, rep_ms: float) -> int:
    """Warm-up count that covers `seconds` at `rep_ms` per repetition, never below `minimum`."""
    if seconds <= 0 or rep_ms <= 0:
        return minimum
    return max(minimum, math.ceil(seconds * 1000.0 / rep_ms))
