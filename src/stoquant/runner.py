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
class ValidatedBenchResponse:
    """Accepted native response with its original JSON payload preserved."""

    payload: dict[str, Any]
    configuration: dict[str, Any]
    samples_ms: list[int | float]
    stage_samples_ms: dict[str, list[int | float]]
    capture_and_instantiate_ms: int | float | None


class BenchResponseError(ValueError):
    """The native process succeeded but did not return the requested bench run."""


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
    *,
    count: int,
) -> dict[str, Any]:
    """Run the resident CUDA path untimed until `seconds` have passed."""
    start = time.monotonic()
    processes = 0
    while time.monotonic() - start < seconds:
        payload, error = run_bench_process(
            binary,
            input_path,
            count=count,
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
    count: int,
    bits: int,
    backend: str,
    extra_args: Sequence[str],
    seed: int,
    warmups: int,
    reps: int,
    k1: str = "reference",
    creationflags: int = 0,
    process: BenchProcess = REAL_BENCH_PROCESS,
) -> tuple[ValidatedBenchResponse | None, str | None]:
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
    try:
        response = validate_bench_response(
            payload,
            count=count,
            bits=bits,
            backend=backend,
            extra_args=extra_args,
            seed=seed,
            warmups=warmups,
            reps=reps,
            k1=k1,
        )
    except BenchResponseError as exc:
        return None, str(exc)
    return response, None


_STAGE_KEYS = ("k1_ms", "k2_ms", "k3_ms", "h2d_ms", "d2h_ms", "cpu_ms")


def _option_value(extra_args: Sequence[str], option: str, default: str | None = None) -> str | None:
    value = default
    i = 0
    while i < len(extra_args):
        arg = extra_args[i]
        if arg == option:
            if i + 1 >= len(extra_args):
                raise BenchResponseError(f"requested {option} has no value")
            value = extra_args[i + 1]
            i += 2
            continue
        if arg.startswith(option + "="):
            value = arg[len(option) + 1 :]
        i += 1
    return value


def _expected_configuration(
    *,
    count: int,
    bits: int,
    backend: str,
    extra_args: Sequence[str],
    seed: int,
    warmups: int,
    reps: int,
    k1: str,
) -> dict[str, Any]:
    defaults = {
        "cpu": "host-host",
        "cpu-avx2": "host-host",
        "cuda": CUDA_RESIDENT.boundary,
    }
    if backend not in defaults:
        raise BenchResponseError(f"requested backend {backend!r} is unsupported")
    boundary = _option_value(extra_args, "--boundary", defaults[backend])
    policy = _option_value(extra_args, "--transfer-policy")
    transfers = boundary in {"host-origin", "gpu-origin"}
    policy = policy or ("pageable" if transfers else "none")
    if policy not in {"none", "pageable", "pinned"}:
        raise BenchResponseError(f"requested transfer policy {policy!r} is unsupported")
    if (not transfers and policy != "none") or (transfers and policy == "none"):
        raise BenchResponseError(f"requested transfer policy {policy!r} has no transfer boundary")

    step = 0 if boundary == "resident-graph" else 1
    expected: dict[str, Any] = {
        "backend": backend,
        "bits": bits,
        "count": count,
        "seed": seed,
        "tensor_id": 0,
        "invocation_id": 0,
        "warmup": warmups,
        "reps": reps,
        "repetition_invocation_ids": [i * step for i in range(reps)],
        "warmup_invocation_ids": [(reps + i) * step for i in range(warmups)],
        "boundary": boundary,
        "transfer_policy": policy,
        "block_size": int(_option_value(extra_args, "--block-size", "256") or "256"),
        "grid_size": int(_option_value(extra_args, "--grid-size", "0") or "0"),
        "prescribed_scale": None,
    }
    if backend == "cuda":
        expected["k1"] = k1
    if backend == "cpu-avx2":
        requested_threads = _option_value(extra_args, "--threads")
        expected["threads"] = int(requested_threads) if requested_threads is not None else None
    return expected


def _finite_nonnegative_samples(
    payload: dict[str, Any], field: str, count: int
) -> list[int | float]:
    if field not in payload:
        raise BenchResponseError(f"stoquant bench response is missing {field}")
    samples = payload.get(field)
    if not isinstance(samples, list):
        raise BenchResponseError(f"stoquant bench response {field} must be an array")
    if len(samples) != count:
        raise BenchResponseError(
            f"stoquant bench response {field} has {len(samples)} samples; expected {count}"
        )
    for i, sample in enumerate(samples):
        if isinstance(sample, bool) or not isinstance(sample, (int, float)):
            raise BenchResponseError(f"stoquant bench response {field}[{i}] must be a number")
        if not _valid_timing(sample):
            raise BenchResponseError(
                f"stoquant bench response {field}[{i}] must be finite and nonnegative"
            )
    return samples


def _valid_timing(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        return False
    return isinstance(value, int) or math.isfinite(value)


def _validate_echoed_value(configuration: dict[str, Any], field: str, expected: Any) -> None:
    if field not in configuration:
        raise BenchResponseError(f"stoquant bench response configuration is missing {field}")
    actual = configuration[field]
    if field in {"backend", "boundary", "transfer_policy", "k1"}:
        valid_type = isinstance(actual, str)
    elif field in {"repetition_invocation_ids", "warmup_invocation_ids"}:
        valid_type = isinstance(actual, list) and all(type(item) is int for item in actual)
    elif field == "prescribed_scale":
        valid_type = actual is None
    else:
        valid_type = type(actual) is int
    if not valid_type or actual != expected:
        raise BenchResponseError(
            f"stoquant bench response configuration {field} is {actual!r}; expected {expected!r}"
        )


def _required_stages(backend: str, boundary: str) -> tuple[str, ...]:
    if backend == "cuda":
        if boundary == "resident-graph":
            return ()
        if boundary == "host-origin":
            return ("k1_ms", "k2_ms", "k3_ms", "h2d_ms", "d2h_ms")
        if boundary == "gpu-origin":
            return ("k1_ms", "k2_ms", "k3_ms", "d2h_ms")
        if boundary == "resident":
            return ("k1_ms", "k2_ms", "k3_ms")
    if backend == "cpu" and boundary == "gpu-origin":
        return ("d2h_ms", "cpu_ms")
    if backend == "cpu" and boundary == "host-host":
        return ()
    if backend == "cpu-avx2" and boundary == "host-host":
        return ()
    raise BenchResponseError(
        f"stoquant bench response has unsupported backend/boundary {backend!r}/{boundary!r}"
    )


def validate_bench_response(
    payload: dict[str, Any],
    *,
    count: int,
    bits: int,
    backend: str,
    extra_args: Sequence[str],
    seed: int,
    warmups: int,
    reps: int,
    k1: str = "reference",
) -> ValidatedBenchResponse:
    """Validate the complete native response against the exact requested run."""
    expected = _expected_configuration(
        count=count,
        bits=bits,
        backend=backend,
        extra_args=extra_args,
        seed=seed,
        warmups=warmups,
        reps=reps,
        k1=k1,
    )
    configuration = payload.get("configuration")
    if not isinstance(configuration, dict):
        raise BenchResponseError("stoquant bench response configuration must be an object")
    if backend != "cuda" and "k1" in configuration:
        raise BenchResponseError(
            "stoquant bench response configuration has k1 for a non-CUDA backend"
        )
    if backend != "cpu-avx2" and "threads" in configuration:
        raise BenchResponseError(
            "stoquant bench response configuration has threads for a non-AVX2 backend"
        )
    for key, value in expected.items():
        if key == "threads" and value is None:
            if (
                key not in configuration
                or type(configuration[key]) is not int
                or configuration[key] <= 0
            ):
                raise BenchResponseError(
                    "stoquant bench response configuration threads must be a positive integer"
                )
        else:
            _validate_echoed_value(configuration, key, value)

    samples = _finite_nonnegative_samples(payload, "samples_ms", reps)
    stages = _required_stages(backend, expected["boundary"])
    stage_samples = {field: _finite_nonnegative_samples(payload, field, reps) for field in stages}
    unexpected_stages = [field for field in _STAGE_KEYS if field in payload and field not in stages]
    if unexpected_stages:
        raise BenchResponseError(
            "stoquant bench response has unexpected stage arrays: " + ", ".join(unexpected_stages)
        )

    capture = payload.get("capture_and_instantiate_ms")
    if expected["boundary"] == "resident-graph":
        if "capture_and_instantiate_ms" not in payload:
            raise BenchResponseError(
                "stoquant bench response is missing capture_and_instantiate_ms for resident-graph"
            )
        if isinstance(capture, bool) or not isinstance(capture, (int, float)):
            raise BenchResponseError(
                "stoquant bench response capture_and_instantiate_ms must be a number"
            )
        if not _valid_timing(capture):
            raise BenchResponseError(
                "stoquant bench response capture_and_instantiate_ms must be finite and nonnegative"
            )
    elif "capture_and_instantiate_ms" in payload:
        raise BenchResponseError(
            "stoquant bench response has capture_and_instantiate_ms outside resident-graph"
        )

    return ValidatedBenchResponse(payload, configuration, samples, stage_samples, capture)


def in_process_warmups(minimum: int, seconds: float, rep_ms: float) -> int:
    """Warm-up count that covers `seconds` at `rep_ms` per repetition, never below `minimum`."""
    if seconds <= 0 or rep_ms <= 0:
        return minimum
    return max(minimum, math.ceil(seconds * 1000.0 / rep_ms))
