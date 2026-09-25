"""Benchmark driver for in-process mco2 CPU comparator and CUDA pipelines.

Orchestrates input generation, per-case correctness gating, timed benchmark
runs, provenance collection, statistical summary computation, and snapshot
creation as specified by the course technical contract and benchmark protocol.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import os
import platform
import shutil
import struct
import subprocess
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from mco2_oracle import decode_record, reference_fp64

HEADER_STRUCT = struct.Struct("<4sBBHQf")
DEFAULT_COUNTS = (1 << 10, 1 << 14, 1 << 18, 1 << 22)
DEFAULT_BITS = (4, 8)
DEFAULT_WARMUPS = 10
DEFAULT_REPS = 30
DEFAULT_INPUT_SEED = 2026
DEFAULT_COMPRESSION_SEED = 42
# Six trials run every ordering of the three paths once, balancing both the
# position of each path and the path that precedes it.
DEFAULT_TRIALS = 6

# Claim rule, fixed before any snapshot is generated. A CUDA speedup or
# slowdown is claimable only when its trial-median range excludes 1.0, its
# boundaries are not inverted, and neither it nor the comparator has a
# between-trial median spread above SPREAD_THRESHOLD.
SPREAD_THRESHOLD = 1.25
QUANTILE_METHOD = "linear"
CLAIM_RULE = (
    "A CUDA case supports a speedup or slowdown claim only when (1) its verdict is "
    "'faster' (CPU trial-median minimum / CUDA trial-median maximum > 1) or 'slower' "
    "(CPU trial-median maximum / CUDA trial-median minimum < 1), (2) boundary_inversion "
    "is false (host-origin pooled median is not below resident pooled median), and "
    f"(3) both the CUDA case and the CPU comparator have spread_ratio <= {SPREAD_THRESHOLD} "
    "(maximum over minimum trial median)."
)

GPU_STATE_FIELDS = (
    "pstate",
    "clocks.sm",
    "clocks.mem",
    "clocks.max.sm",
    "temperature.gpu",
    "power.draw",
    "clocks_event_reasons.active",
)
BUILD_RECIPE = "build-cuda"


def find_binary(root: Path) -> Path:
    binary_name = "mco2.exe" if os.name == "nt" else "mco2"
    path = root / "build" / binary_name
    if not path.is_file():
        raise FileNotFoundError(
            f"mco2 binary not found at {path}. Build it first with just build-cuda."
        )
    return path


def run_command(args: Sequence[str], cwd: Path) -> str:
    result = subprocess.run(
        args,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(f"Command {' '.join(args)} failed (exit {result.returncode}): {detail}")
    return result.stdout.strip()


def collect_git_provenance(root: Path) -> dict[str, Any]:
    try:
        rev = run_command(["git", "rev-parse", "HEAD"], root)
        short_rev = run_command(["git", "rev-parse", "--short", "HEAD"], root)
        status_output = run_command(["git", "status", "--porcelain"], root)
        dirty_files = [line for line in status_output.splitlines() if line.strip()]
    except (subprocess.SubprocessError, OSError, RuntimeError):
        rev = "unknown"
        short_rev = "unknown"
        dirty_files = ["<git status unavailable>"]
    return {
        "code_revision": rev,
        "code_revision_short": short_rev,
        "git_dirty": bool(dirty_files),
        "dirty_files": dirty_files,
    }


def collect_build_commands(root: Path) -> dict[str, Any]:
    """Read the exact compile commands from the build recipe via `just --dry-run`."""
    just = shutil.which("just")
    commands: list[str] = []
    if just is not None:
        proc = subprocess.run(
            [just, "--dry-run", BUILD_RECIPE],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode == 0:
            # just echoes dry-run commands on stderr.
            text = proc.stderr if proc.stderr.strip() else proc.stdout
            commands = [line.strip() for line in text.splitlines() if line.strip()]

    host_flags: list[str] | None = None
    nvcc_flags: list[str] | None = None
    for command in commands:
        tokens = command.split()
        if tokens and tokens[0].endswith("with-msvc.ps1"):
            tokens = tokens[1:]
        if not tokens:
            continue
        tool = Path(tokens[0]).name.lower()
        compiles = any(t.lower() in ("/c", "-c") for t in tokens)
        if host_flags is None and compiles and tool in ("cl.exe", "cl", "gcc", "cc", "clang"):
            host_flags = strip_compile_io(tokens[1:])
        elif nvcc_flags is None and compiles and tool in ("nvcc", "nvcc.exe"):
            nvcc_flags = strip_compile_io(tokens[1:])

    return {
        "source": f"just --dry-run {BUILD_RECIPE}",
        "commands": commands,
        "comparator_c": " ".join(host_flags) if host_flags is not None else "unknown",
        "cuda_nvcc": " ".join(nvcc_flags) if nvcc_flags is not None else "unknown",
        "_host_tokens": host_flags,
    }


def strip_compile_io(tokens: Sequence[str]) -> list[str]:
    """Drop the compile switch, source file, and output path, keeping the flags."""
    kept: list[str] = []
    skip_next = False
    for token in tokens:
        if skip_next:
            skip_next = False
            continue
        lowered = token.lower()
        if lowered in ("/c", "-c"):
            continue
        if lowered == "-o":
            skip_next = True
            continue
        if lowered.startswith("/fo"):
            continue
        if lowered.endswith((".c", ".cu", ".obj", ".o")):
            continue
        kept.append(token)
    return kept


def query_gpu_state() -> dict[str, str]:
    """Record the GPU's clock, power, and thermal state at one instant."""
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu={','.join(GPU_STATE_FIELDS)}",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    except (subprocess.SubprocessError, OSError):
        return {"status": "unavailable"}
    if proc.returncode != 0 or not proc.stdout.strip():
        return {"status": "unavailable"}
    values = [v.strip() for v in proc.stdout.strip().splitlines()[0].split(",")]
    if len(values) != len(GPU_STATE_FIELDS):
        return {"status": "unparsed", "raw": proc.stdout.strip()}
    state = dict(zip(GPU_STATE_FIELDS, values, strict=True))
    state["captured_at_utc"] = datetime.now(UTC).isoformat()
    return state


def collect_hardware_and_toolchain(root: Path) -> dict[str, Any]:
    # Host CPU
    host_cpu = platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER") or platform.machine()

    # GPU name, compute cap, driver
    gpu_name = "unknown"
    driver_version = "unknown"
    compute_cap = "unknown"
    try:
        smi_out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,compute_cap", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=False,
        )
        if smi_out.returncode == 0 and smi_out.stdout.strip():
            parts = [p.strip() for p in smi_out.stdout.strip().splitlines()[0].split(",")]
            if len(parts) >= 3:
                gpu_name, driver_version, compute_cap = parts[0], parts[1], parts[2]
    except (subprocess.SubprocessError, OSError, IndexError):
        pass

    # CUDA toolkit version
    cuda_toolkit = "unknown"
    try:
        nvcc_out = subprocess.run(
            ["nvcc", "--version"],
            capture_output=True,
            text=True,
            check=False,
        )
        if nvcc_out.returncode == 0 and nvcc_out.stdout.strip():
            for line in nvcc_out.stdout.splitlines():
                if "release" in line:
                    cuda_toolkit = line.strip()
                    break
    except (subprocess.SubprocessError, OSError):
        pass

    # C compiler version
    c_compiler = "unknown"
    try:
        if os.name == "nt":
            script = root / "scripts" / "with-msvc.ps1"
            if script.is_file():
                cl_proc = subprocess.run(
                    [
                        "powershell.exe",
                        "-NoProfile",
                        "-ExecutionPolicy",
                        "Bypass",
                        "-File",
                        str(script),
                        "cl.exe",
                    ],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                output = cl_proc.stderr or cl_proc.stdout
                for line in output.splitlines():
                    if "Microsoft (R) C/C++" in line or "Optimizing Compiler" in line:
                        c_compiler = line.strip()
                        break
        else:
            cc = os.environ.get("CC", "gcc")
            cc_out = subprocess.run([cc, "--version"], capture_output=True, text=True, check=False)
            if cc_out.returncode == 0:
                c_compiler = cc_out.stdout.splitlines()[0].strip()
    except (subprocess.SubprocessError, OSError):
        pass

    return {
        "hardware": {
            "gpu_name": gpu_name,
            "compute_capability": compute_cap,
            "host_cpu": host_cpu,
        },
        "toolkit_and_driver": {
            "cuda_toolkit": cuda_toolkit,
            "driver_version": driver_version,
            "c_compiler": c_compiler,
        },
        "transfer_policy": "pageable",
    }


def generate_inputs(
    counts: Sequence[int],
    directory: Path,
    seed: int = DEFAULT_INPUT_SEED,
) -> dict[int, dict[str, Any]]:
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    provenance: dict[int, dict[str, Any]] = {}

    for count in counts:
        values = rng.normal(size=count).astype(np.float32)
        bytes_data = values.astype("<f4").tobytes()
        sha256 = hashlib.sha256(bytes_data).hexdigest()
        file_path = directory / f"input_n{count}.f32"
        file_path.write_bytes(bytes_data)

        provenance[count] = {
            "count": count,
            "filename": file_path.name,
            "generator": "numpy.random.default_rng",
            "bit_generator": type(rng.bit_generator).__name__,
            "seed": seed,
            "sha256": sha256,
            "_path": str(file_path),
        }

    return provenance


def generate_msvc_vectorization_report(
    root: Path,
    output_file: Path,
    host_flags: Sequence[str] | None,
    object_dir: Path,
) -> str:
    """Recompile the comparator's C sources with its exact build flags plus /Qvec-report:2."""
    output_text = ""
    script = root / "scripts" / "with-msvc.ps1"
    if os.name == "nt" and script.is_file() and host_flags is not None:
        object_dir.mkdir(parents=True, exist_ok=True)
        # A quoted path ending in "\" escapes its closing quote through the
        # PowerShell wrapper, so the directory ends in "/" instead.
        try:
            object_arg = str(object_dir.resolve().relative_to(root.resolve()))
        except ValueError:
            object_arg = str(object_dir.resolve())
        object_arg = object_arg.replace("\\", "/")
        compile_args = [
            "cl.exe",
            *host_flags,
            "/Qvec-report:2",
            "/c",
            "src\\main.c",
            "src\\codec.c",
            "src\\quantizer.c",
            "src\\rng_cpu.c",
            f"/Fo:{object_arg}/",
        ]
        cmd = [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(script),
            *compile_args,
        ]
        res = subprocess.run(cmd, cwd=root, capture_output=True, text=True, check=False)
        diagnostics = (res.stdout + "\n" + res.stderr).strip()
        # cl reports absolute source paths; keep the published report root-relative.
        diagnostics = diagnostics.replace(str(root.resolve()) + "\\", "")
        output_text = (
            f"Command: {' '.join(compile_args)}\nExit code: {res.returncode}\n\n" + diagnostics
        )

    if not output_text:
        output_text = "MSVC vectorization report not available on this platform/configuration."

    output_file.write_text(output_text, encoding="utf-8")
    return output_text


def verify_correctness(
    binary: Path,
    input_path: Path,
    count: int,
    bits: int,
    *,
    backends: Sequence[str] = ("cpu", "cuda"),
    seed: int = DEFAULT_COMPRESSION_SEED,
    tensor_id: int = 0,
    invocation_id: int = 0,
    tmp_dir: Path,
    force_fail: bool = False,
) -> tuple[bool, dict[str, Any]]:
    """Gating check before timing each case.

    Checks:
    1. CPU bench record == CPU compress record.
    2. If CUDA backend enabled: CUDA bench record == CUDA compress record
       and CPU record == CUDA record bit-for-bit.
    3. Decoded record satisfies Layer 2 mathematical bounds against FP64 oracle.
    """
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
    cuda_bench_res_path = tmp_dir / f"cuda_bench_res_b{bits}_n{count}.msq"
    cuda_bench_ho_path = tmp_dir / f"cuda_bench_ho_b{bits}_n{count}.msq"

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

    if include_cuda:
        # CUDA compress
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

        # CUDA bench resident --record-output
        res_cuda_res = subprocess.run(
            [
                str(binary),
                "bench",
                "--input",
                str(input_path),
                "--record-output",
                str(cuda_bench_res_path),
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
                "--boundary",
                "resident",
                "--warmup",
                "0",
                "--reps",
                "1",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if res_cuda_res.returncode != 0:
            return False, {
                "status": "failed",
                "error_message": f"CUDA resident bench record failed: {res_cuda_res.stderr.strip()}",
            }

        # CUDA bench host-origin --record-output
        res_cuda_ho = subprocess.run(
            [
                str(binary),
                "bench",
                "--input",
                str(input_path),
                "--record-output",
                str(cuda_bench_ho_path),
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
                "--boundary",
                "host-origin",
                "--warmup",
                "0",
                "--reps",
                "1",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if res_cuda_ho.returncode != 0:
            return False, {
                "status": "failed",
                "error_message": f"CUDA host-origin bench record failed: {res_cuda_ho.stderr.strip()}",
            }

        cuda_comp_bytes = cuda_comp_path.read_bytes()
        cuda_res_bytes = cuda_bench_res_path.read_bytes()
        cuda_ho_bytes = cuda_bench_ho_path.read_bytes()

        byte_identical_to_compress = (
            cuda_res_bytes == cuda_comp_bytes and cuda_ho_bytes == cuda_comp_bytes
        )
        cpu_cuda_byte_identical = cpu_comp_bytes == cuda_comp_bytes

        if not byte_identical_to_compress:
            return False, {
                "status": "failed",
                "byte_identical_to_compress": False,
                "cpu_cuda_byte_identical": cpu_cuda_byte_identical,
                "error_message": "CUDA bench record output is not byte-identical to compress output.",
            }

        if not cpu_cuda_byte_identical:
            return False, {
                "status": "failed",
                "byte_identical_to_compress": True,
                "cpu_cuda_byte_identical": False,
                "error_message": "CPU and CUDA records are not byte-identical.",
            }
    else:
        cpu_cuda_byte_identical = True

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


def compute_statistics(samples_ms: Sequence[float]) -> dict[str, float]:
    arr = np.asarray(samples_ms, dtype=np.float64)
    if len(arr) == 0:
        return {}
    q25 = float(np.percentile(arr, 25, method=QUANTILE_METHOD))
    q75 = float(np.percentile(arr, 75, method=QUANTILE_METHOD))
    return {
        "median_ms": float(np.median(arr)),
        "iqr_ms": float(q75 - q25),
        "q25_ms": q25,
        "q75_ms": q75,
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
        "mean_ms": float(np.mean(arr)),
        "std_ms": float(np.std(arr)),
    }


def compute_case_statistics(trial_samples: Sequence[Sequence[float]]) -> dict[str, Any]:
    """Pooled statistics plus the between-trial spread of per-trial medians."""
    pooled = [sample for samples in trial_samples for sample in samples]
    stats: dict[str, Any] = compute_statistics(pooled)
    trial_medians = [float(np.median(np.asarray(s, dtype=np.float64))) for s in trial_samples]
    low = min(trial_medians)
    high = max(trial_medians)
    stats["trial_medians_ms"] = trial_medians
    stats["trial_median_min_ms"] = low
    stats["trial_median_max_ms"] = high
    stats["spread_ratio"] = high / low if low > 0 else float("inf")
    stats["unstable"] = stats["spread_ratio"] > SPREAD_THRESHOLD
    return stats


def compare_to_comparator(cpu_stats: dict[str, Any], cuda_stats: dict[str, Any]) -> dict[str, Any]:
    """Point speedup from pooled medians and a conservative range from trial medians."""
    point = cpu_stats["median_ms"] / cuda_stats["median_ms"]
    low = cpu_stats["trial_median_min_ms"] / cuda_stats["trial_median_max_ms"]
    high = cpu_stats["trial_median_max_ms"] / cuda_stats["trial_median_min_ms"]
    if low > 1.0:
        verdict = "faster"
    elif high < 1.0:
        verdict = "slower"
    else:
        verdict = "inconclusive"
    return {
        "speedup_vs_cpu": point,
        "speedup_low": low,
        "speedup_high": high,
        "verdict": verdict,
    }


def trial_orders(paths: Sequence[tuple[str, str, list[str]]], trials: int) -> list[list[int]]:
    """Cycle through every ordering of the paths in a fixed lexicographic sequence."""
    perms = [list(p) for p in itertools.permutations(range(len(paths)))]
    return [perms[t % len(perms)] for t in range(trials)]


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
        "--warmup",
        str(warmups),
        "--reps",
        str(reps),
        *extra_args,
    ]
    proc = subprocess.run(bench_cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        return None, f"mco2 bench failed: {proc.stderr.strip()}"
    return json.loads(proc.stdout), None


def failed_row(
    count: int, bits: int, backend: str, boundary: str, warmups: int, reps: int, trials: int
) -> dict[str, Any]:
    row = dict.fromkeys(SUMMARY_FIELDS, "")
    row.update(
        {
            "count": count,
            "bits": bits,
            "backend": backend,
            "boundary": boundary,
            "correctness": "failed",
            "warmup": warmups,
            "reps": reps,
            "trials": trials,
        }
    )
    return row


SUMMARY_FIELDS = [
    "count",
    "bits",
    "backend",
    "boundary",
    "correctness",
    "warmup",
    "reps",
    "trials",
    "median_ms",
    "iqr_ms",
    "trial_median_min_ms",
    "trial_median_max_ms",
    "spread_ratio",
    "speedup_vs_c",
    "speedup_low",
    "speedup_high",
    "verdict",
    "boundary_inversion",
    "unstable",
    "claim_supported",
]


def run_benchmark_matrix(
    *,
    root: Path,
    output_dir: Path | None = None,
    counts: Sequence[int] = DEFAULT_COUNTS,
    bit_widths: Sequence[int] = DEFAULT_BITS,
    backends: Sequence[str] = ("cpu", "cuda"),
    warmups: int = DEFAULT_WARMUPS,
    reps: int = DEFAULT_REPS,
    trials: int = DEFAULT_TRIALS,
    input_seed: int = DEFAULT_INPUT_SEED,
    compression_seed: int = DEFAULT_COMPRESSION_SEED,
    force_fail: bool = False,
    allow_existing: bool = False,
    allow_dirty: bool = False,
) -> Path:
    if trials < 1:
        raise ValueError("trials must be at least 1")
    if "cpu" not in backends:
        raise ValueError("the CPU comparator is required; include 'cpu' in backends")

    git_prov = collect_git_provenance(root)

    # Determine snapshot directory name
    date_str = datetime.now(UTC).strftime("%Y-%m-%d")
    short_rev = git_prov["code_revision_short"]
    if output_dir is None:
        target_dir = root / "results" / f"{date_str}-{short_rev}"
    else:
        target_dir = output_dir

    if target_dir.exists() and not allow_existing:
        raise FileExistsError(
            f"Snapshot directory already exists: {target_dir}. "
            "Snapshots are frozen and never overwritten. "
            "Pass --allow-existing or an explicit --output-dir if intentional."
        )

    if git_prov["git_dirty"] and not allow_dirty:
        raise RuntimeError(
            "Working tree is dirty; a snapshot must be traceable to a committed revision. "
            "Commit or remove these changes, or pass --allow-dirty for a non-evidence run: "
            + "; ".join(git_prov["dirty_files"])
        )

    binary = find_binary(root)
    toolchain_prov = collect_hardware_and_toolchain(root)
    build_prov = collect_build_commands(root)
    host_tokens = build_prov.pop("_host_tokens")
    gpu_state_start = query_gpu_state() if "cuda" in backends else None

    target_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = target_dir / "_temp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    # 1. Inputs generation
    input_meta = generate_inputs(counts, temp_dir / "inputs", seed=input_seed)
    input_files: dict[int, Path] = {c: Path(meta["_path"]) for c, meta in input_meta.items()}
    clean_input_meta: dict[int, dict[str, Any]] = {
        c: {k: v for k, v in meta.items() if not k.startswith("_")}
        for c, meta in input_meta.items()
    }

    # 2. Vectorization report, compiled with the comparator's own flags
    vec_report_path = target_dir / "msvc_vectorization_report.txt"
    generate_msvc_vectorization_report(root, vec_report_path, host_tokens, temp_dir / "vec_obj")

    paths: list[tuple[str, str, list[str]]] = [("cpu", "comparator", [])]
    if "cuda" in backends:
        paths.append(("cuda", "resident", ["--boundary", "resident"]))
        paths.append(("cuda", "host-origin", ["--boundary", "host-origin"]))
    orders = trial_orders(paths, trials)
    order_labels = [[f"{paths[i][0]}-{paths[i][1]}" for i in order] for order in orders]

    # 3. Benchmark cases
    case_results: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for count in counts:
        input_path = input_files[count]

        for bits in bit_widths:
            payload_bytes = count if bits == 8 else (count + 1) // 2

            # Gate on correctness once, outside every timed process.
            passed, correctness_info = verify_correctness(
                binary=binary,
                input_path=input_path,
                count=count,
                bits=bits,
                backends=backends,
                seed=compression_seed,
                tensor_id=0,
                invocation_id=0,
                tmp_dir=temp_dir,
                force_fail=force_fail,
            )

            cases: list[dict[str, Any]] = []
            for backend, boundary, _ in paths:
                cases.append(
                    {
                        "case_id": f"case_{backend}_{boundary}_bits{bits}_n{count}",
                        "count": count,
                        "bits": bits,
                        "backend": backend,
                        "timing_boundary": boundary,
                        "transfer_policy": toolchain_prov["transfer_policy"],
                        "seed": compression_seed,
                        "tensor_id": 0,
                        "invocation_id": 0,
                        "warmup": warmups,
                        "reps": reps,
                        "trials": trials,
                        "header_bytes": HEADER_STRUCT.size,
                        "payload_bytes": payload_bytes,
                        "code_revision": git_prov["code_revision"],
                        "code_revision_short": git_prov["code_revision_short"],
                        "git_dirty": git_prov["git_dirty"],
                        "build_flags": {
                            "comparator_c": build_prov["comparator_c"],
                            "cuda_nvcc": build_prov["cuda_nvcc"],
                        },
                        "hardware": toolchain_prov["hardware"],
                        "toolkit_and_driver": toolchain_prov["toolkit_and_driver"],
                        "input_provenance": clean_input_meta[count],
                        "correctness": correctness_info,
                        "trial_runs": [],
                        "samples_ms": [],
                        "statistics": None,
                    }
                )

            if passed:
                # Each trial runs every path as its own process, in a rotated order.
                # Every trial reuses the same invocation identifiers, so trials are
                # replicates of an identical workload.
                for trial_index, order in enumerate(orders):
                    for position, path_index in enumerate(order):
                        case = cases[path_index]
                        if case["correctness"]["status"] != "passed":
                            continue
                        backend, _, extra_args = paths[path_index]
                        payload, error = run_bench_process(
                            binary,
                            input_path,
                            bits=bits,
                            backend=backend,
                            extra_args=extra_args,
                            seed=compression_seed,
                            warmups=warmups,
                            reps=reps,
                        )
                        if payload is None:
                            case["correctness"] = {"status": "failed", "error_message": error}
                            continue
                        configuration = payload.get("configuration", {})
                        run: dict[str, Any] = {
                            "trial": trial_index,
                            "position": position,
                            "order": order_labels[trial_index],
                            "repetition_invocation_ids": configuration.get(
                                "repetition_invocation_ids"
                            ),
                            "warmup_invocation_ids": configuration.get("warmup_invocation_ids"),
                            "samples_ms": payload.get("samples_ms", []),
                        }
                        for key in ("k1_ms", "k2_ms", "k3_ms"):
                            if key in payload:
                                run[key] = payload[key]
                        case["trial_runs"].append(run)

            for case in cases:
                if case["correctness"]["status"] != "passed":
                    case["trial_runs"] = []
                    continue
                runs = case["trial_runs"]
                first = runs[0]
                for run in runs[1:]:
                    if (
                        run["repetition_invocation_ids"] != first["repetition_invocation_ids"]
                        or run["warmup_invocation_ids"] != first["warmup_invocation_ids"]
                    ):
                        raise RuntimeError(
                            f"{case['case_id']}: trials used different invocation identifiers"
                        )
                case["repetition_invocation_ids"] = first["repetition_invocation_ids"]
                case["warmup_invocation_ids"] = first["warmup_invocation_ids"]
                case["samples_ms"] = [s for run in runs for s in run["samples_ms"]]
                case["statistics"] = compute_case_statistics([run["samples_ms"] for run in runs])

            # Comparisons and flags, only between cases that all passed.
            cpu_case = cases[0]
            by_boundary = {c["timing_boundary"]: c for c in cases}
            inversion = False
            resident = by_boundary.get("resident")
            host_origin = by_boundary.get("host-origin")
            if (
                resident is not None
                and host_origin is not None
                and resident["statistics"] is not None
                and host_origin["statistics"] is not None
            ):
                inversion = (
                    host_origin["statistics"]["median_ms"] < resident["statistics"]["median_ms"]
                )

            for case in cases:
                stats = case["statistics"]
                if stats is None:
                    continue
                if case is cpu_case:
                    stats.update(
                        {
                            "speedup_vs_cpu": 1.0,
                            "speedup_low": 1.0,
                            "speedup_high": 1.0,
                            "verdict": "comparator",
                            "boundary_inversion": False,
                            "claim_supported": None,
                        }
                    )
                    continue
                cpu_stats = cpu_case["statistics"]
                if cpu_stats is None:
                    continue
                stats.update(compare_to_comparator(cpu_stats, stats))
                stats["boundary_inversion"] = inversion
                stats["claim_supported"] = (
                    stats["verdict"] != "inconclusive"
                    and not inversion
                    and not stats["unstable"]
                    and not cpu_stats["unstable"]
                )

            for case in cases:
                (target_dir / f"{case['case_id']}.json").write_text(
                    json.dumps(case, indent=2), encoding="utf-8"
                )
                case_results.append(case)
                stats = case["statistics"]
                if (
                    case["correctness"]["status"] != "passed"
                    or stats is None
                    or "verdict" not in stats
                ):
                    summary_rows.append(
                        failed_row(
                            count,
                            bits,
                            case["backend"],
                            case["timing_boundary"],
                            warmups,
                            reps,
                            trials,
                        )
                    )
                    continue
                claim = stats["claim_supported"]
                summary_rows.append(
                    {
                        "count": count,
                        "bits": bits,
                        "backend": case["backend"],
                        "boundary": case["timing_boundary"],
                        "correctness": "passed",
                        "warmup": warmups,
                        "reps": reps,
                        "trials": trials,
                        "median_ms": f"{stats['median_ms']:.6f}",
                        "iqr_ms": f"{stats['iqr_ms']:.6f}",
                        "trial_median_min_ms": f"{stats['trial_median_min_ms']:.6f}",
                        "trial_median_max_ms": f"{stats['trial_median_max_ms']:.6f}",
                        "spread_ratio": f"{stats['spread_ratio']:.4f}",
                        "speedup_vs_c": f"{stats['speedup_vs_cpu']:.4f}",
                        "speedup_low": f"{stats['speedup_low']:.4f}",
                        "speedup_high": f"{stats['speedup_high']:.4f}",
                        "verdict": stats["verdict"],
                        "boundary_inversion": str(stats["boundary_inversion"]).lower(),
                        "unstable": str(stats["unstable"]).lower(),
                        "claim_supported": "" if claim is None else str(claim).lower(),
                    }
                )

    gpu_state_end = query_gpu_state() if "cuda" in backends else None

    shutil.rmtree(temp_dir, ignore_errors=True)

    # 4. Summary CSV
    csv_path = target_dir / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(summary_rows)

    # 5. Run manifest
    manifest_data = {
        "manifest_version": "2.0",
        "date": date_str,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "git_provenance": git_prov,
        "hardware": toolchain_prov["hardware"],
        "toolkit_and_driver": toolchain_prov["toolkit_and_driver"],
        "build_flags": build_prov,
        "transfer_policy": toolchain_prov["transfer_policy"],
        "gpu_state": {
            "note": (
                "Instantaneous nvidia-smi readings before the first and after the last "
                "timed process; they describe the GPU around the run, not clock stability "
                "during it."
            ),
            "start": gpu_state_start,
            "end": gpu_state_end,
        },
        "matrix_parameters": {
            "counts": list(counts),
            "bit_widths": list(bit_widths),
            "backends": list(backends),
            "warmup": warmups,
            "reps": reps,
            "trials": trials,
            "trial_orders": order_labels,
            "invocation_scheme": (
                "every trial reuses base invocation 0; identifiers per repetition are "
                "copied from each mco2 bench configuration"
            ),
            "input_seed": input_seed,
            "compression_seed": compression_seed,
        },
        "statistics_method": {
            "pooled": "median and IQR over all measured repetitions of all trials",
            "quantile_method": f"numpy.percentile(method='{QUANTILE_METHOD}')",
            "numpy_version": np.__version__,
            "speedup_point": "CPU pooled median / CUDA pooled median",
            "speedup_range": (
                "[CPU min trial median / CUDA max trial median, "
                "CPU max trial median / CUDA min trial median]"
            ),
            "spread_ratio": "max trial median / min trial median",
            "spread_threshold": SPREAD_THRESHOLD,
            "claim_rule": CLAIM_RULE,
        },
        "inputs": clean_input_meta,
        "summary_csv": csv_path.name,
        "msvc_vectorization_report": vec_report_path.name,
        "cases": [c["case_id"] for c in case_results],
        "all_cases_passed": all(c["correctness"]["status"] == "passed" for c in case_results),
    }

    manifest_path = target_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")

    return target_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="mco2 benchmark driver and snapshot creator")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory to save snapshot")
    parser.add_argument(
        "--counts", type=int, nargs="+", default=DEFAULT_COUNTS, help="Element counts"
    )
    parser.add_argument(
        "--bits", type=int, nargs="+", default=DEFAULT_BITS, help="Bit widths (4 or 8)"
    )
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUPS, help="Warmup runs")
    parser.add_argument("--reps", type=int, default=DEFAULT_REPS, help="Measured repetitions")
    parser.add_argument(
        "--trials",
        type=int,
        default=DEFAULT_TRIALS,
        help="Independent processes per path, each trial in a rotated path order",
    )
    parser.add_argument(
        "--input-seed", type=int, default=DEFAULT_INPUT_SEED, help="Seed for input generation"
    )
    parser.add_argument(
        "--compression-seed",
        type=int,
        default=DEFAULT_COMPRESSION_SEED,
        help="Seed for compression",
    )
    parser.add_argument(
        "--backends",
        type=str,
        nargs="+",
        default=["cpu", "cuda"],
        help="Backends to benchmark (cpu, cuda)",
    )
    parser.add_argument(
        "--force-fail", action="store_true", help="Force correctness failure for tests"
    )
    parser.add_argument(
        "--allow-existing", action="store_true", help="Allow writing into existing folder"
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Run from an uncommitted tree (records the dirty files; not for evidence)",
    )

    args = parser.parse_args()
    root = Path(__file__).resolve().parent

    try:
        snapshot_dir = run_benchmark_matrix(
            root=root,
            output_dir=args.output_dir,
            counts=args.counts,
            bit_widths=args.bits,
            backends=args.backends,
            warmups=args.warmup,
            reps=args.reps,
            trials=args.trials,
            input_seed=args.input_seed,
            compression_seed=args.compression_seed,
            force_fail=args.force_fail,
            allow_existing=args.allow_existing,
            allow_dirty=args.allow_dirty,
        )
        print(f"Benchmark completed successfully. Snapshot written to: {snapshot_dir}")
    except (RuntimeError, ValueError, OSError, subprocess.SubprocessError) as exc:
        print(f"Benchmark driver failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
