"""Benchmark driver for in-process mco2 CPU comparator and CUDA pipelines.

Orchestrates input generation, per-case correctness gating, timed benchmark
runs, provenance collection, statistical summary computation, and snapshot
creation as specified by the course technical contract and benchmark protocol.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import ctypes
import hashlib
import itertools
import json
import math
import os
import platform
import shutil
import struct
import subprocess
import sys
import time
from collections.abc import Iterator, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from mco2_oracle import decode_record, reference_fp64

HEADER_STRUCT = struct.Struct("<4sBBHQf")
DEFAULT_COUNTS = tuple(1 << exponent for exponent in range(10, 27))
DEFAULT_BITS = (4, 8)
DEFAULT_WARMUPS = 10
DEFAULT_REPS = 30
DEFAULT_INPUT_SEED = 2026
DEFAULT_COMPRESSION_SEED = 42
DEFAULT_CASE_ORDER_SEED = 612
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
STAGE_KEYS = ("k1_ms", "k2_ms", "k3_ms", "h2d_ms", "d2h_ms")
# Twenty-four trials run every ordering of the four paths once, balancing both the
# position of each path and the path that precedes it.
DEFAULT_TRIALS = 24

# Claim rules, fixed before any snapshot is generated (protocol revision 3).
# A direction claim (faster or slower) needs a conservative verdict and no
# boundary inversion. A magnitude claim also needs both sides stable, with the
# stability spread taken between the 10th and 90th percentiles of trial medians
# so one outlier process does not veto a case. The revision 2 rule, which gated
# every claim on the max/min spread, is still computed for comparison.
SPREAD_THRESHOLD = 1.25
STABILITY_THRESHOLD = 1.25
STABILITY_PERCENTILES = (10, 90)
QUANTILE_METHOD = "linear"
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 31
BOOTSTRAP_LEVEL = 0.95
VERDICT_RULE = (
    "verdict is 'faster' when baseline trial-median minimum / candidate trial-median maximum "
    "> 1, 'slower' when baseline trial-median maximum / candidate trial-median minimum < 1, "
    "and 'inconclusive' otherwise"
)
DIRECTION_RULE = (
    "direction_supported: verdict is not 'inconclusive' and boundary_inversion is false "
    "(host-origin pooled median is not below the pooled median of any resident path)"
)
MAGNITUDE_RULE = (
    "magnitude_supported: direction_supported and both the candidate and the baseline have "
    f"spread_p90_p10 <= {STABILITY_THRESHOLD} (90th over 10th percentile of trial medians, "
    f"numpy method '{QUANTILE_METHOD}')"
)
CLAIM_RULE_REV2 = (
    "claim_supported_rev2: verdict is not 'inconclusive', boundary_inversion is false, and "
    f"both the candidate and the baseline have spread_ratio <= {SPREAD_THRESHOLD} "
    "(maximum over minimum trial median)"
)
BOOTSTRAP_RULE = (
    f"{BOOTSTRAP_LEVEL:.0%} percentile bootstrap: resample each path's trial medians with "
    f"replacement, independently, {BOOTSTRAP_RESAMPLES} times from "
    f"numpy.random.default_rng({BOOTSTRAP_SEED}); the statistic is the baseline median of "
    "resampled trial medians over the candidate median of resampled trial medians"
)

# Every benchmark and probe process runs off physical core 0 (logical CPUs 0
# and 1), which ran about 30% slower in the issue #30 diagnosis.
EXCLUDED_LOGICAL_CPUS = (0, 1)

# Readiness: a sweep starts on a freshly rebooted, quiet machine.
MAX_UPTIME_SECONDS = 30 * 60
WINDOW_ALLOWLIST = (
    "claude",
    "explorer",
    "TextInputHost",
    "ShellExperienceHost",
    "StartMenuExperienceHost",
    "SearchHost",
    "LockApp",
)
# nvidia-smi clocks_event_reasons bits. GpuIdle is the only benign one.
CLOCK_EVENT_REASONS = {
    0x1: "GpuIdle",
    0x2: "ApplicationsClocksSetting",
    0x4: "SwPowerCap",
    0x8: "HwSlowdown",
    0x10: "SyncBoost",
    0x20: "SwThermalSlowdown",
    0x40: "HwThermalSlowdown",
    0x80: "HwPowerBrakeSlowdown",
    0x100: "DisplayClockSetting",
}
BENIGN_CLOCK_EVENTS = 0x1

# The pilot runs the sweep's code path on four sizes: the smallest, the
# launch-bound 2^14, a mid size, and the largest.
PILOT_COUNTS = (1 << 10, 1 << 14, 1 << 20, 1 << 26)

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
    cuda_bench_graph_path = tmp_dir / f"cuda_bench_graph_b{bits}_n{count}.msq"
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

        # CUDA bench resident-graph --record-output
        res_cuda_graph = subprocess.run(
            [
                str(binary),
                "bench",
                "--input",
                str(input_path),
                "--record-output",
                str(cuda_bench_graph_path),
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
                "resident-graph",
                "--warmup",
                "0",
                "--reps",
                "1",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if res_cuda_graph.returncode != 0:
            return False, {
                "status": "failed",
                "error_message": f"CUDA resident-graph bench record failed: {res_cuda_graph.stderr.strip()}",
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
        cuda_graph_bytes = cuda_bench_graph_path.read_bytes()
        cuda_ho_bytes = cuda_bench_ho_path.read_bytes()

        byte_identical_to_compress = (
            cuda_res_bytes == cuda_comp_bytes
            and cuda_graph_bytes == cuda_comp_bytes
            and cuda_ho_bytes == cuda_comp_bytes
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
    p_low, p_high = np.percentile(
        np.asarray(trial_medians, dtype=np.float64), STABILITY_PERCENTILES, method=QUANTILE_METHOD
    )
    stats["trial_medians_ms"] = trial_medians
    stats["trial_median_min_ms"] = low
    stats["trial_median_max_ms"] = high
    stats["spread_ratio"] = high / low if low > 0 else float("inf")
    stats["unstable_rev2"] = stats["spread_ratio"] > SPREAD_THRESHOLD
    stats["spread_p90_p10"] = float(p_high / p_low) if p_low > 0 else float("inf")
    stats["stable"] = stats["spread_p90_p10"] <= STABILITY_THRESHOLD
    return stats


def bootstrap_speedup_ci(
    baseline_medians: Sequence[float],
    candidate_medians: Sequence[float],
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
    level: float = BOOTSTRAP_LEVEL,
) -> tuple[float, float]:
    """Percentile bootstrap interval of baseline median / candidate median over trials."""
    rng = np.random.default_rng(seed)
    base = np.asarray(baseline_medians, dtype=np.float64)
    cand = np.asarray(candidate_medians, dtype=np.float64)
    base_draws = base[rng.integers(0, len(base), size=(resamples, len(base)))]
    cand_draws = cand[rng.integers(0, len(cand), size=(resamples, len(cand)))]
    ratios = np.median(base_draws, axis=1) / np.median(cand_draws, axis=1)
    tail = 100.0 * (1.0 - level) / 2.0
    low, high = np.percentile(ratios, (tail, 100.0 - tail), method=QUANTILE_METHOD)
    return float(low), float(high)


def compare_to_comparator(cpu_stats: dict[str, Any], cuda_stats: dict[str, Any]) -> dict[str, Any]:
    """Point speedup from pooled medians, a conservative range, and a bootstrap CI."""
    point = cpu_stats["median_ms"] / cuda_stats["median_ms"]
    low = cpu_stats["trial_median_min_ms"] / cuda_stats["trial_median_max_ms"]
    high = cpu_stats["trial_median_max_ms"] / cuda_stats["trial_median_min_ms"]
    if low > 1.0:
        verdict = "faster"
    elif high < 1.0:
        verdict = "slower"
    else:
        verdict = "inconclusive"
    ci_low, ci_high = bootstrap_speedup_ci(
        cpu_stats["trial_medians_ms"], cuda_stats["trial_medians_ms"]
    )
    return {
        "speedup_vs_cpu": point,
        "speedup_low": low,
        "speedup_high": high,
        "speedup_ci_low": ci_low,
        "speedup_ci_high": ci_high,
        "verdict": verdict,
    }


def compare_to_resident(
    resident_stats: dict[str, Any], graph_stats: dict[str, Any]
) -> dict[str, Any]:
    """Point speedup of resident-graph over plain resident, range, and bootstrap CI."""
    point = resident_stats["median_ms"] / graph_stats["median_ms"]
    low = resident_stats["trial_median_min_ms"] / graph_stats["trial_median_max_ms"]
    high = resident_stats["trial_median_max_ms"] / graph_stats["trial_median_min_ms"]
    if low > 1.0:
        verdict = "faster"
    elif high < 1.0:
        verdict = "slower"
    else:
        verdict = "inconclusive"
    ci_low, ci_high = bootstrap_speedup_ci(
        resident_stats["trial_medians_ms"], graph_stats["trial_medians_ms"]
    )
    return {
        "speedup_vs_resident": point,
        "speedup_low": low,
        "speedup_high": high,
        "speedup_ci_low": ci_low,
        "speedup_ci_high": ci_high,
        "verdict": verdict,
    }


def claim_support(
    verdict: str, inversion: bool, baseline: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, bool]:
    """The revision 3 direction and magnitude claims, plus the revision 2 claim."""
    direction = verdict != "inconclusive" and not inversion
    return {
        "direction_supported": direction,
        "magnitude_supported": direction and baseline["stable"] and candidate["stable"],
        "claim_supported_rev2": (
            direction and not baseline["unstable_rev2"] and not candidate["unstable_rev2"]
        ),
    }


def boundary_inversion(medians: dict[str, float]) -> bool:
    """Host-origin adds copies to a resident path, so it must not be faster than one."""
    host_origin = medians.get("host-origin")
    if host_origin is None:
        return False
    return any(
        host_origin < median
        for boundary, median in medians.items()
        if boundary.startswith("resident")
    )


def uptime_seconds() -> float:
    if os.name == "nt":
        kernel32 = ctypes.windll.kernel32
        kernel32.GetTickCount64.restype = ctypes.c_uint64
        return kernel32.GetTickCount64() / 1000.0
    return float(Path("/proc/uptime").read_text(encoding="utf-8").split()[0])


def list_app_windows() -> list[dict[str, str]]:
    """Top-level windows with a title, as (process, title) pairs."""
    if os.name != "nt":
        return []
    script = (
        "[Console]::OutputEncoding = [Text.Encoding]::UTF8; "
        "Get-Process | Where-Object { $_.MainWindowHandle -ne 0 -and $_.MainWindowTitle } | "
        "ForEach-Object { $_.ProcessName + [char]9 + $_.MainWindowTitle }"
    )
    proc = subprocess.run(
        ["powershell.exe", "-NoProfile", "-Command", script],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    windows = []
    for line in proc.stdout.splitlines():
        name, _, title = line.partition("\t")
        if name.strip():
            windows.append({"process": name.strip(), "title": title.strip()})
    return windows


def list_processes(name: str) -> list[int]:
    if os.name == "nt":
        script = f"(Get-Process -Name '{name}' -ErrorAction SilentlyContinue).Id"
        proc = subprocess.run(
            ["powershell.exe", "-NoProfile", "-Command", script],
            capture_output=True,
            text=True,
            check=False,
        )
    else:
        proc = subprocess.run(["pgrep", "-x", name], capture_output=True, text=True, check=False)
    return [int(pid) for pid in proc.stdout.split() if pid.isdigit()]


def query_power_plan() -> str:
    if os.name != "nt":
        return "unavailable"
    proc = subprocess.run(
        ["powercfg", "/getactivescheme"], capture_output=True, text=True, check=False
    )
    return proc.stdout.strip() or "unavailable"


def query_hags() -> str:
    """Hardware-accelerated GPU scheduling: HwSchMode 2 is on, 1 is off, unset is the default."""
    if os.name != "nt":
        return "unavailable"
    import winreg

    try:
        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\GraphicsDrivers"
        ) as key:
            value, _ = winreg.QueryValueEx(key, "HwSchMode")
    except OSError:
        return "unset"
    return str(value)


def probe_readiness_facts(root: Path) -> dict[str, Any]:
    """Gather the machine facts the readiness check judges, plus context it only records."""
    gpu_state = query_gpu_state()
    return {
        "uptime_seconds": uptime_seconds(),
        "app_windows": list_app_windows(),
        "mco2_pids": list_processes("mco2"),
        "git_dirty_files": collect_git_provenance(root)["dirty_files"],
        "gpu_clock_event_reasons": gpu_state.get("clocks_event_reasons.active"),
        "power_plan": query_power_plan(),
        "hags_hwschmode": query_hags(),
    }


def check_readiness(
    facts: dict[str, Any], allowlist: Sequence[str] = WINDOW_ALLOWLIST
) -> list[str]:
    """Named reasons the machine is not ready for an evidence sweep; empty when ready."""
    failures = []
    uptime = facts["uptime_seconds"]
    if uptime > MAX_UPTIME_SECONDS:
        failures.append(
            f"uptime {uptime / 60:.0f} min exceeds {MAX_UPTIME_SECONDS // 60} min; reboot first"
        )
    allowed = {name.lower() for name in allowlist}
    for window in facts["app_windows"]:
        if window["process"].lower() not in allowed:
            failures.append(f"open app window: {window['process']} ({window['title']})")
    if facts["mco2_pids"]:
        pids = ", ".join(str(pid) for pid in facts["mco2_pids"])
        failures.append(f"mco2 already running (pid {pids})")
    if facts["git_dirty_files"]:
        failures.append("dirty git tree: " + "; ".join(facts["git_dirty_files"]))
    try:
        reasons = int(facts["gpu_clock_event_reasons"], 16)
    except (TypeError, ValueError):
        reasons = None
    if reasons is None:
        failures.append("GPU clock-event reasons unavailable")
    else:
        active = reasons & ~BENIGN_CLOCK_EVENTS
        if active:
            names = [name for bit, name in CLOCK_EVENT_REASONS.items() if active & bit]
            failures.append(f"GPU clock-event reasons active: {', '.join(names) or hex(active)}")
    return failures


def affinity_mask_excluding(excluded: Sequence[int], current_mask: int) -> int:
    mask = current_mask
    for cpu in excluded:
        mask &= ~(1 << cpu)
    if mask == 0:
        raise RuntimeError(f"no logical CPU left after excluding {list(excluded)}")
    return mask


def _kernel32() -> Any:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    kernel32.GetProcessAffinityMask.argtypes = (
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(ctypes.c_size_t),
    )
    kernel32.SetProcessAffinityMask.argtypes = (ctypes.c_void_p, ctypes.c_size_t)
    return kernel32


def get_process_affinity() -> int:
    if os.name == "nt":
        kernel32 = _kernel32()
        process_mask = ctypes.c_size_t()
        system_mask = ctypes.c_size_t()
        if not kernel32.GetProcessAffinityMask(
            kernel32.GetCurrentProcess(), ctypes.byref(process_mask), ctypes.byref(system_mask)
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        return process_mask.value
    return sum(1 << cpu for cpu in os.sched_getaffinity(0))


def set_process_affinity(mask: int) -> None:
    """Restrict this process; every child it starts inherits the mask."""
    if os.name == "nt":
        kernel32 = _kernel32()
        if not kernel32.SetProcessAffinityMask(kernel32.GetCurrentProcess(), mask):
            raise ctypes.WinError(ctypes.get_last_error())
        return
    os.sched_setaffinity(0, {cpu for cpu in range(mask.bit_length()) if mask >> cpu & 1})


@contextlib.contextmanager
def process_affinity(mask: int) -> Iterator[None]:
    previous = get_process_affinity()
    set_process_affinity(mask)
    try:
        yield
    finally:
        set_process_affinity(previous)


def compute_stage_medians(runs: Sequence[dict[str, Any]]) -> dict[str, float] | None:
    """Median of each timed stage and of the untimed remainder of each repetition."""
    keys = [k for k in STAGE_KEYS if all(k in run for run in runs)]
    if not runs or not keys:
        return None
    medians = {
        key: float(
            np.median(np.concatenate([np.asarray(run[key], dtype=np.float64) for run in runs]))
        )
        for key in keys
    }
    wall = np.concatenate([np.asarray(run["samples_ms"], dtype=np.float64) for run in runs])
    staged = sum(
        np.concatenate([np.asarray(run[key], dtype=np.float64) for run in runs]) for key in keys
    )
    medians["other_ms"] = float(np.median(wall - staged))
    return medians


def case_order(
    counts: Sequence[int], bit_widths: Sequence[int], seed: int
) -> list[tuple[int, int]]:
    """Every (count, bits) case in a seeded random order."""
    cases = [(count, bits) for count in counts for bits in bit_widths]
    permutation = np.random.default_rng(seed).permutation(len(cases))
    return [cases[i] for i in permutation]


def warm_up_gpu(binary: Path, input_path: Path, seconds: float) -> dict[str, Any]:
    """Run the resident CUDA path untimed until `seconds` have passed."""
    start = time.monotonic()
    processes = 0
    while time.monotonic() - start < seconds:
        payload, error = run_bench_process(
            binary,
            input_path,
            bits=8,
            backend="cuda",
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


def in_process_warmups(minimum: int, seconds: float, rep_ms: float) -> int:
    """Warm-up count that covers `seconds` at `rep_ms` per repetition, never below `minimum`."""
    if seconds <= 0 or rep_ms <= 0:
        return minimum
    return max(minimum, math.ceil(seconds * 1000.0 / rep_ms))


def compact_invocation_ids(ids: Sequence[int] | None) -> dict[str, int] | list[int] | None:
    """Store a contiguous identifier run as its bounds; revision 2 warm-ups reach 10^4-10^5."""
    if not ids:
        return None if ids is None else []
    if list(ids) != list(range(ids[0], ids[0] + len(ids))):
        return list(ids)
    return {"first": ids[0], "last": ids[-1], "count": len(ids)}


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


def csv_flag(value: bool | None) -> str:
    return "" if value is None else str(value).lower()


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
    "spread_p90_p10",
    "stable",
    "unstable_rev2",
    "speedup_vs_c",
    "speedup_low",
    "speedup_high",
    "speedup_ci_low",
    "speedup_ci_high",
    "verdict",
    "boundary_inversion",
    "direction_supported",
    "magnitude_supported",
    "claim_supported_rev2",
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
    case_order_seed: int = DEFAULT_CASE_ORDER_SEED,
    gpu_warmup_seconds: float = DEFAULT_GPU_WARMUP_SECONDS,
    case_warmup_seconds: float = DEFAULT_CASE_WARMUP_SECONDS,
    in_process_warmup_seconds: float = DEFAULT_IN_PROCESS_WARMUP_SECONDS,
    force_fail: bool = False,
    allow_existing: bool = False,
    allow_dirty: bool = False,
    pilot: bool = False,
    readiness_facts: dict[str, Any] | None = None,
    ignore_readiness: bool = False,
) -> Path:
    if trials < 1:
        raise ValueError("trials must be at least 1")
    if "cpu" not in backends:
        raise ValueError("the CPU comparator is required; include 'cpu' in backends")

    # Revision 3: every probe and benchmark process runs off physical core 0.
    previous_mask = get_process_affinity()
    mask = affinity_mask_excluding(EXCLUDED_LOGICAL_CPUS, previous_mask)
    with process_affinity(mask):
        git_prov = collect_git_provenance(root)

        # Determine snapshot directory name
        now = datetime.now(UTC)
        date_str = now.strftime("%Y-%m-%d")
        short_rev = git_prov["code_revision_short"]
        if output_dir is not None:
            target_dir = output_dir
        elif pilot:
            target_dir = root / "results" / "pilots" / f"{now:%Y-%m-%dT%H%M%S}-{short_rev}"
        else:
            target_dir = root / "results" / f"{date_str}-{short_rev}"

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

        # Revision 3: judge the machine before any benchmark process starts. A pilot is
        # non-evidence by design, so it records the check without enforcing it.
        facts = probe_readiness_facts(root) if readiness_facts is None else readiness_facts
        failures = check_readiness(facts)
        if failures and not (ignore_readiness or pilot):
            raise RuntimeError(
                "Machine is not ready for an evidence sweep; fix these or pass --ignore-readiness "
                "for a non-evidence run: " + " | ".join(failures)
            )
        non_evidence_reasons = []
        if pilot:
            non_evidence_reasons.append("pilot run")
        if failures:
            non_evidence_reasons.append(
                "readiness check failed and was overridden"
                if ignore_readiness
                else "readiness check failed (pilot; not enforced)"
            )
        if git_prov["git_dirty"]:
            non_evidence_reasons.append("dirty git tree")
        run_conditions = {
            "evidence": not non_evidence_reasons,
            "non_evidence_reasons": non_evidence_reasons,
            "pilot": pilot,
            "readiness": {
                "passed": not failures,
                "failures": failures,
                "overridden": bool(failures) and ignore_readiness,
                "enforced": not pilot,
                "facts": facts,
                "max_uptime_seconds": MAX_UPTIME_SECONDS,
                "window_allowlist": list(WINDOW_ALLOWLIST),
            },
        }

        run_conditions["affinity"] = {
            "excluded_logical_cpus": list(EXCLUDED_LOGICAL_CPUS),
            "mask": hex(mask),
            "previous_mask": hex(previous_mask),
            "applied_to": "driver process before any benchmark or probe process; inherited",
        }
        return sweep_matrix(
            root=root,
            target_dir=target_dir,
            date_str=date_str,
            git_prov=git_prov,
            run_conditions=run_conditions,
            counts=counts,
            bit_widths=bit_widths,
            backends=backends,
            warmups=warmups,
            reps=reps,
            trials=trials,
            input_seed=input_seed,
            compression_seed=compression_seed,
            case_order_seed=case_order_seed,
            gpu_warmup_seconds=gpu_warmup_seconds,
            case_warmup_seconds=case_warmup_seconds,
            in_process_warmup_seconds=in_process_warmup_seconds,
            force_fail=force_fail,
        )


def sweep_matrix(
    *,
    root: Path,
    target_dir: Path,
    date_str: str,
    git_prov: dict[str, Any],
    run_conditions: dict[str, Any],
    counts: Sequence[int],
    bit_widths: Sequence[int],
    backends: Sequence[str],
    warmups: int,
    reps: int,
    trials: int,
    input_seed: int,
    compression_seed: int,
    case_order_seed: int,
    gpu_warmup_seconds: float,
    case_warmup_seconds: float,
    in_process_warmup_seconds: float,
    force_fail: bool,
) -> Path:
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
        paths.append(("cuda", "resident-graph", ["--boundary", "resident-graph"]))
        paths.append(("cuda", "host-origin", ["--boundary", "host-origin"]))
    orders = trial_orders(paths, trials)
    order_labels = [[f"{paths[i][0]}-{paths[i][1]}" for i in order] for order in orders]
    ordered_cases = case_order(counts, bit_widths, case_order_seed)

    # Bring the GPU to steady clocks on the largest input before any timed process.
    gpu_warmup = None
    gpu_state_after_warmup = None
    if "cuda" in backends and gpu_warmup_seconds > 0:
        gpu_warmup = warm_up_gpu(binary, input_files[max(counts)], gpu_warmup_seconds)
        gpu_state_after_warmup = query_gpu_state()

    # 3. Benchmark cases
    case_results: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for execution_index, (count, bits) in enumerate(ordered_cases):
        input_path = input_files[count]
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
                    "in_process_warmup": None,
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
                    "execution_index": execution_index,
                    "case_warmup": None,
                    "trial_runs": [],
                    "samples_ms": [],
                    "statistics": None,
                    "stage_medians_ms": None,
                }
            )

        if passed and "cuda" in backends and case_warmup_seconds > 0:
            # Restore GPU clocks after the untimed correctness gate and any
            # long CPU processes of the previous case.
            case_warmup = warm_up_gpu(binary, input_path, case_warmup_seconds)
            case_warmup["gpu_state_after"] = query_gpu_state()
            for case in cases:
                case["case_warmup"] = case_warmup

        if passed and in_process_warmup_seconds > 0:
            # Revision 2: size each path's in-process warm-up from an untimed probe
            # so every timed process first runs about the same wall time untimed.
            for case, (backend, _, extra_args) in zip(cases, paths, strict=True):
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
                probe_ms = float(np.median(payload.get("samples_ms", [])))
                case["warmup"] = in_process_warmups(warmups, in_process_warmup_seconds, probe_ms)
                case["in_process_warmup"] = {
                    "target_seconds": in_process_warmup_seconds,
                    "probe_warmup": warmups,
                    "probe_reps": reps,
                    "probe_median_ms": probe_ms,
                    "warmup": case["warmup"],
                }

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
                        warmups=case["warmup"],
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
                        "repetition_invocation_ids": configuration.get("repetition_invocation_ids"),
                        "warmup_invocation_ids": compact_invocation_ids(
                            configuration.get("warmup_invocation_ids")
                        ),
                        "samples_ms": payload.get("samples_ms", []),
                    }
                    for key in ("capture_ms", "capture_and_instantiate_ms"):
                        if key in payload:
                            run[key] = payload[key]
                    for key in STAGE_KEYS:
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
            case["stage_medians_ms"] = compute_stage_medians(runs)

        # Comparisons and flags, only between cases that all passed.
        cpu_case = cases[0]
        inversion = boundary_inversion(
            {
                c["timing_boundary"]: c["statistics"]["median_ms"]
                for c in cases
                if c is not cpu_case and c["statistics"] is not None
            }
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
                        "speedup_ci_low": 1.0,
                        "speedup_ci_high": 1.0,
                        "verdict": "comparator",
                        "boundary_inversion": False,
                        "direction_supported": None,
                        "magnitude_supported": None,
                        "claim_supported_rev2": None,
                    }
                )
                continue
            cpu_stats = cpu_case["statistics"]
            if cpu_stats is None:
                continue
            stats.update(compare_to_comparator(cpu_stats, stats))
            stats["boundary_inversion"] = inversion
            stats.update(claim_support(stats["verdict"], inversion, cpu_stats, stats))

        resident_case = next((c for c in cases if c["timing_boundary"] == "resident"), None)
        graph_case = next((c for c in cases if c["timing_boundary"] == "resident-graph"), None)
        if resident_case is not None and graph_case is not None:
            res_stats = resident_case.get("statistics")
            graph_stats = graph_case.get("statistics")
            if res_stats is not None and graph_stats is not None:
                vs_res = compare_to_resident(res_stats, graph_stats)
                vs_res_claims = claim_support(vs_res["verdict"], inversion, res_stats, graph_stats)
                graph_vs_res = {
                    "speedup_vs_resident": vs_res["speedup_vs_resident"],
                    "speedup_low": vs_res["speedup_low"],
                    "speedup_high": vs_res["speedup_high"],
                    "speedup_ci_low": vs_res["speedup_ci_low"],
                    "speedup_ci_high": vs_res["speedup_ci_high"],
                    "verdict": vs_res["verdict"],
                    "boundary_inversion": inversion,
                    "direction_supported": vs_res_claims["direction_supported"],
                    "magnitude_supported": vs_res_claims["magnitude_supported"],
                    "claim_supported_rev2": vs_res_claims["claim_supported_rev2"],
                }
                graph_stats["vs_resident"] = graph_vs_res
                graph_case["vs_resident"] = graph_vs_res

        for case in cases:
            (target_dir / f"{case['case_id']}.json").write_text(
                json.dumps(case, indent=2), encoding="utf-8"
            )
            case_results.append(case)
            stats = case["statistics"]
            if case["correctness"]["status"] != "passed" or stats is None or "verdict" not in stats:
                summary_rows.append(
                    failed_row(
                        count,
                        bits,
                        case["backend"],
                        case["timing_boundary"],
                        case["warmup"],
                        reps,
                        trials,
                    )
                )
                continue
            summary_rows.append(
                {
                    "count": count,
                    "bits": bits,
                    "backend": case["backend"],
                    "boundary": case["timing_boundary"],
                    "correctness": "passed",
                    "warmup": case["warmup"],
                    "reps": reps,
                    "trials": trials,
                    "median_ms": f"{stats['median_ms']:.6f}",
                    "iqr_ms": f"{stats['iqr_ms']:.6f}",
                    "trial_median_min_ms": f"{stats['trial_median_min_ms']:.6f}",
                    "trial_median_max_ms": f"{stats['trial_median_max_ms']:.6f}",
                    "spread_ratio": f"{stats['spread_ratio']:.4f}",
                    "spread_p90_p10": f"{stats['spread_p90_p10']:.4f}",
                    "stable": csv_flag(stats["stable"]),
                    "unstable_rev2": csv_flag(stats["unstable_rev2"]),
                    "speedup_vs_c": f"{stats['speedup_vs_cpu']:.4f}",
                    "speedup_low": f"{stats['speedup_low']:.4f}",
                    "speedup_high": f"{stats['speedup_high']:.4f}",
                    "speedup_ci_low": f"{stats['speedup_ci_low']:.4f}",
                    "speedup_ci_high": f"{stats['speedup_ci_high']:.4f}",
                    "verdict": stats["verdict"],
                    "boundary_inversion": csv_flag(stats["boundary_inversion"]),
                    "direction_supported": csv_flag(stats["direction_supported"]),
                    "magnitude_supported": csv_flag(stats["magnitude_supported"]),
                    "claim_supported_rev2": csv_flag(stats["claim_supported_rev2"]),
                }
            )

    gpu_state_end = query_gpu_state() if "cuda" in backends else None

    path_rank = {f"{backend}-{boundary}": i for i, (backend, boundary, _) in enumerate(paths)}
    summary_rows.sort(
        key=lambda r: (r["count"], r["bits"], path_rank[f"{r['backend']}-{r['boundary']}"])
    )
    case_results.sort(
        key=lambda c: (
            c["count"],
            c["bits"],
            path_rank[f"{c['backend']}-{c['timing_boundary']}"],
        )
    )

    shutil.rmtree(temp_dir, ignore_errors=True)

    # 4. Summary CSV
    csv_path = target_dir / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(summary_rows)

    # 5. Run manifest
    manifest_data = {
        "manifest_version": "3.0",
        "date": date_str,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "git_provenance": git_prov,
        "hardware": toolchain_prov["hardware"],
        "toolkit_and_driver": toolchain_prov["toolkit_and_driver"],
        "build_flags": build_prov,
        "transfer_policy": toolchain_prov["transfer_policy"],
        "gpu_state": {
            "note": (
                "Instantaneous nvidia-smi readings before the warm-up, after it, and after "
                "the last timed process; they describe the GPU around the run, not clock "
                "stability during it."
            ),
            "start": gpu_state_start,
            "warmup": gpu_warmup,
            "after_warmup": gpu_state_after_warmup,
            "end": gpu_state_end,
        },
        "matrix_parameters": {
            "counts": list(counts),
            "bit_widths": list(bit_widths),
            "backends": list(backends),
            "warmup": warmups,
            "in_process_warmup_seconds": in_process_warmup_seconds,
            "in_process_warmup_rule": (
                "per path: max(warmup, ceil(target_seconds * 1000 / probe median ms)), "
                "probe = one untimed mco2 bench process with the base warmup and reps"
                if in_process_warmup_seconds > 0
                else "disabled; every path uses the base warmup"
            ),
            "reps": reps,
            "trials": trials,
            "trial_orders": order_labels,
            "case_order_seed": case_order_seed,
            "case_order": [{"count": c, "bits": b} for c, b in ordered_cases],
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
            "spread_p90_p10": (
                f"p{STABILITY_PERCENTILES[1]} / p{STABILITY_PERCENTILES[0]} of the trial medians"
            ),
            "stability_threshold": STABILITY_THRESHOLD,
            "claim_rules": {
                "verdict": VERDICT_RULE,
                "direction_supported": DIRECTION_RULE,
                "magnitude_supported": MAGNITUDE_RULE,
                "claim_supported_rev2": CLAIM_RULE_REV2,
            },
            "bootstrap": {
                "resamples": BOOTSTRAP_RESAMPLES,
                "seed": BOOTSTRAP_SEED,
                "level": BOOTSTRAP_LEVEL,
                "rule": BOOTSTRAP_RULE,
            },
        },
        "run_conditions": run_conditions,
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
        "--case-order-seed",
        type=int,
        default=DEFAULT_CASE_ORDER_SEED,
        help="Seed for the random order of (count, bits) cases",
    )
    parser.add_argument(
        "--gpu-warmup-seconds",
        type=float,
        default=DEFAULT_GPU_WARMUP_SECONDS,
        help="Untimed CUDA work before the first timed process (0 disables)",
    )
    parser.add_argument(
        "--case-warmup-seconds",
        type=float,
        default=DEFAULT_CASE_WARMUP_SECONDS,
        help="Untimed CUDA work before each case's timed processes (0 disables)",
    )
    parser.add_argument(
        "--warmup-seconds",
        type=float,
        default=DEFAULT_IN_PROCESS_WARMUP_SECONDS,
        help=(
            "Minimum untimed in-process warm-up per timed process, for every path; "
            "--warmup is the floor (0 disables)"
        ),
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
    parser.add_argument(
        "--pilot",
        action="store_true",
        help=(
            "Non-evidence pilot at four sizes and both bit widths, written under "
            "results/pilots/; records the readiness check without enforcing it"
        ),
    )
    parser.add_argument(
        "--ignore-readiness",
        action="store_true",
        help="Run despite a failed readiness check (marks the snapshot non-evidence)",
    )

    args = parser.parse_args()
    root = Path(__file__).resolve().parent

    try:
        snapshot_dir = run_benchmark_matrix(
            root=root,
            output_dir=args.output_dir,
            counts=PILOT_COUNTS if args.pilot else args.counts,
            bit_widths=DEFAULT_BITS if args.pilot else args.bits,
            backends=args.backends,
            warmups=args.warmup,
            reps=args.reps,
            trials=args.trials,
            input_seed=args.input_seed,
            compression_seed=args.compression_seed,
            case_order_seed=args.case_order_seed,
            gpu_warmup_seconds=args.gpu_warmup_seconds,
            case_warmup_seconds=args.case_warmup_seconds,
            in_process_warmup_seconds=args.warmup_seconds,
            force_fail=args.force_fail,
            allow_existing=args.allow_existing,
            allow_dirty=args.allow_dirty,
            pilot=args.pilot,
            ignore_readiness=args.ignore_readiness,
        )
        print(f"Benchmark completed successfully. Snapshot written to: {snapshot_dir}")
    except (RuntimeError, ValueError, OSError, subprocess.SubprocessError) as exc:
        print(f"Benchmark driver failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
