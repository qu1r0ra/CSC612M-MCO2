"""Benchmark driver for in-process mco2 CPU comparator and CUDA pipelines.

Orchestrates input generation, per-case correctness gating, timed benchmark
runs, provenance collection, statistical summary computation, and snapshot
creation as specified by the course technical contract and benchmark protocol.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
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

CPU_BUILD_FLAGS = (
    "/O2 /W4 /std:c11 /fp:strict /D_CRT_SECURE_NO_WARNINGS /Isrc /Ithird_party/random123/include"
)
CUDA_HOST_BUILD_FLAGS = (
    "/O2 /W4 /std:c11 /fp:strict /D_CRT_SECURE_NO_WARNINGS /DMCO2_ENABLE_CUDA /Isrc "
    "/Ithird_party/random123/include"
)
CUDA_NVCC_BUILD_FLAGS = (
    "-O2 -arch=native -Isrc -Ithird_party/random123/include --fmad=false --ftz=false "
    "--prec-div=true --prec-sqrt=true -Xcompiler /wd4068"
)


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
        is_dirty = bool(status_output.strip())
    except (subprocess.SubprocessError, OSError, RuntimeError):
        rev = "unknown"
        short_rev = "unknown"
        is_dirty = True
    return {
        "code_revision": rev,
        "code_revision_short": short_rev,
        "git_dirty": is_dirty,
    }


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
        "build_flags": {
            "cpu": CPU_BUILD_FLAGS,
            "cuda_host": CUDA_HOST_BUILD_FLAGS,
            "cuda_nvcc": CUDA_NVCC_BUILD_FLAGS,
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


def generate_msvc_vectorization_report(root: Path, output_file: Path) -> str:
    """Run MSVC cl.exe with /Qvec-report:2 to inspect vectorization decisions."""
    output_text = ""
    if os.name == "nt":
        script = root / "scripts" / "with-msvc.ps1"
        build_dir = root / "build"
        build_dir.mkdir(parents=True, exist_ok=True)
        if script.is_file():
            cmd = [
                "powershell.exe",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(script),
                "cl.exe",
                "/nologo",
                "/O2",
                "/W4",
                "/std:c11",
                "/fp:strict",
                "/Qvec-report:2",
                "/D_CRT_SECURE_NO_WARNINGS",
                "/Isrc",
                "/Ithird_party/random123/include",
                "/c",
                "src\\main.c",
                "src\\codec.c",
                "src\\quantizer.c",
                "src\\rng_cpu.c",
                "/Fo:build\\",
            ]
            res = subprocess.run(cmd, cwd=root, capture_output=True, text=True, check=False)
            output_text = (res.stdout + "\n" + res.stderr).strip()

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
    q25 = float(np.percentile(arr, 25))
    q75 = float(np.percentile(arr, 75))
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


def run_benchmark_matrix(
    *,
    root: Path,
    output_dir: Path | None = None,
    counts: Sequence[int] = DEFAULT_COUNTS,
    bit_widths: Sequence[int] = DEFAULT_BITS,
    backends: Sequence[str] = ("cpu", "cuda"),
    warmups: int = DEFAULT_WARMUPS,
    reps: int = DEFAULT_REPS,
    input_seed: int = DEFAULT_INPUT_SEED,
    compression_seed: int = DEFAULT_COMPRESSION_SEED,
    force_fail: bool = False,
    allow_existing: bool = False,
) -> Path:
    binary = find_binary(root)
    git_prov = collect_git_provenance(root)
    toolchain_prov = collect_hardware_and_toolchain(root)

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

    target_dir.mkdir(parents=True, exist_ok=True)
    existing_inputs_dir = target_dir / "inputs"
    if existing_inputs_dir.exists():
        shutil.rmtree(existing_inputs_dir, ignore_errors=True)

    temp_dir = target_dir / "_temp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    # 1. Inputs generation
    inputs_dir = temp_dir / "inputs"
    input_meta = generate_inputs(counts, inputs_dir, seed=input_seed)
    input_files: dict[int, Path] = {c: Path(meta["_path"]) for c, meta in input_meta.items()}
    clean_input_meta: dict[int, dict[str, Any]] = {
        c: {k: v for k, v in meta.items() if not k.startswith("_")}
        for c, meta in input_meta.items()
    }

    # 2. Vectorization report
    vec_report_path = target_dir / "msvc_vectorization_report.txt"
    generate_msvc_vectorization_report(root, vec_report_path)

    # 3. Benchmark cases
    case_results: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for count in counts:
        input_path = input_files[count]

        for bits in bit_widths:
            # First: Gate on correctness before any timed run!
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

            paths = []
            if "cpu" in backends:
                paths.append(("cpu", "comparator", []))
            if "cuda" in backends:
                paths.append(("cuda", "resident", ["--boundary", "resident"]))
                paths.append(("cuda", "host-origin", ["--boundary", "host-origin"]))

            cpu_median_ms: float | None = None

            for backend, boundary, extra_args in paths:
                case_id = f"case_{backend}_{boundary}_bits{bits}_n{count}"
                case_json_path = target_dir / f"{case_id}.json"

                payload_bytes = count if bits == 8 else (count + 1) // 2

                case_data: dict[str, Any] = {
                    "case_id": case_id,
                    "count": count,
                    "bits": bits,
                    "backend": backend,
                    "timing_boundary": boundary,
                    "transfer_policy": toolchain_prov["transfer_policy"],
                    "seed": compression_seed,
                    "tensor_id": 0,
                    "invocation_id": 0,
                    "repetition_invocation_ids": list(range(reps)),
                    "warmup_invocation_ids": list(range(reps, reps + warmups)),
                    "warmup": warmups,
                    "reps": reps,
                    "header_bytes": HEADER_STRUCT.size,
                    "payload_bytes": payload_bytes,
                    "code_revision": git_prov["code_revision"],
                    "code_revision_short": git_prov["code_revision_short"],
                    "git_dirty": git_prov["git_dirty"],
                    "build_flags": toolchain_prov["build_flags"],
                    "hardware": toolchain_prov["hardware"],
                    "toolkit_and_driver": toolchain_prov["toolkit_and_driver"],
                    "input_provenance": clean_input_meta[count],
                    "correctness": correctness_info,
                    "samples_ms": [],
                    "statistics": None,
                }

                if not passed:
                    # Marking failed with no speed figures
                    case_results.append(case_data)
                    case_json_path.write_text(json.dumps(case_data, indent=2), encoding="utf-8")
                    summary_rows.append(
                        {
                            "count": count,
                            "bits": bits,
                            "backend": backend,
                            "boundary": boundary,
                            "correctness": "failed",
                            "warmup": warmups,
                            "reps": reps,
                            "median_ms": "",
                            "iqr_ms": "",
                            "speedup_vs_c": "",
                        }
                    )
                    continue

                # Run timed benchmark
                bench_cmd = [
                    str(binary),
                    "bench",
                    "--input",
                    str(input_path),
                    "--seed",
                    str(compression_seed),
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

                bench_proc = subprocess.run(bench_cmd, capture_output=True, text=True, check=False)
                if bench_proc.returncode != 0:
                    case_data["correctness"] = {
                        "status": "failed",
                        "error_message": f"mco2 bench failed: {bench_proc.stderr.strip()}",
                    }
                    case_results.append(case_data)
                    case_json_path.write_text(json.dumps(case_data, indent=2), encoding="utf-8")
                    summary_rows.append(
                        {
                            "count": count,
                            "bits": bits,
                            "backend": backend,
                            "boundary": boundary,
                            "correctness": "failed",
                            "warmup": warmups,
                            "reps": reps,
                            "median_ms": "",
                            "iqr_ms": "",
                            "speedup_vs_c": "",
                        }
                    )
                    continue

                bench_payload = json.loads(bench_proc.stdout)
                samples = bench_payload.get("samples_ms", [])
                case_data["samples_ms"] = samples
                if "k1_ms" in bench_payload:
                    case_data["k1_ms"] = bench_payload["k1_ms"]
                    case_data["k2_ms"] = bench_payload["k2_ms"]
                    case_data["k3_ms"] = bench_payload["k3_ms"]

                stats = compute_statistics(samples)
                if backend == "cpu":
                    cpu_median_ms = stats["median_ms"]
                    stats["speedup_vs_cpu"] = 1.0
                else:
                    if cpu_median_ms is not None and stats["median_ms"] > 0:
                        stats["speedup_vs_cpu"] = cpu_median_ms / stats["median_ms"]
                    else:
                        stats["speedup_vs_cpu"] = 0.0

                case_data["statistics"] = stats
                case_results.append(case_data)
                case_json_path.write_text(json.dumps(case_data, indent=2), encoding="utf-8")

                summary_rows.append(
                    {
                        "count": count,
                        "bits": bits,
                        "backend": backend,
                        "boundary": boundary,
                        "correctness": "passed",
                        "warmup": warmups,
                        "reps": reps,
                        "median_ms": f"{stats['median_ms']:.6f}",
                        "iqr_ms": f"{stats['iqr_ms']:.6f}",
                        "speedup_vs_c": f"{stats['speedup_vs_cpu']:.4f}",
                    }
                )

    # Clean up temp files
    try:
        shutil.rmtree(temp_dir, ignore_errors=True)
    except OSError:
        pass

    # 4. Summary CSV
    csv_path = target_dir / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "count",
                "bits",
                "backend",
                "boundary",
                "correctness",
                "warmup",
                "reps",
                "median_ms",
                "iqr_ms",
                "speedup_vs_c",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    # 5. Run manifest
    manifest_data = {
        "manifest_version": "1.0",
        "date": date_str,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "git_provenance": git_prov,
        "hardware": toolchain_prov["hardware"],
        "toolkit_and_driver": toolchain_prov["toolkit_and_driver"],
        "build_flags": toolchain_prov["build_flags"],
        "transfer_policy": toolchain_prov["transfer_policy"],
        "matrix_parameters": {
            "counts": list(counts),
            "bit_widths": list(bit_widths),
            "warmup": warmups,
            "reps": reps,
            "input_seed": input_seed,
            "compression_seed": compression_seed,
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
            input_seed=args.input_seed,
            compression_seed=args.compression_seed,
            force_fail=args.force_fail,
            allow_existing=args.allow_existing,
        )
        print(f"Benchmark completed successfully. Snapshot written to: {snapshot_dir}")
    except (RuntimeError, ValueError, OSError, subprocess.SubprocessError) as exc:
        print(f"Benchmark driver failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
