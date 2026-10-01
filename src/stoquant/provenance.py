"""Provenance: git state, build commands, hardware and toolchain, GPU state, binary lookup."""

from __future__ import annotations

import json
import os
import platform
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NamedTuple

from stoquant.build_stamp import tree_fingerprint, verify_stamp

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


@dataclass(frozen=True)
class BuildCommands:
    """Parsed build provenance with compile arguments kept out of its JSON record."""

    source: str
    commands: list[str]
    comparator_c: str
    avx2_c: str
    cuda_nvcc: str
    host_tokens: list[str] | None
    avx2_tokens: list[str] | None

    def as_record(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "commands": list(self.commands),
            "comparator_c": self.comparator_c,
            "avx2_c": self.avx2_c,
            "cuda_nvcc": self.cuda_nvcc,
        }


class VerifiedBinary(NamedTuple):
    path: Path
    stamp: dict[str, Any]


def find_binary(
    root: Path,
    *,
    require_cuda: bool = True,
    git_prov: dict[str, Any] | None = None,
    binary_name: str | None = None,
) -> VerifiedBinary:
    binary_name = binary_name or ("stoquant.exe" if os.name == "nt" else "stoquant")
    path = root / "build" / binary_name
    if not path.is_file():
        raise FileNotFoundError(
            f"stoquant binary not found at {path}. Build it first with just build-cuda."
        )
    git = collect_git_provenance(root) if git_prov is None else git_prov
    if git["code_revision"] == "unknown":
        raise RuntimeError("cannot verify a build stamp without git provenance")
    stamp = verify_stamp(
        root,
        path,
        revision=git["code_revision"],
        tree_dirty=git["git_dirty"],
        fingerprint=git["tree_fingerprint"],
        require_cuda=require_cuda,
        expected_recipe=("build-cpu", "build-cuda") if not require_cuda else "build-cuda",
    )
    return VerifiedBinary(path, stamp)


def query_device_attributes(root: Path) -> dict[str, Any] | None:
    """Read device constants without running the streaming probe."""

    name = "stream_probe.exe" if os.name == "nt" else "stream_probe"
    path = root / "build" / name
    if not path.is_file():
        return None
    return json.loads(run_command([str(path), "--device-only"], root))


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
        fingerprint = tree_fingerprint(root)
    except (subprocess.SubprocessError, OSError, RuntimeError):
        rev = "unknown"
        short_rev = "unknown"
        dirty_files = ["<git status unavailable>"]
        fingerprint = "unknown"
    return {
        "code_revision": rev,
        "code_revision_short": short_rev,
        "git_dirty": bool(dirty_files),
        "dirty_files": dirty_files,
        "tree_fingerprint": fingerprint,
    }


def collect_build_commands(stamp: dict[str, Any]) -> BuildCommands:
    """Read compiler commands from the verified binary's build stamp."""
    commands = stamp.get("commands")
    if not isinstance(commands, list) or not all(isinstance(command, str) for command in commands):
        raise RuntimeError("verified build stamp has invalid compiler commands")
    parsed = parse_build_commands(commands)
    return BuildCommands(
        source=f"verified build stamp ({stamp.get('recipe', 'unknown')})",
        commands=parsed.commands,
        comparator_c=parsed.comparator_c,
        avx2_c=parsed.avx2_c,
        cuda_nvcc=parsed.cuda_nvcc,
        host_tokens=parsed.host_tokens,
        avx2_tokens=parsed.avx2_tokens,
    )


def parse_build_commands(commands: Sequence[str]) -> BuildCommands:
    """Flags of the scalar comparator, the AVX2 comparator and the CUDA kernels.

    Each C entry is the compile of its own source file, so the per-file flags of the
    AVX2 translation unit never stand in for the scalar comparator's.
    """
    host_flags: list[str] | None = None
    avx2_flags: list[str] | None = None
    nvcc_flags: list[str] | None = None
    for command in commands:
        tokens = command.split()
        if tokens and tokens[0].endswith("with-msvc.ps1"):
            tokens = tokens[1:]
        if not tokens:
            continue
        tool = Path(tokens[0]).name.lower()
        compiles = any(t.lower() in ("/c", "-c") for t in tokens)
        sources = {t.replace("\\", "/").rsplit("/", 1)[-1] for t in tokens[1:]}
        if compiles and tool in ("cl.exe", "cl", "gcc", "cc", "clang"):
            if host_flags is None and "quantizer.c" in sources:
                host_flags = strip_compile_io(tokens[1:])
            elif avx2_flags is None and "quantizer_avx2.c" in sources:
                avx2_flags = strip_compile_io(tokens[1:])
        elif nvcc_flags is None and compiles and tool in ("nvcc", "nvcc.exe"):
            nvcc_flags = strip_compile_io(tokens[1:])

    return BuildCommands(
        source="parsed compile commands",
        commands=list(commands),
        comparator_c=" ".join(host_flags) if host_flags is not None else "unknown",
        avx2_c=" ".join(avx2_flags) if avx2_flags is not None else "unknown",
        cuda_nvcc=" ".join(nvcc_flags) if nvcc_flags is not None else "unknown",
        host_tokens=host_flags,
        avx2_tokens=avx2_flags,
    )


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
    state: dict[str, str] = dict(zip(GPU_STATE_FIELDS, values, strict=True))
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
            script = root / "tools" / "with-msvc.ps1"
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
    }
