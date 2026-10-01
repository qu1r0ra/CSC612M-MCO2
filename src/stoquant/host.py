"""Host control: readiness checks, process affinity, power-plan and HAGS probes."""

from __future__ import annotations

import contextlib
import ctypes
import json
import math
import os
import re
import subprocess
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

from stoquant.provenance import collect_git_provenance, query_gpu_state

# Every benchmark and probe process runs off physical core 0 (logical CPUs 0
# and 1), which ran about 30% slower in the issue #30 diagnosis.
EXCLUDED_LOGICAL_CPUS = (0, 1)

# Evidence runs require at least 4 GiB of available physical memory.
MIN_AVAILABLE_PHYSICAL_BYTES = 4 * 1024**3
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


def _checked_probe_output(proc: subprocess.CompletedProcess[str], name: str) -> str:
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout).strip() or "no diagnostic output"
        raise RuntimeError(f"{name} exited with status {proc.returncode}: {detail}")
    if proc.stderr.strip():
        raise RuntimeError(f"{name} wrote to stderr: {proc.stderr.strip()}")
    return proc.stdout


def _parse_json_array(output: str, name: str) -> list[Any]:
    try:
        value = json.loads(output)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"{name} returned unparseable JSON: {error.msg}") from None
    if not isinstance(value, list):
        raise RuntimeError(  # noqa: TRY004 - malformed tool output is a probe failure.
            f"{name} returned unparseable JSON: expected an array"
        )
    return value


def uptime_seconds() -> float:
    if os.name == "nt":
        kernel32 = ctypes.windll.kernel32
        kernel32.GetTickCount64.restype = ctypes.c_uint64
        uptime = kernel32.GetTickCount64() / 1000.0
    else:
        uptime = float(Path("/proc/uptime").read_text(encoding="utf-8").split()[0])
    if not math.isfinite(uptime) or uptime < 0:
        raise ValueError("uptime probe returned an invalid value")
    return uptime


def physical_memory_status() -> dict[str, int | float]:
    """Return physical memory totals and used percentage for readiness."""
    if os.name == "nt":

        class MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatusEx()
        status.dwLength = ctypes.sizeof(status)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        global_memory_status_ex = kernel32.GlobalMemoryStatusEx
        global_memory_status_ex.argtypes = (ctypes.POINTER(MemoryStatusEx),)
        global_memory_status_ex.restype = ctypes.c_bool
        if not global_memory_status_ex(ctypes.byref(status)):
            raise ctypes.WinError(ctypes.get_last_error())
        total_bytes = int(status.ullTotalPhys)
        available_bytes = int(status.ullAvailPhys)
    else:
        total_bytes = int(os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
        available_bytes = int(os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))

    if total_bytes <= 0 or available_bytes < 0 or available_bytes > total_bytes:
        raise ValueError("physical-memory probe returned invalid byte counts")
    return {
        "total_bytes": total_bytes,
        "available_bytes": available_bytes,
        "used_percent": (total_bytes - available_bytes) * 100.0 / total_bytes,
    }


def list_processes(name: str) -> list[int]:
    if os.name == "nt":
        script = (
            "$ErrorActionPreference = 'Stop'; "
            f"$ids = @(Get-Process | Where-Object {{ $_.ProcessName -eq '{name}' }} | "
            "ForEach-Object { $_.Id }); "
            "ConvertTo-Json -InputObject $ids -Compress"
        )
        proc = subprocess.run(
            ["powershell.exe", "-NoProfile", "-Command", script],
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
        )
        output = _checked_probe_output(proc, "list_processes").strip()
        ids = _parse_json_array(output, "list_processes")
        if any(type(pid) is not int or pid <= 0 for pid in ids):
            raise RuntimeError("list_processes returned an unparseable process ID")
        return ids
    else:
        proc = subprocess.run(["pgrep", "-x", name], capture_output=True, text=True, check=False)
        if proc.returncode == 1 and not proc.stderr.strip() and not proc.stdout.strip():
            return []
        output = _checked_probe_output(proc, "list_processes")
    pids = output.split()
    if any(not pid.isdecimal() for pid in pids):
        raise RuntimeError(f"list_processes returned unparseable output: {output.strip()!r}")
    return [int(pid) for pid in pids]


def query_power_plan() -> str:
    if os.name != "nt":
        return "unavailable"
    proc = subprocess.run(
        ["powercfg", "/getactivescheme"], capture_output=True, text=True, check=False
    )
    output = _checked_probe_output(proc, "query_power_plan").strip()
    guid = r"[0-9a-fA-F]{8}-(?:[0-9a-fA-F]{4}-){3}[0-9a-fA-F]{12}"
    if not re.search(guid, output):
        raise RuntimeError("query_power_plan returned unparseable output")
    return output


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
    except FileNotFoundError:
        return "unset"
    if type(value) is not int:
        raise RuntimeError("query_hags returned an unparseable registry value")
    return str(value)


def probe_readiness_facts(root: Path, git_prov: dict[str, Any] | None = None) -> dict[str, Any]:
    """Gather the machine facts the readiness check judges, plus context it only records."""
    probe_failures: dict[str, str] = {}
    context_probe_failures: dict[str, str] = {}

    def probe(name: str, call: Any, fallback: Any) -> Any:
        try:
            return call()
        except Exception as error:  # noqa: BLE001 - readiness names every failed probe.
            detail = str(error).strip() or type(error).__name__
            probe_failures[name] = f"{type(error).__name__}: {detail}"
            return fallback

    gpu_state = probe("query_gpu_state", query_gpu_state, {})
    if not isinstance(gpu_state, dict):
        probe_failures["query_gpu_state"] = "returned an unparseable result"
        gpu_state = {}
    git_info = (
        git_prov
        if git_prov is not None
        else probe("collect_git_provenance", lambda: collect_git_provenance(root), {})
    )
    if not isinstance(git_info, dict) or not isinstance(git_info.get("dirty_files"), list):
        probe_failures["collect_git_provenance"] = "returned an unparseable result"
        dirty_files: list[str] = ["<git status unavailable>"]
    else:
        dirty_files = git_info["dirty_files"]

    facts: dict[str, Any] = {
        "physical_memory": probe("physical_memory_status", physical_memory_status, {}),
        "stoquant_pids": probe("list_processes", lambda: list_processes("stoquant"), []),
        "git_dirty_files": dirty_files,
        "gpu_clock_event_reasons": gpu_state.get("clocks_event_reasons.active"),
        "power_plan": probe("query_power_plan", query_power_plan, "unavailable"),
        "hags_hwschmode": probe("query_hags", query_hags, "unavailable"),
    }
    try:
        facts["uptime_seconds"] = uptime_seconds()
    except Exception as error:  # noqa: BLE001 - uptime is recorded context, not a readiness gate.
        detail = str(error).strip() or type(error).__name__
        context_probe_failures["uptime_seconds"] = f"{type(error).__name__}: {detail}"
        facts["uptime_seconds"] = None
    if probe_failures:
        facts["probe_failures"] = probe_failures
    if context_probe_failures:
        facts["context_probe_failures"] = context_probe_failures
    return facts


def check_readiness(facts: dict[str, Any]) -> list[str]:
    """Named reasons the machine is not ready for an evidence sweep; empty when ready."""
    probe_failures = facts.get("probe_failures", {})
    if not isinstance(probe_failures, dict):
        failures = ["host probe failed: probe_readiness_facts: unparseable failure details"]
        failed_probe_names: set[str] = set()
    else:
        failures = [
            f"host probe failed: {name}: {detail}"
            for name, detail in sorted(probe_failures.items())
        ]
        failed_probe_names = set(probe_failures)
    memory = facts.get("physical_memory")
    if (
        not isinstance(memory, dict)
        or type(memory.get("total_bytes")) is not int
        or type(memory.get("available_bytes")) is not int
        or memory["total_bytes"] <= 0
        or memory["available_bytes"] < 0
        or memory["available_bytes"] > memory["total_bytes"]
    ):
        if "physical_memory_status" not in failed_probe_names:
            failures.append("host probe failed: physical_memory_status: invalid memory facts")
    elif memory["available_bytes"] < MIN_AVAILABLE_PHYSICAL_BYTES:
        failures.append(
            f"available physical memory {memory['available_bytes'] / 1024**3:.2f} GiB "
            f"is below {MIN_AVAILABLE_PHYSICAL_BYTES / 1024**3:.0f} GiB minimum"
        )
    if facts["stoquant_pids"]:
        pids = ", ".join(str(pid) for pid in facts["stoquant_pids"])
        failures.append(f"stoquant already running (pid {pids})")
    try:
        reasons = int(facts["gpu_clock_event_reasons"], 16)
    except (TypeError, ValueError):
        reasons = None
    if reasons is None:
        if "query_gpu_state" not in failed_probe_names:
            failures.append(
                "host probe failed: query_gpu_state: GPU clock-event reasons unavailable"
            )
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


@contextlib.contextmanager
def benchmark_process_affinity() -> Iterator[dict[str, Any]]:
    """Apply the benchmark core-exclusion policy and expose its recorded state."""
    previous_mask = get_process_affinity()
    mask = affinity_mask_excluding(EXCLUDED_LOGICAL_CPUS, previous_mask)
    with process_affinity(mask):
        yield {
            "excluded_logical_cpus": EXCLUDED_LOGICAL_CPUS,
            "mask": mask,
            "previous_mask": previous_mask,
        }
