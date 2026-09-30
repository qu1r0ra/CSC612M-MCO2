"""Host control: readiness checks, process affinity, power-plan, HAGS and window probes."""

from __future__ import annotations

import contextlib
import ctypes
import os
import subprocess
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

from stoquant.provenance import collect_git_provenance, query_gpu_state

# Every benchmark and probe process runs off physical core 0 (logical CPUs 0
# and 1), which ran about 30% slower in the issue #30 diagnosis.
EXCLUDED_LOGICAL_CPUS = (0, 1)

# Readiness: a sweep starts on a freshly rebooted, quiet machine. Besides these
# Windows shell hosts, only the processes that launched the sweep may own windows.
MAX_UPTIME_SECONDS = 30 * 60
WINDOW_ALLOWLIST = (
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


def list_launcher_processes() -> list[str]:
    """Process names from this driver up its parent chain, innermost first."""
    if os.name != "nt":
        return []
    script = (
        "$byId = @{}; Get-CimInstance Win32_Process | "
        "ForEach-Object { $byId[[int]$_.ProcessId] = $_ }; "
        f"$id = {os.getpid()}; $seen = @{{}}; "
        "while ($byId.ContainsKey($id) -and -not $seen.ContainsKey($id)) { "
        "$seen[$id] = 1; $p = $byId[$id]; "
        "[IO.Path]::GetFileNameWithoutExtension($p.Name); $id = [int]$p.ParentProcessId }"
    )
    proc = subprocess.run(
        ["powershell.exe", "-NoProfile", "-Command", script],
        capture_output=True,
        text=True,
        check=False,
    )
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


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


def probe_readiness_facts(root: Path, git_prov: dict[str, Any] | None = None) -> dict[str, Any]:
    """Gather the machine facts the readiness check judges, plus context it only records."""
    gpu_state = query_gpu_state()
    return {
        "uptime_seconds": uptime_seconds(),
        "app_windows": list_app_windows(),
        "launcher_processes": list_launcher_processes(),
        "stoquant_pids": list_processes("stoquant"),
        "git_dirty_files": (collect_git_provenance(root) if git_prov is None else git_prov)[
            "dirty_files"
        ],
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
    allowed = {name.lower() for name in [*allowlist, *facts["launcher_processes"]]}
    for window in facts["app_windows"]:
        if window["process"].lower() not in allowed:
            failures.append(f"open app window: {window['process']} ({window['title']})")
    if facts["stoquant_pids"]:
        pids = ", ".join(str(pid) for pid in facts["stoquant_pids"])
        failures.append(f"stoquant already running (pid {pids})")
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
