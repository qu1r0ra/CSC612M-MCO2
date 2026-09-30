"""Host control: readiness checks, process affinity, power-plan, HAGS and window probes."""

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


def list_app_windows() -> list[dict[str, str]]:
    """Top-level windows with a title, as (process, title) pairs."""
    if os.name != "nt":
        return []
    script = (
        "$ErrorActionPreference = 'Stop'; "
        "[Console]::OutputEncoding = [Text.Encoding]::UTF8; "
        "$windows = @(Get-Process | "
        "Where-Object { $_.MainWindowHandle -ne 0 -and $_.MainWindowTitle } | "
        "ForEach-Object { [PSCustomObject]@{ process = $_.ProcessName; title = $_.MainWindowTitle } }); "
        "ConvertTo-Json -InputObject $windows -Compress"
    )
    proc = subprocess.run(
        ["powershell.exe", "-NoProfile", "-Command", script],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    output = _checked_probe_output(proc, "list_app_windows").strip()
    rows = _parse_json_array(output, "list_app_windows")
    windows = []
    for row in rows:
        if (
            not isinstance(row, dict)
            or not isinstance(row.get("process"), str)
            or not row["process"].strip()
            or not isinstance(row.get("title"), str)
            or not row["title"].strip()
        ):
            raise RuntimeError("list_app_windows returned an unparseable row")
        windows.append({"process": row["process"].strip(), "title": row["title"].strip()})
    return windows


def list_launcher_processes() -> list[str]:
    """Process names from this driver up its parent chain, innermost first."""
    if os.name != "nt":
        return []
    script = (
        "$ErrorActionPreference = 'Stop'; "
        "$byId = @{}; Get-CimInstance Win32_Process | "
        "ForEach-Object { $byId[[int]$_.ProcessId] = $_ }; "
        f"$id = {os.getpid()}; $seen = @{{}}; $processes = @(); "
        "while ($byId.ContainsKey($id) -and -not $seen.ContainsKey($id)) { "
        "$seen[$id] = 1; $p = $byId[$id]; "
        "$processes += [IO.Path]::GetFileNameWithoutExtension($p.Name); "
        "$id = [int]$p.ParentProcessId }; "
        "ConvertTo-Json -InputObject $processes -Compress"
    )
    proc = subprocess.run(
        ["powershell.exe", "-NoProfile", "-Command", script],
        capture_output=True,
        text=True,
        check=False,
    )
    output = _checked_probe_output(proc, "list_launcher_processes").strip()
    processes = _parse_json_array(output, "list_launcher_processes")
    if not processes or any(not isinstance(name, str) or not name.strip() for name in processes):
        raise RuntimeError("list_launcher_processes returned an unparseable process list")
    return [name.strip() for name in processes]


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
        "uptime_seconds": probe("uptime_seconds", uptime_seconds, 0.0),
        "app_windows": probe("list_app_windows", list_app_windows, []),
        "launcher_processes": probe("list_launcher_processes", list_launcher_processes, []),
        "stoquant_pids": probe("list_processes", lambda: list_processes("stoquant"), []),
        "git_dirty_files": dirty_files,
        "gpu_clock_event_reasons": gpu_state.get("clocks_event_reasons.active"),
        "power_plan": probe("query_power_plan", query_power_plan, "unavailable"),
        "hags_hwschmode": probe("query_hags", query_hags, "unavailable"),
    }
    if probe_failures:
        facts["probe_failures"] = probe_failures
    return facts


def check_readiness(
    facts: dict[str, Any], allowlist: Sequence[str] = WINDOW_ALLOWLIST
) -> list[str]:
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
