"""Shared fixtures and helpers for the split matrix-driver tests."""

import pytest

from stoquant import layout
from stoquant.matrix import run_benchmark_matrix

ROOT = layout.ROOT
CUDA_TEST = pytest.mark.cuda

READY_FACTS = {
    "uptime_seconds": 600.0,
    "app_windows": [
        {"process": "WindowsTerminal", "title": "Terminal"},
        {"process": "TextInputHost", "title": "Windows Input Experience"},
    ],
    "launcher_processes": ["python", "uv", "just", "pwsh", "WindowsTerminal", "explorer"],
    "stoquant_pids": [],
    "git_dirty_files": [],
    "gpu_clock_event_reasons": "0x0000000000000001",
    "power_plan": "Balanced",
    "hags_hwschmode": "unset",
}


def run_cpu_snapshot(output, **kwargs):
    return run_benchmark_matrix(
        root=ROOT,
        output_dir=output,
        counts=[1024],
        bit_widths=[4],
        backends=["cpu"],
        warmups=1,
        reps=1,
        trials=1,
        in_process_warmup_seconds=0,
        allow_dirty=True,
        **kwargs,
    )


def fake_stats(median: float) -> dict:
    return {
        "median_ms": median,
        "trial_median_min_ms": median * 0.99,
        "trial_median_max_ms": median * 1.01,
        "trial_medians_ms": [median * 0.99, median, median * 1.01],
        "stable": True,
        "unstable_rev2": False,
    }
