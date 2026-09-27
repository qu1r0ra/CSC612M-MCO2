"""Measure the reference K1 scale reduction against device memory bandwidth (issue #23).

Records the baseline that any K1 change is judged against, before the change:

- the device's theoretical DRAM bandwidth from its attributes (2 * memory clock *
  bus width / 8, the DDR convention of the CUDA Best Practices Guide);
- the achievable streaming-read ceiling from `build/stream_probe`;
- K1's effective bandwidth on the CUDA resident path at every matrix cell, over
  several processes, each with the protocol's in-process warm-up.

Effective bandwidth counts the contract-forced minimum traffic: two full reads of
the input, one for max|x| and one for the squared terms, which need max|x| first.
Fractions of peak are reported only where the input exceeds the L2 cache; below
that, K1 runs from L2 and is bound by its kernel launches, so the launch count is
recorded instead.

    just k1-baseline --output-dir results/<date>-<sha>-k1-baseline
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import statistics
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from benchmark_driver import (
    BUILD_RECIPE,
    DEFAULT_BITS,
    DEFAULT_COMPRESSION_SEED,
    DEFAULT_COUNTS,
    DEFAULT_IN_PROCESS_WARMUP_SECONDS,
    collect_build_commands,
    collect_git_provenance,
    collect_hardware_and_toolchain,
    compute_case_statistics,
    find_binary,
    generate_inputs,
    in_process_warmups,
    query_gpu_state,
    run_bench_process,
    warm_up_gpu,
)

REDUCTION_THREADS = 256
BYTES_PER_VALUE = 4
K1_INPUT_READS = 2
DEFAULT_PROCESSES = 12
DEFAULT_REPS = 30
MIN_WARMUPS = 10
CALIBRATION_REPS = 10
GPU_WARMUP_SECONDS = 20.0
ORDER_SEED = 23
PROBE_RECIPE = "build-stream-probe"
PEAK_RULE = (
    "theoretical_peak_gbps = 2 * memory_clock_khz * 1e3 * bus_width_bits / 8 / 1e9 "
    "(DDR convention, CUDA C++ Best Practices Guide)"
)
EFFECTIVE_RULE = (
    "effective_gbps = K1_INPUT_READS * 4 * count / k1 pooled median (ms) / 1e6; "
    "the contract forces two full input reads (max|x|, then the squared terms)"
)
REGIME_RULE = "dram when 4 * count > L2 size, else l2-resident; fractions only for dram"


def k1_launch_count(count: int) -> int:
    """Kernels K1 launches: max, reduce-max, sum, one per pair-tree level, finish."""
    blocks = -(-count // REDUCTION_THREADS)
    padded = 1
    levels = 0
    while padded < blocks:
        padded <<= 1
        levels += 1
    return 4 + levels


def k1_min_bytes(count: int) -> int:
    return K1_INPUT_READS * BYTES_PER_VALUE * count


def theoretical_peak_gbps(memory_clock_khz: int, bus_width_bits: int) -> float:
    return 2.0 * memory_clock_khz * 1e3 * bus_width_bits / 8.0 / 1e9


def regime(count: int, l2_bytes: int) -> str:
    return "dram" if count * BYTES_PER_VALUE > l2_bytes else "l2-resident"


def probe_ceiling_gbps(probe: dict[str, Any], l2_bytes: int) -> float:
    """Best streaming-read bandwidth over the probe sizes that exceed L2."""
    dram = [size["best_gbps"] for size in probe["sizes"] if size["bytes"] > l2_bytes]
    if not dram:
        raise ValueError("the probe has no size larger than L2; raise --max-exp")
    return max(dram)


def summarize_cell(
    count: int,
    bits: int,
    trial_k1_samples: list[list[float]],
    stage_trial_medians: dict[str, list[float]],
    device: dict[str, Any],
    ceiling_gbps: float,
) -> dict[str, Any]:
    k1 = compute_case_statistics(trial_k1_samples)
    min_bytes = k1_min_bytes(count)
    effective = min_bytes / k1["median_ms"] / 1e6
    in_dram = regime(count, device["l2_bytes"]) == "dram"
    peak = theoretical_peak_gbps(device["memory_clock_khz"], device["bus_width_bits"])
    ideal_ms = min_bytes / ceiling_gbps / 1e6
    return {
        "count": count,
        "bits": bits,
        "regime": regime(count, device["l2_bytes"]),
        "launches": k1_launch_count(count),
        "min_bytes": min_bytes,
        "k1": k1,
        "effective_gbps": effective,
        "fraction_of_peak": effective / peak if in_dram else None,
        "fraction_of_ceiling": effective / ceiling_gbps if in_dram else None,
        "ideal_ms_at_ceiling": ideal_ms if in_dram else None,
        "headroom": k1["median_ms"] / ideal_ms if in_dram else None,
        **{
            f"{stage[:-3]}_median_ms": statistics.median(values)
            for stage, values in stage_trial_medians.items()
        },
    }


def probe_path(root: Path) -> Path:
    path = root / "build" / ("stream_probe.exe" if os.name == "nt" else "stream_probe")
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found. Build it first with just {PROBE_RECIPE}.")
    return path


def dry_run(root: Path, recipe: str) -> list[str]:
    proc = subprocess.run(
        ["just", "--dry-run", recipe], cwd=root, capture_output=True, text=True, check=False
    )
    text = proc.stderr if proc.stderr.strip() else proc.stdout
    return [line.strip() for line in text.splitlines() if line.strip()]


def run_probe(probe: Path) -> dict[str, Any]:
    proc = subprocess.run([str(probe)], capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"stream probe failed: {proc.stderr.strip()}")
    return json.loads(proc.stdout)


def run_resident(binary: Path, input_path: Path, bits: int, warmups: int, reps: int) -> dict:
    payload, error = run_bench_process(
        binary,
        input_path,
        bits=bits,
        backend="cuda",
        extra_args=["--boundary", "resident"],
        seed=DEFAULT_COMPRESSION_SEED,
        warmups=warmups,
        reps=reps,
    )
    if payload is None:
        raise RuntimeError(error)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--counts", type=int, nargs="+", default=list(DEFAULT_COUNTS))
    parser.add_argument("--bits", type=int, nargs="+", default=list(DEFAULT_BITS))
    parser.add_argument("--processes", type=int, default=DEFAULT_PROCESSES)
    parser.add_argument("--reps", type=int, default=DEFAULT_REPS)
    parser.add_argument("--gpu-warmup-seconds", type=float, default=GPU_WARMUP_SECONDS)
    parser.add_argument(
        "--in-process-warmup-seconds", type=float, default=DEFAULT_IN_PROCESS_WARMUP_SECONDS
    )
    parser.add_argument(
        "--allow-dirty", action="store_true", help="Permit a dirty tree (non-evidence runs only)"
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    git = collect_git_provenance(root)
    if git["git_dirty"] and not args.allow_dirty:
        sys.exit("Working tree is dirty; a baseline must trace to a committed revision.")
    out = args.output_dir
    if out.exists() and any(out.iterdir()):
        sys.exit(f"{out} is not empty; snapshots are never overwritten.")
    out.mkdir(parents=True, exist_ok=True)
    binary = find_binary(root)
    probe = run_probe(probe_path(root))
    device = probe["device"]
    ceiling = probe_ceiling_gbps(probe, device["l2_bytes"])
    (out / "stream_probe.json").write_text(json.dumps(probe, indent=2), encoding="utf-8")

    cells = [(count, bits) for count in args.counts for bits in args.bits]
    records: list[dict[str, Any]] = []
    gpu_start = query_gpu_state()
    with tempfile.TemporaryDirectory(prefix="mco2-k1-") as scratch:
        inputs = generate_inputs(args.counts, Path(scratch))
        paths = {count: Path(inputs[count]["_path"]) for count in args.counts}
        warm = warm_up_gpu(binary, paths[max(args.counts)], args.gpu_warmup_seconds)
        warmups = {}
        for count, bits in cells:
            payload = run_resident(binary, paths[count], bits, MIN_WARMUPS, CALIBRATION_REPS)
            rep_ms = statistics.median(payload["samples_ms"])
            warmups[count, bits] = in_process_warmups(
                MIN_WARMUPS, args.in_process_warmup_seconds, rep_ms
            )
        rng = random.Random(ORDER_SEED)
        for process in range(args.processes):
            order = list(cells)
            rng.shuffle(order)
            for position, (count, bits) in enumerate(order):
                payload = run_resident(binary, paths[count], bits, warmups[count, bits], args.reps)
                records.append(
                    {
                        "process": process,
                        "position": position,
                        "count": count,
                        "bits": bits,
                        "warmups": warmups[count, bits],
                        "reps": args.reps,
                        "samples_ms": payload["samples_ms"],
                        "k1_ms": payload["k1_ms"],
                        "k2_ms": payload["k2_ms"],
                        "k3_ms": payload["k3_ms"],
                    }
                )
            print(f"process round {process + 1}/{args.processes}", flush=True)
    gpu_end = query_gpu_state()

    cell_summaries = []
    for count, bits in cells:
        rows = [r for r in records if r["count"] == count and r["bits"] == bits]
        cell_summaries.append(
            summarize_cell(
                count,
                bits,
                [r["k1_ms"] for r in rows],
                {
                    stage: [statistics.median(r[stage]) for r in rows]
                    for stage in ("k2_ms", "k3_ms", "samples_ms")
                },
                device,
                ceiling,
            )
        )

    with (out / "processes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["process", "position", "count", "bits", "warmups", "reps"]
            + [f"{key}_median" for key in ("samples_ms", "k1_ms", "k2_ms", "k3_ms")]
        )
        for r in records:
            writer.writerow(
                [r["process"], r["position"], r["count"], r["bits"], r["warmups"], r["reps"]]
                + [statistics.median(r[key]) for key in ("samples_ms", "k1_ms", "k2_ms", "k3_ms")]
            )
    (out / "processes.json").write_text(json.dumps(records), encoding="utf-8")

    build = collect_build_commands(root)
    build.pop("_host_tokens", None)
    summary = {
        "created_utc": datetime.now(UTC).isoformat(),
        "purpose": "issue #23 K1 baseline before any K1 change",
        "git": git,
        "build": {BUILD_RECIPE: build["commands"], PROBE_RECIPE: dry_run(root, PROBE_RECIPE)},
        **collect_hardware_and_toolchain(root),
        "device": device,
        "rules": {"peak": PEAK_RULE, "effective": EFFECTIVE_RULE, "regime": REGIME_RULE},
        "theoretical_peak_gbps": theoretical_peak_gbps(
            device["memory_clock_khz"], device["bus_width_bits"]
        ),
        "probe_ceiling_gbps": ceiling,
        "protocol": {
            "path": "cuda resident",
            "processes": args.processes,
            "reps_per_process": args.reps,
            "in_process_warmup_seconds": args.in_process_warmup_seconds,
            "min_warmups": MIN_WARMUPS,
            "order": f"cells shuffled per process round, random.Random({ORDER_SEED})",
            "seed": DEFAULT_COMPRESSION_SEED,
        },
        "inputs": [{k: v for k, v in inputs[c].items() if k != "_path"} for c in args.counts],
        "initial_gpu_warm_up": warm,
        "gpu_state_start": gpu_start,
        "gpu_state_end": gpu_end,
        "cells": cell_summaries,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for cell in cell_summaries:
        fraction = cell["fraction_of_peak"]
        print(
            f"n=2^{cell['count'].bit_length() - 1} b={cell['bits']} {cell['regime']:>11} "
            f"k1={cell['k1']['median_ms']:.4f} ms  {cell['effective_gbps']:.1f} GB/s  "
            + (
                f"{fraction:.1%} of peak"
                if fraction is not None
                else f"{cell['launches']} launches"
            )
        )


if __name__ == "__main__":
    main()
