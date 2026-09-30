"""Measure the reference K1 scale reduction against device memory bandwidth (issue #23).

Records the baseline that any K1 change is judged against, before the change:

- the device's theoretical DRAM bandwidth from its attributes (2 * memory clock *
  bus width / 8, the DDR convention of the CUDA Best Practices Guide);
- the achievable streaming-read ceiling from `build/stream_probe`;
- K1's effective bandwidth on the CUDA resident path at every matrix cell, over
  several processes, each with the protocol's in-process warm-up.

Effective bandwidth counts the contract-forced minimum traffic: two full reads of
the input, one for max|x| and one for the squared terms, which need max|x| first.
Fractions of peak are reported only where the input is at least four times the L2
cache, so neither K1's second read nor the probe's repeated scans can be served
from L2. Smaller inputs are labelled transitional (1-4x L2) or l2-resident, and
the launch count is recorded instead.

    just k1-baseline --output-dir results/<date>-<sha>-k1-baseline
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from stoquant import layout
from stoquant.design import DEFAULT_BITS, DEFAULT_COUNTS
from stoquant.resident_experiment import (
    DEFAULT_PROCESSES,
    DEFAULT_REPS,
    GPU_WARMUP_SECONDS,
    MIN_WARMUPS,
    ORDER_SEED,
    ResidentExperimentConfig,
    ResidentExperimentRun,
    run_resident_experiment,
)
from stoquant.runner import (
    DEFAULT_COMPRESSION_SEED,
    DEFAULT_IN_PROCESS_WARMUP_SECONDS,
    REAL_BENCH_PROCESS,
    BenchProcess,
)
from stoquant.stats import compute_case_statistics

# SQ_CUDA_REDUCTION_THREADS in native/quantizer_cuda.cu.
REDUCTION_THREADS = 256
BYTES_PER_VALUE = 4
K1_INPUT_READS = 2
DRAM_L2_MULTIPLE = 4
# One size past the largest matrix count, so the ceiling is not set at the range edge.
PROBE_MAX_EXP = 27
PEAK_RULE = (
    "theoretical_peak_gbps = 2 * memory_clock_khz * 1e3 * bus_width_bits / 8 / 1e9 "
    "(DDR convention, CUDA C++ Best Practices Guide); the RTX 5060 vendor figure is "
    "448 GB/s (28 Gbps x 128-bit GDDR7)"
)
EFFECTIVE_RULE = (
    "effective_gbps = K1_INPUT_READS * 4 * count / k1 pooled median (ms) / 1e6; "
    "the contract forces two full input reads (max|x|, then the squared terms)"
)
REGIME_RULE = (
    f"dram when 4 * count >= {DRAM_L2_MULTIPLE} * L2 size, transitional when 4 * count > L2 "
    "size, else l2-resident; fractions, headroom and the probe ceiling use dram sizes only"
)


STAGE_KEYS = {
    "k2_ms": "k2_median_ms",
    "k3_ms": "k3_median_ms",
    "samples_ms": "total_median_ms",
}


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


def regime_for_bytes(size_bytes: int, l2_bytes: int) -> str:
    if size_bytes >= DRAM_L2_MULTIPLE * l2_bytes:
        return "dram"
    return "transitional" if size_bytes > l2_bytes else "l2-resident"


def regime(count: int, l2_bytes: int) -> str:
    return regime_for_bytes(count * BYTES_PER_VALUE, l2_bytes)


def probe_ceiling(probe: dict[str, Any], l2_bytes: int) -> dict[str, Any]:
    """Best streaming-read bandwidth over the probe sizes in the dram regime."""
    dram = [s for s in probe["sizes"] if regime_for_bytes(s["bytes"], l2_bytes) == "dram"]
    if not dram:
        raise ValueError(
            f"the probe has no size of at least {DRAM_L2_MULTIPLE}x L2; raise --max-exp"
        )
    best = max(dram, key=lambda s: s["best_gbps"])
    return {"gbps": best["best_gbps"], "count": best["count"], "bytes": best["bytes"]}


def probe_ceiling_gbps(probe: dict[str, Any], l2_bytes: int) -> float:
    return probe_ceiling(probe, l2_bytes)["gbps"]


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
    cell_regime = regime(count, device["l2_bytes"])
    in_dram = cell_regime == "dram"
    peak = theoretical_peak_gbps(device["memory_clock_khz"], device["bus_width_bits"])
    ideal_ms = min_bytes / ceiling_gbps / 1e6
    return {
        "count": count,
        "bits": bits,
        "regime": cell_regime,
        "launches": k1_launch_count(count),
        "min_bytes": min_bytes,
        "k1": k1,
        "effective_gbps": effective,
        "fraction_of_peak": effective / peak if in_dram else None,
        "fraction_of_ceiling": effective / ceiling_gbps if in_dram else None,
        "ideal_ms_at_ceiling": ideal_ms if in_dram else None,
        "headroom": k1["median_ms"] / ideal_ms if in_dram else None,
        **{
            STAGE_KEYS[stage]: statistics.median(values)
            for stage, values in stage_trial_medians.items()
        },
    }


def analyze_baseline(run: ResidentExperimentRun, out: Path) -> None:
    ceiling = probe_ceiling(run.probe, run.device["l2_bytes"])
    cell_summaries = []
    for count, bits in run.cells:
        rows = [
            record for record in run.records if record["count"] == count and record["bits"] == bits
        ]
        cell_summaries.append(
            summarize_cell(
                count,
                bits,
                [record["k1_ms"] for record in rows],
                {
                    stage: [statistics.median(record[stage]) for record in rows]
                    for stage in ("k2_ms", "k3_ms", "samples_ms")
                },
                run.device,
                ceiling["gbps"],
            )
        )

    with (out / "processes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["process", "position", "count", "bits", "warmups", "reps"]
            + [f"{key}_median" for key in ("samples_ms", "k1_ms", "k2_ms", "k3_ms")]
        )
        for record in run.records:
            writer.writerow(
                [
                    record["process"],
                    record["position"],
                    record["count"],
                    record["bits"],
                    record["warmups"],
                    record["reps"],
                ]
                + [
                    statistics.median(record[key])
                    for key in ("samples_ms", "k1_ms", "k2_ms", "k3_ms")
                ]
            )
    (out / "processes.json").write_text(json.dumps(run.records), encoding="utf-8")

    summary = {
        "created_utc": datetime.now(UTC).isoformat(),
        "purpose": "issue #23 K1 baseline before any K1 change",
        "git": run.git,
        "build": run.build_record,
        **run.hardware,
        "device": run.device,
        "rules": {"peak": PEAK_RULE, "effective": EFFECTIVE_RULE, "regime": REGIME_RULE},
        "theoretical_peak_gbps": theoretical_peak_gbps(
            run.device["memory_clock_khz"], run.device["bus_width_bits"]
        ),
        "probe_ceiling_gbps": ceiling["gbps"],
        "probe_ceiling_size": {"count": ceiling["count"], "bytes": ceiling["bytes"]},
        "gpu_state_probe": {"start": run.gpu_probe_start, "end": run.gpu_probe_end},
        "protocol": {
            "path": "cuda resident",
            "processes": run.processes,
            "reps_per_process": run.reps,
            "in_process_warmup_seconds": run.in_process_warmup_seconds,
            "min_warmups": MIN_WARMUPS,
            "order": f"cells shuffled per process round, random.Random({ORDER_SEED})",
            "seed": DEFAULT_COMPRESSION_SEED,
            "affinity_mask": run.affinity_mask,
            "excluded_logical_cpus": list(run.excluded_logical_cpus),
        },
        "inputs": list(run.inputs.values()),
        "initial_gpu_warm_up": run.initial_gpu_warm_up,
        "gpu_state_start": run.gpu_state_start,
        "gpu_state_end": run.gpu_state_end,
        "cells": cell_summaries,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for cell in cell_summaries:
        fraction = cell["fraction_of_peak"]
        print(
            f"n=2^{cell['count'].bit_length() - 1} b={cell['bits']} {cell['regime']:>12} "
            f"k1={cell['k1']['median_ms']:.4f} ms  {cell['effective_gbps']:.1f} GB/s  "
            + (
                f"{fraction:.1%} of peak"
                if fraction is not None
                else f"{cell['launches']} launches"
            )
        )


def run_k1_baseline(
    config: ResidentExperimentConfig,
    adapter: BenchProcess = REAL_BENCH_PROCESS,
) -> Path:
    return run_resident_experiment(
        config,
        variants=("reference",),
        analyze=analyze_baseline,
        adapter=adapter,
    )


def main(argv: list[str] | None = None, prog: str | None = None) -> None:
    parser = argparse.ArgumentParser(prog=prog, description=__doc__)
    parser.add_argument("--output-dir", type=Path)
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
    args = parser.parse_args(argv)
    config = ResidentExperimentConfig(
        root=layout.ROOT,
        kind="k1-baseline",
        counts=args.counts,
        bits=args.bits,
        processes=args.processes,
        reps=args.reps,
        gpu_warmup_seconds=args.gpu_warmup_seconds,
        in_process_warmup_seconds=args.in_process_warmup_seconds,
        output_dir=args.output_dir,
        allow_dirty=args.allow_dirty,
    )
    run_k1_baseline(config)


if __name__ == "__main__":
    main()
