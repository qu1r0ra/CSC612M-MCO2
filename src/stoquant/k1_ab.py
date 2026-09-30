"""Same-tree A/B of the reference and optimized K1 (issue #23).

Both variants run from one binary on the CUDA resident path, interleaved per
process round, under the revision 3 conditions (core 0 excluded, pooled
statistics, conservative verdict). Before timing, every cell's record is
compressed with both variants and must be byte-identical.

The measurand is K1 stage time. The keep rule is fixed in
`docs/benchmark-protocol.md` before any A/B run: the optimized K1 is kept only if
every `dram` cell is `faster` with `direction_supported`. The resident total is
reported, but no decision depends on it. F5 draws the stage medians of both arms.

    just k1-ab --output-dir results/<date>-<sha>-k1-ab
    just k1-ab --figure results/<date>-<sha>-k1-ab
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from stoquant import layout
from stoquant.design import DEFAULT_BITS, DEFAULT_COUNTS
from stoquant.k1_bandwidth import (
    DEFAULT_PROCESSES,
    DEFAULT_REPS,
    GPU_WARMUP_SECONDS,
    REDUCTION_THREADS,
    probe_ceiling,
    summarize_cell,
    theoretical_peak_gbps,
)
from stoquant.plotting import pyplot
from stoquant.resident_experiment import (
    MIN_WARMUPS,
    ResidentExperimentConfig,
    ResidentExperimentRecord,
    ResidentExperimentRun,
    run_resident_experiment,
    variant_order,
)
from stoquant.runner import (
    DEFAULT_COMPRESSION_SEED,
    DEFAULT_IN_PROCESS_WARMUP_SECONDS,
    REAL_BENCH_PROCESS,
    BenchProcess,
)
from stoquant.snapshot_store import derived_directory
from stoquant.stats import (
    claim_support,
    compare_speedup,
    compute_case_statistics,
    compute_stage_medians,
)

VARIANTS = ("reference", "optimized")
__all__ = ["variant_order"]
# SQ_CUDA_K1_TREE_SPAN in native/quantizer_cuda.cu.
TREE_SPAN = 2048
K1_AB_PILOT_COUNTS = (1 << 10, 1 << 14, 1 << 20, 1 << 25, 1 << 26)
PILOT_PROCESSES = 2
F5_COUNTS = (1 << 14, 1 << 18, 1 << 22, 1 << 25, 1 << 26)
F5_STAGES = (
    ("k1_ms", "K1 scale", "#4C72B0"),
    ("k2_ms", "K2 quantize", "#55A868"),
    ("k3_ms", "K3 pack", "#C44E52"),
    ("other_ms", "other", "#8C8C8C"),
)
KEEP_RULE = (
    "the optimized K1 is kept only if every dram cell (4 * count >= 4 * L2) has K1 verdict "
    "'faster' with direction_supported (revision 3; no boundary, so no inversion). "
    "magnitude_supported and the bootstrap CI are reported; no decision depends on them"
)
ORDER_RULE = (
    "cells shuffled per process round with random.Random(ORDER_SEED); within a cell the "
    "reference runs first in even rounds and the optimized K1 first in odd rounds"
)


def optimized_launch_count(count: int) -> int:
    """Kernels the optimized K1 launches: max, reduce-max, warp sums, tree levels, finish."""
    blocks = -(-count // REDUCTION_THREADS)
    active = 1
    while active < blocks:
        active <<= 1
    launches = 4
    while active > 1:
        active //= min(active, TREE_SPAN)
        launches += 1
    return launches


def compare_arms(
    reference_trials: list[list[float]], optimized_trials: list[list[float]]
) -> dict[str, Any]:
    """Reference over optimized under the revision 3 verdict and claims."""
    reference = compute_case_statistics(reference_trials)
    optimized = compute_case_statistics(optimized_trials)
    comparison = compare_speedup(reference, optimized, "speedup")
    comparison.update(claim_support(comparison["verdict"], False, reference, optimized))
    return comparison


def keep_decision(cells: list[dict[str, Any]]) -> dict[str, Any]:
    dram = [c for c in cells if c["regime"] == "dram"]
    failing = [
        {"count": c["count"], "bits": c["bits"], "verdict": c["k1_comparison"]["verdict"]}
        for c in dram
        if not (
            c["k1_comparison"]["verdict"] == "faster" and c["k1_comparison"]["direction_supported"]
        )
    ]
    slowdowns = [
        {"count": c["count"], "bits": c["bits"], "regime": c["regime"]}
        for c in cells
        if c["regime"] != "dram"
        and c["k1_comparison"]["verdict"] == "slower"
        and c["k1_comparison"]["direction_supported"]
    ]
    return {
        "rule": KEEP_RULE,
        "dram_cells": len(dram),
        "kept": bool(dram) and not failing,
        "failing_cells": failing,
        "descriptive_slowdowns": slowdowns,
    }


def summarize_ab_cell(
    count: int,
    bits: int,
    rows: Sequence[ResidentExperimentRecord],
    device: dict[str, Any],
    ceiling: float,
) -> dict[str, Any]:
    arms = {}
    for variant in VARIANTS:
        runs = [r for r in rows if r["variant"] == variant]
        arm = summarize_cell(
            count,
            bits,
            [r["k1_ms"] for r in runs],
            {
                stage: [statistics.median(r[stage]) for r in runs]
                for stage in ("k2_ms", "k3_ms", "samples_ms")
            },
            device,
            ceiling,
        )
        if variant == "optimized":
            arm["launches"] = optimized_launch_count(count)
        arm["stage_medians_ms"] = compute_stage_medians(runs)
        arm["total"] = compute_case_statistics([r["samples_ms"] for r in runs])
        arms[variant] = arm
    return {
        "count": count,
        "bits": bits,
        "regime": arms["reference"]["regime"],
        "arms": arms,
        "k1_comparison": compare_arms(
            [r["k1_ms"] for r in rows if r["variant"] == "reference"],
            [r["k1_ms"] for r in rows if r["variant"] == "optimized"],
        ),
        "total_comparison": compare_arms(
            [r["samples_ms"] for r in rows if r["variant"] == "reference"],
            [r["samples_ms"] for r in rows if r["variant"] == "optimized"],
        ),
    }


def power_label(count: int) -> str:
    return f"$2^{{{count.bit_length() - 1}}}$"


def plot_f5(summary: dict[str, Any], out: Path) -> None:
    """Stage medians of both arms, as a share of the reference arm's stage total."""
    plt = pyplot()
    cells = {(c["count"], c["bits"]): c for c in summary["cells"]}
    bit_widths = sorted({c["bits"] for c in summary["cells"]})
    counts = [n for n in F5_COUNTS if any((n, b) in cells for b in bit_widths)]
    fig, axes = plt.subplots(
        1, len(bit_widths), figsize=(5.5 * len(bit_widths), 4.4), sharey=True, squeeze=False
    )
    width = 0.38
    for ax, bits in zip(axes[0], bit_widths, strict=True):
        for i, count in enumerate(counts):
            cell = cells.get((count, bits))
            if cell is None:
                continue
            reference = cell["arms"]["reference"]["stage_medians_ms"]
            scale = sum(max(reference[key], 0.0) for key, _, _ in F5_STAGES)
            for offset, variant in zip((-width / 2, width / 2), VARIANTS, strict=True):
                medians = cell["arms"][variant]["stage_medians_ms"]
                bottom = 0.0
                for key, label, color in F5_STAGES:
                    share = 100.0 * max(medians[key], 0.0) / scale
                    ax.bar(
                        i + offset,
                        share,
                        width,
                        bottom=bottom,
                        color=color,
                        edgecolor="white",
                        linewidth=0.4,
                        label=label,
                    )
                    bottom += share
                total = sum(max(medians[key], 0.0) for key, _, _ in F5_STAGES)
                ax.text(
                    i + offset,
                    bottom + 1,
                    f"{variant[0].upper()}\n{total:.3g}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )
        ax.set_xticks(range(len(counts)), [power_label(n) for n in counts])
        ax.set_ylim(0, 125)
        ax.set_title(f"{bits}-bit")
        ax.set_xlabel("elements (R: reference, O: optimized K1; label: stage sum, ms)")
    axes[0][0].set_ylabel("stage median, % of reference stage sum")
    handles, labels = [], []
    for ax in axes[0]:
        for handle, label in zip(*ax.get_legend_handles_labels(), strict=True):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=8)
    fig.suptitle("F5. CUDA resident stage medians before and after the K1 change")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(out, dpi=200)
    plt.close(fig)


def is_default_design(args: argparse.Namespace, counts: Sequence[int], processes: int) -> bool:
    """True only for the resolved default design; any reduction clears the evidence flag."""
    return (
        not (args.pilot or args.allow_dirty)
        and tuple(counts) == DEFAULT_COUNTS
        and tuple(args.bits) == DEFAULT_BITS
        and processes == DEFAULT_PROCESSES
        and args.reps == DEFAULT_REPS
        and args.gpu_warmup_seconds == GPU_WARMUP_SECONDS
        and args.in_process_warmup_seconds == DEFAULT_IN_PROCESS_WARMUP_SECONDS
    )


def analyze_k1_ab(run: ResidentExperimentRun, out: Path) -> None:
    ceiling = probe_ceiling(run.probe, run.device["l2_bytes"])
    cell_summaries = [
        summarize_ab_cell(
            count,
            bits,
            [
                record
                for record in run.records
                if record["count"] == count and record["bits"] == bits
            ],
            run.device,
            ceiling["gbps"],
        )
        for count, bits in run.cells
    ]
    (out / "processes.json").write_text(json.dumps(run.records), encoding="utf-8")
    summary = {
        "created_utc": datetime.now(UTC).isoformat(),
        "purpose": "issue #23 same-tree A/B of the reference and optimized K1",
        "evidence": run.evidence,
        "git": run.git,
        "build": run.build_record,
        **run.hardware,
        "device": run.device,
        "theoretical_peak_gbps": theoretical_peak_gbps(
            run.device["memory_clock_khz"], run.device["bus_width_bits"]
        ),
        "probe_ceiling_gbps": ceiling["gbps"],
        "probe_ceiling_size": {"count": ceiling["count"], "bytes": ceiling["bytes"]},
        "gpu_state_probe": {"start": run.gpu_probe_start, "end": run.gpu_probe_end},
        "protocol": {
            "path": "cuda resident",
            "variants": list(VARIANTS),
            "measurand": "k1_ms (K1 stage events); the resident total is descriptive",
            "processes_per_variant": run.processes,
            "reps_per_process": run.reps,
            "in_process_warmup_seconds": run.in_process_warmup_seconds,
            "min_warmups": MIN_WARMUPS,
            "order": ORDER_RULE,
            "seed": DEFAULT_COMPRESSION_SEED,
            "affinity_mask": run.affinity_mask,
            "excluded_logical_cpus": list(run.excluded_logical_cpus),
            "readiness": "recorded, not enforced",
        },
        "readiness_facts": run.readiness,
        "record_identity": run.record_identity,
        "inputs": list(run.inputs.values()),
        "initial_gpu_warm_up": run.initial_gpu_warm_up,
        "gpu_state_start": run.gpu_state_start,
        "gpu_state_end": run.gpu_state_end,
        "cells": cell_summaries,
        "decision": keep_decision(cell_summaries),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plot_f5(summary, derived_directory(out) / "f5_k1_stages.png")
    for cell in cell_summaries:
        k1 = cell["k1_comparison"]
        arms = cell["arms"]
        speedup = k1["speedup"]
        speedup_text = (
            f"x{speedup:.2f}"
            if isinstance(speedup, (int, float)) and math.isfinite(speedup)
            else "speedup unavailable"
        )
        print(
            f"n=2^{cell['count'].bit_length() - 1} b={cell['bits']} {cell['regime']:>12} "
            f"ref={arms['reference']['k1']['median_ms']:.4f} "
            f"opt={arms['optimized']['k1']['median_ms']:.4f} ms  "
            f"{speedup_text} {k1['verdict']} direction={k1['direction_supported']} "
            f"magnitude={k1['magnitude_supported']}"
        )
    decision = summary["decision"]
    print(f"kept={decision['kept']} over {decision['dram_cells']} dram cells")


def run_k1_ab(
    config: ResidentExperimentConfig,
    adapter: BenchProcess = REAL_BENCH_PROCESS,
) -> Path:
    return run_resident_experiment(
        config,
        variants=VARIANTS,
        analyze=analyze_k1_ab,
        adapter=adapter,
    )


def main(argv: list[str] | None = None, prog: str | None = None) -> None:
    parser = argparse.ArgumentParser(prog=prog, description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--figure", type=Path, help="Re-render F5 from an A/B snapshot and stop")
    parser.add_argument(
        "--pilot", action="store_true", help="Pilot cells and processes; never evidence"
    )
    parser.add_argument("--counts", type=int, nargs="+")
    parser.add_argument("--bits", type=int, nargs="+", default=list(DEFAULT_BITS))
    parser.add_argument("--processes", type=int)
    parser.add_argument("--reps", type=int, default=DEFAULT_REPS)
    parser.add_argument("--gpu-warmup-seconds", type=float, default=GPU_WARMUP_SECONDS)
    parser.add_argument(
        "--in-process-warmup-seconds", type=float, default=DEFAULT_IN_PROCESS_WARMUP_SECONDS
    )
    parser.add_argument(
        "--allow-dirty", action="store_true", help="Permit a dirty tree (non-evidence runs only)"
    )
    args = parser.parse_args(argv)

    if args.figure is not None:
        summary = json.loads((args.figure / "summary.json").read_text(encoding="utf-8"))
        plot_f5(summary, derived_directory(args.figure) / "f5_k1_stages.png")
        return
    counts = args.counts or list(K1_AB_PILOT_COUNTS if args.pilot else DEFAULT_COUNTS)
    processes = args.processes or (PILOT_PROCESSES if args.pilot else DEFAULT_PROCESSES)
    evidence = is_default_design(args, counts, processes)
    config = ResidentExperimentConfig(
        root=layout.ROOT,
        kind="k1-ab",
        counts=counts,
        bits=args.bits,
        processes=processes,
        reps=args.reps,
        gpu_warmup_seconds=args.gpu_warmup_seconds,
        in_process_warmup_seconds=args.in_process_warmup_seconds,
        output_dir=args.output_dir,
        allow_dirty=args.allow_dirty,
        pilot=args.pilot,
        evidence=evidence,
        record_readiness=True,
    )
    run_k1_ab(config)


if __name__ == "__main__":
    main()
