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
import hashlib
import json
import random
import statistics
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from benchmark_driver import (
    BUILD_RECIPE,
    DEFAULT_BITS,
    DEFAULT_COMPRESSION_SEED,
    DEFAULT_COUNTS,
    DEFAULT_IN_PROCESS_WARMUP_SECONDS,
    EXCLUDED_LOGICAL_CPUS,
    affinity_mask_excluding,
    claim_support,
    collect_build_commands,
    collect_git_provenance,
    collect_hardware_and_toolchain,
    compare_speedup,
    compute_case_statistics,
    compute_stage_medians,
    find_binary,
    generate_inputs,
    get_process_affinity,
    in_process_warmups,
    probe_readiness_facts,
    process_affinity,
    query_gpu_state,
    run_bench_process,
    warm_up_gpu,
)
from k1_bandwidth import (
    CALIBRATION_REPS,
    DEFAULT_PROCESSES,
    DEFAULT_REPS,
    GPU_WARMUP_SECONDS,
    MIN_WARMUPS,
    ORDER_SEED,
    PROBE_RECIPE,
    REDUCTION_THREADS,
    dry_run,
    file_sha256,
    probe_ceiling,
    probe_path,
    run_probe,
    summarize_cell,
    theoretical_peak_gbps,
)

VARIANTS = ("reference", "optimized")
# MCO2_CUDA_K1_TREE_SPAN in src/quantizer_cuda.cu.
TREE_SPAN = 2048
PILOT_COUNTS = (1 << 10, 1 << 14, 1 << 20, 1 << 25, 1 << 26)
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


def variant_order(process: int) -> tuple[str, ...]:
    return VARIANTS if process % 2 == 0 else VARIANTS[::-1]


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


def record_hash(binary: Path, input_path: Path, bits: int, variant: str, out: Path) -> str:
    command = [
        str(binary),
        "compress",
        "--input",
        str(input_path),
        "--output",
        str(out),
        "--seed",
        str(DEFAULT_COMPRESSION_SEED),
        "--bits",
        str(bits),
        "--tensor-id",
        "0",
        "--invocation-id",
        "0",
        "--backend",
        "cuda",
        "--k1",
        variant,
    ]
    proc = subprocess.run(command, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"mco2 compress --k1 {variant} failed: {proc.stderr.strip()}")
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    out.unlink()
    return digest


def run_arm(binary: Path, input_path: Path, bits: int, variant: str, warmups: int, reps: int):
    payload, error = run_bench_process(
        binary,
        input_path,
        bits=bits,
        backend="cuda",
        extra_args=["--boundary", "resident", "--k1", variant],
        seed=DEFAULT_COMPRESSION_SEED,
        warmups=warmups,
        reps=reps,
    )
    if payload is None:
        raise RuntimeError(error)
    if payload["configuration"].get("k1") != variant:
        raise RuntimeError(f"bench did not record k1={variant}")
    return payload


def summarize_ab_cell(
    count: int, bits: int, rows: list[dict[str, Any]], device: dict[str, Any], ceiling: float
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
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
    args = parser.parse_args()

    if args.figure is not None:
        summary = json.loads((args.figure / "summary.json").read_text(encoding="utf-8"))
        plot_f5(summary, args.figure / "f5_k1_stages.png")
        return
    if args.output_dir is None:
        sys.exit("--output-dir is required unless --figure is given.")
    counts = args.counts or list(PILOT_COUNTS if args.pilot else DEFAULT_COUNTS)
    processes = args.processes or (PILOT_PROCESSES if args.pilot else DEFAULT_PROCESSES)
    evidence = not (args.pilot or args.allow_dirty or args.counts or args.processes)

    root = Path(__file__).resolve().parent
    git = collect_git_provenance(root)
    if git["git_dirty"] and not args.allow_dirty:
        sys.exit("Working tree is dirty; an A/B run must trace to a committed revision.")
    out = args.output_dir
    if out.exists() and any(out.iterdir()):
        sys.exit(f"{out} is not empty; snapshots are never overwritten.")
    binary = find_binary(root)
    probe_binary = probe_path(root)
    build = collect_build_commands(root)
    build.pop("_host_tokens", None)
    if not build["commands"]:
        sys.exit(f"just --dry-run {BUILD_RECIPE} failed; build provenance cannot be recorded.")
    try:
        probe_commands = dry_run(root, PROBE_RECIPE)
    except RuntimeError as error:
        sys.exit(str(error))
    build_record = {
        BUILD_RECIPE: build["commands"],
        PROBE_RECIPE: probe_commands,
        "sha256": {
            binary.name: file_sha256(binary),
            probe_binary.name: file_sha256(probe_binary),
        },
    }

    cells = [(count, bits) for count in counts for bits in args.bits]
    records: list[dict[str, Any]] = []
    readiness = probe_readiness_facts(root)
    # Revision 3: every probe and benchmark process runs off physical core 0.
    mask = affinity_mask_excluding(EXCLUDED_LOGICAL_CPUS, get_process_affinity())
    with process_affinity(mask), tempfile.TemporaryDirectory(prefix="mco2-k1ab-") as scratch:
        gpu_start = query_gpu_state()
        inputs = generate_inputs(counts, Path(scratch))
        paths = {count: Path(inputs[count]["_path"]) for count in counts}
        identity = []
        for count, bits in cells:
            hashes = {
                variant: record_hash(
                    binary, paths[count], bits, variant, Path(scratch) / "record.msq"
                )
                for variant in VARIANTS
            }
            identity.append({"count": count, "bits": bits, "record_sha256": hashes})
            if len(set(hashes.values())) != 1:
                sys.exit(f"records differ at count={count} bits={bits}: {hashes}")
        warm = warm_up_gpu(binary, paths[max(counts)], args.gpu_warmup_seconds)
        gpu_probe_start = query_gpu_state()
        probe = run_probe(probe_binary)
        gpu_probe_end = query_gpu_state()
        device = probe["device"]
        ceiling = probe_ceiling(probe, device["l2_bytes"])
        out.mkdir(parents=True, exist_ok=True)
        (out / "stream_probe.json").write_text(json.dumps(probe, indent=2), encoding="utf-8")
        warmups = {}
        for count, bits in cells:
            for variant in VARIANTS:
                payload = run_arm(
                    binary, paths[count], bits, variant, MIN_WARMUPS, CALIBRATION_REPS
                )
                rep_ms = statistics.median(payload["samples_ms"])
                warmups[count, bits, variant] = in_process_warmups(
                    MIN_WARMUPS, args.in_process_warmup_seconds, rep_ms
                )
        rng = random.Random(ORDER_SEED)
        for process in range(processes):
            order = list(cells)
            rng.shuffle(order)
            for position, (count, bits) in enumerate(order):
                for slot, variant in enumerate(variant_order(process)):
                    n = warmups[count, bits, variant]
                    payload = run_arm(binary, paths[count], bits, variant, n, args.reps)
                    records.append(
                        {
                            "process": process,
                            "position": position,
                            "slot": slot,
                            "variant": variant,
                            "count": count,
                            "bits": bits,
                            "warmups": n,
                            "reps": args.reps,
                            "samples_ms": payload["samples_ms"],
                            "k1_ms": payload["k1_ms"],
                            "k2_ms": payload["k2_ms"],
                            "k3_ms": payload["k3_ms"],
                        }
                    )
            print(f"process round {process + 1}/{processes}", flush=True)
        gpu_end = query_gpu_state()

    cell_summaries = [
        summarize_ab_cell(
            count,
            bits,
            [r for r in records if r["count"] == count and r["bits"] == bits],
            device,
            ceiling["gbps"],
        )
        for count, bits in cells
    ]
    (out / "processes.json").write_text(json.dumps(records), encoding="utf-8")
    summary = {
        "created_utc": datetime.now(UTC).isoformat(),
        "purpose": "issue #23 same-tree A/B of the reference and optimized K1",
        "evidence": evidence,
        "git": git,
        "build": build_record,
        **collect_hardware_and_toolchain(root),
        "device": device,
        "theoretical_peak_gbps": theoretical_peak_gbps(
            device["memory_clock_khz"], device["bus_width_bits"]
        ),
        "probe_ceiling_gbps": ceiling["gbps"],
        "probe_ceiling_size": {"count": ceiling["count"], "bytes": ceiling["bytes"]},
        "gpu_state_probe": {"start": gpu_probe_start, "end": gpu_probe_end},
        "protocol": {
            "path": "cuda resident",
            "variants": list(VARIANTS),
            "measurand": "k1_ms (K1 stage events); the resident total is descriptive",
            "processes_per_variant": processes,
            "reps_per_process": args.reps,
            "in_process_warmup_seconds": args.in_process_warmup_seconds,
            "min_warmups": MIN_WARMUPS,
            "order": ORDER_RULE,
            "seed": DEFAULT_COMPRESSION_SEED,
            "affinity_mask": mask,
            "excluded_logical_cpus": list(EXCLUDED_LOGICAL_CPUS),
            "readiness": "recorded, not enforced",
        },
        "readiness_facts": readiness,
        "record_identity": identity,
        "inputs": [{k: v for k, v in inputs[c].items() if k != "_path"} for c in counts],
        "initial_gpu_warm_up": warm,
        "gpu_state_start": gpu_start,
        "gpu_state_end": gpu_end,
        "cells": cell_summaries,
        "decision": keep_decision(cell_summaries),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plot_f5(summary, out / "f5_k1_stages.png")
    for cell in cell_summaries:
        k1 = cell["k1_comparison"]
        arms = cell["arms"]
        print(
            f"n=2^{cell['count'].bit_length() - 1} b={cell['bits']} {cell['regime']:>12} "
            f"ref={arms['reference']['k1']['median_ms']:.4f} "
            f"opt={arms['optimized']['k1']['median_ms']:.4f} ms  "
            f"x{k1['speedup']:.2f} {k1['verdict']} direction={k1['direction_supported']} "
            f"magnitude={k1['magnitude_supported']}"
        )
    decision = summary["decision"]
    print(f"kept={decision['kept']} over {decision['dram_cells']} dram cells")


if __name__ == "__main__":
    main()
