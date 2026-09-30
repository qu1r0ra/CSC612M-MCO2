"""Render the size-sweep figures, the T1 summary table and the crossover report.

Reads a frozen snapshot (manifest.json plus one JSON per case) and writes
F1-F4 as PNG files and ``report.md`` into the output directory, which defaults
to the snapshot's ``derived/`` folder. Raw snapshot files are never rewritten.
"""

from __future__ import annotations

import argparse
import itertools
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import numpy as np

from stoquant.design import (
    AVX2_COMPARATOR,
    COMPARATOR,
    CUDA_GPU_ORIGIN,
    CUDA_HOST_ORIGIN,
    CUDA_RESIDENT,
    CUDA_RESIDENT_GRAPH,
    REPORT_LABELS,
    REPORT_PATHS,
)
from stoquant.plotting import pyplot
from stoquant.schema import read_snapshot
from stoquant.snapshot_store import derived_directory
from stoquant.stats import (
    compute_stage_medians,
    direction_supported,
    magnitude_supported,
    rev2_supported,
)

# "optimized" is the opt-in AVX2 CPU comparator (issue #22): an F1 line and a
# descriptive F2 line only. It never enters T1, the crossovers or a claim.
PATHS = tuple(path.key for path in REPORT_PATHS)
CUDA_PATHS = tuple(path.key for path in REPORT_PATHS if path.backend == "cuda")
STAGE_PATHS = (CUDA_RESIDENT.key, CUDA_HOST_ORIGIN.key)
LABELS = dict(zip(PATHS, REPORT_LABELS, strict=True))
COLORS = dict(zip(PATHS, ("#555555", "#CC79A7", "#0072B2", "#009E73", "#E69F00"), strict=True))
MARKERS = dict(zip(PATHS, ("s", "v", "o", "D", "^"), strict=True))
STAGE_COUNTS = (1 << 14, 1 << 18, 1 << 22, 1 << 26)
STAGES = (
    ("k1_ms", "scale (K1)", "#0072B2"),
    ("k2_ms", "rounding (K2)", "#009E73"),
    ("k3_ms", "packing (K3)", "#CC79A7"),
    ("h2d_ms", "copy H2D", "#E69F00"),
    ("d2h_ms", "copy D2H", "#F0C050"),
    ("other_ms", "remaining overhead", "#BBBBBB"),
)
FIGURES = {
    "f1": "f1_time_vs_elements.png",
    "f2": "f2_speedup_vs_elements.png",
    "f3": "f3_stage_breakdown.png",
    "f4": "f4_graph_vs_resident.png",
}
REPORT = "report.md"


def index_cases(cases: list[dict[str, Any]]) -> dict[tuple[int, int, str], dict[str, Any]]:
    """Passing revision 3 cases keyed by (count, bits, timing boundary).

    Publication extension paths (gpu-origin, pinned) share a timing boundary with a
    revision 3 path or have none here, so they stay out of F1-F4 and the crossovers.
    Sparse and model-shaped inputs are separate families and stay out as well.
    """
    return {
        (c["count"], c["bits"], c["timing_boundary"]): c
        for c in cases
        if c["correctness"]["status"] == "passed"
        and c["statistics"] is not None
        and c["timing_boundary"] in PATHS
        and c.get("transfer_policy") != "pinned"
        and c.get("input_family", "dense") == "dense"
    }


def stage_medians(case: dict[str, Any]) -> dict[str, float] | None:
    # Snapshots before manifest 2.1 carry per-run stage times but no medians.
    return case.get("stage_medians_ms") or compute_stage_medians(case["trial_runs"])


def speedup_interval(stats: dict[str, Any]) -> tuple[float | None, float | None]:
    """The bootstrap CI, or the trial-median range for snapshots without one."""
    low, high = stats.get("speedup_ci_low"), stats.get("speedup_ci_high")
    if low is not None and high is not None:
        return low, high
    return stats.get("speedup_low"), stats.get("speedup_high")


def has_speedup_data(stats: dict[str, Any], point_key: str = "speedup_vs_cpu") -> bool:
    """Whether a comparison has a point and finite interval to draw or print."""
    point = stats.get(point_key)
    low, high = speedup_interval(stats)
    return all(
        isinstance(value, (int, float)) and math.isfinite(value) for value in (point, low, high)
    )


def plot_interval(ax, x, y, bounds, color, label) -> None:
    """Draw the point line and each interval as a segment from its own bounds.

    The bootstrap CI comes from trial medians and the point from pooled medians,
    so the point may lie outside its CI; error bars relative to it would go negative.
    """
    ax.plot(x, y, color=color, linewidth=1, label=label)
    ax.vlines(x, bounds[:, 0], bounds[:, 1], color=color, linewidth=1)


def power_label(count: int) -> str:
    return f"2^{count.bit_length() - 1}" if count & (count - 1) == 0 else str(count)


def find_crossovers(
    indexed: dict[tuple[int, int, str], dict[str, Any]],
    counts: list[int],
    bits: int,
    path: str,
    supported: Callable[[dict[str, Any]], bool] = direction_supported,
) -> dict[str, Any]:
    """Apply the protocol's crossover rule to one path at one bit width.

    The crossover is the interval between adjacent sizes whose verdicts differ
    (faster vs slower) with the claim ``supported`` on both sides: the
    direction claim for revision 3, the revision 2 claim for comparison.
    """
    points = []
    for count in counts:
        case = indexed.get((count, bits, path))
        if case is None:
            continue
        stats = case["statistics"]
        points.append((count, stats["verdict"], supported(stats)))

    flips = [
        (a[0], b[0], a[1], b[1])
        for a, b in itertools.pairwise(points)
        if a[2] and b[2] and {a[1], b[1]} == {"faster", "slower"}
    ]
    supported_slower = [p[0] for p in points if p[2] and p[1] == "slower"]
    supported_faster = [p[0] for p in points if p[2] and p[1] == "faster"]
    if len(flips) == 1:
        low, high, before, after = flips[0]
        return {
            "status": "resolved",
            "interval": [low, high],
            "direction": f"{before} to {after}",
        }
    return {
        "status": "not resolved",
        "flips": [list(f[:2]) for f in flips],
        "largest_supported_slower": max(supported_slower) if supported_slower else None,
        "smallest_supported_faster": min(supported_faster) if supported_faster else None,
    }


def plot_time(indexed, counts, bit_widths, out: Path) -> None:
    plt = pyplot()
    fig, axes = plt.subplots(
        1, len(bit_widths), figsize=(5.5 * len(bit_widths), 4.2), sharey=True, squeeze=False
    )
    for ax, bits in zip(axes[0], bit_widths, strict=True):
        for path in PATHS:
            rows = [indexed[(n, bits, path)] for n in counts if (n, bits, path) in indexed]
            if not rows:
                continue
            x = np.array([r["count"] for r in rows])
            stats = [r["statistics"] for r in rows]
            ax.fill_between(
                x,
                [s["trial_median_min_ms"] for s in stats],
                [s["trial_median_max_ms"] for s in stats],
                color=COLORS[path],
                alpha=0.25,
                linewidth=0,
            )
            ax.plot(
                x,
                [s["median_ms"] for s in stats],
                marker=MARKERS[path],
                markersize=4,
                color=COLORS[path],
                label=LABELS[path],
            )
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_title(f"{bits}-bit")
        ax.set_xlabel("elements")
        ax.grid(True, which="major", alpha=0.3)
    axes[0][0].set_ylabel("pooled median time (ms)")
    axes[0][0].legend(loc="upper left", fontsize=8)
    fig.suptitle("F1. Time vs elements (band: range of trial medians)")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_descriptive_speedup(ax, indexed, counts, bits) -> None:
    """The AVX2 comparator as a dashed CI line with no claim markers."""
    path = AVX2_COMPARATOR.key
    rows = [indexed[(n, bits, path)] for n in counts if (n, bits, path) in indexed]
    rows = [r for r in rows if has_speedup_data(r["statistics"])]
    if not rows:
        return
    x = np.array([r["count"] for r in rows])
    y = np.array([r["statistics"]["speedup_vs_cpu"] for r in rows])
    bounds = np.array([speedup_interval(r["statistics"]) for r in rows])
    ax.plot(x, y, color=COLORS[path], linewidth=1, linestyle="--", label=LABELS[path])
    ax.vlines(x, bounds[:, 0], bounds[:, 1], color=COLORS[path], linewidth=1)


def plot_speedup(indexed, counts, bit_widths, out: Path) -> None:
    plt = pyplot()
    fig, axes = plt.subplots(
        1, len(bit_widths), figsize=(5.5 * len(bit_widths), 4.2), sharey=True, squeeze=False
    )
    for ax, bits in zip(axes[0], bit_widths, strict=True):
        ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
        plot_descriptive_speedup(ax, indexed, counts, bits)
        for path in CUDA_PATHS:
            rows = [indexed[(n, bits, path)] for n in counts if (n, bits, path) in indexed]
            rows = [r for r in rows if has_speedup_data(r["statistics"])]
            if not rows:
                continue
            x = np.array([r["count"] for r in rows])
            stats = [r["statistics"] for r in rows]
            y = np.array([s["speedup_vs_cpu"] for s in stats])
            bounds = np.array([speedup_interval(s) for s in stats])
            plot_interval(ax, x, y, bounds, COLORS[path], LABELS[path])
            direction = np.array([direction_supported(s) for s in stats])
            magnitude = np.array([bool(magnitude_supported(s)) for s in stats])
            ax.scatter(
                x[magnitude], y[magnitude], marker=MARKERS[path], color=COLORS[path], zorder=3
            )
            only = direction & ~magnitude
            ax.scatter(
                x[only],
                y[only],
                marker=MARKERS[path],
                facecolors="white",
                edgecolors=COLORS[path],
                zorder=3,
            )
            ax.scatter(
                x[~direction],
                y[~direction],
                marker=MARKERS[path],
                facecolors="white",
                edgecolors=COLORS[path],
                alpha=0.35,
                zorder=3,
            )
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_title(f"{bits}-bit")
        ax.set_xlabel("elements")
        ax.grid(True, which="major", alpha=0.3)
    axes[0][0].set_ylabel("speedup vs C comparator")
    axes[0][0].legend(loc="upper left", fontsize=8)
    fig.suptitle(
        "F2. Speedup vs elements, 95% bootstrap CI "
        "(filled: magnitude; hollow: direction only; faded: neither; dashed: descriptive)"
    )
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_stages(indexed, stage_counts, bit_widths, out: Path) -> None:
    plt = pyplot()
    fig, axes = plt.subplots(
        1, len(bit_widths), figsize=(5.5 * len(bit_widths), 4.4), sharey=True, squeeze=False
    )
    width = 0.38
    for ax, bits in zip(axes[0], bit_widths, strict=True):
        for offset, path in zip((-width / 2, width / 2), STAGE_PATHS, strict=True):
            for i, count in enumerate(stage_counts):
                case = indexed.get((count, bits, path))
                if case is None:
                    continue
                medians = stage_medians(case)
                if not medians:
                    continue
                parts = {key: max(medians.get(key, 0.0), 0.0) for key, _, _ in STAGES}
                total = sum(parts.values())
                bottom = 0.0
                for key, label, color in STAGES:
                    share = 100.0 * parts[key] / total if total > 0 else 0.0
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
                ax.text(
                    i + offset,
                    101,
                    f"{'R' if path == 'resident' else 'H'}\n{case['statistics']['median_ms']:.3g}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )
        ax.set_xticks(range(len(stage_counts)), [power_label(n) for n in stage_counts])
        ax.set_ylim(0, 118)
        ax.set_title(f"{bits}-bit")
        ax.set_xlabel("elements (R: resident, H: host-origin; label: median ms)")
    axes[0][0].set_ylabel("share of stage medians (%)")
    handles, labels = [], []
    for ax in axes[0]:
        for handle, label in zip(*ax.get_legend_handles_labels(), strict=True):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=8)
    else:
        fig.text(0.5, 0.5, "no stage timings in this snapshot", ha="center")
    fig.suptitle("F3. CUDA stage breakdown")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_graph_vs_resident(indexed, counts, bit_widths, out: Path) -> None:
    plt = pyplot()
    fig, axes = plt.subplots(
        1, len(bit_widths), figsize=(5.5 * len(bit_widths), 4.2), sharey=True, squeeze=False
    )
    has_any = False
    for ax, bits in zip(axes[0], bit_widths, strict=True):
        ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
        rows = [
            indexed[(n, bits, CUDA_RESIDENT_GRAPH.key)]
            for n in counts
            if (n, bits, CUDA_RESIDENT_GRAPH.key) in indexed
        ]
        rows = [
            r
            for r in rows
            if has_speedup_data(
                ((r.get("statistics") or {}).get("vs_resident") or {}),
                "speedup_vs_resident",
            )
        ]
        if not rows:
            ax.set_xscale("log", base=2)
            ax.set_title(f"{bits}-bit")
            ax.set_xlabel("elements")
            ax.grid(True, which="major", alpha=0.3)
            continue
        has_any = True
        x = np.array([r["count"] for r in rows])
        vs_res_list = [r["statistics"]["vs_resident"] for r in rows]
        y = np.array([v["speedup_vs_resident"] for v in vs_res_list])
        bounds = np.array(
            [
                (
                    (v["speedup_ci_low"], v["speedup_ci_high"])
                    if "speedup_ci_low" in v
                    else (v["speedup_low"], v["speedup_high"])
                )
                for v in vs_res_list
            ]
        )
        plot_interval(
            ax, x, y, bounds, COLORS[CUDA_RESIDENT_GRAPH.key], LABELS[CUDA_RESIDENT_GRAPH.key]
        )
        direction = np.array([direction_supported(v) for v in vs_res_list])
        magnitude = np.array([bool(magnitude_supported(v)) for v in vs_res_list])
        ax.scatter(
            x[magnitude],
            y[magnitude],
            marker=MARKERS[CUDA_RESIDENT_GRAPH.key],
            color=COLORS[CUDA_RESIDENT_GRAPH.key],
            zorder=3,
        )
        only = direction & ~magnitude
        ax.scatter(
            x[only],
            y[only],
            marker=MARKERS[CUDA_RESIDENT_GRAPH.key],
            facecolors="white",
            edgecolors=COLORS[CUDA_RESIDENT_GRAPH.key],
            zorder=3,
        )
        ax.scatter(
            x[~direction],
            y[~direction],
            marker=MARKERS[CUDA_RESIDENT_GRAPH.key],
            facecolors="white",
            edgecolors=COLORS[CUDA_RESIDENT_GRAPH.key],
            alpha=0.35,
            zorder=3,
        )
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_title(f"{bits}-bit")
        ax.set_xlabel("elements")
        ax.grid(True, which="major", alpha=0.3)
    axes[0][0].set_ylabel("speedup vs plain resident")
    if has_any:
        axes[0][0].legend(loc="upper left", fontsize=8)
    else:
        fig.text(0.5, 0.5, "no resident-graph comparison in this snapshot", ha="center")
    fig.suptitle(
        "F4. CUDA Graph speedup vs plain resident, 95% bootstrap CI "
        "(filled: magnitude; hollow: direction only; faded: neither)"
    )
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def fmt_ms(value: float) -> str:
    return f"{value:.4g}"


def yes_no(value: bool | None) -> str:
    return "—" if value is None else ("yes" if value else "no")


def t1_table(
    indexed, counts, bit_widths, cuda_paths: tuple[str, ...] | list[str] = CUDA_PATHS
) -> list[str]:
    lines = [
        (
            "| Elements | Bits | Path | C (ms) | CUDA (ms) | Speedup | Trial range | 95% CI | "
            "Verdict | Direction | Magnitude | Rev 2 |"
        ),
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for count in counts:
        for bits in bit_widths:
            cpu = indexed.get((count, bits, COMPARATOR.key))
            c_ms = fmt_ms(cpu["statistics"]["median_ms"]) if cpu else "failed"
            for path in cuda_paths:
                cells = [power_label(count), str(bits), LABELS[path], c_ms]
                case = indexed.get((count, bits, path))
                if case is None or "verdict" not in case["statistics"]:
                    cells += ["failed", *["—"] * 7]
                    lines.append("| " + " | ".join(cells) + " |")
                    continue
                stats = case["statistics"]
                speedup = stats.get("speedup_vs_cpu")
                speedup_text = (
                    f"{speedup:.3g}×"
                    if isinstance(speedup, (int, float)) and math.isfinite(speedup)
                    else "—"
                )
                low, high = stats.get("speedup_low"), stats.get("speedup_high")
                trial_range = (
                    f"[{low:.3g}, {high:.3g}]"
                    if isinstance(low, (int, float))
                    and isinstance(high, (int, float))
                    and math.isfinite(low)
                    and math.isfinite(high)
                    else "—"
                )
                ci_low, ci_high = stats.get("speedup_ci_low"), stats.get("speedup_ci_high")
                ci = (
                    f"[{ci_low:.3g}, {ci_high:.3g}]"
                    if isinstance(ci_low, (int, float))
                    and isinstance(ci_high, (int, float))
                    and math.isfinite(ci_low)
                    and math.isfinite(ci_high)
                    else "—"
                )
                cells += [
                    fmt_ms(stats["median_ms"]),
                    speedup_text,
                    trial_range,
                    ci,
                    stats["verdict"],
                    yes_no(direction_supported(stats)),
                    yes_no(magnitude_supported(stats)),
                    yes_no(rev2_supported(stats)),
                ]
                lines.append("| " + " | ".join(cells) + " |")
    return lines


def crossover_lines(crossovers: dict[tuple[int, str], dict[str, Any]]) -> list[str]:
    lines = []
    for (bits, path), result in crossovers.items():
        name = f"{LABELS[path]}, {bits}-bit"
        if result["status"] == "resolved":
            low, high = result["interval"]
            lines.append(
                f"- {name}: resolved between {power_label(low)} and {power_label(high)} "
                f"elements ({result['direction']})."
            )
            continue
        slower = result["largest_supported_slower"]
        faster = result["smallest_supported_faster"]
        detail = []
        if result["flips"]:
            detail.append(f"{len(result['flips'])} supported flips")
        detail.append(
            "largest supported slower size: " + (power_label(slower) if slower else "none")
        )
        detail.append(
            "smallest supported faster size: " + (power_label(faster) if faster else "none")
        )
        lines.append(f"- {name}: not resolved ({'; '.join(detail)}).")
    return lines


def render_report(
    snapshot: Path,
    output_dir: Path | None = None,
    stage_counts=STAGE_COUNTS,
    *,
    k1_ab: Path | None = None,
    second_platform: Path | None = None,
) -> dict[str, Any]:
    typed_manifest, typed_cases = read_snapshot(snapshot)
    manifest = cast(dict[str, Any], typed_manifest)
    cases = [cast(dict[str, Any], case) for case in typed_cases]
    if any(
        case.get("input_family", "dense") != "dense"
        or case.get("timing_boundary") == CUDA_GPU_ORIGIN.boundary
        or case.get("transfer_policy") == "pinned"
        for case in cases
    ):
        from stoquant.publication import render_publication_report

        return render_publication_report(
            snapshot, manifest, cases, output_dir, k1_ab, second_platform
        )
    out = output_dir if output_dir is not None else derived_directory(snapshot)
    out.mkdir(parents=True, exist_ok=True)
    indexed = index_cases(cases)
    counts = sorted({c["count"] for c in cases})
    bit_widths = sorted({c["bits"] for c in cases})
    chosen = [n for n in stage_counts if n in counts] or counts

    plot_time(indexed, counts, bit_widths, out / FIGURES["f1"])
    plot_speedup(indexed, counts, bit_widths, out / FIGURES["f2"])
    plot_stages(indexed, chosen, bit_widths, out / FIGURES["f3"])
    plot_graph_vs_resident(indexed, counts, bit_widths, out / FIGURES["f4"])

    present_cuda_paths: list[str] = [
        p for p in CUDA_PATHS if any(c.get("timing_boundary") == p for c in cases)
    ]
    if not present_cuda_paths:
        present_cuda_paths = list(CUDA_PATHS)

    crossovers = {
        (bits, path): find_crossovers(indexed, counts, bits, path)
        for bits in bit_widths
        for path in present_cuda_paths
    }
    crossovers_rev2 = {
        (bits, path): find_crossovers(indexed, counts, bits, path, rev2_supported)
        for bits in bit_widths
        for path in present_cuda_paths
    }
    revision = manifest["git_provenance"]["code_revision_short"]
    report = [
        f"# Size sweep report ({snapshot.name})",
        "",
        (
            f"Generated by `bench_report.py` from snapshot revision `{revision}`, "
            f"{len(cases)} cases, all passed: {manifest['all_cases_passed']}."
        ),
        "",
        "## Crossover",
        "",
        "Located from the direction claim on both sides of the flip.",
        "",
        *crossover_lines(crossovers),
        "",
        "### Revision 2 crossover",
        "",
        "Located from the unchanged revision 2 claim, for comparison.",
        "",
        *crossover_lines(crossovers_rev2),
        "",
        "## T1. Summary",
        "",
        (
            "Times are pooled medians. Speedup is C median over CUDA median. The trial "
            "range decides the verdict; the 95% CI is the bootstrap interval over trial "
            "medians, so a speedup can lie outside its CI. Direction, magnitude and "
            "revision 2 are the claim rules recorded in "
            "the manifest; — marks a field the snapshot predates."
        ),
        "",
        *t1_table(indexed, counts, bit_widths, cuda_paths=present_cuda_paths),
        "",
        "## Figures",
        "",
        *[f"- ![{key.upper()}]({name})" for key, name in FIGURES.items()],
        "",
    ]
    (out / REPORT).write_text("\n".join(report), encoding="utf-8", newline="\n")
    return {
        "figures": [out / name for name in FIGURES.values()],
        "report": out / REPORT,
        "crossovers": crossovers,
        "crossovers_rev2": crossovers_rev2,
    }


def main(argv: list[str] | None = None, prog: str | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog=prog, description="Render benchmark snapshot figures and report"
    )
    parser.add_argument("snapshot", type=Path, help="Snapshot directory under results/")
    parser.add_argument(
        "--output-dir", type=Path, default=None, help="Write here instead of the snapshot"
    )
    parser.add_argument("--k1-ab", type=Path, help="K1 A/B snapshot for publication bandwidth")
    parser.add_argument("--second-platform", type=Path, help="optional second-platform snapshot")
    args = parser.parse_args(argv)
    result = render_report(
        args.snapshot,
        args.output_dir,
        k1_ab=args.k1_ab,
        second_platform=args.second_platform,
    )
    for path in [*result["figures"], result["report"]]:
        print(path)


if __name__ == "__main__":
    main()
