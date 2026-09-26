"""Render the size-sweep figures, the T1 summary table and the crossover report.

Reads a frozen snapshot (manifest.json plus one JSON per case) and writes
F1-F3 as PNG files and ``report.md`` into the output directory, which defaults
to the snapshot itself. Nothing in the snapshot is rewritten.
"""

from __future__ import annotations

import argparse
import itertools
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from benchmark_driver import compute_stage_medians

PATHS = ("comparator", "resident", "resident-graph", "host-origin")
CUDA_PATHS = ("resident", "resident-graph", "host-origin")
STAGE_PATHS = ("resident", "host-origin")
LABELS = {
    "comparator": "C comparator",
    "resident": "CUDA resident",
    "resident-graph": "CUDA resident-graph",
    "host-origin": "CUDA host-origin",
}
COLORS = {
    "comparator": "#555555",
    "resident": "#0072B2",
    "resident-graph": "#009E73",
    "host-origin": "#E69F00",
}
MARKERS = {"comparator": "s", "resident": "o", "resident-graph": "D", "host-origin": "^"}
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


def load_snapshot(snapshot: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = json.loads((snapshot / "manifest.json").read_text(encoding="utf-8"))
    cases = [
        json.loads((snapshot / f"{case_id}.json").read_text(encoding="utf-8"))
        for case_id in manifest["cases"]
    ]
    return manifest, cases


def index_cases(cases: list[dict[str, Any]]) -> dict[tuple[int, int, str], dict[str, Any]]:
    """Passing cases keyed by (count, bits, timing boundary)."""
    return {
        (c["count"], c["bits"], c["timing_boundary"]): c
        for c in cases
        if c["correctness"]["status"] == "passed" and c["statistics"] is not None
    }


def stage_medians(case: dict[str, Any]) -> dict[str, float] | None:
    # Snapshots before manifest 2.1 carry per-run stage times but no medians.
    return case.get("stage_medians_ms") or compute_stage_medians(case["trial_runs"])


def direction_supported(stats: dict[str, Any]) -> bool:
    # Revision 2 snapshots predate the field; the rule is recomputable from them.
    if "direction_supported" in stats:
        return bool(stats["direction_supported"])
    return stats["verdict"] in ("faster", "slower") and not stats.get("boundary_inversion")


def magnitude_supported(stats: dict[str, Any]) -> bool | None:
    return stats.get("magnitude_supported")


def rev2_supported(stats: dict[str, Any]) -> bool:
    return bool(stats.get("claim_supported_rev2", stats.get("claim_supported")))


def speedup_interval(stats: dict[str, Any]) -> tuple[float, float]:
    """The bootstrap CI, or the trial-median range for snapshots without one."""
    if "speedup_ci_low" in stats:
        return stats["speedup_ci_low"], stats["speedup_ci_high"]
    return stats["speedup_low"], stats["speedup_high"]


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


def plot_speedup(indexed, counts, bit_widths, out: Path) -> None:
    fig, axes = plt.subplots(
        1, len(bit_widths), figsize=(5.5 * len(bit_widths), 4.2), sharey=True, squeeze=False
    )
    for ax, bits in zip(axes[0], bit_widths, strict=True):
        ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
        for path in CUDA_PATHS:
            rows = [indexed[(n, bits, path)] for n in counts if (n, bits, path) in indexed]
            rows = [r for r in rows if "speedup_vs_cpu" in r["statistics"]]
            if not rows:
                continue
            x = np.array([r["count"] for r in rows])
            stats = [r["statistics"] for r in rows]
            y = np.array([s["speedup_vs_cpu"] for s in stats])
            bounds = np.array([speedup_interval(s) for s in stats])
            err = np.array([y - bounds[:, 0], bounds[:, 1] - y])
            ax.errorbar(
                x, y, yerr=err, color=COLORS[path], linewidth=1, capsize=2, label=LABELS[path]
            )
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
        "(filled: magnitude; hollow: direction only; faded: neither)"
    )
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_stages(indexed, stage_counts, bit_widths, out: Path) -> None:
    fig, axes = plt.subplots(
        1, len(bit_widths), figsize=(5.5 * len(bit_widths), 4.4), sharey=True, squeeze=False
    )
    width = 0.38
    for ax, bits in zip(axes[0], bit_widths, strict=True):
        for offset, path in zip((-width / 2, width / 2), STAGE_PATHS, strict=True):
            for i, count in enumerate(stage_counts):
                case = indexed.get((count, bits, path))
                medians = stage_medians(case) if case is not None else None
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
    fig, axes = plt.subplots(
        1, len(bit_widths), figsize=(5.5 * len(bit_widths), 4.2), sharey=True, squeeze=False
    )
    has_any = False
    for ax, bits in zip(axes[0], bit_widths, strict=True):
        ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
        rows = [
            indexed[(n, bits, "resident-graph")]
            for n in counts
            if (n, bits, "resident-graph") in indexed
        ]
        rows = [
            r
            for r in rows
            if (r.get("vs_resident") and "speedup_vs_resident" in r["vs_resident"])
            or (
                r.get("statistics")
                and r["statistics"].get("vs_resident")
                and "speedup_vs_resident" in r["statistics"]["vs_resident"]
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
        vs_res_list = [r.get("vs_resident") or r["statistics"]["vs_resident"] for r in rows]
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
        err = np.array([y - bounds[:, 0], bounds[:, 1] - y])
        ax.errorbar(
            x,
            y,
            yerr=err,
            color=COLORS["resident-graph"],
            linewidth=1,
            capsize=2,
            label=LABELS["resident-graph"],
        )
        direction = np.array([direction_supported(v) for v in vs_res_list])
        magnitude = np.array([bool(magnitude_supported(v)) for v in vs_res_list])
        ax.scatter(
            x[magnitude],
            y[magnitude],
            marker=MARKERS["resident-graph"],
            color=COLORS["resident-graph"],
            zorder=3,
        )
        only = direction & ~magnitude
        ax.scatter(
            x[only],
            y[only],
            marker=MARKERS["resident-graph"],
            facecolors="white",
            edgecolors=COLORS["resident-graph"],
            zorder=3,
        )
        ax.scatter(
            x[~direction],
            y[~direction],
            marker=MARKERS["resident-graph"],
            facecolors="white",
            edgecolors=COLORS["resident-graph"],
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
            cpu = indexed.get((count, bits, "comparator"))
            c_ms = fmt_ms(cpu["statistics"]["median_ms"]) if cpu else "failed"
            for path in cuda_paths:
                cells = [power_label(count), str(bits), LABELS[path], c_ms]
                case = indexed.get((count, bits, path))
                if case is None or "verdict" not in case["statistics"]:
                    cells += ["failed", *["—"] * 7]
                    lines.append("| " + " | ".join(cells) + " |")
                    continue
                stats = case["statistics"]
                ci = (
                    f"[{stats['speedup_ci_low']:.3g}, {stats['speedup_ci_high']:.3g}]"
                    if "speedup_ci_low" in stats
                    else "—"
                )
                cells += [
                    fmt_ms(stats["median_ms"]),
                    f"{stats['speedup_vs_cpu']:.3g}×",
                    f"[{stats['speedup_low']:.3g}, {stats['speedup_high']:.3g}]",
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
    snapshot: Path, output_dir: Path | None = None, stage_counts=STAGE_COUNTS
) -> dict[str, Any]:
    manifest, cases = load_snapshot(snapshot)
    out = output_dir or snapshot
    out.mkdir(parents=True, exist_ok=True)
    indexed = index_cases(cases)
    counts = sorted({c["count"] for c in cases})
    bit_widths = sorted({c["bits"] for c in cases})
    chosen = [n for n in stage_counts if n in counts] or counts

    plot_time(indexed, counts, bit_widths, out / FIGURES["f1"])
    plot_speedup(indexed, counts, bit_widths, out / FIGURES["f2"])
    plot_stages(indexed, chosen, bit_widths, out / FIGURES["f3"])
    plot_graph_vs_resident(indexed, counts, bit_widths, out / FIGURES["f4"])

    present_cuda_paths = [
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
            "medians. Direction, magnitude and revision 2 are the claim rules recorded in "
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
    (out / REPORT).write_text("\n".join(report), encoding="utf-8")
    return {
        "figures": [out / name for name in FIGURES.values()],
        "report": out / REPORT,
        "crossovers": crossovers,
        "crossovers_rev2": crossovers_rev2,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Render F1-F3, T1 and the crossover report")
    parser.add_argument("snapshot", type=Path, help="Snapshot directory under results/")
    parser.add_argument(
        "--output-dir", type=Path, default=None, help="Write here instead of the snapshot"
    )
    args = parser.parse_args()
    result = render_report(args.snapshot, args.output_dir)
    for path in [*result["figures"], result["report"]]:
        print(path)


if __name__ == "__main__":
    main()
