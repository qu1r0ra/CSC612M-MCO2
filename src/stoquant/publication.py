"""Render the opt-in publication matrix without changing revision 3 reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from stoquant.plotting import pyplot
from stoquant.report import (
    FIGURES,
    REPORT,
    STAGES,
    direction_supported,
    find_crossovers,
    load_snapshot,
    magnitude_supported,
    plot_graph_vs_resident,
    power_label,
    speedup_interval,
    stage_medians,
)

PUBLICATION_FIGURES = {**FIGURES, "bandwidth": "appendix_bandwidth.png"}
PATH_LABELS = {
    "cpu-comparator": "C comparator",
    "cpu-avx2-optimized": "AVX2 comparator (descriptive)",
    "cuda-resident": "CUDA resident",
    "cuda-resident-graph": "CUDA resident-graph",
    "cuda-host-origin": "CUDA host-origin pageable",
    "cuda-host-origin-pinned": "CUDA host-origin pinned",
    "cpu-gpu-origin": "CPU GPU-origin pageable",
    "cuda-gpu-origin": "CUDA GPU-origin pageable",
    "cpu-gpu-origin-pinned": "CPU GPU-origin pinned",
    "cuda-gpu-origin-pinned": "CUDA GPU-origin pinned",
}


def _path_colors(plt) -> dict[str, Any]:
    # The stubs type tab10 as a plain Colormap; it is a ListedColormap.
    colors: list[Any] = plt.cm.tab10.colors  # pyright: ignore[reportAttributeAccessIssue]
    return dict(zip(PATH_LABELS, colors, strict=True))


TRANSFER_PAIRS = (
    ("cuda-host-origin", "cuda-host-origin-pinned"),
    ("cpu-gpu-origin", "cpu-gpu-origin-pinned"),
    ("cuda-gpu-origin", "cuda-gpu-origin-pinned"),
)


def _usable(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        case
        for case in cases
        if case.get("correctness", {}).get("status") == "passed"
        and case.get("statistics") is not None
    ]


def _path(case: dict[str, Any]) -> str:
    if "path_label" in case:
        return case["path_label"]
    suffix = "-pinned" if case.get("transfer_policy") == "pinned" else ""
    return f"{case['backend']}-{case['timing_boundary']}{suffix}"


def _key(case: dict[str, Any]) -> str:
    return case.get("input_key", f"n{case['count']}")


def _rows(cases: list[dict[str, Any]], path: str, bits: int) -> list[dict[str, Any]]:
    return sorted(
        [case for case in cases if _path(case) == path and case["bits"] == bits],
        key=lambda case: (case["count"], _key(case)),
    )


def _mark_claims(ax, x, y, stats, color: str, marker: str = "o") -> None:
    direction = np.array([direction_supported(item) for item in stats])
    magnitude = np.array([bool(magnitude_supported(item)) for item in stats])
    ax.scatter(x[magnitude], y[magnitude], marker=marker, color=color, zorder=3)
    only = direction & ~magnitude
    ax.scatter(x[only], y[only], marker=marker, facecolors="white", edgecolors=color, zorder=3)
    neither = ~direction
    ax.scatter(
        x[neither],
        y[neither],
        marker=marker,
        facecolors="white",
        edgecolors=color,
        alpha=0.35,
        zorder=3,
    )


def plot_publication_time(
    cases: list[dict[str, Any]], out: Path, second: list[dict[str, Any]] | None
) -> None:
    panels = [("primary platform", cases), ("second platform", second)]
    plt = pyplot()
    colors = _path_colors(plt)
    fig, axes = plt.subplots(
        len(panels), 2, figsize=(13, 4.6 * len(panels)), squeeze=False, sharey="row"
    )
    for row, (platform, collection) in enumerate(panels):
        for column, bits in enumerate((4, 8)):
            ax = axes[row, column]
            if collection is None:
                ax.set_axis_off()
                ax.text(
                    0.5,
                    0.5,
                    "Second-platform matrix pending",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                continue
            for path, label in PATH_LABELS.items():
                selected = _rows(collection, path, bits)
                if not selected:
                    continue
                x = np.array([case["count"] for case in selected])
                stats = [case["statistics"] for case in selected]
                color = colors[path]
                ax.fill_between(
                    x,
                    [item["trial_median_min_ms"] for item in stats],
                    [item["trial_median_max_ms"] for item in stats],
                    color=color,
                    alpha=0.15,
                )
                ax.plot(x, [item["median_ms"] for item in stats], color=color, label=label)
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            ax.set_title(f"{platform}, {bits}-bit")
            ax.set_xlabel("elements")
            ax.grid(True, alpha=0.25)
        axes[row, 0].set_ylabel("pooled median time (ms)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=7)
    fig.suptitle("F1. Publication matrix time and trial-median range")
    fig.tight_layout(rect=(0, 0.1, 1, 0.96))
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_publication_speedup(cases: list[dict[str, Any]], out: Path) -> None:
    plt = pyplot()
    colors = _path_colors(plt)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, bits in zip(axes, (4, 8), strict=True):
        ax.axhline(1, color="black", linewidth=0.8, linestyle="--")
        for path, label in PATH_LABELS.items():
            if path == "cpu-comparator":
                continue
            selected = _rows(cases, path, bits)
            if not selected:
                continue
            stats = [case["statistics"] for case in selected]
            if not all("speedup_vs_cpu" in item for item in stats):
                continue
            x = np.array([case["count"] for case in selected])
            y = np.array([item["speedup_vs_cpu"] for item in stats])
            bounds = np.array([speedup_interval(item) for item in stats])
            color = colors[path]
            descriptive = path == "cpu-avx2-optimized"
            ax.plot(x, y, color=color, linestyle="--" if descriptive else "-", label=label)
            ax.vlines(x, bounds[:, 0], bounds[:, 1], color=color, linewidth=0.8)
            if not descriptive:
                _mark_claims(ax, x, y, stats, color)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_title(f"{bits}-bit")
        ax.set_xlabel("elements")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("speedup vs recorded baseline")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=7)
    fig.suptitle("F2. Publication matrix speedup and 95% bootstrap CI")
    fig.tight_layout(rect=(0, 0.1, 1, 0.96))
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_publication_stages(cases: list[dict[str, Any]], out: Path) -> None:
    plt = pyplot()
    paths = [
        path for path in PATH_LABELS if path.startswith("cuda-") and path != "cuda-resident-graph"
    ]
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    for ax, bits in zip(axes, (4, 8), strict=True):
        selected = [
            case
            for case in cases
            if case["bits"] == bits and _path(case) in paths and stage_medians(case)
        ]
        counts = sorted({case["count"] for case in selected})
        chosen = [n for n in (1 << 14, 1 << 18, 1 << 22, 1 << 26) if n in counts]
        if not chosen:
            chosen = counts[:4]
        width = 0.8 / len(paths)
        for index, path in enumerate(paths):
            for position, count in enumerate(chosen):
                case = next(
                    (item for item in selected if item["count"] == count and _path(item) == path),
                    None,
                )
                if case is None:
                    continue
                stages = stage_medians(case) or {}
                total = sum(max(stages.get(name, 0), 0) for name, _, _ in STAGES)
                bottom = 0.0
                for name, label, color in STAGES:
                    share = 100 * max(stages.get(name, 0), 0) / total if total else 0
                    ax.bar(
                        position - 0.4 + (index + 0.5) * width,
                        share,
                        width,
                        bottom=bottom,
                        color=color,
                        label=label,
                    )
                    bottom += share
        ax.set_xticks(range(len(chosen)), [power_label(n) for n in chosen])
        ax.set_title(f"{bits}-bit")
        ax.set_xlabel("elements")
    axes[0].set_ylabel("share of stage medians (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        unique = dict(zip(labels, handles, strict=True))
        fig.legend(
            unique.values(),
            unique.keys(),
            loc="lower center",
            bbox_to_anchor=(0.5, 0.07),
            ncol=3,
            fontsize=7,
        )
    fig.text(
        0.5,
        0.01,
        "Within each count, left to right: " + ", ".join(PATH_LABELS[path] for path in paths),
        ha="center",
        fontsize=7,
    )
    fig.suptitle("F3. CUDA stages, including GPU-origin transfer")
    fig.tight_layout(rect=(0, 0.18, 1, 0.96))
    fig.savefig(out, dpi=200)
    plt.close(fig)


def _stage_ms(case: dict[str, Any], stage: str) -> float:
    if stage == "pipeline":
        return case["statistics"]["median_ms"]
    medians = stage_medians(case)
    if medians is None:
        raise ValueError(f"resident case n={case['count']} lacks stage times")
    return medians["k1_ms"]


def plot_bandwidth(
    manifest: dict[str, Any], cases: list[dict[str, Any]], k1_ab: Path, out: Path
) -> dict[str, float]:
    summary = json.loads((k1_ab / "summary.json").read_text(encoding="utf-8"))
    if not summary.get("evidence") or summary.get("git", {}).get("git_dirty"):
        raise ValueError("K1 A/B bandwidth source must be clean evidence")
    device = manifest.get("device")
    if not device:
        raise ValueError("publication matrix manifest lacks device attributes")
    probe = json.loads((k1_ab / "stream_probe.json").read_text(encoding="utf-8"))
    attributes = ("name", "memory_clock_khz", "bus_width_bits", "l2_bytes")
    if any(device[name] != probe["device"][name] for name in attributes):
        raise ValueError("matrix device and K1 A/B probe attributes differ")
    l2 = int(device["l2_bytes"])
    peak = 2 * int(device["memory_clock_khz"]) * 1e3 * int(device["bus_width_bits"]) / 8 / 1e9
    dram_probe = [size for size in probe["sizes"] if size["bytes"] >= 4 * l2]
    if not dram_probe:
        raise ValueError("K1 A/B probe has no DRAM-sized measurement")
    ceiling = max(float(size["best_gbps"]) for size in dram_probe)
    plt = pyplot()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    resident = [case for case in cases if _path(case) == "cuda-resident"]
    if not resident:
        raise ValueError("publication matrix lacks CUDA resident cases")
    for bits in (4, 8):
        selected = sorted((c for c in resident if c["bits"] == bits), key=lambda c: c["count"])
        for stage, style in (("k1", "-"), ("pipeline", "--")):
            ax.plot(
                [c["count"] for c in selected],
                [8 * c["count"] / (_stage_ms(c, stage) * 1e6) for c in selected],
                linestyle=style,
                label=f"{bits}-bit {stage}",
            )
    minimum = min(case["count"] for case in resident)
    maximum = max(case["count"] for case in resident)
    for start, end, label, color in (
        (minimum, l2 / 4, "l2-resident", "#dbe9f6"),
        (l2 / 4, l2, "transitional", "#fcecc9"),
        (l2, maximum, "dram", "#e0f0d9"),
    ):
        if end > start:
            ax.axvspan(start, end, color=color, alpha=0.35, label=label)
    ax.axhline(peak, color="black", linestyle=":", label=f"device peak {peak:.1f} GB/s")
    ax.axhline(ceiling, color="gray", linestyle="-.", label=f"probe ceiling {ceiling:.1f} GB/s")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("elements")
    ax.set_ylabel("effective two-read bandwidth (GB/s)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return {"peak_gbps": peak, "ceiling_gbps": ceiling}


def _comparison(item: dict[str, Any]) -> str:
    if not item:
        return "—"
    speedup = item.get("speedup_vs_pageable", item.get("speedup_vs_cpu"))
    if speedup is None:
        return "—"
    ci = speedup_interval(item)
    return (
        f"{speedup:.3g}× [{ci[0]:.3g}, {ci[1]:.3g}]; "
        f"{item['verdict']}; direction={direction_supported(item)}; "
        f"magnitude={bool(magnitude_supported(item))}"
    )


def _report_lines(cases: list[dict[str, Any]], manifest: dict[str, Any]) -> list[str]:
    family = manifest.get("matrix_parameters", {}).get("input_family") or cases[0].get(
        "input_family", "dense"
    )
    lines = [
        f"# Publication matrix report ({family})",
        "",
        (
            f"Revision `{manifest['git_provenance']['code_revision_short']}`; {len(cases)} "
            f"cases; all passed: {manifest['all_cases_passed']}."
        ),
        "",
        "## Input family",
        "",
        "| Family | Inputs | Cells | Distinct counts |",
        "|---|---:|---:|---:|",
        (
            f"| {family} | {len({_key(c) for c in cases})} | {len(cases)} | "
            f"{len({c['count'] for c in cases})} |"
        ),
        "",
        "### Input provenance",
        "",
        "| Input | Elements | Source detail |",
        "|---|---:|---|",
    ]
    by_input = {_key(case): case for case in cases}
    for key, case in sorted(by_input.items()):
        provenance = case.get("input_provenance", {})
        if family == "sparse":
            detail = (
                f"zero target {provenance.get('zero_fraction_target', '—')}; "
                f"realised {provenance.get('zero_fraction_realised', '—')}"
            )
        elif family == "model":
            detail = (
                f"{provenance.get('tensor_name', key)}; "
                f"shape {provenance.get('tensor_shape', '—')}; "
                f"{provenance.get('tensor_kind', '—')}"
            )
        else:
            detail = provenance.get("generator", "dense input")
        lines.append(f"| {key} | {case['count']} | {detail} |")
    lines += [
        "",
        "## Path comparisons",
        "",
        (
            "GPU-origin CUDA uses the policy-matched CPU GPU-origin baseline. Other CUDA "
            "paths use the scalar C comparator. AVX2 and cross-boundary GPU-origin "
            "comparisons are descriptive."
        ),
        "",
        "| Input | Elements | Bits | Path | Policy | Baseline | Median ms | Speedup and CI | Inversion |",
        "|---|---:|---:|---|---|---|---:|---|---|",
    ]
    for case in sorted(cases, key=lambda c: (_key(c), c["bits"], _path(c))):
        stats = case["statistics"]
        lines.append(
            f"| {_key(case)} | {case['count']} | {case['bits']} | {_path(case)} | "
            f"{case.get('transfer_policy', 'none')} | {stats.get('baseline', '—') or '—'} | "
            f"{stats['median_ms']:.4g} | {_comparison(stats)} | "
            f"{bool(stats.get('boundary_inversion'))} |"
        )
    lines += [
        "",
        "## Transfer policy",
        "",
        "Pinned is compared with its pageable twin under the recorded claim rule.",
        "",
        "| Input | Bits | Path | Pinned vs pageable |",
        "|---|---:|---|---|",
    ]
    by_key = {(_key(case), case["bits"], _path(case)): case for case in cases}
    for key, bits in sorted({(_key(case), case["bits"]) for case in cases}):
        for _, pinned in TRANSFER_PAIRS:
            case = by_key.get((key, bits, pinned))
            if case:
                lines.append(
                    f"| {key} | {bits} | {pinned} | "
                    f"{_comparison(case['statistics'].get('vs_pageable', {}))} |"
                )
    lines += ["", "## Boundary inversions", ""]
    inverted = [case for case in cases if case["statistics"].get("boundary_inversion")]
    if inverted:
        lines.extend(
            f"- {_key(case)}, {case['bits']}-bit, {_path(case)}: direction and magnitude vetoed."
            for case in inverted
        )
    else:
        lines.append("None recorded.")
    if family in ("dense", "sparse"):
        lines += ["", "## Direction-based crossovers", ""]
        paths = [
            path
            for path in PATH_LABELS
            if path.startswith("cuda-") and path != "cuda-resident-graph"
        ]
        for bits in (4, 8):
            for path in paths:
                indexed = {(case["count"], bits, path): case for case in _rows(cases, path, bits)}
                result = find_crossovers(indexed, sorted({c["count"] for c in cases}), bits, path)
                if result["status"] == "resolved":
                    low, high = result["interval"]
                    lines.append(
                        f"- {path}, {bits}-bit: {power_label(low)} to {power_label(high)}."
                    )
                else:
                    lines.append(f"- {path}, {bits}-bit: not resolved.")
    lines += ["", "## Figures", ""]
    lines.extend(f"- ![{key.upper()}]({name})" for key, name in PUBLICATION_FIGURES.items())
    lines.append("")
    return lines


def render_publication_report(
    snapshot: Path,
    manifest: dict[str, Any],
    cases: list[dict[str, Any]],
    output_dir: Path | None,
    k1_ab: Path | None,
    second_platform: Path | None,
) -> dict[str, Any]:
    out = output_dir or snapshot
    out.mkdir(parents=True, exist_ok=True)
    usable = _usable(cases)
    if len(usable) != len(cases):
        raise ValueError(
            "publication report requires every case to pass correctness and statistics"
        )
    second = None
    if second_platform is not None:
        _, second_cases = load_snapshot(second_platform)
        second = _usable(second_cases)
    if k1_ab is None:
        candidates = sorted(snapshot.parent.glob("*-k1-ab/summary.json"))
        if len(candidates) != 1:
            raise ValueError("pass --k1-ab with the publication K1 A/B snapshot")
        k1_ab = candidates[0].parent
    plot_publication_time(usable, out / FIGURES["f1"], second)
    plot_publication_speedup(usable, out / FIGURES["f2"])
    plot_publication_stages(usable, out / FIGURES["f3"])
    legacy = {
        (case["count"], case["bits"], case["timing_boundary"]): case
        for case in usable
        if _path(case) == f"cuda-{case['timing_boundary']}"
    }
    plot_graph_vs_resident(
        legacy,
        sorted({case["count"] for case in usable}),
        [4, 8],
        out / FIGURES["f4"],
    )
    bandwidth = plot_bandwidth(manifest, usable, k1_ab, out / PUBLICATION_FIGURES["bandwidth"])
    (out / REPORT).write_text("\n".join(_report_lines(usable, manifest)), encoding="utf-8")
    return {
        "figures": [out / name for name in PUBLICATION_FIGURES.values()],
        "report": out / REPORT,
        "bandwidth": bandwidth,
    }
