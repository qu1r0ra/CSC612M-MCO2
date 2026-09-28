import json
from pathlib import Path

import pytest

from bench_report import FIGURES, STAGE_PATHS, find_crossovers, index_cases, render_report

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def make_case(
    count, bits, boundary, median, verdict, supported, *, stages=True, rev2=None, magnitude=None
):
    backend = "cpu" if boundary == "comparator" else "cuda"
    stats = {
        "median_ms": median,
        "trial_median_min_ms": median * 0.95,
        "trial_median_max_ms": median * 1.05,
    }
    if boundary != "comparator":
        speedup = 1.0 / median
        stats.update(
            {
                "speedup_vs_cpu": speedup,
                "speedup_low": speedup * 0.9,
                "speedup_high": speedup * 1.1,
                "speedup_ci_low": speedup * 0.95,
                "speedup_ci_high": speedup * 1.05,
                "verdict": verdict,
                "direction_supported": supported,
                "magnitude_supported": supported if magnitude is None else magnitude,
                "claim_supported_rev2": supported if rev2 is None else rev2,
            }
        )
        if boundary == "resident-graph":
            stats["vs_resident"] = {
                "speedup_vs_resident": 1.1,
                "speedup_low": 1.05,
                "speedup_high": 1.15,
                "speedup_ci_low": 1.06,
                "speedup_ci_high": 1.14,
                "verdict": "faster",
                "direction_supported": True,
                "magnitude_supported": True,
                "claim_supported_rev2": True,
            }
    else:
        stats.update(
            {
                "verdict": "comparator",
                "direction_supported": None,
                "magnitude_supported": None,
                "claim_supported_rev2": None,
            }
        )
    run = {"samples_ms": [median, median]}
    if boundary == "resident-graph":
        run["capture_and_instantiate_ms"] = 0.5
    elif backend == "cuda" and stages:
        run.update({"k1_ms": [0.5 * median] * 2, "k2_ms": [0.1 * median] * 2})
        run["k3_ms"] = [0.1 * median] * 2
        if boundary == "host-origin":
            run.update({"h2d_ms": [0.1 * median] * 2, "d2h_ms": [0.05 * median] * 2})
    case = {
        "case_id": f"case_{backend}_{boundary}_bits{bits}_n{count}",
        "count": count,
        "bits": bits,
        "backend": backend,
        "timing_boundary": boundary,
        "correctness": {"status": "passed"},
        "statistics": stats,
        "trial_runs": [run, run],
    }
    return case


def write_snapshot(root, cases):
    root.mkdir()
    for case in cases:
        (root / f"{case['case_id']}.json").write_text(json.dumps(case), encoding="utf-8")
    manifest = {
        "git_provenance": {"code_revision_short": "abc1234"},
        "all_cases_passed": True,
        "cases": [c["case_id"] for c in cases],
        "device": {
            "name": "Fixture GPU",
            "memory_clock_khz": 14000000,
            "bus_width_bits": 128,
            "l2_bytes": 1024,
        },
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def sweep_cases():
    # CUDA time ratio vs C: slower at 2^10, faster from 2^12 on.
    cases = []
    for count, ratio in ((1024, 4.0), (4096, 0.5), (16384, 0.1)):
        for bits in (4, 8):
            cases.append(make_case(count, bits, "comparator", 1.0, None, None))
            verdict = "slower" if ratio > 1 else "faster"
            cases.append(make_case(count, bits, "resident", ratio, verdict, True))
            cases.append(
                make_case(
                    count,
                    bits,
                    "resident-graph",
                    ratio * 0.9,
                    "slower" if (ratio * 0.9) > 1 else "faster",
                    True,
                    stages=False,
                )
            )
            # Host-origin at 2^12 keeps its direction but loses magnitude and the rev 2 claim.
            cases.append(
                make_case(
                    count,
                    bits,
                    "host-origin",
                    ratio * 2,
                    verdict,
                    True,
                    rev2=count != 4096,
                    magnitude=count != 4096,
                )
            )
    return cases


def test_render_report_writes_figures_and_table(tmp_path):
    snapshot = tmp_path / "snap"
    write_snapshot(snapshot, sweep_cases())
    before = sorted(p.name for p in snapshot.iterdir())
    out = tmp_path / "out"

    result = render_report(snapshot, out)

    assert sorted(p.name for p in snapshot.iterdir()) == before
    for name in FIGURES.values():
        data = (out / name).read_bytes()
        assert data.startswith(PNG_MAGIC)
        assert len(data) > 10_000
    report = (out / "report.md").read_text(encoding="utf-8")
    assert "abc1234" in report
    assert report.count("| 2^") == 18  # one row per size, bit width and CUDA path
    assert "CUDA resident-graph" in report
    assert "resident-graph" not in STAGE_PATHS
    header = next(line for line in report.splitlines() if line.startswith("| Elements"))
    for column in ("95% CI", "Direction", "Magnitude", "Rev 2"):
        assert column in header
    assert (
        "| 2^12 | 8 | CUDA host-origin | 1 | 1 | 1× | [0.9, 1.1] | [0.95, 1.05] | faster | yes | no | no |"
        in report
    )
    assert "CUDA resident, 4-bit: resolved between 2^10 and 2^12" in report
    assert "CUDA resident-graph, 4-bit: resolved between 2^10 and 2^12" in report
    assert "### Revision 2 crossover" in report
    # Direction alone resolves host-origin; the rev 2 rule does not.
    assert result["crossovers"][(8, "host-origin")]["status"] == "resolved"
    assert result["crossovers_rev2"][(8, "host-origin")]["status"] == "not resolved"


def test_render_report_defaults_to_the_snapshot_folder(tmp_path):
    snapshot = tmp_path / "snap"
    write_snapshot(snapshot, sweep_cases())
    render_report(snapshot)
    for name in [*FIGURES.values(), "report.md"]:
        assert (snapshot / name).is_file()


def test_stage_figure_tolerates_missing_stage_times(tmp_path):
    # Pre-2.1 snapshots lack copy timings; a case may carry no stage times at all.
    snapshot = tmp_path / "snap"
    cases = [make_case(16384, 8, "comparator", 1.0, None, None)]
    cases += [
        make_case(16384, 8, b, 0.1, "faster", True, stages=False)
        for b in ("resident", "host-origin")
    ]
    write_snapshot(snapshot, cases)
    result = render_report(snapshot, tmp_path / "out")
    assert all(path.is_file() for path in result["figures"])


def test_crossover_needs_direction_support_on_both_sides():
    cases = []
    for count, verdict, supported in (
        (1024, "slower", True),
        (2048, "faster", False),
        (4096, "faster", True),
    ):
        cases.append(make_case(count, 8, "resident", 1.0, verdict, supported))
    indexed = index_cases(cases)
    result = find_crossovers(indexed, [1024, 2048, 4096], 8, "resident")
    assert result["status"] == "not resolved"
    assert result["largest_supported_slower"] == 1024
    assert result["smallest_supported_faster"] == 4096

    cases[1] = make_case(2048, 8, "resident", 1.0, "faster", True)
    result = find_crossovers(index_cases(cases), [1024, 2048, 4096], 8, "resident")
    assert result == {
        "status": "resolved",
        "interval": [1024, 2048],
        "direction": "slower to faster",
    }


def test_extension_cases_stay_out_of_the_revision_3_index():
    pageable = make_case(1024, 8, "host-origin", 2.0, "faster", True)
    pageable["transfer_policy"] = "pageable"
    pinned = make_case(1024, 8, "host-origin", 1.0, "faster", True)
    pinned["transfer_policy"] = "pinned"
    gpu_origin = make_case(1024, 8, "gpu-origin", 1.5, "faster", True)
    gpu_origin["transfer_policy"] = "pageable"
    indexed = index_cases([pageable, pinned, gpu_origin])
    assert indexed == {(1024, 8, "host-origin"): pageable}


def test_rev2_snapshot_fields_still_render(tmp_path):
    # Frozen revision 2 snapshots carry claim_supported and no CI or direction fields.
    cases = sweep_cases()
    for case in cases:
        stats = case["statistics"]
        stats["claim_supported"] = stats.pop("claim_supported_rev2")
        for key in (
            "direction_supported",
            "magnitude_supported",
            "speedup_ci_low",
            "speedup_ci_high",
        ):
            stats.pop(key, None)
    snapshot = tmp_path / "snap"
    write_snapshot(snapshot, cases)
    result = render_report(snapshot, tmp_path / "out")
    report = result["report"].read_text(encoding="utf-8")
    assert "| — | faster | yes | — |" in report
    assert result["crossovers"][(4, "resident")]["status"] == "resolved"


def test_render_report_draws_a_ci_that_excludes_the_point(tmp_path):
    # The point speedup is a ratio of pooled medians and the CI is bootstrapped
    # from trial medians, so the point can sit outside its CI (seen in the #34 pilot).
    cases = sweep_cases()
    for case in cases:
        stats = case["statistics"]
        if case["timing_boundary"] == "resident-graph":
            stats["speedup_ci_low"] = stats["speedup_vs_cpu"] * 0.97
            stats["speedup_ci_high"] = stats["speedup_vs_cpu"] * 0.99
            stats["vs_resident"]["speedup_ci_low"] = 1.11
            stats["vs_resident"]["speedup_ci_high"] = 1.12
    snapshot = tmp_path / "snap"
    write_snapshot(snapshot, cases)

    render_report(snapshot, tmp_path / "out")

    for key in ("f2", "f4"):
        assert (tmp_path / "out" / FIGURES[key]).read_bytes().startswith(PNG_MAGIC)


def test_avx2_comparator_is_drawn_but_stays_out_of_t1_and_crossovers(tmp_path, monkeypatch):
    import bench_report

    cases = sweep_cases()
    for count in (1024, 4096, 16384):
        for bits in (4, 8):
            avx2 = make_case(count, bits, "optimized", 0.2, "faster", None)
            avx2["backend"] = "cpu-avx2"
            avx2["statistics"].update(
                {
                    "descriptive": True,
                    "direction_supported": None,
                    "magnitude_supported": None,
                    "claim_supported_rev2": None,
                }
            )
            cases.append(avx2)
    snapshot = tmp_path / "snap"
    write_snapshot(snapshot, cases)
    drawn = []
    original = bench_report.plot_descriptive_speedup

    def spy(ax, indexed, counts, bits):
        drawn.append(bits)
        original(ax, indexed, counts, bits)

    monkeypatch.setattr(bench_report, "plot_descriptive_speedup", spy)

    result = render_report(snapshot, tmp_path / "out")

    assert (1024, 8, "optimized") in index_cases(cases)
    assert drawn == [4, 8]
    report = result["report"].read_text(encoding="utf-8")
    assert report.count("| 2^") == 18
    assert "AVX2" not in report
    assert all(path != "optimized" for _, path in result["crossovers"])


def publication_cases(family="dense"):
    cases = []
    paths = (
        ("cpu", "comparator", "none"),
        ("cpu-avx2", "optimized", "none"),
        ("cuda", "resident", "none"),
        ("cuda", "resident-graph", "none"),
        ("cuda", "host-origin", "pageable"),
        ("cuda", "host-origin", "pinned"),
        ("cpu", "gpu-origin", "pageable"),
        ("cuda", "gpu-origin", "pageable"),
        ("cpu", "gpu-origin", "pinned"),
        ("cuda", "gpu-origin", "pinned"),
    )
    for count in (1024, 4096):
        for bits in (4, 8):
            for index, (backend, boundary, policy) in enumerate(paths):
                case = make_case(count, bits, boundary, 1 + index / 10, "faster", True)
                suffix = "-pinned" if policy == "pinned" else ""
                case.update(
                    backend=backend,
                    transfer_policy=policy,
                    path_label=f"{backend}-{boundary}{suffix}",
                    input_family=family,
                    input_key=f"n{count}" if family != "model" else f"tensor_{count}",
                    case_id=f"case_{backend}_{boundary}{suffix}_bits{bits}_n{count}",
                )
                case["statistics"]["baseline"] = (
                    "cpu-gpu-origin"
                    if backend == "cuda" and boundary == "gpu-origin"
                    else "cpu-comparator"
                )
                case["statistics"]["boundary_inversion"] = index == 5 and count == 1024
                if policy == "pinned":
                    case["statistics"]["vs_pageable"] = {
                        "speedup_vs_pageable": 1.2,
                        "speedup_low": 1.1,
                        "speedup_high": 1.3,
                        "speedup_ci_low": 1.15,
                        "speedup_ci_high": 1.25,
                        "verdict": "faster",
                        "direction_supported": True,
                        "magnitude_supported": True,
                    }
                cases.append(case)
    return cases


def write_k1_summary(root):
    root.mkdir()
    cells = [
        {
            "count": count,
            "bits": bits,
            "arms": {
                "reference": {"effective_gbps": 100.0},
                "optimized": {"effective_gbps": 400.0},
            },
        }
        for count in (1024, 4096)
        for bits in (4, 8)
    ]
    (root / "summary.json").write_text(
        json.dumps(
            {
                "evidence": True,
                "git": {"git_dirty": False},
                "theoretical_peak_gbps": 448.0,
                "probe_ceiling_gbps": 430.0,
                "cells": cells,
            }
        ),
        encoding="utf-8",
    )
    (root / "stream_probe.json").write_text(
        json.dumps(
            {
                "device": {
                    "name": "Fixture GPU",
                    "memory_clock_khz": 14000000,
                    "bus_width_bits": 128,
                    "l2_bytes": 1024,
                },
                "sizes": [{"bytes": 4096, "best_gbps": 430.0}],
            }
        ),
        encoding="utf-8",
    )


def test_publication_report_includes_extensions_and_bandwidth(tmp_path, monkeypatch):
    import publication_report

    snapshot = tmp_path / "dense"
    write_snapshot(snapshot, publication_cases())
    k1 = tmp_path / "k1-ab"
    write_k1_summary(k1)
    drawn = []
    original = publication_report.plot_publication_time

    def spy(cases, out, second):
        drawn.append(second)
        return original(cases, out, second)

    monkeypatch.setattr(publication_report, "plot_publication_time", spy)
    out = tmp_path / "out"
    result = render_report(snapshot, out, k1_ab=k1)
    assert drawn == [None]
    assert result["bandwidth"] == {"peak_gbps": 448.0, "ceiling_gbps": 430.0}
    assert all(path.read_bytes().startswith(PNG_MAGIC) for path in result["figures"])
    report = result["report"].read_text(encoding="utf-8")
    assert "cuda-gpu-origin-pinned" in report
    assert "cpu-gpu-origin-pinned" in report
    assert "Pinned vs pageable" in report
    assert "direction and magnitude vetoed" in report
    assert "| dense | 2 | 40 | 2 |" in report

    second = tmp_path / "second"
    write_snapshot(second, publication_cases())
    render_report(snapshot, tmp_path / "second-out", k1_ab=k1, second_platform=second)
    assert drawn[-1] is not None
    assert len(drawn[-1]) == 40


def test_sparse_and_model_publication_reports(tmp_path):
    k1 = tmp_path / "k1-ab"
    write_k1_summary(k1)
    for family in ("sparse", "model"):
        snapshot = tmp_path / family
        write_snapshot(snapshot, publication_cases(family))
        report = render_report(snapshot, tmp_path / f"{family}-out", k1_ab=k1)["report"]
        assert f"# Publication matrix report ({family})" in report.read_text(encoding="utf-8")
        assert "## Input family" in report.read_text(encoding="utf-8")


def test_publication_bandwidth_rejects_same_name_device_mismatch(tmp_path):
    snapshot = tmp_path / "dense"
    write_snapshot(snapshot, publication_cases())
    k1 = tmp_path / "k1-ab"
    write_k1_summary(k1)
    manifest = json.loads((snapshot / "manifest.json").read_text(encoding="utf-8"))
    manifest["device"]["bus_width_bits"] = 192
    (snapshot / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="attributes differ"):
        render_report(snapshot, tmp_path / "out", k1_ab=k1)


def test_revision_3_frozen_report_remains_byte_identical(tmp_path):
    snapshot = Path(__file__).resolve().parents[1] / "results" / "2026-09-26-a1d2439"
    report = render_report(snapshot, tmp_path / "old-output")["report"]
    assert report.read_bytes() == (snapshot / "report.md").read_bytes()


GOLDEN = Path(__file__).resolve().parent / "golden"


def render_publication_goldens(root):
    """Render each publication golden under root; returns {golden name: report path}."""
    root.mkdir(parents=True)
    k1 = root / "k1-ab"
    write_k1_summary(k1)
    reports = {}
    for family in ("dense", "sparse", "model"):
        snapshot = root / family
        write_snapshot(snapshot, publication_cases(family))
        out = root / f"{family}-out"
        reports[f"publication_{family}.md"] = render_report(snapshot, out, k1_ab=k1)["report"]
    return reports


def test_publication_reports_remain_byte_identical(tmp_path):
    first = render_publication_goldens(tmp_path / "a")
    elsewhere = render_publication_goldens(tmp_path / "somewhere-else")
    for name, report in first.items():
        assert report.read_bytes() == elsewhere[name].read_bytes(), name
        assert report.read_bytes() == (GOLDEN / name).read_bytes(), name
