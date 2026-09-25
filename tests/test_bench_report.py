import json

from bench_report import FIGURES, find_crossovers, index_cases, render_report

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def make_case(count, bits, boundary, median, verdict, supported, *, stages=True):
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
                "verdict": verdict,
                "claim_supported": supported,
            }
        )
    else:
        stats.update({"verdict": "comparator", "claim_supported": None})
    run = {"samples_ms": [median, median]}
    if backend == "cuda" and stages:
        run.update({"k1_ms": [0.5 * median] * 2, "k2_ms": [0.1 * median] * 2})
        run["k3_ms"] = [0.1 * median] * 2
        if boundary == "host-origin":
            run.update({"h2d_ms": [0.1 * median] * 2, "d2h_ms": [0.05 * median] * 2})
    return {
        "case_id": f"case_{backend}_{boundary}_bits{bits}_n{count}",
        "count": count,
        "bits": bits,
        "backend": backend,
        "timing_boundary": boundary,
        "correctness": {"status": "passed"},
        "statistics": stats,
        "trial_runs": [run, run],
    }


def write_snapshot(root, cases):
    root.mkdir()
    for case in cases:
        (root / f"{case['case_id']}.json").write_text(json.dumps(case), encoding="utf-8")
    manifest = {
        "git_provenance": {"code_revision_short": "abc1234"},
        "all_cases_passed": True,
        "cases": [c["case_id"] for c in cases],
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
            cases.append(make_case(count, bits, "host-origin", ratio * 2, verdict, count != 4096))
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
    assert report.count("| 2^") == 6
    assert "†" in report  # host-origin at 2^12 lacks claim support
    assert "CUDA resident, 4-bit: resolved between 2^10 and 2^12" in report
    assert result["crossovers"][(8, "host-origin")]["status"] == "not resolved"


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


def test_crossover_needs_claim_support_on_both_sides():
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
