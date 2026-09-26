import json

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
