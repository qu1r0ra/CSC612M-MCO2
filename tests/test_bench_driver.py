import csv
import itertools
import json
import os
import subprocess
from pathlib import Path

import pytest

import benchmark_driver
from benchmark_driver import (
    BOOTSTRAP_SEED,
    DEFAULT_COUNTS,
    DEFAULT_TRIALS,
    EXCLUDED_LOGICAL_CPUS,
    PILOT_COUNTS,
    SUMMARY_FIELDS,
    BenchPath,
    affinity_mask_excluding,
    bootstrap_speedup_ci,
    boundary_inversion,
    build_paths,
    case_order,
    check_readiness,
    claim_support,
    compact_invocation_ids,
    compare_case_group,
    compare_to_comparator,
    compute_case_statistics,
    get_process_affinity,
    in_process_warmups,
    run_benchmark_matrix,
    trial_design,
    trial_orders,
    williams_rows,
)

ROOT = Path(__file__).resolve().parents[1]

CUDA_SKIP = pytest.mark.skipif(
    os.environ.get("MCO2_TEST_CUDA") != "1",
    reason="run with `just test-cuda` on a CUDA device",
)


def test_driver_cpu_only_produces_valid_snapshot(tmp_path):
    snapshot_dir = tmp_path / "test-cpu-snapshot"
    result_dir = run_benchmark_matrix(
        root=ROOT,
        output_dir=snapshot_dir,
        counts=[1024],
        bit_widths=[4, 8],
        backends=["cpu"],
        warmups=1,
        reps=2,
        trials=2,
        in_process_warmup_seconds=0,
        allow_existing=True,
        allow_dirty=True,
        readiness_facts=READY_FACTS,
    )
    assert result_dir == snapshot_dir

    manifest_path = snapshot_dir / "manifest.json"
    assert manifest_path.is_file()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["manifest_version"] == "3.1"
    assert manifest["transfer_policies"] == ["pageable"]
    assert manifest["matrix_parameters"]["paths"] == ["cpu-comparator"]
    assert manifest["matrix_parameters"]["trial_design"] == "all-permutations"
    assert len(manifest["cases"]) == 2  # 1 size * 2 bit widths * 1 CPU path
    assert manifest["all_cases_passed"] is True
    assert manifest["gpu_state"]["warmup"] is None
    assert manifest["matrix_parameters"]["in_process_warmup_seconds"] == 0
    assert "disabled" in manifest["matrix_parameters"]["in_process_warmup_rule"]
    conditions = manifest["run_conditions"]
    assert conditions["readiness"]["passed"] is True
    assert conditions["readiness"]["overridden"] is False
    assert conditions["pilot"] is False
    assert conditions["affinity"]["excluded_logical_cpus"] == list(EXCLUDED_LOGICAL_CPUS)
    assert conditions["evidence"] is (not manifest["git_provenance"]["git_dirty"])
    assert set(manifest["statistics_method"]["claim_rules"]) == {
        "verdict",
        "direction_supported",
        "magnitude_supported",
        "claim_supported_rev2",
        "baselines",
    }
    assert manifest["statistics_method"]["bootstrap"]["seed"] == BOOTSTRAP_SEED

    csv_path = snapshot_dir / manifest["summary_csv"]
    assert csv_path.is_file()
    with csv_path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    for row in rows:
        assert row["backend"] == "cpu"
        assert row["correctness"] == "passed"
        assert float(row["speedup_vs_c"]) == 1.0


@CUDA_SKIP
def test_driver_tiny_matrix_produces_valid_snapshot(tmp_path):
    snapshot_dir = tmp_path / "path with space" / "test-snapshot"
    result_dir = run_benchmark_matrix(
        root=ROOT,
        output_dir=snapshot_dir,
        counts=[1024, 2048],
        bit_widths=[4, 8],
        backends=["cpu", "cuda"],
        warmups=1,
        reps=2,
        trials=2,
        case_order_seed=7,
        gpu_warmup_seconds=0.5,
        case_warmup_seconds=0.2,
        in_process_warmup_seconds=0.01,
        allow_existing=True,
        allow_dirty=True,
        readiness_facts=READY_FACTS,
    )
    assert result_dir == snapshot_dir

    # 1. Manifest
    manifest_path = snapshot_dir / "manifest.json"
    assert manifest_path.is_file()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["manifest_version"] == "3.1"
    assert "date" in manifest
    assert "created_at_utc" in manifest
    assert "git_provenance" in manifest
    for key in ("code_revision", "code_revision_short", "git_dirty", "dirty_files"):
        assert key in manifest["git_provenance"]
    assert "hardware" in manifest
    for key in ("gpu_name", "compute_capability", "host_cpu"):
        assert key in manifest["hardware"]
    assert "toolkit_and_driver" in manifest
    for key in ("cuda_toolkit", "driver_version", "c_compiler"):
        assert key in manifest["toolkit_and_driver"]
    assert "build_flags" in manifest
    for key in ("source", "commands", "comparator_c", "cuda_nvcc"):
        assert key in manifest["build_flags"]
    assert "/DMCO2_ENABLE_CUDA" in manifest["build_flags"]["comparator_c"]
    assert "--fmad=false" in manifest["build_flags"]["cuda_nvcc"]
    assert manifest["transfer_policies"] == ["pageable"]
    assert "transfer_policy" not in manifest["toolkit_and_driver"]
    gpu_state = manifest["gpu_state"]
    assert set(gpu_state) == {"note", "start", "warmup", "after_warmup", "end"}
    assert gpu_state["warmup"]["processes"] >= 1
    assert gpu_state["warmup"]["seconds_elapsed"] >= 0.5
    assert "2048" in gpu_state["warmup"]["workload"]
    assert gpu_state["after_warmup"] is not None
    params = manifest["matrix_parameters"]
    assert params["case_order_seed"] == 7
    assert params["case_order"] == [
        {"count": c, "bits": b} for c, b in case_order([1024, 2048], [4, 8], 7)
    ]
    assert sorted((c["count"], c["bits"]) for c in params["case_order"]) == [
        (1024, 4),
        (1024, 8),
        (2048, 4),
        (2048, 8),
    ]
    assert params["trials"] == 2
    assert params["warmup"] == 1
    assert params["in_process_warmup_seconds"] == 0.01
    assert len(params["trial_orders"]) == 2
    assert params["trial_orders"][0] != params["trial_orders"][1]
    # Check that DEFAULT_TRIALS generates all 24 distinct orderings for the 4 paths
    assert params["paths"] == [
        "cpu-comparator",
        "cuda-resident",
        "cuda-resident-graph",
        "cuda-host-origin",
    ]
    assert params["trial_design"] == "all-permutations"
    all_orders = trial_orders(build_paths(["cpu", "cuda"]), DEFAULT_TRIALS)
    assert len({tuple(o) for o in all_orders}) == 24
    method = manifest["statistics_method"]
    assert method["spread_threshold"] == 1.25
    assert "linear" in method["quantile_method"]

    # Inputs provenance
    assert set(manifest["inputs"]) == {"1024", "2048"}
    inp = manifest["inputs"]["1024"]
    assert inp["count"] == 1024
    assert inp["generator"] == "numpy.random.default_rng"
    assert "seed" in inp
    assert len(inp["sha256"]) == 64
    assert "_path" not in inp
    assert not (snapshot_dir / "inputs").exists()
    assert not (snapshot_dir / "_temp").exists()

    # Vectorization report
    vec_report_path = snapshot_dir / manifest["msvc_vectorization_report"]
    assert vec_report_path.is_file()
    vec_report = vec_report_path.read_text(encoding="utf-8")
    assert "/DMCO2_ENABLE_CUDA" in vec_report
    assert "/Qvec-report:2" in vec_report
    assert "Exit code: 0" in vec_report
    assert str(ROOT.resolve()) not in vec_report

    # Cases
    # 2 sizes * 2 bits (4, 8) * 4 paths (cpu, cuda-resident, cuda-resident-graph, cuda-host-origin) = 16 cases
    assert len(manifest["cases"]) == 16
    assert manifest["all_cases_passed"] is True

    # 2. Case JSONs
    warmup_by_case = {}
    for case_id in manifest["cases"]:
        case_file = snapshot_dir / f"{case_id}.json"
        assert case_file.is_file()
        case_data = json.loads(case_file.read_text(encoding="utf-8"))

        assert case_data["case_id"] == case_id
        assert case_data["count"] in (1024, 2048)
        assert case_data["case_warmup"]["processes"] >= 1
        assert case_data["case_warmup"]["gpu_state_after"] is not None
        assert case_data["bits"] in (4, 8)
        assert params["case_order"][case_data["execution_index"]] == {
            "count": case_data["count"],
            "bits": case_data["bits"],
        }
        assert case_data["backend"] in ("cpu", "cuda")
        assert case_data["timing_boundary"] in (
            "comparator",
            "resident",
            "resident-graph",
            "host-origin",
        )
        expected_policy = "pageable" if case_data["timing_boundary"] == "host-origin" else "none"
        assert case_data["transfer_policy"] == expected_policy
        assert case_data["path_label"] == f"{case_data['backend']}-{case_data['timing_boundary']}"
        probe = case_data["in_process_warmup"]
        assert probe["target_seconds"] == 0.01
        assert probe["probe_warmup"] == 1
        assert probe["probe_reps"] == 2
        assert probe["probe_median_ms"] >= 0
        assert case_data["warmup"] == probe["warmup"] >= 1
        warmup_by_case[(case_data["count"], case_data["bits"], case_data["timing_boundary"])] = (
            case_data["warmup"]
        )
        assert case_data["reps"] == 2
        assert case_data["trials"] == 2
        if case_data["timing_boundary"] == "resident-graph":
            # The captured graph replays the base invocation on every run.
            assert case_data["repetition_invocation_ids"] == [0, 0]
            assert case_data["warmup_invocation_ids"] == compact_invocation_ids(
                [0] * case_data["warmup"]
            )
        else:
            assert case_data["repetition_invocation_ids"] == [0, 1]
            assert case_data["warmup_invocation_ids"] == {
                "first": 2,
                "last": 1 + case_data["warmup"],
                "count": case_data["warmup"],
            }
        assert case_data["header_bytes"] == 20
        count = case_data["count"]
        assert case_data["payload_bytes"] == (count if case_data["bits"] == 8 else count // 2)
        assert case_data["correctness"]["status"] == "passed"
        assert case_data["correctness"]["byte_identical_to_compress"] is True
        assert case_data["correctness"]["cpu_cuda_byte_identical"] is True
        assert case_data["correctness"]["layer2_scale_within_bound"] is True
        assert case_data["correctness"]["layer2_reconstruction_within_bound"] is True
        assert "_path" not in case_data["input_provenance"]

        assert len(case_data["samples_ms"]) == 4  # 2 trials * 2 reps, pooled
        assert all(sample >= 0 for sample in case_data["samples_ms"])
        runs = case_data["trial_runs"]
        assert [run["trial"] for run in runs] == [0, 1]
        for run in runs:
            assert len(run["samples_ms"]) == 2
            assert run["repetition_invocation_ids"] == case_data["repetition_invocation_ids"]
            if case_data["timing_boundary"] == "resident-graph":
                assert run["capture_and_instantiate_ms"] > 0
                for k in ("k1_ms", "k2_ms", "k3_ms", "h2d_ms", "d2h_ms"):
                    assert k not in run
            elif case_data["backend"] == "cuda":
                for k in ("k1_ms", "k2_ms", "k3_ms"):
                    assert len(run[k]) == 2
                    assert all(s >= 0 for s in run[k])
            if case_data["timing_boundary"] == "host-origin":
                for k in ("h2d_ms", "d2h_ms"):
                    assert len(run[k]) == 2
                    assert all(s > 0 for s in run[k])
            else:
                assert "h2d_ms" not in run and "d2h_ms" not in run

        stages = case_data["stage_medians_ms"]
        if case_data["backend"] == "cpu" or case_data["timing_boundary"] == "resident-graph":
            assert stages is None
        else:
            expected = {"k1_ms", "k2_ms", "k3_ms", "other_ms"}
            if case_data["timing_boundary"] == "host-origin":
                expected |= {"h2d_ms", "d2h_ms"}
            assert set(stages) == expected

        stats = case_data["statistics"]
        assert stats is not None
        assert stats["median_ms"] >= 0
        assert stats["iqr_ms"] >= 0
        assert stats["speedup_vs_cpu"] >= 0
        assert len(stats["trial_medians_ms"]) == 2
        assert stats["spread_ratio"] >= 1.0
        if case_data["backend"] == "cpu":
            assert stats["speedup_vs_cpu"] == 1.0
            assert stats["verdict"] == "comparator"
        else:
            assert stats["speedup_low"] <= stats["speedup_vs_cpu"] <= stats["speedup_high"]
            assert stats["verdict"] in ("faster", "slower", "inconclusive")
            assert stats["speedup_ci_low"] <= stats["speedup_ci_high"]
            assert isinstance(stats["direction_supported"], bool)
            assert isinstance(stats["magnitude_supported"], bool)
            assert isinstance(stats["claim_supported_rev2"], bool)
            assert stats["direction_supported"] or not stats["magnitude_supported"]

        if case_data["timing_boundary"] == "resident-graph":
            assert "vs_resident" in case_data["statistics"]
            assert "vs_resident" in stats
            vs_res = case_data["statistics"]["vs_resident"]
            assert vs_res["speedup_vs_resident"] > 0
            assert vs_res["speedup_low"] <= vs_res["speedup_vs_resident"] <= vs_res["speedup_high"]
            assert vs_res["speedup_ci_low"] <= vs_res["speedup_ci_high"]
            assert vs_res["verdict"] in ("faster", "slower", "inconclusive")
            assert isinstance(vs_res["direction_supported"], bool)
            assert isinstance(vs_res["magnitude_supported"], bool)

    # 3. Summary CSV
    csv_path = snapshot_dir / manifest["summary_csv"]
    assert csv_path.is_file()
    with csv_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == 16
    assert reader.fieldnames == SUMMARY_FIELDS
    order = [
        "cpu-comparator",
        "cuda-resident",
        "cuda-resident-graph",
        "cuda-host-origin",
    ]
    keys = [
        (int(r["count"]), int(r["bits"]), order.index(f"{r['backend']}-{r['boundary']}"))
        for r in rows
    ]
    assert keys == sorted(keys)
    for row in rows:
        assert row["count"] in ("1024", "2048")
        assert row["bits"] in ("4", "8")
        assert row["correctness"] == "passed"
        assert float(row["median_ms"]) >= 0
        assert float(row["iqr_ms"]) >= 0
        assert float(row["speedup_vs_c"]) >= 0
        key = (int(row["count"]), int(row["bits"]), row["boundary"])
        assert int(row["warmup"]) == warmup_by_case[key]


@CUDA_SKIP
def test_driver_extension_matrix_adds_gpu_origin_and_pinned_paths(tmp_path):
    snapshot_dir = tmp_path / "extension"
    run_benchmark_matrix(
        root=ROOT,
        output_dir=snapshot_dir,
        counts=[1024],
        bit_widths=[8],
        backends=["cpu", "cuda"],
        warmups=1,
        reps=2,
        trials=18,
        gpu_warmup_seconds=0.2,
        case_warmup_seconds=0.1,
        in_process_warmup_seconds=0.01,
        allow_existing=True,
        allow_dirty=True,
        readiness_facts=READY_FACTS,
        boundaries=["gpu-origin"],
        transfer_policies=["pageable", "pinned"],
    )
    manifest = json.loads((snapshot_dir / "manifest.json").read_text(encoding="utf-8"))
    params = manifest["matrix_parameters"]
    labels = [
        "cpu-comparator",
        "cuda-resident",
        "cuda-resident-graph",
        "cuda-host-origin",
        "cuda-host-origin-pinned",
        "cpu-gpu-origin",
        "cuda-gpu-origin",
        "cpu-gpu-origin-pinned",
        "cuda-gpu-origin-pinned",
    ]
    assert manifest["transfer_policies"] == ["pageable", "pinned"]
    assert params["paths"] == labels
    assert params["trial_design"] == "williams"
    assert len(params["trial_orders"]) == 18
    assert manifest["all_cases_passed"] is True
    assert len(manifest["cases"]) == 9

    cases = {}
    for case_id in manifest["cases"]:
        case = json.loads((snapshot_dir / f"{case_id}.json").read_text(encoding="utf-8"))
        cases[case["path_label"]] = case
    assert sorted(cases) == sorted(labels)
    assert cases["cuda-host-origin-pinned"]["transfer_policy"] == "pinned"
    assert cases["cuda-host-origin-pinned"]["timing_boundary"] == "host-origin"
    assert cases["cpu-gpu-origin"]["case_id"] == "case_cpu_gpu-origin_bits8_n1024"
    for case in cases.values():
        assert case["correctness"]["status"] == "passed"
        assert case["correctness"]["byte_identical_to_compress"] is True
        assert len(case["trial_runs"]) == 18
    assert set(cases["cpu-gpu-origin"]["stage_medians_ms"]) == {"d2h_ms", "cpu_ms", "other_ms"}
    assert set(cases["cuda-gpu-origin-pinned"]["stage_medians_ms"]) == {
        "k1_ms",
        "k2_ms",
        "k3_ms",
        "d2h_ms",
        "other_ms",
    }
    stats = {label: case["statistics"] for label, case in cases.items()}
    assert stats["cuda-gpu-origin"]["baseline"] == "cpu-gpu-origin"
    assert stats["cuda-gpu-origin-pinned"]["baseline"] == "cpu-gpu-origin-pinned"
    assert stats["cpu-gpu-origin-pinned"]["baseline"] == "cpu-comparator"
    assert "vs_pageable" in stats["cuda-host-origin-pinned"]
    assert "vs_pageable" in stats["cpu-gpu-origin-pinned"]
    assert stats["cuda-gpu-origin"]["vs_comparator"]["descriptive"] is True

    with (snapshot_dir / manifest["summary_csv"]).open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert reader.fieldnames == SUMMARY_FIELDS
    assert [row["path_label"] for row in rows] == labels
    assert [row["transfer_policy"] for row in rows] == [
        "none",
        "none",
        "none",
        "pageable",
        "pinned",
        "pageable",
        "pageable",
        "pinned",
        "pinned",
    ]
    assert rows[6]["baseline"] == "cpu-gpu-origin"


@CUDA_SKIP
def test_driver_forced_failure_marks_failed_without_speed_figures(tmp_path):
    snapshot_dir = tmp_path / "failed-snapshot"
    result_dir = run_benchmark_matrix(
        root=ROOT,
        output_dir=snapshot_dir,
        counts=[1024],
        bit_widths=[4],
        backends=["cpu", "cuda"],
        warmups=1,
        reps=2,
        trials=2,
        gpu_warmup_seconds=0,
        case_warmup_seconds=0,
        force_fail=True,
        allow_existing=True,
        allow_dirty=True,
        readiness_facts=READY_FACTS,
    )
    assert result_dir == snapshot_dir

    manifest = json.loads((snapshot_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["all_cases_passed"] is False

    # Check case JSON
    for case_id in manifest["cases"]:
        case_file = snapshot_dir / f"{case_id}.json"
        case_data = json.loads(case_file.read_text(encoding="utf-8"))
        assert case_data["correctness"]["status"] == "failed"
        assert case_data["samples_ms"] == []
        assert case_data["trial_runs"] == []
        assert case_data["statistics"] is None

    # Check CSV has no speed figures
    with (snapshot_dir / "summary.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 4  # 1 size * 1 bit width * 4 paths
    for row in rows:
        assert row["correctness"] == "failed"
        assert row["median_ms"] == ""
        assert row["iqr_ms"] == ""
        assert row["speedup_vs_c"] == ""
        assert row["verdict"] == ""
        assert row["direction_supported"] == ""
        assert row["magnitude_supported"] == ""
        assert row["claim_supported_rev2"] == ""


def test_driver_refuses_to_overwrite_existing_snapshot(tmp_path):
    snapshot_dir = tmp_path / "existing-snapshot"
    snapshot_dir.mkdir(parents=True)

    with pytest.raises(FileExistsError, match="already exists"):
        run_benchmark_matrix(
            root=ROOT,
            output_dir=snapshot_dir,
            counts=[1024],
            bit_widths=[4],
            warmups=1,
            reps=1,
            allow_existing=False,
            allow_dirty=True,
        )


def test_in_process_warmups_cover_the_target_time():
    assert in_process_warmups(10, 1.0, 0.05) == 20000
    assert in_process_warmups(10, 1.0, 0.3) == 3334
    # Slow repetitions never drop below the base warm-up.
    assert in_process_warmups(10, 1.0, 500.0) == 10
    assert in_process_warmups(10, 0.0, 0.05) == 10
    assert in_process_warmups(10, 1.0, 0.0) == 10


def test_compact_invocation_ids_keeps_bounds_of_contiguous_runs():
    assert compact_invocation_ids([2, 3, 4]) == {"first": 2, "last": 4, "count": 3}
    assert compact_invocation_ids([5, 7]) == [5, 7]
    assert compact_invocation_ids([4, 4, 4]) == {"repeated": 4, "count": 3}
    assert compact_invocation_ids([]) == []
    assert compact_invocation_ids(None) is None


def test_default_counts_are_the_power_of_two_sweep():
    # Tests pass explicit tiny counts; the default grid is the full sweep.
    assert DEFAULT_COUNTS == tuple(2**e for e in range(10, 27))


def test_driver_refuses_dirty_tree(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    (repo / "untracked.txt").write_text("dirty", encoding="utf-8")

    with pytest.raises(RuntimeError, match="dirty"):
        run_benchmark_matrix(
            root=repo,
            output_dir=tmp_path / "out",
            counts=[1024],
            bit_widths=[4],
            backends=["cpu"],
            warmups=1,
            reps=1,
            trials=1,
        )
    assert not (tmp_path / "out").exists()


def test_trial_orders_cover_every_permutation_once():
    paths = build_paths(["cpu", "cuda"])
    orders = trial_orders(paths, DEFAULT_TRIALS)
    assert DEFAULT_TRIALS == 24
    assert len({tuple(order) for order in orders}) == 24
    for position in range(4):
        assert sorted(order[position] for order in orders) == sorted([0, 1, 2, 3] * 6)


def test_speedup_verdicts_use_trial_median_ranges():
    cpu = compute_case_statistics([[10.0, 10.0], [11.0, 11.0]])
    fast = compute_case_statistics([[1.0, 1.0], [1.1, 1.1]])
    slow = compute_case_statistics([[20.0, 20.0], [22.0, 22.0]])
    overlap = compute_case_statistics([[9.0, 9.0], [12.0, 12.0]])

    assert compare_to_comparator(cpu, fast)["verdict"] == "faster"
    assert compare_to_comparator(cpu, slow)["verdict"] == "slower"
    result = compare_to_comparator(cpu, overlap)
    assert result["verdict"] == "inconclusive"
    assert result["speedup_low"] < 1.0 < result["speedup_high"]
    assert cpu["spread_ratio"] == pytest.approx(1.1)
    assert cpu["unstable_rev2"] is False
    assert overlap["unstable_rev2"] is True  # 12 / 9 > 1.25


def test_p90_p10_spread_ignores_one_outlier_trial():
    # Linear percentiles of [1..10]: P10 = 1.9, P90 = 9.1.
    stats = compute_case_statistics([[float(m)] for m in range(1, 11)])
    assert stats["spread_p90_p10"] == pytest.approx(9.1 / 1.9)
    assert stats["stable"] is False

    medians = [1.0] * 23 + [3.0]
    stats = compute_case_statistics([[m] for m in medians])
    assert stats["spread_ratio"] == pytest.approx(3.0)
    assert stats["unstable_rev2"] is True
    assert stats["spread_p90_p10"] == pytest.approx(1.0)
    assert stats["stable"] is True


def test_direction_claim_survives_instability_but_magnitude_does_not():
    cpu = compute_case_statistics([[10.0], [10.2], [10.1], [10.3]])
    # Always faster than the comparator, but its trial medians spread 2x.
    noisy = compute_case_statistics([[1.0], [2.0], [1.0], [2.0]])
    comparison = compare_to_comparator(cpu, noisy)
    assert comparison["verdict"] == "faster"
    assert noisy["stable"] is False

    claims = claim_support(comparison["verdict"], False, cpu, noisy)
    assert claims == {
        "direction_supported": True,
        "magnitude_supported": False,
        "claim_supported_rev2": False,
    }
    # An unstable comparator also blocks the magnitude claim.
    steady = compute_case_statistics([[1.0], [1.01], [1.02], [1.0]])
    assert claim_support("faster", False, noisy, steady)["magnitude_supported"] is False
    assert claim_support("faster", False, cpu, steady) == {
        "direction_supported": True,
        "magnitude_supported": True,
        "claim_supported_rev2": True,
    }
    assert claim_support("faster", True, cpu, steady)["direction_supported"] is False
    assert claim_support("inconclusive", False, cpu, steady)["direction_supported"] is False


def test_bootstrap_ci_is_deterministic_and_brackets_the_point_estimate():
    cpu = [10.0, 10.4, 9.8, 10.1, 10.3, 9.9]
    cuda = [2.0, 2.2, 1.9, 2.1, 2.05, 1.95]
    first = bootstrap_speedup_ci(cpu, cuda)
    assert first == bootstrap_speedup_ci(cpu, cuda)
    low, high = first
    assert low < 10.05 / 2.025 < high
    assert bootstrap_speedup_ci(cpu, cuda, seed=BOOTSTRAP_SEED + 1) != first

    comparison = compare_to_comparator(
        compute_case_statistics([[m] for m in cpu]), compute_case_statistics([[m] for m in cuda])
    )
    assert (comparison["speedup_ci_low"], comparison["speedup_ci_high"]) == first


def test_boundary_inversion_covers_every_resident_path():
    assert boundary_inversion({"resident": 1.0, "host-origin": 2.0}) is False
    assert boundary_inversion({"resident": 3.0, "host-origin": 2.0}) is True
    assert boundary_inversion({"resident": 1.0, "resident-graph": 2.5, "host-origin": 2.0}) is True
    assert boundary_inversion({"resident": 1.0}) is False


def test_boundary_inversion_nests_gpu_origin_between_resident_and_host_origin():
    assert boundary_inversion({"resident": 1.0, "gpu-origin": 1.5, "host-origin": 2.0}) is False
    assert boundary_inversion({"resident": 2.0, "gpu-origin": 1.5, "host-origin": 3.0}) is True
    assert boundary_inversion({"resident": 1.0, "gpu-origin": 2.5, "host-origin": 2.0}) is True
    # Each transfer policy is checked on its own group.
    medians = {
        "resident": 1.0,
        "host-origin": 3.0,
        "gpu-origin": 2.0,
        "host-origin-pinned": 1.8,
        "gpu-origin-pinned": 1.5,
    }
    assert boundary_inversion(medians) is False
    assert boundary_inversion(medians, "-pinned") is False
    medians["gpu-origin-pinned"] = 0.5
    assert boundary_inversion(medians) is False
    assert boundary_inversion(medians, "-pinned") is True


def test_default_paths_are_the_revision_3_paths():
    assert [path.label for path in build_paths(["cpu", "cuda"])] == [
        "cpu-comparator",
        "cuda-resident",
        "cuda-resident-graph",
        "cuda-host-origin",
    ]
    assert [path.label for path in build_paths(["cpu"])] == ["cpu-comparator"]
    assert BenchPath("cuda", "host-origin", "pageable").extra_args == [
        "--boundary",
        "host-origin",
    ]
    assert BenchPath("cpu", "comparator", "none").extra_args == []


def test_extension_paths_follow_a_fixed_order():
    paths = build_paths(["cpu", "cuda"], ["gpu-origin"], ["pinned", "pageable"])
    assert [path.label for path in paths] == [
        "cpu-comparator",
        "cuda-resident",
        "cuda-resident-graph",
        "cuda-host-origin",
        "cuda-host-origin-pinned",
        "cpu-gpu-origin",
        "cuda-gpu-origin",
        "cpu-gpu-origin-pinned",
        "cuda-gpu-origin-pinned",
    ]
    assert paths[-1].extra_args == ["--boundary", "gpu-origin", "--transfer-policy", "pinned"]
    assert paths[5].extra_args == ["--boundary", "gpu-origin"]


@pytest.mark.parametrize(
    ("backends", "boundaries", "policies", "message"),
    [
        (["cpu", "cuda"], [], ["pinned"], "include 'pageable'"),
        (["cpu"], ["gpu-origin"], ["pageable"], "needs the 'cuda' backend"),
        (["cpu"], [], ["pageable", "pinned"], "needs the 'cuda' backend"),
        (["cpu", "cuda"], ["device-origin"], ["pageable"], "unknown extension boundaries"),
    ],
)
def test_bad_path_selections_are_rejected(backends, boundaries, policies, message):
    with pytest.raises(ValueError, match=message):
        build_paths(backends, boundaries, policies)


@pytest.mark.parametrize("n", [5, 6, 7, 8, 9])
def test_williams_rows_balance_position_and_predecessor(n):
    rows = williams_rows(n)
    assert len(rows) == (n if n % 2 == 0 else 2 * n)
    repeats = len(rows) // n
    for row in rows:
        assert sorted(row) == list(range(n))
    for position in range(n):
        assert sorted(row[position] for row in rows) == sorted(list(range(n)) * repeats)
    pairs: dict[tuple[int, int], int] = {}
    for row in rows:
        for before, after in itertools.pairwise(row):
            pairs[(before, after)] = pairs.get((before, after), 0) + 1
    assert len(pairs) == n * (n - 1)
    assert set(pairs.values()) == {repeats}


def test_trial_design_switches_to_williams_above_four_paths():
    paths = build_paths(["cpu", "cuda"], ["gpu-origin"], ["pageable", "pinned"])
    assert trial_design(4) == "all-permutations"
    assert trial_design(len(paths)) == "williams"
    orders = trial_orders(paths, 36)
    assert orders[:18] == williams_rows(9)
    assert orders[18:] == williams_rows(9)
    with pytest.raises(ValueError, match="multiple of 18"):
        trial_orders(paths, 24)


def test_extension_rejects_bad_trials_before_touching_disk(tmp_path):
    with pytest.raises(ValueError, match="multiple of 18"):
        run_benchmark_matrix(
            root=ROOT,
            output_dir=tmp_path / "out",
            counts=[1024],
            bit_widths=[8],
            backends=["cpu", "cuda"],
            trials=24,
            boundaries=["gpu-origin"],
            transfer_policies=["pageable", "pinned"],
            readiness_facts=READY_FACTS,
        )
    assert not (tmp_path / "out").exists()


def fake_stats(median: float) -> dict:
    return {
        "median_ms": median,
        "trial_median_min_ms": median * 0.99,
        "trial_median_max_ms": median * 1.01,
        "trial_medians_ms": [median * 0.99, median, median * 1.01],
        "stable": True,
        "unstable_rev2": False,
    }


def test_case_group_uses_policy_matched_baselines_and_groups():
    paths = build_paths(["cpu", "cuda"], ["gpu-origin"], ["pageable", "pinned"])
    medians = {
        "cpu-comparator": 10.0,
        "cuda-resident": 1.0,
        "cuda-resident-graph": 0.8,
        "cuda-host-origin": 4.0,
        "cuda-host-origin-pinned": 3.0,
        "cpu-gpu-origin": 12.0,
        "cuda-gpu-origin": 2.0,
        "cpu-gpu-origin-pinned": 11.0,
        # Faster than resident: an inversion in the pinned group only.
        "cuda-gpu-origin-pinned": 0.5,
    }
    cases = [{"statistics": fake_stats(medians[path.label])} for path in paths]
    compare_case_group(cases, paths)
    stats = {path.label: case["statistics"] for path, case in zip(paths, cases, strict=True)}

    assert stats["cpu-comparator"]["verdict"] == "comparator"
    assert stats["cuda-host-origin"]["baseline"] == "cpu-comparator"
    assert stats["cpu-gpu-origin"]["baseline"] == "cpu-comparator"
    assert stats["cuda-gpu-origin"]["baseline"] == "cpu-gpu-origin"
    assert stats["cuda-gpu-origin-pinned"]["baseline"] == "cpu-gpu-origin-pinned"
    assert stats["cuda-gpu-origin"]["speedup_vs_cpu"] == 6.0
    # A CPU path that adds a download must not beat the comparator; it does not here.
    assert stats["cpu-gpu-origin"]["verdict"] == "slower"
    assert stats["cpu-gpu-origin"]["boundary_inversion"] is False

    for label in ("cuda-resident", "cuda-host-origin", "cuda-gpu-origin"):
        assert stats[label]["boundary_inversion"] is False
        assert stats[label]["direction_supported"] is True
    for label in ("cuda-host-origin-pinned", "cuda-gpu-origin-pinned"):
        assert stats[label]["boundary_inversion"] is True
        assert stats[label]["direction_supported"] is False

    vs_pageable = stats["cuda-host-origin-pinned"]["vs_pageable"]
    assert vs_pageable["speedup_vs_pageable"] == 4.0 / 3.0
    assert vs_pageable["verdict"] == "faster"
    assert vs_pageable["boundary_inversion"] is True
    assert vs_pageable["direction_supported"] is False
    assert "vs_pageable" not in stats["cuda-host-origin"]

    vs_comparator = stats["cuda-gpu-origin"]["vs_comparator"]
    assert vs_comparator["speedup_vs_comparator"] == 5.0
    assert vs_comparator["descriptive"] is True
    assert "direction_supported" not in vs_comparator
    assert stats["cuda-resident-graph"]["vs_resident"]["boundary_inversion"] is False


def test_cpu_gpu_origin_faster_than_the_comparator_is_an_inversion():
    paths = build_paths(["cpu", "cuda"], ["gpu-origin"], ["pageable"])
    medians = [10.0, 1.0, 0.8, 4.0, 9.0, 2.0]
    cases = [{"statistics": fake_stats(m)} for m in medians]
    compare_case_group(cases, paths)
    stats = {path.label: case["statistics"] for path, case in zip(paths, cases, strict=True)}
    assert stats["cpu-gpu-origin"]["boundary_inversion"] is True
    assert stats["cuda-gpu-origin"]["boundary_inversion"] is True
    assert stats["cuda-host-origin"]["boundary_inversion"] is False


def test_inverted_pageable_twin_vetoes_the_pinning_comparison():
    paths = build_paths(["cpu", "cuda"], ["gpu-origin"], ["pageable", "pinned"])
    medians = {
        "cpu-comparator": 10.0,
        "cuda-resident": 1.0,
        "cuda-resident-graph": 0.8,
        "cuda-host-origin": 4.0,
        "cuda-host-origin-pinned": 3.0,
        "cpu-gpu-origin": 12.0,
        # Faster than resident: an inversion in the pageable group only.
        "cuda-gpu-origin": 0.5,
        "cpu-gpu-origin-pinned": 11.0,
        "cuda-gpu-origin-pinned": 1.5,
    }
    cases = [{"statistics": fake_stats(medians[path.label])} for path in paths]
    compare_case_group(cases, paths)
    stats = {path.label: case["statistics"] for path, case in zip(paths, cases, strict=True)}
    assert stats["cuda-gpu-origin"]["boundary_inversion"] is True
    assert stats["cuda-gpu-origin-pinned"]["boundary_inversion"] is False
    assert stats["cuda-gpu-origin-pinned"]["direction_supported"] is True
    vs_pageable = stats["cuda-gpu-origin-pinned"]["vs_pageable"]
    assert vs_pageable["boundary_inversion"] is True
    assert vs_pageable["direction_supported"] is False


READY_FACTS = {
    "uptime_seconds": 600.0,
    "app_windows": [
        {"process": "WindowsTerminal", "title": "Terminal"},
        {"process": "TextInputHost", "title": "Windows Input Experience"},
    ],
    "launcher_processes": ["python", "uv", "just", "pwsh", "WindowsTerminal", "explorer"],
    "mco2_pids": [],
    "git_dirty_files": [],
    "gpu_clock_event_reasons": "0x0000000000000001",
    "power_plan": "Balanced",
    "hags_hwschmode": "unset",
}


def test_readiness_passes_on_a_quiet_fresh_machine():
    assert check_readiness(READY_FACTS) == []


def test_readiness_passes_with_only_shell_windows_and_no_launcher_chain():
    shell_only = {
        **READY_FACTS,
        "app_windows": [{"process": "TextInputHost", "title": "Windows Input Experience"}],
        "launcher_processes": [],
    }
    assert check_readiness(shell_only) == []


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ({"uptime_seconds": 31 * 60.0}, "uptime 31 min"),
        (
            {"app_windows": [*READY_FACTS["app_windows"], {"process": "firefox", "title": "x"}]},
            "open app window: firefox",
        ),
        # The terminal window passes only because the terminal launched the sweep.
        ({"launcher_processes": []}, "open app window: WindowsTerminal"),
        ({"mco2_pids": [4242]}, "mco2 already running (pid 4242)"),
        ({"git_dirty_files": [" M benchmark_driver.py"]}, "dirty git tree"),
        ({"gpu_clock_event_reasons": "0x0000000000000024"}, "SwPowerCap, SwThermalSlowdown"),
        ({"gpu_clock_event_reasons": None}, "clock-event reasons unavailable"),
        ({"gpu_clock_event_reasons": "[N/A]"}, "clock-event reasons unavailable"),
    ],
)
def test_readiness_names_each_failure(change, reason):
    failures = check_readiness({**READY_FACTS, **change})
    assert len(failures) == 1
    assert reason in failures[0]


def test_pilot_is_four_sizes_from_the_sweep_grid():
    assert len(PILOT_COUNTS) == 4
    assert set(PILOT_COUNTS) <= set(DEFAULT_COUNTS)


def test_affinity_mask_excludes_core_zero():
    assert affinity_mask_excluding(EXCLUDED_LOGICAL_CPUS, 0b1111_1111_1111) == 0b1111_1111_1100
    # Built from the current mask, so CPUs the process never had stay excluded.
    assert affinity_mask_excluding(EXCLUDED_LOGICAL_CPUS, 0b1010_1111) == 0b1010_1100
    with pytest.raises(RuntimeError, match="no logical CPU"):
        affinity_mask_excluding((0, 1), 0b11)


def run_cpu_snapshot(out, **kwargs):
    return run_benchmark_matrix(
        root=ROOT,
        output_dir=out,
        counts=[1024],
        bit_widths=[4],
        backends=["cpu"],
        warmups=1,
        reps=1,
        trials=1,
        in_process_warmup_seconds=0,
        allow_dirty=True,
        **kwargs,
    )


def test_failed_readiness_stops_the_sweep_before_any_process(tmp_path):
    busy = {**READY_FACTS, "uptime_seconds": 5 * 3600.0}
    with pytest.raises(RuntimeError, match="uptime 300 min"):
        run_cpu_snapshot(tmp_path / "out", readiness_facts=busy)
    assert not (tmp_path / "out").exists()


def test_readiness_override_marks_the_snapshot_non_evidence(tmp_path):
    busy = {**READY_FACTS, "mco2_pids": [4242]}
    before = get_process_affinity()
    out = run_cpu_snapshot(tmp_path / "out", readiness_facts=busy, ignore_readiness=True)
    assert get_process_affinity() == before
    conditions = json.loads((out / "manifest.json").read_text(encoding="utf-8"))["run_conditions"]
    assert conditions["evidence"] is False
    assert conditions["readiness"]["overridden"] is True
    assert "readiness check failed and was overridden" in conditions["non_evidence_reasons"]


def test_pilot_records_readiness_without_enforcing_it(tmp_path):
    busy = {**READY_FACTS, "uptime_seconds": 5 * 3600.0}
    out = run_cpu_snapshot(tmp_path / "out", readiness_facts=busy, pilot=True)
    conditions = json.loads((out / "manifest.json").read_text(encoding="utf-8"))["run_conditions"]
    assert conditions["pilot"] is True
    assert conditions["evidence"] is False
    assert conditions["readiness"]["passed"] is False
    assert conditions["readiness"]["overridden"] is False
    assert conditions["readiness"]["enforced"] is False
    assert "pilot run" in conditions["non_evidence_reasons"]


def test_failed_sweep_restores_the_affinity_mask(tmp_path, monkeypatch):
    def missing_binary(root):
        raise FileNotFoundError("no binary")

    monkeypatch.setattr(benchmark_driver, "find_binary", missing_binary)
    before = get_process_affinity()
    with pytest.raises(FileNotFoundError):
        run_cpu_snapshot(tmp_path / "out", readiness_facts=READY_FACTS)
    assert get_process_affinity() == before
