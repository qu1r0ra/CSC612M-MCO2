import csv
import json
import subprocess

import pytest
from _bench_test_support import CUDA_TEST, READY_FACTS, ROOT

from stoquant.design import (
    DEFAULT_COUNTS,
    DEFAULT_TRIALS,
    build_paths,
    case_order,
    trial_orders,
)
from stoquant.host import (
    EXCLUDED_LOGICAL_CPUS,
)
from stoquant.matrix import run_benchmark_matrix
from stoquant.runner import FakeBenchProcess, in_process_warmups
from stoquant.schema import SUMMARY_FIELDS
from stoquant.stats import (
    BOOTSTRAP_SEED,
)
from stoquant.sweep import (
    compact_invocation_ids,
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
        allow_dirty=True,
        readiness_facts=READY_FACTS,
    )
    assert result_dir == snapshot_dir

    manifest_path = snapshot_dir / "manifest.json"
    assert manifest_path.is_file()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["manifest_version"] == "3.3"
    assert manifest["build_stamp"]["recipe"] in ("build-cpu", "build-cuda")
    assert manifest["build_stamp"]["binary_sha256"]
    assert manifest["transfer_policies"] == ["pageable"]
    assert manifest["matrix_parameters"]["paths"] == ["cpu-comparator"]
    assert manifest["matrix_parameters"]["k1"] == "reference"
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


def test_driver_rejects_unknown_k1_variant(tmp_path):
    with pytest.raises(ValueError, match="unknown K1 variant"):
        run_benchmark_matrix(root=ROOT, output_dir=tmp_path / "invalid", k1="other")


@CUDA_TEST
def test_optimized_k1_reaches_all_cuda_paths_and_is_recorded(tmp_path):
    snapshot = run_benchmark_matrix(
        root=ROOT,
        output_dir=tmp_path / "optimized-k1",
        counts=[1024],
        bit_widths=[8],
        backends=["cpu", "cpu-avx2", "cuda"],
        boundaries=["gpu-origin"],
        transfer_policies=["pageable", "pinned"],
        warmups=1,
        reps=2,
        trials=10,
        gpu_warmup_seconds=0,
        case_warmup_seconds=0,
        in_process_warmup_seconds=0,
        allow_dirty=True,
        readiness_facts=READY_FACTS,
        k1="optimized",
    )
    manifest = json.loads((snapshot / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["matrix_parameters"]["k1"] == "optimized"
    assert len(manifest["matrix_parameters"]["paths"]) == 10
    assert manifest["all_cases_passed"] is True
    for case_id in manifest["cases"]:
        case = json.loads((snapshot / f"{case_id}.json").read_text(encoding="utf-8"))
        expected = "optimized" if case["backend"] == "cuda" else None
        assert case["configuration"]["k1"] == expected


@CUDA_TEST
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
        allow_dirty=True,
        readiness_facts=READY_FACTS,
    )
    assert result_dir == snapshot_dir

    # 1. Manifest
    manifest_path = snapshot_dir / "manifest.json"
    assert manifest_path.is_file()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["manifest_version"] == "3.3"
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
    assert "/DSQ_ENABLE_CUDA" in manifest["build_flags"]["comparator_c"]
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
    assert params["k1"] == "reference"
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
    assert "/DSQ_ENABLE_CUDA" in vec_report
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
        assert case_data["configuration"]["k1"] == (
            "reference" if case_data["backend"] == "cuda" else None
        )
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


@CUDA_TEST
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


@CUDA_TEST
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
        bench_process=FakeBenchProcess(
            [subprocess.CompletedProcess(["stoquant", "compress"], 1, "", "forced failure")]
        ),
        allow_dirty=True,
        readiness_facts=READY_FACTS,
    )
    assert result_dir == snapshot_dir

    manifest = json.loads((snapshot_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["all_cases_passed"] is False
    assert manifest["run_conditions"]["evidence"] is False
    assert any(
        reason.startswith("failed benchmark cases:")
        for reason in manifest["run_conditions"]["non_evidence_reasons"]
    )

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


@pytest.mark.parametrize(
    ("family", "extra", "keys"),
    [
        ("sparse", {"counts": [1024]}, ["sparse_n1024"]),
        ("model", {"counts": [], "model_limit": 2}, ["model_conv1.weight", "model_bn1.weight"]),
    ],
)
def test_driver_cpu_only_family_snapshot(tmp_path, family, extra, keys):
    snapshot_dir = tmp_path / f"{family}-snapshot"
    run_benchmark_matrix(
        root=ROOT,
        output_dir=snapshot_dir,
        bit_widths=[8],
        backends=["cpu"],
        warmups=1,
        reps=2,
        trials=2,
        in_process_warmup_seconds=0,
        allow_dirty=True,
        readiness_facts=READY_FACTS,
        input_family=family,
        **extra,
    )
    manifest = json.loads((snapshot_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["all_cases_passed"] is True
    assert manifest["matrix_parameters"]["input_family"] == family
    assert sorted(manifest["inputs"]) == sorted(keys)
    assert sorted(manifest["cases"]) == sorted(f"case_cpu_comparator_bits8_{key}" for key in keys)
    assert all(entry["input"] in keys for entry in manifest["matrix_parameters"]["case_order"])
    for case_id in manifest["cases"]:
        case_data = json.loads((snapshot_dir / f"{case_id}.json").read_text(encoding="utf-8"))
        assert case_data["input_family"] == family
        assert case_data["input_provenance"]["input_key"] == case_data["input_key"]
    with (snapshot_dir / manifest["summary_csv"]).open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert sorted(row["input_key"] for row in rows) == sorted(keys)
    assert {row["input_family"] for row in rows} == {family}
