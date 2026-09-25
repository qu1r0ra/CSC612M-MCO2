import csv
import json
import os
import subprocess
from pathlib import Path

import pytest

from benchmark_driver import (
    DEFAULT_COUNTS,
    DEFAULT_TRIALS,
    SUMMARY_FIELDS,
    compare_to_comparator,
    compute_case_statistics,
    run_benchmark_matrix,
    trial_orders,
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
        allow_existing=True,
        allow_dirty=True,
    )
    assert result_dir == snapshot_dir

    manifest_path = snapshot_dir / "manifest.json"
    assert manifest_path.is_file()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["manifest_version"] == "2.0"
    assert manifest["transfer_policy"] == "pageable"
    assert len(manifest["cases"]) == 2  # 1 size * 2 bit widths * 1 CPU path
    assert manifest["all_cases_passed"] is True

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
        counts=[1024],
        bit_widths=[4, 8],
        backends=["cpu", "cuda"],
        warmups=1,
        reps=2,
        trials=2,
        allow_existing=True,
        allow_dirty=True,
    )
    assert result_dir == snapshot_dir

    # 1. Manifest
    manifest_path = snapshot_dir / "manifest.json"
    assert manifest_path.is_file()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["manifest_version"] == "2.0"
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
    assert manifest["transfer_policy"] == "pageable"
    assert set(manifest["gpu_state"]) == {"note", "start", "end"}
    params = manifest["matrix_parameters"]
    assert params["trials"] == 2
    assert len(params["trial_orders"]) == 2
    assert params["trial_orders"][0] != params["trial_orders"][1]
    method = manifest["statistics_method"]
    assert method["spread_threshold"] == 1.25
    assert "linear" in method["quantile_method"]

    # Inputs provenance
    assert "1024" in manifest["inputs"]
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
    # 1 size (1024) * 2 bits (4, 8) * 3 paths (cpu, cuda-resident, cuda-host-origin) = 6 cases
    assert len(manifest["cases"]) == 6
    assert manifest["all_cases_passed"] is True

    # 2. Case JSONs
    for case_id in manifest["cases"]:
        case_file = snapshot_dir / f"{case_id}.json"
        assert case_file.is_file()
        case_data = json.loads(case_file.read_text(encoding="utf-8"))

        assert case_data["case_id"] == case_id
        assert case_data["count"] == 1024
        assert case_data["bits"] in (4, 8)
        assert case_data["backend"] in ("cpu", "cuda")
        assert case_data["timing_boundary"] in ("comparator", "resident", "host-origin")
        assert case_data["transfer_policy"] == "pageable"
        assert case_data["warmup"] == 1
        assert case_data["reps"] == 2
        assert case_data["trials"] == 2
        assert case_data["repetition_invocation_ids"] == [0, 1]
        assert case_data["warmup_invocation_ids"] == [2]
        assert len(case_data["warmup_invocation_ids"]) == 1
        assert case_data["header_bytes"] == 20
        assert case_data["payload_bytes"] == (1024 if case_data["bits"] == 8 else 512)
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
            assert run["repetition_invocation_ids"] == [0, 1]
            if case_data["backend"] == "cuda":
                for k in ("k1_ms", "k2_ms", "k3_ms"):
                    assert len(run[k]) == 2
                    assert all(s >= 0 for s in run[k])

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
            assert isinstance(stats["claim_supported"], bool)

    # 3. Summary CSV
    csv_path = snapshot_dir / manifest["summary_csv"]
    assert csv_path.is_file()
    with csv_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == 6
    assert reader.fieldnames == SUMMARY_FIELDS
    for row in rows:
        assert row["count"] == "1024"
        assert row["bits"] in ("4", "8")
        assert row["correctness"] == "passed"
        assert float(row["median_ms"]) >= 0
        assert float(row["iqr_ms"]) >= 0
        assert float(row["speedup_vs_c"]) >= 0


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
        force_fail=True,
        allow_existing=True,
        allow_dirty=True,
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

    assert len(rows) == 3  # 1 size * 1 bit width * 3 paths
    for row in rows:
        assert row["correctness"] == "failed"
        assert row["median_ms"] == ""
        assert row["iqr_ms"] == ""
        assert row["speedup_vs_c"] == ""
        assert row["verdict"] == ""
        assert row["claim_supported"] == ""


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


def test_full_matrix_counts_never_run_in_tests():
    # Verify that test suite does not include the full matrix element counts
    # (2^14, 2^18, 2^22), only 2^10 is used for testing driver.
    assert DEFAULT_COUNTS == (1024, 16384, 262144, 4194304)


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
    paths = [("cpu", "comparator", []), ("cuda", "resident", []), ("cuda", "host-origin", [])]
    orders = trial_orders(paths, DEFAULT_TRIALS)
    assert DEFAULT_TRIALS == 6
    assert len({tuple(order) for order in orders}) == 6
    for position in range(3):
        assert sorted(order[position] for order in orders) == [0, 0, 1, 1, 2, 2]


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
    assert cpu["unstable"] is False
    assert overlap["unstable"] is True  # 12 / 9 > 1.25
