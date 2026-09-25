import csv
import json
import os
from pathlib import Path

import pytest

from benchmark_driver import (
    DEFAULT_COUNTS,
    run_benchmark_matrix,
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
        allow_existing=True,
    )
    assert result_dir == snapshot_dir

    manifest_path = snapshot_dir / "manifest.json"
    assert manifest_path.is_file()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["manifest_version"] == "1.0"
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
    snapshot_dir = tmp_path / "test-snapshot"
    result_dir = run_benchmark_matrix(
        root=ROOT,
        output_dir=snapshot_dir,
        counts=[1024],
        bit_widths=[4, 8],
        backends=["cpu", "cuda"],
        warmups=1,
        reps=2,
        allow_existing=True,
    )
    assert result_dir == snapshot_dir

    # 1. Manifest
    manifest_path = snapshot_dir / "manifest.json"
    assert manifest_path.is_file()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["manifest_version"] == "1.0"
    assert "date" in manifest
    assert "created_at_utc" in manifest
    assert "git_provenance" in manifest
    for key in ("code_revision", "code_revision_short", "git_dirty"):
        assert key in manifest["git_provenance"]
    assert "hardware" in manifest
    for key in ("gpu_name", "compute_capability", "host_cpu"):
        assert key in manifest["hardware"]
    assert "toolkit_and_driver" in manifest
    for key in ("cuda_toolkit", "driver_version", "c_compiler"):
        assert key in manifest["toolkit_and_driver"]
    assert "build_flags" in manifest
    for key in ("cpu", "cuda_host", "cuda_nvcc"):
        assert key in manifest["build_flags"]
    assert manifest["transfer_policy"] == "pageable"

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
    assert len(vec_report_path.read_text(encoding="utf-8")) > 0

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
        assert len(case_data["repetition_invocation_ids"]) == 2
        assert len(case_data["warmup_invocation_ids"]) == 1
        assert case_data["header_bytes"] == 20
        assert case_data["payload_bytes"] == (1024 if case_data["bits"] == 8 else 512)
        assert case_data["correctness"]["status"] == "passed"
        assert case_data["correctness"]["byte_identical_to_compress"] is True
        assert case_data["correctness"]["cpu_cuda_byte_identical"] is True
        assert case_data["correctness"]["layer2_scale_within_bound"] is True
        assert case_data["correctness"]["layer2_reconstruction_within_bound"] is True
        assert "_path" not in case_data["input_provenance"]

        assert len(case_data["samples_ms"]) == 2
        assert all(sample >= 0 for sample in case_data["samples_ms"])
        if case_data["backend"] == "cuda":
            for k in ("k1_ms", "k2_ms", "k3_ms"):
                assert len(case_data[k]) == 2
                assert all(s >= 0 for s in case_data[k])

        stats = case_data["statistics"]
        assert stats is not None
        assert stats["median_ms"] >= 0
        assert stats["iqr_ms"] >= 0
        assert stats["speedup_vs_cpu"] >= 0
        if case_data["backend"] == "cpu":
            assert stats["speedup_vs_cpu"] == 1.0

    # 3. Summary CSV
    csv_path = snapshot_dir / manifest["summary_csv"]
    assert csv_path.is_file()
    with csv_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == 6
    expected_fields = [
        "count",
        "bits",
        "backend",
        "boundary",
        "correctness",
        "warmup",
        "reps",
        "median_ms",
        "iqr_ms",
        "speedup_vs_c",
    ]
    assert reader.fieldnames == expected_fields
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
        force_fail=True,
        allow_existing=True,
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
        )


def test_full_matrix_counts_never_run_in_tests():
    # Verify that test suite does not include the full matrix element counts
    # (2^14, 2^18, 2^22), only 2^10 is used for testing driver.
    assert DEFAULT_COUNTS == (1024, 16384, 262144, 4194304)
