import json

import pytest
from _bench_test_support import ROOT

from stoquant import cli
from stoquant.runner import REAL_BENCH_PROCESS
from stoquant.sweep import (
    SweepInputs,
    SweepSettings,
    write_case_outputs,
)


def test_passing_unbaselined_case_keeps_measured_summary_values(tmp_path):
    case = {
        "case_id": "candidate-case",
        "backend": "cuda",
        "timing_boundary": "resident",
        "transfer_policy": "none",
        "path_label": "cuda-resident",
        "warmup": 1,
        "correctness": {"status": "passed"},
        "statistics": {
            "median_ms": 1.25,
            "iqr_ms": 0.2,
            "trial_median_min_ms": 1.1,
            "trial_median_max_ms": 1.4,
            "spread_ratio": 1.27,
            "spread_p90_p10": 1.1,
            "stable": True,
            "unstable_rev2": True,
        },
    }

    rows = write_case_outputs(
        tmp_path,
        [case],
        SweepInputs(files={}, clean_meta={}, counts={"n1024": 1024}),
        "n1024",
        4,
        SweepSettings(
            root=ROOT,
            target_dir=tmp_path,
            date_str="2026-09-30",
            git_prov={},
            run_conditions={},
            counts=(1024,),
            bit_widths=(4,),
            backends=("cuda",),
            paths=(),
            transfer_policies=(),
            warmups=1,
            reps=2,
            trials=1,
            input_seed=0,
            compression_seed=0,
            case_order_seed=0,
            gpu_warmup_seconds=0,
            case_warmup_seconds=0,
            in_process_warmup_seconds=0,
            bench_process=REAL_BENCH_PROCESS,
            input_family="dense",
            model_tensors=None,
            model_limit=None,
            k1="reference",
            monitoring=None,
        ),
    )

    assert len(rows) == 1
    row = rows[0]
    assert row is not None
    assert row["correctness"] == "passed"
    assert row["median_ms"] == "1.250000"
    assert row["baseline"] == ""
    assert row["verdict"] == ""


def test_matrix_cli_returns_failure_for_non_evidence_snapshot(tmp_path, monkeypatch, capsys):
    snapshot = tmp_path / "failed-snapshot"
    snapshot.mkdir()
    (snapshot / "manifest.json").write_text(
        json.dumps(
            {
                "all_cases_passed": False,
                "run_conditions": {"non_evidence_reasons": ["failed benchmark cases: case-1"]},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(cli, "run_benchmark_matrix", lambda **_: snapshot)

    with pytest.raises(SystemExit) as error:
        cli.main(["--output-dir", str(snapshot)])

    assert error.value.code == 1
    assert "snapshot is non-evidence" in capsys.readouterr().err
