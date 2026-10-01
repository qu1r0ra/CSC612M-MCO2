"""Contract tests for the bench-process adapter."""

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from stoquant.design import build_paths
from stoquant.runner import FakeBenchProcess, RealBenchProcess, run_bench_process
from stoquant.sweep import SweepOrder, SweepSettings, run_trials, warm_up_case


def test_real_process_times_out_with_a_clear_error():
    process = RealBenchProcess(timeout_seconds=0.05)
    with pytest.raises(subprocess.TimeoutExpired):
        process.run([sys.executable, "-c", "import time; time.sleep(5)"])


def test_fake_replays_json_and_preserves_bench_argv():
    payload = {"samples_ms": [1.0], "configuration": {"k1": "optimized"}}
    fake = FakeBenchProcess([subprocess.CompletedProcess([], 0, json.dumps(payload), "")])
    result, error = run_bench_process(
        Path("stoquant.exe"),
        Path("input.f32"),
        bits=8,
        backend="cuda",
        extra_args=["--boundary", "resident"],
        seed=42,
        warmups=2,
        reps=3,
        k1="optimized",
        process=fake,
    )
    assert error is None
    assert result == payload
    assert fake.commands == [
        [
            "stoquant.exe",
            "bench",
            "--input",
            "input.f32",
            "--seed",
            "42",
            "--bits",
            "8",
            "--tensor-id",
            "0",
            "--invocation-id",
            "0",
            "--backend",
            "cuda",
            "--k1",
            "optimized",
            "--warmup",
            "2",
            "--reps",
            "3",
            "--boundary",
            "resident",
        ]
    ]


def test_invalid_json_and_duplicate_k1_fail_closed():
    fake = FakeBenchProcess([subprocess.CompletedProcess([], 0, "not JSON", "")])
    payload, error = run_bench_process(
        Path("stoquant.exe"),
        Path("input.f32"),
        bits=8,
        backend="cuda",
        extra_args=[],
        seed=42,
        warmups=0,
        reps=1,
        process=fake,
    )
    assert payload is None
    assert error == "stoquant bench returned invalid JSON"
    with pytest.raises(ValueError, match="K1 variant"):
        run_bench_process(
            Path("stoquant.exe"),
            Path("input.f32"),
            bits=8,
            backend="cuda",
            extra_args=["--k1", "optimized"],
            seed=42,
            warmups=0,
            reps=1,
            process=fake,
        )


def test_bad_calibration_probe_fails_only_its_case():
    fake = FakeBenchProcess(
        [
            subprocess.CompletedProcess([], 0, '{"samples_ms": []}', ""),
            subprocess.CompletedProcess([], 0, '{"samples_ms": [2.0]}', ""),
        ]
    )
    settings = SimpleNamespace(
        cuda=False,
        case_warmup_seconds=0,
        in_process_warmup_seconds=1.0,
        warmups=1,
        reps=1,
        compression_seed=42,
        k1="reference",
        paths=build_paths(["cpu", "cpu-avx2"]),
        bench_process=fake,
    )
    gate = {
        "status": "passed",
        "byte_identical_to_compress": True,
        "record_sha256": {"cpu": "abc123"},
        "seed": 42,
    }
    cases = [{"correctness": dict(gate)} for _ in range(2)]
    warm_up_case(cases, Path("stoquant.exe"), Path("input.f32"), 8, cast(SweepSettings, settings))
    assert cases[0]["correctness"]["status"] == "failed"
    assert cases[0]["correctness"]["gate_status"] == "passed"
    assert cases[0]["correctness"]["byte_identical_to_compress"] is True
    assert cases[0]["correctness"]["record_sha256"] == {"cpu": "abc123"}
    assert cases[0]["correctness"]["seed"] == 42
    assert "empty or non-finite" in cases[0]["correctness"]["error_message"]
    assert cases[1]["correctness"]["status"] == "passed"
    assert cases[1]["warmup"] == 500


def test_trial_failure_preserves_the_complete_correctness_gate():
    gate = {
        "status": "passed",
        "byte_identical_to_compress": True,
        "records": [{"backend": "cpu", "sha256": "abc123"}],
        "oracle": {"version": 2},
        "error_message": "correctness metadata detail",
    }
    case = {
        "case_id": "case_cpu_comparator_bits4_n1024",
        "correctness": dict(gate),
        "warmup": 1,
        "trial_runs": [],
    }
    settings = SimpleNamespace(
        k1="reference",
        compression_seed=42,
        reps=1,
        paths=build_paths(["cpu"]),
        bench_process=FakeBenchProcess(
            [subprocess.CompletedProcess([], 1, "", "native trial failed")]
        ),
    )
    order = SweepOrder([[0]], [["cpu-comparator"]], [], None, None)

    run_trials(
        [case],
        Path("stoquant.exe"),
        Path("input.f32"),
        4,
        order,
        cast(SweepSettings, settings),
    )

    assert case["correctness"]["status"] == "failed"
    assert case["correctness"]["gate_status"] == "passed"
    assert case["correctness"]["byte_identical_to_compress"] is True
    assert case["correctness"]["records"] == gate["records"]
    assert case["correctness"]["oracle"] == gate["oracle"]
    assert case["correctness"]["gate_error_message"] == gate["error_message"]
    assert case["correctness"]["failure_phase"] == "trial"
    assert case["correctness"]["error_message"] == "stoquant bench failed: native trial failed"
