"""Contract tests for the bench-process adapter."""

import math
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from _bench_response_support import bench_payload, process_result

from stoquant.design import build_paths
from stoquant.runner import FakeBenchProcess, RealBenchProcess, run_bench_process
from stoquant.sweep import SweepOrder, SweepSettings, run_trials, warm_up_case


def test_real_process_times_out_with_a_clear_error():
    process = RealBenchProcess(timeout_seconds=0.05)
    with pytest.raises(subprocess.TimeoutExpired):
        process.run([sys.executable, "-c", "import time; time.sleep(5)"])


def test_real_process_keeps_inherited_environment_and_timeout(monkeypatch):
    calls = []

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr("stoquant.runner.subprocess.run", fake_run)
    process = RealBenchProcess(timeout_seconds=17)

    process.run(["stoquant.exe", "bench"], creationflags=9)

    argv, kwargs = calls[0]
    assert argv == ["stoquant.exe", "bench"]
    assert kwargs["timeout"] == 17
    assert kwargs["creationflags"] == 9
    assert "env" not in kwargs


def test_fake_replays_json_and_preserves_bench_argv():
    payload = bench_payload(
        count=4,
        bits=8,
        backend="cuda",
        seed=42,
        warmups=2,
        reps=3,
        extra_args=["--boundary", "resident"],
        k1="optimized",
    )
    fake = FakeBenchProcess([process_result(payload)])
    result, error = run_bench_process(
        Path("stoquant.exe"),
        Path("input.f32"),
        count=4,
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
    assert result is not None
    assert result.payload == payload
    assert result.samples_ms == [2.0, 2.0, 2.0]
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
        count=4,
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
            count=4,
            bits=8,
            backend="cuda",
            extra_args=["--k1", "optimized"],
            seed=42,
            warmups=0,
            reps=1,
            process=fake,
        )


@pytest.mark.parametrize(
    ("backend", "extra_args", "k1"),
    [
        ("cpu", [], "reference"),
        ("cpu-avx2", [], "reference"),
        ("cpu-avx2", ["--threads", "3"], "reference"),
        ("cpu", ["--boundary", "gpu-origin"], "reference"),
        ("cuda", ["--boundary", "resident"], "reference"),
        ("cuda", ["--boundary", "resident-graph"], "optimized"),
        ("cuda", ["--boundary", "host-origin"], "reference"),
        ("cuda", ["--boundary", "host-origin", "--transfer-policy", "pinned"], "optimized"),
        ("cuda", ["--boundary", "gpu-origin", "--transfer-policy", "pinned"], "reference"),
    ],
)
def test_valid_response_variants_pass_with_zero_timings(backend, extra_args, k1):
    payload = bench_payload(
        count=5,
        bits=4,
        backend=backend,
        seed=42,
        warmups=1,
        reps=2,
        extra_args=extra_args,
        k1=k1,
        sample=0,
    )
    fake = FakeBenchProcess([process_result(payload)])

    response, error = run_bench_process(
        Path("stoquant.exe"),
        Path("input.f32"),
        count=5,
        bits=4,
        backend=backend,
        extra_args=extra_args,
        seed=42,
        warmups=1,
        reps=2,
        k1=k1,
        process=fake,
    )

    assert error is None
    assert response is not None
    assert response.payload == payload
    if backend == "cuda" and "resident-graph" in extra_args:
        assert response.configuration["repetition_invocation_ids"] == [0, 0]
        assert response.configuration["warmup_invocation_ids"] == [0]
        assert response.capture_and_instantiate_ms == 0


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda payload: payload["samples_ms"].pop(), "samples_ms has 1 samples; expected 2"),
        (lambda payload: payload["samples_ms"].append(2.0), "samples_ms has 3 samples; expected 2"),
        (
            lambda payload: payload["samples_ms"].__setitem__(0, True),
            "samples_ms[0] must be a number",
        ),
        (
            lambda payload: payload["samples_ms"].__setitem__(0, "2"),
            "samples_ms[0] must be a number",
        ),
        (
            lambda payload: payload["samples_ms"].__setitem__(0, math.nan),
            "must be finite and nonnegative",
        ),
        (
            lambda payload: payload["samples_ms"].__setitem__(0, math.inf),
            "must be finite and nonnegative",
        ),
        (
            lambda payload: payload["samples_ms"].__setitem__(0, -0.1),
            "must be finite and nonnegative",
        ),
    ],
)
def test_invalid_wall_samples_fail_at_public_runner_boundary(mutation, message):
    payload = bench_payload(count=4, bits=8, backend="cpu", seed=42, warmups=0, reps=2)
    mutation(payload)
    response, error = run_bench_process(
        Path("stoquant.exe"),
        Path("input.f32"),
        count=4,
        bits=8,
        backend="cpu",
        extra_args=[],
        seed=42,
        warmups=0,
        reps=2,
        process=FakeBenchProcess([process_result(payload)]),
    )

    assert response is None
    assert error is not None and message in error


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda payload: payload.pop("k2_ms"), "missing"),
        (lambda payload: payload["k1_ms"].pop(), "k1_ms has 1 samples; expected 2"),
        (lambda payload: payload["k3_ms"].__setitem__(0, "fast"), "k3_ms[0] must be a number"),
        (lambda payload: payload["k1_ms"].__setitem__(0, -1), "finite and nonnegative"),
        (lambda payload: payload.__setitem__("cpu_ms", [0.0, 0.0]), "unexpected stage arrays"),
    ],
)
def test_invalid_stage_responses_fail_at_public_runner_boundary(mutation, message):
    payload = bench_payload(
        count=4,
        bits=8,
        backend="cuda",
        seed=42,
        warmups=1,
        reps=2,
        extra_args=["--boundary", "resident"],
    )
    mutation(payload)
    response, error = run_bench_process(
        Path("stoquant.exe"),
        Path("input.f32"),
        count=4,
        bits=8,
        backend="cuda",
        extra_args=["--boundary", "resident"],
        seed=42,
        warmups=1,
        reps=2,
        process=FakeBenchProcess([process_result(payload)]),
    )

    assert response is None
    assert error is not None and message in error


@pytest.mark.parametrize(
    ("field", "wrong"),
    [
        ("backend", "cpu-avx2"),
        ("bits", 4),
        ("count", 5),
        ("seed", 9),
        ("warmup", 3),
        ("transfer_policy", "pinned"),
    ],
)
def test_wrong_echoed_configuration_fails(field, wrong):
    payload = bench_payload(count=4, bits=8, backend="cpu", seed=42, warmups=1, reps=2)
    payload["configuration"][field] = wrong
    response, error = run_bench_process(
        Path("stoquant.exe"),
        Path("input.f32"),
        count=4,
        bits=8,
        backend="cpu",
        extra_args=[],
        seed=42,
        warmups=1,
        reps=2,
        process=FakeBenchProcess([process_result(payload)]),
    )

    assert response is None
    assert error is not None and f"configuration {field}" in error


@pytest.mark.parametrize("field", ["repetition_invocation_ids", "warmup_invocation_ids"])
def test_wrong_invocation_identifiers_fail(field):
    payload = bench_payload(
        count=4,
        bits=8,
        backend="cuda",
        seed=42,
        warmups=1,
        reps=2,
        extra_args=["--boundary", "resident-graph"],
    )
    payload["configuration"][field] = [1, 1] if field == "repetition_invocation_ids" else [1]
    response, error = run_bench_process(
        Path("stoquant.exe"),
        Path("input.f32"),
        count=4,
        bits=8,
        backend="cuda",
        extra_args=["--boundary", "resident-graph"],
        seed=42,
        warmups=1,
        reps=2,
        process=FakeBenchProcess([process_result(payload)]),
    )

    assert response is None
    assert error is not None and f"configuration {field}" in error


def test_graph_capture_is_required_and_validated():
    payload = bench_payload(
        count=4,
        bits=8,
        backend="cuda",
        seed=42,
        warmups=0,
        reps=1,
        extra_args=["--boundary", "resident-graph"],
    )
    payload["capture_and_instantiate_ms"] = -1
    response, error = run_bench_process(
        Path("stoquant.exe"),
        Path("input.f32"),
        count=4,
        bits=8,
        backend="cuda",
        extra_args=["--boundary", "resident-graph"],
        seed=42,
        warmups=0,
        reps=1,
        process=FakeBenchProcess([process_result(payload)]),
    )

    assert response is None
    assert error is not None and "capture_and_instantiate_ms" in error


def test_missing_graph_capture_is_rejected():
    payload = bench_payload(
        count=4,
        bits=8,
        backend="cuda",
        seed=42,
        warmups=0,
        reps=1,
        extra_args=["--boundary", "resident-graph"],
    )
    payload.pop("capture_and_instantiate_ms")

    response, error = run_bench_process(
        Path("stoquant.exe"),
        Path("input.f32"),
        count=4,
        bits=8,
        backend="cuda",
        extra_args=["--boundary", "resident-graph"],
        seed=42,
        warmups=0,
        reps=1,
        process=FakeBenchProcess([process_result(payload)]),
    )

    assert response is None
    assert error is not None and "missing capture_and_instantiate_ms" in error


def test_avx2_default_thread_count_must_be_present_and_positive():
    payload = bench_payload(count=4, bits=8, backend="cpu-avx2", seed=42, warmups=0, reps=1)
    payload["configuration"].pop("threads")

    response, error = run_bench_process(
        Path("stoquant.exe"),
        Path("input.f32"),
        count=4,
        bits=8,
        backend="cpu-avx2",
        extra_args=[],
        seed=42,
        warmups=0,
        reps=1,
        process=FakeBenchProcess([process_result(payload)]),
    )

    assert response is None
    assert error is not None and "threads must be a positive integer" in error


def test_avx2_explicit_thread_count_must_match_request():
    extra_args = ["--threads", "3"]
    payload = bench_payload(
        count=4,
        bits=8,
        backend="cpu-avx2",
        seed=42,
        warmups=0,
        reps=1,
        extra_args=extra_args,
    )
    payload["configuration"]["threads"] = 4

    response, error = run_bench_process(
        Path("stoquant.exe"),
        Path("input.f32"),
        count=4,
        bits=8,
        backend="cpu-avx2",
        extra_args=extra_args,
        seed=42,
        warmups=0,
        reps=1,
        process=FakeBenchProcess([process_result(payload)]),
    )

    assert response is None
    assert error is not None and "configuration threads is 4; expected 3" in error


def test_bad_calibration_probe_fails_only_its_case():
    fake = FakeBenchProcess(
        [
            process_result(
                bench_payload(count=8, bits=8, backend="cpu", seed=42, warmups=1, reps=1)
                | {"samples_ms": []}
            ),
            process_result(
                bench_payload(count=8, bits=8, backend="cpu-avx2", seed=42, warmups=1, reps=1)
            ),
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
    cases = [{"correctness": dict(gate), "count": 8} for _ in range(2)]
    warm_up_case(cases, Path("stoquant.exe"), Path("input.f32"), 8, cast(SweepSettings, settings))
    assert cases[0]["correctness"]["status"] == "failed"
    assert cases[0]["correctness"]["gate_status"] == "passed"
    assert cases[0]["correctness"]["byte_identical_to_compress"] is True
    assert cases[0]["correctness"]["record_sha256"] == {"cpu": "abc123"}
    assert cases[0]["correctness"]["seed"] == 42
    assert "samples_ms has 0 samples" in cases[0]["correctness"]["error_message"]
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
        "count": 8,
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
