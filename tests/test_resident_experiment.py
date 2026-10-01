import subprocess
from contextlib import contextmanager
from pathlib import Path

import pytest
from _bench_response_support import bench_payload_for_argv, process_result

from stoquant import k1_ab, k1_bandwidth, resident_experiment
from stoquant.provenance import BuildCommands, VerifiedBinary


class FakeResidentProcess:
    def __init__(self, malformed_reps=None):
        self.commands = []
        self.malformed_reps = malformed_reps

    def run(self, argv, *, creationflags=0):
        command = list(argv)
        self.commands.append(command)
        if command[1] == "compress":
            output = Path(command[command.index("--output") + 1])
            output.write_bytes(b"same record for each K1 variant")
            return subprocess.CompletedProcess(command, 0, "", "")
        if command[1] == "bench":
            variant = command[command.index("--k1") + 1]
            payload = bench_payload_for_argv(command)
            level = 2.0 if variant == "reference" else 1.0
            payload["samples_ms"] = [level] * len(payload["samples_ms"])
            for key in ("k1_ms", "k2_ms", "k3_ms"):
                if key in payload:
                    payload[key] = [level / 2] * len(payload[key])
            reps = int(command[command.index("--reps") + 1])
            if reps == self.malformed_reps:
                payload["samples_ms"].pop()
            return process_result(payload)
        raise AssertionError(f"unexpected fake process command: {command!r}")


@pytest.mark.parametrize(
    ("experiment", "malformed_reps"),
    [("baseline", None), ("ab", None), ("baseline", 10), ("baseline", 3)],
)
def test_k1_experiments_use_shared_runner_and_fake_process_adapter(
    experiment, malformed_reps, tmp_path, monkeypatch
):
    root = tmp_path / "repo"
    binary = root / "build" / "stoquant.exe"
    probe_binary = root / "build" / "stream_probe.exe"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"fake binary")
    probe_binary.write_bytes(b"fake probe")
    git = {
        "git_dirty": False,
        "dirty_files": [],
        "code_revision_short": "test123",
        "code_revision": "0" * 40,
        "tree_fingerprint": "clean tree",
    }
    probe = {
        "device": {
            "memory_clock_khz": 14_001_000,
            "bus_width_bits": 128,
            "l2_bytes": 1024,
        },
        "sizes": [{"count": 1 << 18, "bytes": 1 << 20, "best_gbps": 100.0}],
    }
    affinity_state = {
        "mask": 0b1100,
        "previous_mask": 0b1111,
        "excluded_logical_cpus": (0, 1),
    }

    @contextmanager
    def fake_affinity():
        yield affinity_state

    monkeypatch.setattr(resident_experiment, "benchmark_process_affinity", fake_affinity)
    monkeypatch.setattr(resident_experiment, "collect_git_provenance", lambda _: git)
    monkeypatch.setattr(
        resident_experiment,
        "find_binary",
        lambda *_args, **_kwargs: VerifiedBinary(binary, {"verified": True}),
    )
    monkeypatch.setattr(
        resident_experiment,
        "collect_build_commands",
        lambda _: BuildCommands(
            source="test",
            commands=["just build-cuda"],
            comparator_c="unknown",
            avx2_c="unknown",
            cuda_nvcc="unknown",
            host_tokens=[],
            avx2_tokens=None,
        ),
    )
    monkeypatch.setattr(
        resident_experiment,
        "verify_stamp",
        lambda *_args, **_kwargs: {
            "recipe": "build-stream-probe",
            "commands": [
                "nvcc -O2 --Werror all-warnings native/stream_probe.cu -o build/stream_probe.exe"
            ],
        },
    )
    monkeypatch.setattr(resident_experiment, "file_sha256", lambda _: "a" * 64)
    monkeypatch.setattr(resident_experiment, "probe_path", lambda _: probe_binary)
    monkeypatch.setattr(resident_experiment, "probe_readiness_facts", lambda *_: {"ready": True})
    monkeypatch.setattr(resident_experiment, "query_gpu_state", lambda: {"pstate": "P0"})
    monkeypatch.setattr(resident_experiment, "run_probe", lambda _: probe)
    monkeypatch.setattr(resident_experiment, "collect_hardware_and_toolchain", lambda _: {})
    warmups = []

    def fake_gpu_warmup(*args, **kwargs):
        warmups.append((args, kwargs))
        return {"seconds_requested": args[2], "processes": 0}

    monkeypatch.setattr(resident_experiment, "warm_up_gpu", fake_gpu_warmup)
    analyzed = []

    def analyze(run, output):
        analyzed.append(run)
        (output / "analysis.txt").write_text("analyzed", encoding="utf-8")

    if experiment == "baseline":
        monkeypatch.setattr(k1_bandwidth, "analyze_baseline", analyze)
        run_experiment = k1_bandwidth.run_k1_baseline
        variants = ("reference",)
    else:
        monkeypatch.setattr(k1_ab, "analyze_k1_ab", analyze)
        run_experiment = k1_ab.run_k1_ab
        variants = k1_ab.VARIANTS

    output = root / "results" / f"{experiment}-snapshot"
    config = resident_experiment.ResidentExperimentConfig(
        root=root,
        kind=f"k1-{experiment}",
        counts=(1024,),
        bits=(4,),
        processes=2,
        reps=3,
        gpu_warmup_seconds=0.0,
        in_process_warmup_seconds=0.0,
        output_dir=output,
        record_readiness=experiment == "ab",
    )
    adapter = FakeResidentProcess(malformed_reps)

    if malformed_reps is not None:
        expected_count = malformed_reps - 1
        with pytest.raises(
            RuntimeError,
            match=f"samples_ms has {expected_count} samples; expected {malformed_reps}",
        ):
            run_experiment(config, adapter=adapter)
        assert not output.exists()
        assert (output.with_name(output.name + ".partial")).is_dir()
        return

    result = run_experiment(config, adapter=adapter)

    assert result == output
    assert output.is_dir()
    assert (output / "analysis.txt").read_text(encoding="utf-8") == "analyzed"
    assert len(analyzed) == 1
    run = analyzed[0]
    assert run.affinity_mask == affinity_state["mask"]
    assert run.excluded_logical_cpus == affinity_state["excluded_logical_cpus"]
    assert run.readiness == ({"ready": True} if experiment == "ab" else None)
    assert len(warmups) == 1
    assert len(run.records) == 2 * len(variants)
    measured = [
        command
        for command in adapter.commands
        if command[1] == "bench" and int(command[command.index("--reps") + 1]) == config.reps
    ]
    assert [command[command.index("--k1") + 1] for command in measured] == [
        variant
        for process_round in range(config.processes)
        for variant in (variants if process_round % 2 == 0 else tuple(reversed(variants)))
    ]
    assert len([command for command in adapter.commands if command[1] == "compress"]) == (
        len(variants) if len(variants) > 1 else 0
    )
    assert len(run.record_identity) == (1 if len(variants) > 1 else 0)
    if len(variants) > 1:
        hashes = run.record_identity[0]["record_sha256"]
        assert len(set(hashes.values())) == 1
