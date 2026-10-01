import json

import pytest
from _bench_test_support import READY_FACTS, ROOT, run_cpu_snapshot

from stoquant import cli, host, sweep
from stoquant.design import (
    DEFAULT_COUNTS,
    PILOT_PROFILE,
)
from stoquant.host import (
    EXCLUDED_LOGICAL_CPUS,
    affinity_mask_excluding,
    check_readiness,
    get_process_affinity,
)
from stoquant.matrix import run_benchmark_matrix
from stoquant.monitoring import ConfigurationError


def test_readiness_passes_on_a_quiet_fresh_machine():
    assert check_readiness(READY_FACTS) == []


def test_readiness_does_not_restrict_uptime():
    old_uptime = {**READY_FACTS, "uptime_seconds": 5 * 3600.0}
    assert check_readiness(old_uptime) == []


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        (
            {
                "physical_memory": {
                    **READY_FACTS["physical_memory"],
                    "available_bytes": 3 * 1024**3,
                }
            },
            "available physical memory 3.00 GiB",
        ),
        ({"stoquant_pids": [4242]}, "stoquant already running (pid 4242)"),
        ({"gpu_clock_event_reasons": "0x0000000000000024"}, "SwPowerCap, SwThermalSlowdown"),
        ({"gpu_clock_event_reasons": None}, "clock-event reasons unavailable"),
        ({"gpu_clock_event_reasons": "[N/A]"}, "clock-event reasons unavailable"),
    ],
)
def test_readiness_names_each_failure(change, reason):
    failures = check_readiness({**READY_FACTS, **change})
    assert len(failures) == 1
    assert reason in failures[0]


def test_readiness_names_a_failed_probe_from_the_probe_seam(monkeypatch):
    monkeypatch.setattr(host, "query_gpu_state", lambda: {"clocks_event_reasons.active": "0x1"})
    monkeypatch.setattr(host, "uptime_seconds", lambda: 600.0)
    monkeypatch.setattr(
        host,
        "physical_memory_status",
        lambda: (_ for _ in ()).throw(RuntimeError("memory probe failed")),
    )
    monkeypatch.setattr(host, "list_processes", lambda _name: [])
    monkeypatch.setattr(host, "collect_git_provenance", lambda _root: {"dirty_files": []})
    monkeypatch.setattr(host, "query_power_plan", lambda: "Balanced")
    monkeypatch.setattr(host, "query_hags", lambda: "unset")

    facts = host.probe_readiness_facts(ROOT)
    failures = check_readiness(facts)

    assert failures == [
        "host probe failed: physical_memory_status: RuntimeError: memory probe failed"
    ]


def test_process_probe_rejects_non_json_output(monkeypatch):
    if host.os.name != "nt":
        pytest.skip("Windows process listing uses PowerShell")
    result = host.subprocess.CompletedProcess(
        args=["powershell.exe"],
        returncode=0,
        stdout="not a process ID list",
        stderr="",
    )
    monkeypatch.setattr(host.subprocess, "run", lambda *_args, **_kwargs: result)

    with pytest.raises(RuntimeError, match="list_processes returned unparseable JSON"):
        host.list_processes("stoquant")


def test_dirty_tree_is_judged_only_by_the_sweep_gate():
    facts = {**READY_FACTS, "git_dirty_files": [" M src/stoquant/design.py"]}
    assert check_readiness(facts) == []


def test_readiness_fails_while_a_stoquant_process_runs(monkeypatch):
    monkeypatch.setattr(host, "query_gpu_state", lambda: {"clocks_event_reasons.active": "0x1"})
    monkeypatch.setattr(host, "uptime_seconds", lambda: 600.0)
    monkeypatch.setattr(host, "collect_git_provenance", lambda root: {"dirty_files": []})
    monkeypatch.setattr(host, "query_power_plan", lambda: "Balanced")
    monkeypatch.setattr(host, "query_hags", lambda: "unset")
    monkeypatch.setattr(host, "list_processes", lambda name: [4242] if name == "stoquant" else [])

    failures = check_readiness(host.probe_readiness_facts(ROOT))

    assert failures == ["stoquant already running (pid 4242)"]


def test_pilot_is_four_sizes_from_the_sweep_grid():
    assert len(PILOT_PROFILE.counts) == 4
    assert set(PILOT_PROFILE.counts) <= set(DEFAULT_COUNTS)


def test_affinity_mask_excludes_core_zero():
    assert affinity_mask_excluding(EXCLUDED_LOGICAL_CPUS, 0b1111_1111_1111) == 0b1111_1111_1100
    # Built from the current mask, so CPUs the process never had stay excluded.
    assert affinity_mask_excluding(EXCLUDED_LOGICAL_CPUS, 0b1010_1111) == 0b1010_1100
    with pytest.raises(RuntimeError, match="no logical CPU"):
        affinity_mask_excluding((0, 1), 0b11)


def test_failed_readiness_stops_the_sweep_before_any_process(tmp_path):
    low_memory = {
        **READY_FACTS,
        "physical_memory": {
            **READY_FACTS["physical_memory"],
            "available_bytes": 3 * 1024**3,
        },
    }
    with pytest.raises(RuntimeError, match="available physical memory 3.00 GiB"):
        run_cpu_snapshot(tmp_path / "out", readiness_facts=low_memory)
    assert not (tmp_path / "out").exists()


def test_invalid_monitoring_configuration_fails_before_binary_lookup(tmp_path, monkeypatch):
    monkeypatch.setenv("STOQUANT_MONITORING_HEARTBEAT_SECONDS", "not-an-integer")

    def unexpected_binary_lookup(*_args, **_kwargs):
        pytest.fail("binary lookup started before monitoring validation")

    monkeypatch.setattr(sweep, "find_binary", unexpected_binary_lookup)
    with pytest.raises(ConfigurationError, match="positive integer"):
        run_cpu_snapshot(tmp_path / "out", readiness_facts=READY_FACTS)
    assert not (tmp_path / "out").exists()


def test_duplicate_counts_are_rejected_by_api_and_cli_before_work(tmp_path):
    with pytest.raises(ValueError, match="counts values must be unique"):
        run_benchmark_matrix(
            root=ROOT,
            output_dir=tmp_path / "api",
            counts=[1024, 1024],
            backends=["cpu"],
        )
    with pytest.raises(SystemExit):
        cli.main(["--counts", "1024", "1024"])
    assert not (tmp_path / "api").exists()


def test_readiness_override_marks_the_snapshot_non_evidence(tmp_path):
    busy = {**READY_FACTS, "stoquant_pids": [4242]}
    before = get_process_affinity()
    out = run_cpu_snapshot(tmp_path / "out", readiness_facts=busy, ignore_readiness=True)
    assert get_process_affinity() == before
    conditions = json.loads((out / "manifest.json").read_text(encoding="utf-8"))["run_conditions"]
    assert conditions["evidence"] is False
    assert conditions["readiness"]["overridden"] is True
    assert "readiness check failed and was overridden" in conditions["non_evidence_reasons"]


def test_readiness_probe_override_records_named_failure_as_non_evidence(tmp_path):
    probe_failure = {
        **READY_FACTS,
        "probe_failures": {"physical_memory_status": "memory probe failed"},
    }

    out = run_cpu_snapshot(
        tmp_path / "probe-failure", readiness_facts=probe_failure, ignore_readiness=True
    )
    conditions = json.loads((out / "manifest.json").read_text(encoding="utf-8"))["run_conditions"]

    assert conditions["evidence"] is False
    assert conditions["readiness"]["overridden"] is True
    assert (
        "host probe failed: physical_memory_status: memory probe failed"
        in conditions["readiness"]["failures"]
    )
    assert "readiness check failed and was overridden" in conditions["non_evidence_reasons"]


def test_pilot_records_readiness_without_enforcing_it(tmp_path):
    low_memory = {
        **READY_FACTS,
        "physical_memory": {
            **READY_FACTS["physical_memory"],
            "available_bytes": 3 * 1024**3,
        },
    }
    out = run_cpu_snapshot(tmp_path / "out", readiness_facts=low_memory, pilot=True)
    conditions = json.loads((out / "manifest.json").read_text(encoding="utf-8"))["run_conditions"]
    assert conditions["pilot"] is True
    assert conditions["evidence"] is False
    assert conditions["readiness"]["passed"] is False
    assert conditions["readiness"]["overridden"] is False
    assert conditions["readiness"]["enforced"] is False
    assert "pilot run" in conditions["non_evidence_reasons"]


def test_manifest_records_monitoring_without_topic_or_token(tmp_path, monkeypatch):
    monkeypatch.setenv("STOQUANT_MONITORING_PROVIDER", "ntfy")
    monkeypatch.setenv("STOQUANT_MONITORING_HEARTBEAT_SECONDS", "3600")
    monkeypatch.setenv("NTFY_TOPIC", "manifest-must-not-contain-topic")
    monkeypatch.setenv("NTFY_TOKEN", "manifest-must-not-contain-token")

    snapshot = run_cpu_snapshot(tmp_path / "monitoring", readiness_facts=READY_FACTS)
    manifest_text = (snapshot / "manifest.json").read_text(encoding="utf-8")
    manifest = json.loads(manifest_text)

    assert manifest["monitoring"] == {
        "provider": "ntfy",
        "heartbeat_interval_seconds": 3600,
    }
    assert "manifest-must-not-contain-topic" not in manifest_text
    assert "manifest-must-not-contain-token" not in manifest_text


def test_failed_sweep_restores_the_affinity_mask(tmp_path, monkeypatch):
    def missing_binary(root, **kwargs):
        raise FileNotFoundError("no binary")

    monkeypatch.setattr(sweep, "find_binary", missing_binary)
    before = get_process_affinity()
    with pytest.raises(FileNotFoundError):
        run_cpu_snapshot(tmp_path / "out", readiness_facts=READY_FACTS)
    assert get_process_affinity() == before
