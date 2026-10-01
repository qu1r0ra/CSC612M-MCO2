"""Benchmark matrix validation and snapshot lifecycle."""

from __future__ import annotations

import os
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from stoquant.design import (
    DEFAULT_BITS,
    DEFAULT_CASE_ORDER_SEED,
    DEFAULT_COUNTS,
    DEFAULT_TRIALS,
    K1_VARIANTS,
    TRANSFER_POLICIES,
    build_paths,
    trial_orders,
)
from stoquant.host import (
    MIN_AVAILABLE_PHYSICAL_BYTES,
    benchmark_process_affinity,
    check_readiness,
    probe_readiness_facts,
)
from stoquant.inputs import DEFAULT_INPUT_SEED, INPUT_FAMILIES, select_model_tensors
from stoquant.monitoring import parse_monitoring_config
from stoquant.provenance import collect_git_provenance
from stoquant.runner import (
    DEFAULT_CASE_WARMUP_SECONDS,
    DEFAULT_COMPRESSION_SEED,
    DEFAULT_GPU_WARMUP_SECONDS,
    DEFAULT_IN_PROCESS_WARMUP_SECONDS,
    DEFAULT_REPS,
    DEFAULT_WARMUPS,
    REAL_BENCH_PROCESS,
    BenchProcess,
)
from stoquant.snapshot_store import (
    ACTIVE_SNAPSHOT_REGISTRATION_ENV,
    check_snapshot,
    create_snapshot,
    register_active_snapshot,
    snapshot_path,
)
from stoquant.sweep import Sweep, SweepSettings


def run_benchmark_matrix(
    *,
    root: Path,
    output_dir: Path | None = None,
    counts: Sequence[int] = DEFAULT_COUNTS,
    bit_widths: Sequence[int] = DEFAULT_BITS,
    backends: Sequence[str] = ("cpu", "cuda"),
    warmups: int = DEFAULT_WARMUPS,
    reps: int = DEFAULT_REPS,
    trials: int = DEFAULT_TRIALS,
    input_seed: int = DEFAULT_INPUT_SEED,
    compression_seed: int = DEFAULT_COMPRESSION_SEED,
    case_order_seed: int = DEFAULT_CASE_ORDER_SEED,
    gpu_warmup_seconds: float = DEFAULT_GPU_WARMUP_SECONDS,
    case_warmup_seconds: float = DEFAULT_CASE_WARMUP_SECONDS,
    in_process_warmup_seconds: float = DEFAULT_IN_PROCESS_WARMUP_SECONDS,
    bench_process: BenchProcess = REAL_BENCH_PROCESS,
    allow_dirty: bool = False,
    pilot: bool = False,
    readiness_facts: dict[str, Any] | None = None,
    ignore_readiness: bool = False,
    boundaries: Sequence[str] = (),
    transfer_policies: Sequence[str] = ("pageable",),
    input_family: str = "dense",
    model_tensors: str | None = None,
    model_limit: int | None = None,
    k1: str = "reference",
) -> Path:
    monitoring = parse_monitoring_config(root).manifest_metadata()
    if len(set(counts)) != len(counts):
        raise ValueError("--counts values must be unique")
    if k1 not in K1_VARIANTS:
        raise ValueError(f"unknown K1 variant {k1!r}; choose from {K1_VARIANTS}")
    if input_family not in INPUT_FAMILIES:
        raise ValueError(f"unknown input family {input_family!r}; choose from {INPUT_FAMILIES}")
    if input_family != "model" and (model_tensors is not None or model_limit is not None):
        raise ValueError("--model-tensors and --model-limit apply only to the model family")
    if input_family == "model":
        model_tensors = model_tensors or "distinct"
        select_model_tensors(model_tensors, model_limit)
    if trials < 1:
        raise ValueError("trials must be at least 1")
    if "cpu" not in backends:
        raise ValueError("the CPU comparator is required; include 'cpu' in backends")
    # Reject a bad path selection or trial count before touching the machine or disk.
    paths = build_paths(backends, boundaries, transfer_policies)
    trial_orders(paths, trials)

    # Revision 3: every probe and benchmark process runs off physical core 0.
    with benchmark_process_affinity() as affinity:
        git_prov = collect_git_provenance(root)

        target_dir = output_dir or snapshot_path(
            root, git_prov["code_revision_short"], input_family, pilot=pilot
        )
        date_str = datetime.now().astimezone().strftime("%Y-%m-%d")
        check_snapshot(target_dir, git_prov, allow_dirty=allow_dirty)

        # Revision 3: judge the machine before any benchmark process starts. A pilot is
        # non-evidence by design, so it records the check without enforcing it.
        facts = (
            probe_readiness_facts(root, git_prov) if readiness_facts is None else readiness_facts
        )
        failures = check_readiness(facts)
        if failures and not (ignore_readiness or pilot):
            raise RuntimeError(
                "Machine is not ready for an evidence sweep; fix these or pass --ignore-readiness "
                "for a non-evidence run: " + " | ".join(failures)
            )
        non_evidence_reasons = []
        if pilot:
            non_evidence_reasons.append("pilot run")
        if failures:
            non_evidence_reasons.append(
                "readiness check failed and was overridden"
                if ignore_readiness
                else "readiness check failed (pilot; not enforced)"
            )
        if git_prov["git_dirty"]:
            non_evidence_reasons.append("dirty git tree")
        run_conditions = {
            "evidence": not non_evidence_reasons,
            "non_evidence_reasons": non_evidence_reasons,
            "pilot": pilot,
            "readiness": {
                "passed": not failures,
                "failures": failures,
                "overridden": bool(failures) and ignore_readiness,
                "enforced": not pilot,
                "facts": facts,
                "min_available_physical_bytes": MIN_AVAILABLE_PHYSICAL_BYTES,
            },
        }

        run_conditions["affinity"] = {
            **affinity,
            "excluded_logical_cpus": list(affinity["excluded_logical_cpus"]),
            "mask": hex(affinity["mask"]),
            "previous_mask": hex(affinity["previous_mask"]),
            "applied_to": "driver process before any benchmark or probe process; inherited",
        }
        snapshot = create_snapshot(target_dir, git_prov, allow_dirty=allow_dirty)
        registration = os.environ.get(ACTIVE_SNAPSHOT_REGISTRATION_ENV)
        registration_file = Path(registration) if registration is not None else None
        register_active_snapshot(snapshot.partial, registration_file)
        settings = SweepSettings(
            root=root,
            target_dir=snapshot.partial,
            date_str=date_str,
            git_prov=git_prov,
            run_conditions=run_conditions,
            counts=tuple(counts),
            bit_widths=tuple(bit_widths),
            backends=tuple(backends),
            paths=tuple(paths),
            transfer_policies=tuple(p for p in TRANSFER_POLICIES if p in transfer_policies),
            warmups=warmups,
            reps=reps,
            trials=trials,
            input_seed=input_seed,
            compression_seed=compression_seed,
            case_order_seed=case_order_seed,
            gpu_warmup_seconds=gpu_warmup_seconds,
            case_warmup_seconds=case_warmup_seconds,
            in_process_warmup_seconds=in_process_warmup_seconds,
            bench_process=bench_process,
            input_family=input_family,
            model_tensors=model_tensors,
            model_limit=model_limit,
            k1=k1,
            monitoring=monitoring,
        )
        Sweep(settings).run()
        final_snapshot = snapshot.commit()
        register_active_snapshot(final_snapshot, registration_file)
        return final_snapshot
