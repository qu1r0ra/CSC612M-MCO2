"""Benchmark driver for in-process stoquant CPU comparator and CUDA pipelines.

Orchestrates input generation, per-case correctness gating, timed benchmark
runs, provenance collection, statistical summary computation, and snapshot
creation as specified by the course technical contract and benchmark protocol.
"""

from __future__ import annotations

import argparse
import csv
import math
import shutil
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NamedTuple, cast

import numpy as np

from stoquant import layout
from stoquant.correctness import verify_correctness
from stoquant.design import (
    DEFAULT_BITS,
    DEFAULT_CASE_ORDER_SEED,
    DEFAULT_COUNTS,
    DEFAULT_TRIALS,
    EXTENSION_BOUNDARIES,
    K1_VARIANTS,
    TRANSFER_POLICIES,
    BenchPath,
    build_paths,
    case_order,
    trial_design,
    trial_orders,
)
from stoquant.host import (
    MAX_UPTIME_SECONDS,
    WINDOW_ALLOWLIST,
    benchmark_process_affinity,
    check_readiness,
    probe_readiness_facts,
)
from stoquant.inputs import (
    DEFAULT_INPUT_SEED,
    INPUT_FAMILIES,
    MODEL_TENSOR_SETS,
    InputProvenance,
    generate_family_inputs,
    select_model_tensors,
)
from stoquant.monitoring import MonitoringMetadata, parse_monitoring_config
from stoquant.oracle import HEADER_STRUCT
from stoquant.provenance import (
    collect_build_commands,
    collect_git_provenance,
    collect_hardware_and_toolchain,
    find_binary,
    query_device_attributes,
    query_gpu_state,
)
from stoquant.runner import (
    DEFAULT_CASE_WARMUP_SECONDS,
    DEFAULT_COMPRESSION_SEED,
    DEFAULT_GPU_WARMUP_SECONDS,
    DEFAULT_IN_PROCESS_WARMUP_SECONDS,
    DEFAULT_REPS,
    DEFAULT_WARMUPS,
    REAL_BENCH_PROCESS,
    BenchProcess,
    in_process_warmups,
    run_bench_process,
    warm_up_gpu,
)
from stoquant.schema import (
    CaseRecord,
    MatrixManifest,
    read_manifest,
    write_case,
)
from stoquant.schema import (
    write_manifest as write_manifest_record,
)
from stoquant.snapshot_store import (
    RunPlan,
    check_snapshot,
    create_snapshot,
    snapshot_path,
    write_run_plan,
)
from stoquant.stats import (
    BASELINE_RULE,
    BOOTSTRAP_LEVEL,
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_RULE,
    BOOTSTRAP_SEED,
    CLAIM_RULE_REV2,
    DIRECTION_RULE,
    MAGNITUDE_RULE,
    QUANTILE_METHOD,
    SPREAD_THRESHOLD,
    STABILITY_PERCENTILES,
    STABILITY_THRESHOLD,
    STAGE_KEYS,
    VERDICT_RULE,
    compare_case_group,
    compute_case_statistics,
    compute_stage_medians,
)
from stoquant.vectorization import AVX2_SOURCES, generate_msvc_vectorization_report

# The pilot runs the sweep's code path on four sizes: the smallest, the
# launch-bound 2^14, a mid size, and the largest.
MATRIX_PILOT_COUNTS = (1 << 10, 1 << 14, 1 << 20, 1 << 26)


def compact_invocation_ids(ids: Sequence[int] | None) -> dict[str, int] | list[int] | None:
    """Store a contiguous or repeated identifier run compactly; warm-ups reach 10^4-10^5.

    Resident-graph replays one invocation, so its identifiers repeat.
    """
    if not ids:
        return None if ids is None else []
    if len(ids) > 1 and all(i == ids[0] for i in ids):
        return {"repeated": ids[0], "count": len(ids)}
    if list(ids) != list(range(ids[0], ids[0] + len(ids))):
        return list(ids)
    return {"first": ids[0], "last": ids[-1], "count": len(ids)}


def failed_row(
    count: int,
    bits: int,
    backend: str,
    boundary: str,
    warmups: int,
    reps: int,
    trials: int,
    *,
    transfer_policy: str,
    path_label: str,
    input_family: str = "dense",
    input_key: str = "",
) -> dict[str, Any]:
    row: dict[str, Any] = dict.fromkeys(SUMMARY_FIELDS, "")
    row.update(
        {
            "count": count,
            "input_family": input_family,
            "input_key": input_key or f"n{count}",
            "bits": bits,
            "backend": backend,
            "boundary": boundary,
            "transfer_policy": transfer_policy,
            "path_label": path_label,
            "correctness": "failed",
            "warmup": warmups,
            "reps": reps,
            "trials": trials,
        }
    )
    return row


def csv_flag(value: bool | None) -> str:
    return "" if value is None else str(value).lower()


SUMMARY_FIELDS = [
    "count",
    "bits",
    "backend",
    "boundary",
    "transfer_policy",
    "path_label",
    "correctness",
    "warmup",
    "reps",
    "trials",
    "median_ms",
    "iqr_ms",
    "trial_median_min_ms",
    "trial_median_max_ms",
    "spread_ratio",
    "spread_p90_p10",
    "stable",
    "unstable_rev2",
    "baseline",
    "speedup_vs_c",
    "speedup_low",
    "speedup_high",
    "speedup_ci_low",
    "speedup_ci_high",
    "verdict",
    "boundary_inversion",
    "direction_supported",
    "magnitude_supported",
    "claim_supported_rev2",
    "input_family",
    "input_key",
]


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
                "max_uptime_seconds": MAX_UPTIME_SECONDS,
                "window_allowlist": list(WINDOW_ALLOWLIST),
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
        sweep_matrix(
            root=root,
            target_dir=snapshot.partial,
            date_str=date_str,
            git_prov=git_prov,
            run_conditions=run_conditions,
            counts=counts,
            bit_widths=bit_widths,
            backends=backends,
            paths=paths,
            transfer_policies=[p for p in TRANSFER_POLICIES if p in transfer_policies],
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
        return snapshot.commit()


@dataclass(frozen=True)
class SweepSettings:
    """The sweep parameters `sweep_matrix` receives, shared by its phases."""

    counts: Sequence[int]
    bit_widths: Sequence[int]
    backends: Sequence[str]
    paths: Sequence[BenchPath]
    transfer_policies: Sequence[str]
    warmups: int
    reps: int
    trials: int
    input_seed: int
    compression_seed: int
    case_order_seed: int
    gpu_warmup_seconds: float
    case_warmup_seconds: float
    in_process_warmup_seconds: float
    bench_process: BenchProcess
    input_family: str
    model_tensors: str | None
    model_limit: int | None
    k1: str

    @property
    def cuda(self) -> bool:
        return "cuda" in self.backends

    @property
    def has_avx2(self) -> bool:
        return any(path.backend == "cpu-avx2" for path in self.paths)

    @property
    def dense(self) -> bool:
        return self.input_family == "dense"


class RunSetup(NamedTuple):
    binary: Path
    build_stamp: dict[str, Any]
    toolchain_prov: dict[str, Any]
    build_prov: dict[str, Any]
    host_tokens: list[str]
    avx2_tokens: list[str]
    device_attributes: dict[str, Any] | None
    gpu_state_start: dict[str, Any] | None
    temp_dir: Path


class SweepInputs(NamedTuple):
    files: dict[str, Path]
    # Keys keep the generator's order, which ranks inputs in the summary.
    clean_meta: dict[str, InputProvenance]
    counts: dict[str, int]


class SweepOrder(NamedTuple):
    orders: list[list[int]]
    order_labels: list[list[str]]
    ordered_cases: list[tuple[str, int]]
    gpu_warmup: dict[str, Any] | None
    gpu_state_after_warmup: dict[str, Any] | None


def sweep_matrix(
    *,
    root: Path,
    target_dir: Path,
    date_str: str,
    git_prov: dict[str, Any],
    run_conditions: dict[str, Any],
    counts: Sequence[int],
    bit_widths: Sequence[int],
    backends: Sequence[str],
    paths: Sequence[BenchPath],
    transfer_policies: Sequence[str],
    warmups: int,
    reps: int,
    trials: int,
    input_seed: int,
    compression_seed: int,
    case_order_seed: int,
    gpu_warmup_seconds: float,
    case_warmup_seconds: float,
    in_process_warmup_seconds: float,
    bench_process: BenchProcess = REAL_BENCH_PROCESS,
    input_family: str = "dense",
    model_tensors: str | None = None,
    model_limit: int | None = None,
    k1: str = "reference",
    monitoring: MonitoringMetadata | None = None,
) -> Path:
    settings = SweepSettings(
        counts=counts,
        bit_widths=bit_widths,
        backends=backends,
        paths=paths,
        transfer_policies=transfer_policies,
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
    )
    setup = set_up_run(root, target_dir, settings, git_prov)
    inputs = generate_inputs(setup, settings)
    run_plan: RunPlan = {
        "total_cases": len(inputs.clean_meta) * len(settings.bit_widths) * len(paths),
        "input_family": settings.input_family,
    }
    write_run_plan(target_dir, run_plan)
    vec_report_path = write_vectorization_reports(root, target_dir, setup, settings)
    order = order_and_warm_up(setup, inputs, settings)

    case_results: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for execution_index, (input_key, bits) in enumerate(order.ordered_cases):
        # Gate on correctness once, outside every timed process.
        passed, correctness_info = verify_correctness(
            binary=setup.binary,
            input_path=inputs.files[input_key],
            count=inputs.counts[input_key],
            bits=bits,
            backends=backends,
            paths=paths,
            seed=compression_seed,
            tensor_id=0,
            invocation_id=0,
            tmp_dir=setup.temp_dir,
            process=bench_process,
            k1=k1,
        )
        cases = build_case_stubs(
            setup, git_prov, inputs, input_key, bits, correctness_info, execution_index, settings
        )
        if passed:
            warm_up_case(cases, setup.binary, inputs.files[input_key], bits, settings)
            run_trials(cases, setup.binary, inputs.files[input_key], bits, order, settings)
        compute_case_group_statistics(cases, paths)
        summary_rows.extend(
            write_case_outputs(target_dir, cases, inputs, input_key, bits, settings)
        )
        case_results.extend(cases)

    gpu_state_end = query_gpu_state() if settings.cuda else None
    csv_path = write_summary(target_dir, setup, inputs, case_results, summary_rows, settings)
    write_manifest(
        target_dir,
        date_str=date_str,
        git_prov=git_prov,
        run_conditions=run_conditions,
        setup=setup,
        inputs=inputs,
        order=order,
        gpu_state_end=gpu_state_end,
        csv_path=csv_path,
        vec_report_path=vec_report_path,
        case_results=case_results,
        settings=settings,
        monitoring=monitoring,
    )
    return target_dir


def set_up_run(
    root: Path, target_dir: Path, settings: SweepSettings, git_prov: dict[str, Any]
) -> RunSetup:
    """Binary, provenance and starting GPU state, then the snapshot and scratch folders."""
    verified = find_binary(root, require_cuda=settings.cuda, git_prov=git_prov)
    toolchain_prov = collect_hardware_and_toolchain(root)
    build_commands = collect_build_commands(root)
    build_prov = build_commands.as_record()
    host_tokens = build_commands.host_tokens or []
    avx2_tokens = build_commands.avx2_tokens or []
    publication = len(settings.paths) == 10 and settings.k1 == "optimized"
    device_attributes = query_device_attributes(root) if publication else None
    gpu_state_start = query_gpu_state() if settings.cuda else None

    temp_dir = target_dir / "_temp"
    temp_dir.mkdir()
    return RunSetup(
        verified.path,
        verified.stamp,
        toolchain_prov,
        build_prov,
        host_tokens,
        avx2_tokens,
        device_attributes,
        gpu_state_start,
        temp_dir,
    )


def generate_inputs(setup: RunSetup, settings: SweepSettings) -> SweepInputs:
    input_meta = generate_family_inputs(
        settings.input_family,
        settings.counts,
        setup.temp_dir / "inputs",
        settings.input_seed,
        settings.model_tensors or "distinct",
        settings.model_limit,
    )
    return SweepInputs(
        files={key: item.path for key, item in input_meta.items()},
        clean_meta={key: item.provenance for key, item in input_meta.items()},
        counts={key: item.provenance["count"] for key, item in input_meta.items()},
    )


def write_vectorization_reports(
    root: Path, target_dir: Path, setup: RunSetup, settings: SweepSettings
) -> Path:
    """Compile with the comparator's own flags; returns the scalar report's path."""
    vec_report_path = target_dir / "msvc_vectorization_report.txt"
    generate_msvc_vectorization_report(
        root, vec_report_path, setup.host_tokens, setup.temp_dir / "vec_obj"
    )
    if settings.has_avx2:
        generate_msvc_vectorization_report(
            root,
            target_dir / "msvc_vectorization_report_avx2.txt",
            setup.avx2_tokens,
            setup.temp_dir / "vec_obj_avx2",
            AVX2_SOURCES,
        )
    return vec_report_path


def order_and_warm_up(setup: RunSetup, inputs: SweepInputs, settings: SweepSettings) -> SweepOrder:
    """Trial and case orders, then the pre-loop GPU warm-up."""
    paths = settings.paths
    orders = trial_orders(paths, settings.trials)
    order_labels = [[paths[i].label for i in order] for order in orders]
    ordered_cases = case_order(
        list(inputs.clean_meta), settings.bit_widths, settings.case_order_seed
    )
    largest_input = max(inputs.clean_meta, key=lambda key: inputs.counts[key])

    # Bring the GPU to steady clocks on the largest input before any timed process.
    gpu_warmup = None
    gpu_state_after_warmup = None
    if settings.cuda and settings.gpu_warmup_seconds > 0:
        gpu_warmup = warm_up_gpu(
            setup.binary,
            inputs.files[largest_input],
            settings.gpu_warmup_seconds,
            settings.k1,
            settings.bench_process,
        )
        gpu_state_after_warmup = query_gpu_state()
    return SweepOrder(orders, order_labels, ordered_cases, gpu_warmup, gpu_state_after_warmup)


def build_case_stubs(
    setup: RunSetup,
    git_prov: dict[str, Any],
    inputs: SweepInputs,
    input_key: str,
    bits: int,
    correctness_info: dict[str, Any],
    execution_index: int,
    settings: SweepSettings,
) -> list[dict[str, Any]]:
    """One case record per path, before any warm-up or trial fills it."""
    count = inputs.counts[input_key]
    payload_bytes = count if bits == 8 else (count + 1) // 2
    build_prov = setup.build_prov
    toolchain_prov = setup.toolchain_prov
    k1 = settings.k1
    cases: list[dict[str, Any]] = []
    for path in settings.paths:
        cases.append(
            {
                "case_id": f"case_{path.backend}_{path.key}_bits{bits}_{input_key}",
                "count": count,
                "input_family": settings.input_family,
                "input_key": input_key,
                "bits": bits,
                "backend": path.backend,
                "timing_boundary": path.boundary,
                "transfer_policy": path.policy,
                "path_label": path.label,
                "configuration": {"k1": k1 if path.backend == "cuda" else None},
                "seed": settings.compression_seed,
                "tensor_id": 0,
                "invocation_id": 0,
                "warmup": settings.warmups,
                "in_process_warmup": None,
                "reps": settings.reps,
                "trials": settings.trials,
                "header_bytes": HEADER_STRUCT.size,
                "payload_bytes": payload_bytes,
                "code_revision": git_prov["code_revision"],
                "code_revision_short": git_prov["code_revision_short"],
                "git_dirty": git_prov["git_dirty"],
                "build_flags": {
                    "comparator_c": build_prov["comparator_c"],
                    "cuda_nvcc": build_prov["cuda_nvcc"],
                    **({"avx2_c": build_prov["avx2_c"]} if settings.has_avx2 else {}),
                },
                "hardware": toolchain_prov["hardware"],
                "toolkit_and_driver": toolchain_prov["toolkit_and_driver"],
                "input_provenance": inputs.clean_meta[input_key],
                "correctness": correctness_info,
                "execution_index": execution_index,
                "case_warmup": None,
                "trial_runs": [],
                "samples_ms": [],
                "statistics": None,
                "stage_medians_ms": None,
            }
        )
    return cases


def warm_up_case(
    cases: list[dict[str, Any]],
    binary: Path,
    input_path: Path,
    bits: int,
    settings: SweepSettings,
) -> None:
    """GPU clock warm-up, then each path's in-process warm-up count from an untimed probe."""
    if settings.cuda and settings.case_warmup_seconds > 0:
        # Restore GPU clocks after the untimed correctness gate and any
        # long CPU processes of the previous case.
        case_warmup = warm_up_gpu(
            binary, input_path, settings.case_warmup_seconds, settings.k1, settings.bench_process
        )
        case_warmup["gpu_state_after"] = query_gpu_state()
        for case in cases:
            case["case_warmup"] = case_warmup

    if settings.in_process_warmup_seconds > 0:
        # Revision 2: size each path's in-process warm-up from an untimed probe
        # so every timed process first runs about the same wall time untimed.
        warmups = settings.warmups
        reps = settings.reps
        for case, path in zip(cases, settings.paths, strict=True):
            payload, error = run_bench_process(
                binary,
                input_path,
                bits=bits,
                backend=path.backend,
                extra_args=path.extra_args,
                seed=settings.compression_seed,
                warmups=warmups,
                reps=reps,
                k1=settings.k1,
                process=settings.bench_process,
            )
            if payload is None:
                case["correctness"] = {"status": "failed", "error_message": error}
                continue
            samples = payload.get("samples_ms")
            if (
                not isinstance(samples, list)
                or not samples
                or not all(
                    isinstance(sample, (int, float)) and math.isfinite(sample) for sample in samples
                )
            ):
                case["correctness"] = {
                    "status": "failed",
                    "error_message": "calibration probe returned empty or non-finite samples",
                }
                continue
            probe_ms = float(np.median(samples))
            case["warmup"] = in_process_warmups(
                warmups, settings.in_process_warmup_seconds, probe_ms
            )
            case["in_process_warmup"] = {
                "target_seconds": settings.in_process_warmup_seconds,
                "probe_warmup": warmups,
                "probe_reps": reps,
                "probe_median_ms": probe_ms,
                "warmup": case["warmup"],
            }


def run_trials(
    cases: list[dict[str, Any]],
    binary: Path,
    input_path: Path,
    bits: int,
    order: SweepOrder,
    settings: SweepSettings,
) -> None:
    # Each trial runs every path as its own process, in a balanced order.
    # Every trial reuses the same invocation identifiers, so trials are
    # replicates of an identical workload.
    k1 = settings.k1
    for trial_index, trial_order in enumerate(order.orders):
        for position, path_index in enumerate(trial_order):
            case = cases[path_index]
            if case["correctness"]["status"] != "passed":
                continue
            path = settings.paths[path_index]
            payload, error = run_bench_process(
                binary,
                input_path,
                bits=bits,
                backend=path.backend,
                extra_args=path.extra_args,
                seed=settings.compression_seed,
                warmups=case["warmup"],
                reps=settings.reps,
                k1=k1,
                process=settings.bench_process,
            )
            if payload is None:
                case["correctness"] = {"status": "failed", "error_message": error}
                continue
            configuration = payload.get("configuration", {})
            if path.backend == "cuda" and configuration.get("k1") != k1:
                raise RuntimeError(
                    f"{case['case_id']}: stoquant bench ran K1 "
                    f"{configuration.get('k1')!r}, expected {k1!r}"
                )
            if configuration.get("transfer_policy") != path.policy:
                raise RuntimeError(
                    f"{case['case_id']}: stoquant bench ran transfer policy "
                    f"{configuration.get('transfer_policy')!r}, expected {path.policy!r}"
                )
            run: dict[str, Any] = {
                "trial": trial_index,
                "position": position,
                "order": order.order_labels[trial_index],
                "repetition_invocation_ids": configuration.get("repetition_invocation_ids"),
                "warmup_invocation_ids": compact_invocation_ids(
                    configuration.get("warmup_invocation_ids")
                ),
                "samples_ms": payload.get("samples_ms", []),
            }
            if "threads" in configuration:
                run["threads"] = configuration["threads"]
            if "capture_and_instantiate_ms" in payload:
                run["capture_and_instantiate_ms"] = payload["capture_and_instantiate_ms"]
            for key in STAGE_KEYS:
                if key in payload:
                    run[key] = payload[key]
            case["trial_runs"].append(run)


def compute_case_group_statistics(cases: list[dict[str, Any]], paths: Sequence[BenchPath]) -> None:
    """Pooled statistics per passing case, then comparisons across the group."""
    for case in cases:
        if case["correctness"]["status"] != "passed":
            case["trial_runs"] = []
            continue
        runs = case["trial_runs"]
        first = runs[0]
        for run in runs[1:]:
            if (
                run["repetition_invocation_ids"] != first["repetition_invocation_ids"]
                or run["warmup_invocation_ids"] != first["warmup_invocation_ids"]
            ):
                raise RuntimeError(
                    f"{case['case_id']}: trials used different invocation identifiers"
                )
        if "threads" in first:
            if any(run.get("threads") != first["threads"] for run in runs):
                raise RuntimeError(f"{case['case_id']}: trials ran different thread counts")
            case["threads"] = first["threads"]
        case["repetition_invocation_ids"] = first["repetition_invocation_ids"]
        case["warmup_invocation_ids"] = first["warmup_invocation_ids"]
        case["samples_ms"] = [s for run in runs for s in run["samples_ms"]]
        case["statistics"] = compute_case_statistics([run["samples_ms"] for run in runs])
        case["stage_medians_ms"] = compute_stage_medians(runs)

    # Comparisons and flags, only between cases that all passed.
    compare_case_group(cases, paths)


def write_case_outputs(
    target_dir: Path,
    cases: list[dict[str, Any]],
    inputs: SweepInputs,
    input_key: str,
    bits: int,
    settings: SweepSettings,
) -> list[dict[str, Any]]:
    """Write each case's JSON; returns the case group's summary rows."""
    count = inputs.counts[input_key]
    input_family = settings.input_family
    reps = settings.reps
    trials = settings.trials
    rows: list[dict[str, Any]] = []
    for case in cases:
        write_case(target_dir / f"{case['case_id']}.json", cast(CaseRecord, case))
        stats = case["statistics"]
        if case["correctness"]["status"] != "passed":
            rows.append(
                failed_row(
                    count,
                    bits,
                    case["backend"],
                    case["timing_boundary"],
                    case["warmup"],
                    reps,
                    trials,
                    transfer_policy=case["transfer_policy"],
                    path_label=case["path_label"],
                    input_family=input_family,
                    input_key=input_key,
                )
            )
            continue
        row: dict[str, Any] = dict.fromkeys(SUMMARY_FIELDS, "")
        row.update(
            {
                "count": count,
                "input_family": input_family,
                "input_key": input_key,
                "bits": bits,
                "backend": case["backend"],
                "boundary": case["timing_boundary"],
                "transfer_policy": case["transfer_policy"],
                "path_label": case["path_label"],
                "correctness": "passed",
                "warmup": case["warmup"],
                "reps": reps,
                "trials": trials,
            }
        )
        if stats is not None:
            for source, destination, precision in (
                ("median_ms", "median_ms", 6),
                ("iqr_ms", "iqr_ms", 6),
                ("trial_median_min_ms", "trial_median_min_ms", 6),
                ("trial_median_max_ms", "trial_median_max_ms", 6),
                ("spread_ratio", "spread_ratio", 4),
                ("spread_p90_p10", "spread_p90_p10", 4),
                ("speedup_vs_cpu", "speedup_vs_c", 4),
                ("speedup_low", "speedup_low", 4),
                ("speedup_high", "speedup_high", 4),
                ("speedup_ci_low", "speedup_ci_low", 4),
                ("speedup_ci_high", "speedup_ci_high", 4),
            ):
                if source in stats:
                    row[destination] = f"{stats[source]:.{precision}f}"
            for source in ("stable", "unstable_rev2", "boundary_inversion"):
                if source in stats:
                    row[source] = csv_flag(stats[source])
            for source in (
                "baseline",
                "verdict",
                "direction_supported",
                "magnitude_supported",
                "claim_supported_rev2",
            ):
                if source in stats:
                    row[source] = (
                        csv_flag(stats[source])
                        if source
                        in {"direction_supported", "magnitude_supported", "claim_supported_rev2"}
                        else stats[source]
                    )
        rows.append(row)
    return rows


def write_summary(
    target_dir: Path,
    setup: RunSetup,
    inputs: SweepInputs,
    case_results: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    settings: SweepSettings,
) -> Path:
    """Sort rows and cases in place, drop the scratch folder, then write summary.csv."""
    path_rank = {path.label: i for i, path in enumerate(settings.paths)}
    input_rank = {key: i for i, key in enumerate(inputs.clean_meta)}
    summary_rows.sort(
        key=lambda r: (
            r["count"],
            input_rank[r["input_key"]],
            r["bits"],
            path_rank[r["path_label"]],
        )
    )
    case_results.sort(
        key=lambda c: (
            c["count"],
            input_rank[c["input_key"]],
            c["bits"],
            path_rank[c["path_label"]],
        )
    )

    shutil.rmtree(setup.temp_dir, ignore_errors=True)

    csv_path = target_dir / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(summary_rows)
    return csv_path


def write_manifest(
    target_dir: Path,
    *,
    date_str: str,
    git_prov: dict[str, Any],
    run_conditions: dict[str, Any],
    setup: RunSetup,
    inputs: SweepInputs,
    order: SweepOrder,
    gpu_state_end: dict[str, Any] | None,
    csv_path: Path,
    vec_report_path: Path,
    case_results: list[dict[str, Any]],
    settings: SweepSettings,
    monitoring: MonitoringMetadata | None,
) -> None:
    toolchain_prov = setup.toolchain_prov
    device_attributes = setup.device_attributes
    input_family = settings.input_family
    input_counts = inputs.counts
    in_process_warmup_seconds = settings.in_process_warmup_seconds
    paths = settings.paths
    dense = settings.dense
    failed_case_ids = [
        case["case_id"] for case in case_results if case["correctness"]["status"] != "passed"
    ]
    conditions = dict(run_conditions)
    reasons = list(conditions["non_evidence_reasons"])
    if failed_case_ids:
        reasons.append("failed benchmark cases: " + ", ".join(failed_case_ids))
    conditions["evidence"] = bool(conditions["evidence"] and not failed_case_ids)
    conditions["non_evidence_reasons"] = reasons
    manifest_data = {
        "manifest_version": "3.2",
        "date": date_str,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "git_provenance": git_prov,
        **({"monitoring": monitoring} if monitoring is not None else {}),
        "hardware": toolchain_prov["hardware"],
        **({"device": device_attributes} if device_attributes is not None else {}),
        "toolkit_and_driver": toolchain_prov["toolkit_and_driver"],
        "build_flags": setup.build_prov,
        "build_stamp": setup.build_stamp,
        "transfer_policies": list(settings.transfer_policies),
        "gpu_state": {
            "note": (
                "Instantaneous nvidia-smi readings before the warm-up, after it, and after "
                "the last timed process; they describe the GPU around the run, not clock "
                "stability during it."
            ),
            "start": setup.gpu_state_start,
            "warmup": order.gpu_warmup,
            "after_warmup": order.gpu_state_after_warmup,
            "end": gpu_state_end,
        },
        "matrix_parameters": {
            "k1": settings.k1,
            "input_family": input_family,
            "counts": (
                list(settings.counts) if input_family != "model" else list(input_counts.values())
            ),
            "model_tensors": (
                (settings.model_tensors or "distinct") if input_family == "model" else None
            ),
            "model_limit": settings.model_limit,
            "bit_widths": list(settings.bit_widths),
            "backends": list(settings.backends),
            "warmup": settings.warmups,
            "in_process_warmup_seconds": in_process_warmup_seconds,
            "in_process_warmup_rule": (
                "per path: max(warmup, ceil(target_seconds * 1000 / probe median ms)), "
                "probe = one untimed stoquant bench process with the base warmup and reps"
                if in_process_warmup_seconds > 0
                else "disabled; every path uses the base warmup"
            ),
            "reps": settings.reps,
            "trials": settings.trials,
            "paths": [path.label for path in paths],
            **(
                {
                    "cpu_avx2_threads": sorted(
                        {c["threads"] for c in case_results if "threads" in c}
                    ),
                    "cpu_avx2_threads_rule": (
                        "stoquant default: logical processors in the inherited affinity mask "
                        "(Windows; online processors elsewhere)"
                    ),
                }
                if settings.has_avx2
                else {}
            ),
            "trial_design": trial_design(len(paths)),
            "trial_orders": order.order_labels,
            "case_order_seed": settings.case_order_seed,
            "case_order": [
                {"count": input_counts[k], "bits": b}
                if dense
                else {"input": k, "count": input_counts[k], "bits": b}
                for k, b in order.ordered_cases
            ],
            "invocation_scheme": (
                "every trial reuses base invocation 0; identifiers per repetition are "
                "copied from each stoquant bench configuration"
            ),
            "input_seed": settings.input_seed,
            "compression_seed": settings.compression_seed,
        },
        "statistics_method": {
            "pooled": "median and IQR over all measured repetitions of all trials",
            "quantile_method": f"numpy.percentile(method='{QUANTILE_METHOD}')",
            "numpy_version": np.__version__,
            "speedup_point": "baseline pooled median / candidate pooled median",
            "speedup_range": (
                "[baseline min trial median / candidate max trial median, "
                "baseline max trial median / candidate min trial median]"
            ),
            "spread_ratio": "max trial median / min trial median",
            "spread_threshold": SPREAD_THRESHOLD,
            "spread_p90_p10": (
                f"p{STABILITY_PERCENTILES[1]} / p{STABILITY_PERCENTILES[0]} of the trial medians"
            ),
            "stability_threshold": STABILITY_THRESHOLD,
            "claim_rules": {
                "verdict": VERDICT_RULE,
                "direction_supported": DIRECTION_RULE,
                "magnitude_supported": MAGNITUDE_RULE,
                "claim_supported_rev2": CLAIM_RULE_REV2,
                "baselines": BASELINE_RULE,
            },
            "bootstrap": {
                "resamples": BOOTSTRAP_RESAMPLES,
                "seed": BOOTSTRAP_SEED,
                "level": BOOTSTRAP_LEVEL,
                "rule": BOOTSTRAP_RULE,
            },
        },
        "run_conditions": conditions,
        # Dense inputs stay keyed by count, as in revision 3.
        "inputs": {
            (meta["count"] if dense else key): meta for key, meta in inputs.clean_meta.items()
        },
        "summary_csv": csv_path.name,
        "msvc_vectorization_report": vec_report_path.name,
        "cases": [c["case_id"] for c in case_results],
        "all_cases_passed": not failed_case_ids,
    }

    write_manifest_record(target_dir / "manifest.json", cast(MatrixManifest, manifest_data))


def main(argv: list[str] | None = None, prog: str | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog=prog, description="stoquant benchmark driver and snapshot creator"
    )
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory to save snapshot")
    parser.add_argument(
        "--counts", type=int, nargs="+", default=DEFAULT_COUNTS, help="Element counts"
    )
    parser.add_argument(
        "--bits", type=int, nargs="+", default=DEFAULT_BITS, help="Bit widths (4 or 8)"
    )
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUPS, help="Warmup runs")
    parser.add_argument("--reps", type=int, default=DEFAULT_REPS, help="Measured repetitions")
    parser.add_argument(
        "--trials",
        type=int,
        default=DEFAULT_TRIALS,
        help="Independent processes per path, each trial in a balanced path order (all permutations, or Williams rows)",
    )
    parser.add_argument(
        "--input-seed", type=int, default=DEFAULT_INPUT_SEED, help="Seed for input generation"
    )
    parser.add_argument(
        "--compression-seed",
        type=int,
        default=DEFAULT_COMPRESSION_SEED,
        help="Seed for compression",
    )
    parser.add_argument(
        "--case-order-seed",
        type=int,
        default=DEFAULT_CASE_ORDER_SEED,
        help="Seed for the random order of (count, bits) cases",
    )
    parser.add_argument(
        "--gpu-warmup-seconds",
        type=float,
        default=DEFAULT_GPU_WARMUP_SECONDS,
        help="Untimed CUDA work before the first timed process (0 disables)",
    )
    parser.add_argument(
        "--case-warmup-seconds",
        type=float,
        default=DEFAULT_CASE_WARMUP_SECONDS,
        help="Untimed CUDA work before each case's timed processes (0 disables)",
    )
    parser.add_argument(
        "--warmup-seconds",
        type=float,
        default=DEFAULT_IN_PROCESS_WARMUP_SECONDS,
        help=(
            "Minimum untimed in-process warm-up per timed process, for every path; "
            "--warmup is the floor (0 disables)"
        ),
    )
    parser.add_argument(
        "--backends",
        type=str,
        nargs="+",
        default=["cpu", "cuda"],
        help=(
            "Backends to benchmark (cpu, cuda, cpu-avx2); cpu-avx2 adds the "
            "descriptive AVX2 comparator path"
        ),
    )
    parser.add_argument("--k1", choices=K1_VARIANTS, default="reference", help="CUDA K1 variant")
    parser.add_argument(
        "--boundaries",
        type=str,
        nargs="*",
        choices=EXTENSION_BOUNDARIES,
        default=[],
        help="Publication extension boundaries to add to the revision 3 paths",
    )
    parser.add_argument(
        "--transfer-policies",
        type=str,
        nargs="+",
        choices=TRANSFER_POLICIES,
        default=["pageable"],
        help="Host transfer policies for transfer boundaries; pinned needs pageable",
    )
    parser.add_argument(
        "--input-family",
        choices=INPUT_FAMILIES,
        default="dense",
        help="Input family: dense normal, sparse (90%% zeros), or model-shaped tensors",
    )
    parser.add_argument(
        "--model-tensors",
        choices=MODEL_TENSOR_SETS,
        default=None,
        help="Model family: one tensor per distinct shape and kind (default), or all tensors",
    )
    parser.add_argument(
        "--model-limit",
        type=int,
        default=None,
        help="Model family: keep only the first N selected tensors",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Run from an uncommitted tree (records the dirty files; not for evidence)",
    )
    parser.add_argument(
        "--pilot",
        action="store_true",
        help=(
            "Non-evidence pilot at four sizes and both bit widths, written under "
            "results/pilots/; records the readiness check without enforcing it"
        ),
    )
    parser.add_argument(
        "--ignore-readiness",
        action="store_true",
        help="Run despite a failed readiness check (marks the snapshot non-evidence)",
    )

    args = parser.parse_args(argv)
    if len(set(args.counts)) != len(args.counts):
        parser.error("--counts values must be unique")
    root = layout.ROOT

    try:
        snapshot_dir = run_benchmark_matrix(
            root=root,
            output_dir=args.output_dir,
            counts=MATRIX_PILOT_COUNTS if args.pilot else args.counts,
            bit_widths=DEFAULT_BITS if args.pilot else args.bits,
            backends=args.backends,
            warmups=args.warmup,
            reps=args.reps,
            trials=args.trials,
            input_seed=args.input_seed,
            compression_seed=args.compression_seed,
            case_order_seed=args.case_order_seed,
            gpu_warmup_seconds=args.gpu_warmup_seconds,
            case_warmup_seconds=args.case_warmup_seconds,
            in_process_warmup_seconds=args.warmup_seconds,
            allow_dirty=args.allow_dirty,
            pilot=args.pilot,
            ignore_readiness=args.ignore_readiness,
            boundaries=args.boundaries,
            transfer_policies=args.transfer_policies,
            input_family=args.input_family,
            model_tensors=args.model_tensors,
            model_limit=args.model_limit,
            k1=args.k1,
        )
        manifest = read_manifest(snapshot_dir / "manifest.json")
        if not manifest["all_cases_passed"]:
            reasons = manifest["run_conditions"]["non_evidence_reasons"]
            print(
                "Benchmark completed with failed cases; snapshot is non-evidence. "
                f"Reasons: {'; '.join(reasons)}. Snapshot: {snapshot_dir}",
                file=sys.stderr,
            )
            sys.exit(1)
        print(f"Benchmark completed successfully. Snapshot written to: {snapshot_dir}")
    except (RuntimeError, ValueError, OSError, subprocess.SubprocessError) as exc:
        print(f"Benchmark driver failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
