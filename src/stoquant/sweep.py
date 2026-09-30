"""Benchmark driver for in-process stoquant CPU comparator and CUDA pipelines.

Orchestrates input generation, per-case correctness gating, timed benchmark
runs, provenance collection, statistical summary computation, and snapshot
creation as specified by the course technical contract and benchmark protocol.
"""

from __future__ import annotations

import math
import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NamedTuple, cast

import numpy as np

from stoquant.correctness import verify_correctness
from stoquant.design import (
    PUBLICATION_PATHS,
    BenchPath,
    case_order,
    trial_design,
    trial_orders,
)
from stoquant.inputs import (
    InputProvenance,
    generate_family_inputs,
)
from stoquant.monitoring import MonitoringMetadata
from stoquant.native_build import source_lists
from stoquant.oracle import HEADER_STRUCT
from stoquant.provenance import (
    collect_build_commands,
    collect_hardware_and_toolchain,
    find_binary,
    query_device_attributes,
    query_gpu_state,
)
from stoquant.runner import (
    BenchProcess,
    in_process_warmups,
    run_bench_process,
    warm_up_gpu,
)
from stoquant.schema import (
    SUMMARY_FIELDS,
    CaseRecord,
    MatrixManifest,
    SummaryRow,
    write_case,
    write_summary_csv,
)
from stoquant.schema import (
    write_manifest as write_manifest_record,
)
from stoquant.snapshot_store import (
    RunPlan,
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
from stoquant.vectorization import generate_msvc_vectorization_report


@dataclass(frozen=True)
class SweepSettings:
    """All inputs and output context required by one sweep."""

    root: Path
    target_dir: Path
    date_str: str
    git_prov: dict[str, Any]
    run_conditions: dict[str, Any]
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
    monitoring: MonitoringMetadata | None

    @property
    def cuda(self) -> bool:
        return "cuda" in self.backends

    @property
    def has_avx2(self) -> bool:
        return any(path.backend == "cpu-avx2" for path in self.paths)

    @property
    def dense(self) -> bool:
        return self.input_family == "dense"

    @property
    def requires_publication_device_attributes(self) -> bool:
        return self.k1 == "optimized" and tuple(self.paths) == PUBLICATION_PATHS


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
) -> SummaryRow:
    row = empty_summary_row()
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


def empty_summary_row() -> SummaryRow:
    return cast(SummaryRow, dict.fromkeys(SUMMARY_FIELDS, ""))


def csv_flag(value: bool | None) -> str:
    return "" if value is None else str(value).lower()


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


class Sweep:
    """Run a matrix sweep from one validated, immutable settings object."""

    def __init__(self, settings: SweepSettings):
        self.settings = settings

    def run(self) -> Path:
        settings = self.settings
        root = settings.root
        target_dir = settings.target_dir
        git_prov = settings.git_prov
        setup = set_up_run(root, target_dir, settings, git_prov)
        inputs = generate_inputs(setup, settings)
        run_plan: RunPlan = {
            "total_cases": len(inputs.clean_meta) * len(settings.bit_widths) * len(settings.paths),
            "input_family": settings.input_family,
        }
        write_run_plan(target_dir, run_plan)
        vec_report_path = write_vectorization_reports(root, target_dir, setup, settings)
        order = order_and_warm_up(setup, inputs, settings)

        case_results: list[dict[str, Any]] = []
        summary_rows: list[SummaryRow] = []
        for execution_index, (input_key, bits) in enumerate(order.ordered_cases):
            passed, correctness_info = verify_correctness(
                binary=setup.binary,
                input_path=inputs.files[input_key],
                count=inputs.counts[input_key],
                bits=bits,
                backends=settings.backends,
                paths=settings.paths,
                seed=settings.compression_seed,
                tensor_id=0,
                invocation_id=0,
                tmp_dir=setup.temp_dir,
                process=settings.bench_process,
                k1=settings.k1,
            )
            cases = build_case_stubs(
                setup,
                git_prov,
                inputs,
                input_key,
                bits,
                correctness_info,
                execution_index,
                settings,
            )
            if passed:
                warm_up_case(cases, setup.binary, inputs.files[input_key], bits, settings)
                run_trials(cases, setup.binary, inputs.files[input_key], bits, order, settings)
            compute_case_group_statistics(cases, settings.paths)
            summary_rows.extend(
                write_case_outputs(target_dir, cases, inputs, input_key, bits, settings)
            )
            case_results.extend(cases)

        gpu_state_end = query_gpu_state() if settings.cuda else None
        path_rank = {path.label: i for i, path in enumerate(settings.paths)}
        input_rank = {key: i for i, key in enumerate(inputs.clean_meta)}
        summary_rows.sort(
            key=lambda row: (
                row["count"],
                input_rank[row["input_key"]],
                row["bits"],
                path_rank[row["path_label"]],
            )
        )
        case_results.sort(
            key=lambda case: (
                case["count"],
                input_rank[case["input_key"]],
                case["bits"],
                path_rank[case["path_label"]],
            )
        )
        shutil.rmtree(setup.temp_dir, ignore_errors=True)
        csv_path = write_summary_csv(target_dir / "summary.csv", summary_rows)
        write_manifest(
            target_dir,
            date_str=settings.date_str,
            git_prov=git_prov,
            run_conditions=settings.run_conditions,
            setup=setup,
            inputs=inputs,
            order=order,
            gpu_state_end=gpu_state_end,
            csv_path=csv_path,
            vec_report_path=vec_report_path,
            case_results=case_results,
            settings=settings,
            monitoring=settings.monitoring,
        )
        return target_dir


def fail_case_execution(case: dict[str, Any], phase: str, error: str) -> None:
    """Keep the complete correctness-gate record when later work fails."""
    correctness = case["correctness"]
    correctness.setdefault("gate_status", correctness.get("status"))
    if "error_message" in correctness:
        correctness.setdefault("gate_error_message", correctness["error_message"])
    correctness["status"] = "failed"
    correctness["failure_phase"] = phase
    correctness["error_message"] = error


def set_up_run(
    root: Path, target_dir: Path, settings: SweepSettings, git_prov: dict[str, Any]
) -> RunSetup:
    """Binary, provenance and starting GPU state, then the snapshot and scratch folders."""
    verified = find_binary(root, require_cuda=settings.cuda, git_prov=git_prov)
    toolchain_prov = collect_hardware_and_toolchain(root)
    build_commands = collect_build_commands(verified.stamp)
    build_prov = build_commands.as_record()
    host_tokens = build_commands.host_tokens or []
    avx2_tokens = build_commands.avx2_tokens or []
    device_attributes = (
        query_device_attributes(root) if settings.requires_publication_device_attributes else None
    )
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
    sources = source_lists(root)
    vec_report_path = target_dir / "msvc_vectorization_report.txt"
    generate_msvc_vectorization_report(
        root,
        vec_report_path,
        setup.host_tokens,
        setup.temp_dir / "vec_obj",
        sources["native_host_sources"],
    )
    if settings.has_avx2:
        generate_msvc_vectorization_report(
            root,
            target_dir / "msvc_vectorization_report_avx2.txt",
            setup.avx2_tokens,
            setup.temp_dir / "vec_obj_avx2",
            sources["native_avx2_sources"],
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
                "correctness": dict(correctness_info),
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
        try:
            case_warmup = warm_up_gpu(
                binary,
                input_path,
                settings.case_warmup_seconds,
                settings.k1,
                settings.bench_process,
            )
        except (RuntimeError, OSError, subprocess.SubprocessError) as exc:
            for case in cases:
                fail_case_execution(case, "gpu-warmup", str(exc))
            return
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
                fail_case_execution(
                    case, "calibration-warmup", error or "benchmark calibration failed"
                )
                continue
            samples = payload.get("samples_ms")
            if (
                not isinstance(samples, list)
                or not samples
                or not all(
                    isinstance(sample, (int, float)) and math.isfinite(sample) for sample in samples
                )
            ):
                fail_case_execution(
                    case,
                    "calibration-warmup",
                    "calibration probe returned empty or non-finite samples",
                )
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
                fail_case_execution(case, "trial", error or "benchmark trial failed")
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
) -> list[SummaryRow]:
    """Write each case's JSON; returns the case group's summary rows."""
    count = inputs.counts[input_key]
    input_family = settings.input_family
    reps = settings.reps
    trials = settings.trials
    rows: list[SummaryRow] = []
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
        row = empty_summary_row()
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
                    value = stats[source]
                    if isinstance(value, (int, float)) and math.isfinite(value):
                        row[destination] = f"{value:.{precision}f}"
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
