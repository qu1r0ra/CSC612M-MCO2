"""Benchmark driver for in-process stoquant CPU comparator and CUDA pipelines.

Orchestrates input generation, per-case correctness gating, timed benchmark
runs, provenance collection, statistical summary computation, and snapshot
creation as specified by the course technical contract and benchmark protocol.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

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
    EXCLUDED_LOGICAL_CPUS,
    MAX_UPTIME_SECONDS,
    WINDOW_ALLOWLIST,
    affinity_mask_excluding,
    check_readiness,
    get_process_affinity,
    probe_readiness_facts,
    process_affinity,
)
from stoquant.inputs import (
    DEFAULT_INPUT_SEED,
    INPUT_FAMILIES,
    MODEL_TENSOR_SETS,
    generate_family_inputs,
    select_model_tensors,
)
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
    in_process_warmups,
    run_bench_process,
    warm_up_gpu,
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
PILOT_COUNTS = (1 << 10, 1 << 14, 1 << 20, 1 << 26)


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
    row = dict.fromkeys(SUMMARY_FIELDS, "")
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
    force_fail: bool = False,
    allow_existing: bool = False,
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
    previous_mask = get_process_affinity()
    mask = affinity_mask_excluding(EXCLUDED_LOGICAL_CPUS, previous_mask)
    with process_affinity(mask):
        git_prov = collect_git_provenance(root)

        # Determine snapshot directory name
        now = datetime.now(UTC)
        date_str = now.strftime("%Y-%m-%d")
        short_rev = git_prov["code_revision_short"]
        if output_dir is not None:
            target_dir = output_dir
        elif pilot:
            target_dir = root / "results" / "pilots" / f"{now:%Y-%m-%dT%H%M%S}-{short_rev}"
        else:
            suffix = "" if input_family == "dense" else f"-{input_family}"
            target_dir = root / "results" / f"{date_str}-{short_rev}{suffix}"

        if target_dir.exists() and not allow_existing:
            raise FileExistsError(
                f"Snapshot directory already exists: {target_dir}. "
                "Snapshots are frozen and never overwritten. "
                "Pass --allow-existing or an explicit --output-dir if intentional."
            )

        if git_prov["git_dirty"] and not allow_dirty:
            raise RuntimeError(
                "Working tree is dirty; a snapshot must be traceable to a committed revision. "
                "Commit or remove these changes, or pass --allow-dirty for a non-evidence run: "
                + "; ".join(git_prov["dirty_files"])
            )

        # Revision 3: judge the machine before any benchmark process starts. A pilot is
        # non-evidence by design, so it records the check without enforcing it.
        facts = probe_readiness_facts(root) if readiness_facts is None else readiness_facts
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
            "excluded_logical_cpus": list(EXCLUDED_LOGICAL_CPUS),
            "mask": hex(mask),
            "previous_mask": hex(previous_mask),
            "applied_to": "driver process before any benchmark or probe process; inherited",
        }
        return sweep_matrix(
            root=root,
            target_dir=target_dir,
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
            force_fail=force_fail,
            input_family=input_family,
            model_tensors=model_tensors,
            model_limit=model_limit,
            k1=k1,
        )


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
    force_fail: bool,
    input_family: str = "dense",
    model_tensors: str | None = None,
    model_limit: int | None = None,
    k1: str = "reference",
) -> Path:
    binary = find_binary(root)
    toolchain_prov = collect_hardware_and_toolchain(root)
    build_prov = collect_build_commands(root)
    host_tokens = build_prov.pop("_host_tokens")
    avx2_tokens = build_prov.pop("_avx2_tokens")
    has_avx2 = any(path.backend == "cpu-avx2" for path in paths)
    publication = len(paths) == 10 and k1 == "optimized"
    device_attributes = query_device_attributes(root) if publication else None
    gpu_state_start = query_gpu_state() if "cuda" in backends else None

    target_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = target_dir / "_temp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    # 1. Inputs generation
    input_meta = generate_family_inputs(
        input_family,
        counts,
        temp_dir / "inputs",
        input_seed,
        model_tensors or "distinct",
        model_limit,
    )
    input_files: dict[str, Path] = {key: Path(meta["_path"]) for key, meta in input_meta.items()}
    clean_input_meta: dict[str, dict[str, Any]] = {
        key: {k: v for k, v in meta.items() if not k.startswith("_")}
        for key, meta in input_meta.items()
    }
    input_counts = {key: meta["count"] for key, meta in input_meta.items()}
    dense = input_family == "dense"

    # 2. Vectorization report, compiled with the comparator's own flags
    vec_report_path = target_dir / "msvc_vectorization_report.txt"
    generate_msvc_vectorization_report(root, vec_report_path, host_tokens, temp_dir / "vec_obj")
    if has_avx2:
        generate_msvc_vectorization_report(
            root,
            target_dir / "msvc_vectorization_report_avx2.txt",
            avx2_tokens,
            temp_dir / "vec_obj_avx2",
            AVX2_SOURCES,
        )

    orders = trial_orders(paths, trials)
    order_labels = [[paths[i].label for i in order] for order in orders]
    ordered_cases = case_order(list(input_meta), bit_widths, case_order_seed)
    largest_input = max(input_meta, key=lambda key: input_counts[key])

    # Bring the GPU to steady clocks on the largest input before any timed process.
    gpu_warmup = None
    gpu_state_after_warmup = None
    if "cuda" in backends and gpu_warmup_seconds > 0:
        gpu_warmup = warm_up_gpu(binary, input_files[largest_input], gpu_warmup_seconds, k1)
        gpu_state_after_warmup = query_gpu_state()

    # 3. Benchmark cases
    case_results: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for execution_index, (input_key, bits) in enumerate(ordered_cases):
        input_path = input_files[input_key]
        count = input_counts[input_key]
        payload_bytes = count if bits == 8 else (count + 1) // 2

        # Gate on correctness once, outside every timed process.
        passed, correctness_info = verify_correctness(
            binary=binary,
            input_path=input_path,
            count=count,
            bits=bits,
            backends=backends,
            paths=paths,
            seed=compression_seed,
            tensor_id=0,
            invocation_id=0,
            tmp_dir=temp_dir,
            force_fail=force_fail,
            k1=k1,
        )

        cases: list[dict[str, Any]] = []
        for path in paths:
            cases.append(
                {
                    "case_id": f"case_{path.backend}_{path.key}_bits{bits}_{input_key}",
                    "count": count,
                    "input_family": input_family,
                    "input_key": input_key,
                    "bits": bits,
                    "backend": path.backend,
                    "timing_boundary": path.boundary,
                    "transfer_policy": path.policy,
                    "path_label": path.label,
                    "configuration": {"k1": k1 if path.backend == "cuda" else None},
                    "seed": compression_seed,
                    "tensor_id": 0,
                    "invocation_id": 0,
                    "warmup": warmups,
                    "in_process_warmup": None,
                    "reps": reps,
                    "trials": trials,
                    "header_bytes": HEADER_STRUCT.size,
                    "payload_bytes": payload_bytes,
                    "code_revision": git_prov["code_revision"],
                    "code_revision_short": git_prov["code_revision_short"],
                    "git_dirty": git_prov["git_dirty"],
                    "build_flags": {
                        "comparator_c": build_prov["comparator_c"],
                        "cuda_nvcc": build_prov["cuda_nvcc"],
                        **({"avx2_c": build_prov["avx2_c"]} if has_avx2 else {}),
                    },
                    "hardware": toolchain_prov["hardware"],
                    "toolkit_and_driver": toolchain_prov["toolkit_and_driver"],
                    "input_provenance": clean_input_meta[input_key],
                    "correctness": correctness_info,
                    "execution_index": execution_index,
                    "case_warmup": None,
                    "trial_runs": [],
                    "samples_ms": [],
                    "statistics": None,
                    "stage_medians_ms": None,
                }
            )

        if passed and "cuda" in backends and case_warmup_seconds > 0:
            # Restore GPU clocks after the untimed correctness gate and any
            # long CPU processes of the previous case.
            case_warmup = warm_up_gpu(binary, input_path, case_warmup_seconds, k1)
            case_warmup["gpu_state_after"] = query_gpu_state()
            for case in cases:
                case["case_warmup"] = case_warmup

        if passed and in_process_warmup_seconds > 0:
            # Revision 2: size each path's in-process warm-up from an untimed probe
            # so every timed process first runs about the same wall time untimed.
            for case, path in zip(cases, paths, strict=True):
                payload, error = run_bench_process(
                    binary,
                    input_path,
                    bits=bits,
                    backend=path.backend,
                    extra_args=path.extra_args,
                    seed=compression_seed,
                    warmups=warmups,
                    reps=reps,
                    k1=k1,
                )
                if payload is None:
                    case["correctness"] = {"status": "failed", "error_message": error}
                    continue
                probe_ms = float(np.median(payload.get("samples_ms", [])))
                case["warmup"] = in_process_warmups(warmups, in_process_warmup_seconds, probe_ms)
                case["in_process_warmup"] = {
                    "target_seconds": in_process_warmup_seconds,
                    "probe_warmup": warmups,
                    "probe_reps": reps,
                    "probe_median_ms": probe_ms,
                    "warmup": case["warmup"],
                }

        if passed:
            # Each trial runs every path as its own process, in a rotated order.
            # Every trial reuses the same invocation identifiers, so trials are
            # replicates of an identical workload.
            for trial_index, order in enumerate(orders):
                for position, path_index in enumerate(order):
                    case = cases[path_index]
                    if case["correctness"]["status"] != "passed":
                        continue
                    path = paths[path_index]
                    payload, error = run_bench_process(
                        binary,
                        input_path,
                        bits=bits,
                        backend=path.backend,
                        extra_args=path.extra_args,
                        seed=compression_seed,
                        warmups=case["warmup"],
                        reps=reps,
                        k1=k1,
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
                        "order": order_labels[trial_index],
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

        for case in cases:
            (target_dir / f"{case['case_id']}.json").write_text(
                json.dumps(case, indent=2), encoding="utf-8"
            )
            case_results.append(case)
            stats = case["statistics"]
            if case["correctness"]["status"] != "passed" or stats is None or "verdict" not in stats:
                summary_rows.append(
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
            summary_rows.append(
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
                    "median_ms": f"{stats['median_ms']:.6f}",
                    "iqr_ms": f"{stats['iqr_ms']:.6f}",
                    "trial_median_min_ms": f"{stats['trial_median_min_ms']:.6f}",
                    "trial_median_max_ms": f"{stats['trial_median_max_ms']:.6f}",
                    "spread_ratio": f"{stats['spread_ratio']:.4f}",
                    "spread_p90_p10": f"{stats['spread_p90_p10']:.4f}",
                    "stable": csv_flag(stats["stable"]),
                    "unstable_rev2": csv_flag(stats["unstable_rev2"]),
                    "baseline": stats["baseline"],
                    "speedup_vs_c": f"{stats['speedup_vs_cpu']:.4f}",
                    "speedup_low": f"{stats['speedup_low']:.4f}",
                    "speedup_high": f"{stats['speedup_high']:.4f}",
                    "speedup_ci_low": f"{stats['speedup_ci_low']:.4f}",
                    "speedup_ci_high": f"{stats['speedup_ci_high']:.4f}",
                    "verdict": stats["verdict"],
                    "boundary_inversion": csv_flag(stats["boundary_inversion"]),
                    "direction_supported": csv_flag(stats["direction_supported"]),
                    "magnitude_supported": csv_flag(stats["magnitude_supported"]),
                    "claim_supported_rev2": csv_flag(stats["claim_supported_rev2"]),
                }
            )

    gpu_state_end = query_gpu_state() if "cuda" in backends else None

    path_rank = {path.label: i for i, path in enumerate(paths)}
    input_rank = {key: i for i, key in enumerate(input_meta)}
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

    shutil.rmtree(temp_dir, ignore_errors=True)

    # 4. Summary CSV
    csv_path = target_dir / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(summary_rows)

    # 5. Run manifest
    monitoring_provider = os.environ.get("STOQUANT_MONITORING_PROVIDER")
    monitoring = None
    if monitoring_provider:
        try:
            heartbeat_interval = int(os.environ["STOQUANT_MONITORING_HEARTBEAT_SECONDS"])
        except (KeyError, ValueError) as exc:
            raise RuntimeError(
                "Invalid run-monitoring metadata in the process environment."
            ) from exc
        if heartbeat_interval < 1:
            raise RuntimeError("Invalid run-monitoring metadata in the process environment.")
        monitoring = {
            "provider": monitoring_provider,
            "heartbeat_interval_seconds": heartbeat_interval,
        }

    manifest_data = {
        "manifest_version": "3.1",
        "date": date_str,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "git_provenance": git_prov,
        **({"monitoring": monitoring} if monitoring is not None else {}),
        "hardware": toolchain_prov["hardware"],
        **({"device": device_attributes} if device_attributes is not None else {}),
        "toolkit_and_driver": toolchain_prov["toolkit_and_driver"],
        "build_flags": build_prov,
        "transfer_policies": list(transfer_policies),
        "gpu_state": {
            "note": (
                "Instantaneous nvidia-smi readings before the warm-up, after it, and after "
                "the last timed process; they describe the GPU around the run, not clock "
                "stability during it."
            ),
            "start": gpu_state_start,
            "warmup": gpu_warmup,
            "after_warmup": gpu_state_after_warmup,
            "end": gpu_state_end,
        },
        "matrix_parameters": {
            "k1": k1,
            "input_family": input_family,
            "counts": list(counts) if input_family != "model" else list(input_counts.values()),
            "model_tensors": (model_tensors or "distinct") if input_family == "model" else None,
            "model_limit": model_limit,
            "bit_widths": list(bit_widths),
            "backends": list(backends),
            "warmup": warmups,
            "in_process_warmup_seconds": in_process_warmup_seconds,
            "in_process_warmup_rule": (
                "per path: max(warmup, ceil(target_seconds * 1000 / probe median ms)), "
                "probe = one untimed stoquant bench process with the base warmup and reps"
                if in_process_warmup_seconds > 0
                else "disabled; every path uses the base warmup"
            ),
            "reps": reps,
            "trials": trials,
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
                if has_avx2
                else {}
            ),
            "trial_design": trial_design(len(paths)),
            "trial_orders": order_labels,
            "case_order_seed": case_order_seed,
            "case_order": [
                {"count": input_counts[k], "bits": b}
                if dense
                else {"input": k, "count": input_counts[k], "bits": b}
                for k, b in ordered_cases
            ],
            "invocation_scheme": (
                "every trial reuses base invocation 0; identifiers per repetition are "
                "copied from each stoquant bench configuration"
            ),
            "input_seed": input_seed,
            "compression_seed": compression_seed,
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
        "run_conditions": run_conditions,
        # Dense inputs stay keyed by count, as in revision 3.
        "inputs": {
            (meta["count"] if dense else key): meta for key, meta in clean_input_meta.items()
        },
        "summary_csv": csv_path.name,
        "msvc_vectorization_report": vec_report_path.name,
        "cases": [c["case_id"] for c in case_results],
        "all_cases_passed": all(c["correctness"]["status"] == "passed" for c in case_results),
    }

    manifest_path = target_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")

    return target_dir


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
        help="Independent processes per path, each trial in a rotated path order",
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
        "--force-fail", action="store_true", help="Force correctness failure for tests"
    )
    parser.add_argument(
        "--allow-existing", action="store_true", help="Allow writing into existing folder"
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
    root = layout.ROOT

    try:
        snapshot_dir = run_benchmark_matrix(
            root=root,
            output_dir=args.output_dir,
            counts=PILOT_COUNTS if args.pilot else args.counts,
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
            force_fail=args.force_fail,
            allow_existing=args.allow_existing,
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
        print(f"Benchmark completed successfully. Snapshot written to: {snapshot_dir}")
    except (RuntimeError, ValueError, OSError, subprocess.SubprocessError) as exc:
        print(f"Benchmark driver failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
