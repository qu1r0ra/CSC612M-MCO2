"""Typed JSON contract for matrix case and manifest snapshots."""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, NotRequired, Required, TypedDict, cast

if TYPE_CHECKING:
    from stoquant.inputs import InputProvenance
    from stoquant.monitoring import MonitoringMetadata


class SummaryRow(TypedDict):
    count: int | str
    bits: int | str
    backend: str
    boundary: str
    transfer_policy: str
    path_label: str
    correctness: str
    warmup: int | str
    reps: int | str
    trials: int | str
    median_ms: str
    iqr_ms: str
    trial_median_min_ms: str
    trial_median_max_ms: str
    spread_ratio: str
    spread_p90_p10: str
    stable: str
    unstable_rev2: str
    baseline: str
    speedup_vs_c: str
    speedup_low: str
    speedup_high: str
    speedup_ci_low: str
    speedup_ci_high: str
    verdict: str
    boundary_inversion: str
    direction_supported: str
    magnitude_supported: str
    claim_supported_rev2: str
    input_family: str
    input_key: str


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


class CaseRecord(TypedDict, total=False):
    case_id: Required[str]
    input_key: str
    input_family: str
    count: int
    bits: int
    backend: str
    timing_boundary: str
    transfer_policy: str
    path_label: str
    configuration: dict[str, Any]
    seed: int
    tensor_id: int
    invocation_id: int
    warmup: int
    in_process_warmup: int | None
    reps: int
    trials: int
    header_bytes: int
    payload_bytes: int
    code_revision: str
    code_revision_short: str
    git_dirty: bool
    build_flags: dict[str, Any]
    hardware: dict[str, Any]
    toolkit_and_driver: dict[str, Any]
    correctness: dict[str, Any]
    execution_index: int
    case_warmup: dict[str, Any] | None
    samples_ms: list[float]
    statistics: dict[str, Any] | None
    trial_runs: list[dict[str, Any]]
    stage_medians_ms: dict[str, float] | None
    input_provenance: InputProvenance


class MatrixManifest(TypedDict):
    manifest_version: NotRequired[str]
    date: NotRequired[str]
    created_at_utc: NotRequired[str]
    git_provenance: NotRequired[dict[str, Any]]
    monitoring: NotRequired[MonitoringMetadata]
    hardware: NotRequired[dict[str, Any]]
    device: NotRequired[dict[str, Any]]
    toolkit_and_driver: NotRequired[dict[str, Any]]
    build_flags: NotRequired[dict[str, Any]]
    build_stamp: NotRequired[dict[str, Any]]
    transfer_policies: NotRequired[list[str]]
    gpu_state: NotRequired[dict[str, Any]]
    matrix_parameters: NotRequired[dict[str, Any]]
    cpu_comparison_rule: NotRequired[str]
    timing_contract: NotRequired[str]
    statistic_definition: NotRequired[str]
    comparison_rules: NotRequired[dict[str, str]]
    cases: list[str]
    summary_csv: NotRequired[str]
    msvc_vectorization_report: NotRequired[str]
    vectorization_rule: NotRequired[str]
    all_cases_passed: bool
    failed_cases: NotRequired[list[str]]
    run_conditions: dict[str, Any]
    statistics_method: NotRequired[dict[str, Any]]
    inputs: NotRequired[dict[str | int, InputProvenance]]


def write_summary_csv(path: Path, rows: Sequence[SummaryRow]) -> Path:
    """Write typed matrix summary rows using the shared publication column order."""
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _write_json(path: Path, record: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(record, indent=2),
        encoding="utf-8",
        newline="\n",
    )


def _read_record(path: Path, description: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{description} must contain a JSON object: {path}")
    return value


def write_case(path: Path, case: CaseRecord) -> None:
    _write_json(path, cast(dict[str, Any], case))


def read_case(path: Path) -> CaseRecord:
    return cast(CaseRecord, _read_record(path, "case record"))


def write_manifest(path: Path, manifest: MatrixManifest) -> None:
    _write_json(path, cast(dict[str, Any], manifest))


def read_manifest(path: Path) -> MatrixManifest:
    return cast(MatrixManifest, _read_record(path, "matrix manifest"))


def read_snapshot(snapshot: Path) -> tuple[MatrixManifest, list[CaseRecord]]:
    manifest = read_manifest(snapshot / "manifest.json")
    case_ids = manifest.get("cases")
    if not isinstance(case_ids, list) or any(
        not isinstance(case_id, str) or Path(case_id).name != case_id for case_id in case_ids
    ):
        raise ValueError(f"matrix manifest has no case list: {snapshot / 'manifest.json'}")
    return manifest, [read_case(snapshot / f"{case_id}.json") for case_id in case_ids]
