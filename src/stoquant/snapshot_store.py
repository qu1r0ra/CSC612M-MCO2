"""Write-once snapshot lifecycle and derived-output locations."""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, TypedDict


class RunPlan(TypedDict):
    total_cases: int
    input_family: str


@dataclass(frozen=True)
class SnapshotProgress:
    snapshot: str | None
    completed_cases: int
    total_cases: int | None
    input_family: str | None


ACTIVE_SNAPSHOT_REGISTRATION_ENV = "STOQUANT_MONITORING_SNAPSHOT_REGISTRATION"


def snapshot_path(
    root: Path,
    revision: str,
    kind: str = "matrix",
    *,
    pilot: bool = False,
    now: datetime | None = None,
) -> Path:
    """Name a run from the host's local date, revision, and experiment kind."""
    stamp = now or datetime.now().astimezone()
    suffix = "" if kind in ("matrix", "dense") else f"-{kind}"
    if pilot:
        return root / "results" / "pilots" / f"{stamp:%Y-%m-%dT%H%M%S}-{revision}{suffix}"
    return root / "results" / f"{stamp:%Y-%m-%d}-{revision}{suffix}"


@dataclass
class Snapshot:
    final: Path
    partial: Path
    committed: bool = False

    def commit(self) -> Path:
        """Publish a complete run without ever replacing an existing snapshot."""
        if self.committed:
            raise RuntimeError("snapshot was already committed")
        if self.final.exists():
            raise FileExistsError(f"Snapshot directory already exists: {self.final}")
        self.partial.rename(self.final)
        self.committed = True
        return self.final


def check_snapshot(target: Path, git_prov: dict[str, Any], *, allow_dirty: bool = False) -> None:
    """Refuse a dirty run or any name already reserved by a run."""
    if git_prov["git_dirty"] and not allow_dirty:
        raise RuntimeError(
            "Working tree is dirty; commit first or pass --allow-dirty for a non-evidence run: "
            + "; ".join(git_prov["dirty_files"])
        )
    if target.exists():
        raise FileExistsError(f"Snapshot directory already exists: {target}")
    partial = target.with_name(target.name + ".partial")
    if partial.exists():
        raise FileExistsError(f"Partial snapshot directory already exists: {partial}")


def create_snapshot(
    target: Path, git_prov: dict[str, Any], *, allow_dirty: bool = False
) -> Snapshot:
    """Apply both gates before reserving a partial directory for one run."""
    check_snapshot(target, git_prov, allow_dirty=allow_dirty)
    partial = target.with_name(target.name + ".partial")
    partial.parent.mkdir(parents=True, exist_ok=True)
    partial.mkdir()
    return Snapshot(target, partial)


def derived_directory(snapshot: Path) -> Path:
    """Replace only regenerateable output beneath a completed snapshot."""
    out = snapshot / "derived"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    return out


def write_run_plan(snapshot: Path, plan: RunPlan) -> None:
    (snapshot / "run-plan.json").write_text(
        json.dumps(plan, indent=2) + "\n", encoding="utf-8", newline="\n"
    )


def register_active_snapshot(snapshot: Path, registration_file: Path | None) -> None:
    """Publish the active snapshot location for a supervising notifier, when present."""
    if registration_file is None:
        return
    registration_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = registration_file.with_name(f"{registration_file.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps({"snapshot": str(snapshot.resolve())}) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(registration_file)


def registered_snapshot(registration_file: Path | None) -> Path | None:
    """Read a driver's active-snapshot registration, ignoring absent or incomplete state."""
    if registration_file is None:
        return None
    try:
        record = json.loads(registration_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(record, dict):
        return None
    path = record.get("snapshot")
    if not isinstance(path, str):
        return None
    snapshot = Path(path)
    return snapshot if snapshot.is_dir() else None


def progress(snapshot: Path | None, root: Path) -> SnapshotProgress | None:
    if snapshot is None or not snapshot.is_dir():
        return None
    completed = len(list(snapshot.glob("case_*.json")))
    plan = snapshot / "run-plan.json"
    total: int | None = None
    family: str | None = None
    if plan.is_file():
        try:
            record = json.loads(plan.read_text(encoding="utf-8"))
            if isinstance(record, dict):
                total_value = record.get("total_cases")
                family_value = record.get("input_family")
                if isinstance(total_value, int) and total_value >= 0:
                    total = total_value
                if isinstance(family_value, str):
                    family = family_value
        except (OSError, ValueError, KeyError, TypeError):
            pass
    try:
        label = snapshot.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        label = None
    return SnapshotProgress(label, completed, total, family)


def discover(root: Path) -> set[Path]:
    """Return completed and in-progress snapshots, including pilot runs."""
    results = root / "results"
    if not results.is_dir():
        return set()
    found = set()
    for path in results.iterdir():
        if not path.is_dir():
            continue
        if path.name == "pilots":
            found.update(child for child in path.iterdir() if child.is_dir())
        else:
            found.add(path)
    return found


def new_since(root: Path, previous: set[Path]) -> Path | None:
    """Find the newest snapshot reserved since the caller's starting point."""
    candidates = discover(root) - previous
    return max(candidates, key=lambda path: path.stat().st_mtime, default=None)
