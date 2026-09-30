"""Write-once snapshot lifecycle and derived-output locations."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


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


def progress(snapshot: Path | None) -> tuple[int | None, int | None]:
    if snapshot is None or not snapshot.is_dir():
        return None, None
    completed = len(list(snapshot.glob("case_*.json")))
    plan = snapshot / "run-plan.json"
    if plan.is_file():
        try:
            total = int(json.loads(plan.read_text(encoding="utf-8"))["total_cases"])
            return completed, total
        except (OSError, ValueError, KeyError, TypeError):
            pass
    manifest = snapshot / "manifest.json"
    if manifest.is_file():
        try:
            return completed, len(json.loads(manifest.read_text(encoding="utf-8"))["cases"])
        except (OSError, ValueError, KeyError, TypeError):
            pass
    return completed, None


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
