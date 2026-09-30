import json
from datetime import UTC, datetime

import pytest

from stoquant.snapshot_store import (
    create_snapshot,
    derived_directory,
    progress,
    snapshot_path,
)

CLEAN = {"git_dirty": False, "dirty_files": []}


def test_names_include_local_date_revision_kind_and_pilot(tmp_path):
    local = UTC
    now = datetime(2026, 9, 30, 0, 15, tzinfo=local)
    assert snapshot_path(tmp_path, "abc123", "matrix", now=now).name == "2026-09-30-abc123"
    assert snapshot_path(tmp_path, "abc123", "sparse", now=now).name == "2026-09-30-abc123-sparse"
    pilot = snapshot_path(tmp_path, "abc123", "k1-ab", pilot=True, now=now)
    assert pilot.parent.name == "pilots"
    assert pilot.name == "2026-09-30T001500-abc123-k1-ab"


def test_store_refuses_existing_and_dirty_runs(tmp_path):
    target = tmp_path / "run"
    with pytest.raises(RuntimeError, match="dirty"):
        create_snapshot(target, {"git_dirty": True, "dirty_files": ["file.py"]})
    assert not target.exists()
    target.mkdir()
    with pytest.raises(FileExistsError, match="already exists"):
        create_snapshot(target, CLEAN)


def test_failure_preserves_partial_then_commit_renames_atomically(tmp_path):
    target = tmp_path / "run"
    first = create_snapshot(target, CLEAN)
    (first.partial / "case_0.json").write_text("{}", encoding="utf-8")
    assert first.partial.is_dir() and not target.exists()
    with pytest.raises(FileExistsError, match="Partial"):
        create_snapshot(target, CLEAN)
    first.commit()
    assert target.joinpath("case_0.json").read_bytes() == b"{}"
    assert not first.partial.exists()
    with pytest.raises(RuntimeError, match="already committed"):
        first.commit()


def test_derived_outputs_replace_only_derived_and_progress_uses_plan(tmp_path):
    snapshot = create_snapshot(tmp_path / "run", CLEAN)
    raw = snapshot.partial / "manifest.json"
    raw.write_text(json.dumps({"cases": [1, 2, 3]}), encoding="utf-8")
    (snapshot.partial / "run-plan.json").write_text(
        json.dumps({"total_cases": 4}), encoding="utf-8"
    )
    (snapshot.partial / "case_0.json").write_text("{}", encoding="utf-8")
    assert progress(snapshot.partial) == (1, 4)
    before = raw.read_bytes()
    derived = derived_directory(snapshot.partial)
    (derived / "report.md").write_text("old", encoding="utf-8")
    assert derived_directory(snapshot.partial) == derived
    assert not (derived / "report.md").exists()
    assert raw.read_bytes() == before
