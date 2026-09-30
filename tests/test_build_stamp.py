"""The binary locator refuses build facts it cannot verify."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from stoquant.build_stamp import compiler_commands, recipe_commands, sha256, stamp_path
from stoquant.provenance import find_binary


@pytest.fixture
def stamped_binary(tmp_path: Path) -> tuple[Path, Path, dict[str, object]]:
    (tmp_path / "justfile").write_text(
        "build-cpu:\n    echo cl.exe /O2 /c native/main.c\n"
        "build-cuda:\n    echo nvcc -O2 -DSQ_ENABLE_CUDA native/main.cu\n"
        "build-cuda-fault:\n    echo nvcc -DSQ_CUDA_FAULT_INJECTION native/main.cu\n",
        encoding="utf-8",
    )
    binary = tmp_path / "build" / "stoquant.exe"
    binary.parent.mkdir()
    binary.write_bytes(b"verified binary")
    commands = recipe_commands(tmp_path, "build-cuda")
    stamp = {
        "version": 1,
        "revision": "a" * 40,
        "tree_dirty": False,
        "tree_fingerprint": "clean tree",
        "recipe": "build-cuda",
        "commands": commands,
        "compiler_flags": compiler_commands(commands),
        "binary_sha256": sha256(binary),
    }
    stamp_path(binary).write_text(json.dumps(stamp), encoding="utf-8")
    git = {
        "code_revision": "a" * 40,
        "git_dirty": False,
        "tree_fingerprint": "clean tree",
    }
    return tmp_path, binary, git


def test_locator_returns_verified_stamp(stamped_binary):
    root, binary, git = stamped_binary
    verified = find_binary(root, git_prov=git)
    assert verified.path == binary
    assert verified.stamp["binary_sha256"] == sha256(binary)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("missing", "build stamp missing"),
        ("stale", "revision is stale"),
        ("hash", "binary hash does not match"),
        ("cpu", "cannot serve CUDA paths"),
        ("fault", "test-only build define"),
    ],
)
def test_locator_refuses_unverifiable_build(stamped_binary, change, message):
    root, binary, git = stamped_binary
    path = stamp_path(binary)
    stamp = json.loads(path.read_text(encoding="utf-8"))
    if change == "missing":
        path.unlink()
    elif change == "stale":
        stamp["revision"] = "b" * 40
    elif change == "hash":
        binary.write_bytes(b"changed binary")
    else:
        recipe = "build-cpu" if change == "cpu" else "build-cuda-fault"
        commands = recipe_commands(root, recipe)
        stamp.update(
            recipe=recipe,
            commands=commands,
            compiler_flags=compiler_commands(commands),
        )
    if change not in ("missing", "hash"):
        path.write_text(json.dumps(stamp), encoding="utf-8")
    with pytest.raises(RuntimeError, match=message):
        find_binary(root, git_prov=git)
