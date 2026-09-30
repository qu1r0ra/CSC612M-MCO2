"""Write and verify native build stamps beside their binaries."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

STAMP_VERSION = 1
TEST_ONLY_DEFINES = ("SQ_CUDA_FAULT_INJECTION",)


def _git(root: Path, *args: str) -> bytes:
    return subprocess.run(["git", *args], cwd=root, capture_output=True, check=True).stdout


def tree_fingerprint(root: Path) -> str:
    """Identify the exact tracked diff and untracked content in a dirty build."""
    digest = hashlib.sha256()
    digest.update(_git(root, "diff", "--binary", "HEAD"))
    for relative in _git(root, "ls-files", "--others", "--exclude-standard", "-z").split(b"\0"):
        if not relative:
            continue
        path = root / relative.decode("utf-8", errors="surrogateescape")
        digest.update(relative)
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def stamp_path(binary: Path) -> Path:
    return binary.with_name(binary.name + ".build-stamp.json")


def recipe_commands(root: Path, recipe: str) -> list[str]:
    result = subprocess.run(
        ["just", "--dry-run", recipe], cwd=root, capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        raise RuntimeError(f"just --dry-run {recipe} failed: {result.stderr.strip()}")
    output = result.stderr if result.stderr.strip() else result.stdout
    return [line.strip() for line in output.splitlines() if line.strip()]


def compiler_commands(commands: list[str]) -> list[str]:
    return [
        line
        for line in commands
        if any(tool in line.lower() for tool in ("cl.exe ", "nvcc ", "gcc ", "clang ", "cc "))
    ]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_stamp(root: Path, binary: Path, recipe: str) -> dict[str, Any]:
    if not binary.is_file():
        raise FileNotFoundError(binary)
    commands = recipe_commands(root, recipe)
    status = _git(root, "status", "--porcelain", "--untracked-files=all")
    stamp = {
        "version": STAMP_VERSION,
        "revision": _git(root, "rev-parse", "HEAD").decode().strip(),
        "tree_dirty": bool(status.strip()),
        "tree_fingerprint": tree_fingerprint(root),
        "recipe": recipe,
        "commands": commands,
        "compiler_flags": compiler_commands(commands),
        "binary_sha256": sha256(binary),
    }
    stamp_path(binary).write_text(json.dumps(stamp, indent=2) + "\n", encoding="utf-8")
    return stamp


def verify_stamp(
    root: Path,
    binary: Path,
    *,
    revision: str,
    tree_dirty: bool,
    fingerprint: str,
    require_cuda: bool,
) -> dict[str, Any]:
    path = stamp_path(binary)
    if not path.is_file():
        raise RuntimeError(f"build stamp missing for {binary}; rebuild the binary")
    try:
        stamp = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"invalid build stamp at {path}") from exc
    if not isinstance(stamp, dict) or stamp.get("version") != STAMP_VERSION:
        raise RuntimeError(f"unsupported build stamp at {path}")
    if stamp.get("binary_sha256") != sha256(binary):
        raise RuntimeError(f"binary hash does not match build stamp at {path}")
    if stamp.get("revision") != revision:
        raise RuntimeError(f"build stamp revision is stale at {path}")
    if stamp.get("tree_dirty") != tree_dirty or stamp.get("tree_fingerprint") != fingerprint:
        raise RuntimeError(f"build stamp tree state is stale at {path}")
    recipe = stamp.get("recipe")
    if recipe not in ("build-cpu", "build-cuda", "build-cuda-fault"):
        raise RuntimeError(f"build stamp has an unsupported recipe at {path}")
    commands = stamp.get("commands")
    if not isinstance(commands, list) or not all(isinstance(command, str) for command in commands):
        raise RuntimeError(f"build stamp commands are invalid at {path}")
    if stamp.get("compiler_flags") != compiler_commands(commands):
        raise RuntimeError(f"build stamp compiler flags are invalid at {path}")
    if any(define in command for command in commands for define in TEST_ONLY_DEFINES):
        raise RuntimeError(f"test-only build define in stamp at {path}")
    if require_cuda and recipe != "build-cuda":
        raise RuntimeError(f"{recipe} binary cannot serve CUDA paths")
    if commands != recipe_commands(root, recipe):
        raise RuntimeError(f"build stamp recipe flags are stale at {path}")
    return stamp


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recipe")
    parser.add_argument("binary", type=Path)
    args = parser.parse_args()
    write_stamp(Path.cwd(), args.binary, args.recipe)


if __name__ == "__main__":
    main()
