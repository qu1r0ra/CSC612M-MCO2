"""Run clang-format over the tracked C and CUDA sources outside third_party.

    python tools/clang_format.py [--check]

With --check, files are left untouched and any file that needs formatting
fails the run.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PATTERNS = ("*.c", "*.h", "*.cu", "*.cuh")


def sources() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files", "--", *PATTERNS],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    )
    return [p for p in out.stdout.splitlines() if not p.startswith("third_party/")]


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--check", action="store_true", help="report without rewriting")
    args = parser.parse_args()
    exe = shutil.which("clang-format")
    if exe is None:
        sys.exit("clang-format not found; run through `uv run`")
    mode = ["--dry-run", "--Werror"] if args.check else ["-i"]
    sys.exit(subprocess.run([exe, *mode, *sources()], cwd=REPO, check=False).returncode)


if __name__ == "__main__":
    main()
