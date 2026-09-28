"""Check that MSVC vectorizes every AVX2 hot loop (issue #22).

Recompiles `native/quantizer_avx2.c` with the exact flags of the `build-cuda`
recipe plus /Qvec-report:2, writes the report, and fails unless each loop
tagged `/* avx2-hot */` is reported as "loop vectorized" (info C5001) and no
inlined copy of it as "loop not vectorized" (info C5002).

    uv run python -m stoquant vec-report [--output build/vec_report_avx2.txt]
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

from stoquant import layout
from stoquant.provenance import collect_build_commands

COMPARATOR_SOURCES = (
    "native\\main.c",
    "native\\codec.c",
    "native\\quantizer.c",
    "native\\rng_cpu.c",
)
AVX2_SOURCES = ("native\\quantizer_avx2.c",)


def generate_msvc_vectorization_report(
    root: Path,
    output_file: Path,
    host_flags: Sequence[str] | None,
    object_dir: Path,
    sources: Sequence[str] = COMPARATOR_SOURCES,
) -> str:
    """Recompile C sources with their exact build flags plus /Qvec-report:2."""
    output_text = ""
    script = root / "tools" / "with-msvc.ps1"
    if os.name == "nt" and script.is_file() and host_flags is not None:
        object_dir.mkdir(parents=True, exist_ok=True)
        # A quoted path ending in "\" escapes its closing quote through the
        # PowerShell wrapper, so the directory ends in "/" instead.
        try:
            object_arg = str(object_dir.resolve().relative_to(root.resolve()))
        except ValueError:
            object_arg = str(object_dir.resolve())
        object_arg = object_arg.replace("\\", "/")
        compile_args = [
            "cl.exe",
            *host_flags,
            "/Qvec-report:2",
            "/c",
            *sources,
            f"/Fo:{object_arg}/",
        ]
        cmd = [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(script),
            *compile_args,
        ]
        res = subprocess.run(cmd, cwd=root, capture_output=True, text=True, check=False)
        diagnostics = (res.stdout + "\n" + res.stderr).strip()
        # cl reports absolute source paths; keep the published report root-relative.
        diagnostics = diagnostics.replace(str(root.resolve()) + "\\", "")
        output_text = (
            f"Command: {' '.join(compile_args)}\nExit code: {res.returncode}\n\n" + diagnostics
        )

    if not output_text:
        output_text = "MSVC vectorization report not available on this platform/configuration."

    output_file.write_text(output_text, encoding="utf-8")
    return output_text


HOT_TAG = "/* avx2-hot */"
SOURCE_NAME = "quantizer_avx2.c"


def hot_lines(source_text: str) -> list[int]:
    return [n for n, line in enumerate(source_text.splitlines(), start=1) if HOT_TAG in line]


def unvectorized_hot_loops(source_text: str, report_text: str) -> list[int]:
    """Tagged lines not reported as "loop vectorized" in every inlined copy."""
    pattern = re.compile(
        rf"(?:^|[\\/\s]){re.escape(SOURCE_NAME)}\((\d+)\)\s*:\s*info (C500[12]):",
        re.IGNORECASE | re.MULTILINE,
    )
    vectorized, scalar = set(), set()
    for m in pattern.finditer(report_text):
        (vectorized if m.group(2).upper() == "C5001" else scalar).add(int(m.group(1)))
    return [n for n in hot_lines(source_text) if n not in vectorized or n in scalar]


def main(argv: list[str] | None = None, prog: str | None = None) -> int:
    parser = argparse.ArgumentParser(prog=prog, description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=Path("build/vec_report_avx2.txt"))
    args = parser.parse_args(argv)
    root = layout.ROOT
    flags = collect_build_commands(root)["_avx2_tokens"]
    if flags is None:
        print("no quantizer_avx2.c compile found in the build recipe", file=sys.stderr)
        return 1
    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root / "build") as objects:
        report = generate_msvc_vectorization_report(
            root, output, flags, Path(objects), AVX2_SOURCES
        )
    source = (layout.NATIVE_DIR / SOURCE_NAME).read_text(encoding="utf-8")
    missing = unvectorized_hot_loops(source, report)
    total = len(hot_lines(source))
    if total == 0:
        print(f"no {HOT_TAG} loops found", file=sys.stderr)
        return 1
    if missing:
        print(f"not vectorized: {SOURCE_NAME} lines {missing}; see {args.output}", file=sys.stderr)
        return 1
    print(f"{total} of {total} avx2-hot loops vectorized; report at {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
