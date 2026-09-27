"""Check that MSVC vectorizes every AVX2 hot loop (issue #22).

Recompiles `src/quantizer_avx2.c` with the exact flags of the `build-cuda`
recipe plus /Qvec-report:2, writes the report, and fails unless each loop
tagged `/* avx2-hot */` is reported as "loop vectorized" (info C5001).

    uv run python scripts/vec_report_check.py [--output build/vec_report_avx2.txt]
"""

from __future__ import annotations

import argparse
import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark_driver import (
    AVX2_SOURCES,
    collect_build_commands,
    generate_msvc_vectorization_report,
)

HOT_TAG = "/* avx2-hot */"
SOURCE_NAME = "quantizer_avx2.c"


def hot_lines(source_text: str) -> list[int]:
    return [n for n, line in enumerate(source_text.splitlines(), start=1) if HOT_TAG in line]


def unvectorized_hot_loops(source_text: str, report_text: str) -> list[int]:
    """Tagged lines the report does not mark as "loop vectorized"."""
    pattern = re.compile(
        rf"(?:^|[\\/\s]){re.escape(SOURCE_NAME)}\((\d+)\)\s*:\s*info C5001: loop vectorized",
        re.IGNORECASE | re.MULTILINE,
    )
    vectorized = {int(m.group(1)) for m in pattern.finditer(report_text)}
    return [n for n in hot_lines(source_text) if n not in vectorized]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=Path("build/vec_report_avx2.txt"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
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
    source = (root / "src" / SOURCE_NAME).read_text(encoding="utf-8")
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
