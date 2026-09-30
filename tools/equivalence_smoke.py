"""Run a bounded, fixed-input byte-parity smoke across the native backends."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from stoquant.provenance import find_binary

ROOT = Path(__file__).resolve().parents[1]
SMOKE_COMMAND_BUDGET_SECONDS = 285.0
INPUT_SEED = 2026
COMPRESSION_SEED = 612
TENSOR_ID = 3
INVOCATION_ID = 5

VARIANTS = (
    ("cpu", "cpu", None),
    ("cpu-avx2", "cpu-avx2", None),
    ("cuda-reference", "cuda", "reference"),
    ("cuda-optimized", "cuda", "optimized"),
)


class SmokeError(RuntimeError):
    """A smoke command failed, timed out, or produced different bytes."""


def fixed_inputs() -> dict[str, np.ndarray]:
    """Small cases for ordinary values, odd packing, zeros, and mixed magnitudes."""
    normal = np.random.default_rng(INPUT_SEED).standard_normal(1024, dtype=np.float32)
    mixed = np.random.default_rng(INPUT_SEED + 1).standard_normal(4099, dtype=np.float32)
    mixed[::5] = np.float32(0.0)
    mixed[1::17] = np.float32(1.0e-5)
    mixed[2::29] = np.float32(-1.0e4)
    mixed[3::37] = np.float32(0.125)
    return {
        "normal_1024": normal,
        "mixed_odd_4099": mixed,
        "zero_513": np.zeros(513, dtype=np.float32),
    }


def run(argv: Sequence[str], deadline: float) -> None:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise SmokeError("the 5-minute behavior-smoke budget expired")
    try:
        result = subprocess.run(
            list(argv),
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=remaining,
        )
    except subprocess.TimeoutExpired as exc:
        raise SmokeError("a native command reached the 5-minute behavior-smoke cap") from exc
    except OSError as exc:
        raise SmokeError(f"could not start {argv[0]}: {exc}") from exc
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise SmokeError(f"command failed ({result.returncode}): {argv[1]}\n{detail}")


def compress_args(
    binary: Path,
    input_path: Path,
    output_path: Path,
    backend: str,
    bits: int,
    k1: str | None,
) -> list[str]:
    args = [
        str(binary),
        "compress",
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        "--seed",
        str(COMPRESSION_SEED),
        "--bits",
        str(bits),
        "--tensor-id",
        str(TENSOR_ID),
        "--invocation-id",
        str(INVOCATION_ID),
        "--backend",
        backend,
    ]
    if k1 is not None:
        args.extend(("--k1", k1))
    return args


def first_difference(left: bytes, right: bytes) -> int | None:
    for index, (a, b) in enumerate(zip(left, right, strict=False)):
        if a != b:
            return index
    return min(len(left), len(right)) if len(left) != len(right) else None


def check_case(
    binary: Path, work: Path, name: str, values: np.ndarray, bits: int, deadline: float
) -> None:
    input_path = work / f"{name}.f32"
    values.astype("<f4", copy=False).tofile(input_path)
    records: dict[str, bytes] = {}
    decoded: dict[str, bytes] = {}

    for label, backend, k1 in VARIANTS:
        record_path = work / f"{name}_b{bits}_{label}.msq"
        run(compress_args(binary, input_path, record_path, backend, bits, k1), deadline)
        records[label] = record_path.read_bytes()

    reference = records["cpu"]
    for label, record in records.items():
        mismatch = first_difference(reference, record)
        if mismatch is not None:
            raise SmokeError(
                f"{name} {bits}-bit {label} record differs from scalar CPU at byte {mismatch}"
            )

    for label, _, _ in VARIANTS:
        record_path = work / f"{name}_b{bits}_{label}.msq"
        decoded_path = work / f"{name}_b{bits}_{label}.decoded.f32"
        run(
            [str(binary), "decompress", "--input", str(record_path), "--output", str(decoded_path)],
            deadline,
        )
        decoded[label] = decoded_path.read_bytes()

    reference_output = decoded["cpu"]
    for label, output in decoded.items():
        mismatch = first_difference(reference_output, output)
        if mismatch is not None:
            raise SmokeError(
                f"{name} {bits}-bit {label} decode differs from scalar CPU at byte {mismatch}"
            )
    expected_bytes = values.size * 4
    if len(reference_output) != expected_bytes:
        raise SmokeError(
            f"{name} {bits}-bit decoded {len(reference_output)} bytes; expected {expected_bytes}"
        )
    print(f"pass: {name}, {bits}-bit, {len(VARIANTS)} encoders and decoders")


def main() -> int:
    started = time.monotonic()
    deadline = started + SMOKE_COMMAND_BUDGET_SECONDS
    try:
        verified = find_binary(ROOT, require_cuda=True)
        with tempfile.TemporaryDirectory(prefix="stoquant-equivalence-smoke-") as temporary:
            work = Path(temporary)
            for name, values in fixed_inputs().items():
                for bits in (4, 8):
                    check_case(verified.path, work, name, values, bits, deadline)
        elapsed = time.monotonic() - started
        if elapsed >= SMOKE_COMMAND_BUDGET_SECONDS:
            raise SmokeError("the 5-minute behavior-smoke budget expired during cleanup")
    except (SmokeError, FileNotFoundError, RuntimeError) as exc:
        print(f"equivalence smoke failed: {exc}", file=sys.stderr)
        return 1

    print(f"behavior smoke passed in {elapsed:.1f}s (cap: 5 minutes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
