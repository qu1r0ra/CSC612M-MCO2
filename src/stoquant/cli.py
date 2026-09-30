"""Command-line interface for the benchmark matrix."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from stoquant import layout
from stoquant.design import (
    DEFAULT_BITS,
    DEFAULT_CASE_ORDER_SEED,
    DEFAULT_COUNTS,
    DEFAULT_TRIALS,
    EXTENSION_BOUNDARIES,
    K1_VARIANTS,
    PILOT_PROFILE,
    TRANSFER_POLICIES,
)
from stoquant.inputs import DEFAULT_INPUT_SEED, INPUT_FAMILIES, MODEL_TENSOR_SETS
from stoquant.matrix import run_benchmark_matrix
from stoquant.runner import (
    DEFAULT_CASE_WARMUP_SECONDS,
    DEFAULT_COMPRESSION_SEED,
    DEFAULT_GPU_WARMUP_SECONDS,
    DEFAULT_IN_PROCESS_WARMUP_SECONDS,
    DEFAULT_REPS,
    DEFAULT_WARMUPS,
)
from stoquant.schema import read_manifest


def main(argv: list[str] | None = None, prog: str | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog=prog, description="stoquant benchmark driver and snapshot creator"
    )
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory to save snapshot")
    parser.add_argument(
        "--counts", type=int, nargs="+", default=DEFAULT_COUNTS, help="Element counts"
    )
    parser.add_argument(
        "--bits", type=int, nargs="+", default=DEFAULT_BITS, help="Bit widths (4 or 8)"
    )
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUPS, help="Warmup runs")
    parser.add_argument("--reps", type=int, default=DEFAULT_REPS, help="Measured repetitions")
    parser.add_argument(
        "--trials",
        type=int,
        default=DEFAULT_TRIALS,
        help="Independent processes per path, each trial in a balanced path order (all permutations, or Williams rows)",
    )
    parser.add_argument(
        "--input-seed", type=int, default=DEFAULT_INPUT_SEED, help="Seed for input generation"
    )
    parser.add_argument(
        "--compression-seed",
        type=int,
        default=DEFAULT_COMPRESSION_SEED,
        help="Seed for compression",
    )
    parser.add_argument(
        "--case-order-seed",
        type=int,
        default=DEFAULT_CASE_ORDER_SEED,
        help="Seed for the random order of (count, bits) cases",
    )
    parser.add_argument(
        "--gpu-warmup-seconds",
        type=float,
        default=DEFAULT_GPU_WARMUP_SECONDS,
        help="Untimed CUDA work before the first timed process (0 disables)",
    )
    parser.add_argument(
        "--case-warmup-seconds",
        type=float,
        default=DEFAULT_CASE_WARMUP_SECONDS,
        help="Untimed CUDA work before each case's timed processes (0 disables)",
    )
    parser.add_argument(
        "--warmup-seconds",
        type=float,
        default=DEFAULT_IN_PROCESS_WARMUP_SECONDS,
        help=(
            "Minimum untimed in-process warm-up per timed process, for every path; "
            "--warmup is the floor (0 disables)"
        ),
    )
    parser.add_argument(
        "--backends",
        type=str,
        nargs="+",
        default=["cpu", "cuda"],
        help=(
            "Backends to benchmark (cpu, cuda, cpu-avx2); cpu-avx2 adds the "
            "descriptive AVX2 comparator path"
        ),
    )
    parser.add_argument("--k1", choices=K1_VARIANTS, default="reference", help="CUDA K1 variant")
    parser.add_argument(
        "--boundaries",
        type=str,
        nargs="*",
        choices=EXTENSION_BOUNDARIES,
        default=[],
        help="Publication extension boundaries to add to the revision 3 paths",
    )
    parser.add_argument(
        "--transfer-policies",
        type=str,
        nargs="+",
        choices=TRANSFER_POLICIES,
        default=["pageable"],
        help="Host transfer policies for transfer boundaries; pinned needs pageable",
    )
    parser.add_argument(
        "--input-family",
        choices=INPUT_FAMILIES,
        default="dense",
        help="Input family: dense normal, sparse (90%% zeros), or model-shaped tensors",
    )
    parser.add_argument(
        "--model-tensors",
        choices=MODEL_TENSOR_SETS,
        default=None,
        help="Model family: one tensor per distinct shape and kind (default), or all tensors",
    )
    parser.add_argument(
        "--model-limit",
        type=int,
        default=None,
        help="Model family: keep only the first N selected tensors",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Run from an uncommitted tree (records the dirty files; not for evidence)",
    )
    parser.add_argument(
        "--pilot",
        action="store_true",
        help=(
            "Non-evidence pilot at four sizes and both bit widths, written under "
            "results/pilots/; records the readiness check without enforcing it"
        ),
    )
    parser.add_argument(
        "--ignore-readiness",
        action="store_true",
        help="Run despite a failed readiness check (marks the snapshot non-evidence)",
    )

    args = parser.parse_args(argv)
    if len(set(args.counts)) != len(args.counts):
        parser.error("--counts values must be unique")
    root = layout.ROOT

    try:
        snapshot_dir = run_benchmark_matrix(
            root=root,
            output_dir=args.output_dir,
            counts=PILOT_PROFILE.counts if args.pilot else args.counts,
            bit_widths=PILOT_PROFILE.bit_widths if args.pilot else args.bits,
            backends=args.backends,
            warmups=args.warmup,
            reps=args.reps,
            trials=args.trials,
            input_seed=args.input_seed,
            compression_seed=args.compression_seed,
            case_order_seed=args.case_order_seed,
            gpu_warmup_seconds=args.gpu_warmup_seconds,
            case_warmup_seconds=args.case_warmup_seconds,
            in_process_warmup_seconds=args.warmup_seconds,
            allow_dirty=args.allow_dirty,
            pilot=args.pilot,
            ignore_readiness=args.ignore_readiness,
            boundaries=args.boundaries,
            transfer_policies=args.transfer_policies,
            input_family=args.input_family,
            model_tensors=args.model_tensors,
            model_limit=args.model_limit,
            k1=args.k1,
        )
        manifest = read_manifest(snapshot_dir / "manifest.json")
        if not manifest["all_cases_passed"]:
            reasons = manifest["run_conditions"]["non_evidence_reasons"]
            print(
                "Benchmark completed with failed cases; snapshot is non-evidence. "
                f"Reasons: {'; '.join(reasons)}. Snapshot: {snapshot_dir}",
                file=sys.stderr,
            )
            sys.exit(1)
        print(f"Benchmark completed successfully. Snapshot written to: {snapshot_dir}")
    except (RuntimeError, ValueError, OSError, subprocess.SubprocessError) as exc:
        print(f"Benchmark driver failed: {exc}", file=sys.stderr)
        sys.exit(1)
