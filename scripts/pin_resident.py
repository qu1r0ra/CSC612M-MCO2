"""Run untraced resident-path processes, optionally pinned to one physical core (issue #30).

The Nsight trace showed that slow resident processes pay a fixed, per-process CPU-side
launch cost. This runner checks whether that level tracks the core a process lands on,
without the profiler: under Nsight, pinned processes hung inside the driver. Each
process gets a hard timeout so a hang stops the run instead of stalling it.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark_driver import (  # noqa: E402
    DEFAULT_COMPRESSION_SEED,
    collect_git_provenance,
    find_binary,
    generate_inputs,
    query_gpu_state,
    warm_up_gpu,
)
from scripts.trace_resident import set_affinity  # noqa: E402

NOT_CORE0 = -1


def apply_affinity(core: int | None) -> None:
    """Pin to one physical core, to every core except core 0 (NOT_CORE0), or unpin."""
    if core != NOT_CORE0:
        set_affinity(core)
        return
    import ctypes
    import os

    mask = ((1 << (os.cpu_count() or 1)) - 1) & ~0b11
    kernel32 = ctypes.windll.kernel32
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    if not kernel32.SetProcessAffinityMask(ctypes.c_void_p(kernel32.GetCurrentProcess()), mask):
        raise OSError(ctypes.get_last_error())


def run_one(
    binary: Path, input_path: Path, bits: int, warmups: int, reps: int, timeout: float
) -> dict[str, Any]:
    cmd = [
        str(binary), "bench", "--input", str(input_path),
        "--seed", str(DEFAULT_COMPRESSION_SEED), "--bits", str(bits),
        "--tensor-id", "0", "--invocation-id", "0", "--backend", "cuda",
        "--warmup", str(warmups), "--reps", str(reps), "--boundary", "resident", "--timings",
    ]  # fmt: skip
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip())
    payload = json.loads(proc.stdout)
    kern = [a + b + c for a, b, c in zip(payload["k1_ms"], payload["k2_ms"], payload["k3_ms"])]
    return {
        "wall_median_ms": statistics.median(payload["samples_ms"]),
        "kernel_median_ms": statistics.median(kern),
        "k1_median_ms": statistics.median(payload["k1_ms"]),
        "k2_median_ms": statistics.median(payload["k2_ms"]),
        "k3_median_ms": statistics.median(payload["k3_ms"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=1 << 14)
    parser.add_argument("--bits", type=int, default=8)
    parser.add_argument("--warmups", type=int, default=6654)
    parser.add_argument("--reps", type=int, default=30)
    parser.add_argument("--unpinned", type=int, default=12, help="Unpinned processes")
    parser.add_argument("--per-core", type=int, default=6, help="Processes per pinned core")
    parser.add_argument(
        "--cores", type=int, nargs="*", default=[0, 1, 2, 3, 4, 5],
        help=f"Physical cores to pin to; {NOT_CORE0} means every core except core 0",
    )  # fmt: skip
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--label", default="pin")
    parser.add_argument("--seed", type=int, default=30)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    binary = find_binary(root)
    input_path = Path(generate_inputs([args.count], args.input_dir)[args.count]["_path"])
    plan: list[int | None] = [None] * args.unpinned
    plan += [c for c in args.cores for _ in range(args.per_core)]
    random.Random(args.seed).shuffle(plan)

    gpu_start = query_gpu_state()
    set_affinity(None)
    warm_up_gpu(binary, input_path, 3.0)
    runs = []
    try:
        for i, core in enumerate(plan):
            apply_affinity(core)
            start = time.time()
            try:
                result = run_one(
                    binary, input_path, args.bits, args.warmups, args.reps, args.timeout
                )
            except subprocess.TimeoutExpired:
                print(f"{i:3d} core={core} TIMEOUT; stopping", flush=True)
                runs.append({"index": i, "core": core, "start_unix": start, "timeout": True})
                break
            finally:
                set_affinity(None)
            runs.append({"index": i, "core": core, "start_unix": start, **result})
            print(
                f"{i:3d} core={core} wall={result['wall_median_ms']:.3f} "
                f"kern={result['kernel_median_ms']:.3f}",
                flush=True,
            )
    finally:
        set_affinity(None)
    out = {
        "label": args.label,
        "count": args.count,
        "bits": args.bits,
        "warmups": args.warmups,
        "reps": args.reps,
        "git": collect_git_provenance(root),
        "gpu_state_start": gpu_start,
        "gpu_state_end": query_gpu_state(),
        "runs": runs,
    }
    path = args.output_dir / f"pin-{args.label}.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
