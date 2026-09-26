"""Trace separate resident-path processes under Nsight Systems (issue #30).

Each process runs `mco2 bench --boundary resident` under `nsys profile --trace=cuda`.
The report is exported to SQLite and reduced to per-repetition timings for the
timed repetitions: GPU span, kernel busy time, gaps between GPU operations, and the
CPU-side launch call duration and launch-to-start latency. Slow and fast processes
can then be compared on where the extra time goes.

Nsight Systems on Windows does not forward the application's stdout, so every
timing comes from the trace itself.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import random
import sqlite3
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

DEFAULT_NSYS = Path(
    r"C:\Program Files\NVIDIA Corporation\Nsight Systems 2026.3.2\target-windows-x64\nsys.exe"
)


def set_affinity(core: int | None) -> None:
    """Pin this process, and every child it starts, to one physical core.

    Both SMT siblings (logical CPUs 2*core and 2*core+1) stay available: with a
    single logical CPU the CUDA synchronize spin starved the driver's helper
    thread and the process hung.
    """
    if core is None:
        mask = (1 << (os.cpu_count() or 1)) - 1
    else:
        mask = 0b11 << (2 * core)
    kernel32 = ctypes.windll.kernel32
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    if not kernel32.SetProcessAffinityMask(ctypes.c_void_p(kernel32.GetCurrentProcess()), mask):
        raise OSError(ctypes.get_last_error())


def profile_one(
    nsys: Path, binary: Path, input_path: Path, bits: int, warmups: int, reps: int, report: Path
) -> None:
    cmd = [
        str(nsys), "profile", "--trace=cuda", "--sample=none", "--cpuctxsw=none",
        "--force-overwrite=true", "-o", str(report),
        str(binary), "bench", "--input", str(input_path),
        "--seed", str(DEFAULT_COMPRESSION_SEED), "--bits", str(bits),
        "--tensor-id", "0", "--invocation-id", "0", "--backend", "cuda",
        "--warmup", str(warmups), "--reps", str(reps), "--boundary", "resident", "--timings",
    ]  # fmt: skip
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip())
    export = [
        str(nsys), "export", "--type", "sqlite", "--force-overwrite=true",
        "-o", str(report.with_suffix(".sqlite")), str(report.with_suffix(".nsys-rep")),
    ]  # fmt: skip
    subprocess.run(export, capture_output=True, text=True, check=True)


def reduce_report(sqlite_path: Path, reps: int) -> dict[str, Any]:
    """Split the GPU timeline into repetitions at each memset; keep the last `reps`."""
    con = sqlite3.connect(sqlite_path)
    ops: list[tuple[int, int, str, int]] = []
    for start, end, corr in con.execute(
        "SELECT start, end, correlationId FROM CUPTI_ACTIVITY_KIND_MEMSET"
    ):
        ops.append((start, end, "memset", corr))
    for start, end, corr in con.execute(
        "SELECT start, end, correlationId FROM CUPTI_ACTIVITY_KIND_KERNEL"
    ):
        ops.append((start, end, "kernel", corr))
    ops.sort()
    api = {
        corr: (start, end)
        for start, end, corr in con.execute(
            "SELECT r.start, r.end, r.correlationId FROM CUPTI_ACTIVITY_KIND_RUNTIME r "
            "JOIN StringIds s ON s.id = r.nameId "
            "WHERE s.value LIKE 'cudaLaunchKernel%' OR s.value LIKE 'cudaMemsetAsync%'"
        )
    }
    con.close()

    groups: list[list[tuple[int, int, str, int]]] = []
    for op in ops:
        if op[2] == "memset" or not groups:
            groups.append([])
        groups[-1].append(op)
    timed = groups[-reps:]

    per_rep = []
    for group in timed:
        gaps = [b[0] - a[1] for a, b in zip(group, group[1:])]
        kernels = [op for op in group if op[2] == "kernel"]
        launch_api = [api[op[3]][1] - api[op[3]][0] for op in kernels if op[3] in api]
        queue = [op[0] - api[op[3]][0] for op in kernels if op[3] in api]
        api_starts = sorted(api[op[3]][0] for op in group if op[3] in api)
        per_rep.append(
            {
                "span_us": (group[-1][1] - group[0][0]) / 1000.0,
                "busy_us": sum(op[1] - op[0] for op in kernels) / 1000.0,
                "gap_median_us": statistics.median(gaps) / 1000.0,
                "launch_api_median_us": statistics.median(launch_api) / 1000.0,
                "launch_to_start_median_us": statistics.median(queue) / 1000.0,
                "api_interval_median_us": statistics.median(
                    b - a for a, b in zip(api_starts, api_starts[1:])
                )
                / 1000.0,
                "ops": len(group),
            }
        )
    summary = {
        key: statistics.median(r[key] for r in per_rep)
        for key in per_rep[0]
        if key != "ops"
    }
    summary["ops_per_rep"] = statistics.median(r["ops"] for r in per_rep)
    summary["reps_found"] = len(groups)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True, help="Raw .nsys-rep/.sqlite")
    parser.add_argument("--counts", type=int, nargs="+", default=[1 << 14])
    parser.add_argument("--bits", type=int, default=8)
    parser.add_argument("--processes", type=int, default=36)
    parser.add_argument("--warmups", type=int, nargs="+", default=[6654])
    parser.add_argument("--reps", type=int, default=30)
    parser.add_argument("--nsys", type=Path, default=DEFAULT_NSYS)
    parser.add_argument("--label", default="as-is")
    parser.add_argument("--seed", type=int, default=30)
    parser.add_argument(
        "--cores", type=int, nargs="+", default=None,
        help="Pin each process to one of these physical cores (a plan dimension)",
    )  # fmt: skip
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    binary = find_binary(root)
    inputs = generate_inputs(args.counts, args.report_dir / "inputs")
    cores: list[int | None] = list(args.cores) if args.cores else [None]
    plan = [(c, w, u) for c in args.counts for w in args.warmups for u in cores] * args.processes
    random.Random(args.seed).shuffle(plan)

    gpu_start = query_gpu_state()
    warm_up_gpu(binary, Path(inputs[args.counts[0]]["_path"]), 3.0)
    set_affinity(None)
    runs = []
    for i, (count, warmups, core) in enumerate(plan):
        report = args.report_dir / f"p{i:03d}_n{count}_w{warmups}_c{core}"
        set_affinity(core)
        start = time.time()
        profile_one(
            args.nsys, binary, Path(inputs[count]["_path"]), args.bits, warmups, args.reps, report
        )
        summary = reduce_report(report.with_suffix(".sqlite"), args.reps)
        runs.append(
            {
                "index": i,
                "count": count,
                "warmups": warmups,
                "core": core,
                "start_unix": start,
                **summary,
            }
        )
        print(
            f"{i:3d} n={count} w={warmups} core={core} span={summary['span_us']:.1f}us "
            f"gap={summary['gap_median_us']:.1f} api={summary['launch_api_median_us']:.1f} "
            f"q={summary['launch_to_start_median_us']:.1f}",
            flush=True,
        )
    set_affinity(None)
    out = {
        "label": args.label,
        "bits": args.bits,
        "reps": args.reps,
        "nsys": str(args.nsys),
        "trace": "cuda",
        "git": collect_git_provenance(root),
        "gpu_state_start": gpu_start,
        "gpu_state_end": query_gpu_state(),
        "runs": runs,
    }
    path = args.output_dir / f"trace-{args.label}.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
