"""Diagnose between-process instability of the CUDA resident path (issue #26).

Runs one fixed resident case in many separate processes under several
conditions, interleaved round-robin so slow drift affects every condition
equally, while `nvidia-smi` logs GPU state in the background. Writes raw
per-process records, the GPU log, and a summary into the output directory.

    uv run python scripts/diag_resident.py --output-dir results/diag-...
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import statistics
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark_driver import (
    DEFAULT_COMPRESSION_SEED,
    collect_git_provenance,
    find_binary,
    generate_inputs,
    query_gpu_state,
    run_bench_process,
    warm_up_gpu,
)

SLOW_RATIO = 1.5
GROUP_SIZE = 6
GPU_LOG_FIELDS = (
    "timestamp",
    "pstate",
    "clocks.sm",
    "clocks.mem",
    "power.draw",
    "utilization.gpu",
    "clocks_event_reasons.active",
)
HIGH_PRIORITY_CLASS = 0x00000080


def conditions_for(rep_ms: float, long_warmup_seconds: float) -> dict[str, dict[str, Any]]:
    long_warmups = max(10, round(long_warmup_seconds * 1000.0 / rep_ms))
    return {
        "baseline": {"warmups": 10, "reps": 30, "gap_s": 0.0, "high_priority": False},
        "long_warmup": {"warmups": long_warmups, "reps": 30, "gap_s": 0.0, "high_priority": False},
        "many_reps": {"warmups": 10, "reps": 300, "gap_s": 0.0, "high_priority": False},
        "idle_gap": {"warmups": 10, "reps": 30, "gap_s": 3.0, "high_priority": False},
        "high_priority": {"warmups": 10, "reps": 30, "gap_s": 0.0, "high_priority": True},
    }


def kernel_sums(payload: dict[str, Any]) -> list[float]:
    return [a + b + c for a, b, c in zip(payload["k1_ms"], payload["k2_ms"], payload["k3_ms"])]


def run_one(binary: Path, input_path: Path, bits: int, spec: dict[str, Any]) -> dict[str, Any]:
    if spec["gap_s"]:
        time.sleep(spec["gap_s"])
    extra = ["--boundary", "resident", "--timings"]
    if spec["high_priority"]:
        # run_bench_process has no creationflags hook; start the same command directly.
        cmd = [
            str(binary), "bench", "--input", str(input_path),
            "--seed", str(DEFAULT_COMPRESSION_SEED), "--bits", str(bits),
            "--tensor-id", "0", "--invocation-id", "0", "--backend", "cuda",
            "--warmup", str(spec["warmups"]), "--reps", str(spec["reps"]), *extra,
        ]  # fmt: skip
        start = time.time()
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            creationflags=HIGH_PRIORITY_CLASS if os.name == "nt" else 0,
        )
        end = time.time()
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.strip())
        payload = json.loads(proc.stdout)
    else:
        start = time.time()
        payload, error = run_bench_process(
            binary,
            input_path,
            bits=bits,
            backend="cuda",
            extra_args=extra,
            seed=DEFAULT_COMPRESSION_SEED,
            warmups=spec["warmups"],
            reps=spec["reps"],
        )
        end = time.time()
        if payload is None:
            raise RuntimeError(error)
    kern = kernel_sums(payload)
    return {
        "start_unix": start,
        "end_unix": end,
        "wall_median_ms": statistics.median(payload["samples_ms"]),
        "kernel_median_ms": statistics.median(kern),
        "kernel_first10_ms": statistics.median(kern[:10]),
        "kernel_last10_ms": statistics.median(kern[-10:]),
        "k1_median_ms": statistics.median(payload["k1_ms"]),
        "k2_median_ms": statistics.median(payload["k2_ms"]),
        "k3_median_ms": statistics.median(payload["k3_ms"]),
    }


def start_gpu_log(path: Path, interval_ms: int) -> subprocess.Popen[bytes]:
    return subprocess.Popen(
        [
            "nvidia-smi",
            f"--query-gpu={','.join(GPU_LOG_FIELDS)}",
            "--format=csv,noheader,nounits",
            f"-lms={interval_ms}",
            f"--filename={path}",
        ]
    )


def read_gpu_log(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != len(GPU_LOG_FIELDS):
            continue
        try:
            stamp = datetime.strptime(parts[0], "%Y/%m/%d %H:%M:%S.%f").astimezone().timestamp()
            rows.append(
                {
                    "t": stamp,
                    "pstate": parts[1],
                    "sm": float(parts[2]),
                    "mem": float(parts[3]),
                    "power": float(parts[4]),
                    "util": float(parts[5]),
                    "reasons": parts[6],
                }
            )
        except ValueError:
            continue
    return rows


def gpu_during(rows: list[dict[str, Any]], start: float, end: float) -> dict[str, Any]:
    inside = [r for r in rows if start <= r["t"] <= end]
    if not inside:
        return {"samples": 0}
    return {
        "samples": len(inside),
        "sm_min": min(r["sm"] for r in inside),
        "mem_min": min(r["mem"] for r in inside),
        "pstates": sorted({r["pstate"] for r in inside}),
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in sorted({(r["count"], r["condition"]) for r in records}):
        count, condition = key
        case = [r for r in records if r["count"] == count]
        floor = min(r["kernel_median_ms"] for r in case)
        rows = [r for r in case if r["condition"] == condition]
        ratios = [r["kernel_median_ms"] / floor for r in rows]
        walls = [r["wall_median_ms"] for r in rows]
        groups = [
            walls[i : i + GROUP_SIZE] for i in range(0, len(walls) - GROUP_SIZE + 1, GROUP_SIZE)
        ]
        spreads = [max(g) / min(g) for g in groups]
        slow = [r for r, q in zip(rows, ratios, strict=True) if q > SLOW_RATIO]
        fast = [r for r, q in zip(rows, ratios, strict=True) if q <= SLOW_RATIO]

        def stage_inflation(
            stage: str,
            slow: list[dict[str, Any]] = slow,
            fast: list[dict[str, Any]] = fast,
        ) -> float | None:
            if not slow or not fast:
                return None
            return round(
                statistics.median(r[stage] for r in slow)
                / statistics.median(r[stage] for r in fast),
                2,
            )

        def slow_gpu(group: list[dict[str, Any]], field: str) -> float | None:
            vals = [r["gpu"][field] for r in group if r["gpu"].get("samples")]
            return statistics.median(vals) if vals else None

        out[f"n{count}/{condition}"] = {
            "processes": len(rows),
            "wall_median_ms": round(statistics.median(walls), 4),
            "slow_fraction": round(len(slow) / len(rows), 2),
            "kernel_ratio_median": round(statistics.median(ratios), 2),
            "kernel_ratio_max": round(max(ratios), 2),
            "spread_ratio_groups_of_6": [round(s, 2) for s in spreads],
            "groups_within_1_25": sum(s <= 1.25 for s in spreads),
            "slow_over_fast_k1": stage_inflation("k1_median_ms"),
            "slow_over_fast_k2": stage_inflation("k2_median_ms"),
            "slow_over_fast_k3": stage_inflation("k3_median_ms"),
            "slow_first10_over_last10": (
                round(
                    statistics.median(r["kernel_first10_ms"] / r["kernel_last10_ms"] for r in slow),
                    2,
                )
                if slow
                else None
            ),
            "sm_min_mhz_slow_vs_fast": [slow_gpu(slow, "sm_min"), slow_gpu(fast, "sm_min")],
            "mem_min_mhz_slow_vs_fast": [slow_gpu(slow, "mem_min"), slow_gpu(fast, "mem_min")],
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--counts", type=int, nargs="+", default=[1 << 14, 1 << 20])
    parser.add_argument("--bits", type=int, default=8)
    parser.add_argument("--processes", type=int, default=36)
    parser.add_argument("--conditions", nargs="+", default=None)
    parser.add_argument("--long-warmup-seconds", type=float, default=1.0)
    parser.add_argument("--gpu-log-ms", type=int, default=50)
    parser.add_argument("--label", default="as-is", help="Desktop state, e.g. as-is or quiet")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    binary = find_binary(root)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    inputs = generate_inputs(args.counts, out / "inputs")
    gpu_log = out / f"gpu_log_{args.label}.csv"
    logger = start_gpu_log(gpu_log, args.gpu_log_ms)
    records: list[dict[str, Any]] = []
    rng = random.Random(26)
    try:
        largest = Path(inputs[max(args.counts)]["_path"])
        warm = warm_up_gpu(binary, largest, 20.0)
        for count in args.counts:
            input_path = Path(inputs[count]["_path"])
            probe = run_one(
                binary,
                input_path,
                args.bits,
                {"warmups": 10, "reps": 30, "gap_s": 0.0, "high_priority": False},
            )
            conds = conditions_for(probe["wall_median_ms"], args.long_warmup_seconds)
            if args.conditions:
                conds = {k: v for k, v in conds.items() if k in args.conditions}
            for index in range(args.processes):
                names = list(conds)
                rng.shuffle(names)
                for name in names:
                    record = run_one(binary, input_path, args.bits, conds[name])
                    record.update(
                        {"count": count, "condition": name, "index": index, **conds[name]}
                    )
                    records.append(record)
                print(f"n={count} process round {index + 1}/{args.processes}", flush=True)
    finally:
        time.sleep(1.0)
        logger.terminate()
        logger.wait()

    rows = read_gpu_log(gpu_log)
    for record in records:
        record["gpu"] = gpu_during(rows, record["start_unix"], record["end_unix"])
    with (out / f"processes_{args.label}.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = [k for k in records[0] if k != "gpu"] + ["gpu"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({**record, "gpu": json.dumps(record["gpu"])})
    summary = {
        "label": args.label,
        "created_utc": datetime.now(UTC).isoformat(),
        "git": collect_git_provenance(root),
        "bits": args.bits,
        "counts": args.counts,
        "processes_per_condition": args.processes,
        "initial_warm_up": warm,
        "gpu_state_end": query_gpu_state(),
        "gpu_log_samples": len(rows),
        "slow_ratio": SLOW_RATIO,
        "results": summarize(records),
    }
    (out / f"summary_{args.label}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["results"], indent=2))


if __name__ == "__main__":
    main()
