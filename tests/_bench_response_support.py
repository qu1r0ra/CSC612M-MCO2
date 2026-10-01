from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any


def option_value(args: Sequence[str], name: str, default: str | None = None) -> str | None:
    value = default
    for index, arg in enumerate(args):
        if arg == name and index + 1 < len(args):
            value = args[index + 1]
        elif arg.startswith(name + "="):
            value = arg[len(name) + 1 :]
    return value


def bench_payload(
    *,
    count: int,
    bits: int,
    backend: str,
    seed: int,
    warmups: int,
    reps: int,
    extra_args: Sequence[str] = (),
    k1: str = "reference",
    sample: float = 2.0,
) -> dict[str, Any]:
    default_boundary = {
        "cpu": "host-host",
        "cpu-avx2": "host-host",
        "cuda": "resident",
    }[backend]
    boundary = option_value(extra_args, "--boundary", default_boundary) or default_boundary
    transfers = boundary in {"host-origin", "gpu-origin"}
    policy = option_value(extra_args, "--transfer-policy", "pageable" if transfers else "none")
    step = 0 if boundary == "resident-graph" else 1
    configuration: dict[str, Any] = {
        "backend": backend,
        "bits": bits,
        "count": count,
        "seed": seed,
        "tensor_id": 0,
        "invocation_id": 0,
        "warmup": warmups,
        "reps": reps,
        "repetition_invocation_ids": [i * step for i in range(reps)],
        "warmup_invocation_ids": [(reps + i) * step for i in range(warmups)],
        "boundary": boundary,
        "transfer_policy": policy,
        "block_size": int(option_value(extra_args, "--block-size", "256") or "256"),
        "grid_size": int(option_value(extra_args, "--grid-size", "0") or "0"),
        "prescribed_scale": None,
    }
    if backend == "cuda":
        configuration["k1"] = k1
    if backend == "cpu-avx2":
        configuration["threads"] = int(option_value(extra_args, "--threads", "4") or "4")

    payload: dict[str, Any] = {
        "configuration": configuration,
        "samples_ms": [sample] * reps,
        "header_bytes": 20,
        "payload_bytes": count if bits == 8 else (count + 1) // 2,
    }
    if backend == "cuda":
        stage_keys = {
            "resident": ("k1_ms", "k2_ms", "k3_ms"),
            "resident-graph": (),
            "host-origin": ("k1_ms", "k2_ms", "k3_ms", "h2d_ms", "d2h_ms"),
            "gpu-origin": ("k1_ms", "k2_ms", "k3_ms", "d2h_ms"),
        }[boundary]
        for key in stage_keys:
            payload[key] = [sample] * reps
        if boundary == "resident-graph":
            payload["capture_and_instantiate_ms"] = sample
    elif backend == "cpu" and boundary == "gpu-origin":
        payload["d2h_ms"] = [sample] * reps
        payload["cpu_ms"] = [sample] * reps
    return payload


def bench_payload_for_argv(argv: Sequence[str]) -> dict[str, Any]:
    options = list(argv[2:])
    input_path = Path(option_value(options, "--input", "") or "")
    count = input_path.stat().st_size // 4
    warmups = int(option_value(options, "--warmup", "10") or "10")
    reps = int(option_value(options, "--reps", "30") or "30")
    backend = option_value(options, "--backend", "cpu") or "cpu"
    k1 = option_value(options, "--k1", "reference") or "reference"
    reps_index = options.index("--reps")
    extra_args = options[reps_index + 2 :]
    return bench_payload(
        count=count,
        bits=int(option_value(options, "--bits", "8") or "8"),
        backend=backend,
        seed=int(option_value(options, "--seed", "1") or "1"),
        warmups=warmups,
        reps=reps,
        extra_args=extra_args,
        k1=k1,
    )


def process_result(payload: dict[str, Any]):
    from subprocess import CompletedProcess

    return CompletedProcess([], 0, json.dumps(payload), "")
