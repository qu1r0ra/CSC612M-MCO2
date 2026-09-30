"""Shared lifecycle for CUDA resident experiments."""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import statistics
import subprocess
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict

from stoquant.design import CUDA_RESIDENT
from stoquant.host import (
    benchmark_process_affinity,
    probe_readiness_facts,
)
from stoquant.inputs import generate_inputs
from stoquant.provenance import (
    BUILD_RECIPE,
    VerifiedBinary,
    collect_build_commands,
    collect_git_provenance,
    collect_hardware_and_toolchain,
    find_binary,
    query_gpu_state,
)
from stoquant.runner import (
    DEFAULT_COMPRESSION_SEED,
    REAL_BENCH_PROCESS,
    BenchProcess,
    compress_args,
    in_process_warmups,
    run_bench_process,
    warm_up_gpu,
)
from stoquant.snapshot_store import check_snapshot, create_snapshot, snapshot_path

DEFAULT_PROCESSES = 12
DEFAULT_REPS = 30
MIN_WARMUPS = 10
CALIBRATION_REPS = 10
GPU_WARMUP_SECONDS = 20.0
ORDER_SEED = 23
PROBE_RECIPE = "build-stream-probe"
DEFAULT_VARIANTS = ("reference", "optimized")


class ResidentExperimentRecord(TypedDict):
    process: int
    position: int
    slot: int
    variant: str
    count: int
    bits: int
    warmups: int
    reps: int
    samples_ms: list[float]
    k1_ms: list[float]
    k2_ms: list[float]
    k3_ms: list[float]


@dataclass(frozen=True)
class ResidentExperimentConfig:
    root: Path
    kind: str
    counts: Sequence[int]
    bits: Sequence[int]
    processes: int
    reps: int
    gpu_warmup_seconds: float
    in_process_warmup_seconds: float
    output_dir: Path | None = None
    allow_dirty: bool = False
    pilot: bool = False
    evidence: bool = False
    record_readiness: bool = False


@dataclass(frozen=True)
class ResidentExperimentRun:
    root: Path
    git: dict[str, Any]
    verified: VerifiedBinary
    binary: Path
    build_record: dict[str, Any]
    hardware: dict[str, Any]
    probe: dict[str, Any]
    device: dict[str, Any]
    gpu_state_start: dict[str, Any]
    gpu_probe_start: dict[str, Any]
    gpu_probe_end: dict[str, Any]
    gpu_state_end: dict[str, Any]
    initial_gpu_warm_up: dict[str, Any]
    readiness: dict[str, Any] | None
    record_identity: list[dict[str, Any]]
    inputs: dict[int, dict[str, Any]]
    cells: list[tuple[int, int]]
    records: list[ResidentExperimentRecord]
    warmups: dict[tuple[int, int, str], int]
    processes: int
    reps: int
    in_process_warmup_seconds: float
    affinity_mask: int
    excluded_logical_cpus: tuple[int, ...]
    evidence: bool
    output_dir: Path


def variant_order(process: int, variants: Sequence[str] = DEFAULT_VARIANTS) -> tuple[str, ...]:
    """Alternate the two arms by process round; keep a single arm unchanged."""
    if len(variants) < 2 or process % 2 == 0:
        return tuple(variants)
    return tuple(reversed(variants))


def dry_run(root: Path, recipe: str) -> list[str]:
    just = shutil.which("just")
    if just is None:
        raise RuntimeError("just is not on PATH; build provenance cannot be recorded")
    proc = subprocess.run(
        [just, "--dry-run", recipe], cwd=root, capture_output=True, text=True, check=False
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"just --dry-run {recipe} failed; build provenance cannot be recorded: "
            f"{proc.stderr.strip()}"
        )
    text = proc.stderr if proc.stderr.strip() else proc.stdout
    return [line.strip() for line in text.splitlines() if line.strip()]


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def probe_path(root: Path) -> Path:
    path = root / "build" / ("stream_probe.exe" if os.name == "nt" else "stream_probe")
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found. Build it first with just {PROBE_RECIPE}.")
    return path


def run_probe(probe: Path) -> dict[str, Any]:
    proc = subprocess.run(
        [str(probe), "--max-exp", "27"], capture_output=True, text=True, check=False
    )
    if proc.returncode != 0:
        raise RuntimeError(f"stream probe failed: {proc.stderr.strip()}")
    return json.loads(proc.stdout)


def run_resident_arm(
    binary: Path,
    input_path: Path,
    bits: int,
    variant: str,
    warmups: int,
    reps: int,
    process: BenchProcess,
) -> dict[str, Any]:
    payload, error = run_bench_process(
        binary,
        input_path,
        bits=bits,
        backend="cuda",
        extra_args=["--boundary", CUDA_RESIDENT.boundary],
        k1=variant,
        seed=DEFAULT_COMPRESSION_SEED,
        warmups=warmups,
        reps=reps,
        process=process,
    )
    if payload is None:
        raise RuntimeError(error)
    if payload.get("configuration", {}).get("k1") != variant:
        raise RuntimeError(f"bench did not record k1={variant}")
    return payload


def _record_hash(
    binary: Path,
    input_path: Path,
    bits: int,
    variant: str,
    out: Path,
    process: BenchProcess,
) -> str:
    command = compress_args(
        binary,
        input_path,
        out,
        seed=DEFAULT_COMPRESSION_SEED,
        bits=bits,
        tensor_id=0,
        invocation_id=0,
        backend="cuda",
        k1=variant,
    )
    result = process.run(command)
    if result.returncode != 0:
        raise RuntimeError(f"stoquant compress --k1 {variant} failed: {result.stderr.strip()}")
    try:
        return file_sha256(out)
    finally:
        out.unlink(missing_ok=True)


def _verify_variant_records(
    binary: Path,
    paths: dict[int, Path],
    cells: Sequence[tuple[int, int]],
    scratch: Path,
    variants: Sequence[str],
    process: BenchProcess,
) -> list[dict[str, Any]]:
    identity = []
    for count, bits in cells:
        hashes = {
            variant: _record_hash(
                binary, paths[count], bits, variant, scratch / "record.msq", process
            )
            for variant in variants
        }
        identity.append({"count": count, "bits": bits, "record_sha256": hashes})
        if len(set(hashes.values())) != 1:
            raise RuntimeError(f"records differ at count={count} bits={bits}: {hashes}")
    return identity


def run_resident_experiment(
    config: ResidentExperimentConfig,
    *,
    variants: Sequence[str],
    analyze: Callable[[ResidentExperimentRun, Path], None],
    adapter: BenchProcess = REAL_BENCH_PROCESS,
) -> Path:
    """Run shared setup and resident rounds, delegate only result analysis and writing."""
    if not variants:
        raise ValueError("a resident experiment needs at least one K1 variant")
    with benchmark_process_affinity() as affinity:
        return _run_resident_experiment(
            config,
            variants=variants,
            analyze=analyze,
            adapter=adapter,
            affinity=affinity,
        )


def _run_resident_experiment(
    config: ResidentExperimentConfig,
    *,
    variants: Sequence[str],
    analyze: Callable[[ResidentExperimentRun, Path], None],
    adapter: BenchProcess,
    affinity: dict[str, Any],
) -> Path:
    git = collect_git_provenance(config.root)
    final = config.output_dir or snapshot_path(
        config.root, git["code_revision_short"], config.kind, pilot=config.pilot
    )
    check_snapshot(final, git, allow_dirty=config.allow_dirty)
    verified = find_binary(config.root, git_prov=git)
    binary = verified.path
    probe_binary = probe_path(config.root)
    build = collect_build_commands(config.root)
    build.pop("_host_tokens", None)
    if not build["commands"]:
        raise RuntimeError(
            f"just --dry-run {BUILD_RECIPE} failed; build provenance cannot be recorded."
        )
    build_record = {
        "build_stamp": verified.stamp,
        BUILD_RECIPE: build["commands"],
        PROBE_RECIPE: dry_run(config.root, PROBE_RECIPE),
        "sha256": {
            binary.name: file_sha256(binary),
            probe_binary.name: file_sha256(probe_binary),
        },
    }
    readiness = probe_readiness_facts(config.root, git) if config.record_readiness else None
    cells = [(count, bits) for count in config.counts for bits in config.bits]
    with tempfile.TemporaryDirectory(prefix=f"stoquant-{config.kind}-") as scratch_name:
        scratch = Path(scratch_name)
        gpu_state_start = query_gpu_state()
        input_rows = generate_inputs(config.counts, scratch)
        inputs = {
            count: {key: value for key, value in input_rows[count].items() if key != "_path"}
            for count in config.counts
        }
        input_paths = {count: Path(input_rows[count]["_path"]) for count in config.counts}
        identity = (
            _verify_variant_records(binary, input_paths, cells, scratch, variants, adapter)
            if len(variants) > 1
            else []
        )
        warm = warm_up_gpu(
            binary,
            input_paths[max(config.counts)],
            config.gpu_warmup_seconds,
            process=adapter,
        )
        gpu_probe_start = query_gpu_state()
        probe = run_probe(probe_binary)
        gpu_probe_end = query_gpu_state()
        device = probe["device"]
        snapshot = create_snapshot(final, git, allow_dirty=config.allow_dirty)
        out = snapshot.partial
        (out / "stream_probe.json").write_text(json.dumps(probe, indent=2), encoding="utf-8")

        warmups: dict[tuple[int, int, str], int] = {}
        for count, bits in cells:
            for variant in variants:
                payload = run_resident_arm(
                    binary,
                    input_paths[count],
                    bits,
                    variant,
                    MIN_WARMUPS,
                    CALIBRATION_REPS,
                    adapter,
                )
                rep_ms = statistics.median(payload["samples_ms"])
                warmups[count, bits, variant] = in_process_warmups(
                    MIN_WARMUPS, config.in_process_warmup_seconds, rep_ms
                )

        rng = random.Random(ORDER_SEED)
        records: list[ResidentExperimentRecord] = []
        for process_round in range(config.processes):
            order = list(cells)
            rng.shuffle(order)
            for position, (count, bits) in enumerate(order):
                for slot, variant in enumerate(variant_order(process_round, variants)):
                    count_warmups = warmups[count, bits, variant]
                    payload = run_resident_arm(
                        binary,
                        input_paths[count],
                        bits,
                        variant,
                        count_warmups,
                        config.reps,
                        adapter,
                    )
                    records.append(
                        {
                            "process": process_round,
                            "position": position,
                            "slot": slot,
                            "variant": variant,
                            "count": count,
                            "bits": bits,
                            "warmups": count_warmups,
                            "reps": config.reps,
                            "samples_ms": payload["samples_ms"],
                            "k1_ms": payload["k1_ms"],
                            "k2_ms": payload["k2_ms"],
                            "k3_ms": payload["k3_ms"],
                        }
                    )
            print(f"process round {process_round + 1}/{config.processes}", flush=True)
        gpu_state_end = query_gpu_state()

        run = ResidentExperimentRun(
            root=config.root,
            git=git,
            verified=verified,
            binary=binary,
            build_record=build_record,
            hardware=collect_hardware_and_toolchain(config.root),
            probe=probe,
            device=device,
            gpu_state_start=gpu_state_start,
            gpu_probe_start=gpu_probe_start,
            gpu_probe_end=gpu_probe_end,
            gpu_state_end=gpu_state_end,
            initial_gpu_warm_up=warm,
            readiness=readiness,
            record_identity=identity,
            inputs=inputs,
            cells=cells,
            records=records,
            warmups=warmups,
            processes=config.processes,
            reps=config.reps,
            in_process_warmup_seconds=config.in_process_warmup_seconds,
            affinity_mask=affinity["mask"],
            excluded_logical_cpus=tuple(affinity["excluded_logical_cpus"]),
            evidence=config.evidence,
            output_dir=out,
        )
        analyze(run, out)
        snapshot.commit()
    return final
