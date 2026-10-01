"""Behaviour-equivalence check for the restructure (issue #47).

`capture` runs small fixed-seed configurations of every command, compresses
records through the native binary, re-renders every frozen snapshot under
`results/`, and stores the normalized fingerprint in the primary checkout's
`build/equivalence/<commit>.json`. `compare` builds a fresh fingerprint the
same way and fails on any difference from a stored baseline.

Compared: field names, CSV headers, row and case order, correctness values,
seed-determined values, record and decompressed bytes, input hashes, and the
bytes of every frozen snapshot and of its regenerated report and figures.
Redacted by key path (the leaf stays, so the key structure is still compared):
timings and everything derived from them in a live run (statistics, speedups,
verdicts, K1 bandwidth and the keep decision), timestamps, git provenance, GPU
and host state, and run conditions that depend on the shell. Collapsed to one
leaf (key names included): path-valued provenance, that is build commands and
flags, binary hashes keyed by binary name, and readiness facts.
Build-stamp revisions and executable hashes are redacted because the golden
commit and identical rebuilds change them without changing program behavior.

Later stages may change only the INVOCATION table below. Everything else in
this file, and the configurations it runs, stays fixed so that a baseline
captured before a stage is comparable with a run after it.

    just equivalence capture
    just equivalence compare [--baseline PATH|COMMIT]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Invocation table: the only part later stages may edit.
# ---------------------------------------------------------------------------
INVOCATION = {
    "binary": REPO / "build" / ("stoquant.exe" if os.name == "nt" else "stoquant"),
    "bench-matrix": [sys.executable, "-m", "stoquant", "bench-matrix"],
    "unbiasedness": [sys.executable, "-m", "stoquant", "unbiasedness"],
    "k1-baseline": [sys.executable, "-m", "stoquant", "k1-baseline"],
    "k1-ab": [sys.executable, "-m", "stoquant", "k1-ab"],
    "figures": [sys.executable, "-m", "stoquant", "figures"],
    # Environment variables removed before every run, so each run uses defaults.
    "scrubbed_env_prefixes": ("STOQUANT_",),
}
# ---------------------------------------------------------------------------

FORMAT_VERSION = 1
PATH_TOKEN = "<path>"

# Live-run configurations. Warm-ups are zero so warm-up counts stay fixed.
NO_WARMUP = ["--gpu-warmup-seconds", "0", "--case-warmup-seconds", "0", "--warmup-seconds", "0"]
BENCH_COMMON = [
    "--bits", "4", "8",
    "--backends", "cpu", "cuda", "cpu-avx2",
    "--boundaries", "gpu-origin",
    "--transfer-policies", "pageable", "pinned",
    "--trials", "10", "--reps", "3", "--warmup", "1",
    *NO_WARMUP,
    "--allow-dirty", "--ignore-readiness",
]  # fmt: skip
K1_COMMON = [
    "--bits", "4", "8", "--processes", "2", "--reps", "3",
    "--gpu-warmup-seconds", "0", "--in-process-warmup-seconds", "0", "--allow-dirty",
]  # fmt: skip
LIVE_RUNS = {
    "bench-dense": ("bench-matrix", ["--counts", "1024", "4096", *BENCH_COMMON]),
    "bench-sparse": (
        "bench-matrix",
        ["--input-family", "sparse", "--counts", "1024", *BENCH_COMMON],
    ),
    "bench-model": (
        "bench-matrix",
        ["--input-family", "model", "--model-limit", "2", *BENCH_COMMON],
    ),
    "unbiasedness": (
        "unbiasedness",
        ["--seeds", "8", "--backends", "cpu", "cuda", "--allow-dirty"],
    ),
    "k1-baseline": ("k1-baseline", ["--counts", "1024", "65536", *K1_COMMON]),
    # F5 draws only its fixed counts, so the A/B uses two of them.
    "k1-ab": ("k1-ab", ["--counts", "16384", "262144", *K1_COMMON]),
}

RECORD_INPUT_SEED = 2026
RECORD_COUNTS = (1000, 65536)
RECORD_SEEDS = (42, 7)
RECORD_BACKENDS = ("cpu", "cpu-avx2", "cuda")

# Key-path rules. Paths join dict keys and list indices with "/".
# COLLAPSE hides a whole subtree, key names included.
COLLAPSE = [
    (r"(.*/)?build_flags", "path-valued provenance"),
    (r"build", "path-valued provenance"),
    (r"(run_conditions/readiness/)?facts|readiness_facts", "path-valued provenance"),
    (r"(.*/)?gpu_state\w*", "gpu state"),
    (r"git_provenance|git", "git provenance"),
    (r"(.*/)?(hardware|toolkit_and_driver)|device", "environment"),
    # Lists whose length depends on timings or on the tree and machine state.
    (r"decision/(kept|failing_cells|descriptive_slowdowns)", "timing-derived"),
    (r"(run_conditions/)?(evidence|non_evidence_reasons)", "run conditions"),
    (r"run_conditions/readiness/(passed|failures|overridden)", "run conditions"),
]
REDACT = [
    (r"(.*/)?(created_at_utc|created_utc|captured_at_utc)|date", "timestamp"),
    (r"(.*/)?(code_revision|code_revision_short|git_dirty)", "git provenance"),
    (r"(.*/)?build_stamp/revision", "git provenance"),
    (r"(.*/)?build_stamp/binary_sha256", "build provenance"),
    (r"(.*/)?numpy_version", "environment"),
    (r"(.*/)?\w*_ms(/.*)?", "timing"),
    (r"statistics/(?!baseline$|descriptive$).*", "timing-derived"),
    (r"cells/\d+(/arms/\w+)?/(k1|total)/.*", "timing-derived"),
    (r"cells/\d+/(k1|total)_comparison/.*", "timing-derived"),
    (
        (
            r"cells/\d+(/arms/\w+)?/(effective_gbps|fraction_of_peak|fraction_of_ceiling"
            r"|ideal_ms_at_ceiling|headroom)"
        ),
        "timing-derived",
    ),
    (r"probe_ceiling_gbps|probe_ceiling_size/.*|sizes/\d+/best_\w+", "timing-derived"),
    (r"initial_gpu_warm_up/seconds_elapsed", "timing"),
    (r"run_conditions/affinity/previous_mask", "run conditions"),
]
CSV_COMPARED_COLUMNS = {
    "summary.csv": {
        "count", "bits", "backend", "boundary", "transfer_policy", "path_label",
        "correctness", "warmup", "reps", "trials", "baseline", "input_family", "input_key",
    },
    "processes.csv": {"process", "position", "count", "bits", "warmups", "reps"},
}  # fmt: skip
# Drawn from seed-determined values only, so hashed like any other file.
SEEDED_FIGURES = {"f_unbiasedness.png"}
VEC_VERDICT = re.compile(
    r"info C500[12]: (loop vectorized|loop not vectorized due to reason '\d+')"
)


def rules(patterns: list[tuple[str, str]]) -> list[tuple[re.Pattern[str], str]]:
    return [(re.compile(p), reason) for p, reason in patterns]


COLLAPSE_RULES = rules(COLLAPSE)
REDACT_RULES = rules(REDACT)


def match(path: str, compiled: list[tuple[re.Pattern[str], str]]) -> str | None:
    for pattern, reason in compiled:
        if pattern.fullmatch(path):
            return reason
    return None


def flatten(value: Any, path: str, out: dict[str, Any]) -> None:
    if path:
        reason = match(path, COLLAPSE_RULES)
        if reason:
            out[path] = f"<collapsed: {reason}>"
            return
    if isinstance(value, dict):
        if not value:
            out[path] = {}
        for key, child in value.items():
            flatten(child, f"{path}/{key}" if path else str(key), out)
    elif isinstance(value, list):
        if not value:
            out[path] = []
        for i, child in enumerate(value):
            flatten(child, f"{path}/{i}" if path else str(i), out)
    else:
        reason = match(path, REDACT_RULES)
        if reason:
            value = f"<redacted: {reason}>"
        out[path] = value


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fingerprint_csv(path: Path) -> dict[str, Any]:
    compared = CSV_COMPARED_COLUMNS.get(path.name)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    out: dict[str, Any] = {"header": rows[0] if rows else []}
    for r, row in enumerate(rows[1:]):
        for column, cell in zip(rows[0], row, strict=True):
            keep = compared is None or column in compared
            out[f"row/{r}/{column}"] = cell if keep else "<redacted: timing-derived>"
    return out


def fingerprint_vec_report(path: Path) -> dict[str, Any]:
    """Per-report verdict counts: function names and source paths change in stage 1."""
    verdicts = Counter(
        m.group(1) for m in VEC_VERDICT.finditer(path.read_text(encoding="utf-8", errors="replace"))
    )
    return dict(sorted(verdicts.items()))


def fingerprint_live_dir(root: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for file in sorted(p for p in root.rglob("*") if p.is_file()):
        name = file.relative_to(root).as_posix()
        if file.suffix == ".json":
            flat: dict[str, Any] = {}
            flatten(json.loads(file.read_text(encoding="utf-8")), "", flat)
            out[name] = flat
        elif file.suffix == ".csv":
            out[name] = fingerprint_csv(file)
        elif file.name.startswith("msvc_vectorization_report"):
            out[name] = fingerprint_vec_report(file)
        elif file.suffix == ".png" and file.name not in SEEDED_FIGURES:
            out[name] = "<present: drawn from timings>"
        else:
            out[name] = sha256(file)
    return out


def hash_tree(root: Path) -> dict[str, str]:
    return {
        p.relative_to(root).as_posix(): sha256(p) for p in sorted(root.rglob("*")) if p.is_file()
    }


def scrub_paths(text: str, *roots: Path) -> str:
    for root in roots:
        for form in {str(root), root.as_posix()}:
            text = text.replace(form, PATH_TOKEN)
    return text


def child_env() -> dict[str, str]:
    prefixes = INVOCATION["scrubbed_env_prefixes"]
    return {k: v for k, v in os.environ.items() if not k.startswith(prefixes)}


def run(argv: Sequence[str | Path], log: Path) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        [str(a) for a in argv],
        cwd=REPO,
        env=child_env(),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    command = subprocess.list2cmdline([str(a) for a in argv])
    log.write_text(
        f"$ {command}\n--- stdout\n{result.stdout}\n--- stderr\n{result.stderr}",
        encoding="utf-8",
    )
    return result


def run_or_fail(argv: Sequence[str | Path], log: Path) -> None:
    result = run(argv, log)
    if result.returncode != 0:
        tail = "\n".join(result.stderr.strip().splitlines()[-15:])
        raise SystemExit(f"command failed ({result.returncode}), log {log}:\n{tail}")


def live_runs(work: Path) -> dict[str, Any]:
    out = {}
    for name, (command, args) in LIVE_RUNS.items():
        print(f"live run: {name}", flush=True)
        target = work / "live" / name
        log = work / "logs" / f"{name}.log"
        # Exit codes are compared: 8 seeds are too few for the unbiasedness gate.
        result = run([*INVOCATION[command], "--output-dir", target, *args], log)
        if not target.exists():
            target = target.with_name(target.name + ".partial")
        if not target.exists():
            raise SystemExit(f"{name} wrote nothing ({result.returncode}), log {log}")
        out[name] = {"exit_code": result.returncode, "files": fingerprint_live_dir(target)}
    return out


def record_inputs() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(RECORD_INPUT_SEED)
    inputs = {f"normal_n{n}": rng.standard_normal(n, dtype=np.float32) for n in RECORD_COUNTS}
    mixed = rng.standard_normal(4099, dtype=np.float32)
    mixed[::3] = 0.0
    mixed[1::7] *= 1.0e4
    inputs["mixed_n4099"] = mixed
    inputs["constant_n512"] = np.full(512, -2.5, dtype=np.float32)
    return inputs


def record_variants() -> list[tuple[str, list[str]]]:
    variants = []
    for backend in RECORD_BACKENDS:
        k1_modes = ("reference", "optimized") if backend == "cuda" else (None,)
        for bits in (4, 8):
            for seed in RECORD_SEEDS:
                for k1 in k1_modes:
                    label = f"{backend}_b{bits}_s{seed}" + (f"_{k1}" if k1 else "")
                    args = ["--backend", backend, "--bits", str(bits), "--seed", str(seed)]
                    if k1:
                        args += ["--k1", k1]
                    variants.append((label, args))
    ids = ["--tensor-id", "3", "--invocation-id", "5"]
    for backend in RECORD_BACKENDS:
        variants.append((f"{backend}_b4_s42_t3_i5", ["--backend", backend, "--bits", "4",
                                                      "--seed", "42", *ids]))  # fmt: skip
    return variants


def records(work: Path) -> dict[str, Any]:
    print("records", flush=True)
    folder = work / "records"
    folder.mkdir(parents=True)
    binary = INVOCATION["binary"]
    out: dict[str, Any] = {}
    for name, values in record_inputs().items():
        source = folder / f"{name}.f32"
        values.astype("<f4").tofile(source)
        out[f"{name}/input"] = sha256(source)
        for label, args in record_variants():
            record = folder / f"{name}_{label}.rec"
            decoded = folder / f"{name}_{label}.out.f32"
            log = work / "logs" / f"record_{name}_{label}"
            run_or_fail(
                [binary, "compress", "--input", source, "--output", record, *args],
                log.with_name(f"{log.name}_compress.log"),
            )
            run_or_fail(
                [binary, "decompress", "--input", record, "--output", decoded],
                log.with_name(f"{log.name}_decompress.log"),
            )
            out[f"{name}/{label}/record"] = sha256(record)
            out[f"{name}/{label}/decompressed"] = sha256(decoded)
    return out


def frozen_snapshots(work: Path) -> dict[str, Any]:
    """Content hashes of every frozen snapshot and of what each renderer makes from it."""
    results = REPO / "results"
    snapshots = (
        sorted(p for p in results.iterdir() if p.is_dir() and p.name != "pilots")
        if results.is_dir()
        else []
    )
    out: dict[str, Any] = {"content": {p.name: hash_tree(p) for p in snapshots}}
    renders: dict[str, Any] = {}
    renders_root = work / "renders"
    for snap in snapshots:
        if (snap / "manifest.json").exists():
            renders[f"figures/{snap.name}"] = render(
                [*INVOCATION["figures"], snap], renders_root / snap.name, work
            )
        if (snap / "summary.json").exists() and snap.name.endswith("-k1-ab"):
            copy = renders_root / f"k1-ab-figure/{snap.name}"
            shutil.copytree(snap, copy)
            (copy / "f5_k1_stages.png").unlink(missing_ok=True)
            log = work / "logs" / f"f5_{snap.name}.log"
            result = run([*INVOCATION["k1-ab"], "--figure", copy], log)
            renders[f"k1-ab-figure/{snap.name}"] = outcome(
                result, copy, work, only="f5_k1_stages.png"
            )
    for bench in (p for p in snapshots if (p / "manifest.json").exists()):
        for ab in (p for p in snapshots if p.name.endswith("-k1-ab")):
            key = f"figures-k1-ab/{bench.name}+{ab.name}"
            renders[key] = render(
                [*INVOCATION["figures"], bench, "--k1-ab", ab],
                renders_root / f"{bench.name}+{ab.name}",
                work,
            )
    out["renders"] = renders
    return out


def render(argv: list[Any], target: Path, work: Path) -> dict[str, Any]:
    print(f"render: {target.name}", flush=True)
    target.mkdir(parents=True)
    result = run([*argv, "--output-dir", target], work / "logs" / f"render_{target.name}.log")
    return outcome(result, target, work)


def outcome(
    result: subprocess.CompletedProcess[str], target: Path, work: Path, only: str | None = None
) -> dict[str, Any]:
    if result.returncode != 0:
        last = result.stderr.strip().splitlines()[-1] if result.stderr.strip() else ""
        return {"error": scrub_paths(last, work, REPO)}
    files = hash_tree(target)
    return {k: v for k, v in files.items() if only is None or Path(k).name == only}


def results_state() -> tuple[str, dict[str, str]]:
    status = subprocess.run(
        ["git", "status", "--porcelain", "--", "results"],
        cwd=REPO, capture_output=True, text=True, check=True,
    ).stdout  # fmt: skip
    return status, hash_tree(REPO / "results")


def build_fingerprint(work: Path) -> dict[str, Any]:
    if not INVOCATION["binary"].exists():
        raise SystemExit(f"missing {INVOCATION['binary']}; build it first")
    before = results_state()
    (work / "logs").mkdir(parents=True)
    fingerprint = {
        "frozen": frozen_snapshots(work),
        "records": records(work),
        "live": live_runs(work),
    }
    if results_state() != before:
        raise SystemExit("results/ changed during the run; restore it before trusting anything")
    return fingerprint


def flatten_fingerprint(value: Any, path: str = "") -> dict[str, Any]:
    if isinstance(value, dict) and value:
        out: dict[str, Any] = {}
        for key, child in value.items():
            out.update(flatten_fingerprint(child, f"{path}/{key}" if path else str(key)))
        return out
    reason = match(path, REDACT_RULES)
    if reason:
        value = f"<redacted: {reason}>"
    return {path: value}


def diff(baseline: dict[str, Any], fresh: dict[str, Any]) -> list[str]:
    old, new = flatten_fingerprint(baseline), flatten_fingerprint(fresh)
    lines = []
    for key in sorted(old.keys() | new.keys()):
        if key not in new:
            lines.append(f"- {key} = {old[key]!r}")
        elif key not in old:
            lines.append(f"+ {key} = {new[key]!r}")
        elif old[key] != new[key]:
            lines.append(f"~ {key}: {old[key]!r} -> {new[key]!r}")
    return lines


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout.strip()


def default_store() -> Path:
    common = Path(git("rev-parse", "--path-format=absolute", "--git-common-dir"))
    return common.parent / "build" / "equivalence"


def work_dir() -> Path:
    work = REPO / "build" / "equivalence-work"
    if work.exists():
        shutil.rmtree(work)
    return work


def capture(args: argparse.Namespace) -> int:
    if git("status", "--porcelain", "--untracked-files=no") and not args.allow_dirty:
        raise SystemExit("tracked files are modified; a baseline must match its commit")
    store = args.store or default_store()
    commit = git("rev-parse", "HEAD")
    target = store / f"{commit}.json"
    if target.exists():
        raise SystemExit(f"baseline {target} already exists; it is never overwritten")
    fingerprint = build_fingerprint(work_dir())
    store.mkdir(parents=True, exist_ok=True)
    payload = {"format": FORMAT_VERSION, "commit": commit, "fingerprint": fingerprint}
    target.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"baseline written: {target}")
    return 0


def resolve_baseline(value: str | None, store: Path) -> Path:
    """Pick the baseline file: a path, a commit prefix in the store, or the only one stored.

    A commit prefix keeps the recipe usable when the primary checkout's path
    contains spaces, which `just` splits when it forwards arguments.
    """
    stored = sorted(store.glob("*.json")) if store.exists() else []
    if value is None:
        if len(stored) != 1:
            raise SystemExit(f"{len(stored)} baselines in {store}; pass --baseline")
        return stored[0]
    if Path(value).is_file():
        return Path(value)
    matches = [path for path in stored if path.stem.startswith(value)]
    if len(matches) != 1:
        raise SystemExit(f"{len(matches)} baselines in {store} match {value!r}")
    return matches[0]


def compare(args: argparse.Namespace) -> int:
    baseline_path = resolve_baseline(args.baseline, args.store or default_store())
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    if baseline.get("format") != FORMAT_VERSION:
        raise SystemExit(f"{baseline_path} has format {baseline.get('format')}")
    fresh = json.loads(json.dumps(build_fingerprint(work_dir())))
    differences = diff(baseline["fingerprint"], fresh)
    compared = len(flatten_fingerprint(fresh))
    if differences:
        print(f"NOT EQUIVALENT to {baseline['commit'][:7]}: {len(differences)} differences")
        for line in differences[: args.limit]:
            print(line)
        if len(differences) > args.limit:
            print(f"... {len(differences) - args.limit} more")
        return 1
    print(f"equivalent to {baseline['commit'][:7]} ({compared} leaves)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    sub = parser.add_subparsers(dest="mode", required=True)
    cap = sub.add_parser("capture", help="Store a baseline for the current commit")
    cap.add_argument("--store", type=Path, help="Baseline folder (default: primary checkout)")
    cap.add_argument("--allow-dirty", action="store_true", help="Capture from modified files")
    cmp_ = sub.add_parser("compare", help="Diff a fresh fingerprint against a baseline")
    cmp_.add_argument(
        "--baseline",
        help="Baseline file or commit prefix in the store (default: the only stored one)",
    )
    cmp_.add_argument("--store", type=Path, help="Baseline folder (default: primary checkout)")
    cmp_.add_argument("--limit", type=int, default=200, help="Differences to print")
    args = parser.parse_args(argv)
    return capture(args) if args.mode == "capture" else compare(args)


if __name__ == "__main__":
    sys.exit(main())
