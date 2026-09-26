"""Correctness layer 3: the full empirical-expectation suite.

For each suite input and bit width, `mco2 expect` decodes the record of every
seed in 1..T on each backend and returns per-element sums of the decoded values
and of their squares. The suite checks that the CPU and CUDA sums are
byte-identical, gates every element's empirical mean on the 5-sigma rule of the
technical contract, reports the variance, and draws `f_unbiasedness.png`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from benchmark_driver import collect_git_provenance, find_binary
from input_families import (
    DEFAULT_INPUT_SEED,
    MODEL_NAME,
    SPARSE_MASK_SEED,
    SPARSE_ZERO_FRACTION,
    course_vector,
    dense_vectors,
    model_tensor_by_name,
    model_tensor_seed,
    model_tensor_values,
    sparsify,
)

SUITE_SEEDS = 4096
SEED_START = 1
SUITE_COUNT = 1 << 14
SUITE_MODEL_TENSOR = "layer1.0.conv1.weight"
SIGMA = 5.0
BIT_WIDTHS = (4, 8)
SIGNED_LIMITS = {4: 7, 8: 127}
BACKENDS = ("cpu", "cuda")
RESULTS = "unbiasedness.json"
FIGURE = "f_unbiasedness.png"
COLORS = {
    "course": "#555555",
    "dense_n16384": "#0072B2",
    "sparse_n16384": "#009E73",
    f"model_{SUITE_MODEL_TENSOR}": "#E69F00",
}
GATE_RULE = (
    "each element with realised rounding probability q strictly between 0 and 1 satisfies "
    "|mean - E| <= 5 * delta * sqrt(q * (1 - q) / T), where E is the decoded expectation "
    "(1 - q) * dec(l) + q * dec(l + 1), delta = |dec(l + 1) - dec(l)|, l = floor(a), "
    "a = min(|x| / scale * s, s) in FP32, and q = floor(fl32(p * 2^32)) / 2^32 is the "
    "Bernoulli probability the rounding comparison realises for p = a - l; every element "
    "with q = 0 has mean exactly dec(l)"
)


def suite_inputs(seed: int = DEFAULT_INPUT_SEED) -> list[dict[str, Any]]:
    """The four suite inputs with their provenance and SHA-256.

    The dense vector is the first draw of the seed's generator, so it matches the
    `n16384` input of a matrix run only when `--counts` starts at 16384.
    """
    course, _ = course_vector()
    dense = dense_vectors([SUITE_COUNT], seed)[0]
    sparse, realised = sparsify(dense)
    tensor = model_tensor_by_name(SUITE_MODEL_TENSOR)
    items = [
        {
            "name": "course",
            "values": course,
            "provenance": {
                "input_family": "course",
                "generator": (
                    "fixed construction: signed zeros, +/-2, +/-2k/7, +/-2k/127, "
                    "and a ramp inside +/-1.95"
                ),
            },
        },
        {
            "name": f"dense_n{SUITE_COUNT}",
            "values": dense,
            "provenance": {
                "input_family": "dense",
                "generator": "numpy.random.default_rng",
                "bit_generator": "PCG64",
                "seed": seed,
                "draw": f"first normal draw of {SUITE_COUNT} values, cast to FP32",
            },
        },
        {
            "name": f"sparse_n{SUITE_COUNT}",
            "values": sparse,
            "provenance": {
                "input_family": "sparse",
                "seed": seed,
                "mask_seed": SPARSE_MASK_SEED,
                "zero_fraction_target": SPARSE_ZERO_FRACTION,
                "zero_fraction_realised": realised,
            },
        },
        {
            "name": f"model_{SUITE_MODEL_TENSOR}",
            "values": model_tensor_values(tensor, seed),
            "provenance": {
                "input_family": "model",
                "model": MODEL_NAME,
                "tensor_name": tensor.name,
                "tensor_shape": list(tensor.shape),
                "seed": model_tensor_seed(tensor, seed),
            },
        },
    ]
    for item in items:
        data = item["values"].astype("<f4").tobytes()
        item["provenance"]["sha256"] = hashlib.sha256(data).hexdigest()
    return items


def run_expect(
    binary: Path, input_path: Path, output_path: Path, backend: str, bits: int, seeds: int
) -> dict[str, Any]:
    result = subprocess.run(
        [
            str(binary),
            "expect",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--seeds",
            str(seeds),
            "--seed-start",
            str(SEED_START),
            "--backend",
            backend,
            "--bits",
            str(bits),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"mco2 expect ({backend}, {bits}-bit) failed: {result.stderr.strip()}")
    return json.loads(result.stdout)


def decoded_level(level: np.ndarray, negative: np.ndarray, s: int, scale: np.float32) -> np.ndarray:
    """Decode a magnitude level as the codec does, in FP32."""
    value = (level.astype(np.float32) / np.float32(s)) * scale
    return np.where(negative & (level != 0), -value, value).astype(np.float64)


def expectation(x: np.ndarray, scale: np.float32, s: int) -> dict[str, np.ndarray]:
    """Per-element decoded expectation, step and realised probability."""
    scaled = np.minimum((np.abs(x) / scale) * np.float32(s), np.float32(s))
    lower = np.floor(scaled)
    p = scaled - lower
    q = np.floor((p * np.float32(4294967296.0)).astype(np.float64)) / 4294967296.0
    negative = np.signbit(x)
    low = decoded_level(lower, negative, s, scale)
    high = decoded_level(np.minimum(lower + 1, np.float32(s)), negative, s, scale)
    return {
        "a": scaled.astype(np.float64),
        "p": p.astype(np.float64),
        "q": q,
        "low": low,
        "expected": (1.0 - q) * low + q * high,
        "delta": np.abs(high - low),
    }


def analyse(
    x: np.ndarray, scale: np.float32, s: int, sums: np.ndarray, squares: np.ndarray, seeds: int
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    e = expectation(x, scale, s)
    mean = sums / seeds
    stochastic = e["q"] > 0.0
    sigma = e["delta"] * np.sqrt(e["q"] * (1.0 - e["q"]) / seeds)
    ratio = np.zeros_like(mean)
    ratio[stochastic] = np.abs(mean - e["expected"])[stochastic] / (SIGMA * sigma[stochastic])
    exact = mean[~stochastic] == e["low"][~stochastic]
    variance = np.maximum(squares / seeds - mean * mean, 0.0)
    expected_variance = e["delta"] ** 2 * e["q"] * (1.0 - e["q"])
    # FP32 rounding of a and of the decoded levels moves E off x by a tiny fraction of a step.
    gap_steps = np.abs(e["expected"] - x.astype(np.float64)) / (float(scale) / s)
    result = {
        "count": int(x.size),
        "scale": float(scale),
        "scale_bits": f"0x{np.float32(scale).view(np.uint32):08x}",
        "signed_limit": s,
        "stochastic_elements": int(stochastic.sum()),
        "deterministic_elements": int((~stochastic).sum()),
        "deterministic_exact": bool(exact.all()),
        "max_error_over_bound": float(ratio.max()) if stochastic.any() else 0.0,
        "elements_over_bound": int((ratio > 1.0).sum()),
        "max_abs_z": float(ratio.max() * SIGMA) if stochastic.any() else 0.0,
        "variance_ratio_pooled": (
            float(variance[stochastic].sum() / expected_variance[stochastic].sum())
            if stochastic.any()
            else None
        ),
        "max_expectation_gap_steps": float(gap_steps.max()),
        "passed": bool(exact.all() and not (ratio > 1.0).any()),
    }
    arrays = {"x": x.astype(np.float64), "mean": mean, "ratio": ratio, "stochastic": stochastic}
    return result, arrays


def run_suite(
    binary: Path,
    work_dir: Path,
    seeds: int = SUITE_SEEDS,
    backends: tuple[str, ...] = BACKENDS,
    input_seed: int = DEFAULT_INPUT_SEED,
) -> tuple[list[dict[str, Any]], dict[tuple[str, int], dict[str, Any]]]:
    work_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    plot_data: dict[tuple[str, int], dict[str, Any]] = {}
    for item in suite_inputs(input_seed):
        x = item["values"]
        input_path = work_dir / f"{item['name']}.f32"
        input_path.write_bytes(x.astype("<f4").tobytes())
        for bits in BIT_WIDTHS:
            raw: dict[str, bytes] = {}
            meta: dict[str, dict[str, Any]] = {}
            for backend in backends:
                out = work_dir / f"{item['name']}_bits{bits}_{backend}.f64"
                meta[backend] = run_expect(binary, input_path, out, backend, bits, seeds)
                raw[backend] = out.read_bytes()
            identical = len({raw[b] for b in backends}) == 1
            sums_all = np.frombuffer(raw[backends[0]], dtype="<f8")
            scale = np.uint32(int(meta[backends[0]]["scale_bits"], 16)).view(np.float32)
            s = SIGNED_LIMITS[bits]
            summary, arrays = analyse(x, scale, s, sums_all[: x.size], sums_all[x.size :], seeds)
            summary.update(
                {
                    "input": item["name"],
                    "bits": bits,
                    "seeds": seeds,
                    "seed_start": SEED_START,
                    "backends": list(backends),
                    "backends_identical": identical,
                    "input_provenance": item["provenance"],
                }
            )
            summary["passed"] = summary["passed"] and identical
            results.append(summary)
            plot_data[(item["name"], bits)] = {**arrays, "scale": float(scale), "s": s}
    return results, plot_data


def envelope(u: np.ndarray, seeds: int) -> np.ndarray:
    frac = np.abs(u) - np.floor(np.abs(u))
    return SIGMA * np.sqrt(frac * (1.0 - frac) / seeds)


def plot_unbiasedness(
    plot_data: dict[tuple[str, int], dict[str, Any]], seeds: int, out: Path
) -> None:
    fig, axes = plt.subplots(len(BIT_WIDTHS), 2, figsize=(11.0, 4.0 * len(BIT_WIDTHS)))
    for row, bits in enumerate(BIT_WIDTHS):
        left, right = axes[row]
        entries = [(name, d) for (name, b), d in plot_data.items() if b == bits]
        span = 0.0
        for name, d in entries:
            step = d["scale"] / d["s"]
            u = d["x"] / step
            span = max(span, float(np.abs(u).max()))
            left.scatter(
                u,
                (d["mean"] - d["x"]) / step,
                s=2,
                color=COLORS.get(name, "#CC79A7"),
                label=name,
                rasterized=True,
            )
        grid = np.linspace(-span, span, 20001)
        bound = envelope(grid, seeds)
        left.fill_between(grid, -bound, bound, color="#BBBBBB", alpha=0.4, lw=0, label="±5σ")
        left.set_xlabel("input (quantization steps)")
        left.set_ylabel("mean − input (quantization steps)")
        left.set_title(f"{bits}-bit: empirical mean over {seeds} seeds")
        left.legend(fontsize=7, markerscale=4, loc="upper right")

        ratios = [d["ratio"][d["stochastic"]] for _, d in entries]
        top = max(1.05, *(float(r.max()) for r in ratios if r.size))
        right.hist(
            ratios,
            bins=np.linspace(0.0, top, 43),
            stacked=True,
            color=[COLORS.get(name, "#CC79A7") for name, _ in entries],
            label=[name for name, _ in entries],
        )
        right.axvline(1.0, color="black", lw=1, ls="--", label="5σ bound")
        right.set_xlabel("|mean − expectation| / 5σ bound")
        right.set_ylabel("elements")
        right.set_title(f"{bits}-bit: error against the bound")
        right.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Correctness layer 3 expectation suite")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for results")
    parser.add_argument("--seeds", type=int, default=SUITE_SEEDS, help="Seeds per case")
    parser.add_argument(
        "--backends", nargs="+", choices=BACKENDS, default=list(BACKENDS), help="Backends"
    )
    parser.add_argument("--input-seed", type=int, default=DEFAULT_INPUT_SEED)
    parser.add_argument("--allow-existing", action="store_true")
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Run from an uncommitted tree (records the dirty files; not for evidence)",
    )
    args = parser.parse_args()
    if args.seeds < 1:
        parser.error("--seeds must be at least 1")
    root = Path(__file__).resolve().parent

    git_prov = collect_git_provenance(root)
    now = datetime.now(UTC)
    target = args.output_dir or (
        root / "results" / f"{now:%Y-%m-%d}-{git_prov['code_revision_short']}-unbiasedness"
    )
    if target.exists() and not args.allow_existing:
        sys.exit(f"Results directory already exists and snapshots are frozen: {target}")
    if git_prov["git_dirty"] and not args.allow_dirty:
        sys.exit(
            "Working tree is dirty; commit first or pass --allow-dirty for a non-evidence run: "
            + "; ".join(git_prov["dirty_files"])
        )

    created = not target.exists()
    work_dir = target / "_temp"
    try:
        binary = find_binary(root)
        target.mkdir(parents=True, exist_ok=True)
        results, plot_data = run_suite(
            binary, work_dir, args.seeds, tuple(args.backends), args.input_seed
        )
    except (RuntimeError, OSError) as exc:
        shutil.rmtree(target if created else work_dir, ignore_errors=True)
        sys.exit(f"Unbiasedness suite failed: {exc}")
    shutil.rmtree(work_dir, ignore_errors=True)

    plot_unbiasedness(plot_data, args.seeds, target / FIGURE)
    reasons = []
    if git_prov["git_dirty"]:
        reasons.append("dirty git tree")
    if args.seeds != SUITE_SEEDS:
        reasons.append(f"{args.seeds} seeds instead of {SUITE_SEEDS}")
    if tuple(args.backends) != BACKENDS:
        reasons.append("not every backend")
    report = {
        "created_at_utc": now.isoformat(),
        "git_provenance": git_prov,
        "evidence": not reasons,
        "non_evidence_reasons": reasons,
        "seeds": args.seeds,
        "seed_start": SEED_START,
        "sigma": SIGMA,
        "gate_rule": GATE_RULE,
        "variance": "reported as the pooled ratio of observed to expected variance; not gated",
        "numpy_version": np.__version__,
        "figure": FIGURE,
        "all_passed": all(r["passed"] for r in results),
        "cases": results,
    }
    (target / RESULTS).write_text(json.dumps(report, indent=2), encoding="utf-8")
    for r in results:
        print(
            f"{r['input']:>32} {r['bits']}-bit  max err/bound {r['max_error_over_bound']:.3f}  "
            f"var ratio {r['variance_ratio_pooled'] or 0.0:.4f}  "
            f"identical {r['backends_identical']}  "
            f"{'pass' if r['passed'] else 'FAIL'}"
        )
    print(f"Results written to {target}")
    if not report["all_passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
