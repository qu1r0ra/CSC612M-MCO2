"""Statistics, bootstrap CI, speedup and claim comparison, and stage medians."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from stoquant.design import (
    AVX2_COMPARATOR,
    COMPARATOR,
    CPU_GPU_ORIGIN,
    CUDA_GPU_ORIGIN,
    CUDA_HOST_ORIGIN,
    CUDA_RESIDENT,
    CUDA_RESIDENT_GRAPH,
    BenchPath,
)

STAGE_KEYS = ("k1_ms", "k2_ms", "k3_ms", "h2d_ms", "d2h_ms", "cpu_ms")

# Claim rules, fixed before any snapshot is generated (protocol revision 3).
# A direction claim (faster or slower) needs a conservative verdict and no
# boundary inversion. A magnitude claim also needs both sides stable, with the
# stability spread taken between the 10th and 90th percentiles of trial medians
# so one outlier process does not veto a case. The revision 2 rule, which gated
# every claim on the max/min spread, is still computed for comparison.
SPREAD_THRESHOLD = 1.25
STABILITY_THRESHOLD = 1.25
STABILITY_PERCENTILES = (10, 90)
QUANTILE_METHOD = "linear"
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 31
BOOTSTRAP_LEVEL = 0.95
BASELINE_RULE = (
    "every path is measured against a policy-matched CPU baseline: resident, resident-graph, "
    "host-origin, and CPU gpu-origin paths against the CPU comparator (host-host); CUDA "
    "gpu-origin against CPU gpu-origin of the same transfer policy. Pinned paths add "
    "vs_pageable against their pageable twin; CUDA gpu-origin adds a descriptive "
    "vs_comparator with no claim flags"
)
VERDICT_RULE = (
    "verdict is 'faster' when baseline trial-median minimum / candidate trial-median maximum "
    "> 1, 'slower' when baseline trial-median maximum / candidate trial-median minimum < 1, "
    "and 'inconclusive' otherwise"
)
DIRECTION_RULE = (
    "direction_supported: verdict is not 'inconclusive' and boundary_inversion is false "
    "(host-origin pooled median is not below the pooled median of any resident path)"
)
MAGNITUDE_RULE = (
    "magnitude_supported: direction_supported and both the candidate and the baseline have "
    f"spread_p90_p10 <= {STABILITY_THRESHOLD} (90th over 10th percentile of trial medians, "
    f"numpy method '{QUANTILE_METHOD}')"
)
CLAIM_RULE_REV2 = (
    "claim_supported_rev2: verdict is not 'inconclusive', boundary_inversion is false, and "
    f"both the candidate and the baseline have spread_ratio <= {SPREAD_THRESHOLD} "
    "(maximum over minimum trial median)"
)
BOOTSTRAP_RULE = (
    f"{BOOTSTRAP_LEVEL:.0%} percentile bootstrap: resample each path's trial medians with "
    f"replacement, independently, {BOOTSTRAP_RESAMPLES} times from "
    f"numpy.random.default_rng({BOOTSTRAP_SEED}); the statistic is the baseline median of "
    "resampled trial medians over the candidate median of resampled trial medians"
)


def compute_statistics(samples_ms: Sequence[float]) -> dict[str, float]:
    arr = np.asarray(samples_ms, dtype=np.float64)
    if len(arr) == 0:
        return {}
    q25 = float(np.percentile(arr, 25, method=QUANTILE_METHOD))
    q75 = float(np.percentile(arr, 75, method=QUANTILE_METHOD))
    return {
        "median_ms": float(np.median(arr)),
        "iqr_ms": float(q75 - q25),
        "q25_ms": q25,
        "q75_ms": q75,
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
        "mean_ms": float(np.mean(arr)),
        "std_ms": float(np.std(arr)),
    }


def compute_case_statistics(trial_samples: Sequence[Sequence[float]]) -> dict[str, Any]:
    """Pooled statistics plus the between-trial spread of per-trial medians."""
    pooled = [sample for samples in trial_samples for sample in samples]
    stats: dict[str, Any] = compute_statistics(pooled)
    trial_medians = [float(np.median(np.asarray(s, dtype=np.float64))) for s in trial_samples]
    low = min(trial_medians)
    high = max(trial_medians)
    p_low, p_high = np.percentile(
        np.asarray(trial_medians, dtype=np.float64), STABILITY_PERCENTILES, method=QUANTILE_METHOD
    )
    stats["trial_medians_ms"] = trial_medians
    stats["trial_median_min_ms"] = low
    stats["trial_median_max_ms"] = high
    stats["spread_ratio"] = high / low if low > 0 else float("inf")
    stats["unstable_rev2"] = stats["spread_ratio"] > SPREAD_THRESHOLD
    stats["spread_p90_p10"] = float(p_high / p_low) if p_low > 0 else float("inf")
    stats["stable"] = stats["spread_p90_p10"] <= STABILITY_THRESHOLD
    return stats


def bootstrap_speedup_ci(
    baseline_medians: Sequence[float],
    candidate_medians: Sequence[float],
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
    level: float = BOOTSTRAP_LEVEL,
) -> tuple[float, float]:
    """Percentile bootstrap interval of baseline median / candidate median over trials."""
    rng = np.random.default_rng(seed)
    base = np.asarray(baseline_medians, dtype=np.float64)
    cand = np.asarray(candidate_medians, dtype=np.float64)
    base_draws = base[rng.integers(0, len(base), size=(resamples, len(base)))]
    cand_draws = cand[rng.integers(0, len(cand), size=(resamples, len(cand)))]
    ratios = np.median(base_draws, axis=1) / np.median(cand_draws, axis=1)
    tail = 100.0 * (1.0 - level) / 2.0
    low, high = np.percentile(ratios, (tail, 100.0 - tail), method=QUANTILE_METHOD)
    return float(low), float(high)


def compare_speedup(
    baseline: dict[str, Any], candidate: dict[str, Any], point_key: str
) -> dict[str, Any]:
    """Point speedup from pooled medians, a conservative range, and a bootstrap CI."""
    low = baseline["trial_median_min_ms"] / candidate["trial_median_max_ms"]
    high = baseline["trial_median_max_ms"] / candidate["trial_median_min_ms"]
    if low > 1.0:
        verdict = "faster"
    elif high < 1.0:
        verdict = "slower"
    else:
        verdict = "inconclusive"
    ci_low, ci_high = bootstrap_speedup_ci(
        baseline["trial_medians_ms"], candidate["trial_medians_ms"]
    )
    return {
        point_key: baseline["median_ms"] / candidate["median_ms"],
        "speedup_low": low,
        "speedup_high": high,
        "speedup_ci_low": ci_low,
        "speedup_ci_high": ci_high,
        "verdict": verdict,
    }


def claim_support(
    verdict: str, inversion: bool, baseline: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, bool]:
    """The revision 3 direction and magnitude claims, plus the revision 2 claim."""
    direction = verdict != "inconclusive" and not inversion
    return {
        "direction_supported": direction,
        "magnitude_supported": direction and baseline["stable"] and candidate["stable"],
        "claim_supported_rev2": (
            direction and not baseline["unstable_rev2"] and not candidate["unstable_rev2"]
        ),
    }


def baseline_label(path: BenchPath) -> str | None:
    """The policy-matched CPU path each path is measured against."""
    baseline = path.baseline
    return baseline.label if baseline is not None else None


def compare_case_group(cases: Sequence[dict[str, Any]], paths: Sequence[BenchPath]) -> None:
    """Attach verdicts, inversion flags, and claims to one (count, bits) group of cases."""
    by_label = {path.label: case for path, case in zip(paths, cases, strict=True)}
    stats_by_label = {label: case["statistics"] for label, case in by_label.items()}
    cuda_medians = {
        path.key: stats_by_label[path.label]["median_ms"]
        for path in paths
        if path.backend == "cuda" and stats_by_label[path.label] is not None
    }
    comparator = stats_by_label[COMPARATOR.label]
    avx2 = stats_by_label.get(AVX2_COMPARATOR.label)
    # CPU gpu-origin adds a full input download to the comparator, so it must not beat it.
    selected_labels = {path.label for path in paths}
    cpu_inversion = {}
    for policy in ("pageable", "pinned"):
        cpu_path = BenchPath("cpu", CPU_GPU_ORIGIN.boundary, policy)
        candidate = stats_by_label.get(cpu_path.label)
        baseline = stats_by_label.get(cpu_path.nested_paths[0].label)
        if cpu_path.label not in selected_labels:
            cpu_inversion[policy] = False
        else:
            cpu_inversion[policy] = (
                candidate is None
                or baseline is None
                or candidate["median_ms"] < baseline["median_ms"]
            )
    selected_cuda_keys = {path.key for path in paths if path.backend == "cuda"}
    cuda_inversion = {
        "pageable": boundary_inversion(cuda_medians, selected_keys=selected_cuda_keys),
        "pinned": boundary_inversion(cuda_medians, "pinned", selected_keys=selected_cuda_keys),
    }

    def path_inversion(path: BenchPath) -> bool:
        group = "pinned" if path.policy == "pinned" else "pageable"
        if path == AVX2_COMPARATOR:
            # Same boundary and work as the comparator: nothing it could invert.
            return False
        if path.backend == "cpu":
            return cpu_inversion[group]
        if path.boundary == CUDA_GPU_ORIGIN.boundary:
            return cuda_inversion[group] or cpu_inversion[group]
        return cuda_inversion[group]

    for path in paths:
        stats = stats_by_label[path.label]
        if stats is None:
            continue
        if path == COMPARATOR:
            stats.update(
                {
                    "baseline": "",
                    "speedup_vs_cpu": 1.0,
                    "speedup_low": 1.0,
                    "speedup_high": 1.0,
                    "speedup_ci_low": 1.0,
                    "speedup_ci_high": 1.0,
                    "verdict": "comparator",
                    "boundary_inversion": False,
                    "direction_supported": None,
                    "magnitude_supported": None,
                    "claim_supported_rev2": None,
                }
            )
            continue
        inversion = path_inversion(path)
        base_label = baseline_label(path)
        base_stats = stats_by_label.get(base_label) if base_label else None
        if base_stats is None:
            continue
        stats["baseline"] = base_label
        stats.update(compare_speedup(base_stats, stats, "speedup_vs_cpu"))
        stats["boundary_inversion"] = inversion
        if path == AVX2_COMPARATOR:
            # Descriptive only: the paper's claims stay against the scalar comparator.
            stats.update(
                {
                    "descriptive": True,
                    "direction_supported": None,
                    "magnitude_supported": None,
                    "claim_supported_rev2": None,
                }
            )
            continue
        stats.update(claim_support(stats["verdict"], inversion, base_stats, stats))
        if path.backend == "cuda" and avx2 is not None:
            stats["vs_cpu_avx2"] = {
                **compare_speedup(avx2, stats, "speedup_vs_cpu_avx2"),
                "descriptive": True,
            }

        if path.policy == "pinned":
            twin_path = path.pageable_twin
            twin = stats_by_label.get(twin_path.label)
            if twin is not None:
                # Either side's inversion vetoes the pinning comparison.
                pair_inversion = inversion or path_inversion(twin_path)
                vs_pageable = compare_speedup(twin, stats, "speedup_vs_pageable")
                stats["vs_pageable"] = {
                    **vs_pageable,
                    "boundary_inversion": pair_inversion,
                    **claim_support(vs_pageable["verdict"], pair_inversion, twin, stats),
                }
        if (
            path.backend == "cuda"
            and path.boundary == CUDA_GPU_ORIGIN.boundary
            and comparator is not None
        ):
            # Descriptive only: the two sides start from different data locations.
            stats["vs_comparator"] = {
                **compare_speedup(comparator, stats, "speedup_vs_comparator"),
                "descriptive": True,
            }

    resident = stats_by_label.get(CUDA_RESIDENT.label)
    graph = stats_by_label.get(CUDA_RESIDENT_GRAPH.label)
    if resident is not None and graph is not None:
        inversion = cuda_inversion["pageable"]
        vs_res = compare_speedup(resident, graph, "speedup_vs_resident")
        graph["vs_resident"] = {
            **vs_res,
            "boundary_inversion": inversion,
            **claim_support(vs_res["verdict"], inversion, resident, graph),
        }


def boundary_inversion(
    medians: dict[str, float],
    policy: str = "pageable",
    *,
    selected_keys: set[str] | None = None,
) -> bool:
    """CUDA boundaries nest by the work they add; a path must not beat one it contains.

    Keys are CUDA path keys; policy picks one transfer policy. Selected paths
    with missing medians veto dependent claims, while omitted paths do not.
    Host-origin adds copies to a resident path, gpu-origin adds the output copy to
    resident, and host-origin adds the input copy to gpu-origin.
    """
    selected = medians.keys() if selected_keys is None else selected_keys
    for boundary in (CUDA_HOST_ORIGIN.boundary, CUDA_GPU_ORIGIN.boundary):
        path = BenchPath("cuda", boundary, policy)
        if path.key not in selected:
            continue
        candidate = medians.get(path.key)
        if candidate is None:
            return True
        for nested in path.nested_paths:
            if nested.key not in selected:
                continue
            nested_median = medians.get(nested.key)
            if nested_median is None or candidate < nested_median:
                return True
    return False


def compute_stage_medians(runs: Sequence[Mapping[str, Any]]) -> dict[str, float] | None:
    """Median of each timed stage and of the untimed remainder of each repetition."""
    keys = [k for k in STAGE_KEYS if all(k in run for run in runs)]
    if not runs or not keys:
        return None
    medians = {
        key: float(
            np.median(np.concatenate([np.asarray(run[key], dtype=np.float64) for run in runs]))
        )
        for key in keys
    }
    wall = np.concatenate([np.asarray(run["samples_ms"], dtype=np.float64) for run in runs])
    staged = sum(
        np.concatenate([np.asarray(run[key], dtype=np.float64) for run in runs]) for key in keys
    )
    medians["other_ms"] = float(np.median(wall - staged))
    return medians
