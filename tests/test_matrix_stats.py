import pytest
from _bench_test_support import fake_stats

from stoquant.design import (
    build_paths,
)
from stoquant.stats import (
    BOOTSTRAP_SEED,
    bootstrap_speedup_ci,
    boundary_inversion,
    claim_support,
    compare_case_group,
    compare_speedup,
    compute_case_statistics,
)


def test_speedup_verdicts_use_trial_median_ranges():
    cpu = compute_case_statistics([[10.0, 10.0], [11.0, 11.0]])
    fast = compute_case_statistics([[1.0, 1.0], [1.1, 1.1]])
    slow = compute_case_statistics([[20.0, 20.0], [22.0, 22.0]])
    overlap = compute_case_statistics([[9.0, 9.0], [12.0, 12.0]])

    assert compare_speedup(cpu, fast, "speedup_vs_cpu")["verdict"] == "faster"
    assert compare_speedup(cpu, slow, "speedup_vs_cpu")["verdict"] == "slower"
    result = compare_speedup(cpu, overlap, "speedup_vs_cpu")
    assert result["verdict"] == "inconclusive"
    assert result["speedup_low"] < 1.0 < result["speedup_high"]
    assert cpu["spread_ratio"] == pytest.approx(1.1)
    assert cpu["unstable_rev2"] is False
    assert overlap["unstable_rev2"] is True  # 12 / 9 > 1.25


def test_p90_p10_spread_ignores_one_outlier_trial():
    # Linear percentiles of [1..10]: P10 = 1.9, P90 = 9.1.
    stats = compute_case_statistics([[float(m)] for m in range(1, 11)])
    assert stats["spread_p90_p10"] == pytest.approx(9.1 / 1.9)
    assert stats["stable"] is False

    medians = [1.0] * 23 + [3.0]
    stats = compute_case_statistics([[m] for m in medians])
    assert stats["spread_ratio"] == pytest.approx(3.0)
    assert stats["unstable_rev2"] is True
    assert stats["spread_p90_p10"] == pytest.approx(1.0)
    assert stats["stable"] is True


def test_claim_spread_thresholds_are_inclusive_at_exactly_1_25():
    p90_boundary = [1.0, 1.0, 1.05, 1.1, 1.15, 1.2, 1.22, 1.24, 1.25, 1.25]
    p90_exact = compute_case_statistics([[value] for value in p90_boundary])
    p90_over = compute_case_statistics([[value] for value in p90_boundary[:-2] + [1.2501, 1.2501]])
    assert p90_exact["spread_p90_p10"] == pytest.approx(1.25)
    assert p90_exact["stable"] is True
    assert p90_over["spread_p90_p10"] > 1.25
    assert p90_over["stable"] is False

    max_min_exact = compute_case_statistics([[1.0], [1.25]])
    max_min_over = compute_case_statistics([[1.0], [1.2501]])
    assert max_min_exact["spread_ratio"] == pytest.approx(1.25)
    assert max_min_exact["unstable_rev2"] is False
    assert max_min_over["spread_ratio"] > 1.25
    assert max_min_over["unstable_rev2"] is True


def test_verdict_boundaries_are_strict_and_invalid_medians_are_inconclusive():
    baseline = compute_case_statistics([[2.0], [3.0]])
    touches_faster_boundary = compute_case_statistics([[1.0], [2.0]])
    assert compare_speedup(baseline, touches_faster_boundary, "speedup")["verdict"] == (
        "inconclusive"
    )

    touches_slower_boundary = compute_case_statistics([[1.0], [2.0]])
    candidate = compute_case_statistics([[2.0], [3.0]])
    assert compare_speedup(touches_slower_boundary, candidate, "speedup")["verdict"] == (
        "inconclusive"
    )

    for invalid in (0.0, float("nan")):
        measured = compute_case_statistics([[invalid], [1.0]])
        comparison = compare_speedup(baseline, measured, "speedup")
        assert comparison["verdict"] == "inconclusive"
        assert comparison["speedup"] is None
        assert comparison["speedup_ci_low"] is None
        assert claim_support(comparison["verdict"], False, baseline, measured) == {
            "direction_supported": False,
            "magnitude_supported": False,
            "claim_supported_rev2": False,
        }


def test_direction_claim_survives_instability_but_magnitude_does_not():
    cpu = compute_case_statistics([[10.0], [10.2], [10.1], [10.3]])
    # Always faster than the comparator, but its trial medians spread 2x.
    noisy = compute_case_statistics([[1.0], [2.0], [1.0], [2.0]])
    comparison = compare_speedup(cpu, noisy, "speedup_vs_cpu")
    assert comparison["verdict"] == "faster"
    assert noisy["stable"] is False

    claims = claim_support(comparison["verdict"], False, cpu, noisy)
    assert claims == {
        "direction_supported": True,
        "magnitude_supported": False,
        "claim_supported_rev2": False,
    }
    # An unstable comparator also blocks the magnitude claim.
    steady = compute_case_statistics([[1.0], [1.01], [1.02], [1.0]])
    assert claim_support("faster", False, noisy, steady)["magnitude_supported"] is False
    assert claim_support("faster", False, cpu, steady) == {
        "direction_supported": True,
        "magnitude_supported": True,
        "claim_supported_rev2": True,
    }
    assert claim_support("faster", True, cpu, steady)["direction_supported"] is False
    assert claim_support("inconclusive", False, cpu, steady)["direction_supported"] is False


def test_bootstrap_ci_is_deterministic_and_brackets_the_point_estimate():
    cpu = [10.0, 10.4, 9.8, 10.1, 10.3, 9.9]
    cuda = [2.0, 2.2, 1.9, 2.1, 2.05, 1.95]
    first = bootstrap_speedup_ci(cpu, cuda)
    assert first == bootstrap_speedup_ci(cpu, cuda)
    low, high = first
    assert low < 10.05 / 2.025 < high
    assert bootstrap_speedup_ci(cpu, cuda, seed=BOOTSTRAP_SEED + 1) != first

    comparison = compare_speedup(
        compute_case_statistics([[m] for m in cpu]),
        compute_case_statistics([[m] for m in cuda]),
        "speedup_vs_cpu",
    )
    assert (comparison["speedup_ci_low"], comparison["speedup_ci_high"]) == first


def test_boundary_inversion_covers_every_resident_path():
    assert boundary_inversion({"resident": 1.0, "host-origin": 2.0}) is False
    assert boundary_inversion({"resident": 3.0, "host-origin": 2.0}) is True
    assert boundary_inversion({"resident": 1.0, "resident-graph": 2.5, "host-origin": 2.0}) is True
    assert boundary_inversion({"resident": 1.0}) is False


def test_boundary_inversion_nests_gpu_origin_between_resident_and_host_origin():
    assert boundary_inversion({"resident": 1.0, "gpu-origin": 1.5, "host-origin": 2.0}) is False
    assert boundary_inversion({"resident": 2.0, "gpu-origin": 1.5, "host-origin": 3.0}) is True
    assert boundary_inversion({"resident": 1.0, "gpu-origin": 2.5, "host-origin": 2.0}) is True
    # Each transfer policy is checked on its own group.
    medians = {
        "resident": 1.0,
        "host-origin": 3.0,
        "gpu-origin": 2.0,
        "host-origin-pinned": 1.8,
        "gpu-origin-pinned": 1.5,
    }
    assert boundary_inversion(medians) is False
    assert boundary_inversion(medians, "pinned") is False
    medians["gpu-origin-pinned"] = 0.5
    assert boundary_inversion(medians) is False
    assert boundary_inversion(medians, "pinned") is True


def test_boundary_inversion_vetoes_a_selected_partner_without_a_median():
    assert (
        boundary_inversion(
            {"resident": 1.0, "host-origin": 2.0},
            selected_keys={"resident", "resident-graph", "host-origin"},
        )
        is True
    )


def test_boundary_inversion_vetoes_a_selected_host_path_without_a_median():
    assert (
        boundary_inversion(
            {"resident": 1.0},
            selected_keys={"resident", "resident-graph", "host-origin"},
        )
        is True
    )


def test_case_group_uses_policy_matched_baselines_and_groups():
    paths = build_paths(["cpu", "cuda"], ["gpu-origin"], ["pageable", "pinned"])
    medians = {
        "cpu-comparator": 10.0,
        "cuda-resident": 1.0,
        "cuda-resident-graph": 0.8,
        "cuda-host-origin": 4.0,
        "cuda-host-origin-pinned": 3.0,
        "cpu-gpu-origin": 12.0,
        "cuda-gpu-origin": 2.0,
        "cpu-gpu-origin-pinned": 11.0,
        # Faster than resident: an inversion in the pinned group only.
        "cuda-gpu-origin-pinned": 0.5,
    }
    cases = [{"statistics": fake_stats(medians[path.label])} for path in paths]
    compare_case_group(cases, paths)
    stats = {path.label: case["statistics"] for path, case in zip(paths, cases, strict=True)}

    assert stats["cpu-comparator"]["verdict"] == "comparator"
    assert stats["cuda-host-origin"]["baseline"] == "cpu-comparator"
    assert stats["cpu-gpu-origin"]["baseline"] == "cpu-comparator"
    assert stats["cuda-gpu-origin"]["baseline"] == "cpu-gpu-origin"
    assert stats["cuda-gpu-origin-pinned"]["baseline"] == "cpu-gpu-origin-pinned"
    assert stats["cuda-gpu-origin"]["speedup_vs_cpu"] == 6.0
    # A CPU path that adds a download must not beat the comparator; it does not here.
    assert stats["cpu-gpu-origin"]["verdict"] == "slower"
    assert stats["cpu-gpu-origin"]["boundary_inversion"] is False

    for label in ("cuda-resident", "cuda-host-origin", "cuda-gpu-origin"):
        assert stats[label]["boundary_inversion"] is False
        assert stats[label]["direction_supported"] is True
    for label in ("cuda-host-origin-pinned", "cuda-gpu-origin-pinned"):
        assert stats[label]["boundary_inversion"] is True
        assert stats[label]["direction_supported"] is False

    vs_pageable = stats["cuda-host-origin-pinned"]["vs_pageable"]
    assert vs_pageable["speedup_vs_pageable"] == 4.0 / 3.0
    assert vs_pageable["verdict"] == "faster"
    assert vs_pageable["boundary_inversion"] is True
    assert vs_pageable["direction_supported"] is False
    assert "vs_pageable" not in stats["cuda-host-origin"]

    vs_comparator = stats["cuda-gpu-origin"]["vs_comparator"]
    assert vs_comparator["speedup_vs_comparator"] == 5.0
    assert vs_comparator["descriptive"] is True
    assert "direction_supported" not in vs_comparator
    assert stats["cuda-resident-graph"]["vs_resident"]["boundary_inversion"] is False


def test_cuda_claims_fail_closed_when_a_selected_veto_path_failed():
    paths = build_paths(["cpu", "cuda"])
    medians = {
        "cpu-comparator": 10.0,
        "cuda-resident": 1.0,
        "cuda-resident-graph": None,
        "cuda-host-origin": 4.0,
    }
    cases = [
        {"statistics": None if medians[path.label] is None else fake_stats(medians[path.label])}
        for path in paths
    ]

    compare_case_group(cases, paths)
    stats = {path.label: case["statistics"] for path, case in zip(paths, cases, strict=True)}

    host_origin = stats["cuda-host-origin"]
    resident = stats["cuda-resident"]
    assert host_origin is not None
    assert resident is not None
    assert host_origin["boundary_inversion"] is True
    assert host_origin["direction_supported"] is False
    assert resident["direction_supported"] is False


def test_cpu_gpu_origin_faster_than_the_comparator_is_an_inversion():
    paths = build_paths(["cpu", "cuda"], ["gpu-origin"], ["pageable"])
    medians = [10.0, 1.0, 0.8, 4.0, 9.0, 2.0]
    cases = [{"statistics": fake_stats(m)} for m in medians]
    compare_case_group(cases, paths)
    stats = {path.label: case["statistics"] for path, case in zip(paths, cases, strict=True)}
    assert stats["cpu-gpu-origin"]["boundary_inversion"] is True
    assert stats["cuda-gpu-origin"]["boundary_inversion"] is True
    assert stats["cuda-host-origin"]["boundary_inversion"] is False


def test_inverted_pageable_twin_vetoes_the_pinning_comparison():
    paths = build_paths(["cpu", "cuda"], ["gpu-origin"], ["pageable", "pinned"])
    medians = {
        "cpu-comparator": 10.0,
        "cuda-resident": 1.0,
        "cuda-resident-graph": 0.8,
        "cuda-host-origin": 4.0,
        "cuda-host-origin-pinned": 3.0,
        "cpu-gpu-origin": 12.0,
        # Faster than resident: an inversion in the pageable group only.
        "cuda-gpu-origin": 0.5,
        "cpu-gpu-origin-pinned": 11.0,
        "cuda-gpu-origin-pinned": 1.5,
    }
    cases = [{"statistics": fake_stats(medians[path.label])} for path in paths]
    compare_case_group(cases, paths)
    stats = {path.label: case["statistics"] for path, case in zip(paths, cases, strict=True)}
    assert stats["cuda-gpu-origin"]["boundary_inversion"] is True
    assert stats["cuda-gpu-origin-pinned"]["boundary_inversion"] is False
    assert stats["cuda-gpu-origin-pinned"]["direction_supported"] is True
    vs_pageable = stats["cuda-gpu-origin-pinned"]["vs_pageable"]
    assert vs_pageable["boundary_inversion"] is True
    assert vs_pageable["direction_supported"] is False


def test_cuda_resident_claims_fail_closed_when_host_origin_veto_failed():
    paths = build_paths(["cpu", "cuda"])
    medians = {
        "cpu-comparator": 10.0,
        "cuda-resident": 1.0,
        "cuda-resident-graph": 0.9,
        "cuda-host-origin": None,
    }
    cases = [
        {"statistics": None if medians[path.label] is None else fake_stats(medians[path.label])}
        for path in paths
    ]

    compare_case_group(cases, paths)
    stats = {path.label: case["statistics"] for path, case in zip(paths, cases, strict=True)}

    for label in ("cuda-resident", "cuda-resident-graph"):
        result = stats[label]
        assert result is not None
        assert result["boundary_inversion"] is True
        assert result["direction_supported"] is False
        assert result["magnitude_supported"] is False
