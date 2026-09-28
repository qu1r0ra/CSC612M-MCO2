import re

import pytest

from stoquant import layout
from stoquant.k1_ab import (
    TREE_SPAN,
    VARIANTS,
    compare_arms,
    keep_decision,
    optimized_launch_count,
    plot_f5,
    summarize_ab_cell,
    variant_order,
)

DEVICE = {"memory_clock_khz": 14_001_000, "bus_width_bits": 128, "l2_bytes": 25_165_824}
SOURCE = layout.NATIVE_DIR / "quantizer_cuda.cu"


def test_tree_span_matches_the_cuda_source():
    text = SOURCE.read_text(encoding="utf-8")
    assert re.search(rf"#define MCO2_CUDA_K1_TREE_SPAN {TREE_SPAN}\b", text)


def test_optimized_launch_count_matches_the_protocol():
    # max, reduce-max, warp sums, one launch per 2048-to-1 tree level, finish.
    assert optimized_launch_count(1) == 4
    assert optimized_launch_count(256) == 4
    assert optimized_launch_count(257) == 5
    assert optimized_launch_count(256 * 2048) == 5
    assert optimized_launch_count(256 * 2048 + 1) == 6
    assert optimized_launch_count(1 << 25) == 6
    assert optimized_launch_count(1 << 26) == 6


def test_variant_order_alternates_so_each_arm_holds_each_slot_equally():
    orders = [variant_order(p) for p in range(12)]
    for slot in range(2):
        firsts = [order[slot] for order in orders]
        assert firsts.count("reference") == firsts.count("optimized") == 6
    assert all(sorted(order) == sorted(VARIANTS) for order in orders)


def trials(level, n=12, reps=5):
    return [[level * (1 + 0.001 * i)] * reps for i in range(n)]


def test_compare_arms_reports_a_faster_direction_claim():
    result = compare_arms(trials(2.0), trials(1.0))
    assert result["verdict"] == "faster"
    assert result["direction_supported"] and result["magnitude_supported"]
    assert result["speedup"] == pytest.approx(2.0)


def test_compare_arms_overlapping_ranges_are_inconclusive():
    result = compare_arms(trials(1.0), trials(1.005))
    assert result["verdict"] == "inconclusive"
    assert not result["direction_supported"]


def cell(count, verdict, direction, regime="dram"):
    return {
        "count": count,
        "bits": 8,
        "regime": regime,
        "k1_comparison": {"verdict": verdict, "direction_supported": direction},
    }


def test_keep_needs_every_dram_cell_supported_faster():
    kept = keep_decision(
        [cell(1 << 25, "faster", True), cell(1 << 10, "slower", True, "l2-resident")]
    )
    assert kept["kept"] and kept["dram_cells"] == 1
    assert kept["descriptive_slowdowns"] == [{"count": 1 << 10, "bits": 8, "regime": "l2-resident"}]
    rejected = keep_decision([cell(1 << 25, "faster", True), cell(1 << 26, "inconclusive", False)])
    assert not rejected["kept"]
    assert rejected["failing_cells"] == [{"count": 1 << 26, "bits": 8, "verdict": "inconclusive"}]


def test_keep_is_false_without_dram_cells():
    assert not keep_decision([cell(1 << 10, "faster", True, "l2-resident")])["kept"]


def rows(count, bits):
    out = []
    for process in range(4):
        for variant, k1 in (("reference", 2.0), ("optimized", 1.0)):
            out.append(
                {
                    "process": process,
                    "variant": variant,
                    "count": count,
                    "bits": bits,
                    "samples_ms": [k1 + 1.5 + 0.01 * process] * 3,
                    "k1_ms": [k1 + 0.01 * process] * 3,
                    "k2_ms": [0.5] * 3,
                    "k3_ms": [0.5] * 3,
                }
            )
    return out


def test_summarize_ab_cell_and_f5_render(tmp_path):
    count = 1 << 25
    summary_cell = summarize_ab_cell(count, 8, rows(count, 8), DEVICE, 430.0)
    assert summary_cell["regime"] == "dram"
    assert summary_cell["arms"]["reference"]["launches"] == 21
    assert summary_cell["arms"]["optimized"]["launches"] == 6
    assert summary_cell["k1_comparison"]["verdict"] == "faster"
    stages = summary_cell["arms"]["optimized"]["stage_medians_ms"]
    assert stages["k1_ms"] == pytest.approx(1.015)
    assert stages["other_ms"] == pytest.approx(0.5)
    out = tmp_path / "f5.png"
    plot_f5({"cells": [summary_cell]}, out)
    assert out.stat().st_size > 0
