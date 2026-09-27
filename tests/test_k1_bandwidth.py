import pytest

from k1_bandwidth import (
    k1_launch_count,
    k1_min_bytes,
    probe_ceiling_gbps,
    regime,
    summarize_cell,
    theoretical_peak_gbps,
)

DEVICE = {"memory_clock_khz": 14_001_000, "bus_width_bits": 128, "l2_bytes": 25_165_824}


def test_launch_count_counts_every_k1_kernel():
    # max, reduce-max, sum, one pair launch per tree level, finish.
    assert k1_launch_count(1) == 4
    assert k1_launch_count(256) == 4
    assert k1_launch_count(257) == 5
    assert k1_launch_count(1 << 22) == 18
    assert k1_launch_count(1 << 26) == 22
    assert k1_launch_count((1 << 16) + 1) == 4 + 9


def test_min_bytes_is_two_full_input_reads():
    assert k1_min_bytes(1 << 26) == 2 * 4 * (1 << 26)


def test_theoretical_peak_uses_the_ddr_convention():
    assert theoretical_peak_gbps(14_001_000, 128) == pytest.approx(448.032)


def test_regime_splits_at_the_l2_size():
    assert regime(1 << 22, DEVICE["l2_bytes"]) == "l2-resident"
    assert regime(1 << 23, DEVICE["l2_bytes"]) == "dram"


def test_probe_ceiling_takes_the_best_dram_size():
    probe = {
        "sizes": [
            {"count": 1 << 22, "bytes": 1 << 24, "best_gbps": 1200.0},
            {"count": 1 << 24, "bytes": 1 << 26, "best_gbps": 410.0},
            {"count": 1 << 26, "bytes": 1 << 28, "best_gbps": 425.0},
        ]
    }
    assert probe_ceiling_gbps(probe, DEVICE["l2_bytes"]) == 425.0
    with pytest.raises(ValueError):
        probe_ceiling_gbps({"sizes": probe["sizes"][:1]}, DEVICE["l2_bytes"])


def test_summarize_cell_reports_fractions_only_above_l2():
    trials = [[3.2, 3.3, 3.1], [3.2, 3.25, 3.15]]
    stage = {"k2_ms": [1.2, 1.3], "k3_ms": [0.9, 1.0]}
    big = summarize_cell(1 << 26, 8, trials, stage, DEVICE, 425.0)
    assert big["regime"] == "dram"
    assert big["launches"] == 22
    assert big["k1"]["median_ms"] == pytest.approx(3.2)
    assert big["effective_gbps"] == pytest.approx(2 * 4 * (1 << 26) / 3.2 / 1e6)
    assert big["fraction_of_peak"] == pytest.approx(big["effective_gbps"] / 448.032)
    assert big["fraction_of_ceiling"] == pytest.approx(big["effective_gbps"] / 425.0)
    assert big["ideal_ms_at_ceiling"] == pytest.approx(2 * 4 * (1 << 26) / 425.0 / 1e6)
    assert big["headroom"] == pytest.approx(3.2 / big["ideal_ms_at_ceiling"])
    assert big["k2_median_ms"] == pytest.approx(1.25)

    small = summarize_cell(1 << 22, 8, trials, stage, DEVICE, 425.0)
    assert small["regime"] == "l2-resident"
    assert small["fraction_of_peak"] is None
    assert small["fraction_of_ceiling"] is None
    assert small["headroom"] is None
    assert small["effective_gbps"] > 0
