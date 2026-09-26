import os
from pathlib import Path

import numpy as np
import pytest

from unbiasedness import analyse, expectation, plot_unbiasedness, run_suite

ROOT = Path(__file__).resolve().parents[1]
BINARY = ROOT / "build" / ("mco2.exe" if os.name == "nt" else "mco2")


def _exact_sums(x: np.ndarray, scale: np.float32, s: int, seeds: int):
    """Sums and sums of squares equal to their expectations over `seeds` draws."""
    e = expectation(x, scale, s)
    high = e["low"] + np.where(np.signbit(x), -1.0, 1.0) * e["delta"]
    squares = ((1.0 - e["q"]) * e["low"] ** 2 + e["q"] * high**2) * seeds
    return e, e["expected"] * seeds, squares


def test_analyse_passes_exact_expectation_and_flags_bias():
    x = np.asarray([0.0, -0.0, 2.0, -2.0, 0.3, -0.7, 1.1, 1.9], dtype=np.float32)
    scale, s, seeds = np.float32(2.0), 7, 4096
    e, sums, squares = _exact_sums(x, scale, s, seeds)

    summary, arrays = analyse(x, scale, s, sums, squares, seeds)
    assert summary["passed"] is True
    assert summary["deterministic_exact"] is True
    assert summary["deterministic_elements"] == 4
    assert summary["max_error_over_bound"] == pytest.approx(0.0, abs=1e-9)
    assert summary["variance_ratio_pooled"] == pytest.approx(1.0)
    assert summary["max_expectation_gap_steps"] < 1e-5
    assert arrays["stochastic"].sum() == 4

    biased = sums.copy()
    biased[4] += 6.0 * e["delta"][4] * np.sqrt(e["q"][4] * (1 - e["q"][4]) * seeds)
    summary, _ = analyse(x, scale, s, biased, squares, seeds)
    assert summary["passed"] is False
    assert summary["elements_over_bound"] == 1

    shifted = sums.copy()
    shifted[2] += 1e-6
    summary, _ = analyse(x, scale, s, shifted, squares, seeds)
    assert summary["deterministic_exact"] is False
    assert summary["passed"] is False


@pytest.mark.skipif(not BINARY.exists(), reason="build the executable first")
def test_run_suite_cpu_smoke_and_figure(tmp_path):
    results, plot_data = run_suite(BINARY, tmp_path / "work", seeds=32, backends=("cpu",))
    assert len(results) == 8
    assert {r["input"] for r in results} == {
        "course",
        "dense_n16384",
        "sparse_n16384",
        "model_layer1.0.conv1.weight",
    }
    for r in results:
        assert r["backends_identical"] is True
        assert r["deterministic_exact"] is True
        assert r["seeds"] == 32
    figure = tmp_path / "f_unbiasedness.png"
    plot_unbiasedness(plot_data, 32, figure)
    assert figure.stat().st_size > 0
