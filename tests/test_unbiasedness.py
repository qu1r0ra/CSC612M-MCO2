import argparse
import os

import numpy as np
import pytest

from stoquant import layout
from stoquant.inputs import DEFAULT_INPUT_SEED
from stoquant.oracle import SIGNED_LIMITS, compress_record_fp32, decode_record
from stoquant.unbiasedness import (
    BIT_WIDTHS,
    SUITE_SEEDS,
    analyse,
    expectation,
    non_evidence_reasons,
    plot_unbiasedness,
    run_suite,
)

ROOT = layout.ROOT
BINARY = ROOT / "build" / ("stoquant.exe" if os.name == "nt" else "stoquant")


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
    results, plot_data = run_suite(BINARY, tmp_path / "work", seeds=SUITE_SEEDS, backends=("cpu",))
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
        assert r["passed"] is True
        assert r["seeds"] == SUITE_SEEDS
    figure = tmp_path / "f_unbiasedness.png"
    plot_unbiasedness(plot_data, SUITE_SEEDS, figure)
    assert figure.stat().st_size > 0


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("target", ["sums", "squares"])
def test_analyse_fails_on_any_nonfinite_mean_or_sum(bad, target):
    x = np.asarray([0.0, 0.3, -0.7, 1.1], dtype=np.float32)
    scale, s, seeds = np.float32(2.0), 7, 4096
    _, sums, squares = _exact_sums(x, scale, s, seeds)
    (sums if target == "sums" else squares)[0] = bad
    summary, _ = analyse(x, scale, s, sums, squares, seeds)
    assert summary["passed"] is False
    assert summary["finite"] is False


@pytest.mark.parametrize("bits", BIT_WIDTHS)
def test_expectation_matches_the_oracle_rounding_threshold(bits):
    """The realised probability q flips the oracle's rounding exactly at word == q * 2^32."""
    s = SIGNED_LIMITS[bits]
    x = np.linspace(-1.9, 1.9, 61, dtype=np.float32)
    scale = np.float32(2.0)
    e = expectation(x, scale, s)
    threshold = (e["q"] * 2.0**32).astype(np.uint64)
    assert np.all(e["q"] * 2.0**32 == threshold)

    def decoded(words):
        record = compress_record_fp32(x, bits=bits, scale=float(scale), words=words)
        return decode_record(record).astype(np.float64)

    below = decoded(np.maximum(threshold, 1).astype(np.uint32) - 1)
    at = decoded(threshold.astype(np.uint32))
    high = e["low"] + np.where(np.signbit(x), -1.0, 1.0) * e["delta"]
    stochastic = e["q"] > 0
    np.testing.assert_allclose(at, e["low"], rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(below[stochastic], high[stochastic], rtol=1e-6, atol=1e-9)


@pytest.mark.parametrize(
    "changes",
    [
        {"seeds": 32},
        {"backends": ["cpu"]},
        {"input_seed": DEFAULT_INPUT_SEED + 1},
    ],
)
def test_any_non_default_design_clears_the_evidence_flag(changes):
    from stoquant.unbiasedness import BACKENDS

    values = {"seeds": SUITE_SEEDS, "backends": list(BACKENDS), "input_seed": DEFAULT_INPUT_SEED}
    assert non_evidence_reasons(argparse.Namespace(**values), False) == []
    assert non_evidence_reasons(argparse.Namespace(**{**values, **changes}), False)
    assert non_evidence_reasons(argparse.Namespace(**values), True)
