import itertools

import pytest
from _bench_test_support import READY_FACTS, ROOT

from stoquant.design import (
    DEFAULT_TRIALS,
    BenchPath,
    build_paths,
    trial_design,
    trial_orders,
    williams_rows,
)
from stoquant.matrix import run_benchmark_matrix


def test_trial_orders_cover_every_permutation_once():
    paths = build_paths(["cpu", "cuda"])
    orders = trial_orders(paths, DEFAULT_TRIALS)
    assert DEFAULT_TRIALS == 24
    assert len({tuple(order) for order in orders}) == 24
    for position in range(4):
        assert sorted(order[position] for order in orders) == sorted([0, 1, 2, 3] * 6)


def test_default_paths_are_the_revision_3_paths():
    assert [path.label for path in build_paths(["cpu", "cuda"])] == [
        "cpu-comparator",
        "cuda-resident",
        "cuda-resident-graph",
        "cuda-host-origin",
    ]
    assert [path.label for path in build_paths(["cpu"])] == ["cpu-comparator"]
    assert BenchPath("cuda", "host-origin", "pageable").extra_args == [
        "--boundary",
        "host-origin",
    ]
    assert BenchPath("cpu", "comparator", "none").extra_args == []


def test_extension_paths_follow_a_fixed_order():
    paths = build_paths(["cpu", "cuda"], ["gpu-origin"], ["pinned", "pageable"])
    assert [path.label for path in paths] == [
        "cpu-comparator",
        "cuda-resident",
        "cuda-resident-graph",
        "cuda-host-origin",
        "cuda-host-origin-pinned",
        "cpu-gpu-origin",
        "cuda-gpu-origin",
        "cpu-gpu-origin-pinned",
        "cuda-gpu-origin-pinned",
    ]
    assert paths[-1].extra_args == ["--boundary", "gpu-origin", "--transfer-policy", "pinned"]
    assert paths[5].extra_args == ["--boundary", "gpu-origin"]


@pytest.mark.parametrize(
    ("backends", "boundaries", "policies", "message"),
    [
        (["cpu", "cuda"], [], ["pinned"], "include 'pageable'"),
        (["cpu"], ["gpu-origin"], ["pageable"], "needs the 'cuda' backend"),
        (["cpu"], [], ["pageable", "pinned"], "needs the 'cuda' backend"),
        (["cpu", "cuda"], ["device-origin"], ["pageable"], "unknown extension boundaries"),
    ],
)
def test_bad_path_selections_are_rejected(backends, boundaries, policies, message):
    with pytest.raises(ValueError, match=message):
        build_paths(backends, boundaries, policies)


@pytest.mark.parametrize("n", [5, 6, 7, 8, 9])
def test_williams_rows_balance_position_and_predecessor(n):
    rows = williams_rows(n)
    assert len(rows) == (n if n % 2 == 0 else 2 * n)
    repeats = len(rows) // n
    for row in rows:
        assert sorted(row) == list(range(n))
    for position in range(n):
        assert sorted(row[position] for row in rows) == sorted(list(range(n)) * repeats)
    pairs: dict[tuple[int, int], int] = {}
    for row in rows:
        for before, after in itertools.pairwise(row):
            pairs[(before, after)] = pairs.get((before, after), 0) + 1
    assert len(pairs) == n * (n - 1)
    assert set(pairs.values()) == {repeats}


def test_trial_design_switches_to_williams_above_four_paths():
    paths = build_paths(["cpu", "cuda"], ["gpu-origin"], ["pageable", "pinned"])
    assert trial_design(4) == "all-permutations"
    assert trial_design(len(paths)) == "williams"
    orders = trial_orders(paths, 36)
    assert orders[:18] == williams_rows(9)
    assert orders[18:] == williams_rows(9)
    with pytest.raises(ValueError, match="multiple of 18"):
        trial_orders(paths, 24)


def test_extension_rejects_bad_trials_before_touching_disk(tmp_path):
    with pytest.raises(ValueError, match="multiple of 18"):
        run_benchmark_matrix(
            root=ROOT,
            output_dir=tmp_path / "out",
            counts=[1024],
            bit_widths=[8],
            backends=["cpu", "cuda"],
            trials=24,
            boundaries=["gpu-origin"],
            transfer_policies=["pageable", "pinned"],
            readiness_facts=READY_FACTS,
        )
    assert not (tmp_path / "out").exists()
