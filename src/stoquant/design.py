"""Experimental design: bench paths, path sets, Williams rows, trial and case ordering."""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

DEFAULT_COUNTS = tuple(1 << exponent for exponent in range(10, 27))
DEFAULT_BITS = (4, 8)


@dataclass(frozen=True)
class SweepProfile:
    """Named matrix settings applied by the command-line interface."""

    name: str
    counts: tuple[int, ...]
    bit_widths: tuple[int, ...]


PILOT_PROFILE = SweepProfile(
    name="pilot",
    counts=(1 << 10, 1 << 14, 1 << 20, 1 << 26),
    bit_widths=DEFAULT_BITS,
)

DEFAULT_CASE_ORDER_SEED = 612

# Twenty-four trials run every ordering of the four paths once, balancing both the
# position of each path and the path that precedes it.
DEFAULT_TRIALS = 24
# Publication matrix extension (issue #20): opt-in boundaries and transfer policies.
# The four revision 3 paths stay the default; the extension adds paths, never changes them.
EXTENSION_BOUNDARIES = ("gpu-origin",)
TRANSFER_POLICIES = ("pageable", "pinned")
# All orderings stay affordable up to four paths; beyond that a Williams design
# balances position and immediate predecessor in n (even) or 2n (odd) rows.
MAX_PERMUTATION_PATHS = 4
# Issue #22: the opt-in multithreaded AVX2 comparator. It is timed on the same
# host-host boundary as the scalar comparator, which stays every claim's baseline.
HOST_BOUNDARIES = ("comparator", "optimized")
K1_VARIANTS = ("reference", "optimized")


@dataclass(frozen=True)
class BenchPath:
    """One timed path: a backend, its timing boundary, and its host transfer policy.

    The comparator is the CPU host-host path, and "optimized" is the AVX2 CPU path on
    the same boundary. Resident paths move no host data, so their policy is "none";
    transfer boundaries carry "pageable" or "pinned".
    """

    backend: str
    boundary: str
    policy: str

    @property
    def key(self) -> str:
        return f"{self.boundary}-pinned" if self.policy == "pinned" else self.boundary

    @property
    def label(self) -> str:
        return f"{self.backend}-{self.key}"

    @property
    def extra_args(self) -> list[str]:
        args = [] if self.boundary in HOST_BOUNDARIES else ["--boundary", self.boundary]
        if self.policy == "pinned":
            args += ["--transfer-policy", "pinned"]
        return args

    @property
    def baseline(self) -> BenchPath | None:
        """The policy-matched path used for claims against this path."""
        if self == COMPARATOR:
            return None
        if self.backend == "cuda" and self.boundary == "gpu-origin":
            return BenchPath("cpu", "gpu-origin", self.policy)
        return COMPARATOR

    @property
    def pageable_twin(self) -> BenchPath:
        return BenchPath(self.backend, self.boundary, "pageable")

    @property
    def nested_paths(self) -> tuple[BenchPath, ...]:
        """Paths whose measured work is contained in this path's work."""
        if self.backend == "cpu" and self.boundary == CPU_GPU_ORIGIN.boundary:
            return (COMPARATOR,)
        if self.backend == "cuda" and self.boundary == CUDA_GPU_ORIGIN.boundary:
            return (CUDA_RESIDENT,)
        if self.backend == "cuda" and self.boundary == CUDA_HOST_ORIGIN.boundary:
            return (
                CUDA_RESIDENT,
                CUDA_RESIDENT_GRAPH,
                BenchPath("cuda", CUDA_GPU_ORIGIN.boundary, self.policy),
            )
        return ()


COMPARATOR = BenchPath("cpu", "comparator", "none")
AVX2_COMPARATOR = BenchPath("cpu-avx2", "optimized", "none")
CUDA_RESIDENT = BenchPath("cuda", "resident", "none")
CUDA_RESIDENT_GRAPH = BenchPath("cuda", "resident-graph", "none")
CUDA_HOST_ORIGIN = BenchPath("cuda", "host-origin", "pageable")
CUDA_HOST_ORIGIN_PINNED = BenchPath("cuda", "host-origin", "pinned")
CPU_GPU_ORIGIN = BenchPath("cpu", "gpu-origin", "pageable")
CUDA_GPU_ORIGIN = BenchPath("cuda", "gpu-origin", "pageable")
CPU_GPU_ORIGIN_PINNED = BenchPath("cpu", "gpu-origin", "pinned")
CUDA_GPU_ORIGIN_PINNED = BenchPath("cuda", "gpu-origin", "pinned")

REPORT_PATHS = (
    COMPARATOR,
    AVX2_COMPARATOR,
    CUDA_RESIDENT,
    CUDA_RESIDENT_GRAPH,
    CUDA_HOST_ORIGIN,
)
PUBLICATION_PATHS = (
    *REPORT_PATHS,
    CUDA_HOST_ORIGIN_PINNED,
    CPU_GPU_ORIGIN,
    CUDA_GPU_ORIGIN,
    CPU_GPU_ORIGIN_PINNED,
    CUDA_GPU_ORIGIN_PINNED,
)
REPORT_LABELS = (
    "C comparator",
    "AVX2 comparator (descriptive)",
    "CUDA resident",
    "CUDA resident-graph",
    "CUDA host-origin",
)
PUBLICATION_LABELS = (
    "C comparator",
    "AVX2 comparator (descriptive)",
    "CUDA resident",
    "CUDA resident-graph",
    "CUDA host-origin pageable",
    "CUDA host-origin pinned",
    "CPU GPU-origin pageable",
    "CUDA GPU-origin pageable",
    "CPU GPU-origin pinned",
    "CUDA GPU-origin pinned",
)
PINNED_PAIRS = (
    (CUDA_HOST_ORIGIN, CUDA_HOST_ORIGIN_PINNED),
    (CPU_GPU_ORIGIN, CPU_GPU_ORIGIN_PINNED),
    (CUDA_GPU_ORIGIN, CUDA_GPU_ORIGIN_PINNED),
)


def build_paths(
    backends: Sequence[str],
    boundaries: Sequence[str] = (),
    transfer_policies: Sequence[str] = ("pageable",),
) -> list[BenchPath]:
    """The revision 3 paths, then the selected extension paths in a fixed order."""
    unknown = sorted(set(boundaries) - set(EXTENSION_BOUNDARIES))
    if unknown:
        raise ValueError(f"unknown extension boundaries: {unknown}")
    unknown = sorted(set(transfer_policies) - set(TRANSFER_POLICIES))
    if unknown:
        raise ValueError(f"unknown transfer policies: {unknown}")
    if "pageable" not in transfer_policies:
        raise ValueError("pinned paths are compared with their pageable twins; include 'pageable'")
    extended = bool(boundaries) or "pinned" in transfer_policies
    if extended and "cuda" not in backends:
        raise ValueError("the publication extension needs the 'cuda' backend")
    policies = [policy for policy in TRANSFER_POLICIES if policy in transfer_policies]

    paths = [COMPARATOR]
    if "cpu-avx2" in backends:
        paths.append(AVX2_COMPARATOR)
    if "cuda" in backends:
        paths += [CUDA_RESIDENT, CUDA_RESIDENT_GRAPH, CUDA_HOST_ORIGIN]
        if "pinned" in policies:
            paths.append(CUDA_HOST_ORIGIN_PINNED)
    if "gpu-origin" in boundaries:
        for policy in policies:
            paths += [
                BenchPath("cpu", "gpu-origin", policy),
                BenchPath("cuda", "gpu-origin", policy),
            ]
    return paths


def trial_design(path_count: int) -> str:
    return "all-permutations" if path_count <= MAX_PERMUTATION_PATHS else "williams"


def williams_rows(n: int) -> list[list[int]]:
    """Williams balanced Latin square: each path once per position and after every other."""
    first = [0]
    low, high = 1, n - 1
    while len(first) < n:
        first.append(low)
        low += 1
        if len(first) < n:
            first.append(high)
            high -= 1
    rows = [[(x + i) % n for x in first] for i in range(n)]
    if n % 2 == 1:
        rows += [list(reversed(row)) for row in rows]
    return rows


def case_order(keys: Sequence[Any], bit_widths: Sequence[int], seed: int) -> list[tuple[Any, int]]:
    """Every (input, bits) case in a seeded random order; dense inputs are keyed by count."""
    cases = [(key, bits) for key in keys for bits in bit_widths]
    permutation = np.random.default_rng(seed).permutation(len(cases))
    return [cases[i] for i in permutation]


def trial_orders(paths: Sequence[Any], trials: int) -> list[list[int]]:
    """Trial path orders for the design that fits the number of paths.

    Up to four paths, cycle through every ordering in a fixed lexicographic sequence.
    Beyond that, repeat the Williams rows; trials must be a whole number of designs.
    """
    if trial_design(len(paths)) == "all-permutations":
        perms = [list(p) for p in itertools.permutations(range(len(paths)))]
        return [perms[t % len(perms)] for t in range(trials)]
    rows = williams_rows(len(paths))
    if trials % len(rows) != 0:
        raise ValueError(
            f"{len(paths)} paths use a Williams design of {len(rows)} orders; "
            f"trials must be a multiple of {len(rows)}, not {trials}"
        )
    return [rows[t % len(rows)] for t in range(trials)]
