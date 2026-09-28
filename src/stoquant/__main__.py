"""Command-line entry point: `python -m stoquant <command>`.

Recipes call it through `python -m` rather than the `stoquant` console
launcher, because the readiness check treats any running process named
stoquant as a competing benchmark.
"""

from __future__ import annotations

import argparse
import importlib
import sys

COMMANDS = {
    "figures": ("stoquant.report", "Render snapshot figures and the report"),
    "bench-matrix": ("stoquant.driver", "Run the benchmark matrix"),
    "unbiasedness": ("stoquant.unbiasedness", "Run the Layer 3 expectation suite"),
    "k1-baseline": ("stoquant.k1_bandwidth", "Measure K1 against device bandwidth"),
    "k1-ab": ("stoquant.k1_ab", "A/B the reference and optimized K1"),
    "vec-report": ("stoquant.vectorization", "Check AVX2 hot-loop vectorization"),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="stoquant", description="Stochastic-rounding quantization benchmarks"
    )
    sub = parser.add_subparsers(dest="command", metavar="command", required=True)
    for name, (_, summary) in COMMANDS.items():
        sub.add_parser(name, help=summary, add_help=False)
    args, rest = parser.parse_known_args(argv)
    module = importlib.import_module(COMMANDS[args.command][0])
    result = module.main(rest, prog=f"stoquant {args.command}")
    return int(result or 0)


if __name__ == "__main__":
    sys.exit(main())
