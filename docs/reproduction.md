# Reproduction path

Status: implementation pending.

This document is the public entry point for reproducing the course implementation.

## Before implementation exists

The repository currently contains the accepted contract and architecture decision but not the native C++/CUDA executable.
Do not interpret a successful documentation check as a successful CUDA build or benchmark.

## Local quality preflight

After changing Python reference, analysis, or tooling code, run `just format` and then `just lint`.
Before declaring a code change ready, run `just verify`, which includes Ruff linting and a formatting check.

## Required implementation documentation

Before claiming reproducibility, add the following verified commands and outputs:

1. Toolchain and GPU inventory commands.
2. Native CPU and CUDA build commands.
3. RNG known-answer checks on CPU and GPU.
4. Python oracle, codec, decoder, and malformed-input test commands.
5. Benchmark command with matrix, warmup, repetition, and timing-boundary configuration.
6. Result-inspection command that verifies manifests, raw samples, byte counts, and correctness status.

The implementation must record the exact compiler, CUDA toolkit, dependency revision, build flags, hardware, and transfer policy.
Preserve verified results and provenance in a frozen snapshot for downstream analysis.

## Scope boundary

The course implementation does not require full federated training, network transport, or compatibility with other wire formats.
Host-ready means packed bytes in host memory.
Remote GPU experiment execution remains author-run and is outside this repository scaffold.
