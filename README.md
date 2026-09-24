# CSC612M-MCO2

Public course implementation repository for the CSC612M MCO2 data-level-parallelism project.

## Status

The public technical contract, benchmark protocol, and course-deliverables guide define this repository's implementation-facing requirements.
The C-host/CUDA toolchain and the Philox generator are verified on the RTX 5060; the quantizer, codec, and measurements have not started.

## Start here

- [Technical contract](docs/technical-contract.md)
- [Reproduction path](docs/reproduction.md)
- [Benchmark protocol](docs/benchmark-protocol.md)
- [Course deliverables](docs/course-deliverables.md)
- [Architecture ADR](docs/adr/0001-cuda-stochastic-quantization-architecture.md)

This repository is independently understandable for implementation, build, and reproduction.
Project-wide planning and issue tracking are handled separately from this public implementation repository.

## Build status

`just test-rng` builds the native C/CUDA executable and runs the RNG checks on CPU and GPU.
See [Reproduction path](docs/reproduction.md) for the verified toolchain and what the build does not yet cover.

## Quality gates

For every code change, run `just format`, then `just lint`, and `just verify` before closeout.
For C or CUDA changes, also run `just test-rng`.
