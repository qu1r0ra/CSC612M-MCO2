# CSC612M-MCO2

Public course implementation repository for the CSC612M MCO2 data-level-parallelism project.

## Status

The public technical contract, benchmark protocol, and course-deliverables guide define this repository's implementation-facing requirements.
The executable implementation and measurements have not started.

## Start here

- [Technical contract](docs/technical-contract.md)
- [Reproduction path](docs/reproduction.md)
- [Benchmark protocol](docs/benchmark-protocol.md)
- [Course deliverables](docs/course-deliverables.md)
- [Architecture ADR](docs/adr/0001-cuda-stochastic-quantization-architecture.md)

This repository is independently understandable for implementation, build, and reproduction.
Project-wide planning and issue tracking are handled separately from this public implementation repository.

## Build status

The native C++/CUDA build is not implemented by this scaffold.
When implementation begins, add the exact build and test commands to `docs/reproduction.md` before claiming reproducibility.

## Quality gates

For every code change, run `just format`, then `just lint`, and `just verify` before closeout.
Add the applicable compiler, test, and benchmark checks as those tools become available.
