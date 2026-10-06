# CSC612M-MCO2

Public course implementation repository for the CSC612M-MCO2 data-level-parallelism project. The code, including its Python package, command and binary, is named `stoquant` ([ADR 0004](docs/adr/0004-stoquant-naming.md)).

## Status

The public technical contract, benchmark protocol, and course-deliverables guide define how to implement and measure this course project.
The CPU scalar and AVX2 quantizers, CUDA 8-bit and 4-bit quantizers, record codec, decoder, Philox generator, and NumPy oracle are implemented and verified on the RTX 5060.
The benchmark matrix with its paper extensions, the Layer 3 unbiasedness suite, and the K1 bandwidth baseline and A/B tools are implemented.
The publication evidence is the strict rerun at source revision `46c1294`: the dense, sparse and model matrices in `results/2026-10-05-46c1294-*-strict-rerun`, with a comparative review in `results/2026-10-05-46c1294-review`. The K1 A/B and unbiasedness runs at `5d0841b` complete it.
The earlier `ec31947` matrices stay in the tree as the same-code comparator that the review reads. Other superseded snapshots are archived in Git history ([ADR 0005](docs/adr/0005-snapshot-lifecycle.md)); test fixtures are under `tests/golden/`.

## Start here

- [Technical contract](docs/technical-contract.md)
- [Reproduction path](docs/reproduction.md)
- [Run notifications](docs/run-notifications.md)
- [Benchmark protocol](docs/benchmark-protocol.md)
- [Course deliverables](docs/course-deliverables.md)
- [Architecture ADR](docs/adr/0001-cuda-stochastic-quantization-architecture.md)

The implementation, build, and reproduction instructions live in this repository. Project planning and issue tracking live in the private paper repository.

## Build status

`just test-cpu` builds and verifies the CPU compression/decompression pipeline.
`just test-rng` builds the native C/CUDA executable and runs the RNG checks on CPU and GPU.
`just test-cuda` checks CUDA 8-bit and 4-bit parity, launch-geometry independence, determinism, and acceptance behavior on a CUDA device.
See [Reproduction path](docs/reproduction.md) for commands, dependencies, and current coverage.

## Quality gates

For every code change, run `just format`, `just lint`, and `just verify` before closeout.
For C or CUDA changes, also run `just test-rng`.
For CUDA quantizer or CLI changes, also run `just test-cuda`.
For CPU quantizer, codec, or CLI changes, run `just test-cpu`.
