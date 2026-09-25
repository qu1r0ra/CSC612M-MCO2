# CSC612M-MCO2

Public course implementation repository for the CSC612M MCO2 data-level-parallelism project.

## Status

The public technical contract, benchmark protocol, and course-deliverables guide define how to implement and measure this course project.
The CPU 8-bit and 4-bit quantizers, record codec, decoder, and NumPy oracle are implemented.
The C-host/CUDA toolchain, Philox generator, and CUDA 8-bit and 4-bit quantizers with launch-geometry independence are verified on the RTX 5060. Acceptance timings for 2^22 elements at 8-bit and 4-bit are recorded; the benchmark matrix remains for later work.

## Start here

- [Technical contract](docs/technical-contract.md)
- [Reproduction path](docs/reproduction.md)
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
