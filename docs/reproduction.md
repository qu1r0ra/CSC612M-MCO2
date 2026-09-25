# Reproduction path

Status: The CPU 8-bit pipeline and Philox checks are implemented. CUDA quantization and benchmark work remain.

This document is the public entry point for reproducing the course implementation.

## What exists

The repository builds a CPU command-line tool, `build/mco2`, and a CUDA RNG check, `build/test_rng`.
The CPU tool compresses raw little-endian FP32 vectors into version-1 8-bit records. It decodes each record to raw FP32.
The NumPy oracle independently implements Philox4x32-10 and the CPU quantization path.
CUDA quantization and benchmark work need their own tests and measurements.

## Verified toolchain

Verified on 2026-09-25:

| Component | Version |
| --- | --- |
| GPU | NVIDIA GeForce RTX 5060, compute capability 12.0 (`sm_120`), driver 610.88 |
| CUDA toolkit | 13.4 (`nvcc` V13.4.59) |
| C compiler | MSVC `cl` 19.51.36260 for x64 (Visual Studio Community 2026 18.10, toolset 14.51) |
| RNG dependency | Random123 v1.14.0, commit `726a093`, vendored in `third_party/random123` |
| CUDA build flags | `-O2 -arch=sm_120 -Isrc -Ithird_party/random123/include -Xcompiler /wd4068` |
| CPU build flags | `/O2 /W4 /std:c11 /fp:strict` |

`nvcc --list-gpu-arch` includes `compute_120`, and `nvcc` accepted MSVC 19.51 without `-allow-unsupported-compiler`.
The `/wd4068` flag silences MSVC warnings about CUDA-only pragmas in the vendored Random123 headers.

## Build and RNG check

Requires `just`, the CUDA toolkit, and an NVIDIA GPU.
On Windows, also install Visual Studio with the x64 C++ tools; `scripts/with-msvc.ps1` enters its developer environment, so no developer prompt is needed.

```powershell
just toolchain
$env:CUDA_ARCH = 'sm_120'
just test-rng
```

`CUDA_ARCH` defaults to `native`, which targets the GPU in the build machine.
Set it explicitly when recording a result.
`just test-rng` builds `build/test_rng`, prints the detected GPU, and exits nonzero on any failure.
It checks:

- Upstream Philox4x32-10 known-answer vectors on CPU and GPU.
- The seed, group, tensor, and invocation mapping from the technical contract, including 64-bit group splitting.
- Rejection of tensor and invocation identifiers above `2^32-1`.
- Bitwise CPU/GPU agreement for 1,000,003 elements under six block and grid geometries, including the capped automatic grid.
- The Bernoulli threshold at `p=0`, `p=1`, `p=1-2^-24`, and `p=0.5` on CPU and GPU.

## CPU pipeline

Requires `just`, Python 3.11 or newer, `uv`, and a C compiler. On Windows, `scripts/with-msvc.ps1` enters the x64 MSVC environment.

```powershell
just test-cpu
```

`just test-cpu` builds the native CPU CLI and two C test executables, then runs the NumPy/pytest oracle and CLI suite. The first run resolves NumPy and pytest from the checked-in `pyproject.toml` and `uv.lock`.

The CLI consumes and produces raw little-endian FP32 files. A seeded compression computes the FP32 L2 scale and uses the logical Philox stream:

```powershell
just build-cpu
build\mco2.exe compress --input input.f32 --output tensor.msq --seed 123 --tensor-id 7 --invocation-id 9
build\mco2.exe decompress --input tensor.msq --output reconstructed.f32
```

For layer-1 checks, pass a prescribed FP32 scale and a raw little-endian uint32 word file containing one word per input element:

```powershell
build\mco2.exe compress --input input.f32 --output tensor.msq --seed 123 --scale 2.0 --words prescribed-words.u32
```

The CLI requires `--seed` for every compression command. `--words` supplies the random decisions for prescribed-scale tests. `--bits` accepts only `8` in this slice. The exact header, pairwise FP32 reduction, and layer-2 bounds are in [the technical contract](technical-contract.md).

The suite checks prescribed-word records, seeded CPU/oracle byte equality, edge inputs, malformed headers, and both FP32 bounds against the FP64 oracle.

On Linux, including Google Colab, `build-rng` and `test-rng` call `nvcc` with the system host compiler.
The Linux CUDA path has not been run; record the Colab GPU model and host compiler with its first run.

## Local quality preflight

After changing Python reference, analysis, or tooling code, run `just format` and then `just lint`.
Before declaring a code change ready, run `just verify`, which includes Ruff linting and a formatting check.

## Documentation still required

Before claiming reproducibility of the full pipeline, add the following verified commands and outputs:

1. Google Colab build and RNG check, with the Colab GPU model.
2. CUDA-vs-CPU codec comparisons and decoder malformed-input cases beyond the CPU suite.
3. Benchmark command with matrix, warmup, repetition, and timing-boundary configuration.
4. Result-inspection command that verifies manifests, raw samples, byte counts, and correctness status.

The implementation must record the exact compiler, CUDA toolkit, dependency revision, build flags, hardware, and transfer policy.
Preserve verified results and provenance in a frozen snapshot for downstream analysis.

## Scope boundary

The course implementation does not require full federated training, network transport, or compatibility with other wire formats.
Host-ready means packed bytes in host memory.
Remote GPU experiment execution remains author-run and is outside this repository scaffold.
