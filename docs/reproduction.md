# Reproduction path

Status: toolchain and RNG verified; quantizer, codec, and benchmark pending.

This document is the public entry point for reproducing the course implementation.

## What exists

The repository builds one native executable, `build/test_rng`: a C host driver and C CPU path, calling CUDA kernels through `extern "C"` launch functions.
It checks the Philox generator and its logical mapping on CPU and GPU.
The quantizer, codec, decoder, and benchmark are not implemented yet; a passing RNG check is not evidence for them.

## Verified toolchain

Verified on 2026-09-25:

| Component | Version |
| --- | --- |
| GPU | NVIDIA GeForce RTX 5060, compute capability 12.0 (`sm_120`), driver 610.88 |
| CUDA toolkit | 13.4 (`nvcc` V13.4.59) |
| Host compiler used by `nvcc` | MSVC `cl` 19.51.36260 for x64 (Visual Studio Community 2026 18.10, toolset 14.51) |
| RNG dependency | Random123 v1.14.0, commit `726a093`, vendored in `third_party/random123` |
| Build flags | `-O2 -arch=sm_120 -Isrc -Ithird_party/random123/include -Xcompiler /wd4068` |

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

On Linux, including Google Colab, the `[unix]` recipes call `nvcc` directly with the system host compiler.
This path has not yet been run; record the Colab GPU model and host compiler with its first run.

## Local quality preflight

After changing Python reference, analysis, or tooling code, run `just format` and then `just lint`.
Before declaring a code change ready, run `just verify`, which includes Ruff linting and a formatting check.

## Documentation still required

Before claiming reproducibility of the full pipeline, add the following verified commands and outputs:

1. Google Colab build and RNG check, with the Colab GPU model.
2. Python oracle, codec, decoder, and malformed-input test commands.
3. Benchmark command with matrix, warmup, repetition, and timing-boundary configuration.
4. Result-inspection command that verifies manifests, raw samples, byte counts, and correctness status.

The implementation must record the exact compiler, CUDA toolkit, dependency revision, build flags, hardware, and transfer policy.
Preserve verified results and provenance in a frozen snapshot for downstream analysis.

## Scope boundary

The course implementation does not require full federated training, network transport, or compatibility with other wire formats.
Host-ready means packed bytes in host memory.
Remote GPU experiment execution remains author-run and is outside this repository scaffold.
