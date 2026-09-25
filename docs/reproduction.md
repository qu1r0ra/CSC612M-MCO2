# Reproduction path

Status: The CPU 8-bit and 4-bit pipelines, full decoder validation, and Layer 3 unbiasedness checks are implemented. CUDA quantization and benchmark work remain.

This document is the public entry point for reproducing the course implementation.

## What exists

The repository builds a CPU command-line tool, `build/mco2`, and a CUDA RNG check, `build/test_rng`.
The CPU tool compresses raw little-endian FP32 vectors into version-1 8-bit or 4-bit records. It decodes each record to raw FP32.
The NumPy oracle independently implements Philox4x32-10 and the CPU quantization path at both 8-bit and 4-bit widths.
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

# 8-bit compression (default) and decompression
build\mco2.exe compress --input input.f32 --output tensor_q8.msq --bits 8 --seed 123 --tensor-id 7 --invocation-id 9
build\mco2.exe decompress --input tensor_q8.msq --output reconstructed_q8.f32

# 4-bit compression and decompression
build\mco2.exe compress --input input.f32 --output tensor_q4.msq --bits 4 --seed 123 --tensor-id 7 --invocation-id 9
build\mco2.exe decompress --input tensor_q4.msq --output reconstructed_q4.f32
```

For layer-1 checks, pass a prescribed FP32 scale and a raw little-endian uint32 word file containing one word per input element:

```powershell
build\mco2.exe compress --input input.f32 --output tensor_q8.msq --bits 8 --seed 123 --scale 2.0 --words prescribed-words.u32
build\mco2.exe compress --input input.f32 --output tensor_q4.msq --bits 4 --seed 123 --scale 2.0 --words prescribed-words.u32
```

The CLI requires `--seed` for every compression command. `--words` supplies the random decisions for prescribed-scale tests. `--bits` accepts `4` or `8` (defaulting to `8`). The exact header layout, 4-bit nibble packing, pairwise FP32 reduction, and layer-2 bounds are in [the technical contract](technical-contract.md).

### Verified test commands

Run the full CPU verification suite or run specific components directly:

1. **Full CPU suite**:
   ```powershell
   just test-cpu
   ```
   Builds `build/mco2`, `build/test_codec`, and `build/test_quantizer`, executes the native C test binaries, and runs the pytest test suites.

2. **C unit tests (codec and quantizer)**:
   ```powershell
   just build-cpu
   .\build\test_codec.exe
   .\build\test_quantizer.exe
   ```
   - `test_codec.exe` verifies 4-bit and 8-bit header encoding, 4-bit nibble decoding, odd-length records, and every decoder rejection branch (magic mismatch, unsupported version, unsupported bit width, nonzero reserved bytes, nonfinite/negative scale, payload length mismatch, nonzero padding nibble, out-of-range codes, and malformed zero-scale records).
   - `test_quantizer.exe` verifies FP32 L2 norm reduction, 4-bit and 8-bit quantization against prescribed words, signed zero (+0 decode), large magnitudes without intermediate overflow via max-rescaling, saturation boundaries, odd lengths (1, 3, 5, 257), and the 4,096-seed C Layer 3 unbiasedness loop.

3. **NumPy oracle tests**:
   ```powershell
   uv run pytest tests/test_oracle.py
   ```
   Verifies Python Philox4x32-10, 4-bit and 8-bit FP32 quantization against prescribed words, FP64 reference bounds, record serialization, and decoder error checking.

4. **CLI, decoder rejection, edge case, and Layer 3 tests**:
   ```powershell
   uv run pytest tests/test_cli.py
   ```
   To run specific subsets:
   ```powershell
   # Decoder rejection suite (all malformed input checks)
   uv run pytest tests/test_cli.py -k test_decompress_rejects

   # FP32 L2 scale overflow test
   uv run pytest tests/test_cli.py -k test_compress_fp32_norm_overflow_rejects

   # Edge cases (signed zero, large magnitudes, saturation, non-multiple & odd lengths)
   uv run pytest tests/test_cli.py -k "test_compress_edge_cases or test_compress_signed_zero"

   # Layer 3 empirical unbiasedness across 4,096 seeds
   uv run pytest tests/test_cli.py -k test_layer3_unbiasedness
   ```

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
