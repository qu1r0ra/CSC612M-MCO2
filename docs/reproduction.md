# Reproduction path

Status: The CPU 8-bit and 4-bit pipelines, the CUDA 8-bit and 4-bit pipelines, launch-geometry independence, full decoder validation, Layer 3 unbiasedness checks, and the full course benchmark matrix snapshot are implemented.

This document is the public entry point for reproducing the course implementation.

## What exists

The repository builds the CPU command-line tool `build/mco2`, the CUDA-enabled command-line tool `build/mco2`, and the CUDA RNG check `build/test_rng`.
The CPU tool compresses raw little-endian FP32 vectors into version-1 8-bit or 4-bit records. It decodes each record to raw FP32.
The NumPy oracle independently implements Philox4x32-10 and the CPU quantization path at both 8-bit and 4-bit widths.
The CUDA tool supports 8-bit and 4-bit compression and shares the CPU decoder. `just test-cuda` checks byte parity, prescribed-scale parity, the FP64 reconstruction bound, launch-geometry independence, determinism, invalid inputs, and timing output on a local CUDA device.

## Verified toolchain

Verified on 2026-09-25:

| Component | Version |
| --- | --- |
| GPU | NVIDIA GeForce RTX 5060, compute capability 12.0 (`sm_120`), driver 610.88 |
| CUDA toolkit | 13.4 (`nvcc` V13.4.59) |
| C compiler | MSVC `cl` 19.51.36260 for x64 (Visual Studio Community 2026 18.10, toolset 14.51) |
| RNG dependency | Random123 v1.14.0, commit `726a093`, vendored in `third_party/random123` |
| CUDA RNG build flags | `-O2 -arch=sm_120 -Isrc -Ithird_party/random123/include -Xcompiler /wd4068` |
| CUDA quantizer build flags | `-O2 -arch=native -Isrc -Ithird_party/random123/include --fmad=false --ftz=false --prec-div=true --prec-sqrt=true -Xcompiler /wd4068` |
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
$env:CUDA_ARCH = 'native'
just test-cuda
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

`just build-cuda` compiles the C host sources as C and the CUDA kernels with precise FP32 division and square root, flush-to-zero disabled, and fused multiply-add disabled. CUDA quantization accepts 8-bit and 4-bit records. Run `just test-cuda` to build that executable and exercise its acceptance suite.

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

The CUDA backend is selected with `--backend cuda`; it supports both 8-bit and 4-bit records (`--bits 8|4`). Optional `--block-size` and `--grid-size` configure execution geometry. Add `--timings` to emit a single JSON line to stderr with K1, K2, K3 device event times and host measured H2D/D2H copy times. For acceptance timing, use an input of `2^22` FP32 elements and record the output line with the GPU and toolkit from `just toolchain`:

```powershell
# 8-bit CUDA compression with timings
build\mco2.exe compress --input input-2m22.f32 --output tensor-q8.msq --bits 8 --seed 123 --backend cuda --timings

# 4-bit CUDA compression with timings
build\mco2.exe compress --input input-2m22.f32 --output tensor-q4.msq --bits 4 --seed 123 --backend cuda --timings
```

Recorded acceptance runs on 2026-09-25: the input was generated with NumPy `default_rng(2026).normal(size=2^22).astype(float32)`. On the RTX 5060 with driver 610.88, compute capability 12.0, CUDA 13.4 V13.4.59, and MSVC 19.51.36260:
- 8-bit: `{"k1_ms":1.181216,"k2_ms":0.170496,"k3_ms":0.135264,"h2d_ms":3.087400,"d2h_ms":1.346900}` (CUDA and CPU records byte-identical at 4,194,324 bytes).
- 4-bit: `{"k1_ms":1.091520,"k2_ms":0.141280,"k3_ms":0.069504,"h2d_ms":3.094700,"d2h_ms":1.172900}` (CUDA and CPU records byte-identical at 2,097,172 bytes).
This is a local acceptance run, not the benchmark matrix.

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

## Benchmark matrix and frozen snapshot

The benchmark driver measures in-process throughput across the course matrix ($2^{10}, 2^{14}, 2^{18}, 2^{22}$ element counts $\times$ 4 and 8 bits $\times$ C comparator, CUDA resident, CUDA host-origin) using `mco2 bench`. Each matrix cell is gated on Layer 2 correctness and CPU/CUDA byte parity before timing, then every path runs as a separate process in each of six trials with balanced path order. The driver refuses a dirty working tree; commit first, or pass `--allow-dirty` for a non-evidence run.

### Running the benchmark matrix

```powershell
# Build CUDA executable and run the full 24-case course benchmark matrix
just bench-matrix

# Alternatively, run with custom repetition or output directory
uv run python benchmark_driver.py --counts 1024 16384 262144 4194304 --bits 4 8 --warmup 10 --reps 30 --trials 6
```

### Inspecting results and provenance

Snapshots are stored in `results/<date>-<short_rev>/`:
- `manifest.json`: Run-level provenance including git revision and dirty state, hardware, the build commands from `just --dry-run build-cuda`, CUDA toolkit, driver, transfer policy (`pageable`), GPU state before and after the run, trial orders, the statistics method and claim rule, and input hashes.
- `summary.csv`: Per-case pooled median and IQR, trial-median range and spread ratio, speedup vs the C comparator with its range and verdict, and the `boundary_inversion`, `unstable`, and `claim_supported` flags.
- `case_*.json`: Per-trial raw samples (`trial_runs[].samples_ms`, `k1_ms`, `k2_ms`, `k3_ms`) with each trial's path order and invocation identifiers, pooled samples, statistics, configuration, and correctness validation results.
- `msvc_vectorization_report.txt`: MSVC `/Qvec-report:2` diagnostics from recompiling the comparator sources with the exact benchmarked host flags; the command is at the top of the file.

```powershell
# View summary table of latest snapshot
Get-Content (Get-ChildItem results -Directory | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName\summary.csv
```

## Scope boundary

The course implementation does not require full federated training, network transport, or compatibility with other wire formats.
Host-ready means packed bytes in host memory.
Remote GPU experiment execution remains author-run and is outside this repository scaffold.
