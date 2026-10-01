# Reproduction path

Status: The CPU 8-bit and 4-bit pipelines, the CUDA 8-bit and 4-bit pipelines, launch-geometry independence, full decoder validation, Layer 3 unbiasedness checks, and the full course benchmark matrix snapshot are implemented.

This document is the public entry point for reproducing the course implementation.

## What exists

The repository builds the command-line tool `build/stoquant`, as a CPU-only or a CUDA-enabled build, and the CUDA RNG check `build/test_rng`.
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
| CUDA RNG build flags | `-O2 -arch=native -Inative -Ithird_party/random123/include --Werror all-warnings -Xcompiler /W4 -Xcompiler /WX -Xcompiler /wd4068` |
| CUDA quantizer build flags | `-O2 -arch=native -Inative -Ithird_party/random123/include --fmad=false --ftz=false --prec-div=true --prec-sqrt=true --Werror all-warnings -Xcompiler /W4 -Xcompiler /WX -Xcompiler /wd4068` |
| CPU build flags | `/O2 /W4 /WX /std:c11 /fp:strict /Inative /Ithird_party/random123/include` |

`nvcc --list-gpu-arch` includes `compute_120`, and `nvcc` accepted MSVC 19.51 without `-allow-unsupported-compiler`.
The `/wd4068` flag silences MSVC warnings about CUDA-only pragmas in the vendored Random123 headers.

## Build and RNG check

Requires `just`, the CUDA toolkit, and an NVIDIA GPU.
On Windows, also install Visual Studio with the x64 C++ tools; `tools/with-msvc.ps1` enters its developer environment, so no developer prompt is needed.

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

Requires `just`, Python 3.11 or newer, `uv`, and a C compiler. On Windows, `tools/with-msvc.ps1` enters the x64 MSVC environment.

```powershell
just test-cpu
```

`just test-cpu` builds the native CPU CLI and three C test executables, then runs the NumPy/pytest oracle and CLI suite. The first run resolves NumPy and pytest from the checked-in `pyproject.toml` and `uv.lock`.

The CLI consumes and produces raw little-endian FP32 files. A seeded compression computes the FP32 L2 scale and uses the logical Philox stream:

```powershell
just build-cpu

# 8-bit compression (default) and decompression
build\stoquant.exe compress --input input.f32 --output tensor_q8.msq --bits 8 --seed 123 --tensor-id 7 --invocation-id 9
build\stoquant.exe decompress --input tensor_q8.msq --output reconstructed_q8.f32

# 4-bit compression and decompression
build\stoquant.exe compress --input input.f32 --output tensor_q4.msq --bits 4 --seed 123 --tensor-id 7 --invocation-id 9
build\stoquant.exe decompress --input tensor_q4.msq --output reconstructed_q4.f32
```

For layer-1 checks, pass a prescribed FP32 scale and a raw little-endian uint32 word file containing one word per input element:

```powershell
build\stoquant.exe compress --input input.f32 --output tensor_q8.msq --bits 8 --seed 123 --scale 2.0 --words prescribed-words.u32
build\stoquant.exe compress --input input.f32 --output tensor_q4.msq --bits 4 --seed 123 --scale 2.0 --words prescribed-words.u32
```

The CLI requires `--seed` for every compression command. `--words` supplies the random decisions for prescribed-scale tests. `--bits` accepts `4` or `8` (defaulting to `8`). The exact header layout, 4-bit nibble packing, pairwise FP32 reduction, and layer-2 bounds are in [the technical contract](technical-contract.md).

`--backend cpu-avx2` runs the multithreaded AVX2 CPU compressor, which writes the same bytes as `--backend cpu`; `--threads N` sets its team. `just vec-report` checks that MSVC vectorizes its hot loops.

The CUDA backend is selected with `--backend cuda`; it supports both 8-bit and 4-bit records (`--bits 8|4`). Optional `--block-size` and `--grid-size` configure execution geometry, and `--k1 optimized` selects the optimized scale-stage kernels, which write the same bytes as the default `reference`. Add `--timings` to emit a single JSON line to stderr with K1, K2, K3 device event times and host measured H2D/D2H copy times. For acceptance timing, use an input of `2^22` FP32 elements and record the output line with the GPU and toolkit from `just toolchain`:

```powershell
# 8-bit CUDA compression with timings
build\stoquant.exe compress --input input-2m22.f32 --output tensor-q8.msq --bits 8 --seed 123 --backend cuda --timings

# 4-bit CUDA compression with timings
build\stoquant.exe compress --input input-2m22.f32 --output tensor-q4.msq --bits 4 --seed 123 --backend cuda --timings
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
   Builds `build/stoquant`, `build/test_codec`, `build/test_quantizer`, and `build/test_quantizer_avx2`, executes the native C test binaries, and runs the pytest test suites.

2. **C unit tests (codec, quantizer and AVX2 quantizer)**:
   ```powershell
   just build-cpu
   .\build\test_codec.exe
   .\build\test_quantizer.exe
   .\build\test_quantizer_avx2.exe
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

## Benchmark matrix and snapshots

The benchmark driver measures in-process throughput with `stoquant bench` for the C comparator and the CUDA resident, resident-graph, and host-origin paths at 4 and 8 bits. By default it runs the size sweep: every power of two from $2^{10}$ to $2^{26}$ elements (136 cases). Each (size, bits) cell is gated on Layer 2 correctness and CPU/CUDA byte parity before timing. Cells run in a seeded random order after a 20-second GPU warm-up, and every path runs as a separate process in each of 24 trials, one per ordering of the four paths. Each timed process first runs about one second of untimed repetitions of its path. The driver excludes physical core 0 from its own affinity before starting any process (protocol revision 3); children inherit the mask.

An evidence sweep refuses to start unless the readiness check passes: at least 4 GiB of physical RAM is available, no other `stoquant` process is running, the git tree is clean, and the GPU reports no active clock-event reason other than `GpuIdle`. There is no reboot-age or open-window limit; uptime is recorded as context. Start evidence commands through `just` or `uv run python -m stoquant <command>`; the installed `stoquant` console command runs as a process of that name and so fails the check. `--ignore-readiness` runs anyway and marks the snapshot non-evidence; a dirty tree needs `--allow-dirty` as well. Earlier snapshots came from earlier revisions of the driver; reproduce each from its recorded revision. The course snapshot `results/2026-09-25-59c8967` was produced by revision `59c8967`, whose driver ran the four course sizes in ascending order without a warm-up.

### Running the benchmark matrix

`just bench-matrix` requires the local ntfy settings and sends a preflight notification before it starts the benchmark driver. See [Run notifications](run-notifications.md) to configure the benchmark topic and token, verify phone delivery, and review what each alert contains. Notification setup does not change the benchmark protocol or results.

```powershell
# Pilot: four sizes, both bit widths, into results/pilots/ (never evidence)
just bench-matrix --pilot

# Build the CUDA executable and run the 136-case size sweep (about 2.8 hours)
just bench-matrix

# Course sizes only, with the sweep's warm-up and random case order
just bench-matrix --counts 1024 16384 262144 4194304

# Publication extension: GPU-origin paths and pinned transfers (nine paths, trials a multiple of 18)
just bench-matrix --boundaries gpu-origin --transfer-policies pageable pinned --trials 18

# Paper extension input families (non-dense snapshots get a -<family> suffix)
just bench-matrix --input-family sparse --counts 1024 16384 262144 4194304
just bench-matrix --input-family model --model-tensors distinct

# Render F1-F4 and report.md (crossover, T1) into derived/ (dense cases only)
just figures results/<date>-<short_rev>
```

### Layer 3 empirical expectation suite

`just unbiasedness` builds the CUDA executable and runs the full expectation suite of the technical contract (correctness layer 3): four inputs, both bit widths, both backends, 4,096 seeds, about a minute on the RTX 5060. It writes `results/<date>-<short_rev>-unbiasedness/unbiasedness.json` and `derived/f_unbiasedness.png` and exits nonzero if any case fails. Run it from a clean tree; `--allow-dirty`, `--seeds N`, `--backends cpu`, and `--output-dir DIR` give non-evidence runs.

### K1 bandwidth baseline

`just k1-baseline` builds the CUDA executable and `build/stream_probe`, then measures the reference K1 against the device's theoretical and achievable bandwidth over the full matrix. It writes `summary.json`, `processes.csv`, `processes.json`, and `stream_probe.json`, and refuses a dirty tree or an existing output folder. `--allow-dirty`, `--counts`, `--bits`, and `--processes N` give non-evidence runs.

### Inspecting results and provenance

Snapshots are stored in `results/<local-date>-<short_rev>/`. Writers create `<name>.partial` first and rename it when the run completes. An interrupted run keeps the partial folder for diagnosis; start a new run with a new name. Raw files remain write-once. Figures and reports regenerate in `derived/`, as specified by [ADR 0005](adr/0005-snapshot-lifecycle.md).

The matrix run also writes `run-plan.json` for progress notifications. Raw files include:
- `manifest.json`: Run-level provenance including git revision and dirty state, hardware, the verified `build_stamp` and build commands from `just --dry-run build-cuda`, CUDA toolkit, driver, transfer policies (`transfer_policies`, `["pageable"]` by default), the measured `paths` and `trial_design` (manifest 3.2), GPU state before and after the warm-up and after the run, the case order and its seed, trial orders, the statistics method and claim rule, input hashes, and from revision 3 the affinity mask, readiness facts, the memory threshold, power plan and HAGS state, and `evidence` with its `non_evidence_reasons`. Readiness metadata is in manifest version 3.3; older snapshots keep their original version.
- `summary.csv`: Per-case `boundary`, `transfer_policy`, `path_label`, and `baseline`, pooled median and IQR, trial-median range, `spread_ratio` and `spread_p90_p10`, `speedup_vs_c` with its range, bootstrap CI and verdict, and the `stable`, `unstable_rev2`, `boundary_inversion`, `direction_supported`, `magnitude_supported`, and `claim_supported_rev2` flags. Snapshots before revision 3 carry `unstable` and `claim_supported` instead.
- `case_*.json`: Per-trial raw samples (`trial_runs[].samples_ms`, `k1_ms`, `k2_ms`, `k3_ms`, for host-origin `h2d_ms` and `d2h_ms`, for GPU-origin `d2h_ms`, and for CPU GPU-origin `cpu_ms`) with each trial's path order and invocation identifiers, the case's `execution_index`, pooled samples, statistics, `stage_medians_ms`, configuration, and correctness validation results.
- `msvc_vectorization_report.txt`: MSVC `/Qvec-report:2` diagnostics from recompiling the comparator sources with the exact benchmarked host flags; the command is at the top of the file.

**Names across the `stoquant` rename.** Snapshots written after [ADR 0004](adr/0004-stoquant-naming.md) record the `stoquant` binary, `native/` sources, `tools/with-msvc.ps1`, the `SQ_ENABLE_CUDA` build flag, and the readiness fact `stoquant_pids`. Earlier snapshots record `mco2`, `src/`, `scripts/`, `MCO2_ENABLE_CUDA`, and `mco2_pids`. Compare these provenance values within one side of the rename, and reproduce an earlier snapshot from its recorded revision.

```powershell
# View summary table of latest snapshot
Get-Content (Get-ChildItem results -Directory | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName\summary.csv
```

## Restructure equivalence check

The restructure stages (issue #46) used a full behavior fingerprint. `just equivalence` builds the CUDA executable and the stream probe, then runs `tools/equivalence.py`, which drives only command-line interfaces and imports no project code. The full fingerprint remains available for publication-output changes and final evidence pinning; routine implementation and refinement PRs use the bounded smoke below.

```powershell
# On the reference commit, with no modified tracked files: write the baseline
just equivalence capture

# On a later commit: exit 0 when equivalent, exit 1 with the differing leaves listed
just equivalence compare
```

`capture` writes `build/equivalence/<sha>.json` under the primary checkout (the parent of the git common directory), so every worktree shares it. It refuses to overwrite an existing baseline and refuses a tree with modified tracked files unless given `--allow-dirty`. `compare` reads the only stored baseline, or the one named by `--baseline` (a file path or a commit prefix in the store), and never writes to the store. Once the store holds several baselines, name one; a commit prefix such as `--baseline 1475fbf` avoids quoting a primary-checkout path that contains spaces. Both accept `--store DIR`. The stage 0 baseline is keyed to `main` at `03b29f3`.

A change that moves a loop or deletes a function shifts the MSVC vectorization-report counts the fingerprint records, so `compare` fails on those leaves alone. After confirming that only those counts differ, disclose the deviation in the pull request, run `just equivalence capture` on the branch head, and check it with `compare --baseline <sha>`.

Each run fingerprints:

- short benchmark-matrix sweeps (dense, sparse, and model families, with GPU-origin paths and both transfer policies), a small unbiasedness run, and short K1 baseline and K1 A/B runs, all with warm-ups shortened and readiness overridden;
- the record bytes and decompressed output of `compress` and `decompress` over fixed inputs, for every backend, both bit widths, two seeds, both K1 variants, and the tensor and invocation identifiers;
- every frozen snapshot under `results/` except `pilots/`: a content hash of each, plus a report re-rendered from it into a scratch folder. Snapshots that no command can re-render from a stored folder (the unbiasedness and K1 baseline snapshots, the issue #26 diagnosis, and the issue #30 trace) are fingerprinted by content hash alone; the live unbiasedness and K1 baseline runs cover their code.

Measured values are redacted. These include timings, timestamps, and bandwidth and statistics that depend on timing. Subtrees that describe the machine, the tree, or timing-dependent decisions are collapsed, so even their length is ignored: build flags, git provenance, hardware, toolkit and driver, device, GPU state, readiness facts and verdicts, evidence status and its reasons, and the K1 A/B decision lists (the keep rule itself is unit-tested on fixed cells). Paths are scrubbed, and the binary name is normalized. Figures drawn from timings are recorded only as present; the unbiasedness figure is hashed. Redaction rules match key paths from each output file's root, so the output layout is part of the compared behavior. Everything else must match exactly: case lists and order, configuration, correctness results, exit codes, record and file hashes, CSV key columns, and vectorization verdict counts. The run also fails if any file under `results/` changes.

Later stages may change only the `INVOCATION` table at the top of the tool, which holds the binary path, the command that starts each script, and the environment-variable prefixes to scrub. A stage that needs any other change to the tool is not behavior-preserving.

### Routine implementation PR smoke

Build the CUDA binary once, then run the bounded smoke:

```powershell
just build-cuda
just equivalence-smoke
```

`just equivalence-smoke` stops its native commands by 285 seconds, leaving time for cleanup and reporting within a five-minute cap. It runs fixed normal, mixed odd-length, and all-zero inputs at 4 and 8 bits through the scalar CPU, AVX2 CPU, and CUDA reference and optimized K1 paths. Every record must match the scalar CPU record byte for byte, and each record must decode to the same output. The CUDA build stamp is checked before any smoke case runs. Temporary inputs and outputs are removed when the command exits.

This is a selected behavior smoke against implementations in the current tree. It does not compare the whole current tree with a fingerprint from an earlier revision, nor cover snapshot rendering, monitoring, or every benchmark and report output. Run focused tests for the changed modules as usual. Use the full capture/compare workflow when publication output contracts or claim rules change and for the final whole-output audit associated with an evidence pin.

## Scope boundary

The course implementation does not require full federated training, network transport, or compatibility with other wire formats.
Host-ready means packed bytes in host memory.
Remote GPU experiment execution remains author-run and is outside this repository scaffold.
