# Technical contract

Status: The 8-bit and 4-bit CPU pipelines, the 8-bit and 4-bit CUDA pipelines, launch-geometry independence, full decoder validation, Layer 3 unbiasedness checks, and the in-process benchmark timing paths and driver snapshot are implemented.

This document carries the technical requirements needed to understand and reproduce the course implementation.
The code and tests define current behavior. This contract freezes the 8-bit and 4-bit record formats, numerical rules, overflow policy, Layer 3 acceptance criteria, and benchmark timing boundaries.

## Course scope

The week-13 course submission requires every item in this contract except these later extensions:

- The full empirical-expectation suite in correctness layer 3; the course requires only the 1,024-element subset.

The benchmark protocol records its own course scope.

## Input and quantizer

- Inputs are finite FP32 vectors with one scale per tensor.
- Empty and all-zero inputs use scale zero and encode every payload element as the center code `s` (`127` for 8-bit, `7` for 4-bit).
- Nonfinite inputs are rejected explicitly.
- The scale is the tensor L2 norm. Max-rescaling avoids overflow from direct squaring.
- If the FP32 L2 norm overflows, the tool exits nonzero with a distinct explicit error (`FP32 L2 scale overflow`). The scale is never saturated.
- Pass 1 finds the maximum absolute value. Pass 2 computes `(abs(x)/max_abs)^2` in FP32, pairwise-sums zero-padded blocks of 256, pairwise-sums the block results, then computes `max_abs*sqrt(sum)`. This fixes the reduction order.
- The NumPy oracle independently implements Philox4x32-10, the same FP32 reduction and code path, and an FP64 reference used for numerical checks.
- For bit width `b`, let `s = 2^(b-1)-1` and compute `a = min((abs(x)/scale)*s, s)` in that operation order.
- Let `l = floor(a)` and choose `k = sign(x)*(l + Bernoulli(a-l))`.
- Decode with `scale*k/s`. Signed zero yields `k = 0` and decodes to `+0`.
- This CPU pipeline implements `b=8` (`s=127`) and `b=4` (`s=7`).

### CUDA quantization pipeline

The CUDA backend is selected with `--backend cuda` (default `cpu`) and accepts both 8-bit (`--bits 8`) and 4-bit (`--bits 4`) records. It writes byte-identical headers and payloads to the CPU backend for the same input, seed, identifiers, and prescribed words. A CPU-only executable rejects `--backend cuda` with an explicit unavailable-backend error.

- **K1 (Scale reduction)**: Computes the maximum absolute value, 256-element normalized square sums, and the fixed pairwise reduction of block sums. Its logical block boundaries (256 threads per block) and adjacent-pair reduction tree match the CPU implementation bit for bit. The reduction is structured as a grid-stride loop, producing identical scale values across all grid sizes. The CUDA translation unit is built with `--fmad=false`, `--ftz=false`, `--prec-div=true`, and `--prec-sqrt=true`; fast math is disabled.
- **K2 (Stochastic rounding)**: Assigns one CUDA thread to each 4-element Philox group in a grid-stride loop, with remainder handling for final partial groups. Key and counter follow the contract mapping. Each thread computes `a = min((|x|/scale)*s, s)`, `l = floor(a)`, `p = a - l`, and the 64-bit Bernoulli decision threshold, generating unsigned codes `k + s` (`s=127` for 8-bit, `s=7` for 4-bit). A prescribed-word variant reads words from device memory for Layer 1 verification.
- **K3 (Payload packing)**: Runs in a grid-stride loop with one thread per output byte. For 8-bit records, each thread copies one code byte to the payload. For 4-bit records, each thread packs two codes into one byte: the lower-index element `2*i` is stored in the low nibble (bits 0-3), the higher-index element `2*i + 1` is stored in the high nibble (bits 4-7), and the unused high nibble of an odd-length payload is strictly zeroed.
- **Launch geometry**: `--block-size` and `--grid-size` configure K2 and K3 launch geometry (and `--grid-size` configures K1 grid); K1's block size is fixed at 256 by the reduction order. The grid-stride loop structure guarantees byte-identical output across all valid block and grid configurations, including grids smaller than the element count. Specifying launch geometry options with `--backend cpu` is rejected as an invalid argument.
- **Timing boundary**: `--timings` writes one JSON object to stderr with `k1_ms`, `k2_ms`, `k3_ms`, `h2d_ms`, and `d2h_ms`. CUDA events measure resident kernel execution across K1, K2, and K3; host elapsed time measures the input, words, scale, status, and payload copies. The timing line describes one invocation and does not substitute for the benchmark protocol.

### In-process benchmark timing paths and boundaries

The `mco2 bench` subcommand measures in-process compression throughput without process startup, dynamic memory allocation, or file I/O inside the measured intervals. It loads the input file and preallocates all host and device buffers once, runs `--warmup N` (default 10) warmups and `--reps N` (default 30) measured repetitions, and writes one JSON object to stdout containing configuration echo, raw per-repetition wall-clock samples in milliseconds, per-repetition K1/K2/K3 event times for CUDA runs, per-repetition H2D event times (`h2d_ms`) for `host-origin` runs, per-repetition D2H event times (`d2h_ms`) for `host-origin` and `gpu-origin` runs, per-repetition CPU compression times (`cpu_ms`) for CPU `gpu-origin` runs, and header and payload byte counts. The configuration echo includes `boundary` and `transfer_policy`.

- **Timing boundaries**:
  - `resident` (CUDA-only): measures device input to device packed bytes across K1, K2, and K3, synchronizing the CUDA device before stopping the monotonic clock.
  - `resident-graph` (CUDA-only): measures device input to device packed bytes using a pre-captured CUDA Graph (1 memset plus the same K1-K3 kernel launches as `resident`) executed via `cudaGraphLaunch` on the benchmark stream, synchronizing the CUDA device before stopping the monotonic clock. Capture and instantiation happen once, outside timing, and are recorded as `capture_and_instantiate_ms`. K1-K3 stage timings are absent for this boundary, so its runs carry none of the six per-repetition CUDA events that `resident` records. Before capture, the device payload is overwritten with the bitwise complement of the preflight payload, so a graph that skipped K3 cannot pass. After timing, the last launch's outputs are read back: its payload and scale must match the uncaptured preflight record byte for byte, and its status and validation flags must pass the same result checks as `resident`; otherwise the benchmark fails.
  - `host-origin` (CUDA-only): measures the complete host-to-host execution path: Host-to-Device input transfer, K1, K2, K3 execution, and Device-to-Host packed payload and scale transfers, all issued with `cudaMemcpyAsync` on the benchmark stream and followed by `cudaDeviceSynchronize`, synchronizing the CUDA device before stopping the monotonic clock.
  - `gpu-origin` with `--backend cuda`: the input is uploaded to the device once, outside timing. Each repetition runs K1, K2, and K3 on the resident input, then copies the packed payload and scale to the host with `cudaMemcpyAsync` on the benchmark stream, synchronizing the CUDA device before stopping the monotonic clock. As for `host-origin`, the wall time also includes the small status and validation-flag copies; `d2h_ms` is the event time of the payload and scale copies only.
  - `gpu-origin` with `--backend cpu`: the input is uploaded to the device once, outside timing. Each repetition copies the full input from the device into a host landing buffer and waits for the copy (`d2h_ms`, from CUDA events around it), then runs the single-thread C compressor on the landing buffer (`cpu_ms`, from the monotonic clock). The wall time covers both. This path needs a CUDA build; a CPU-only build rejects it as CUDA-unavailable.
  - C comparator (`--backend cpu`, the default `host-host` boundary): measures single-thread C execution on preallocated host buffers from host input to host packed bytes. It implements one path because CPU resident and host-origin are identical. With `--backend cpu`, `--boundary` accepts only `gpu-origin`; `--block-size` and `--grid-size` are rejected as invalid arguments.
- **Transfer policy**: `--transfer-policy pageable|pinned` (default `pageable`) selects the host buffers that a timed transfer touches. `pageable` uses `malloc`; `pinned` uses `cudaHostAlloc(..., cudaHostAllocDefault)` and `cudaFreeHost`. Under `pinned`, every host buffer of a timed copy is page-locked: the payload, scale, status, and validation-flag buffers, the CPU `gpu-origin` landing buffer, and for `host-origin` a page-locked copy of the input, made once outside timing. The option applies only to `host-origin` and `gpu-origin`; with any other boundary it is rejected, and the echoed policy is `none`. `pageable` reflects standard host tensor integration and is the revision 3 policy.
- **Invocation identifiers**: Repetition `r` (0-indexed) uses invocation identifier `base + r`; warmup `w` uses `base + reps + w`. Identifiers that would exceed $2^{32}-1$ fail before timing begins. `resident-graph` is the exception: its captured K2 arguments fix the base invocation's RNG stream, so every warmup and repetition replays invocation `base`, and the JSON reports `base` for each. The work per launch is the same; only the stochastic-rounding draws repeat. The driver stores a repeated run as `{"repeated": base, "count": n}`.
- **Driver paths**: By default the benchmark driver (`benchmark_driver.py`) runs the four revision 3 paths: `cpu-comparator`, `cuda-resident`, `cuda-resident-graph`, and `cuda-host-origin`. `--boundaries gpu-origin` adds `cpu-gpu-origin` and `cuda-gpu-origin` for each transfer policy, and `--transfer-policies pageable pinned` adds `cuda-host-origin-pinned` and the pinned `gpu-origin` pair. The policy list must include `pageable`, and both options need the `cuda` backend. Paths run in a fixed order: comparator, resident, resident-graph, host-origin, pinned host-origin, then for each policy CPU and CUDA `gpu-origin`. Each case records `transfer_policy` and `path_label`; pageable paths keep their revision 3 case identifiers, and a pinned path's identifier carries a `-pinned` suffix.
- **Driver trials**: The driver runs every path of a matrix cell as a separate `mco2 bench` process in each of `--trials N` trials (default 24). With at most four paths, the trials cycle through every path ordering; with four, the 24 trials run each ordering once ($4! = 24$), so each path occupies each position and follows each other path equally often. With more than four paths the driver uses a Williams design: the first row is `0, 1, n-1, 2, n-2, ...`, row `i` adds `i` to each entry modulo `n`, and for odd `n` the reversed rows are appended. Across one design length (`n` rows for even `n`, `2n` for odd; 18 for the nine extension paths) each path occupies each position, and follows each other path, equally often. `--trials` must be a multiple of the design length, checked before any process runs. The manifest records `paths` and `trial_design` (`all-permutations` or `williams`). Every trial reuses base invocation identifier 0, making trials replicates of one workload; the driver copies each process's identifiers from its JSON output. Each case reports the pooled median and IQR over all trials (`numpy.percentile`, `method="linear"`), the per-trial medians, their spread ratio (maximum over minimum), and `spread_p90_p10` (90th over 10th percentile of the trial medians, same method). The driver excludes logical CPUs 0 and 1 (physical core 0) from its own process affinity before it starts any benchmark or probe process, so every path runs off core 0; the manifest records the mask.
- **Baselines**: Each path is compared with a CPU baseline of the same transfer policy, recorded as `baseline` (empty for the comparator itself). `resident`, `resident-graph`, `host-origin` (both policies), and CPU `gpu-origin` (both policies) use the comparator. CUDA `gpu-origin` uses CPU `gpu-origin` of the same policy. A pinned path also reports `vs_pageable` against its pageable twin under the same rules. CUDA `gpu-origin` also reports `vs_comparator`, marked `descriptive`, with a speedup and verdict but no claim flags. `speedup_vs_c` and its range, CI, verdict, and claims are always against `baseline`.
- **Speedup and claim rule** (protocol revision 3): A case's point speedup is its baseline's pooled median over its own pooled median. Its range divides the baseline's minimum trial median by the case's maximum and the baseline's maximum by the case's minimum. The verdict is `faster` when the whole range exceeds 1, `slower` when it lies below 1, and `inconclusive` otherwise. Each speedup also reports a 95% percentile bootstrap interval (10,000 independent resamples of each side's trial medians from `numpy.random.default_rng(31)`; each resample's statistic is the baseline's median resampled trial median over the case's); no claim depends on it. `boundary_inversion` marks a path that beats a path whose work it contains, which is physically implausible. CUDA boundaries are checked within each transfer policy: host-origin must not be below any resident path (`resident`, `resident-graph`), `gpu-origin` must not be below `resident`, and host-origin must not be below `gpu-origin`. CPU `gpu-origin` must not be below the comparator. An inverted policy group flags every CUDA path in it; the resident paths, which have no transfer, belong to the pageable group. CPU `gpu-origin` is flagged by its own rule, and CUDA `gpu-origin` by its policy group or by its CPU baseline's rule. A pinned path's `vs_pageable` comparison is flagged when either the pinned path or its pageable twin is. With the default paths only the host-origin rule applies, as in revision 3. A case is `stable` when `spread_p90_p10` is at most 1.25. Two claims follow:
  - `direction_supported`: the verdict is `faster` or `slower` and the cell is not inverted;
  - `magnitude_supported`: `direction_supported`, and both the case and its baseline are stable.

  `resident-graph` is also compared with `resident` under the same rules, with `resident` as the baseline. The revision 2 rule (conclusive verdict, no inversion, both spread ratios at most 1.25) is still computed as `claim_supported_rev2` for comparison only. These thresholds were fixed before the revision 3 pilot and sweep; earlier snapshots keep the `claim_supported` rule they recorded.
- **Snapshot provenance**: The driver refuses a dirty working tree unless `--allow-dirty` is passed, and then records the dirty files; evidence snapshots come from a clean committed revision. Build flags are the commands `just --dry-run build-cuda` prints, and the MSVC vectorization report recompiles the comparator sources with exactly those host flags plus `/Qvec-report:2`. `nvidia-smi` clock, power, thermal, and clock-event readings before the run, after the initial GPU warm-up, after each case's warm-up, and after the run are recorded as context, not as proof of in-run clock stability.
- **Case order and warm-up**: The driver runs (size, bits) cells in a random order from `numpy.random.default_rng(--case-order-seed)` (default 612) and records the order in the manifest. Before the first timed process it runs the resident path untimed on the largest input for `--gpu-warmup-seconds` (default 20); before each cell's timed processes, after its correctness gate, it does the same on the cell's input for `--case-warmup-seconds` (default 3).
- **Stage medians**: Each CUDA case, and each CPU `gpu-origin` case, records `stage_medians_ms`: the pooled median of each stage time it has (for CPU `gpu-origin`, `d2h_ms` and `cpu_ms`), and `other_ms`, the pooled median of each repetition's wall time minus its stage times. Stage medians diagnose where time goes; they do not sum to the headline median.
- **Record validation**: The optional `--record-output` writes the base-configuration record once outside timing. The benchmark driver verifies that every path's benchmark record, under every boundary and transfer policy, is byte-identical to `mco2 compress` across both backends, and verifies Layer 2 decoding against the FP64 reference oracle before timing begins.

### FP32 scale and reconstruction bound

For a nonzero vector, let `N` be its element count and `M = ceil(N/256)` its number of blocks. Let `eta = 2^-149` be the smallest positive FP32 subnormal, `u = 2^-24`, and `gamma_K = K*u/(1-K*u)` where `K = 12 + ceil(log2(M))`.

When intermediate values are finite and round to nearest, the conservative relative scale bound against FP64 L2 norm `S64` is:

`abs(S32-S64)/S64 <= epsilon`, where `epsilon = gamma_K + 4*N*eta + eta/(2*S64)`.

The `gamma_K` term covers the FP32 operation depth through division, squaring, the two pairwise reductions, square root, and final multiply. The additive terms cover underflow during normalization and reduction and a subnormal final scale. The 8-bit CUDA path preserves the same operation and reduction order with fused multiply-add and fast math disabled, so its computed FP32 scale matches the CPU result exactly.

Each stochastic code is one of the two integers adjacent to its unrounded magnitude. For CPU and FP64 reconstructions that use the same Philox words, a conservative per-element bound for bit width `b` (with `s = 2^(b-1) - 1`) is:

`abs(y32-y64) <= S64 * (2/s + 2*epsilon/(1-epsilon) + 5*u)` for `epsilon < 1`.

For 8-bit records (`s = 127`):
`abs(y32-y64) <= S64 * (2/127 + 2*epsilon/(1-epsilon) + 5*u)`

For 4-bit records (`s = 7`):
`abs(y32-y64) <= S64 * (2/7 + 2*epsilon/(1-epsilon) + 5*u)`

The first term allows one code step for each path. The remaining terms cover scale drift and FP32 dequantization. Tests exercise both bounds on a multi-block vector at 8-bit and 4-bit widths. Prescribed-scale layer-1 cases require exact code and byte equality.

## Randomness

Use shared Philox4x32-10 behavior on CPU and CUDA.
The generator is Random123 v1.14.0 (commit `726a093cd9a73f3ec3c8d7a70ff10ed8efec8d13`), vendored under `third_party/random123` with its BSD-style license.
The logical mapping uses a 64-bit seed, a 64-bit group index, a 32-bit tensor identifier, and a 32-bit invocation identifier.
Element `i` uses group `floor(i/4)` and lane `i mod 4`.
Launch geometry must not change this mapping.

The Philox words are laid out as follows, low word first:

| Philox input | Word 0 | Word 1 | Word 2 | Word 3 |
| --- | --- | --- | --- | --- |
| Key | seed bits 0-31 | seed bits 32-63 | n/a | n/a |
| Counter | group bits 0-31 | group bits 32-63 | tensor identifier | invocation identifier |

Convert a 32-bit word `r` to a Bernoulli decision with `r < floor(p*2^32)`, holding the threshold in 64 bits because `p=1` gives `2^32`.
Test `p=0` and `p=1` explicitly.
The stream constructor rejects any tensor or invocation identifier above `2^32-1`. This prevents counter truncation and preserves distinct logical streams.

## Packed codec

- The CPU pipeline implements both 8-bit (`b=8`, `s=127`) and 4-bit (`b=4`, `s=7`) records using the shared 20-byte header format.
- Store unsigned code `k+s`. Valid stored codes are `0` through `2*s` (`0` through `254` for 8-bit; `0` through `14` for 4-bit).
- Packing layout:
  - For 8-bit records, each byte holds one element code. Payload size is `N` bytes.
  - For 4-bit records, two elements are packed into each byte: the lower-index element `2*i` is stored in the low nibble (bits 0-3), and the higher-index element `2*i + 1` is stored in the high nibble (bits 4-7). For odd element count `N`, payload size is `ceil(N / 2) = floor((N + 1) / 2)` bytes, and the unused high nibble (bits 4-7) of the final byte must be zero.
- The 20-byte header is serialized field-by-field in little-endian order, without compiler padding:

| Offset | Size | Field | Frozen value or encoding |
| ---: | ---: | --- | --- |
| 0 | 4 | Magic | ASCII `MSQ1` |
| 4 | 1 | Version | `1` |
| 5 | 1 | Bit width | `4` or `8` |
| 6 | 2 | Reserved | Zero |
| 8 | 8 | Element count | Unsigned uint64, little-endian |
| 16 | 4 | Scale | IEEE-754 FP32 bits, little-endian |

- The decoder strictly validates records and rejects malformed inputs with distinct error messages:
  1. Header magic mismatch: must match ASCII `MSQ1` (`0x3151534D`).
  2. Unsupported version: must be `1`.
  3. Unsupported bit width: must be `4` or `8`.
  4. Nonzero reserved bytes: offsets 6-7 must be `0x0000`.
  5. Nonfinite or negative scale: scale must be finite and `>= 0.0f`.
  6. Unexpected payload length: truncated records or trailing bytes beyond expected payload size (`N` bytes for 8-bit, `(N+1)/2` bytes for 4-bit).
  7. Nonzero padding nibble: for 4-bit records with odd `N`, the high nibble of the final byte must be zero.
  8. Out-of-range code: any code byte (8-bit) or nibble (4-bit) exceeding `2*s` (`> 254` for 8-bit, `> 14` for 4-bit) is rejected.
  9. Malformed zero-scale record: for `scale == 0.0f` with `N > 0`, every payload element must be the exact center code `s` (`127` for 8-bit, `7` for 4-bit). Any code other than `s` is rejected.
- Empty (`N=0`) and all-zero records have scale zero. Empty records have a 0-byte payload.
- The CPU CLI reads and writes raw little-endian FP32 vectors. Shape reconstruction uses a common external manifest.

## Correctness layers

1. **Exact codes and bytes**: Prescribed scales and prescribed RNG words yield bit-for-bit identical codes and headers between CPU C implementation and NumPy oracle at both 4-bit and 8-bit widths.
2. **Numerical reconstruction bounds**: Seeded CPU reconstruction against FP64 reference satisfies the documented conservative per-element bounds at both 4-bit and 8-bit widths:
   `abs(y32 - y64) <= S64 * (2/s + 2*epsilon/(1-epsilon) + 5*u)` with `s=127` (8-bit) and `s=7` (4-bit).
   Relative scale error satisfies `abs(S32 - S64) / S64 <= epsilon`.
3. **Empirical unbiasedness (course subset)**:
   - Evaluated on a fixed 1,024-element input vector across $T = 4,096$ fixed, independent Philox seeds (`seed = 1..4096`, `tensor_id = 0`, `invocation_id = 0`) at both 4-bit and 8-bit widths.
   - For each element $i \in \{0, \dots, N-1\}$, let $a_i = \min((|x_i| / \text{scale}) \cdot s, s)$ and $p_i = a_i - \lfloor a_i \rfloor$.
   - Acceptance rule: each element's sample mean decoded value $\bar{y}_i = \frac{1}{T} \sum_{t=1}^T \hat{x}_{i,t}$ must satisfy:
     $$|\bar{y}_i - x_i| \le 5 \cdot \frac{\text{scale}}{s} \cdot \sqrt{\frac{p_i(1-p_i)}{T}}$$
   - Exact equality $|\bar{y}_i - x_i| = 0$ must hold whenever $p_i \in \{0, 1\}$.
   - Edge case coverage: signed zero (preserving $+0$ decode), large magnitudes without intermediate overflow via max-rescaling, saturation boundaries ($|x| \ge \text{scale}$), lengths not multiples of 4 or 256, and odd payload lengths.
   - Scale overflow policy: if the FP32 L2 norm overflows, the tool exits nonzero with distinct error message `FP32 L2 scale overflow`. The scale is never saturated.
   - **Full expectation suite (paper extension)**, `unbiasedness.py`, spec [qu1r0ra/CSC612M-MCO2-Paper#21](https://github.com/qu1r0ra/CSC612M-MCO2-Paper/issues/21):
     - Inputs: the course-subset vector (1,024 elements, design scale 2.0 overridden by the computed L2 scale), a dense $2^{14}$ vector, the sparse $2^{14}$ vector, and the model tensor `layer1.0.conv1.weight` (36,864 elements), all from the benchmark protocol's input families. Every input runs at 4 and 8 bits on both backends, $T = 4,096$ seeds (`seed = 1..4096`, `tensor_id = 0`, `invocation_id = 0`), with the computed FP32 L2 scale.
     - `mco2 expect --input X.f32 --output S.f64 --seeds T [--seed-start 1] [--bits 4|8] [--backend cpu|cuda]` compresses and decodes the input once per seed and writes the per-element FP64 sums of the decoded values, then their sums of squares, as little-endian FP64. Its JSON line records the scale bits. The CPU and CUDA outputs must be byte-identical.
     - Target: the rule compares the mean with the expectation of the implemented quantizer, not with $x_i$. In FP32, $a_i = \min((|x_i|/	ext{scale}) \cdot s, s)$, $l_i = \lfloor a_i 
  floor$, $p_i = a_i - l_i$. The codec rounds up when the 32-bit word is below $\lfloor 	ext{fl}_{32}(p_i \cdot 2^{32}) 
  floor$, so the realised probability is $q_i = \lfloor 	ext{fl}_{32}(p_i \cdot 2^{32}) 
  floor / 2^{32}$. With $	ext{dec}(k) = 	ext{fl}_{32}(	ext{fl}_{32}(k/s) \cdot 	ext{scale})$, signed like $x_i$: $E_i = (1-q_i)\,	ext{dec}(l_i) + q_i\,	ext{dec}(l_i+1)$ and $\Delta_i = |	ext{dec}(l_i+1) - 	ext{dec}(l_i)|$.
     - Acceptance rule: where $q_i > 0$, $|ar{y}_i - E_i| \le 5 \cdot \Delta_i \cdot \sqrt{q_i(1-q_i)/T}$; where $q_i = 0$, $ar{y}_i = 	ext{dec}(l_i)$ exactly. The suite fails if any element fails or the backends differ.
     - Reported, not gated: the pooled ratio of observed to expected variance, and the largest $|E_i - x_i|$ in steps of $	ext{scale}/s$ (FP32 rounding of $a_i$ and of the decoded levels).
     - Output: `just unbiasedness` writes `results/<date>-<short_rev>-unbiasedness/` with `unbiasedness.json` and `f_unbiasedness.png` (per bit width, mean error against $x/	ext{step}$ with the 5σ envelope, and the histogram of error over bound). It refuses an existing directory or a dirty tree unless `--allow-existing` or `--allow-dirty` is passed; `evidence` is false for a dirty tree, fewer seeds, or one backend.
