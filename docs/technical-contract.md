# Technical contract

Status: The 8-bit and 4-bit CPU pipelines, full decoder validation, and Layer 3 unbiasedness checks are implemented. The CUDA pipeline currently supports 8-bit records; CUDA 4-bit support and broader launch-geometry comparisons remain follow-on work.

This document carries the technical requirements needed to understand and reproduce the course implementation.
The code and tests define current behavior. This contract freezes the 8-bit and 4-bit record formats, numerical rules, overflow policy, and Layer 3 acceptance criteria.

## Course scope

The week-13 course submission requires every item in this contract except these later extensions:

- CUDA 4-bit quantization and broader launch-geometry comparisons.
- The full empirical-expectation suite in correctness layer 3; the course requires only a small unbiasedness check across independent seeds.

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

### CUDA 8-bit path

The CUDA backend accepts 8-bit records. It writes the same header and payload bytes as the CPU backend for the same input, seed, identifiers, and prescribed words. A CPU-only executable rejects `--backend cuda` with an explicit unavailable-backend error.

K1 computes the maximum absolute value, 256-element normalized square sums, and the fixed pairwise reduction of block sums. Its logical block boundaries and reduction tree match the CPU implementation. The CUDA translation unit is built with `--fmad=false`, `--ftz=false`, `--prec-div=true`, and `--prec-sqrt=true`; fast math is disabled.

K2 assigns one CUDA thread to each four-word Philox group. Grid-stride processing preserves the group and lane mapping for any launch geometry and handles the final partial group. It applies the same FP32 scaling, clipping, Bernoulli threshold, and signed-code conversion as the CPU path. K3 copies the byte codes into the 8-bit record payload.

`--timings` writes one JSON object to stderr with `k1_ms`, `k2_ms`, `k3_ms`, `h2d_ms`, and `d2h_ms`. CUDA events measure the three device stages; host elapsed time measures the input, words, scale, status, and payload copies. The timing line describes one invocation and does not substitute for the benchmark protocol.

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
