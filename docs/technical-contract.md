# Technical contract

Status: The 8-bit CPU pipeline and decoder are implemented. CUDA quantization and 4-bit packing remain outside this slice.

This document carries the technical requirements needed to understand and reproduce the course implementation.
The code and tests define current behavior. This contract freezes the 8-bit record format and numerical rules.

## Course scope

The week-13 course submission requires every item in this contract except these later extensions:

- Malformed-record validation beyond header magic, version, reserved bytes, element count, and payload length.
- The full empirical-expectation suite in correctness layer 3; the course requires only a small unbiasedness check across independent seeds.

The benchmark protocol records its own course scope.

## Input and quantizer

- Inputs are finite FP32 vectors with one scale per tensor.
- Empty and all-zero inputs use scale zero and encode every payload element as the center code `127`.
- Nonfinite inputs are rejected explicitly.
- The scale is the tensor L2 norm. Max-rescaling avoids overflow from direct squaring.
- Pass 1 finds the maximum absolute value. Pass 2 computes `(abs(x)/max_abs)^2` in FP32, pairwise-sums zero-padded blocks of 256, pairwise-sums the block results, then computes `max_abs*sqrt(sum)`. This fixes the reduction order.
- The NumPy oracle independently implements Philox4x32-10, the same FP32 reduction and code path, and an FP64 reference used for numerical checks.
- For bit width `b`, let `s = 2^(b-1)-1` and compute `a = min((abs(x)/scale)*s, s)` in that operation order.
- Let `l = floor(a)` and choose `k = sign(x)*(l + Bernoulli(a-l))`.
- Decode with `scale*k/s`.
- This CPU slice implements `b=8`, so `s=127` and the unsigned payload byte is `k+127`.

### FP32 scale and reconstruction bound

For a nonzero vector, let `N` be its element count and `M = ceil(N/256)` its number of blocks. Let `eta = 2^-149` be the smallest positive FP32 subnormal, `u = 2^-24`, and `gamma_K = K*u/(1-K*u)` where `K = 12 + ceil(log2(M))`.

When intermediate values are finite and round to nearest, the conservative relative scale bound against FP64 L2 norm `S64` is:

`abs(S32-S64)/S64 <= epsilon`, where `epsilon = gamma_K + 4*N*eta + eta/(2*S64)`.

The `gamma_K` term covers the FP32 operation depth through division, squaring, the two pairwise reductions, square root, and final multiply. The additive terms cover underflow during normalization and reduction and a subnormal final scale. CUDA must use its own tested scale bound.

Each stochastic code is one of the two integers adjacent to its unrounded magnitude. For CPU and FP64 reconstructions that use the same Philox words, a conservative per-element bound is:

`abs(y32-y64) <= S64 * (2/127 + 2*epsilon/(1-epsilon) + 5*u)` for `epsilon < 1`.

The first term allows one code step for each path. The remaining terms cover scale drift and FP32 dequantization. Tests exercise both bounds on a multi-block vector. Prescribed-scale layer-1 cases require exact code and byte equality.

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

- The CPU slice implements 8-bit codes and reserves the same header format for later 4-bit payloads.
- Store unsigned code `k+s`; for 8-bit output, valid codes are `0` through `254`.
- The 20-byte header is serialized field-by-field in little-endian order, without compiler padding:

| Offset | Size | Field | Frozen value or encoding |
| ---: | ---: | --- | --- |
| 0 | 4 | Magic | ASCII `MSQ1` |
| 4 | 1 | Version | `1` |
| 5 | 1 | Bit width | `8` for this CPU slice |
| 6 | 2 | Reserved | Zero |
| 8 | 8 | Element count | Unsigned uint64, little-endian |
| 16 | 4 | Scale | IEEE-754 FP32 bits, little-endian |

- The payload has exactly `N` bytes and header bytes count in compression ratios.
- The decoder validates magic, version, bit width, reserved bytes, count, payload length, valid codes, and zero-scale records.
- Empty and all-zero records have zero scale; any nonempty zero-scale record contains only the center code `127`.
- The CPU CLI reads and writes raw little-endian FP32 vectors. Shape reconstruction uses a common external manifest.

## Correctness layers

1. Exact codes and bytes for prescribed scales and prescribed RNG words.
2. Numerical reconstruction against the FP64 oracle with documented FP32 error bounds.
3. Empirical expectation checks across independent seeds with fixed sample counts and acceptance rules (see Course scope).

Include signed zero, representable extremes, saturation boundaries, non-multiple block sizes, odd payload lengths, and malformed records.
Evaluate empirical expectation across independent seeds. Compare codes byte for byte only when both paths use the same scale and RNG words.
