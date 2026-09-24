# Technical contract

Status: accepted course scope; executable implementation pending.

This document carries the technical requirements needed to understand and reproduce the course implementation.
The code and tests become execution authority once they exist.

## Course scope

Everything in this contract is required for the week-13 course submission except the items listed below, which are later extensions and must not block course completion:

- Malformed-record validation beyond header magic, version, reserved bytes, element count, and payload length.
- The full empirical-expectation suite in correctness layer 3; the course requires only a small unbiasedness check across independent seeds.

The benchmark protocol records its own course scope.

## Input and quantizer

- Inputs are finite FP32 vectors with one scale per tensor.
- Empty and all-zero inputs use scale zero.
- Nonfinite inputs are rejected explicitly.
- The scale is the tensor L2 norm, computed with max-rescaling and a reduction that avoids overflow from direct squaring.
- The working implementation uses FP32 arithmetic and an FP64 oracle for numerical checks.
- For bit width `b`, let `s = 2^(b-1)-1` and `a = s*abs(x)/scale` clamped to `[0,s]`.
- Let `l = floor(a)` and choose `k = sign(x)*(l + Bernoulli(a-l))`.
- Decode with `scale*k/s`.

## Randomness

Use shared Philox4x32-10 behavior on CPU and CUDA.
The generator is Random123 v1.14.0 (commit `726a093cd9a73f3ec3c8d7a70ff10ed8efec8d13`), vendored under `third_party/random123` with its BSD-style license.
The logical mapping uses a 64-bit seed, a 64-bit group index, a 32-bit tensor identifier, and a 32-bit invocation identifier.
Element `i` uses group `floor(i/4)` and lane `i mod 4`.
Launch geometry must not change this mapping.

The Philox words are laid out as follows, low word first:

| Philox input | Word 0 | Word 1 | Word 2 | Word 3 |
| --- | --- | --- | --- | --- |
| Key | seed bits 0-31 | seed bits 32-63 | — | — |
| Counter | group bits 0-31 | group bits 32-63 | tensor identifier | invocation identifier |

Convert a 32-bit word `r` to a Bernoulli decision with `r < floor(p*2^32)`, holding the threshold in 64 bits because `p=1` gives `2^32`.
Test `p=0` and `p=1` explicitly.
Identifier overflow means a tensor or invocation identifier above `2^32-1`: the stream constructor rejects it rather than truncating it, so two logical streams never share a counter.

## Packed codec

- Support 4-bit and 8-bit codes.
- Store unsigned code `k+s`; valid codes are `0` through `2s`.
- In 4-bit output, the lower-index element occupies the low nibble.
- Zero the unused high nibble for an odd-length payload.
- Use an explicit little-endian header with a four-byte magic, one-byte version, two reserved zero bytes, a uint64 element count, and an FP32 scale.
- The header is 20 bytes and must be serialized without compiler padding.
- The exact magic and version are implementation decisions that must be frozen before backend comparisons.
- Payload length is `ceil(N*b/8)` and header bytes count in compression ratios.
- The decoder validates header fields, payload length, reserved fields, valid codes, padding, and zero-scale records.
- Shape reconstruction uses a common external manifest.

## Correctness layers

1. Exact codes and bytes for prescribed scales and prescribed RNG words.
2. Numerical reconstruction against the FP64 oracle with documented FP32 error bounds.
3. Empirical expectation checks across independent seeds with fixed sample counts and acceptance rules (see Course scope).

Include signed zero, representable extremes, saturation boundaries, non-multiple block sizes, odd payload lengths, and malformed records.
Do not infer correctness from one seed or require identical codes when independently reduced scales differ.
