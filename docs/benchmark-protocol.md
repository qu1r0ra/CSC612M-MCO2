# Benchmark protocol

Status: accepted protocol; executable benchmark pending.

The benchmark must expose the cost of normalization, random-number generation, rounding, packing, and required transfers.
Do not headline a CUDA speedup from Python interpreter overhead or from separately timed stages summed into a complete path.

## Comparison backends

- Use a native C++/CUDA executable for the compiled CPU and CUDA backends; use Python for the reference and analysis.
- Keep the course's sequential Python reference explicit and scalar. A vectorized implementation may be useful for comparison, but it is not the sequential baseline.
- Use a single-threaded compiled CPU comparator. Record compiler vectorization settings and identify a scalar configuration for the course comparison.

## Timing boundaries

| Boundary | CPU path | CUDA path |
| --- | --- | --- |
| Resident computation | Host input to host packed bytes | Device input to device packed bytes |
| GPU-origin, host-ready output | Full input D2H, then CPU compression | CUDA compression, then packed output D2H |
| Host-origin, host-ready output | CPU compression | Input H2D, CUDA compression, packed output D2H |

Use synchronized wall-clock timing for complete paths and CUDA events for device-stage diagnosis.
Keep allocation, warmup, transfer-buffer type, synchronization policy, process startup, and file-I/O treatment explicit.
Verify decoding outside the measured compression interval.

## Matrix

- Main platform: the local RTX 5060, subject to fresh inventory and successful build verification.
- Element counts: `2^10`, `2^14`, `2^18`, and `2^22`.
- Bit widths: 4 and 8.
- Input families: dense centered values and sparse values.
- Add one pinned synthetic collection shaped like reference-model tensors without requiring training data.
- Use 10 warmups and at least 30 measured repetitions per case.
- Report median and interquartile range.

## Required provenance

Each case records seeds, invocation identifiers, code revision, build flags, hardware, transfer policy, header and payload bytes, timing boundary, and correctness status.
Preserve raw samples and configuration with the result.
Inspect stability before making crossover claims, and report a missing crossover or slowdown honestly.
