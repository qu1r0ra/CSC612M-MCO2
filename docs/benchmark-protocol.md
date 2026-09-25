# Benchmark protocol

Status: accepted protocol; executable benchmark implemented with frozen course snapshot committed under `results/`.

The benchmark must expose the cost of normalization, random-number generation, rounding, packing, and required transfers.
Do not headline a CUDA speedup from Python interpreter overhead or from separately timed stages summed into a complete path.

## Course scope

The week-13 course submission requires the protocol below except these later extensions, which must not block course completion:

- The GPU-origin, host-ready timing boundary.
- The sparse input family and the synthetic collection shaped like reference-model tensors.
- Crossover analysis (locating the element count where CUDA overtakes the comparator). The course run includes only a between-process stability check; the crossover study is the size sweep below.

The in-process benchmark driver is implemented in `benchmark_driver.py` and can be reproduced with `just bench-matrix`. The course matrix evidence is the frozen snapshot `results/2026-09-25-59c8967`; its `summary.csv` is the source for every reported figure. In that snapshot, `claim_supported` holds for these cases only (speedup vs the C comparator as pooled-median point estimate [trial-median range]):

| Elements | Bits | CUDA boundary | Speedup |
| --- | --- | --- | --- |
| `2^22` | 4 | resident | 221× [206, 266] |
| `2^22` | 8 | resident | 211× [202, 226] |
| `2^22` | 4 | host-origin | 28.9× [28.1, 31.2] |
| `2^22` | 8 | host-origin | 26.1× [25.4, 26.8] |
| `2^18` | 4 | host-origin | 12.1× [11.0, 13.9] |
| `2^18` | 8 | host-origin | 11.6× [9.9, 12.9] |
| `2^10` | 8 | resident | 0.18× [0.16, 0.19] (slowdown) |

Every other case supports no claim: its verdict is inconclusive, or the case or its comparator is unstable. No boundary inversion occurred, and the vectorization report shows no vectorized comparator loop.

Known limitations: case order is fixed by element count, smallest first, and path order is balanced only within a case, so GPU clock ramp-up from idle (recorded as P8 at the start of this run) falls on the smallest cases. The claim rule flags such cases as unstable rather than reporting them. An earlier six-trial run at revision `8622430` was discarded because its vectorization report failed to compile. Under the same thresholds, its claim set differed at the margins: 2^22 4-bit was unsupported because its comparator was unstable, 2^18 8-bit host-origin was unsupported, and 2^10 8-bit resident was not a supported slowdown. Cases near the thresholds should therefore be read as marginal.

## Size sweep (paper extension)

This protocol was fixed before the sweep ran. The course snapshot above stays the course evidence; the sweep is a separate snapshot.

- Grid: every power of two from `2^10` to `2^26` elements (17 sizes) at 4 and 8 bits, for the C comparator, CUDA resident, and CUDA host-origin paths. This is the driver default.
- Per case: 10 warmups, 30 measured repetitions, 6 trials covering every path ordering, the same statistics, and the same claim rule. The rule is unchanged from the course run.
- Case order: the 34 (size, bits) cases run in a random order drawn with `numpy.random.default_rng(612)`. The manifest records the seed and the order, and each case JSON records its `execution_index`. Path order within a case stays balanced across trials.
- GPU warm-up: before the first timed process, the driver runs the resident path untimed on the largest input for 20 seconds. Before each case's timed processes, after its correctness gate, it runs the same work on that case's input for 3 seconds, so clocks recover after the gate and after long CPU processes of the previous case. The manifest records `nvidia-smi` state before and after the initial warm-up and after the last timed process; each case JSON records its state after its own warm-up.
- Copy timing: the host-origin path records H2D and D2H times for every repetition with CUDA events. Each case also records the median of each stage, plus `other_ms`, the median of wall time minus the timed stages. The wall-clock total stays the headline measure. Stage times are diagnosis only, never summed into a path time.
- Crossover: for one path at one bit width, the crossover is the interval between two adjacent grid sizes whose verdicts differ (`slower` on one side, `faster` on the other), with `claim_supported` true on both sides. If no such interval exists, or more than one exists, the crossover is reported as not resolved, bracketed by the largest supported slower size and the smallest supported faster size.
- Outputs: `just figures <snapshot>` writes F1 (time vs elements, log-log, band = trial-median range), F2 (speedup vs elements with a 1× line; filled markers where `claim_supported`, hollow otherwise), F3 (CUDA stage shares at `2^14`, `2^18`, `2^22`, `2^26`), and `report.md` (crossover and the T1 table) into the snapshot folder.

## Comparison backends

- Build the compiled CPU and CUDA backends as one native executable: a C host driver and single-thread C comparator, with CUDA kernels reached through `extern "C"` launch functions. Use Python for the reference and analysis.
- The course sequential/parallel comparison is the single-thread C comparator against CUDA. Record compiler vectorization settings and identify a scalar configuration for that comparison.
- The scalar Python reference is the correctness oracle; its timings may appear as an additional row but never as the headline baseline. A vectorized Python implementation may be useful for comparison.

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
- Element counts: `2^10`, `2^14`, `2^18`, and `2^22` for the course run; the size sweep covers every power of two to `2^26`.
- Bit widths: 4 and 8.
- Input families: dense centered values and sparse values.
- Add one pinned synthetic collection shaped like reference-model tensors without requiring training data.
- Use 10 warmups and at least 30 measured repetitions per case.
- Report median and interquartile range.
- Run each path as a separate process in several trials with balanced path order (default 6 trials, every ordering of the three paths once), and report each case's per-trial medians and spread ratio. Within-process IQR understates run-to-run variation, especially for sub-millisecond GPU paths dominated by launch and synchronization latency.

## Required provenance

Each case records seeds, invocation identifiers, code revision, build flags, hardware, transfer policy, header and payload bytes, timing boundary, and correctness status.
Preserve raw samples and configuration with the result.
A speedup or slowdown is claimed only for cases with `claim_supported = true` under the claim rule in the technical contract; every other case, including slowdowns and unstable cases, is reported as measured with its flags and supports no claim.
