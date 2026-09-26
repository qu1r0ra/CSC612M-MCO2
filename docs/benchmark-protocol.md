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

Known limitations: case order is fixed by element count, smallest first, and path order is balanced only within a case, so GPU clock ramp-up from idle (recorded as P8 at the start of this run) falls on the smallest cases. The claim rule flags such cases as unstable rather than reporting them. An earlier six-trial run at revision `8622430` was discarded because its vectorization report failed to compile. Under the same thresholds, its claim set differed at the margins: 2^22 4-bit was unsupported because its comparator was unstable, 2^18 8-bit host-origin was unsupported, and 2^10 8-bit resident was not a supported slowdown. Cases near the thresholds should therefore be read as marginal. The size sweep's run 1 (below) shows the resident path's between-process instability is not confined to small cases.

## Size sweep (paper extension)

This protocol was fixed before the sweep ran. The course snapshot above stays the course evidence; the sweep is a separate snapshot.

- Grid: every power of two from `2^10` to `2^26` elements (17 sizes) at 4 and 8 bits, for the C comparator, CUDA resident, and CUDA host-origin paths. This is the driver default.
- Per case: 10 warmups, 30 measured repetitions, 6 trials covering every path ordering, the same statistics, and the same claim rule. The rule is unchanged from the course run.
- Case order: the 34 (size, bits) cases run in a random order drawn with `numpy.random.default_rng(612)`. The manifest records the seed and the order, and each case JSON records its `execution_index`. Path order within a case stays balanced across trials.
- GPU warm-up: before the first timed process, the driver runs the resident path untimed on the largest input for 20 seconds. Before each case's timed processes, after its correctness gate, it runs the same work on that case's input for 3 seconds, so clocks recover after the gate and after long CPU processes of the previous case. The manifest records `nvidia-smi` state before and after the initial warm-up and after the last timed process; each case JSON records its state after its own warm-up.
- Copy timing: the host-origin path records H2D and D2H times for every repetition with CUDA events. Each case also records the median of each stage, plus `other_ms`, the median of wall time minus the timed stages. The wall-clock total stays the headline measure. Stage times are diagnosis only, never summed into a path time.
- Crossover: for one path at one bit width, the crossover is the interval between two adjacent grid sizes whose verdicts differ (`slower` on one side, `faster` on the other), with `claim_supported` true on both sides (from revision 3, `direction_supported`). If no such interval exists, or more than one exists, the crossover is reported as not resolved, bracketed by the largest supported slower size and the smallest supported faster size.
- Outputs: `just figures <snapshot>` writes F1 (time vs elements, log-log, band = trial-median range), F2 (speedup vs elements with a 1× line; filled markers where `claim_supported`, hollow otherwise; from revision 3, filled for a magnitude claim and hollow for a direction claim only, faded otherwise, with a bootstrap CI), F3 (CUDA stage shares at `2^14`, `2^18`, `2^22`, `2^26`), F4 from revision 3 (`resident-graph` speedup over `resident`), and `report.md` (crossover and the T1 table) into the snapshot folder.

### Sweep result (run 1)

Snapshot `results/2026-09-25-674bd5b`, from clean revision `674bd5b`: 102 cases, all passing the correctness gates. The T1 table and crossover lines are in [its `report.md`](../results/2026-09-25-674bd5b/report.md), and the figures are [F1](../results/2026-09-25-674bd5b/f1_time_vs_elements.png), [F2](../results/2026-09-25-674bd5b/f2_speedup_vs_elements.png) and [F3](../results/2026-09-25-674bd5b/f3_stage_breakdown.png). The GPU stayed in P1 at 2,932–2,947 MHz SM with no active clock-event reasons at the start, after the warm-up, and at the end.

- Crossover: not resolved for any path or bit width. No slowdown has claim support. The smallest supported faster sizes are `2^18` (host-origin, 4-bit), `2^20` (host-origin, 8-bit), and `2^26` (resident, both widths). Pooled-median point estimates cross 1× between `2^13` and `2^14` for 4-bit resident and between `2^14` and `2^15` for host-origin at both widths; 8-bit resident rises above 1× at `2^13`, dips below at `2^14`, and stays above from `2^15`. These point estimates support no claim.
- Supported claims: resident at `2^26` only (257× 4-bit, 230× 8-bit). Host-origin at `2^18` 4-bit, `2^20` 8-bit, `2^23` 8-bit, and every case from `2^24` up (27–31×).
- Resident instability: the resident path is unstable (spread ratio above 1.25) at 32 of 34 cells, with ratios of 2–7 below `2^22`. The slow trials inflate the summed K1+K2+K3 event times, not just the wall time, so the slowdown is on the device side. They are not clustered after any particular preceding path, and they persist from the first ten repetitions of the process to the last ten, so they are neither clock ramp-up within a process nor a carry-over from the comparator. This data does not identify the cause; contention on the display GPU under WDDM and per-process power-state behavior are candidates. The course snapshot showed the same pattern less often (resident kernel medians more than 1.5× the case minimum in 6 of 48 trials, against 79 of 204 here).
- Stage shares: the scale step K1 is 62–79% of the resident stage time at every plotted size. For host-origin at `2^22` and above, H2D is 64–78% and D2H 10–20%.

Known limitations of the sweep:

- Resident between-process instability, above, limits claims to the top of the grid. The warm-ups set clocks before each case but do not hold them within or between the timed processes, and the `nvidia-smi` readings are instantaneous.
- The comparator is also flagged unstable at 6 of 34 cells (`2^11` and `2^12` at 4-bit, `2^11`, `2^15`, `2^21` at 8-bit, and `2^22` at 4-bit), which removes claim support there regardless of the CUDA path.
- Host transfers use pageable memory. Pinned buffers would shorten the H2D share that dominates host-origin time.
- The comparator is scalar and single-threaded, as the course requires. Speedups against it overstate the gain over an optimized CPU implementation.
- All results come from one RTX 5060 on Windows, which is also the display GPU.

Run 1 stays the pre-registered result. A revised protocol that targets resident stability must be recorded before its own run and produces a separate snapshot.

### Revision 2 (resident stability)

This revision was fixed before its sweep ran. Run 1 stays the pre-registered result; revision 2 is a separate snapshot.

#### Diagnosis

`scripts/diag_resident.py` runs the resident path, 8-bit, at `2^14` and `2^20` as 36 separate processes for each condition, with conditions interleaved in a shuffled order. It logs `nvidia-smi` clocks and power state every 50 ms. A process counts as slow when its kernel-event median exceeds 1.5× the fastest process of that size. The spread column groups consecutive processes in sixes, as one case's six trials would be, and counts the groups within the 1.25 limit. The evidence is in `results/diag-2026-09-26-issue-26/`; its inputs are regenerated from the fixed seed and not committed.

| Run | Condition | Slow processes `2^14` | Slow processes `2^20` | Groups within 1.25, `2^14` / `2^20` |
| --- | --- | --- | --- | --- |
| desktop as used | base (10 warmups, 30 reps) | 56% | 44% | 0/6 / 1/6 |
| desktop as used | 1 s in-process warm-up | 17% | 3% | 1/6 / 5/6 |
| quiet desktop | base | 44% | 53% | 0/6 / 0/6 |
| quiet desktop | 1 s in-process warm-up | 3% | 8% | 1/6 / 4/6 |
| quiet desktop | 3 s in-process warm-up | 11% | 3% | 1/6 / 2/6 |
| SM clock locked at 2,400 MHz | base | 28% | 39% | 1/6 / 0/6 |
| SM clock locked at 2,400 MHz | 1 s in-process warm-up | 6% | 11% | 0/6 / 2/6 |

Findings:

- The slowdown belongs to the timed process, not to the GPU state between processes. Slow processes inflate all three kernel stages (at `2^14` under the clock lock, K1 about 2.4×, K2 3.7×, K3 4.7×). They are scattered in time and do not follow any particular preceding process. Run 1's out-of-process warm-ups (20 s at the start, 3 s per case) run in other processes and do not carry into the timed one.
- Untimed repetitions inside the timed process, sized to about 1 s, cut the slow fraction from 40–55% to 3–17%. 3 s did no better than 1 s.
- Ruled out as the main cause: display and application contention (pausing the animated wallpaper and closing heavy applications changed nothing), SM clock ramping (with the SM clock held at 2,392 MHz, slow processes still inflated all stages at the same clock as fast ones), more measured repetitions (300), high process priority, and a 3 s idle gap before each process.
- Warm-up does not fully remove the slow processes. At `2^14` at most one group in six stayed within 1.25 under any condition, so a small-size resident claim remains unlikely.

The user applied the clock lock (`nvidia-smi -lgc 2400,2400`) only for that diagnosis run and reset it (`nvidia-smi -rgc`) before the revision 2 pilot, whose manifest shows 2,947 MHz SM under load. The memory clock was never locked. The agent changed no GPU, power, or system setting.

#### Change

The one change from run 1 is a time-based in-process warm-up, applied to every path, the C comparator included, so each path gets the same treatment:

- After a case's correctness gate and its 3 s GPU warm-up, the driver runs one untimed probe process per path with the base 10 warmups and 30 repetitions.
- Each path's timed processes then use `max(10, ceil(1000 / probe median ms))` warm-up repetitions, so every timed process first runs about one second of untimed work inside the same process. Paths whose repetitions take longer than 100 ms keep 10.
- Each case JSON records the target, the probe median, and the resulting count under `in_process_warmup`; its `warmup` field and the summary `warmup` column hold the per-path count. The manifest (version 2.2) records the target and the rule. Warm-up invocation identifiers are stored as `{first, last, count}` because the counts reach 10^4–10^5.
- `just bench-matrix` uses this by default; `--warmup-seconds 0` restores the run 1 behaviour.

Unchanged: the grid, 30 measured repetitions, 6 trials with balanced path order, the randomized case order and its seed, the 20 s and 3 s GPU warm-ups, the statistics, the 1.25 spread threshold, the claim rule, and the crossover rule. GPU clocks, power, and desktop settings stay at their defaults.

#### Expected outcome

The warm-up raises the chance that a case passes but does not guarantee it. In the diagnosis, a `2^20` group of six passed about three times in four with the warm-up; a crossover needs two adjacent sizes to pass together, for both the CUDA path and the comparator. A pilot on four 8-bit sizes, from an uncommitted tree and not evidence, matched this: resident failed the spread limit at `2^14` (2.68) and `2^20` (2.21), host-origin passed at `2^20` (1.17) and fell to 1.30–1.35 at `2^10` and `2^14` (2–5.8 in run 1), and the comparator failed at `2^14` (1.38). No parameter changed after the pilot.

If no path and bit width resolves a crossover under the unchanged rules, the paper reports that the resident path cannot be measured stably between processes on this machine, and makes host-origin claims only. That outcome is decided here, before the run.

### Sweep result (revision 2)

Snapshot `results/2026-09-26-2c190af` comes from clean revision `2c190af`. All 102 cases passed the correctness gates, and the sweep took 2,707 s (about 45 minutes). The T1 table and crossover lines are in [its `report.md`](../results/2026-09-26-2c190af/report.md), and the figures are [F1](../results/2026-09-26-2c190af/f1_time_vs_elements.png), [F2](../results/2026-09-26-2c190af/f2_speedup_vs_elements.png) and [F3](../results/2026-09-26-2c190af/f3_stage_breakdown.png). The manifest records the GPU state at three points, with no active clock-event reasons at any of them:

- start: P3 at 870 MHz SM (idle);
- after the warm-up: P1 at 2,940 MHz;
- end: P8 at 517 MHz.

No clock lock was in effect.

- **Crossover:** not resolved for any path or bit width, and no slowdown has claim support. The smallest supported faster sizes are:

  | Path | 4-bit | 8-bit |
  | --- | --- | --- |
  | Resident | `2^14` | `2^19` |
  | Host-origin | `2^17` | `2^19` |

- **Outcome:** the rule stated before the run applies. The paper reports that the resident path cannot be measured stably between processes on this machine, and it makes host-origin claims only.
- **Supported claims:** host-origin is faster at 4-bit from `2^17` up (7.7–32.5×) and at 8-bit from `2^19` up (17.1–29.5×). That is 18 cells, all `faster`.
- **Resident, descriptive only (no claim):** whether a size passes the spread limit does not follow from the size.
  - At 4-bit, resident passes at `2^13` and `2^14`, fails from `2^15` to `2^20` (1.28–2.50), and passes again from `2^21`.
  - At 8-bit, resident fails at `2^18` (2.63), one size below its first pass.
  - Resident passes the spread limit with claim support in 15 of 34 cells (2 in run 1). The resident speedups in those cells, 61–321× at `2^19` and above, carry no claim.
- **Effect of the warm-up:** resident trials whose kernel median exceeds 1.5× the case minimum fell from 79 of 204 in run 1 to 10 of 204. Unstable cells fell from 32 to 18 of 34 for resident and from 24 to 15 of 34 for host-origin.
- **Remaining instability is still on the device side:** in the 18 unstable resident cells, the spread of per-trial K1+K2+K3 medians tracks the wall-time spread.
  - The 9 failures above 2 come from the remaining slow trials. Their kernel spreads are 2.27–2.91.
  - Only 5 of the 18 marginal failures have a kernel spread within 1.25.
  - The warm-up makes the slowdown rarer. It does not remove it.

Known limitations, in addition to those of run 1:

- **Comparator instability:** the comparator is flagged unstable at 4 of 34 cells. This removes claim support there regardless of the CUDA path.

  | Size | Bits | Spread |
  | --- | --- | --- |
  | `2^12` | 8 | 1.26 |
  | `2^13` | 4 | 1.45 |
  | `2^14` | 8 | 1.28 |
  | `2^15` | 8 | 1.29 |

- **Diagnosis coverage:** the cause of the remaining slow processes was not identified. The diagnosis ruled out display contention, SM clock ramping, repetition count, process priority, and idle gaps, but it did not test other WDDM scheduling behaviour or memory clock behaviour.

### Revision 3 (resident claim)

This revision was fixed and merged before its pilot and its sweep ran. The run 1 and revision 2 snapshots stay as recorded; revision 3 is a separate snapshot. Spec: [qu1r0ra/CSC612M-MCO2-Paper#31](https://github.com/qu1r0ra/CSC612M-MCO2-Paper/issues/31); diagnosis: [#30](https://github.com/qu1r0ra/CSC612M-MCO2-Paper/issues/30). The decision record is [ADR 0002](adr/0002-core-0-exclusion-and-split-claims.md).

#### Diagnosis

The evidence is in [`results/trace-2026-09-26-issue-30/`](../results/trace-2026-09-26-issue-30/README.md). It ran from an uncommitted tree and supports no claim.

- **The resident path is launch-bound below about `2^21`.** Under Nsight Systems, kernel busy time was 9.0 µs per repetition in all 72 traced processes at `2^14`, 8-bit. Span per repetition ran from 136 to 920 µs, so GPU compute was under 7% of it.
- **Slow processes are slow on the CPU side, and the level is per process.** The `cudaLaunchKernel` median fell into discrete levels (about 6, 9, 30 and 48 µs). A process kept its level for its whole life, and the GPU gaps followed the API interval.
- **Physical core 0 is about 30% slower.** Untraced processes pinned to core 0 had a median of 0.184–0.188 ms, against 0.139–0.146 ms on every other core. This held on a quiet and on a busy desktop. Unpinned processes that land on core 0 form the tail that failed revision 2's spread rule.
- **Excluding core 0 removes the tail.** At `2^14`, the spread across processes fell to 1.21× on the quiet desktop and 1.22× on the busy one, with no outliers.
- **Unreproduced:** the slow launch levels seen in the traced run (spans 3–6× the fastest), which followed a long session, did not come back after a reboot, with or without heavy apps open. Their cause is unresolved.

The agent changed no GPU clock, power plan, HAGS, or other system setting for this diagnosis.

#### Changes

- **Core 0 excluded.** The driver sets its own process affinity to exclude logical CPUs 0 and 1 (physical core 0) before it starts any benchmark or probe process; children inherit it. This applies to every path, the C comparator included. The manifest records the mask.
- **Claim split.** Each comparison reports two claims in place of `claim_supported`:
  - `direction_supported`: the conservative verdict is `faster` or `slower`, and the cell has no boundary inversion.
  - `magnitude_supported`: `direction_supported`, and both sides are stable.
  - The revision 2 claim is still computed as `claim_supported_rev2`, for comparison only.
- **Percentile stability.** `stable` means `spread_p90_p10 <= 1.25`, the 90th over the 10th percentile of trial medians (`numpy` method `linear`), so one outlier process no longer vetoes a case. The max/min `spread_ratio` is still reported.
- **Bootstrap CI.** Each speedup reports a 95% percentile bootstrap interval: 10,000 resamples of each side's trial medians, independently, from `numpy.random.default_rng(31)`. Each resample's statistic is the comparator's median resampled trial median over the CUDA one. The CI is reported; no claim depends on it.
- **Graph path.** `resident-graph` captures one resident repetition (1 memset and the same K1–K3 kernel launches, geometry and buffers as `resident`; 13 operations at `2^14`, more at larger sizes as the K1 reduction deepens) as a CUDA Graph once, outside timing, and times each repetition as one graph launch plus a device synchronize. It is compared with the comparator like every other path, and with `resident` under the same rules (F4). Its one-time capture-and-instantiate cost is recorded per process.
- **24 trials.** Each case runs 24 trials, every ordering of the four paths once, so each path holds each position and follows each other path equally often. The sweep takes about 3.2 hours.
- **Direction-based crossover.** The crossover rule is unchanged except that it uses `direction_supported` on both sides in place of `claim_supported`. The report also gives the crossover under the revision 2 claim, for comparison.
- **Readiness check.** An evidence sweep refuses to start unless:
  - uptime is at most 30 minutes (a fresh reboot);
  - no app window is open outside `WINDOW_ALLOWLIST` in `benchmark_driver.py` (the Claude app and Windows shell hosts);
  - no `mco2` process is running;
  - the git tree is clean;
  - the GPU reports no clock-event reason other than `GpuIdle` (idle is not throttling).
  The manifest (version 3.0) records these facts, the power plan, and the HAGS state. `--ignore-readiness` runs anyway and marks the snapshot non-evidence.
- **Boundary inversion against every resident path.** A cell is inverted when host-origin is faster than `resident` or `resident-graph`.
- **Pilot.** `--pilot` runs the sweep's code path at `2^10`, `2^14`, `2^20` and `2^26`, both bit widths, into `results/pilots/`. The only difference from a sweep is that the readiness check is recorded, not enforced. A pilot is never evidence.

Unchanged: the grid, 30 measured repetitions, the time-based in-process warm-up (1 s), the 20 s and 3 s GPU warm-ups, the randomized case order and its seed, the pooled statistics, the conservative verdict, and the correctness gates. GPU clocks, power, and desktop settings stay at their defaults; the user, not the agent, reboots and closes apps before the sweep.

#### Pilot rule

A pilot runs from the merged tree before the sweep. After the pilot, only outright bugs may be fixed, each with a test and recorded here or in its pull request. No parameter or threshold changes after the pilot.

Fixes after the pilot at `3b578d2` (`results/pilots/2026-09-26T100830-3b578d2`, not committed):

- `just figures` failed on F2 and F4 because it drew each 95% CI as error bars relative to the point speedup. The point is a ratio of pooled medians and the CI comes from trial medians, so the point can lie outside its CI (`2^14`, 8-bit, `resident-graph`: 17.471 against [17.456, 17.460]). Each CI is now drawn as a segment between its own bounds. Test: `test_render_report_draws_a_ci_that_excludes_the_point`.

#### Pre-registered outcome

Revision 3's result is final for resident stability: whatever it supports is what the paper claims, and no revision 4 is made for stability. The paper reports, per path and bit width, which direction claims and which magnitude claims hold, the graph-vs-resident finding, both crossovers, and the graph capture cost. It reports the diagnosis and the core-0 finding either way, and discloses the unexplained 3–6× levels as a limitation.

### Sweep result (revision 3)

Snapshot `results/2026-09-26-a1d2439` comes from clean revision `a1d2439`, the merged tree after the pilot fix. The readiness check passed and was enforced, with no override, 5 minutes after a fresh reboot; only the Claude app and the Windows input host were open. The manifest records `evidence: true` and affinity mask `0xffc`. All 136 cases passed the correctness gates. The sweep ran from 11:34 to 14:22 UTC (about 2 h 48 min). The T1 table and crossover lines are in [its `report.md`](../results/2026-09-26-a1d2439/report.md), and the figures are [F1](../results/2026-09-26-a1d2439/f1_time_vs_elements.png), [F2](../results/2026-09-26-a1d2439/f2_speedup_vs_elements.png), [F3](../results/2026-09-26-a1d2439/f3_stage_breakdown.png) and [F4](../results/2026-09-26-a1d2439/f4_graph_vs_resident.png). The GPU reported no active clock-event reason at any of three readings:

- start: P8 at 405 MHz SM (idle);
- after the warm-up: P1 at 2,955 MHz;
- end: P8 at 262 MHz.

No clock lock or other system setting was in effect.

- **Claims:** every comparison has both a direction claim and a magnitude claim: 34 of 34 cells for each path against the comparator, and 34 of 34 for the graph against resident. No cell is inverted.

  | Path | Supported slower | Supported faster | Speedup range |
  | --- | --- | --- | --- |
  | Resident | `2^10`–`2^12` | `2^13`–`2^26` | 0.19–309× |
  | Resident-graph | none | `2^10`–`2^26` | 1.06–317× |
  | Host-origin | `2^10`–`2^13` | `2^14`–`2^26` | 0.10–31.3× |

- **Crossover (direction rule):** the same at both bit widths.

  | Path | Crossover |
  | --- | --- |
  | Resident | between `2^12` and `2^13` |
  | Host-origin | between `2^13` and `2^14` |
  | Resident-graph | none; faster from `2^10`, the smallest size |

  Under the revision 2 claim the crossovers are the same, except that host-origin 8-bit is unresolved (supported slower up to `2^12`, supported faster from `2^18`). The revision 2 claim fails in 2 resident and 8 host-origin cells, all on the max/min spread of the CUDA side.
- **Stability:** every case is stable. The largest `spread_p90_p10` is 1.160 for resident (`2^10`, 8-bit), 1.105 for resident-graph, 1.245 for host-origin (`2^17`, 4-bit), and 1.018 for the comparator. Host-origin's largest spread is 0.005 under the threshold.
- **Graph vs resident:** the graph is faster than resident in all 34 cells, with both claims. The speedup is 5.2–6.5× from `2^10` to `2^15`, falls to 2.0–2.1× at `2^20`, and is 1.03× at `2^26`. Up to `2^15` the graph takes 0.020–0.023 ms per repetition, against 0.105–0.141 ms for resident.
- **Graph capture cost:** the one-time capture and instantiation took a per-case median of 0.171–0.229 ms (0.158–0.369 ms over the 24 processes of each case), rising slightly with size. At `2^14` the median, 0.180 ms, equals the saving from about 1.6 repetitions (0.114 ms each). A single call with capture included would therefore be slower than resident; the timed repetitions exclude capture by design.

Known limitations:

- **Unexplained launch levels:** the slow launch levels seen before the revision 3 diagnosis (spans 3–6× the fastest, after a long session) were not reproduced, and their cause is unknown. This sweep ran right after a reboot; a desktop that has run for a long time may be slower and less stable than these results show.
- **Core 0 cause untested:** core 0 was excluded because it measured about 30% slower. Why it is slower was not tested.
- **One machine, one sweep:** every result comes from one laptop, one reboot, and one sweep, and describes that machine under those conditions.

## Comparison backends

- Build the compiled CPU and CUDA backends as one native executable: a C host driver and single-thread C comparator, with CUDA kernels reached through `extern "C"` launch functions. Use Python for the reference and analysis.
- The course sequential/parallel comparison is the single-thread C comparator against CUDA. Record compiler vectorization settings and identify a scalar configuration for that comparison.
- The scalar Python reference is the correctness oracle; its timings may appear as an additional row but never as the headline baseline. A vectorized Python implementation may be useful for comparison.

## Timing boundaries

| Boundary | CPU path | CUDA path |
| --- | --- | --- |
| Resident computation | Host input to host packed bytes | Device input to device packed bytes (`resident` / `resident-graph`) |
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
- Run each path as a separate process in several trials with balanced path order (default 24 trials, every ordering of the four paths once), and report each case's per-trial medians and spread ratio. Within-process IQR understates run-to-run variation, especially for sub-millisecond GPU paths dominated by launch and synchronization latency.

## Required provenance

Each case records seeds, invocation identifiers, code revision, build flags, hardware, transfer policy, header and payload bytes, timing boundary, and correctness status.
Preserve raw samples and configuration with the result.
From revision 3, a speedup or slowdown direction is claimed only where `direction_supported = true`, and its size only where `magnitude_supported = true`, under the claim rule in the technical contract. Every other case is reported as measured with its flags and supports no claim. Snapshots before revision 3 use `claim_supported` under the rule they recorded.
