# Benchmark protocol

Status: accepted protocol; executable benchmark implemented. Superseded snapshot folders were removed from the working tree under ADR 0005; their contents remain in Git history at the revisions that produced them.

**Naming note.** The revisions below were fixed before [ADR 0004](adr/0004-stoquant-naming.md) renamed the code, and their text keeps the names they were fixed with. Read each old name as its current name from this table; every rule the text states is unchanged.

| Name in the protocol text | Current name |
| --- | --- |
| `mco2` binary, process and subcommands (`mco2 bench`, `mco2 compress`, `mco2 --threads`) | `stoquant` (`build/stoquant`) |
| `benchmark_driver.py` | `src/stoquant/matrix.py` (entry point), split across `provenance`, `inputs`, `vectorization`, `host` (readiness), `design`, `runner`, `correctness`, `stats` and `oracle` |
| `bench_report.py` | `src/stoquant/report.py` |

The benchmark must expose the cost of normalization, random-number generation, rounding, packing, and required transfers.
Do not headline a CUDA speedup from Python interpreter overhead or from separately timed stages summed into a complete path.

## Course scope

The week-13 course submission requires the protocol below except these later extensions, which must not block course completion:

- The GPU-origin, host-ready timing boundary.
- The sparse input family and the synthetic collection shaped like reference-model tensors.
- Crossover analysis (locating the element count where CUDA overtakes the comparator). The course run includes only a between-process stability check; the crossover study is the size sweep below.

The in-process benchmark driver is implemented in `src/stoquant/matrix.py` and can be reproduced with `just bench-matrix`. The historical course matrix result is archived at [revision 59c8967](https://github.com/qu1r0ra/CSC612M-MCO2/tree/59c8967/results/2026-09-25-59c8967); its `summary.csv` is the source for every figure reported here. In that snapshot, `claim_supported` holds for these cases only (speedup vs the C comparator as pooled-median point estimate [trial-median range]):

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

This protocol was fixed before the sweep ran. The course snapshot above remains the course reference; the sweep is a separate snapshot.

- Grid: every power of two from `2^10` to `2^26` elements (17 sizes) at 4 and 8 bits, for the C comparator, CUDA resident, and CUDA host-origin paths. This is the driver default.
- Per case: 10 warmups, 30 measured repetitions, 6 trials covering every path ordering, the same statistics, and the same claim rule. The rule is unchanged from the course run.
- Case order: the 34 (size, bits) cases run in a random order drawn with `numpy.random.default_rng(612)`. The manifest records the seed and the order, and each case JSON records its `execution_index`. Path order within a case stays balanced across trials.
- GPU warm-up: before the first timed process, the driver runs the resident path untimed on the largest input for 20 seconds. Before each case's timed processes, after its correctness gate, it runs the same work on that case's input for 3 seconds, so clocks recover after the gate and after long CPU processes of the previous case. The manifest records `nvidia-smi` state before and after the initial warm-up and after the last timed process; each case JSON records its state after its own warm-up.
- Copy timing: the host-origin path records H2D and D2H times for every repetition with CUDA events. Each case also records the median of each stage, plus `other_ms`, the median of wall time minus the timed stages. The wall-clock total stays the headline measure. Stage times are diagnosis only, never summed into a path time.
- Crossover: for one path at one bit width, the crossover is the interval between two adjacent grid sizes whose verdicts differ (`slower` on one side, `faster` on the other), with `claim_supported` true on both sides (from revision 3, `direction_supported`). If no such interval exists, or more than one exists, the crossover is reported as not resolved, bracketed by the largest supported slower size and the smallest supported faster size.
- Outputs: `just figures <snapshot>` writes F1 (time vs elements, log-log, band = trial-median range), F2 (speedup vs elements with a 1× line; filled markers where `claim_supported`, hollow otherwise; from revision 3, filled for a magnitude claim and hollow for a direction claim only, faded otherwise, with a bootstrap CI), F3 (CUDA stage shares at `2^14`, `2^18`, `2^22`, `2^26`), F4 from revision 3 (`resident-graph` speedup over `resident`), and `report.md` (crossover and the T1 table) into the snapshot's `derived/` folder. The linked snapshots are archived in Git history with their original layout.

### Sweep result (run 1)

The size-sweep run 1 snapshot at clean revision `674bd5b` is [archived here](https://github.com/qu1r0ra/CSC612M-MCO2/tree/674bd5b/results/2026-09-25-674bd5b): 102 cases, all passing the correctness gates. The T1 table and crossover lines are in [its report](https://github.com/qu1r0ra/CSC612M-MCO2/blob/674bd5b/results/2026-09-25-674bd5b/report.md), and its figures are [F1](https://github.com/qu1r0ra/CSC612M-MCO2/blob/674bd5b/results/2026-09-25-674bd5b/f1_time_vs_elements.png), [F2](https://github.com/qu1r0ra/CSC612M-MCO2/blob/674bd5b/results/2026-09-25-674bd5b/f2_speedup_vs_elements.png) and [F3](https://github.com/qu1r0ra/CSC612M-MCO2/blob/674bd5b/results/2026-09-25-674bd5b/f3_stage_breakdown.png). The GPU stayed in P1 at 2,932–2,947 MHz SM with no active clock-event reasons at the start, after the warm-up, and at the end.

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

The resident path was measured at 8-bit, at `2^14` and `2^20`, as 36 separate processes for each condition, with conditions interleaved in a shuffled order. GPU clocks and power state were logged every 50 ms. A process counts as slow when its kernel-event median exceeds 1.5× the fastest process of that size. The spread column groups consecutive processes in sixes, as one case's six trials would be, and counts the groups within the 1.25 limit. These were diagnostic runs and support no claim; their raw outputs have been retired.

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

The user applied the clock lock (`nvidia-smi -lgc 2400,2400`) only for that diagnosis run and reset it (`nvidia-smi -rgc`) before the revision 2 pilot, whose manifest shows 2,947 MHz SM under load. The memory clock was never locked. No other GPU, power, or system setting was changed.

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

The snapshot from clean revision `2c190af` is [archived here](https://github.com/qu1r0ra/CSC612M-MCO2/tree/2c190af/results/2026-09-26-2c190af). All 102 cases passed the correctness gates, and the sweep took 2,707 s (about 45 minutes). The T1 table and crossover lines are in [its report](https://github.com/qu1r0ra/CSC612M-MCO2/blob/2c190af/results/2026-09-26-2c190af/report.md), and its figures are [F1](https://github.com/qu1r0ra/CSC612M-MCO2/blob/2c190af/results/2026-09-26-2c190af/f1_time_vs_elements.png), [F2](https://github.com/qu1r0ra/CSC612M-MCO2/blob/2c190af/results/2026-09-26-2c190af/f2_speedup_vs_elements.png) and [F3](https://github.com/qu1r0ra/CSC612M-MCO2/blob/2c190af/results/2026-09-26-2c190af/f3_stage_breakdown.png). The manifest records the GPU state at three points, with no active clock-event reasons at any of them:

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

The diagnosis ran from an uncommitted tree and supports no claim. Its raw outputs have been retired.

- **The resident path is launch-bound below about `2^21`.** Under Nsight Systems, kernel busy time was 9.0 µs per repetition in all 72 traced processes at `2^14`, 8-bit. Span per repetition ran from 136 to 920 µs, so GPU compute was under 7% of it.
- **Slow processes are slow on the CPU side, and the level is per process.** The `cudaLaunchKernel` median fell into discrete levels (about 6, 9, 30 and 48 µs). A process kept its level for its whole life, and the GPU gaps followed the API interval.
- **Physical core 0 is about 30% slower.** Untraced processes pinned to core 0 had a median of 0.184–0.188 ms, against 0.139–0.146 ms on every other core. This held on a quiet and on a busy desktop. Unpinned processes that land on core 0 form the tail that failed revision 2's spread rule.
- **Excluding core 0 removes the tail.** At `2^14`, the spread across processes fell to 1.21× on the quiet desktop and 1.22× on the busy one, with no outliers.
- **Unreproduced:** the slow launch levels seen in the traced run (spans 3–6× the fastest), which followed a long session, did not come back after a reboot, with or without heavy apps open. Their cause is unresolved.

No GPU clock, power plan, HAGS, or other system setting was changed for this diagnosis.

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
  - no application window is open other than the Windows shell (`WINDOW_ALLOWLIST` in `src/stoquant/host.py`) and the session that launched the sweep (the driver's parent processes, recorded as `launcher_processes`);
  - no `stoquant` process is running;
  - the git tree is clean;
  - the GPU reports no clock-event reason other than `GpuIdle` (idle is not throttling).
  The manifest (version 3.0) records these facts, the power plan, and the HAGS state. `--ignore-readiness` runs anyway and marks the snapshot non-evidence.
- **Boundary inversion against every resident path.** A cell is inverted when host-origin is faster than `resident` or `resident-graph`.
- **Pilot.** `--pilot` runs the sweep's code path at `2^10`, `2^14`, `2^20` and `2^26`, both bit widths, into `results/pilots/`. The only difference from a sweep is that the readiness check is recorded, not enforced. A pilot is never evidence.

Unchanged: the grid, 30 measured repetitions, the time-based in-process warm-up (1 s), the 20 s and 3 s GPU warm-ups, the randomized case order and its seed, the pooled statistics, the conservative verdict, and the correctness gates. GPU clocks, power, and desktop settings stay at their defaults. The original revision 3 preparation called for a reboot and closed apps; the current readiness amendment below supersedes that preparation rule.

#### Pilot rule

A pilot runs from the merged tree before the sweep. After the pilot, only outright bugs may be fixed, each with a test and recorded here or in its pull request. No parameter or threshold changes after the pilot.

Fixes after the pilot at `3b578d2` (`results/pilots/2026-09-26T100830-3b578d2`, not committed):

- `just figures` failed on F2 and F4 because it drew each 95% CI as error bars relative to the point speedup. The point is a ratio of pooled medians and the CI comes from trial medians, so the point can lie outside its CI (`2^14`, 8-bit, `resident-graph`: 17.471 against [17.456, 17.460]). Each CI is now drawn as a segment between its own bounds. Test: `test_render_report_draws_a_ci_that_excludes_the_point`.

#### Implementation note

Issue #37 changed how the readiness check allows the session that launched the sweep. `WINDOW_ALLOWLIST` used to name that session's application; it now holds only Windows shell hosts. A window also passes when its process is one of the driver's parent processes, which the manifest records as `launcher_processes` among the readiness facts. The manifest stays at version 3.0. The rule's intent, a freshly rebooted and quiet machine, is unchanged, and a sweep launched from the same desktop session as the revision 3 sweep passes under both forms, so this is not a new revision. The grid, thresholds, claim rules, and every other readiness condition are unchanged.

#### Pre-registered outcome

Revision 3's result is final for resident stability: whatever it supports is what the paper claims, and no revision 4 is made for stability. The paper reports, per path and bit width, which direction claims and which magnitude claims hold, the graph-vs-resident finding, both crossovers, and the graph capture cost. It reports the diagnosis and the core-0 finding either way, and discloses the unexplained 3–6× levels as a limitation.

### Sweep result (revision 3)

The snapshot from clean revision `a1d2439`, the merged tree after the pilot fix, is [archived here](https://github.com/qu1r0ra/CSC612M-MCO2/tree/a1d2439/results/2026-09-26-a1d2439). The readiness check passed and was enforced, with no override, 5 minutes after a fresh reboot. The manifest records `evidence: true` and affinity mask `0xffc`. All 136 cases passed the correctness gates. The sweep ran from 11:34 to 14:22 UTC (about 2 h 48 min). The T1 table and crossover lines are in [its report](https://github.com/qu1r0ra/CSC612M-MCO2/blob/a1d2439/results/2026-09-26-a1d2439/report.md), and its figures are [F1](https://github.com/qu1r0ra/CSC612M-MCO2/blob/a1d2439/results/2026-09-26-a1d2439/f1_time_vs_elements.png), [F2](https://github.com/qu1r0ra/CSC612M-MCO2/blob/a1d2439/results/2026-09-26-a1d2439/f2_speedup_vs_elements.png), [F3](https://github.com/qu1r0ra/CSC612M-MCO2/blob/a1d2439/results/2026-09-26-a1d2439/f3_stage_breakdown.png) and [F4](https://github.com/qu1r0ra/CSC612M-MCO2/blob/a1d2439/results/2026-09-26-a1d2439/f4_graph_vs_resident.png). The GPU reported no active clock-event reason at any of three readings:

- start: P8 at 405 MHz SM (idle);
- after the warm-up: P1 at 2,955 MHz;
- end: P8 at 262 MHz.

No clock lock or other system setting was in effect.

- **Claims:** every comparison has both a direction claim and a magnitude claim: 34 of 34 cells for each path against the comparator, and 34 of 34 for the graph against resident. No cell is inverted, and the supported ranges below are the same at 4-bit and 8-bit.

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
- **Stability:** every case is stable. The largest `spread_p90_p10` is 1.160 for resident (`2^10`, 8-bit), 1.105 for resident-graph (`2^10`, 4-bit), 1.245 for host-origin (`2^17`, 4-bit), and 1.018 for the comparator (`2^10`, 8-bit). Host-origin's largest spread is 0.005 under the threshold.
- **Graph vs resident:** the graph is faster than resident in all 34 cells, with both claims. The speedup is 5.2–6.5× from `2^10` to `2^15`, falls to 2.0–2.1× at `2^20`, and is 1.03× at `2^26`. Up to `2^15` the graph takes 0.020–0.023 ms per repetition, against 0.105–0.141 ms for resident.
- **Graph capture cost:** the one-time capture and instantiation took a per-case median of 0.171–0.229 ms (0.158–0.369 ms over the 24 processes of each case), higher at the largest sizes. At `2^14`, 4-bit, the median, 0.180 ms, equals the saving from about 1.6 repetitions (0.114 ms each). A single call with capture included would therefore be slower than resident; the timed repetitions exclude capture by design.

Known limitations (the evidence behind them is in the revision 3 [diagnosis](#diagnosis-1)):

- **Unexplained launch levels:** the slow launch levels seen before the revision 3 diagnosis (spans 3–6× the fastest, after a long session) were not reproduced, and their cause is unknown. This sweep ran right after a reboot; a desktop that has run for a long time may be slower and less stable than these results show.
- **Core 0 cause untested:** core 0 was excluded because it measured about 30% slower. Why it is slower was not tested.
- **One machine, one sweep:** every result comes from one laptop, one reboot, and one sweep, and describes that machine under those conditions.

### Publication matrix extension

This section adds paths to revision 3; it is not a new revision. The revision 3 defaults, claim rules, thresholds, readiness check, and pilot rule are unchanged, and the revision 3 snapshots and their results stand as recorded. A run with the default options measures exactly the revision 3 paths and records the same case identifiers. Spec: [qu1r0ra/CSC612M-MCO2-Paper#20](https://github.com/qu1r0ra/CSC612M-MCO2-Paper/issues/20); decision record: [ADR 0003](adr/0003-gpu-origin-boundary-and-transfer-policy.md). The technical contract holds the exact timing windows and rules.

- **Opt-in paths.** `--boundaries gpu-origin` adds the GPU-origin, host-ready boundary for both backends. `--transfer-policies pageable pinned` adds page-locked host buffers for every timed transfer. With both, a cell has nine paths:
  - `cpu-comparator`, `cuda-resident`, `cuda-resident-graph`, `cuda-host-origin` (the revision 3 paths);
  - `cuda-host-origin-pinned`;
  - `cpu-gpu-origin` and `cuda-gpu-origin`, then their `-pinned` variants.

  `resident-graph` has no GPU-origin variant: graph variants of other paths stay out of scope, as in the revision 3 spec.
- **GPU-origin.** The input starts on the device, uploaded once outside timing. The CPU path times a full input D2H into a host landing buffer and then CPU compression; the CUDA path times K1–K3 and then the packed output D2H. `d2h_ms` and the CPU path's `cpu_ms` diagnose where time goes; no claim uses them.
- **Transfer policy.** `pinned` allocates every host buffer that a timed copy touches with `cudaHostAlloc`. Resident paths and the comparator have no timed transfer and record `none`.
- **Baselines.** Every path is compared with a CPU baseline of the same transfer policy: the comparator for resident, host-origin and CPU GPU-origin paths, and CPU GPU-origin for CUDA GPU-origin. Pinned paths also report the comparison with their pageable twin (`vs_pageable`). CUDA GPU-origin also reports its ratio to the comparator (`vs_comparator`), which is descriptive and supports no claim.
- **Inversion.** CUDA boundaries nest by work: resident ⊂ GPU-origin ⊂ host-origin, checked within each policy. CPU GPU-origin must not beat the comparator. An inversion vetoes the direction and magnitude claims of every CUDA path in the inverted policy group (resident paths sit in the pageable group), as in revision 3; a CPU GPU-origin inversion vetoes that path and its CUDA twin.
- **Trial order.** Above four paths the driver uses a Williams design instead of all orderings, since 9! orderings cannot run. One design is n orders for an even number n of paths and 2n for an odd one, and `--trials` must be a multiple of it. The nine-path matrix needs a multiple of 18, so a nine-path run, pilots included, must pass `--trials` explicitly because the default 24 is rejected; the extension default for #25 is set with that sweep. The manifest (version 3.2) records `paths`, `trial_design`, `transfer_policies`, and the verified `build_stamp`.
- **Correctness.** Each path's benchmark record, under each boundary and policy, must be byte-identical to `stoquant compress` before timing.
- **Evidence status.** Issue #20 adds tests and a tiny non-evidence matrix only. Extension sweeps feed the publication matrix of issue #25 and follow the pilot rule above.

### Input families (paper extension)

`--input-family dense|sparse|model` selects the inputs of a matrix run; the default is `dense`, and a dense run records the revision 3 case identifiers and input hashes. Spec: [qu1r0ra/CSC612M-MCO2-Paper#21](https://github.com/qu1r0ra/CSC612M-MCO2-Paper/issues/21). Every family uses the same correctness gate, statistics, and claim rules; `input_provenance` and the manifest's `inputs` record how each input was made.

- **Dense.** Standard-normal FP32 values from `numpy.random.default_rng([count, 2026])`, one generator per count. Input bytes are unchanged when other counts or a different count order are requested; provenance records `[count, seed]`. Case identifiers end in `n{count}`.
- **Sparse.** The dense vector of the same seed and count, with each element set to `+0.0` independently with probability 0.9 by a Bernoulli mask from `default_rng(1021)`. The manifest records the mask seed and the target and realised zero fractions. Case identifiers end in `sparse_n{count}`.
- **Model.** Synthetic tensors with the 62 parameter shapes of ResNet-18 for CIFAR-10 (11,173,962 parameters; names follow the common CIFAR implementation, e.g. `layer2.0.shortcut.0.weight`). No training data or trained weights are used. Convolution and linear weights are He-normal, $N(0, 2/\text{fan\_in})$; biases and batch-norm parameters are $N(0, 0.01^2)$. Tensor `i` in module order uses `default_rng([2026, i])`. `--model-tensors distinct` (default) keeps the first tensor of each distinct (shape, kind), 17 tensors; `all` keeps all 62; `--model-limit N` keeps the first N. The model family ignores `--counts`. Case identifiers end in `model_{tensor name}`.
- **Snapshots.** A non-pilot run of a non-dense family writes `results/<date>-<short_rev>-<family>/`. Summary rows carry `input_family` and `input_key`. `src/stoquant/report.py` keeps revision 3 rendering unchanged and uses a separate publication renderer for extension and non-dense cases.
- **Model option guard.** `--model-tensors` and `--model-limit` are rejected with any other family.

### AVX2 comparator (paper extension)

`--backends cpu cuda cpu-avx2` adds `cpu-avx2-optimized`, a multithreaded AVX2 build of the CPU compressor, timed on the comparator's host-host boundary. Spec: [qu1r0ra/CSC612M-MCO2-Paper#22](https://github.com/qu1r0ra/CSC612M-MCO2-Paper/issues/22). It answers how much of the CUDA speedup a tuned CPU recovers; it does not replace the baseline.

- **Opt-in.** A default run has no AVX2 path and records the revision 3 paths and case identifiers. The AVX2 path runs right after the comparator.
- **Baseline.** Every claim stays against the single-thread scalar comparator. The AVX2 path's ratio to the comparator, and each CUDA path's `vs_cpu_avx2` ratio, are descriptive: `descriptive` is true and the claim fields are null. The AVX2 path never raises a boundary inversion and stays out of T1 and the crossovers; `src/stoquant/report.py` draws it as an extra F1 line and a dashed F2 line.
- **Correctness.** Its `compress` output and bench record must be byte-identical to the scalar comparator's before timing. `tests/test_quantizer_avx2.c` checks each stage against its scalar counterpart bit for bit across sizes, team sizes, zeros, subnormals, non-finite inputs, and extreme random words.
- **Build flags.** Only `native/quantizer_avx2.c` is compiled with `/O2 /fp:precise /arch:AVX2 /openmp` (`-mavx2 -fopenmp -ffp-contract=off` elsewhere); the scalar comparator keeps `/fp:strict`. Neither setting contracts `a*b+c`. Each case records both as `build_flags.comparator_c` and `build_flags.avx2_c`.
- **Vectorization.** `just vec-report` recompiles the AVX2 source with its build flags plus `/Qvec-report:2` and fails unless every loop tagged `/* avx2-hot */` is reported as vectorized and none of their inlined copies is reported as not vectorized. The tagged loops are the max|x| scan, the scale terms, the scale tree, the Philox counter setup and rounds, the threshold split and the code combine. The Philox word interleave, the 4-bit nibble pack, and the per-block bit reversal stay scalar, so the report does not show that the whole path is vector code. A run with the AVX2 path saves that report as `msvc_vectorization_report_avx2.txt`. The check is MSVC-only; the gcc build is covered by the byte-identity test alone.
- **Threads.** `stoquant --threads N` (1–256, `cpu-avx2` only) sets the team; the default is the processor count in the process affinity mask on Windows (the online count elsewhere), which the driver's pinning sets to every logical processor off physical core 0 (logical 0 and 1). Each run, case and bench line records the team size it ran, and the manifest records `cpu_avx2_threads`.
- **Trial order.** Five paths need a Williams design of 10 orders, so the default 24 trials is rejected and the run must pass `--trials` explicitly.
- **Threat.** All-core turbo and thermal limits make a multithreaded CPU time more sensitive to machine state than the single-thread comparator. The stability readings and pilot rule apply unchanged.
- **Evidence status.** Issue #22 adds the path, its tests, and a tiny non-evidence run only. Its timed evidence comes from the publication matrix of issue #25.

### K1 bandwidth baseline (issue #23)

Spec: [qu1r0ra/CSC612M-MCO2-Paper#23](https://github.com/qu1r0ra/CSC612M-MCO2-Paper/issues/23). Before any K1 change, `just k1-baseline` freezes how far the reference K1 sits from the memory system, so that a later change is judged against a measured headroom rather than an assumed one.

- **Theoretical peak.** `2 * memory_clock_khz * 1e3 * bus_width_bits / 8 / 1e9` GB/s from `cudaDeviceGetAttribute`, the DDR convention of the CUDA C++ Best Practices Guide. On the RTX 5060 it gives 448 GB/s, which matches the vendor figure for 28 Gbps GDDR7 on a 128-bit bus; the summary records both so the convention can be checked.
- **Achievable ceiling.** `build/stream_probe` reads a float32 buffer once with a grid-stride `float4` kernel at 4, 8, 16, and 32 blocks per multiprocessor. The ceiling is the best time over kept repetitions and grids (the STREAM convention), taken over sizes of at least four times L2. The probe runs after the GPU warm-up, and the GPU state is recorded around it.
- **Effective bandwidth.** The contract forces two full input reads, one for max|x| and one for the squared terms that divide by it, so K1 moves at least `8 * count` bytes. Effective bandwidth is that minimum over the pooled K1 median of the CUDA resident path.
- **Regime.** An input of at least four times L2 is `dram`, one between L2 and four times L2 is `transitional`, and a smaller one is `l2-resident`. Near L2, part of K1's second read and of the probe's repeated scans can still hit L2, so fractions of peak and ceiling, and the headroom (K1 median over the ideal time at the ceiling), are reported only for `dram` cells. Other cells record K1's launch count (`4 + log2` of the padded block count); small cells are likely launch-bound, but the baseline does not measure a launch floor. On the RTX 5060 (24 MiB L2) only the 2^25 and larger counts are `dram`, so headroom and later K1 claims are scoped to them; the probe runs to 2^27 so its ceiling is not set at the edge of its range.
- **Protocol.** Every matrix count at both bit widths, 12 processes per cell in a per-round shuffled order, each with the in-process warm-up of revision 2 and 30 repetitions, after a 20 s GPU warm-up. `compute_case_statistics` gives the pooled median, the between-process spread, and the `stable` flag of revision 3.
- **Evidence status.** The baseline snapshot is `results/<date>-<short_rev>-k1-baseline/`, run from a clean merged tree. `summary.json` records both build recipes' commands and the SHA-256 of `mco2` and `stream_probe`; the tool stops if either dry-run fails. A K1 change is kept only if an A/B run on the same tree is supported as faster under the revision 3 claim rule.
- **Baseline result.** The snapshot from clean revision `f665f6d` is [archived here](https://github.com/qu1r0ra/CSC612M-MCO2/tree/f665f6d/results/2026-09-27-f665f6d-k1-baseline). The theoretical peak is 448.0 GB/s and the probe ceiling 429.5 GB/s, set at 2^27 elements. All four `dram` cells (2^25 and 2^26 at both bit widths) are `stable`; K1 reaches 159–166 GB/s there, 35.5–37.1% of peak and 37.0–38.7% of the ceiling, a headroom of 2.58–2.70. K1 is 62–66% of the resident path's median stage time in those cells. The GPU reported no active clock-event reason at the start or end of the run.
- **Optimized variant.** `--k1 optimized` keeps the reference arithmetic and changes only its mapping. The max pass reads `float4` words grid-stride and reduces in registers with warp shuffles, leaving one partial per block instead of one per 256 elements. In the sum pass one warp builds each 256-leaf subtree: a lane sums its 8 contiguous leaves as a 3-level tree, and shuffles at offsets 1, 2, 4, 8, and 16 add the next 5 levels in the reference pairing. The levels above reduce 2048 inputs per block per launch instead of one level per launch. At 2^25 elements that is 6 launches instead of 21. Tests require byte-identical records to the reference across counts, bit widths, grids, and zero, subnormal, large, and non-finite inputs. Its speed and keep decision are the K1 A/B result below.

### K1 A/B (issue #23)

This rule was fixed and merged before its pilot and its run. `just k1-ab` compares the two K1 variants from one binary, so the only difference between the arms is `--k1`.

- **Measurand.** K1 stage time, `k1_ms`, measured by the K1 event pair on the CUDA resident path. The events bracket the whole variant dispatch, all 21 or 6 launches. The resident total and the K2 and K3 times are reported, but no decision depends on them.
- **Protocol.** Every matrix count at both bit widths, 12 processes per variant per cell, 30 repetitions each, after the in-process warm-up of revision 2 and a 20 s GPU warm-up. Cells are shuffled per process round with the baseline's seed. Within a cell the reference runs first in even rounds and the optimized K1 first in odd rounds, so each arm holds each slot 6 times. Every process runs off physical core 0, as in revision 3. The stream probe runs again so that each arm's bandwidth fraction uses this run's ceiling. The baseline snapshot, which ran without the core-0 exclusion, supplies only the headroom recorded before the change, and F5 takes its "before" from this run's reference arm.
- **Record identity.** Before any timing, the tool compresses every cell's input with both variants and stops unless the records are byte-identical. `summary.json` records both SHA-256 values per cell.
- **Claim.** Each cell compares reference over optimized with the revision 3 conservative verdict, `direction_supported`, `magnitude_supported`, and the 95% bootstrap CI. The comparison has no boundary, so there is no inversion. Claims are scoped to the `dram` cells (2^25 and 2^26 on the RTX 5060). Other cells are descriptive, and any supported slowdown among them is listed in the decision record.
- **Keep rule.** The optimized K1 is kept only if every `dram` cell is `faster` with `direction_supported`. `magnitude_supported` and the CI are reported; no decision depends on them.
- **What keeping means.** The default stays `reference`, so a default run still records the revision 3 case identifiers and timings. If the change is kept, the publication matrix passes `--k1 optimized` explicitly and records it. If it is not kept, the optimized variant stays opt-in and is documented as not supported as faster.
- **Readiness.** The readiness facts are recorded, not enforced; the machine is not rebooted for this run. The arms are interleaved and order-balanced within each process round, so machine drift affects both.
- **Pilot.** `--pilot` runs 2 processes per variant at 2^10, 2^14, 2^20, 2^25, and 2^26 into `results/pilots/`, and is never evidence. After the pilot, only outright bugs may be fixed, each with a test.
- **Outputs.** `summary.json` (cells, comparisons, decision), `processes.json`, `stream_probe.json`, and F5 (`f5_k1_stages.png`): stacked K1, K2, K3, and other stage medians for both arms at 2^14, 2^18, 2^22, 2^25, and 2^26, as a share of the reference arm's stage sum. `just k1-ab --figure <snapshot>` re-renders F5 from `summary.json`.
- **Result.** The K1 A/B snapshot from the clean merged tree at `73ffbf5` is [archived here](https://github.com/qu1r0ra/CSC612M-MCO2/tree/73ffbf5/results/2026-09-27-73ffbf5-k1-ab) with `evidence` true. Every record pair was byte-identical. The probe ceiling was 430.0 GB/s against a 448.0 GB/s peak, and the GPU held P1 at 14001 MHz memory with no active clock-event reason at both probe ends. In the four dram cells the optimized K1 is supported as faster, with both arms stable, so the magnitude claim holds as well:

  | Count | Bits | Reference K1 (ms) | Optimized K1 (ms) | Speedup | 95% CI | Optimized GB/s | Of ceiling |
  |---|---|---|---|---|---|---|---|
  | 2^25 | 4 | 1.666 | 0.672 | 2.48 | 2.47–2.50 | 399.6 | 92.9% |
  | 2^25 | 8 | 1.665 | 0.672 | 2.48 | 2.47–2.49 | 399.5 | 92.9% |
  | 2^26 | 4 | 3.206 | 1.302 | 2.46 | 2.46–2.47 | 412.4 | 95.9% |
  | 2^26 | 8 | 3.217 | 1.305 | 2.46 | 2.46–2.47 | 411.4 | 95.7% |

  The reference arm reached 161–167 GB/s, 37.5–38.9% of the ceiling, in line with the baseline. The keep rule holds over all 4 dram cells, so the optimized K1 is kept. No cell has a supported slowdown. Up to 2^13 elements five of the eight K1 verdicts are inconclusive; from 2^14 up every K1 verdict is faster. The resident total is faster in the dram cells by 1.60–1.65×; it stays descriptive. What keeping means for the default and the publication matrix is the rule above.

### Publication matrix pre-registration (issue #41)

This section fixes the publication matrix before its pilots. The revision 3 snapshot remains the earlier reference-K1 result. The three new family snapshots will be reported as measured, including unsupported directions, unresolved crossovers, and inversions; no parameter, threshold, or outcome will be selected after seeing a pilot.

- **Paths.** Each cell runs `cpu-comparator`, `cpu-avx2-optimized`, `cuda-resident`, `cuda-resident-graph`, `cuda-host-origin`, `cuda-host-origin-pinned`, `cpu-gpu-origin`, `cuda-gpu-origin`, `cpu-gpu-origin-pinned`, and `cuda-gpu-origin-pinned`. These are the nine boundary and transfer-policy paths plus the descriptive AVX2 comparator. Every CUDA invocation explicitly uses `--k1 optimized`; CPU invocations do not take a K1 option. The default driver invocation remains `reference` for revision 3 reproduction.
- **Trials and families.** Each cell has 30 trials: three complete 10-order Williams designs. Dense and sparse each use every power-of-two element count from `2^10` through `2^26` at 4 and 8 bits. The model family uses `--model-tensors distinct`, the 17 distinct ResNet-18 tensor shapes, at both bit widths. All families retain the revision 3 warm-ups, 30 measured repetitions, randomized case-order seed, correctness gates, and pooled statistics.
- **Run conditions.** Each family is one separate run from a clean merged revision. Apply the latest readiness amendment: at least 4 GiB of physical RAM available, with no reboot-age limit; ordinary app windows may remain open. The gate is enforced without override, and the machine is otherwise left undisturbed during measurement. Core 0 remains excluded (affinity mask `0xffc` on the recorded machine). Dense and sparse are estimated at 9–10 hours each, and model at 3–4 hours; the pilots will replace these estimates with measured guide durations. Each non-dense run freezes in a new `results/<date>-<rev>-<family>/` folder.
- **Pilots.** A non-evidence pilot for each family runs from the merged tree before its full run, with `--trials 30` explicitly selected for the ten paths and `just figures` rendered on its pilot snapshot. A pilot records readiness without enforcing it and never supports a paper claim. After a pilot, only an outright bug may be fixed, with a regression test and a recorded repair; changes to the design or thresholds require a new pre-registration and new pilots.
- **Claims.** The revision 3 conservative verdict, boundary-inversion veto, `direction_supported`, `magnitude_supported`, and `spread_p90_p10 <= 1.25` on both sides apply unchanged. Every speedup reports its 95% bootstrap CI. Pinned paths are compared with their pageable twin. Pinned host-origin uses the scalar comparator as baseline; pinned CUDA GPU-origin uses pinned CPU GPU-origin. CUDA GPU-origin also has a descriptive comparison with the scalar comparator. The AVX2 path and its ratios are descriptive, with null claim fields. Every inversion and its vetoed claims are listed. The published result is whatever these rules support, including a result with no supported claim.

- **Publication rendering.** `just figures <matrix-snapshot> --k1-ab <clean-k1-ab-snapshot>` writes F1–F4, an appendix K1 bandwidth plot, and `report.md`. F1 draws every recorded path and reserves a blank second-platform row; `--second-platform <snapshot>` fills that row when a result exists. F2 uses each case's recorded baseline and marks supported magnitude, supported direction, and unsupported directions separately. The report lists each input and path, the pinned-versus-pageable comparisons, every boundary inversion and its veto, and an input-family count table. Dense and sparse grids also get direction-based crossover lines. The bandwidth plot derives device peak from the matrix device attributes and the measured DRAM ceiling from the specified K1 A/B stream probe; its resident K1 and whole-pipeline curves come from the matrix cases. The original revision 3 renderer and its outputs remain byte-identical.

### Readiness amendment (2026-10-01)

This amendment supersedes the application-window, app-closing, reboot-age, and percentage-of-RAM requirements for runs after this date. It changes only the machine readiness gate; the measurement paths, timings, trial design, correctness checks, and claim rules remain fixed. Earlier snapshots retain the readiness facts and protocol version under which they were collected.

- **Before an evidence sweep:** require at least 4 GiB of available physical RAM; run no other `stoquant` process; use a clean Git tree; and require the GPU to report no active clock-event reason other than `GpuIdle`. The driver records total and available physical RAM and the measured use percentage in the manifest, sampling immediately before launch because available memory changes over time. It fails closed if the memory probe cannot provide valid data. Open application windows, including Settings and the terminal running the sweep, are permitted. Uptime is recorded for context but has no pass/fail limit. Power plan and HAGS remain recorded context, not readiness conditions.
- **Paired ticket pilots:** before launching each arm, the maintainer checks the 4 GiB RAM floor and remaining readiness conditions, then runs both arms in the same session. Apps need not be closed, and no reboot-age limit applies. Pilots record readiness without enforcing it and are never evidence; the manual preflight ensures both arms start under the intended conditions.
- **Machine settings:** GPU clocks, power plan, HAGS, and other system settings stay as the user left them. No reboot or app closing is required solely to satisfy readiness.
- **Manifest:** this readiness metadata change is recorded as manifest version 3.3. It does not alter prior snapshots or paper claims.

## Comparison backends

- Build the compiled CPU and CUDA backends as one native executable: a C host driver and single-thread C comparator, with CUDA kernels reached through `extern "C"` launch functions. Use Python for the reference and analysis.
- The course sequential/parallel comparison is the single-thread C comparator against CUDA. Record compiler vectorization settings and identify a scalar configuration for that comparison.
- The opt-in multithreaded AVX2 comparator (above) is an additional, descriptive CPU row; the scalar comparator stays the headline baseline.
- The scalar comparator keeps two passes over the input that a fused implementation would drop. Its scale step scans the input for finiteness while it takes the maximum, and its encoder then scans it again for finiteness (`sq_validate_input`) and for the first nonzero element (which stops at that element). Both passes are inside the timed comparator interval, so the baseline is a little slower than a fused scalar loop and speedups against it are correspondingly a little larger. Neither is timed separately.
- The scalar Python reference is the correctness oracle; its timings may appear as an additional row but never as the headline baseline. A vectorized Python implementation may be useful for comparison.

### Scalar comparator RNG implementation revision (issue #72, 2026-10-01)

The scalar comparator evaluates Philox4x32-10 once for each group of up to four consecutive elements, then stores lanes 0 through 3 in element order. Element `i` still uses lane `i mod 4` from counter group `floor(i / 4)`; a final partial group writes only its remaining lanes. This removes repeated evaluation of the same four-word block without changing the random stream, compressed record, or decoded output. The scale finiteness scan and the encoder's finiteness and first-nonzero scans remain in the timed comparator path and stay disclosed above. This is an implementation revision only; it does not change timing boundaries, readiness checks, manifest schema, or claim rules.

## Timing boundaries

| Boundary | CPU path | CUDA path |
| --- | --- | --- |
| Resident computation | Host input to host packed bytes | Device input to device packed bytes (`resident` / `resident-graph`) |
| GPU-origin, host-ready output | Full input D2H, then CPU compression | CUDA compression, then packed output D2H |
| Host-origin, host-ready output | CPU compression | Input H2D, CUDA compression, packed output D2H |

Use synchronized wall-clock timing for complete paths and CUDA events for device-stage diagnosis.
Keep allocation, warmup, transfer-buffer type, synchronization policy, process startup, and file-I/O treatment explicit.
Verify decoding outside the measured compression interval.

Timed CUDA repetitions do not copy or inspect the device status or the validation flags. The untimed preflight run and, for `resident-graph`, the readback after timing establish that the result is correct; a wrong result inside a timed repetition that raises no CUDA error is not detected there. The exact contents of each wall window are in the [technical contract](technical-contract.md#in-process-benchmark-timing-paths-and-boundaries).

## Matrix

- Main platform: the local RTX 5060, subject to fresh inventory and successful build verification.
- Element counts: `2^10`, `2^14`, `2^18`, and `2^22` for the course run; the size sweep covers every power of two to `2^26`.
- Bit widths: 4 and 8.
- Input families: dense centered values for the course run; sparse values and a pinned synthetic collection shaped like reference-model tensors, without training data, as the paper extension above.
- Use 10 warmups and at least 30 measured repetitions per case.
- Report median and interquartile range.
- Run each path as a separate process in several trials with balanced path order (default 24 trials, every ordering of the four paths once; a Williams design for the extension's larger path sets), and report each case's per-trial medians and spread ratio. Within-process IQR understates run-to-run variation, especially for sub-millisecond GPU paths dominated by launch and synchronization latency.

## Required provenance

Each case records seeds, invocation identifiers, code revision, build flags, hardware, transfer policy, header and payload bytes, timing boundary, and correctness status.
Preserve raw samples and configuration with the result.
From revision 3, a speedup or slowdown direction is claimed only where `direction_supported = true`, and its size only where `magnitude_supported = true`, under the claim rule in the technical contract. Every other case is reported as measured with its flags and supports no claim. Snapshots before revision 3 use `claim_supported` under the rule they recorded.
