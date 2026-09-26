# Resident-path cause trace (issue #30)

Diagnostic evidence only. The trace ran from an uncommitted working tree, so no claim cites it. Tracking issue: [qu1r0ra/CSC612M-MCO2-Paper#30](https://github.com/qu1r0ra/CSC612M-MCO2-Paper/issues/30).

## Method

`uv run python scripts/trace_resident.py --output-dir <dir> --report-dir <raw-dir> --processes 36 --warmups 10 6654 --label as-is`

- **Workload:** resident path, 8-bit, 2^14. 72 separate processes, 36 with 10 warm-ups and 36 with 6,654 (the revision 2 in-process count for this case), in shuffled order.
- **Profiler:** each process ran under Nsight Systems 2026.3.2, `nsys profile --trace=cuda --sample=none --cpuctxsw=none`, and was exported to SQLite.
- **Repetition boundaries:** each repetition starts with one memset, so the timeline splits into repetitions at each memset. There are 13 GPU operations per repetition: 1 memset and 12 kernels. Only the last 30 (timed) repetitions are kept.
- **Per-process medians** (all in `trace-as-is.json`):
  - GPU span per repetition;
  - kernel busy time;
  - gap between GPU operations;
  - `cudaLaunchKernel` call duration;
  - launch-to-start latency;
  - interval between API calls.
- **No app output:** Nsight Systems on Windows does not forward the application's stdout. Every timing therefore comes from the trace.
- **Raw data:** raw `.nsys-rep` and `.sqlite` files are not committed (about 20 MB per process).

## Findings

1. **The GPU work is constant.** Kernel busy time is 9.0 µs per repetition (range 8.9–9.1) in all 72 processes.
2. **At 2^14 the resident path is launch-bound.** Span per repetition runs from 136 to 920 µs, so GPU compute is under 7% of it. Revision 2 shows a flat resident median of about 0.10–0.20 ms from 2^10 to 2^20. The whole region below about 2^21 is therefore measuring launch cost, not kernel throughput.
3. **Slow processes are slow on the CPU side, and the level is per process.** The `cudaLaunchKernel` median falls into discrete levels. A process stays at one level for its whole life, and the GPU gaps follow the API interval.

   | Launch call median | Span per repetition | 10 warm-ups | 6,654 warm-ups |
   | --- | --- | --- | --- |
   | ~6.3 µs | ~140–175 µs | 17 | 23 |
   | ~8–10 µs | ~195–245 µs | 10 | 12 |
   | ~27–35 µs | ~460–485 µs | 3 | 1 |
   | ~47–50 µs | ~830–920 µs | 6 | 0 |

   This matches the #26 observation that whole processes run 2–3× slower with all three kernel stages inflated. The in-process warm-up removes the worst level but not the ones in between.
4. **At the slowest level, submission queueing also appears.** Launch-to-start latency rises from ~8 µs to ~47 µs.

## Interpretation (working)

Launch cost per process is fixed and CPU-side. That fits which core the process thread lands on and that core's power state. The active power plan was **Balanced** (`381b4222-…`), which allows core parking and frequency scaling. `HwSchMode` is not set in `HKLM\SYSTEM\CurrentControlSet\Control\GraphicsDrivers`, so HAGS is at the driver default; check the Settings page before toggling. A GPU-side cause is ruled out for this case.

## Operational hazards

- **Profiling pinned to one CPU or core hangs.** Pinning the profiled process (by setting the parent process affinity) to one logical CPU, or to one physical core with both SMT siblings, hung `mco2` inside the driver. Windows then reports the process as exited (`HasExited = True`), but it keeps burning a full core and cannot be killed. Only a reboot clears it. Run any pinning test without Nsight and with a per-process timeout. `trace_resident.py --cores` is kept for reference but must not be used under the profiler.
- **Earlier trace flags failed:** `--trace=osrt` is not valid on Windows, and `--trace=wddm` needs administrator rights (it was disabled with a warning).

## Untraced core-placement test (after a clean reboot)

`scripts/pin_resident.py` runs the same case (resident, 8-bit, 2^14, 6,654 warm-ups, 30 timed repetitions) without the profiler. It uses the app's own timings and a 60 s timeout on every process. Pinning works by setting the parent process affinity, which children inherit; `-1` means every core except core 0. Runs used the Balanced power plan and default HAGS, right after a reboot, with few apps open (no browsers or Discord).

| File | Condition | Processes | Median of wall medians (ms) | Range (ms) |
| --- | --- | --- | --- | --- |
| `pin-balanced.json` | core 0 | 6 | 0.185 | 0.168–0.192 |
| `pin-balanced.json` | cores 1–5, each pinned | 30 | 0.141–0.146 per core | 0.121–0.153 |
| `pin-balanced.json` | unpinned | 12 | 0.140 | 0.137–0.193 |
| `pin-not0-quiet.json` | every core except 0 | 18 | 0.139 | 0.123–0.150 |
| `pin-not0-quiet.json` | unpinned | 18 | 0.142 | 0.128–0.175 |
| `pin-core0-quiet.json` | core 0 | 6 | 0.184 | 0.157–0.189 |

Findings:

1. **Pinning without the profiler does not hang.** The hang is specific to running under Nsight.
2. **Core 0 is about 30% slower, every time.** This fits core 0 servicing most device interrupts and DPCs. Unpinned processes that land on core 0 form the tail of the unpinned distribution.
3. **Excluding core 0 removes the tail.** Spread across processes drops to 1.21× with no outliers. This is a property of our own benchmark process, not a system setting.
4. **The 3–6× levels from the traced run did not come back, even with apps open.** A second pass ran with Firefox, ChatGPT, Antigravity, Claude, RustDesk and PowerToys open (`pin-not0-busy.json`, `pin-core0-busy.json`):

   | Condition | Processes | Median (ms) | Range (ms) |
   | --- | --- | --- | --- |
   | every core except 0 | 18 | 0.144 | 0.126–0.154 |
   | unpinned | 18 | 0.142 | 0.127–0.161 |
   | core 0 | 6 | 0.188 | 0.181–0.340 |

   Open apps alone do not reproduce the large levels, and excluding core 0 keeps the spread at 1.22×. The traced run happened after a long session, so the large levels are most likely state that built up over time (such as long uptime, many earlier GPU processes, or Discord, which was not open for this pass). Their exact cause is unresolved. The practical controls are a fresh reboot before an evidence sweep, core 0 excluded, and the open apps recorded.

## Next steps (issue #30)

1. After a clean reboot, run an untraced baseline and a core-pinning test using the app's own timings, with a timeout.
2. If the result implicates CPU power management, the user switches the power plan to High performance (no reboot) and the short diagnosis is rerun.
3. If the cause is still unresolved, the user toggles HAGS (reboot) and the diagnosis is rerun.
4. Candidate structural fix for revision 3: capture the 13 per-repetition operations in a CUDA Graph, so each repetition makes one launch. The kernels and the byte parity contract are unchanged. This is a user-level decision because it changes the evaluated implementation.
5. Revision 3 statistics are unchanged from the #30 plan: direction from the conservative verdict, magnitude from a bootstrap CI of the median, and about 24 trials, pre-registered before a fresh sweep.
