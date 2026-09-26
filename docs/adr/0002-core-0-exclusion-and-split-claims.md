---
status: accepted
updated: 2026-09-26
---
# Core 0 exclusion, split claims, and percentile stability

Every benchmark and probe process runs off physical core 0. Each speedup reports a direction claim and a magnitude claim separately. Stability is judged on the 90th over the 10th percentile of trial medians. These are protocol revision 3 choices, fixed before its sweep; the benchmark protocol's Revision 3 section holds the evidence.

## Consequences

- **Core 0 exclusion.** The issue #30 diagnosis found that the resident path is launch-bound below about `2^21`, and that processes on physical core 0 ran about 30% slower, on a quiet and on a busy desktop. The cause is untested; interrupt servicing on core 0 is one hypothesis. Unpinned processes that landed there were the tail that failed revision 2's stability rule. Excluding logical CPUs 0 and 1 changes only where our own processes run, not any system setting, and it applies to every path, the comparator included, so the comparison stays fair. Do not remove it to "use all cores": the result would again depend on the scheduler's placement.
- **Split claims.** A direction claim (`faster` or `slower`) needs only the conservative verdict and no boundary inversion; a magnitude claim also needs both sides stable. Revision 2 gated every claim on stability, so an unstable spread hid directions that the conservative range already settled. Keep both: a reader can trust the direction wherever it is claimed, and the size only where it is claimed.
- **Percentile spread.** `spread_p90_p10` replaces max/min for `stable`, keeping the 1.25 threshold. With 24 trials, max/min lets a single outlier process veto a case; the 10th–90th percentile ratio tolerates about two slow processes at each end. The max/min ratio is still reported, and the revision 2 claim is still computed as `claim_supported_rev2`.

## Considered options

- **Locking GPU clocks or changing the power plan.** Rejected: the variation is CPU-side and per process, an SM clock lock in the issue #26 diagnosis did not remove it, and these are system settings outside the benchmark's control, left as the user set them.
- **Pinning each process to one core.** Rejected: processes pinned to any one of cores 1–5 ran at 0.141–0.146 ms, no better than the 0.139 ms of processes that only excluded core 0, so pinning adds a choice of core without narrowing the spread.
- **Raising or dropping the stability threshold.** Rejected: that would tune the rule to the data. The threshold stays at 1.25; only the statistic it applies to changes.
- **Gating claims on the bootstrap CI.** Rejected: the CI is reported for each speedup, but the conservative range already gives a direction without distributional assumptions.
