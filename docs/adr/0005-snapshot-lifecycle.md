---
status: accepted
updated: 2026-09-29
---
# Write-once snapshots, derived outputs, and deletion of superseded evidence

A snapshot under `results/` is the raw record of one run: its case JSONs, manifest, summary CSV, and vectorization reports, which are the data files the run itself writes. The run writes these once, into a partial folder that is renamed to its final name only when the run succeeds. After that, nothing rewrites them. Figures and `report.md` are derived outputs, including the figures the unbiasedness and K1 tools draw at the end of their own runs. They go in a `derived/` subfolder of the snapshot, they can be regenerated from the raw files at any time, and regenerating them replaces only `derived/`. Once the pipeline that produced a snapshot is superseded, the snapshot is deleted from the tree, and git history is its archive. A single snapshot-store module owns naming, the exists and dirty gates, partial-then-rename creation, the `derived/` location, and progress queries. Every evidence writer (matrix, unbiasedness, `k1-baseline`, `k1-ab`, pilots) goes through it. This replaces the "Frozen snapshots" rule in the repository `AGENTS.md`. The spec is issue #53, and the audit that motivated it is issue #52.

## Consequences

- **One owner for write-once.** The old rule had about five owners: separate naming, exists checks, and dirty checks in the matrix driver, the unbiasedness tool, and the K1 experiment tools, plus the equivalence tool's `results/` fingerprint and the notifier's layout parsing. `--allow-existing` could bypass the rule. `--allow-existing` is removed, and a writer that cannot get a fresh snapshot from the store cannot write evidence.
- **Failed runs leave no half snapshot.** A sweep that fails leaves a partial folder, never one with the final name. That removes the temptation to finish a failed run in place, which is how a snapshot would come to mix two code states.
- **Raw bytes stay fixed.** Report and figure changes (new plots, corrected labels, revised statistics text) regenerate `derived/`, and the raw files keep their bytes. A reader can trust that the case JSONs and manifest are exactly what the timed run produced.
- **Only current evidence is in the tree.** A snapshot is deleted in the change that supersedes the pipeline that produced it, even if its replacement lands later; the tree may briefly hold no evidence for an experiment. The paper cites only snapshots that exist at the paper's pinned revision. Deleted snapshots stay retrievable through git history at the revision where they last existed.
- **Tests do not depend on evidence.** Golden tests read fixtures under the test tree, never a folder under `results/`, so deleting a superseded snapshot never breaks the suite.

## Considered options

- **Keep every snapshot forever.** Rejected: issue #52 found a comparator making redundant RNG calls and input seeding that depended on the size set, both of which affect earlier results, and a CUDA status path that could report success on failure, which earlier runs cannot prove they never hit. Keeping those snapshots beside the corrected ones invites citing the wrong one. Git history already preserves them without that risk.
- **Rewrite `report.md` and figures inside the raw folder, as before.** Rejected: that makes "frozen" mean "frozen except for some files." No gate can then tell a legitimate regeneration from an accidental overwrite of raw evidence.
- **Put derived outputs in a separate top-level tree.** Rejected: keeping `derived/` inside its snapshot keeps each claim's raw and rendered evidence in one folder, and deleting a superseded snapshot removes both together.
- **Keep `--allow-existing` for resuming interrupted sweeps.** Rejected: a resumed sweep mixes processes from different sessions and possibly different system states in one snapshot. An interrupted sweep is rerun into a fresh snapshot.
