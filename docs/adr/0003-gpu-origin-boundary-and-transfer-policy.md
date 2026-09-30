---
status: accepted
updated: 2026-09-29
---
# GPU-origin boundary, pinned transfers, and policy-matched baselines

The publication matrix adds a GPU-origin, host-ready boundary for both backends and a pinned transfer policy as opt-in paths on top of protocol revision 3, not as a new revision. Each path is compared with a CPU baseline of the same transfer policy, and path orders above four paths follow a Williams design. The benchmark protocol's "Publication matrix extension" section and the technical contract hold the rules; issue #20 holds the spec.

## Consequences

- **Addition, not revision 4.** Revision 3 pre-registered that no revision 4 is made for stability. The extension changes no default, threshold, or claim rule: a run without the new options measures the revision 3 paths with the same case identifiers, and the revision 3 snapshots stand. Do not fold the extension into the defaults; `bench_report.py` indexes only pageable revision 3 paths, so F1–F4 and T1 keep their meaning.
- **CPU GPU-origin as the baseline for CUDA GPU-origin.** When the input starts on the device, a CPU compressor must first download it. Comparing CUDA GPU-origin with the host-host comparator would credit the CUDA path with a copy the CPU path was never charged for. The ratio to the comparator is still reported, marked descriptive, with no claim flags.
- **Every timed-transfer host buffer pinned.** Under `pinned`, the payload, scale, status, validation-flag and landing buffers, and a page-locked copy of the host-origin input, all use `cudaHostAlloc`. Pinning only some of them would mix both policies in one measurement. Pinned paths report both the policy-matched baseline and the comparison with their pageable twin, so the pinning gain is visible on its own.
- **Nested inversion.** Within each policy, resident ⊂ GPU-origin ⊂ host-origin by the work each does, and CPU GPU-origin contains the comparator's work. A path that beats one it contains vetoes the claims of its group, as host-origin below resident did in revision 3.
- **Stage times are diagnosis only.** `d2h_ms` and `cpu_ms` show where time goes. As with revision 3's stage medians, no claim is made from separately timed stages.
- **Williams design above four paths.** 9! orderings cannot run, and cycling through a subset would unbalance positions. A Williams design balances position and immediate predecessor in 18 orders for nine paths, so `--trials` must be a multiple of the design length. The manifest records the design; its version is 3.1.

## Considered options

- **A new protocol revision.** Rejected: it would reopen revision 3's pre-registered outcome for paths that revision 3 never measured.
- **The comparator as the only baseline.** Rejected for CUDA GPU-origin for the reason above; kept for every path whose input starts on the host or on a resident device.
- **A GPU-origin variant of `resident-graph`.** Rejected: the revision 3 spec put graph variants of other paths out of scope, and the graph path answers the launch-overhead question on its own.
- **Random orders or a subset of permutations.** Rejected: neither guarantees that each path holds each position and follows each other path equally often.

## Addendum: build stamp (2026-09-29)

Every native build writes a stamp file beside its binary. The stamp records:

- the full revision and whether the tree was clean;
- the recipe that ran and the exact compiler flags it passed, including any defines;
- the SHA-256 of the binary.

The manifest version is 3.2 with the `build_stamp` field. Version 3.1 was the earlier publication-extension schema.

The binary locator reads the stamp and returns the binary together with its verified build facts, or refuses. It refuses when:

- the stamp is missing;
- the hash does not match the binary;
- the revision is not HEAD;
- the recipe cannot serve the requested paths (for example, a CPU-only build for a CUDA path);
- the flags include a test-only define, such as CUDA fault injection.

The manifest records the verified stamp, so each timing ties back to one exact binary. Issue #53 holds the spec; this lands in its batch B.

- **Why here.** Before this, the manifest stamped the current revision on whatever binary sat in the build folder. A binary left over from an earlier revision, or one built by `build-cpu`, could be stamped as current, and nothing would show it.
- **Considered: record only the binary's hash.** Rejected: a hash identifies a binary but not the source or flags it came from.
- **Considered: rebuild inside every driver run.** Rejected: it puts the compiler into the measurement session and couples the drivers to the toolchain. A verified stamp gives the same guarantee without either cost.
- **Considered: compile the revision into the binary.** Rejected: a per-commit define changes the compiler's inputs at every revision. The object files, and the SASS identity check that timed refactors rely on, would then differ even when the code has not changed.
