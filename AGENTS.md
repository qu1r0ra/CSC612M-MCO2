# CSC612M-MCO2 agent instructions

- Before running, changing, or reporting a benchmark sweep, read `docs/benchmark-protocol.md`; the current revision's section is binding.
- **Pilot before every full sweep.** Run `just bench-matrix --pilot` from the merged, clean tree, then `just figures` on its `results/pilots/` folder, and check that every path passes byte parity and every field, figure, and report renders. After the pilot, fix only outright bugs: each fix gets a test and a line in the protocol or its pull request. Parameters and thresholds stay as pre-registered.
- Snapshots under `results/` are frozen evidence; write new runs to new folders.
- The agent leaves GPU clocks, the power plan, HAGS, and other system settings at their current state; the user reboots, closes apps, and applies any setting.
