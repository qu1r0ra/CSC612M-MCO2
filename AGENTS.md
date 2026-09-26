# CSC612M-MCO2 agent instructions

- Before running, changing, or reporting a benchmark sweep, read `docs/benchmark-protocol.md`; the current revision's section is binding.
- **Pilot before every full sweep:** run `just bench-matrix --pilot` from the merged, clean tree, then `just figures` on its `results/pilots/` folder, and follow the protocol's pilot rule for any fix.
- Snapshots under `results/` are frozen evidence; write new runs to new folders.
- Leave system settings as they are; the user reboots, closes apps, and applies any setting.
