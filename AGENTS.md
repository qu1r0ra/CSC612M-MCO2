# CSC612M-MCO2 agent instructions

- **Benchmark sweeps:** before running, changing, or reporting one, read `docs/benchmark-protocol.md`; its latest revision section is binding.
- **Pilot first:** before every full sweep, run `just bench-matrix --pilot` from the merged, clean tree and `just figures` on its `results/pilots/` folder. Fixes after a pilot follow the protocol's pilot rule.
- **Frozen snapshots:** folders under `results/` are evidence; write each new run to a new folder.
- **System settings** stay as the user left them; the user reboots, closes apps, and applies any setting.
