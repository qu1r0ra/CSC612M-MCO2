# CSC612M-MCO2 agent instructions

- **Benchmark sweeps:** before running, changing, or reporting one, read `docs/benchmark-protocol.md`; its latest revision section is binding.
- **Pilot first:** before every full sweep, run `just bench-matrix --pilot` from the merged, clean tree and `just figures` on its `results/pilots/` folder. Fixes after a pilot follow the protocol's pilot rule.
- **Snapshots:** read `docs/adr/0005-snapshot-lifecycle.md` before changing run creation or derived outputs; raw runs publish once from a partial folder, and reports and figures regenerate in `derived/`.
- **System settings** stay as the user left them; the user reboots, closes apps, and applies any setting.
- **Before committing,** run `just format`; `just verify` is the CI lint gate. Run `uv run pre-commit install` once per clone so the fast fixers run on commit.
- **Line endings:** `.gitattributes` pins text to LF; generated reports use LF on every platform.
