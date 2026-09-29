# CSC612M-MCO2 agent instructions

- **Benchmark sweeps:** before running, changing, or reporting one, read `docs/benchmark-protocol.md`; its latest revision section is binding.
- **Pilot first:** before every full sweep, run `just bench-matrix --pilot` from the merged, clean tree and `just figures` on its `results/pilots/` folder. Fixes after a pilot follow the protocol's pilot rule.
- **Frozen snapshots:** folders under `results/` are evidence; write each new run to a new folder.
- **System settings** stay as the user left them; the user reboots, closes apps, and applies any setting.
- **Before committing,** run `just format`; `just verify` is the CI lint gate. Run `uv run pre-commit install` once per clone so the fast fixers run on commit.
- **Line endings:** on Windows the working tree is CRLF and the index is LF, so a script that edits tracked files reads and writes bytes and keeps each file's existing ending; `git ls-files --eol` shows both.
