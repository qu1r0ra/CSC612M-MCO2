---
status: superseded by ADR-0006
updated: 2026-10-06
---
# The code is named `stoquant`; the repository and record magic keep their names

The course codename `mco2` said nothing about what the code does, so it is replaced by `stoquant` everywhere in the code. That covers the Python distribution and import package, the console command, the binary, and the environment variables. The C API prefix becomes `sq_`/`SQ_`. The repository stays `CSC612M-MCO2`, and the record header magic stays `MSQ1`, because renaming either would cost more than the name is worth.

## Consequences

- **The repository name and code name differ.** The repository name tells the course grader what it is. A paper artifact under double-blind review would be anonymized anyway, so a public URL isn't cited during review. Rename the repository only if the paper goes public, and rely on GitHub's redirect for old links.
- **`MSQ1` stays.** The magic is part of the record format frozen by the technical contract, and the frozen snapshots under `results/` were measured against that format. Changing it would alter the frozen record format for a cosmetic gain.
- **The whole codebase is renamed at once.** A partial rename would keep two names alive indefinitely. The rename also drops the `Q8` infix from C identifiers that apply to both 4-bit and 8-bit records (`sq_status`, `SQ_ERR_*`); 8-bit-only names keep it (`SQ_Q8_BITS`, `sq_q8_make_codes`). The RNG status codes become `SQ_RNG_*` so they stay distinct from the quantizer's `SQ_OK` and `SQ_ERR_*`. Byte-identical record output for fixed seeds guards the rename.
- **Provenance changes in new records.** Manifests written after the rename record the new binary, source paths, build flag, and readiness fact name; [Reproduction path](../reproduction.md) lists them. Compare such values within one side of the rename.

## Considered options

- **Keep `mco2` as the code name.** Rejected: it ties the code's identity to a course deliverable number.
- **`stochastic_quantization` or `msq`.** Rejected: the first is long in every import and binary name. The second is opaque, and the repository never defines what "MSQ" stands for.
- **Rename only the Python package.** Rejected: the binary, environment variables and C API would still carry the old name.
