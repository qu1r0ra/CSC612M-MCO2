---
status: accepted
updated: 2026-10-06
---
# Keep the implementation and course submission repositories separate

The public implementation repository is `qu1r0ra/stochastic-quantization-backends` and owns the complete implementation, tests, benchmark tooling, and engineering history. `qu1r0ra/CSC612M-MCO2-Code` is the course-scoped submission copy; code is staged there from the implementation repository, and each imported code state records its upstream commit so the submitted implementation can be checked for drift. `CSC612M-MCO2-Paper` remains the private repository for planning, manuscript, research coordination, and the sole project tracker. The code package, command, and binary remain `stoquant`, and the record magic remains `MSQ1`, as decided in ADR 0004.

## Consequences

- Keep the old `qu1r0ra/CSC612M-MCO2` repository name unclaimed after the rename so GitHub continues redirecting existing clone and web URLs to the implementation repository.
- Use the implementation repository as the canonical source when updating paper links and recording implementation provenance. The course copy may lag while course deliverables are staged.
