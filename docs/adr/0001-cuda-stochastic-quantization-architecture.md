---
status: accepted
updated: 2026-09-23
---
# CUDA stochastic quantization architecture

Implement a published norm-scaled stochastic quantizer as a native C++/CUDA compression pipeline, with a Python reference, a single-thread compiled CPU comparator, and real 4/8-bit packed output checked by a decoder. This exposes SIMT work while keeping the course deliverable independent of full federated training and unpublished thesis results; the compiled comparator also prevents Python overhead from carrying the performance claim.

## Consequences

Complete the bounded course study first and expand toward a credible conference submission only when correctness, measurements, and prior art justify it. Measure normalization, randomness, packing, and required transfers through explicit memory-location boundaries; host-ready bytes do not establish network performance. Keep implementation and its public technical documentation in this repository.

Before implementation or benchmarking, read the public technical contract and benchmark protocol. Those tracked documents are the authoritative implementation and measurement requirements and are sufficient for public build and reproduction work. Project sequencing, course milestones, and team planning are managed separately from this public repository.

## Considered options

A full thesis integration would add training and unpublished-method dependencies before the CUDA question is answered. A broad quantizer or optimization sweep would dilute the term's correctness and measurement work. Python-only speedup comparisons would leave interpreter overhead as a major confounder. These options are excluded from the course baseline.
