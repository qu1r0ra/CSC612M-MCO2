# Publication matrix report (dense)

Revision `abc1234`; 40 cases; all passed: True.

## Input family

| Family | Inputs | Cells | Distinct counts |
|---|---:|---:|---:|
| dense | 2 | 40 | 2 |

### Input provenance

| Input | Elements | Source detail |
|---|---:|---|
| n1024 | 1024 | dense input |
| n4096 | 4096 | dense input |

## Path comparisons

GPU-origin CUDA uses the policy-matched CPU GPU-origin baseline. Other CUDA paths use the scalar C comparator. AVX2 and cross-boundary GPU-origin comparisons are descriptive.

| Input | Elements | Bits | Path | Policy | Baseline | Median ms | Speedup and CI | Inversion |
|---|---:|---:|---|---|---|---:|---|---|
| n1024 | 1024 | 4 | cpu-avx2-optimized | none | cpu-comparator | 1.1 | 0.909× [0.864, 0.955]; faster; direction=True; magnitude=True | False |
| n1024 | 1024 | 4 | cpu-comparator | none | cpu-comparator | 1 | — | False |
| n1024 | 1024 | 4 | cpu-gpu-origin | pageable | cpu-comparator | 1.6 | 0.625× [0.594, 0.656]; faster; direction=True; magnitude=True | False |
| n1024 | 1024 | 4 | cpu-gpu-origin-pinned | pinned | cpu-comparator | 1.8 | 0.556× [0.528, 0.583]; faster; direction=True; magnitude=True | False |
| n1024 | 1024 | 4 | cuda-gpu-origin | pageable | cpu-gpu-origin | 1.7 | 0.588× [0.559, 0.618]; faster; direction=True; magnitude=True | False |
| n1024 | 1024 | 4 | cuda-gpu-origin-pinned | pinned | cpu-gpu-origin | 1.9 | 0.526× [0.5, 0.553]; faster; direction=True; magnitude=True | False |
| n1024 | 1024 | 4 | cuda-host-origin | pageable | cpu-comparator | 1.4 | 0.714× [0.679, 0.75]; faster; direction=True; magnitude=True | False |
| n1024 | 1024 | 4 | cuda-host-origin-pinned | pinned | cpu-comparator | 1.5 | 0.667× [0.633, 0.7]; faster; direction=True; magnitude=True | True |
| n1024 | 1024 | 4 | cuda-resident | none | cpu-comparator | 1.2 | 0.833× [0.792, 0.875]; faster; direction=True; magnitude=True | False |
| n1024 | 1024 | 4 | cuda-resident-graph | none | cpu-comparator | 1.3 | 0.769× [0.731, 0.808]; faster; direction=True; magnitude=True | False |
| n1024 | 1024 | 8 | cpu-avx2-optimized | none | cpu-comparator | 1.1 | 0.909× [0.864, 0.955]; faster; direction=True; magnitude=True | False |
| n1024 | 1024 | 8 | cpu-comparator | none | cpu-comparator | 1 | — | False |
| n1024 | 1024 | 8 | cpu-gpu-origin | pageable | cpu-comparator | 1.6 | 0.625× [0.594, 0.656]; faster; direction=True; magnitude=True | False |
| n1024 | 1024 | 8 | cpu-gpu-origin-pinned | pinned | cpu-comparator | 1.8 | 0.556× [0.528, 0.583]; faster; direction=True; magnitude=True | False |
| n1024 | 1024 | 8 | cuda-gpu-origin | pageable | cpu-gpu-origin | 1.7 | 0.588× [0.559, 0.618]; faster; direction=True; magnitude=True | False |
| n1024 | 1024 | 8 | cuda-gpu-origin-pinned | pinned | cpu-gpu-origin | 1.9 | 0.526× [0.5, 0.553]; faster; direction=True; magnitude=True | False |
| n1024 | 1024 | 8 | cuda-host-origin | pageable | cpu-comparator | 1.4 | 0.714× [0.679, 0.75]; faster; direction=True; magnitude=True | False |
| n1024 | 1024 | 8 | cuda-host-origin-pinned | pinned | cpu-comparator | 1.5 | 0.667× [0.633, 0.7]; faster; direction=True; magnitude=True | True |
| n1024 | 1024 | 8 | cuda-resident | none | cpu-comparator | 1.2 | 0.833× [0.792, 0.875]; faster; direction=True; magnitude=True | False |
| n1024 | 1024 | 8 | cuda-resident-graph | none | cpu-comparator | 1.3 | 0.769× [0.731, 0.808]; faster; direction=True; magnitude=True | False |
| n4096 | 4096 | 4 | cpu-avx2-optimized | none | cpu-comparator | 1.1 | 0.909× [0.864, 0.955]; faster; direction=True; magnitude=True | False |
| n4096 | 4096 | 4 | cpu-comparator | none | cpu-comparator | 1 | — | False |
| n4096 | 4096 | 4 | cpu-gpu-origin | pageable | cpu-comparator | 1.6 | 0.625× [0.594, 0.656]; faster; direction=True; magnitude=True | False |
| n4096 | 4096 | 4 | cpu-gpu-origin-pinned | pinned | cpu-comparator | 1.8 | 0.556× [0.528, 0.583]; faster; direction=True; magnitude=True | False |
| n4096 | 4096 | 4 | cuda-gpu-origin | pageable | cpu-gpu-origin | 1.7 | 0.588× [0.559, 0.618]; faster; direction=True; magnitude=True | False |
| n4096 | 4096 | 4 | cuda-gpu-origin-pinned | pinned | cpu-gpu-origin | 1.9 | 0.526× [0.5, 0.553]; faster; direction=True; magnitude=True | False |
| n4096 | 4096 | 4 | cuda-host-origin | pageable | cpu-comparator | 1.4 | 0.714× [0.679, 0.75]; faster; direction=True; magnitude=True | False |
| n4096 | 4096 | 4 | cuda-host-origin-pinned | pinned | cpu-comparator | 1.5 | 0.667× [0.633, 0.7]; faster; direction=True; magnitude=True | False |
| n4096 | 4096 | 4 | cuda-resident | none | cpu-comparator | 1.2 | 0.833× [0.792, 0.875]; faster; direction=True; magnitude=True | False |
| n4096 | 4096 | 4 | cuda-resident-graph | none | cpu-comparator | 1.3 | 0.769× [0.731, 0.808]; faster; direction=True; magnitude=True | False |
| n4096 | 4096 | 8 | cpu-avx2-optimized | none | cpu-comparator | 1.1 | 0.909× [0.864, 0.955]; faster; direction=True; magnitude=True | False |
| n4096 | 4096 | 8 | cpu-comparator | none | cpu-comparator | 1 | — | False |
| n4096 | 4096 | 8 | cpu-gpu-origin | pageable | cpu-comparator | 1.6 | 0.625× [0.594, 0.656]; faster; direction=True; magnitude=True | False |
| n4096 | 4096 | 8 | cpu-gpu-origin-pinned | pinned | cpu-comparator | 1.8 | 0.556× [0.528, 0.583]; faster; direction=True; magnitude=True | False |
| n4096 | 4096 | 8 | cuda-gpu-origin | pageable | cpu-gpu-origin | 1.7 | 0.588× [0.559, 0.618]; faster; direction=True; magnitude=True | False |
| n4096 | 4096 | 8 | cuda-gpu-origin-pinned | pinned | cpu-gpu-origin | 1.9 | 0.526× [0.5, 0.553]; faster; direction=True; magnitude=True | False |
| n4096 | 4096 | 8 | cuda-host-origin | pageable | cpu-comparator | 1.4 | 0.714× [0.679, 0.75]; faster; direction=True; magnitude=True | False |
| n4096 | 4096 | 8 | cuda-host-origin-pinned | pinned | cpu-comparator | 1.5 | 0.667× [0.633, 0.7]; faster; direction=True; magnitude=True | False |
| n4096 | 4096 | 8 | cuda-resident | none | cpu-comparator | 1.2 | 0.833× [0.792, 0.875]; faster; direction=True; magnitude=True | False |
| n4096 | 4096 | 8 | cuda-resident-graph | none | cpu-comparator | 1.3 | 0.769× [0.731, 0.808]; faster; direction=True; magnitude=True | False |

## Transfer policy

Pinned is compared with its pageable twin under the recorded claim rule.

| Input | Bits | Path | Pinned vs pageable |
|---|---:|---|---|
| n1024 | 4 | cuda-host-origin-pinned | 1.2× [1.15, 1.25]; faster; direction=True; magnitude=True |
| n1024 | 4 | cpu-gpu-origin-pinned | 1.2× [1.15, 1.25]; faster; direction=True; magnitude=True |
| n1024 | 4 | cuda-gpu-origin-pinned | 1.2× [1.15, 1.25]; faster; direction=True; magnitude=True |
| n1024 | 8 | cuda-host-origin-pinned | 1.2× [1.15, 1.25]; faster; direction=True; magnitude=True |
| n1024 | 8 | cpu-gpu-origin-pinned | 1.2× [1.15, 1.25]; faster; direction=True; magnitude=True |
| n1024 | 8 | cuda-gpu-origin-pinned | 1.2× [1.15, 1.25]; faster; direction=True; magnitude=True |
| n4096 | 4 | cuda-host-origin-pinned | 1.2× [1.15, 1.25]; faster; direction=True; magnitude=True |
| n4096 | 4 | cpu-gpu-origin-pinned | 1.2× [1.15, 1.25]; faster; direction=True; magnitude=True |
| n4096 | 4 | cuda-gpu-origin-pinned | 1.2× [1.15, 1.25]; faster; direction=True; magnitude=True |
| n4096 | 8 | cuda-host-origin-pinned | 1.2× [1.15, 1.25]; faster; direction=True; magnitude=True |
| n4096 | 8 | cpu-gpu-origin-pinned | 1.2× [1.15, 1.25]; faster; direction=True; magnitude=True |
| n4096 | 8 | cuda-gpu-origin-pinned | 1.2× [1.15, 1.25]; faster; direction=True; magnitude=True |

## Boundary inversions

- n1024, 4-bit, cuda-host-origin-pinned: direction and magnitude vetoed.
- n1024, 8-bit, cuda-host-origin-pinned: direction and magnitude vetoed.

## Direction-based crossovers

- cuda-resident, 4-bit: not resolved.
- cuda-host-origin, 4-bit: not resolved.
- cuda-host-origin-pinned, 4-bit: not resolved.
- cuda-gpu-origin, 4-bit: not resolved.
- cuda-gpu-origin-pinned, 4-bit: not resolved.
- cuda-resident, 8-bit: not resolved.
- cuda-host-origin, 8-bit: not resolved.
- cuda-host-origin-pinned, 8-bit: not resolved.
- cuda-gpu-origin, 8-bit: not resolved.
- cuda-gpu-origin-pinned, 8-bit: not resolved.

## Figures

- ![F1](f1_time_vs_elements.png)
- ![F2](f2_speedup_vs_elements.png)
- ![F3](f3_stage_breakdown.png)
- ![F4](f4_graph_vs_resident.png)
- ![BANDWIDTH](appendix_bandwidth.png)
