# Benchmark Results

200 random instances per configuration, roles of up to 2 capabilities, drones and targets placed in a 1,000 × 1,000 area, each run on complete, ring and line networks.

To reproduce these tables: `python -m experiments.benchmark`, then print `python -m experiments.summary > results/benchmark_summary.md`

## Solution Quality
- Quality analysis uses complete graph runs, every topology gives equivalent solution.
- Percentages are relative to Hungarian assignment (optimal assignment, solved globally). 
- Infeasible gives the number of runs which generated no viable assignments based on drone capability/target requirement mismatch, and were excluded from subsequent calculations.

| Drones | Targets | Infeasible | Mean % of optimal | 5th percentile | Worst | Exactly optimal | Coverage | Same as greedy |
|---|---|---|---|---|---|---|---|---|
| 6 | 3 | 6 | 99.3% | 98.7% | 69.5% | 94% | 99.1% | 98% |
| 12 | 6 | 0 | 97.9% | 90.4% | 83.6% | 55% | 96.7% | 92% |
| 24 | 12 | 0 | 96.7% | 92.1% | 89.2% | 10% | 95.7% | 72% |
| 48 | 24 | 0 | 95.4% | 91.5% | 85.8% | 0% | 95.7% | 43% |

## Convergence
Convergence is bounded by $\text{slots} \times \text{diameter}$ in the theoretical worst case.

| Drones | Targets | Network | Diameter | Median rounds | Max rounds | Median rounds / diameter | Worst rounds / bound | Median messages | Same as complete |
|---|---|---|---|---|---|---|---|---|---|
| 6 | 3 | complete | 1 | 3 | 5 | 3.00 | 0.83 | 90 | 100% |
| 6 | 3 | ring | 3 | 6 | 10 | 2.00 | 0.56 | 72 | 100% |
| 6 | 3 | line | 5 | 7 | 15 | 1.40 | 0.47 | 70 | 100% |
| 12 | 6 | complete | 1 | 4 | 8 | 4.00 | 0.36 | 528 | 100% |
| 12 | 6 | ring | 6 | 14 | 28 | 2.33 | 0.25 | 336 | 100% |
| 12 | 6 | line | 11 | 21 | 36 | 1.91 | 0.17 | 462 | 100% |
| 24 | 12 | complete | 1 | 6 | 9 | 6.00 | 0.26 | 3,312 | 100% |
| 24 | 12 | ring | 12 | 38 | 72 | 3.17 | 0.14 | 1,824 | 100% |
| 24 | 12 | line | 23 | 55 | 100 | 2.39 | 0.11 | 2,530 | 100% |
| 48 | 24 | complete | 1 | 8 | 11 | 8.00 | 0.16 | 18,048 | 100% |
| 48 | 24 | ring | 24 | 96 | 151 | 4.00 | 0.08 | 9,216 | 100% |
| 48 | 24 | line | 47 | 140 | 221 | 2.97 | 0.05 | 13,113 | 100% |

Ring and line assignment identical to complete graph: 1600/1600 runs

Runs where the auction exceeded the optimum: 0

Ring and line assignment identical to complete graph: 1600/1600 runs

Runs where the auction exceeded the optimum: 0
