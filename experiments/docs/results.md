| Drones | Targets | Infeasible | Mean % of optimal | 5th percentile | Worst | Exactly optimal | Coverage | Same as greedy |
|---|---|---|---|---|---|---|---|---|
| 6 | 3 | 6 | 99.3% | 98.7% | 69.5% | 94% | 99.1% | 98% |
| 12 | 6 | 0 | 97.9% | 90.4% | 83.6% | 55% | 96.7% | 92% |
| 24 | 12 | 0 | 96.7% | 92.1% | 89.2% | 10% | 95.7% | 72% |
| 48 | 24 | 0 | 95.4% | 91.5% | 85.8% | 0% | 95.7% | 43% |

| Drones | Targets | Network | Diameter | Median rounds | Max rounds | Median rounds / diameter | Worst rounds / bound | Median messages | Same as complete |
|---|---|---|---|---|---|---|---|---|---|
| 6 | 3 | complete | 1 | 3 | 5 | 3.00 | 0.67 | 90 | 100% |
| 6 | 3 | ring | 3 | 5 | 10 | 1.67 | 0.44 | 60 | 100% |
| 6 | 3 | line | 5 | 6 | 15 | 1.20 | 0.30 | 60 | 100% |
| 12 | 6 | complete | 1 | 4 | 7 | 4.00 | 0.36 | 528 | 100% |
| 12 | 6 | ring | 6 | 13 | 28 | 2.17 | 0.25 | 312 | 100% |
| 12 | 6 | line | 11 | 19 | 36 | 1.73 | 0.17 | 418 | 100% |
| 24 | 12 | complete | 1 | 6 | 9 | 6.00 | 0.26 | 3,312 | 100% |
| 24 | 12 | ring | 12 | 36 | 72 | 3.00 | 0.14 | 1,728 | 100% |
| 24 | 12 | line | 23 | 52 | 100 | 2.26 | 0.11 | 2,392 | 100% |
| 48 | 24 | complete | 1 | 8 | 11 | 8.00 | 0.15 | 18,048 | 100% |
| 48 | 24 | ring | 24 | 94 | 140 | 3.92 | 0.07 | 9,024 | 100% |
| 48 | 24 | line | 47 | 136 | 221 | 2.89 | 0.05 | 12,784 | 100% |

Ring and line assignment identical to complete graph: 1600/1600 runs
Runs where the auction exceeded the optimum: 0
