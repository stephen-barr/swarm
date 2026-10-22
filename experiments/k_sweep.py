"""
Grid search for the scarcity weight k.

Uses seeds 0..N-1.
Every k runs on the same drones/targets per instance, so comparisons between k are
paired. Objective: raw score as % of Hungarian (higher is better).

A k is "near-best" in a cell if the 95% bootstrap CI of its paired difference to
the best k includes 0. Expect a plateau of near-best values, not a single winner.

Run from repo root:
    python -m experiments.k_sweep run --seeds 500
    python -m experiments.k_sweep analyze
"""
import argparse
import csv
import time
from concurrent.futures import ProcessPoolExecutor
from typing import NamedTuple

import numpy as np

from swarm.baselines import solveHungarian
from swarm.scenarios import (
    complete_graph,
    line_graph,
    random_instance,
    ring_graph,
    solve,
)

GRAPHS = {"complete": complete_graph, "ring": ring_graph, "line": line_graph}


class Cell(NamedTuple):
    agents: int
    targets: int
    role_size: int
    graph: str

    def label(self):
        return f"{self.agents}a/{self.targets}t role{self.role_size} {self.graph}"


KS = [0.0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0]
CELLS = [
    Cell(agents=12, targets=6, role_size=2, graph="ring"),
    Cell(agents=24, targets=12, role_size=2, graph="ring"),
    Cell(agents=24, targets=12, role_size=2, graph="complete"),
    Cell(agents=48, targets=24, role_size=2, graph="ring"),
]
FIELDS = [*Cell._fields, "seed", "k", "pct_opt", "filled", "rounds"]
RESULTS = "results/k_sweep.csv"

Sweep = dict[Cell, dict[float,
                        dict[int, dict[str,
                                       float]]]]      # cell -> k -> seed -> metrics
FIGURE = "results/k_sweep.png"

# ---------- running ----------


def report_progress(done: int, total: int, start: float, every: int):
    """Print done/total, elapsed and estimated time left every `every` jobs."""
    if done % every and done != total:
        return
    elapsed = time.time() - start
    left = elapsed / done * (total - done)
    print(
        f"  {done}/{total} instances  ({elapsed / 60:.1f} min elapsed, ~{left / 60:.1f} min left)",
        flush=True)


def run_instance(job: tuple[Cell, int]) -> list[dict]:
    cell, seed = job

    def instance(scarcity_k=0.0):
        return random_instance(seed=seed,
                               n_agents=cell.agents,
                               n_targets=cell.targets,
                               max_role_size=cell.role_size,
                               scarcity_k=scarcity_k)

    hungarian, _ = solveHungarian(*instance())
    if hungarian <= 0:
        return []
    rows = []
    for k in KS:
        result = solve(*instance(k), graph=GRAPHS[cell.graph])
        rows.append({
            **cell._asdict(), "seed": seed,
            "k": k,
            "pct_opt": 100 * result.total / hungarian,
            "filled": result.filled,
            "rounds": result.rounds
        })
    return rows


def run(seeds: int, out: str):
    jobs = [(cell, seed) for cell in CELLS for seed in range(seeds)]
    start, every = time.time(), max(1, len(jobs) // 50)
    print(f"running {len(jobs)} instances x {len(KS)} values of k", flush=True)
    with ProcessPoolExecutor() as pool, open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for done, rows in enumerate(pool.map(run_instance, jobs, chunksize=4),
                                    start=1):
            writer.writerows(rows)
            report_progress(done, len(jobs), start, every)
    print(f"wrote {out}")


# ---------- analysis ----------


def bootstrap_ci(x: np.ndarray, reps=10_000, seed=0) -> tuple[float, float]:
    """95% bootstrap interval for the mean of x."""
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), (reps, len(x)))].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def load(path: str) -> Sweep:
    data: Sweep = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            cell = Cell(int(r["agents"]), int(r["targets"]),
                        int(r["role_size"]), r["graph"])
            by_seed = data.setdefault(cell, {}).setdefault(float(r["k"]), {})
            by_seed[int(r["seed"])] = {
                "pct_opt": float(r["pct_opt"]),
                "filled": float(r["filled"]),
                "rounds": float(r["rounds"])
            }
    return data


def analyze(path: str, figure: str):
    data = load(path)
    curves = {}
    for cell in CELLS:
        if cell not in data:
            continue
        by_k = data[cell]
        seeds = sorted(by_k[0.0])

        def metric(k, name):
            return np.array([by_k[k][s][name] for s in seeds])

        score = {k: metric(k, "pct_opt") for k in KS}
        best = max(KS, key=lambda k: score[k].mean())

        print(f"\n{cell.label()}  ({len(seeds)} instances)  best k = {best}")
        print(
            f"{'k':>6}{'% opt':>8}{'Δ vs best':>11}{'95% CI':>18}{'filled':>8}{'rounds':>8}  near-best"
        )
        curve = []
        for k in KS:
            diff = score[k] - score[best]
            lo, hi = bootstrap_ci(diff)
            ci = f"[{lo:.2f}, {hi:.2f}]"
            near = "*" if lo <= 0 <= hi else ""
            print(
                f"{k:>6}{score[k].mean():>8.2f}{diff.mean():>11.2f}{ci:>18}"
                f"{metric(k, 'filled').mean():>8.2f}{metric(k, 'rounds').mean():>8.1f}  {near}"
            )
            curve.append((k, score[k].mean(), *bootstrap_ci(score[k])))
        curves[cell] = curve
    plot(curves, figure)


def plot(curves: dict[Cell, list[tuple]], figure: str):
    """Mean % of Hungarian vs k (log axis); dashed line = k=0 for that cell."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for cell, curve in curves.items():
        ks, means, lows, highs = map(np.array, zip(*curve))
        positive = ks > 0
        line, = ax.plot(ks[positive],
                        means[positive],
                        marker="o",
                        label=cell.label())
        ax.fill_between(ks[positive],
                        lows[positive],
                        highs[positive],
                        color=line.get_color(),
                        alpha=0.15)
        ax.axhline(means[~positive][0],
                   color=line.get_color(),
                   linestyle="--",
                   linewidth=1)
    ax.set_xscale("log")
    ax.set_xlabel("scarcity weight k (dashed: k = 0)")
    ax.set_ylabel("raw score, % of Hungarian")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure, dpi=150)
    print(f"\nwrote {figure}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["run", "analyze"])
    parser.add_argument("--seeds",
                        type=int,
                        default=500,
                        help="tuning instances per cell")
    parser.add_argument("--results", default=RESULTS)
    parser.add_argument("--figure", default=FIGURE)
    args = parser.parse_args()

    if args.command == "run":
        run(args.seeds, args.results)
    else:
        analyze(args.results, args.figure)


if __name__ == "__main__":
    main()
