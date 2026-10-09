import csv
from collections import defaultdict
from typing import NamedTuple

import numpy as np

TOL = 1e-9
INT_FIELDS = {
    "n_agents", "n_targets", "seed", "n_slots", "diameter", "auction_filled",
    "rounds", "messages", "hungarian_filled", "greedy_filled"
}
FLOAT_FIELDS = {"auction_total", "hungarian_total", "greedy_total"}


def loadRows(path="results/sweep.csv") -> list[dict]:
    """Read the sweep CSV, convert each column to its real data type."""
    with open(path, newline="") as f:
        raw = list(csv.DictReader(
            f))                             # create dictionaries for each row, with column name -> value
    rows = []
    for r in raw:
        row = dict(r)                       # convert all data to the correct type
        for k in INT_FIELDS:
            row[k] = int(row[k])
        for k in FLOAT_FIELDS:
            row[k] = float(row[k])
        row["matches_complete"] = bool(row["matches_complete"] == "True")
        rows.append(row)
    return rows


class SizeResult(NamedTuple):
    n_agents: int
    n_targets: int
    infeasible: int
    mean_ratio: float
    p5_ratio: float
    min_ratio: float
    optimal_percent: float
    coverage: float
    same_as_greedy_percent: float


def qualityBySize(groups, configurations) -> list[SizeResult]:
    results = []

    for n_agents, n_targets in configurations:
        complete = groups[(n_agents, n_targets,
                           "complete")]                             # list of rows with complete graph
        feasible = [r for r in complete if r["hungarian_total"] > 0
                    ]                                               # only consider the rows with nonzero allocations

        ratios = np.array([
            r["auction_total"] / r["hungarian_total"] for r in feasible
        ])                                                                      # scoring ratio of rows for n_agents, n_targets
        optimal = np.array([
            abs(r["auction_total"] - r["hungarian_total"]) < TOL
            for r in feasible
        ])                                                                      # boolean, if row is an optimal assignment
        coverage = np.array([
            r["auction_filled"] / r["hungarian_filled"] for r in feasible
        ])                                                                      # ratio of number of slots filled
        equals_greedy = np.array([
            abs(r["auction_total"] - r["greedy_total"]) < TOL for r in feasible
        ])                                                                      # boolean, if row is equivalent to greedy

        results.append(
            SizeResult(
                n_agents=n_agents,
                n_targets=n_targets,
                infeasible=len(complete) - len(feasible),
                mean_ratio=ratios.mean(),
                p5_ratio=np.percentile(ratios, 5),
                min_ratio=ratios.min(),
                optimal_percent=optimal.mean(
                ),                                        # mean of a boolean list returns percent of rows that are optimal
                coverage=coverage.mean(),
                same_as_greedy_percent=equals_greedy.mean()))

    return results


class SizeGraphResult(NamedTuple):
    n_agents: int
    n_targets: int
    graph: str
    diameter: int
    median_rounds: int            # median time of convergence
    max_rounds: int               # worst time of convergence
    max_efficiency_ratio: float   # worst case compared to theoretical bound max(rounds/(slots*diameter))
    median_messages: int          # number of messages sent
    rounds_diameter_ratio: float  # median_rounds / diameter
    matches_complete: float       # fraction of runs with same assignment as complete graph


def convergenceBySizeGraph(groups, configurations):
    """
    Cost of reaching agreement compared over network topologies. 
    """
    results = []
    for n_agents, n_targets in configurations:
        for g in ("complete", "ring", "line"):
            print(f"Beginning run: {n_agents}, {n_targets}, {g}.")
            selected = groups[(n_agents, n_targets,
                               g)]                    # all instances included, feasible or not
            diameters = {
                r["diameter"]
                for r in selected
            }                                         # all diameters are the same size for the same size tuple
            (
                diameter,
            ) = diameters                             # unpack diameters to ensure it is the same size, throws error if not
            rounds = np.array([r["rounds"] for r in selected])
            efficiency = np.array(
                [r["rounds"] / (diameter * r["n_slots"]) for r in selected])
            messages = np.array([r["messages"] for r in selected])
            equals_complete = np.array(
                [r["matches_complete"] for r in selected])

            results.append(
                SizeGraphResult(n_agents=n_agents,
                                n_targets=n_targets,
                                graph=g,
                                diameter=diameter,
                                median_rounds=np.median(rounds),
                                max_rounds=rounds.max(),
                                max_efficiency_ratio=efficiency.max(),
                                median_messages=np.median(messages),
                                rounds_diameter_ratio=np.median(rounds) /
                                diameter,
                                matches_complete=equals_complete.mean()))
    return results


def summary(path="results/sweep.csv"):

    rows = loadRows(path)
    groups = defaultdict(list)
    for r in rows:
        groups[(r["n_agents"], r["n_targets"], r["graph"])].append(
            r)                                                      # dict (n_agents, n_targets, graph) -> list[rows]

    configurations = sorted({(n_a, n_t) for n_a, n_t, _ in groups}) # sort

    quality = qualityBySize(groups, configurations)
    convergence = convergenceBySizeGraph(groups, configurations)

    print(
        "| Drones | Targets | Infeasible | Mean % of optimal | 5th percentile | Worst "
        "| Exactly optimal | Coverage | Same as greedy |")
    print("|---|---|---|---|---|---|---|---|---|")
    for q in quality:
        print(f"| {q.n_agents} | {q.n_targets} | {q.infeasible} "
              f"| {q.mean_ratio:.1%} | {q.p5_ratio:.1%} | {q.min_ratio:.1%} "
              f"| {q.optimal_percent:.0%} | {q.coverage:.1%} "
              f"| {q.same_as_greedy_percent:.0%} |")

    print()
    print(
        "| Drones | Targets | Network | Diameter | Median rounds | Max rounds "
        "| Median rounds / diameter | Worst rounds / bound | Median messages "
        "| Same as complete |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for c in convergence:
        print(
            f"| {c.n_agents} | {c.n_targets} | {c.graph} | {c.diameter} "
            f"| {c.median_rounds:.0f} | {c.max_rounds} "
            f"| {c.rounds_diameter_ratio:.2f} | {c.max_efficiency_ratio:.2f} "
            f"| {c.median_messages:,.0f} | {c.matches_complete:.0%} |")

    others = [r for r in rows if r["graph"] != "complete"]
    above_optimal = [
        r for r in rows if r["auction_total"] > r["hungarian_total"] + TOL
    ]
    print()
    print(f"Ring and line assignment identical to complete graph: "
          f"{sum(r['matches_complete'] for r in others)}/{len(others)} runs")
    print(f"Runs where the auction exceeded the optimum: {len(above_optimal)}")


if __name__ == "__main__":
    summary()
