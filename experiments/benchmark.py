import csv
from typing import NamedTuple

from swarm.agent import buildSlots
from swarm.baselines import greedyBaseline, solveHungarian
from swarm.scenarios import (
    complete_graph,
    getDiameter,
    line_graph,
    random_instance,
    ring_graph,
    solve,
)

SIZES = [(6, 3), (12, 6), (24, 12), (48, 24)]
SEEDS = range(200)
GRAPHS = {"complete": complete_graph, "ring": ring_graph, "line": line_graph}


class Row(NamedTuple):
    n_agents: int
    n_targets: int
    seed: int
    graph: str
    n_slots: int
    diameter: int
    auction_total: float
    auction_filled: int
    rounds: int
    messages: int
    hungarian_total: float
    hungarian_filled: int
    greedy_total: float
    greedy_filled: int
    matches_complete: bool


def runInstance(size: tuple, seed, graphs: dict):
    a, t = random_instance(seed=seed,
                           n_agents=size[0],
                           n_targets=size[1],
                           max_role_size=2)
    hungarian_result = solveHungarian(agents=a, targets=t)
    greedy_result = greedyBaseline(agents=a, targets=t)
    rows = []
    complete_winners = None
    for name, graph in graphs.items():
        a, t = random_instance(seed=seed,
                               n_agents=size[0],
                               n_targets=size[1],
                               max_role_size=2)
        topology = graph([agent.config.agent_id for agent in a])
        auction_result = solve(agents=a, targets=t, graph=graph)

        if name == "complete":
            complete_winners = auction_result.winners

        rows.append(
            Row(
                n_agents=size[0],
                n_targets=size[1],
                seed=seed,
                graph=name,
                n_slots=len(buildSlots(t)),
                diameter=getDiameter(topology),
                auction_total=auction_result.total,
                auction_filled=auction_result.filled,
                rounds=auction_result.rounds,
                messages=auction_result.rounds *
                sum(len(neighbors) for neighbors in topology.values()),
                hungarian_total=hungarian_result[0],
                hungarian_filled=hungarian_result[1],
                greedy_total=greedy_result[0],
                greedy_filled=greedy_result[1],
                matches_complete=auction_result.winners == complete_winners,
            ))
    return rows


class SizeResult(NamedTuple):
    size: tuple[int, int]                   # (number of agents, number of targets)
    infeasible_instances: int               # how many instances have zero possible assignment pairs
    mean_of_optimal: float                  # mean of auction score / hungarian score
    fifth_percent_of_optimal: float         # fifth percentile of auction score / hungarian score
    minimum_percent_of_optimal: float       # worst auction score / hungarian score
    percent_exactly_optimal: float          # number of auctions where auction = hungarian / (number of auctions - infeasible instances)
    coverage: float                         # average of auction slots filled / hungarian slots filled
    percent_same_as_greedy: float           # number of auctions where auction = greedy / (number of auctions - infeasible instances)


def writetoCSV():
    filename = "results/sweep.csv"
    with open(filename, 'w', newline="") as csvfile:
        # creating a csv writer object
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(Row._fields)
        csvwriter.writerows(row for size in SIZES for seed in SEEDS
                            for row in runInstance(size, seed, GRAPHS))


if __name__ == "__main__":
    writetoCSV()
