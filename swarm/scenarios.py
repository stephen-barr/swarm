import random

import numpy as np
from scipy.sparse.csgraph import shortest_path

from swarm.agent import (
    Agent,
    AgentConfig,
    Target,
    buildSlots,
    group_slots,
    index_targets,
)
from swarm.auction import run_auction
from swarm.comms import Channel
from swarm.models import (
    AuctionResult,
    Capability,
    Covariance,
    Frame,
    PlatformClass,
    PositionEstimate,
    ThreeVector,
    role,
)


def line_edges(n) -> tuple[int, list[tuple[int, int]]]:
    return n, [(i, i + 1) for i in range(n - 1)]


def complete_edges(n) -> tuple[int, list[tuple[int, int]]]:
    return n, [(i, j) for i in range(n) for j in range(n) if i < j]


def ring_edges(n) -> tuple[int, list[tuple[int, int]]]:
    assert n >= 3       # ring graph doesn't make sense in these cases
    _, ring = line_edges(n)
    ring.append((0, n - 1))
    return n, ring


def edges_to_topology(
        graph: tuple[int, list[tuple[int, int]]]) -> dict[int, list[int]]:
    n, edges = graph
    topology = {i: [] for i in range(n)}
    for k, v in edges:
        topology[k].append(v)
        topology[v].append(k)
    return topology


# Find the max distance between any two points
def getDiameter(topology: dict[int, list[int]]) -> int:
    a_matrix = np.zeros((len(topology), len(topology)))
    for node, neighbors in topology.items():
        a_matrix[node, neighbors] = 1

    diameter = shortest_path(a_matrix, unweighted=True, directed=False)
    if np.isinf(diameter).any():
        raise ValueError("The graph is disconnected")
    return int(diameter.max())


#-------- helper functions ---------

COV: Covariance = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
STILL: ThreeVector = (0.0, 0.0, 0.0)
CHAR_TIME = 60.0        # seconds; shared by every test target


def at(x: float, y: float, z: float = 0.0) -> PositionEstimate:
    return PositionEstimate(Frame.GLOBAL, (x, y, z), STILL, COV)


def make_agent(agent_id, capabilities, x, y):
    return Agent(
        AgentConfig(agent_id, PlatformClass.GROUP_1_MULTIROTOR,
                    frozenset(capabilities)),
        at(x, y),
    )


def make_target(target_id, position, threat, requirements):
    return Target(target_id, position, threat, CHAR_TIME, requirements)


def complete_graph(ids):
    return edges_to_topology(complete_edges(len(ids)))


def line_graph(ids):
    return edges_to_topology(line_edges(len(ids)))


def ring_graph(ids):
    return edges_to_topology(ring_edges(len(ids)))


###


def solve(agents, targets, graph=complete_graph) -> AuctionResult:
    """Run a distributed auction; return the number of rounds."""
    ids = [a.config.agent_id for a in agents]
    channel = Channel(graph(ids))
    return run_auction(agents, channel, group_slots(buildSlots(targets)),
                       index_targets(targets))


def random_instance(seed, n_agents=6, n_targets=3, max_role_size=1):
    rng = random.Random(seed)
    caps = list(Capability)
    targets = [
        Target(
            target_id=t,
            target_position=at(rng.uniform(-500, 500), rng.uniform(-500, 500)),
            threat_level=rng.uniform(0.1, 1.0),
            threat_char_time=CHAR_TIME,
            requirement_distribution={
                role(*rng.sample(caps, rng.randint(1, max_role_size))): {
                    n: rng.random()
                    for n in (1, 2, 3)
                }
                for _ in range(rng.randint(1, 3))
            },
        ) for t in range(n_targets)
    ]
    agents = [
        make_agent(a, rng.sample(caps, rng.randint(1, 3)),
                   rng.uniform(-500, 500), rng.uniform(-500, 500))
        for a in range(n_agents)
    ]
    return agents, targets
