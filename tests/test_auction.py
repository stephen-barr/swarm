import random

import pytest

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
    Capability,
    Covariance,
    Frame,
    PlatformClass,
    PositionEstimate,
    ThreeVector,
    role,
)

COV: Covariance = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
STILL: ThreeVector = (0.0, 0.0, 0.0)

A = Capability.SURVEILLANCE
B = Capability.MAPPING

# ---- helpers ----


def at(x: float, y: float, z: float = 0.0) -> PositionEstimate:
    return PositionEstimate(Frame.GLOBAL, (x, y, z), STILL, COV)


def make_agent(agent_id, capabilities, x, y, speed=12.0):
    return Agent(
        AgentConfig(agent_id, PlatformClass.GROUP_1_MULTIROTOR,
                    frozenset(capabilities), speed),
        at(x, y),
    )


def complete_graph(ids):
    return {i: [j for j in ids if j != i] for i in ids}


def line_graph(ids):
    ids = sorted(ids)
    return {
        ids[k]: [ids[j] for j in (k - 1, k + 1) if 0 <= j < len(ids)]
        for k in range(len(ids))
    }


def solve(agents, targets, graph=complete_graph) -> int:
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
            requirement_distribution={
                role(*rng.sample(caps, rng.randint(1, max_role_size))): {
                    n: 1.0 / 3
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


# ---- exact cases (unchanged behaviour) ----


def test_no_eligible_agent_leaves_slot_open():
    targets = [Target(0, at(100, 0), 0.5, {role(Capability.STRIKE): {1: 1.0}})]
    agents = [make_agent(0, {A}, 0, 0)]
    solve(agents, targets)
    assert agents[0].slot is None
    assert agents[0].winners == {}


def test_closer_agent_wins_the_only_slot():
    targets = [Target(0, at(100, 0), 0.5, {role(A): {1: 1.0}})]
    agents = [make_agent(0, {A}, 10, 0), make_agent(1, {A}, 400, 0)]
    solve(agents, targets)
    assert agents[0].slot is not None
    assert agents[1].slot is None


def test_identical_agents_both_assigned():
    """Tie-breaking: two identical agents, two identical slots."""
    targets = [Target(0, at(100, 0), 0.5, {role(A): {2: 1.0}})]
    agents = [make_agent(0, {A}, 0, 0), make_agent(1, {A}, 0, 0)]
    solve(agents, targets)
    assert agents[0].slot is not None
    assert agents[1].slot is not None
    assert agents[0].slot != agents[1].slot


def test_mostlikely_picks_by_probability_not_key_order():
    t = Target(0, at(0, 0), 0.5, {role(B): {3: 0.1, 1: 0.7, 2: 0.2}})
    assert t.mostLikely() == {role(B): 1}


# ---- new: roles with several capabilities ----


def test_combined_role_needs_one_drone_with_both():
    targets = [Target(0, at(100, 0), 0.5, {role(A, B): {1: 1.0}})]
    agents = [
        make_agent(0, {A}, 0, 0),
        make_agent(1, {B}, 0, 0),
        make_agent(2, {A, B}, 300, 0)
    ]                                       # farthest, but the only one eligible
    solve(agents, targets)
    assert agents[2].slot is not None
    assert agents[0].slot is None
    assert agents[1].slot is None


def test_separate_roles_filled_by_separate_drones():
    targets = [
        Target(0, at(100, 0), 0.5, {
            role(A): {
                1: 1.0
            },
            role(B): {
                1: 1.0
            }
        })
    ]
    agents = [make_agent(0, {A}, 0, 0), make_agent(1, {B}, 0, 0)]
    solve(agents, targets)
    assert agents[0].slot is not None and agents[0].slot.required == role(A)
    assert agents[1].slot is not None and agents[1].slot.required == role(B)


def test_multi_capable_drone_fills_single_role():
    targets = [Target(0, at(100, 0), 0.5, {role(A): {1: 1.0}})]
    agents = [make_agent(0, {A, B}, 0, 0)]
    solve(agents, targets)
    assert agents[0].slot is not None


# ---- new: distributed-consensus behaviour ----


def test_line_graph_matches_complete_graph_on_simple_case():
    targets = [Target(0, at(100, 0), 0.5, {role(A): {1: 1.0}})]
    near = [make_agent(i, {A}, x, 0) for i, x in enumerate((10, 200, 400))]
    far = [make_agent(i, {A}, x, 0) for i, x in enumerate((10, 200, 400))]
    solve(near, targets, complete_graph)
    solve(far, targets, line_graph)
    assert [a.slot for a in near] == [a.slot for a in far]


# ---- properties over random instances ----

GRAPHS = [complete_graph, line_graph]
ROLE_SIZES = [1, 2]


@pytest.mark.parametrize("graph", GRAPHS)
@pytest.mark.parametrize("role_size", ROLE_SIZES)
@pytest.mark.parametrize("seed", range(50))
def test_conflict_free(seed, role_size, graph):
    agents, targets = random_instance(seed, max_role_size=role_size)
    solve(agents, targets, graph)
    held = [a.slot for a in agents if a.slot is not None]
    assert len(held) == len(set(held)), f"duplicate slot, seed={seed}"


@pytest.mark.parametrize("graph", GRAPHS)
@pytest.mark.parametrize("role_size", ROLE_SIZES)
@pytest.mark.parametrize("seed", range(50))
def test_assignments_respect_capabilities(seed, role_size, graph):
    agents, targets = random_instance(seed, max_role_size=role_size)
    solve(agents, targets, graph)
    for a in agents:
        if a.slot is not None:
            assert a.slot.required <= a.config.capabilities, f"seed={seed}"


@pytest.mark.parametrize("graph", GRAPHS)
@pytest.mark.parametrize("seed", range(50))
def test_terminates(seed, graph):
    agents, targets = random_instance(seed)
    rounds = solve(agents, targets,
                   graph)                   # run_auction raises if it doesn't converge
    assert rounds < 200, f"slow convergence, seed={seed}"


@pytest.mark.parametrize("graph", GRAPHS)
@pytest.mark.parametrize("seed", range(50))
def test_all_tables_agree_after_convergence(seed, graph):
    agents, targets = random_instance(seed)
    solve(agents, targets, graph)
    for a in agents[1:]:
        assert a.winners == agents[0].winners, f"tables disagree, seed={seed}"


@pytest.mark.parametrize("graph", GRAPHS)
@pytest.mark.parametrize("seed", range(50))
def test_held_slots_match_the_shared_table(seed, graph):
    agents, targets = random_instance(seed)
    solve(agents, targets, graph)
    table = agents[0].winners
    for a in agents:
        if a.slot is not None:
            _, winner_id = table[a.slot]
            assert winner_id == a.config.agent_id, f"seed={seed}"


@pytest.mark.parametrize("seed", range(20))
def test_deterministic(seed):
    a1, t1 = random_instance(seed)
    a2, t2 = random_instance(seed)
    solve(a1, t1)
    solve(a2, t2)
    assert [a.slot for a in a1] == [a.slot for a in a2], f"seed={seed}"
