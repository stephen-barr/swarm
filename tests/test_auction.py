import pytest

from swarm.baselines import greedyBaseline, solveHungarian
from swarm.models import (
    Capability,
    role,
)
from swarm.scenarios import (
    at,
    complete_graph,
    line_graph,
    make_agent,
    make_target,
    random_instance,
    ring_graph,
    solve,
)

A = Capability.SURVEILLANCE
B = Capability.MAPPING

# ---- helpers ----
# ---- exact cases (unchanged behaviour) ----


def test_no_eligible_agent_leaves_slot_open():
    targets = [
        make_target(0, at(100, 0), 0.5, {role(Capability.STRIKE): {
                                             1: 1.0
                                         }})
    ]
    agents = [make_agent(0, {A}, 0, 0)]
    solve(agents, targets)
    assert agents[0].slot is None
    assert agents[0].winners == {}


def test_closer_agent_wins_the_only_slot():
    targets = [make_target(0, at(100, 0), 0.5, {role(A): {1: 1.0}})]
    agents = [make_agent(0, {A}, 10, 0), make_agent(1, {A}, 400, 0)]
    solve(agents, targets)
    assert agents[0].slot is not None
    assert agents[1].slot is None


def test_identical_agents_both_assigned():
    """Tie-breaking: two identical agents, two identical slots."""
    targets = [make_target(0, at(100, 0), 0.5, {role(A): {2: 1.0}})]
    agents = [make_agent(0, {A}, 0, 0), make_agent(1, {A}, 0, 0)]
    solve(agents, targets)
    assert agents[0].slot is not None
    assert agents[1].slot is not None
    assert agents[0].slot != agents[1].slot


def test_mostlikely_picks_by_probability_not_key_order():
    t = make_target(0, at(0, 0), 0.5, {role(B): {3: 0.1, 1: 0.7, 2: 0.2}})
    assert t.mostLikely() == {role(B): 1}


# ---- new: roles with several capabilities ----


def test_combined_role_needs_one_drone_with_both():
    targets = [make_target(0, at(100, 0), 0.5, {role(A, B): {1: 1.0}})]
    agents = [
        make_agent(0, {A}, 0, 0),
        make_agent(1, {B}, 0, 0),
        make_agent(2, {A, B}, 300, 0),      # farthest, but the only one eligible
    ]
    solve(agents, targets)
    assert agents[2].slot is not None
    assert agents[0].slot is None
    assert agents[1].slot is None


def test_separate_roles_filled_by_separate_drones():
    targets = [
        make_target(0, at(100, 0), 0.5, {
            role(A): {
                1: 1.0
            },
            role(B): {
                1: 1.0
            },
        })
    ]
    agents = [make_agent(0, {A}, 0, 0), make_agent(1, {B}, 0, 0)]
    solve(agents, targets)
    assert agents[0].slot is not None and agents[0].slot.required == role(A)
    assert agents[1].slot is not None and agents[1].slot.required == role(B)


def test_multi_capable_drone_fills_single_role():
    targets = [make_target(0, at(100, 0), 0.5, {role(A): {1: 1.0}})]
    agents = [make_agent(0, {A, B}, 0, 0)]
    solve(agents, targets)
    assert agents[0].slot is not None


# ---- new: distributed-consensus behaviour ----


def test_line_graph_matches_complete_graph_on_simple_case():
    targets = [make_target(0, at(100, 0), 0.5, {role(A): {1: 1.0}})]
    near = [make_agent(i, {A}, x, 0) for i, x in enumerate((10, 200, 400))]
    far = [make_agent(i, {A}, x, 0) for i, x in enumerate((10, 200, 400))]
    solve(near, targets, complete_graph)
    solve(far, targets, line_graph)
    assert [a.slot for a in near] == [a.slot for a in far]


# ---- properties over random instances ----

GRAPHS = [complete_graph, line_graph, ring_graph]
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
    solution = solve(agents, targets,
                     graph)                 # run_auction raises if it doesn't converge
    assert solution.rounds < 200, f"slow convergence, seed={seed}"


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


@pytest.mark.parametrize("graph", GRAPHS)
@pytest.mark.parametrize("seed", range(50))
def test_distributed_to_hungarian(seed, graph):
    agents, targets = random_instance(seed)
    distr_auction_result = solve(agents, targets, graph)
    hung_total, _ = solveHungarian(
        agents,
        targets=targets,
    )
    assert distr_auction_result.total <= hung_total + 1e-6
    assert hung_total / 2 <= distr_auction_result.total


@pytest.mark.parametrize("graph", GRAPHS)
@pytest.mark.parametrize("seed", range(50))
def test_auction_is_equal_to_greedy_with_no_ties(seed, graph):
    a, t = random_instance(seed=seed, multiple_slots_per_target=False)
    auction_results = solve(a, t, graph)
    greedy_results = greedyBaseline(a, t)
    assert auction_results.total == pytest.approx(greedy_results[0])
    assert auction_results.filled == pytest.approx(greedy_results[1])
