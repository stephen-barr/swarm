import random

import pytest

from agent import (
    Agent,
    AgentConfig,
    Capability,
    Covariance,
    Frame,
    PlatformClass,
    PositionEstimate,
    Target,
    ThreeVector,
    buildSlots,
    group_slots,
    index_targets,
    run_auction,
)

COV: Covariance = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
STILL: ThreeVector = (0.0, 0.0, 0.0)


def at(x: float, y: float, z: float = 0.0) -> PositionEstimate:
    return PositionEstimate(Frame.GLOBAL, (x, y, z), STILL, COV)


def make_agent(agent_id, capabilities, x, y, speed=12.0):
    return Agent(
        AgentConfig(agent_id, PlatformClass.GROUP_1_MULTIROTOR,
                    frozenset(capabilities), speed),
        at(x, y),
    )


def solve(agents, targets):
    """Run an auction and return (winners, rounds)."""
    return run_auction(agents, group_slots(buildSlots(targets)), index_targets(targets))


def random_instance(seed, n_agents=6, n_targets=3):
    rng = random.Random(seed)
    caps = list(Capability)
    targets = [
        Target(
            target_id=t,
            target_position=at(rng.uniform(-500, 500), rng.uniform(-500, 500)),
            threat_level=rng.uniform(0.1, 1.0),
            requirement_distribution={
                cap: {n: 1.0 / 3 for n in (1, 2, 3)}
                for cap in rng.sample(caps, rng.randint(1, 3))
            },
        )
        for t in range(n_targets)
    ]
    agents = [
        make_agent(a, rng.sample(caps, rng.randint(1, 3)),
                   rng.uniform(-500, 500), rng.uniform(-500, 500))
        for a in range(n_agents)
    ]
    return agents, targets


# ---- exact cases ----

def test_no_eligible_agent_leaves_slot_open():
    targets = [Target(0, at(100, 0), 0.5, {Capability.STRIKE: {1: 1.0}})]
    agents = [make_agent(0, {Capability.SURVEILLANCE}, 0, 0)]
    winners, _ = solve(agents, targets)
    assert agents[0].slot is None
    assert winners == {}


def test_closer_agent_wins_the_only_slot():
    targets = [Target(0, at(100, 0), 0.5, {Capability.SURVEILLANCE: {1: 1.0}})]
    agents = [make_agent(0, {Capability.SURVEILLANCE}, 10, 0),
              make_agent(1, {Capability.SURVEILLANCE}, 400, 0)]
    solve(agents, targets)
    assert agents[0].slot is not None
    assert agents[1].slot is None


def test_identical_agents_both_assigned():
    """Tie-breaking: two identical agents, two identical slots."""
    targets = [Target(0, at(100, 0), 0.5, {Capability.SURVEILLANCE: {2: 1.0}})]
    agents = [make_agent(0, {Capability.SURVEILLANCE}, 0, 0),
              make_agent(1, {Capability.SURVEILLANCE}, 0, 0)]
    solve(agents, targets)
    assert agents[0].slot is not None
    assert agents[1].slot is not None
    assert agents[0].slot != agents[1].slot


def test_mostlikely_picks_by_probability_not_key_order():
    t = Target(0, at(0, 0), 0.5, {Capability.MAPPING: {3: 0.1, 1: 0.7, 2: 0.2}})
    assert t.mostLikely() == {Capability.MAPPING: 1}


# ---- properties over random instances ----

@pytest.mark.parametrize("seed", range(50))
def test_conflict_free(seed):
    agents, targets = random_instance(seed)
    solve(agents, targets)
    held = [a.slot for a in agents if a.slot is not None]
    assert len(held) == len(set(held)), f"duplicate slot, seed={seed}"


@pytest.mark.parametrize("seed", range(50))
def test_assignments_respect_capabilities(seed):
    agents, targets = random_instance(seed)
    solve(agents, targets)
    for a in agents:
        if a.slot is not None:
            assert a.slot.capability in a.config.capabilities, f"seed={seed}"


@pytest.mark.parametrize("seed", range(50))
def test_terminates(seed):
    agents, targets = random_instance(seed)
    _, rounds = solve(agents, targets)          # run_auction raises if it doesn't converge
    assert rounds < 50, f"slow convergence, seed={seed}"


@pytest.mark.parametrize("seed", range(20))
def test_deterministic(seed):
    a1, t1 = random_instance(seed)
    a2, t2 = random_instance(seed)
    solve(a1, t1)
    solve(a2, t2)
    assert [a.slot for a in a1] == [a.slot for a in a2], f"seed={seed}"