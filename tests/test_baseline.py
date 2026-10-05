import itertools
import random

import pytest

from swarm.agent import (
    buildSlots,
    indexTargets,
)
from swarm.auction import auctionTotal
from swarm.baselines import solveHungarian
from swarm.models import (
    Capability,
    Covariance,
    ThreeVector,
    role,
)
from swarm.scenarios import at, make_agent, make_target

COV: Covariance = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
STILL: ThreeVector = (0.0, 0.0, 0.0)
CHAR_TIME = 60.0

S = Capability.SURVEILLANCE
M = Capability.MAPPING


def setup(agents, targets):
    slots = buildSlots(targets)
    return slots, indexTargets(targets)


def eligible(agent, slot):
    return slot.required <= agent.config.capabilities


def brute_force(agents, slots, targets_by_id):
    """Best (filled, total) over every possible assignment.

    Each agent takes one eligible slot or none; no slot is used twice.
    Score-first maximises total; coverage-first maximises filled, then total.
    """
    options = [[None] + [j for j, s in enumerate(slots) if eligible(a, s)]
               for a in agents]
    best = (0, 0.0)
    for choice in itertools.product(*options):
        used = [j for j in choice if j is not None]
        if len(used) != len(set(used)):
            continue
        total = sum(agents[i].makeScore(targets_by_id[slots[j].target_id])
                    for i, j in enumerate(choice) if j is not None)
        candidate = (len(used), total)
        key = lambda r: (r[1], r[0])
        if key(candidate) > key(best):
            best = candidate
    return best


def small_instance(seed):
    rng = random.Random(seed)
    targets = [
        make_target(
            t,
            at(rng.uniform(-300, 300), rng.uniform(-300, 300)),
            rng.uniform(0.2, 1.0),
            {role(rng.choice([S, M])): {
                 rng.randint(1, 2): 1.0
             }},
        ) for t in range(2)
    ]
    agents = [
        make_agent(a, rng.sample([S, M], rng.randint(1, 2)),
                   rng.uniform(-300, 300), rng.uniform(-300, 300))
        for a in range(rng.randint(1, 4))
    ]
    return agents, targets


# ---- exact cases ----


def test_no_eligible_pairs_gives_nothing():
    targets = [make_target(0, at(100, 0), 0.5, {role(M): {1: 1.0}})]
    agents = [make_agent(0, {S}, 0, 0)]
    slots, lookup = setup(agents, targets)
    assert solveHungarian(agents, slots, lookup) == (0.0, 0)


def test_single_eligible_pair_scores_exactly():
    targets = [make_target(0, at(100, 0), 0.5, {role(S): {1: 1.0}})]
    agents = [make_agent(0, {S}, 0, 0)]
    slots, lookup = setup(agents, targets)
    total, filled = solveHungarian(agents, slots, lookup)
    assert filled == 1
    assert total == pytest.approx(agents[0].makeScore(targets[0]))


def test_ineligible_agent_is_never_counted():
    targets = [make_target(0, at(100, 0), 0.5, {role(S): {1: 1.0}})]
    agents = [make_agent(0, {S}, 0, 0), make_agent(1, {M}, 0, 0)]
    slots, lookup = setup(agents, targets)
    _, filled = solveHungarian(agents, slots, lookup)
    assert filled == 1


def test_more_agents_than_slots():
    targets = [make_target(0, at(100, 0), 0.5, {role(S): {1: 1.0}})]
    agents = [make_agent(i, {S}, 10 * i, 0) for i in range(4)]
    slots, lookup = setup(agents, targets)
    _, filled = solveHungarian(agents, slots, lookup)
    assert filled == 1


# ---- against brute force ----


@pytest.mark.parametrize("seed", range(100))
def test_matches_brute_force(seed):
    agents, targets = small_instance(seed)
    slots, lookup = setup(agents, targets)
    exp_filled, exp_total = brute_force(agents, slots, lookup)
    total, filled = solveHungarian(
        agents,
        slots,
        lookup,
    )
    assert filled == exp_filled, f"seed={seed}"
    assert total == pytest.approx(exp_total), f"seed={seed}"


# ---- auctionTotal ----


def test_auction_total_of_empty_table_is_zero():
    assert auctionTotal({}) == (0.0, 0)


def test_auction_total_sums_scores_not_ids():
    targets = [make_target(0, at(0, 0), 0.5, {role(S): {2: 1.0}})]
    slots = buildSlots(targets)
    winners = {slots[0]: (12.5, 3), slots[1]: (7.5, 4)}
    total, filled = auctionTotal(winners)
    assert total == pytest.approx(20.0)
    assert filled == 2
