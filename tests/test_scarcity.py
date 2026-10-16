import math

import pytest

from swarm.agent import buildSlots, groupSlots, indexTargets
from swarm.models import Capability, Message, Payload, role
from swarm.scenarios import at, make_agent, make_target

A = Capability.SURVEILLANCE
B = Capability.MAPPING


def census_msg(configs, sender=1, receiver=0):
    """A message whose only content is a census of the given configs."""
    return Message(send_id=sender,
                   receiver_id=receiver,
                   payload=Payload(winners={},
                                   census={c.agent_id: c
                                           for c in configs}),
                   send_step=1)


def agent_with_census(my_caps, others_caps, k):
    """Agent 0 at the origin with scarcity k, after learning the configs of
    drones 1, 2, ... whose capabilities are given by others_caps."""
    a = make_agent(0, my_caps, 0, 0)
    a.scarcity_k = k
    others = [
        make_agent(i + 1, caps, 0, 0).config
        for i, caps in enumerate(others_caps)
    ]
    a.updateCensus([census_msg(others)])
    return a


def setup(targets):
    """The three lookups the auction passes to bid() / auctionPhase()."""
    return groupSlots(buildSlots(targets)), indexTargets(targets)


# ---- test 1: the weight value ----


@pytest.mark.parametrize(
    "others_caps, n",
    [
        ([[B], [B], [B]], 1),     # only agent 0 can fill {A}
        ([[A], [A], [A]], 4),     # everyone can fill {A}
    ])
def test_weight_is_exp_minus_k_n_over_N(others_caps, n):
    k = 2.0
    a = agent_with_census([A], others_caps, k)
    target = make_target(0, at(100, 0), 0.5, {role(A): {1: 1.0}})
    slots_by_role, targets_by_id = setup([target])

    bids = a.bid(slots_by_role, targets_by_id)
    (_, b), = bids.items()        # exactly one slot
    raw = a.makeScore(target)
    N = len(a.census)

    assert a.capability_count[role(A)] == n
    assert b.score == pytest.approx(raw * math.exp(-k * n / N))
    assert N == 4


# ---- test 2: scarcity changes the choice ----


@pytest.mark.parametrize(
    "k, expected_target",
    [
        (0.0, 0),                                     # no weight: the more valuable common target wins
        (2.0, 1),                                     # weighted: the scarce target wins
    ])
def test_versatile_drone_takes_scarce_slot_when_weighted(k, expected_target):
    a = agent_with_census([A, B], [[A], [A], [A]],
                          k)                          # {B} is scarce: only agent 0
    targets = [
        make_target(0, at(100, 0), 0.50, {role(A): {
                                              1: 1.0
                                          }}),        # common role
        make_target(1, at(0, 100), 0.45, {role(B): {
                                              1: 1.0
                                          }}),        # scarce role, same distance
    ]
    slots_by_role, targets_by_id = setup(targets)

    a.auctionPhase(slots_by_role, targets_by_id)

    assert a.slot is not None
    assert a.slot.target_id == expected_target
