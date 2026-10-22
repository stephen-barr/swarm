import math

import numpy as np
import pytest

from swarm.agent import buildSlots, groupSlots, indexTargets
from swarm.models import Capability, Message, Payload, role
from swarm.scenarios import at, make_agent, make_target, solve

S = Capability.SURVEILLANCE
M = Capability.MAPPING


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
    a = make_agent(0, my_caps, 0, 0, scarcity_k=k)
    others = [
        make_agent(i + 1, caps, 0, 0, scarcity_k=k).config
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
        ([[M], [M], [M]], 1),     # only agent 0 can fill {M}
        ([[S], [S], [S]], 4),     # everyone can fill {S}
    ])
def test_weight_is_exp_minus_k_n_over_N(others_caps, n):
    k = 2.0
    a = agent_with_census([S], others_caps, k)
    target = make_target(0, at(100, 0), 0.5, {role(S): {1: 1.0}})
    slots_by_role, targets_by_id = setup([target])

    bids = a.bid(slots_by_role, targets_by_id)
    (_, b), = bids.items()        # exactly one slot
    raw = a.makeScore(target)
    N = len(a.census)

    assert a.capability_count[role(S)] == n
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
    a = agent_with_census([S, M], [[S], [S], [S]],
                          k)                          # {M} is scarce: only agent 0
    targets = [
        make_target(0, at(100, 0), 0.50, {role(S): {
                                              1: 1.0
                                          }}),        # common role
        make_target(1, at(0, 100), 0.45, {role(M): {
                                              1: 1.0
                                          }}),        # scarce role, same distance
    ]
    slots_by_role, targets_by_id = setup(targets)

    a.auctionPhase(slots_by_role, targets_by_id)

    assert a.slot is not None
    assert a.slot.target_id == expected_target


@pytest.mark.parametrize(
    "k, filled, drone0_role, drone1_filled",
    [
        (
            0.0, 1, role(S), False
        ),                                               # plain CBAA: drone 0 takes the valuable SURVEILLANCE slot, MAPPING stays empty
        (2.0, 2, role(M),
         True),                                          # re-scoring: drone 0 moves to the scarce MAPPING slot
    ])
def test_rescoring_moves_versatile_drone_to_scarce_slot(
        k, filled, drone0_role, drone1_filled):
    targets = [
        make_target(0, at(100.0, 0.0), 0.8,
                    {role(S): {
                         1: 1.0
                     }}),                                # common role, valuable, close to drone 0
        make_target(1, at(0.0, 300.0), 0.5,
                    {role(M): {
                         1: 1.0
                     }}),                                # scarce role: only drone 0 can fill it
    ]
    agents = [
        make_agent(0, {S, M}, 50.0, 0.0, scarcity_k=k),  # the versatile drone
        make_agent(1, {S}, 400.0, 0.0,
                   scarcity_k=k),                        # specialists, farther from target 0
        make_agent(2, {S}, 450.0, 50.0, scarcity_k=k),
        make_agent(3, {S}, 450.0, -50.0, scarcity_k=k),
    ]

    auction_results = solve(agents=agents, targets=targets)
    assert auction_results.filled == filled
    assert agents[0].slot is not None
    assert agents[0].slot.required == drone0_role
    assert (agents[1].slot is not None) == drone1_filled
    if drone1_filled:
        assert agents[1].slot is not None
        assert agents[1].slot.required == role(S)


def test_weight_accounts_for_demand():
    k = 2.0
    a = agent_with_census([S, M], [[S], [M], [S, M]], k)
    targets = [
        make_target(0, at(100, 0), 0.5, {role(S): {
                                             1: 1.0
                                         }}),         # 1 slot
        make_target(1, at(0, 100), 0.5, {role(M): {
                                             3: 1.0
                                         }}),         # 3 slots
    ]
    slots_by_role, targets_by_id = setup(targets)
    bids = a.bid(slots_by_role, targets_by_id)

    def weight(r):
        """The scarcity weight on role r's first slot: weighted bid / raw score."""
        slot = slots_by_role[r][0]
        return bids[slot].score / a.makeScore(targets_by_id[slot.target_id])

    n = {r: a.capability_count[r] for r in (role(S), role(M))} # supply
    d = {r: len(slots_by_role[r]) for r in (role(S), role(M))} # demand
    N = len(a.census)
    D = sum(len(slots) for slots in slots_by_role.values())

    assert weight(role(S)) == pytest.approx(
        np.exp(-k * (n[role(S)] / d[role(S)]) / (N / D)))
    assert weight(role(M)) == pytest.approx(
        np.exp(-k * (n[role(M)] / d[role(M)]) / (N / D)))
    assert weight(role(M)) > weight(role(S))
