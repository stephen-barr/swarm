from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from swarm.agent import Agent, Target
from swarm.models import Slot


def createScoreMatrix(
    agents: list[Agent],
    slots: list[Slot],
    targets_by_id: dict[int, Target],
) -> np.ndarray:
    """
    Create a score matrix where m[i,j] is score of agent i, slot j.
    """
    unassigned = 0.0
    m = np.full((len(agents), len(slots)), unassigned)
    for i, agent in enumerate(agents):
        for j, slot in enumerate(slots):
            if agent.config.capabilities >= slot.required:
                m[i, j] = agent.makeScore(targets_by_id[
                    slot.target_id])                     # make score if pairing is eligible
    return m


def solveHungarian(
    agents: list[Agent],
    slots: list[Slot],
    targets_by_id: dict[int, Target],
) -> tuple[float, int]:
    """
    Optimal assignment under the same one-slot-per-agent rule as the auction.

    Centralised algorithm: the function gets every agent's score directly.
    Ineligible pairs are filled with 0. Real scores are strictly
    positive under discounted scoring, so `chosen > 0` separates real
    assignments from forced placeholder pairs.

    Returns (total score, slots filled)
    """
    m = createScoreMatrix(agents, slots, targets_by_id)

    agent_assignment, slot_assignment = linear_sum_assignment(m, maximize=True)

    chosen = m[agent_assignment,
               slot_assignment]             # create an array of only the assigned scores
    real = chosen[chosen > 0]               # boolean mask away any that are unassigned
    return float(real.sum()), int(
        real.size)                          # returns (score, number of filled entries)


def greedyBaseline(
    agents: list[Agent],
    slots: list[Slot],
    targets_by_id: dict[int, Target],
) -> tuple[float, int]:
    """
    Establish the greedy algorithm baseline that differs from auction only on ties
    (same agent,target - different slot).
    Serves as a baseline for divergence of auction from greedy on tie breaking

    Assign the highest scoring pair, zero the corresponding agent and slot scores
    (row and column) and continue until no eligible pairs are left.
    """
    unassigned = 0.0

    m = createScoreMatrix(agents, slots, targets_by_id)
    number_of_assignments = 0
    total_score = 0
    while m.max() > unassigned:
        i, j = np.unravel_index(
            m.argmax(),
            m.shape)              # return the row and column of the maximum score
        total_score += m[i, j]
        number_of_assignments += 1
        m[i, :] = unassigned      # zero out the row and column
        m[:, j] = unassigned
    return total_score, number_of_assignments
