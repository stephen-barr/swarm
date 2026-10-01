from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from swarm.agent import Agent, Target
from swarm.models import FORBIDDEN, Slot, Winners

### Hungarian solver baseline to compare
### distributed solver to optimal solution


def solveHungarian(agents: list[Agent], slots: list[Slot],
                   targets_by_id: dict[int, Target]) -> tuple[float, int]:

    # create a matrix where i,j = score(agent, slot)
    m = np.full((len(agents), len(slots)), FORBIDDEN)
    for i, agent in enumerate(agents):
        for j, slot in enumerate(slots):
            if agent.config.capabilities >= slot.required:
                m[i, j] = agent.makeScore(targets_by_id[slot.target_id])
    agent_assignment, slot_assignment = linear_sum_assignment(m, maximize=True)

    # calculate the score for the
    chosen = m[agent_assignment,
               slot_assignment]             # create an array of only the assigned scores
    real = chosen[chosen
                  != FORBIDDEN]             # boolean mask away any that are unnassigned
    return float(real.sum()), int(
        real.size)                          # returns (score, number of filled entries)


### helper function that returns score, number of assignments
### for distributed solution


def auctionTotal(winners: Winners) -> tuple[float, int]:
    return sum((score for score, _ in winners.values()), 0.0), len(winners)
