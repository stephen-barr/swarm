"""
One agent's auction methods.
Only uses agent's position (to make a score) and winners table (to bid).

"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from swarm.models import (
    LOSES_TO_ALL,
    AgentConfig,
    CountDistribution,
    Frame,
    Message,
    Payload,
    PositionEstimate,
    Role,
    Slot,
    Winners,
)


@dataclass
class Target:
    target_id: int                          # id for target
    target_position: PositionEstimate       # position estimate of target
    threat_level: float                     # how dangerous is this target to my side
    threat_char_time: float                 # time in seconds for the value V of a target to be V/e.
    requirement_distribution: dict[
        Role,
        CountDistribution]                  # what is the estimated probability you need any number of a certain class of drones

    def mostLikely(self) -> dict[Role, int]:
        """
        Function takes in a requirement distribution and naively returns the mode.
        Slots must be integer valued, expectation would return float. The tie 
        breaker is the greater number. To be replaced when incorporating bayesian
        model.
        """
        most_likely = {
            role_: max(distribution, key=lambda n: (distribution[n], n))
            for role_, distribution in self.requirement_distribution.items()
        }                                                                    # return a dict of the mode of the needs per capability of a distribution
        return most_likely


class Agent:

    def __init__(self, config: AgentConfig, agent_position: PositionEstimate):
        if agent_position.frame is not Frame.GLOBAL:
            raise ValueError(
                f"agent {config.agent_id} needs a global-frame position"
            )                                                            # ensure concistency between measurements
        self.config = config
        self.agent_position = agent_position
        self.slot: Slot | None = None                                    # is this agent assigned to a slot
        self.inbox: list[Message[Payload]] = []
        self.winners: Winners = {}                                       # {slot : [score, agent]}
        self.census: dict[int, AgentConfig] = {
            config.agent_id: config
        }                                                                # agent config of every agent
        self.census_to_share: bool = True                                # true if agent's table has changed since last sent

    def outgoing(self) -> Payload:                                  # returns one payload per round
        census = None
        if self.census_to_share:
            census = self.census
            self.census_to_share = False                            # set flag for sharing census to False
        payload_to_be_sent: Payload = Payload(winners=self.winners,
                                              census=census)
        return payload_to_be_sent

    def updateWinners(self, messages: list[Message[Payload]]) -> bool:
        """
        Merge neighbours' tables into this agent's own, slot by slot.
        For each slot, keep whichever bid wins under beats(). Because each slot only
        keeps the best bid, the order messages arrive in doesn't matter, and getting
        the same table twice changes nothing. Bids only ever improve.
        """
        changed = False
        for m in messages:                            # messages for agent is sorted by agent and time
            for slot, bid in m.payload.winners.items():
                if beats(
                        bid,
                        self.winners.get(
                            slot, LOSES_TO_ALL
                        )                             # a slot that is not yet assigned is given LOSES_TO_ALL
                ):                                    # update only the slots that have changed or are not yet present
                    self.winners[slot] = bid
                    changed = True
        return changed

    def updateCensus(self, messages: list[Message[Payload]]) -> bool:
        """
        Merge agent_config information from neighbors into own census.
        """
        changed = False

        for m in messages:
            if m.payload.census is None:
                continue
            for agent_id, agent_config in m.payload.census.items():
                if agent_id not in self.census:
                    self.census[agent_id] = agent_config
                    changed = True
                    self.census_to_share = True
        return changed

    def makeScore(
        self, target: Target
    ) -> float:                                                          # build a naive weight based on threat level and time to arrival
        dist = math.dist(self.agent_position.position,
                         target.target_position.position)
        time_to_arrival = dist / self.config.platform.value.cruise_speed # return the estimated time for the agent to arrive at target
        score = target.threat_level * np.exp(
            -time_to_arrival / target.threat_char_time
        )                                                                # use exponential model to lower score as t increases
        return score

    def bid(self, slots_by_role: dict[Role, list[Slot]],
            targets_by_id: dict[int, Target]) -> dict[Slot, float]:
        scores = {
            t_id: self.makeScore(target)
            for t_id, target in targets_by_id.items()
        }                                             # score all of the targets by id

        ### score only the slots that agent can meet requirements
        return {
            s: scores[s.target_id]
            for required, role_slots in slots_by_role.items()
            if required <= self.config.capabilities for s in role_slots
        }

    def auctionPhase(self, slots_by_role: dict[Role, list[Slot]],
                     targets_by_id: dict[int, Target]) -> None:
        if self.slot is not None:
            return                                                          # already assigned
        bids = self.bid(slots_by_role, targets_by_id)
        contenders: dict[Slot, float] = {
            s: score
            for s, score in bids.items() if s not in self.winners or beats(
                (                                                           # only consider slots that are not filled
                    score, self.config.agent_id),
                self.winners[s])                                            # OR slots that agent can win.
        }
        if not contenders:
            return                                                          # no assignments for the agent

        best_slot: Slot = max(
            contenders, key=lambda s: (contenders[s], s.target_id, s.rank)
        )                                                                  # best slot by score; ties go to the higher target_id, then the higher rank
        self.winners[best_slot] = (
            contenders[best_slot], self.config.agent_id
        )                                                                  # write the agent as the winner to the best slot
        self.slot = best_slot                                              # assign myself the new slot

    def consensusPhase(self):
        if self.slot is None: return        # if unassigned, nothing to do
        _, winner_id = self.winners[self.slot]
        if winner_id != self.config.agent_id:
            self.slot = None                # if outbid on slot, release slot internally
            return

    def receieve(self, messages: list[Message[Payload]]):
        self.updateCensus(messages)
        self.updateWinners(messages)        # update winners from message list
        self.consensusPhase()               # check if you are still assigned


### comparison between two agents
### returns bool comparing (score, id). Id is arbitrary but tie breaks.
### future comparisons will compare time stamps as in async model
def beats(challenger: tuple, incumbent: tuple) -> bool:
    """
    True if `challenger` outranks `incumbent` for the same slot.
    Bids are (score, agent_id) tuples, higher score wins; on equal
    scores, the higher agent_id wins (this is arbitrary, but gets rid of ties
    and ensures convergence).
    """
    return challenger > incumbent


def buildSlots(
    targets: list[Target]
) -> list[Slot]:                                                      # return the possible slots for an array of targets
    return [
        Slot(target.target_id, role_, i) for target in targets
        for role_, n in target.mostLikely().items() for i in range(n)
    ]


def indexTargets(
    targets: list[Target]
) -> dict[int, Target]:           # look up table from id to target
    return {target.target_id: target for target in targets}


def groupSlots(
        slots: list[Slot]
) -> dict[Role, list[Slot]]:      # group slots by capability
    slots_by_role: dict[Role, list[Slot]] = defaultdict(list)
    for s in slots:
        slots_by_role[s.required].append(s)
    return slots_by_role
