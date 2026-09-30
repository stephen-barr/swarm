from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from swarm.models import (
    LOSES_TO_ALL,
    Capability,
    CountDistribution,
    Frame,
    Message,
    PlatformClass,
    PositionEstimate,
    Role,
    Slot,
    Winners,
)


@dataclass
class AgentConfig:
    agent_id: int                           # id for agent
    platform: PlatformClass                 # what platform is the agent
    capabilities: frozenset[Capability]     # what capabilities does the agent have
    cruise_speed: float                     # how fast is the agent


@dataclass
class Target:
    target_id: int                                        # id for target
    target_position: PositionEstimate                     # position estimate of target
    threat_level: float                                   # how dangerous is this target to my side
    threat_char_time: float                               # time in seconds for the value of a target to half
    requirement_distribution: dict[
        Role,
        CountDistribution]                                # what is the estimated probability you need any number of a certain class of drones
    agent_winners: dict[Slot, tuple[float, int]] = field(
        default_factory=dict, init=False)                 # who is assigned to this target

    def mostLikely(self) -> dict[Role, int]:
        most_likely = {
            role_: max(distribution, key=lambda n: distribution[n])
            for role_, distribution in self.requirement_distribution.items()
        }                                                                    # return a dict of the expected needs per capability of a distribution
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
        self.inbox: list[Message] = []
        self.winners: Winners = {}                                       # {slot : [score, agent]}

    def outgoing(self) -> Winners:          # return winners to channel
        return self.winners

    def updateWinners(self, messages_for_agent: list[Message]) -> bool:
        changed = False
        for m in messages_for_agent:                              # messages for agent is sorted by agent and time
            for slot, bid in m.payload.items():
                if beats(
                        bid, self.winners.get(slot, LOSES_TO_ALL)
                ):                                                # update only the slots that have changed or are not yet present
                    self.winners[slot] = bid
                    changed = True
        return changed

    def makeScore(
        self, target: Target
    ) -> float:                                        # build a naive weight based on threat level and time to arrival
        dist = math.dist(self.agent_position.position,
                         target.target_position.position)
        time_to_arrival = dist / self.config.cruise_speed
        score = target.threat_level * np.exp(
            -time_to_arrival / target.threat_char_time)
        return score
                                                       ### A target is only worth reaching within threat_level × 100 seconds

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

### Only consider slots that are not filled OR slots that agent can win.

    def auctionPhase(self, slots_by_role: dict[Role, list[Slot]],
                     targets_by_id: dict[int, Target]) -> None:
        if self.slot is not None:
            return                                                           # already assigned
        bids = self.bid(slots_by_role, targets_by_id)
        contenders: dict[Slot, float] = {
            s: score
            for s, score in bids.items() if s not in self.winners or beats((
                score, self.config.agent_id), self.winners[s])
        }
        if not contenders:
            return                                                           # no assignments for the agent

        best_slot: Slot = max(
            contenders, key=lambda s: (contenders[s], s.target_id, s.rank)
        )                                                                  # take the max of all of the slots by score, then by highest target_id
        self.winners[best_slot] = (
            contenders[best_slot], self.config.agent_id
        )                                                                  # write the agent as the winner to the best slot
        self.slot = best_slot                                              # assign myself the new slot

    def consensusPhase(self):
        if self.slot is None: return        # if unassigned, nothing to do
        _, winner_id = self.winners[self.slot]
        if winner_id != self.config.agent_id:
            self.slot = None
            return

    def receieve(self, messages: list[Message]):
        self.updateWinners(messages)        # update winners from message list
        self.consensusPhase()               # check if you are still assigned


### comparison between two agents
### returns bool comparing (score, id). Id is arbitrary but tie breaks.
### future comparisons will compare time stamps as in async model
def beats(challenger: tuple, incumbent: tuple) -> bool:
    return challenger > incumbent


def buildSlots(
    targets: list[Target]
) -> list[Slot]:                                                      # return the possible slots for an array of targets
    return [
        Slot(target.target_id, role_, i) for target in targets
        for role_, n in target.mostLikely().items() for i in range(n)
    ]


def index_targets(
    targets: list[Target]
) -> dict[int, Target]:           # look up table from id to target
    return {target.target_id: target for target in targets}


def group_slots(
        slots: list[Slot]
) -> dict[Role, list[Slot]]:      # group slots by capability
    slots_by_role: dict[Role, list[Slot]] = defaultdict(list)
    for s in slots:
        slots_by_role[s.required].append(s)
    return slots_by_role
