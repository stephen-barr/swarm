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
    Bid,
    CountDistribution,
    Frame,
    Latest,
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
            )                                                            # ensure consistency between measurements
        self.config = config
        self.agent_position = agent_position
        self.slot: Slot | None = None                                    # is this agent assigned to a slot
        self.inbox: list[Message[Payload]] = []
        self.winners: Winners = {}                                       # {slot : bid}
        self.census: dict[int, AgentConfig] = {
            config.agent_id: config
        }                                                                # agent config of every agent
        self.latest: Latest = {
            config.agent_id: 0
        }                                                                # dict of agent_id : version pairs
        self.census_to_share: bool = True                                # true if agent's table has changed since last sent
        self.latest_to_share: bool = True
        self.version: int = 0                                            # version number dictates rebids, only changes

    def outgoing(self) -> Payload:                                  # returns one payload per round
        census = None
        if self.census_to_share:
            census = self.census
            self.census_to_share = False                            # set flag for sharing census to False
        latest = None
        if self.latest_to_share:
            latest = self.latest
            self.latest_to_share = False
        payload_to_be_sent: Payload = Payload(winners=self.winners,
                                              census=census,
                                              latest=latest)
        return payload_to_be_sent

    def updateWinners(self, messages: list[Message[Payload]]) -> bool:
        """
        Merge neighbours' tables into this agent's own, slot by slot.
        For each slot, keep whichever bid wins under beats(). 
        Checks if the new bid has a higher version than the old.
        """
        changed = False
        for m in messages:                            # messages for agent is sorted by agent and time
            for slot, bid in m.payload.winners.items():
                if beats(
                        bid,
                        self.winners.get(
                            slot, LOSES_TO_ALL
                        )                             # a slot that is not yet assigned is given LOSES_TO_ALL
                ) and bid.version >= self.latest.get(
                        bid.agent, 0
                ):                                    # check that the new message is more recent than the most recent message                                    # update only the slots that have changed or are not yet present
                    self.winners[slot] = bid
                    self.latest[bid.agent] = bid.version
                    self.latest_to_share = True       # flag that latest needs to be shared
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

    def mergeLatest(self, messages: list[Message[Payload]]) -> bool:
        """
        Merge the most recent messages (based on version) to latest table.
        """

        changed = False
        changed_versions = set()
        for m in messages:
            if m.payload.latest is None:
                continue
            for agent_id, new_version in m.payload.latest.items():
                if agent_id not in self.latest or self.latest[
                        agent_id] < new_version:
                    self.latest[
                        agent_id] = new_version       # if the agent id hasnt been heard from yet, or the version is newer add version to latest
                    changed_versions.add(
                        agent_id)                     # add it to the list of agents to change
                    changed = True
                    self.latest_to_share = True       # set flag to share latest
        if changed_versions:
            for s, b in list(self.winners.items()):
                if b.agent in changed_versions:
                    self.winners.pop(
                        s)                            # delete any winners with more recent messages
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
            targets_by_id: dict[int, Target]) -> dict[Slot, Bid]:
        scores = {
            t_id: self.makeScore(target)
            for t_id, target in targets_by_id.items()
        }                                             # score all of the targets by id

        ### score only the slots that agent can meet requirements
        return {
            s:
            Bid(score=scores[s.target_id],
                agent=self.config.agent_id,
                version=self.version)
            for required, role_slots in slots_by_role.items()
            if required <= self.config.capabilities for s in role_slots
        }

    def auctionPhase(self, slots_by_role: dict[Role, list[Slot]],
                     targets_by_id: dict[int, Target]) -> None:
        if self.slot is not None:
            return                                                      # already assigned
        bids = self.bid(slots_by_role, targets_by_id)
        contenders: dict[Slot, Bid] = {
            s: b
            for s, b in bids.items() if s not in self.winners or beats(
                b,                                                      # only consider slots that are not filled
                self.winners[s])                                        # OR slots that agent can win.
        }
        if not contenders:
            return                                                      # no assignments for the agent

        best_slot: Slot = max(
            contenders,
            key=lambda s: (contenders[s].score, s.target_id, s.rank)
        )                                                            # best slot by score; ties go to the higher target_id, then the higher rank
        self.winners[best_slot] = Bid(
            score=contenders[best_slot].score,
            agent=self.config.agent_id,
            version=self.version
        )                                                            # write the agent as the winner to the best slot
        self.slot = best_slot                                        # assign myself the new slot

    def consensusPhase(self):
        if self.slot is None: return        # if unassigned, nothing to do
        bid_agent_id = self.winners[self.slot].agent
        if bid_agent_id != self.config.agent_id:
            self.slot = None                # if outbid on slot, release slot internally
            return

    def receieve(self, messages: list[Message[Payload]]):
        self.updateCensus(messages)
        self.mergeLatest(messages)
        self.updateWinners(messages)        # update winners from message list
        self.consensusPhase()               # check if you are still assigned


### comparison between two agents
### returns bool comparing score, note version has primacy from the other functions
def beats(challenger: Bid, incumbent: Bid) -> bool:
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
