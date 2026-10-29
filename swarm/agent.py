"""
One agent's auction methods.
Agent uses winners table (a dict of slots: bids), census (list of other agents
capabilities) and latest (agent_id : highest message version).
Note that targets are assumed to be known by all and thus not distributed.

"""

from __future__ import annotations

import math
from itertools import combinations

import numpy as np

from swarm.models import (
    LOSES_TO_ALL,
    AgentConfig,
    Bid,
    Frame,
    Latest,
    Message,
    MissionData,
    Payload,
    PositionEstimate,
    Role,
    Slot,
    Target,
    Winners,
)


class Agent:

    def __init__(
            self,
            config: AgentConfig,
            agent_position: PositionEstimate,
            scarcity_k: float = 0.0):                                    # ensure consistency between measurements
        self.config = config
        if agent_position.frame is not Frame.GLOBAL:
            raise ValueError(
                f"agent {config.agent_id} needs a global-frame position"
            )                                                            # ensure agents frame is global
        self.agent_position = agent_position
        self.slot: Slot | None = None                                    # is this agent assigned to a slot
        self.inbox: list[Message[Payload]] = []
        self.winners: Winners = {}                                       # {slot : bid}
        self.last_bids: dict[Slot, Bid] = {}
                                                                         ### Census
        caps = list(config.capabilities)                                 # make a list of capabilities
        self.my_roles = [
            frozenset(cap) for r in
            range(1,
                  len(caps) +
                  1)                                                     # make a list of every combination of roles one possesses
            for cap in combinations(caps, r)
        ]

        self.census: dict[int, AgentConfig] = {       # agent config of every agent
            config.agent_id: config
        }
        self.capability_count: dict[Role, int] = {
            r: 1
            for r in self.my_roles
        }                                             # count of observed number of roles for scoring
        self.census_to_share: bool = True             # true if agent's table has changed since last sent
        self.scarcity_k = scarcity_k                  # how much do you weight the scarcity scoring
        self.reconsider: bool = False                 # ensures the auction does not take an early exit when k != 0

        ### Latest
        self.latest: Latest = {
            config.agent_id: 0
        }                         # dict of agent_id : version pairs
        self.latest_to_share: bool = True

        ### Timing
        self.mission_time: float = 0.0      # how many seconds have elapsed since the beginning of the mission (auction start)

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
        Rejects bids older than last known message.
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
                        bid.agent_id, 0
                ):                                    # check that the new message is at least as recent as the last message                                    # update only the slots that have changed or are not yet present
                    if bid.version > self.latest.get(bid.agent_id, 0):
                        self.latest_to_share = True   # flag that latest needs to be shared
                    self.winners[slot] = bid
                    self.latest[bid.agent_id] = bid.version
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
                if agent_id not in self.census:                                    # agent configs are only added at most once and not changed
                    self.census[
                        agent_id] = agent_config                                   # add the config to the census
                    overlap = self.config.capabilities & agent_config.capabilities # intersection of your capabilities and their capabilities
                    for r in self.capability_count:
                        if r <= overlap:
                            self.capability_count[
                                r] += 1                                            # increment the count of the role for small n scoring
                    changed = True
                    self.census_to_share = True
        return changed

    def mergeLatest(self, messages: list[Message[Payload]]) -> bool:
        """
        Merge the most recent messages (based on version) to latest table.
        """

        changed = False
        changed_versions = set()                               # set of agent_ids that changed versions
        for m in messages:
            if m.payload.latest is None:
                continue
            for agent_id, new_version in m.payload.latest.items():
                if agent_id not in self.latest or self.latest[
                        agent_id] < new_version:
                    self.latest[
                        agent_id] = new_version                # if the agent id hasnt been heard from yet, or the version is newer add version to latest
                    changed_versions.add(
                        agent_id)                              # add it to the list of agents to change
                    changed = True
                    self.latest_to_share = True                # set flag to share latest
        if changed_versions:
            for s, b in list(self.winners.items()):
                if b.agent_id in changed_versions:
                    self.winners.pop(
                        s)                                     # delete any winners with more recent messages
        return changed

    def makeScore(
        self, target: Target
    ) -> float:                                                   # build a naive weight based on threat level and time to arrival
        dist = math.dist(self.agent_position.position,
                         target.target_position.position)
        time_to_arrival = dist / self.config.platform.value.v_max # return the estimated time for the agent to arrive at target
        score = target.threat_level * np.exp(
            -time_to_arrival / target.threat_char_time
        )                                                         # use exponential model to lower score as t increases
        return score

    def bid(self, mission_data: MissionData) -> dict[Slot, Bid]:

        # score all of the targets by id
        scores = {
            t_id: self.makeScore(target)
            for t_id, target in mission_data.targets_by_id.items()
        }
        # total number of slots for the auction
        total_slots = sum(
            len(slots) for slots in mission_data.slots_by_role.values())

        # number of drones agent knows of / number of slots
        drones_slot_ratio = len(self.census) / total_slots

        ### scarcity weight for each role this agent can fill. Calculated as a ratio of
        ### the scarcity of the number of drones per role, relative to the swarm average
        role_to_weight = {
            role:
            math.exp(-self.scarcity_k *
                     (self.capability_count[role] /
                      len(mission_data.slots_by_role[role])) /
                     drones_slot_ratio)
            for role in mission_data.slots_by_role
            if role <= self.config.capabilities
        }

        ### one weighted bid per slot of an eligible role
        bids = {
            s:
            Bid(score=scores[s.target_id] * w,
                agent_id=self.config.agent_id,
                version=self.latest[self.config.agent_id])
            for role, w in role_to_weight.items()
            for s in mission_data.slots_by_role[role]
        }

        self.last_bids = bids     # now used in movement
        return bids

    def auctionPhase(self, mission_data: MissionData) -> None:
        if self.slot is not None and not self.reconsider:
            return                                                      # already assigned and not reconsidering
        bids = self.bid(mission_data)
        contenders: dict[Slot, Bid] = {
            s: b
            for s, b in bids.items() if s not in self.winners or beats(
                b,                                                      # only consider slots that are not filled
                self.winners[s])                                        # OR slots that agent can win.
        }
        if self.slot is not None:
            contenders[self.slot] = bids[
                self.slot]                                              # if agent has a slot add it to contenders

        if not contenders:
            return      # no assignments for the agent

        best_slot: Slot = max(
            contenders,
            key=lambda s: (contenders[s].score, s.target_id, s.rank)
        )                                                            # best slot by score; ties go to the higher target_id, then the higher rank

        if self.slot is not None and best_slot == self.slot and contenders[
                best_slot].score == self.winners[self.slot].score:
            return      # if the best slot is already owned by you simply return and transmit nothing

        self.latest[
            self.config.
            agent_id] += 1                  # increment agent version by one (you are sending a new message)
        self.latest_to_share = True         # set flag to true as your own version changed

        self.winners[best_slot] = Bid(
            score=contenders[best_slot].score,
            agent_id=self.config.agent_id,
            version=self.latest[self.config.agent_id]
        )                                             # write the agent as the winner to the best slot
        if self.slot is not None and self.slot != best_slot:
            self.winners.pop(
                self.slot
            )                                         # remove yourself from the old slot only if you are filling a new slot
        self.slot = best_slot                         # assign myself the new slot

    def consensusPhase(self):
        if self.slot is None: return        # if unassigned, nothing to do
        bid_agent_id = self.winners[self.slot].agent_id
        if bid_agent_id != self.config.agent_id:
            self.slot = None                # if outbid on slot, release slot internally
            self.latest[
                self.config.
                agent_id] += 1              # if outbid flag latest to let other agents know
            self.latest_to_share = True
            return

    def receieve(self, messages: list[Message[Payload]]):
        census_grew = self.updateCensus(messages)
        self.mergeLatest(messages)
        self.updateWinners(messages)                           # update winners from message list
        self.consensusPhase()                                  # check if you are still assigned
        self.reconsider = census_grew and self.scarcity_k != 0 # if the census grew you must reconsider next round, if k=0 default to standard CBAA


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
