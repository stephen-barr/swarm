from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum, auto
from typing import NamedTuple

ThreeVector = tuple[float, float, float]
Covariance = tuple[ThreeVector, ThreeVector, ThreeVector]       # probability of any 
CountDistribution = dict[int, float]

class Frame(Enum):
    GLOBAL = auto()
    LOCAL = auto()


@dataclass(frozen=True)
class PositionEstimate:
    frame: Frame  # either frame or odometry
    position: ThreeVector  # position in (x,y,z)
    velocity: ThreeVector  # Euclidean velocity
    position_covariance: Covariance  # covariance matrix


@dataclass(frozen=True)
class PlatformSpec:
    mass_kg: float  # mass of platform
    cruise_speed: float
    max_altitude: int  # max_altitude
    endurance_s: float  # endurance in seconds
    can_hover: bool


class PlatformClass(Enum):
    GROUP_1_MULTIROTOR = PlatformSpec(2.5, 12.0, 1_200, 35 * 60, True)       # Skydio X2D-like
    GROUP_1_FIXED_WING = PlatformSpec(1.9, 13.0, 1_200, 90 * 60, False)      # RQ-11 Raven-like
    GROUP_2 = PlatformSpec(22.0, 25.0, 3_500, 20 * 3600, False)              # ScanEagle-like
    GROUP_3 = PlatformSpec(210.0, 36.0, 18_000, 8 * 3600, False)             # RQ-7 Shadow-like
    GROUP_4 = PlatformSpec(1_630.0, 55.0, 18_000, 25 * 3600, False)          # MQ-1C Gray Eagle-like
    GROUP_5 = PlatformSpec(4_760.0, 90.0, 50_000, 27 * 3600, False)          # MQ-9 Reaper-like

class Capability(Enum):
    SURVEILLANCE = auto()
    MAPPING = auto()
    COMMS_RELAY = auto()
    SIGNALS_COLLECTION = auto()
    PAYLOAD_DELIVERY = auto()
    SEARCH_AND_RESCUE = auto()
    STRIKE = auto()

class Slot(NamedTuple):
    target_id: int
    capability: Capability
    rank: int          # (target_id, capability, index)
    
Winners = dict[Slot, tuple[float, int]]     # slot assigned to [score, agent]

@dataclass
class AgentConfig:
    agent_id: int                           # id for agent
    platform: PlatformClass                 # what platform is the agent
    capabilities: frozenset[Capability]     # what capabilities does the agent have
    cruise_speed: float                     # how fast is the agent

@dataclass
class Target:
    target_id: int                          # id for target
    target_position: PositionEstimate              # position estimate of target
    threat_level: float                     # how dangerous is this target to my side
    requirement_distribution: dict[Capability, CountDistribution]   # what is the estimated probability you need any number of a certain class of drones

    def mostLikely(self) -> dict[Capability, int]:
        most_likely = {capability: max(distribution, key=lambda n: distribution[n])
            for capability, distribution in self.requirement_distribution.items()}   # return a dict of the expected needs per capability of a distribution
        return most_likely
class Agent:
    def __init__(self, config : AgentConfig, agent_position: PositionEstimate):
        if agent_position.frame is not Frame.GLOBAL:
            raise ValueError(f"agent {config.agent_id} needs a global-frame position")      # ensure concistency between measurements
        self.config = config
        self.agent_position = agent_position
        self.slot: Slot | None = None                       # is this agent assigned to a slot

    def canServe(self, target: Target):
        return bool(self.config.capabilities & target.requirement_distribution.keys())

    def makeScore(self, target: Target) -> float: # build a naive weight based on time to arrival
            dist = math.dist(self.agent_position.position, target.target_position.position)
            time_to_arrival = dist/self.config.cruise_speed
            return target.threat_level*100 - time_to_arrival        # just a naive weight

    def bid(self, slots_by_capability: dict[Capability, list[Slot]], targets_by_id: dict[int, Target]) -> dict[Slot, float]:
            scores = {tid : self.makeScore(t) for tid, t in targets_by_id.items()}     # score all of the targets by id
            
            return {s : scores[s.target_id]                        # return the scores for each set of slots matching capability
                for capability in self.config.capabilities
                for s in slots_by_capability.get(capability, [])}

### Only consider slots that are not filled OR slots that agent can win.
    def auctionPhase(self, slots_by_capability: dict[Capability, list[Slot]], targets_by_id: dict[int, Target], winners: Winners) -> None:
        if self.slot is not None:
            return      # already assigned
        bids = self.bid(slots_by_capability, targets_by_id)
        contenders: dict[Slot, float] = {s : score for s, score in bids.items()
                            if s not in winners or beats((score, self.config.agent_id), winners[s])
                            }
        if not contenders: 
                return      # no assignments for the agent

        best_slot: Slot = max(contenders, key=lambda s: (contenders[s], s.target_id, s.rank))        # take the max of all of the slots by score, then by highest target_id
        winners[best_slot] = (contenders[best_slot], self.config.agent_id)      # write the agent as the winner to the best slot
        self.slot = best_slot       # assign myself the new slot

    def consensusPhase(self, winners: Winners):
        if self.slot is None: return
        _ , id = winners[self.slot]
        if id != self.config.agent_id: 
            self.slot = None
            return
            

### comparison between two agents 
### returns bool comparing (score, id). Id is arbitrary but tie breaks.
### future comparisons will compare time stamps as in async model
def beats(challenger: tuple, incumbent: tuple) -> bool: 
    return challenger > incumbent


def buildSlots(targets: list[Target]) -> list[Slot]:    # return the possible slots for an array of targets
    return[
    Slot(target.target_id, capability, i)
    for target in targets
    for capability, n in target.mostLikely().items()
    for i in range(n)]

def index_targets(targets: list[Target]) -> dict[int, Target]:      # look up table from id to target
    return {target.target_id: target for target in targets}

def group_slots(slots: list[Slot]) -> dict[Capability, list[Slot]]: # group slots by capability
    slots_by_capability: dict[Capability, list[Slot]] = defaultdict(list)
    for s in slots:
        slots_by_capability[s.capability].append(s)        
    return slots_by_capability

def run_auction(agents, slots_by_capability, targets_by_id, max_rounds=100):
    winners: Winners = {}
    for round_num in range(max_rounds):
        before = {a.config.agent_id: a.slot for a in agents}
        for a in agents:
            a.auctionPhase(slots_by_capability, targets_by_id, winners)
        for a in agents:
            a.consensusPhase(winners)
        if before == {a.config.agent_id: a.slot for a in agents}:
            return winners, round_num
    raise RuntimeError("auction did not converge")

def main() -> None:
    zero_cov: Covariance = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    still: ThreeVector = (0.0, 0.0, 0.0)

    def at(x: float, y: float, z: float = 0.0) -> PositionEstimate:
        return PositionEstimate(Frame.GLOBAL, (x, y, z), still, zero_cov)

    targets = [
        Target(
            target_id=0,
            target_position=at(100.0, 0.0),
            threat_level=0.8,
            requirement_distribution={
                Capability.SURVEILLANCE: {1: 0.2, 2: 0.7, 3: 0.1},
            },
        ),
        Target(
            target_id=1,
            target_position=at(0.0, 400.0),
            threat_level=0.3,
            requirement_distribution={
                Capability.SURVEILLANCE: {1: 0.9, 2: 0.1},
                Capability.MAPPING: {1: 0.6, 0: 0.4},
            },
        ),
    ]

    agents = [
        Agent(AgentConfig(0, PlatformClass.GROUP_1_MULTIROTOR,
                          frozenset({Capability.SURVEILLANCE}), 12.0), at(10.0, 0.0)),
        Agent(AgentConfig(1, PlatformClass.GROUP_1_MULTIROTOR,
                          frozenset({Capability.SURVEILLANCE}), 12.0), at(0.0, 350.0)),
        Agent(AgentConfig(2, PlatformClass.GROUP_2,
                          frozenset({Capability.SURVEILLANCE, Capability.MAPPING}), 25.0), at(50.0, 50.0)),
        Agent(AgentConfig(3, PlatformClass.GROUP_2,
                          frozenset({Capability.STRIKE}), 25.0), at(0.0, 0.0)),
    ]

    targets_by_id = index_targets(targets)
    slots = buildSlots(targets)
    slots_by_capability = group_slots(slots)

    print(f"{len(slots)} slots: {[(s.target_id, s.capability.name, s.rank) for s in slots]}\n")

    winners, rounds = run_auction(agents, slots_by_capability, targets_by_id)

    print(f"converged in {rounds + 1} rounds\n")
    for agent in agents:
        if agent.slot is None:
            print(f"agent {agent.config.agent_id}: unassigned")
        else:
            score, _ = winners[agent.slot]
            s = agent.slot
            print(f"agent {agent.config.agent_id}: target {s.target_id} "
                  f"{s.capability.name} rank {s.rank}  (score {score:.1f})")


if __name__ == "__main__":
    main()