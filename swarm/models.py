"""
Contains all of the types imported by other modules. No logic.
"""

import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum, auto
from typing import Generic, NamedTuple, TypeVar

ThreeVector = tuple[float, float, float]
Covariance = tuple[ThreeVector, ThreeVector,
                   ThreeVector]                       # 3x3 covariance matrix
CountDistribution = dict[int, float]                  # distribution of agent needs per target

_g = 9.81


class Frame(Enum):
    GLOBAL = auto()
    LOCAL = auto()


@dataclass(frozen=True)
class PositionEstimate:
    """
    Position estimate of an agent or target.
    Covariance needed due to uncertainty from SLAM or vision front end.
    Velocity not used yet, will correct uncertainty.
    """
    frame: Frame                            # global or local
    position: ThreeVector                   # position in (x,y,z)
    velocity: ThreeVector                   # Euclidean velocity, not used yet, will be used when postion covariance added
    position_covariance: Covariance         # covariance matrix


@dataclass(frozen=True)
class AgentState:
    """Vehicles state as a """
    estimate: PositionEstimate
    bank: float = 0.0   # radians; fixed-wing only, 0 for multirotors


@dataclass(frozen=True)
class PlatformSpec:
    """
    Cruise_speed is currently used.
    All other attributes are for later selection of tasks.
    """
    mass_kg: float                # mass of platform
    v_min: float                  # minimum velocity, 0 if can_hover
    v_cruise: float               # cruise speed, previously cruise_speed
    v_max: float                  # max speed
    a_max: float                  # max acceleration (speeding up)
    a_min: float                  # max negative acceleration (slowing down)
    max_bank: float               # maximum bank in degrees
    max_altitude: int             # max_altitude
    endurance_s: float            # endurance in seconds
    max_roll_rate: float          # deg/s maximum roll rate estimated from wingspan, assumed constant

    @property
    def can_hover(self):
        """If the minimum velocity is zero, the agent, by definition, can hover."""
        return self.v_min == 0.0

    @property
    def max_lateral_accel(self) -> float:
        """Maximum acceleration to turn. Derived from maximum bank angle."""
        return _g * math.tan(math.radians(self.max_bank))

    @property
    def loiter_speed(self) -> float:
        """
        Rule of thumb tolerance of 1.25.
        Can be changed, but must be sufficiently above v_min.
        Good balance of saving fuel and being far enough from min.
        """
        return 1.25 * self.v_min

    @property
    def tau(self) -> float:
        """
        Returns time characteristic. 
        I.e. how many seconds to correct velocity error in harmonic system.
        """
        return self.max_bank / self.max_roll_rate

    @property
    def loiter_radius(self) -> float:
        """
        Smallest radius that can be flow in, calculated from max_lateral_accel.
        Tolerance is given as 1.4, rule of thumb, can be changed, but needs tolerance. """
        return 1.4 * self.loiter_speed * self.loiter_speed / self.max_lateral_accel


### Specifying possible drone classes
class PlatformClass(Enum):
    """
    Example specs per DoD UAS group. Speed envelopes follow public figures for the
    named aircraft; acceleration limits are engineering estimates.
    """
    GROUP_1_MULTIROTOR = PlatformSpec(mass_kg=2.5,
                                      v_min=0.0,
                                      v_cruise=12.0,
                                      v_max=16.0,
                                      a_max=5.0,
                                      a_min=-5.0,
                                      max_bank=27.0,
                                      max_altitude=1_200,
                                      max_roll_rate=300.0,
                                      endurance_s=35 * 60) # Skydio X2D-like
    GROUP_1_FIXED_WING = PlatformSpec(mass_kg=1.9,
                                      v_min=9.0,
                                      v_cruise=13.0,
                                      v_max=22.0,
                                      a_max=2.0,
                                      a_min=-1.5,
                                      max_bank=45.0,
                                      max_altitude=1_200,
                                      max_roll_rate=75,
                                      endurance_s=90 * 60) # RQ-11 Raven-like
    GROUP_2 = PlatformSpec(mass_kg=22.0,
                           v_min=18.0,
                           v_cruise=25.0,
                           v_max=41.0,
                           a_max=1.0,
                           a_min=-1.0,
                           max_bank=35.0,
                           max_altitude=3_500,
                           max_roll_rate=65,
                           endurance_s=20 * 3600)          # ScanEagle-like
    GROUP_3 = PlatformSpec(mass_kg=210.0,
                           v_min=25.0,
                           v_cruise=36.0,
                           v_max=56.0,
                           a_max=1.0,
                           a_min=-0.8,
                           max_bank=30.0,
                           max_altitude=18_000,
                           max_roll_rate=70,
                           endurance_s=8 * 3600)           # RQ-7 Shadow-like
    GROUP_4 = PlatformSpec(mass_kg=1_630.0,
                           v_min=30.0,
                           v_cruise=55.0,
                           v_max=77.0,
                           a_max=0.8,
                           a_min=-0.8,
                           max_bank=30.0,
                           max_altitude=18_000,
                           max_roll_rate=19,
                           endurance_s=25 * 3600)          # MQ-1C Gray Eagle-like
    GROUP_5 = PlatformSpec(mass_kg=4_760.0,
                           v_min=45.0,
                           v_cruise=90.0,
                           v_max=123.0,
                           a_max=1.0,
                           a_min=-1.0,
                           max_bank=30.0,
                           max_altitude=50_000,
                           max_roll_rate=26,
                           endurance_s=27 * 3600)          # MQ-9 Reaper-like


class Capability(Enum):
    SURVEILLANCE = auto()
    MAPPING = auto()
    COMMS_RELAY = auto()
    SIGNALS_COLLECTION = auto()
    PAYLOAD_DELIVERY = auto()
    SEARCH_AND_RESCUE = auto()
    STRIKE = auto()


Role = frozenset[Capability]      # set of capabilites one agent has all of


def role(*caps: Capability) -> Role:        # pass in capabilities get a role
    return frozenset(caps)


@dataclass(frozen=True)
class AgentConfig:
    agent_id: int                           # id for agent
    platform: PlatformClass                 # what platform is the agent
    capabilities: frozenset[Capability]     # what capabilities does the agent have


class Bid(NamedTuple):
    score: float
    agent_id: int
    version: int


class Slot(NamedTuple):           # one need (or slot) for an agent given by a target
    target_id: int
    required: Role                # lets target demand multiple roles for one target
    rank: int                     # (target_id, required, index)


Winners = dict[
    Slot, Bid]          # slot assigned to [score, agent], the best bid an agent knows
                        # order matters as beats() compares score to score then id to id

Census = dict[int, AgentConfig]

Latest = dict[int, int]           # agent_id : version table


class Payload(NamedTuple):
    winners: Winners
    census: Census | None = None
    latest: Latest | None = None


P = TypeVar("P")


@dataclass
class Message(Generic[P]):
    """
    Generic message for any type of payload.

    Message is written by a single drone.

    deliveryStep is a property, so the switch from synchronous to
    asynchronous happens in one place.
    """

    send_id: int
    receiver_id: int
    payload: P
    send_step: int
    message_delay: int = 0

    @property
    def deliveryStep(self) -> int:
        return self.send_step + self.message_delay    # using constant delayed model for now


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


@dataclass(frozen=True)
class AuctionResult:
    dynamics: bool                # were dynamics considered
    converged: bool               # did the auction converge
    rounds: int                   # number of rounds until auction converges
    winners: Winners              # the agreed table
    assignment: dict[
        int, Slot |
        None]                     # agent_id -> slot, assignment[i] -> None implies agent i is unassigned
    total: float                  # sum of winning scores, compared to optimal baseline
    filled: int                   # counts slots with a winner
    completion_time: float = 0    # total time for mission, set at zero so static examples cooperate.


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


def buildSlots(
    targets: list[Target]
) -> list[Slot]:                                                      # return the possible slots for an array of targets
    return [
        Slot(target.target_id, role_, i) for target in targets
        for role_, n in target.mostLikely().items() for i in range(n)
    ]


def raw_score(
    spec: PlatformSpec, position: ThreeVector, target: Target
) -> float:                                                   # naive scoring based on time to arrival, more info in README
    dist = math.dist(position, target.target_position.position)
    return target.threat_level * math.exp(
        -(dist / spec.v_cruise) / target.threat_char_time)


@dataclass(frozen=True)
class MissionData:
    targets_by_id: dict[int, Target]
    slots_by_role: dict[Role, list[Slot]]

    @classmethod
    def fromTargets(cls, targets: list[Target]):
        return cls(indexTargets(targets), groupSlots(buildSlots(targets)))

    def slotPosition(self, slot: Slot) -> ThreeVector:
        """Position of the target this slot belongs to."""
        return self.targets_by_id[slot.target_id].target_position.position


MessagesById = dict[int, list[Message]]     # {agent id : [Messages]}

LOSES_TO_ALL = Bid(float("-inf"), int(-sys.maxsize),
                   int(-sys.maxsize))                 # generic bid that loses to all bids
                                                      # allows updateWinners to not have cases

Graph = tuple[int, list[tuple[int, int]]]
