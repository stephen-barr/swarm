"""
Contains all of the types imported by other modules. No logic.
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import NamedTuple

ThreeVector = tuple[float, float, float]
Covariance = tuple[ThreeVector, ThreeVector,
                   ThreeVector]                       # 3x3 covariance matrix
CountDistribution = dict[int, float]                  # distribution of agent needs per target


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
class PlatformSpec:
    """
    Cruise_speed is currently used.
    All other attributes are for later selection of tasks.
    """
    mass_kg: float      # mass of platform
    cruise_speed: float
    max_altitude: int   # max_altitude
    endurance_s: float  # endurance in seconds
    can_hover: bool


### Specifying possible drone classes
class PlatformClass(Enum):
    """
    Units: kg, m/s, ft, s
    Altitudes are given by drone group ceilings in ft
    All specs are given as an example
    """
    GROUP_1_MULTIROTOR = PlatformSpec(2.5, 12.0, 1_200, 35 * 60,
                                      True)                      # Skydio X2D-like
    GROUP_1_FIXED_WING = PlatformSpec(1.9, 13.0, 1_200, 90 * 60,
                                      False)                     # RQ-11 Raven-like
    GROUP_2 = PlatformSpec(22.0, 25.0, 3_500, 20 * 3600, False)  # ScanEagle-like
    GROUP_3 = PlatformSpec(210.0, 36.0, 18_000, 8 * 3600,
                           False)                                # RQ-7 Shadow-like
    GROUP_4 = PlatformSpec(1_630.0, 55.0, 18_000, 25 * 3600,
                           False)                                # MQ-1C Gray Eagle-like
    GROUP_5 = PlatformSpec(4_760.0, 90.0, 50_000, 27 * 3600,
                           False)                                # MQ-9 Reaper-like


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


class Slot(NamedTuple):           # one need (or slot) for an agent given by a target
    target_id: int
    required: Role                # lets target demand multiple roles for one target
    rank: int                     # (target_id, capability, index)


Winners = dict[Slot, tuple[
    float, int]]                  # slot assigned to [score, agent], the best bid an agent knows
                                  # order matters as beats() compares score to score then id to id


@dataclass
class Message:
    """
    Message is written by a single drone with a payload of Winners.

    deliveryStep is a property, so the switch from synchronous to
    asynchronous happens in one place.
    """
    send_id: int
    receiver_id: int
    payload: Winners    # assume synchronous communication rounds
    send_step: int
    message_delay: int = 0

    @property
    def deliveryStep(self) -> int:
        return self.send_step + self.message_delay    # using constant delayed model for now


@dataclass(frozen=True)
class AuctionResult:
    rounds: int
    winners: Winners    # the agreed table
    assignment: dict[
        int, Slot |
        None]           # agent_id -> slot, assignment[i] -> None implies agent i is unassigned
    total: float        # sum of winning scores, compared to optimal baseline
    filled: int         # counts slots with a winner


MessagesById = dict[int, list[Message]]               # {agent id : [Messages]}
LOSES_TO_ALL = (float("-inf"), float("-inf")
                )                                     # generic bid that loses to all bids
                                                      # allows updateWinners to not have cases

Graph = tuple[int, list[tuple[int, int]]]
