from dataclasses import dataclass
from enum import Enum, auto
from typing import NamedTuple

ThreeVector = tuple[float, float, float]
Covariance = tuple[ThreeVector, ThreeVector, ThreeVector] # probability of any
CountDistribution = dict[int, float]                      # distribution of agent needs per target


class Frame(Enum):
    GLOBAL = auto()
    LOCAL = auto()


@dataclass(frozen=True)
class PositionEstimate:
    frame: Frame                            # global or local
    position: ThreeVector                   # position in (x,y,z)
    velocity: ThreeVector                   # Euclidean velocity
    position_covariance: Covariance         # covariance matrix


### Drone class
@dataclass(frozen=True)
class PlatformSpec:
    mass_kg: float      # mass of platform
    cruise_speed: float
    max_altitude: int   # max_altitude
    endurance_s: float  # endurance in seconds
    can_hover: bool


### Specifying possible drone classes
class PlatformClass(Enum):
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


Role = frozenset[Capability]


def role(*caps: Capability) -> Role:        # pass in capabilities get a role
    return frozenset(caps)


class Slot(NamedTuple):
    target_id: int
    required: Role      # possible multiple capabilities for one slot
    rank: int           # (target_id, capability, index)


Winners = dict[Slot, tuple[float, int]]     # slot assigned to [score, agent]


@dataclass
class Message:
    send_id: int
    receiver_id: int
    payload: Winners    # assume synchronous communication rounds
    send_step: int
    message_delay: int = 0

    @property
    def deliveryStep(self) -> int:
        return self.send_step + self.message_delay    # using contant delayed model for now


@dataclass(frozen=True)
class AuctionResult:
    rounds: int
    winners: Winners                        # the agreed table
    assignment: dict[int, Slot | None]      # agent_id -> slot
    total: float
    filled: int


MessagesById = dict[int, list[Message]]               # {agent id : [Messages]}
LOSES_TO_ALL = (float("-inf"), float("-inf")
                )                                     # generic [score, agent] for look up methods
