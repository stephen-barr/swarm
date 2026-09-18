import random

import pytest

from agent import (
    Agent,
    AgentConfig,
    Capability,
    Covariance,
    Frame,
    PlatformClass,
    PositionEstimate,
    Target,
    ThreeVector,
    buildSlots,
    group_slots,
    index_targets,
    run_auction,
)

