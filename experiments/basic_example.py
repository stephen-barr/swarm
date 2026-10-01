from __future__ import annotations

from swarm.agent import (
    Agent,
    AgentConfig,
    Target,
    buildSlots,
    group_slots,
    index_targets,
)
from swarm.auction import run_auction
from swarm.comms import Channel
from swarm.models import (
    Capability,
    Covariance,
    Frame,
    PlatformClass,
    PositionEstimate,
    ThreeVector,
    role,
)


#### Example
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
            threat_char_time=60.0,
            requirement_distribution={
                role(Capability.SURVEILLANCE): {
                    1: 0.2,
                    2: 0.7,
                    3: 0.1
                }
            },
        ),
        Target(
            target_id=1,
            target_position=at(0.0, 400.0),
            threat_level=0.3,
            threat_char_time=60.0,
            requirement_distribution={
                role(Capability.SURVEILLANCE): {
                    1: 0.9,
                    2: 0.1
                },
                role(Capability.MAPPING): {
                    1: 0.6,
                    0: 0.4
                },
            },
        ),
    ]

    agents = [
        Agent(
            AgentConfig(0, PlatformClass.GROUP_1_MULTIROTOR,
                        frozenset({Capability.SURVEILLANCE}), 12.0),
            at(10.0, 0.0)),
        Agent(
            AgentConfig(1, PlatformClass.GROUP_1_MULTIROTOR,
                        frozenset({Capability.SURVEILLANCE}), 12.0),
            at(0.0, 350.0)),
        Agent(
            AgentConfig(
                2, PlatformClass.GROUP_2,
                frozenset({Capability.SURVEILLANCE, Capability.MAPPING}),
                25.0), at(50.0, 50.0)),
        Agent(
            AgentConfig(3, PlatformClass.GROUP_2,
                        frozenset({Capability.STRIKE}), 25.0), at(0.0, 0.0)),
    ]

    targets_by_id = index_targets(targets)
    slots = buildSlots(targets)
    slots_by_capability = group_slots(slots)

    ids = [a.config.agent_id for a in agents]
    complete = {i: [j for j in ids if j != i] for i in ids}
    channel = Channel(complete)

    print(
        f"{len(slots)} slots: "
        f"{[(s.target_id, '+'.join(sorted(c.name for c in s.required)), s.rank) for s in slots]}\n"
    )

    rounds = run_auction(agents, channel, slots_by_capability, targets_by_id)

    print(f"converged in {rounds} rounds\n")
    for agent in agents:
        if agent.slot is None:
            print(f"agent {agent.config.agent_id}: unassigned")
        else:
            score, _ = agent.winners[agent.slot]
            s = agent.slot
            print(
                f"agent {agent.config.agent_id}: target {s.target_id} "
                f"{'+'.join(sorted(c.name for c in s.required))} rank {s.rank}  (score {score:.1f})"
            )


if __name__ == "__main__":
    main()
