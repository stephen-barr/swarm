from __future__ import annotations

from swarm.agent import (
    Agent,
    AgentConfig,
    Target,
    buildSlots,
    group_slots,
    index_targets,
)
from swarm.comms import Channel
from swarm.models import (
    Capability,
    Covariance,
    Frame,
    PlatformClass,
    PositionEstimate,
    Role,
    Slot,
    ThreeVector,
    role,
)


def run_round(t, agents: list[Agent], channel: Channel, slots_by_capability,
              targets_by_id) -> None:
    channel.beginStep(t)
    for a in agents:
        a.auctionPhase(slots_by_capability, targets_by_id)
    for a in agents:
        channel.transmitMessage(a.config.agent_id, a.outgoing())
    for a in agents:
        a.receieve(channel.deliverMessage(a.config.agent_id))
    channel.endStep()


def run_auction(agents: list[Agent],
                channel: Channel,
                slots_by_capability: dict[Role, list[Slot]],
                targets_by_id: dict[int, Target],
                max_rounds=200) -> int:
    for t in range(1, max_rounds + 1):
        before = [dict(a.winners) for a in agents]
        run_round(t, agents, channel, slots_by_capability, targets_by_id)
        if all(a.winners == b for a, b in zip(agents, before)):
            return t
    raise RuntimeError("auction did not converge")


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
