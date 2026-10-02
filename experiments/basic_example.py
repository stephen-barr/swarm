from __future__ import annotations

from swarm.agent import buildSlots
from swarm.models import Capability, PlatformClass, role
from swarm.scenarios import at, make_agent, make_target, solve

S = Capability.SURVEILLANCE
M = Capability.MAPPING


def main() -> None:
    targets = [
        make_target(0, at(100.0, 0.0), 0.8,
                    {role(S): {
                         1: 0.2,
                         2: 0.7,
                         3: 0.1
                     }}),
        make_target(1, at(0.0, 400.0), 0.3, {
            role(S): {
                1: 0.9,
                2: 0.1
            },
            role(M): {
                1: 0.6,
                0: 0.4
            },
        }),
    ]

    agents = [
        make_agent(0, {S}, 10.0, 0.0),
        make_agent(1, {S}, 0.0, 350.0),
        make_agent(2, {S, M}, 50.0, 50.0, platform=PlatformClass.GROUP_2),
        make_agent(3, {Capability.STRIKE},
                   0.0,
                   0.0,
                   platform=PlatformClass.GROUP_2),
    ]

    slots = buildSlots(targets)
    print(
        f"{len(slots)} slots: "
        f"{[(s.target_id, '+'.join(sorted(c.name for c in s.required)), s.rank) for s in slots]}\n"
    )

    result = solve(agents, targets)

    print(f"converged in {result.rounds} rounds\n")
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
