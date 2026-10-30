"""
Generated demo: a mixed swarm runs a mission with runMission, then the replay is
shown or saved.

    python -m experiments.mission_demo                  # opens a window
    python -m experiments.mission_demo --save mission.mp4
"""
import argparse
import random

import matplotlib.pyplot as plt

from sim.visualize import animateMission, saveAnimation
from swarm.auction import runMission
from swarm.comms import Channel
from swarm.models import Capability, MissionData, Payload, PlatformClass, role
from swarm.scenarios import at, complete_graph, make_agent, make_target

PLATFORMS = [
    PlatformClass.GROUP_1_MULTIROTOR, PlatformClass.GROUP_1_FIXED_WING
]


def build(seed: int, n_agents: int, n_targets: int, k: float):
    rng = random.Random(seed)
    caps = list(Capability)[:4]
    agents = [
        make_agent(i,
                   rng.sample(caps, rng.randint(1, 2)),
                   rng.uniform(-400, 400),
                   rng.uniform(-400, 400),
                   scarcity_k=k,
                   platform=PLATFORMS[i % len(PLATFORMS)])
        for i in range(n_agents)
    ]

    def fillable_role():
        # a role that at least one drone in the swarm can fill: a subset of some drone's caps
        owner = sorted(rng.choice(agents).config.capabilities,
                       key=lambda c: c.value)
        return role(*rng.sample(owner, rng.randint(1, len(owner))))

    targets = [
        make_target(t, at(rng.uniform(-400, 400), rng.uniform(-400, 400)),
                    rng.uniform(0.3, 1.0),
                    {fillable_role(): {
                         rng.randint(1, 2): 1.0
                     }}) for t in range(n_targets)
    ]
    return agents, targets


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--agents", type=int, default=8)
    p.add_argument("--targets", type=int, default=4)
    p.add_argument("--k", type=float, default=0.1)
    p.add_argument("--save",
                   default="",
                   help="output path (.mp4 or .gif); omit to show")
    args = p.parse_args()

    agents, targets = build(args.seed, args.agents, args.targets, args.k)
    mission = MissionData.fromTargets(targets)
    channel = Channel[Payload](complete_graph(
        [a.config.agent_id for a in agents]))
    result = runMission(agents, channel, mission)
    print(f"converged={result.converged}  agreement round={result.rounds}  "
          f"completion={result.completion_time:.1f} s  filled={result.filled}")

    anim = animateMission(
        result,
        agents,
        mission,
        title=f"seed {args.seed}: {args.agents} drones, {args.targets} targets"
    )
    if args.save:
        saveAnimation(anim, args.save)
        print(f"wrote {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
