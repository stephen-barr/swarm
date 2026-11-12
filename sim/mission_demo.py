"""
Generated demo: a mixed swarm runs a mission with runMission, then the replay is
shown or saved.

    python -m sim.mission_demo                          # opens a window
    python -m sim.mission_demo --save mission.mp4
    python -m sim.mission_demo --seed 47 --agents 16 --targets 8 --compare --save cmp.gif
        # --compare: scarcity off (k = 0) next to --k, same drones and targets, one clock
"""
import argparse
import random

import matplotlib.pyplot as plt

from sim.visualize import animateComparison, animateMission, saveAnimation
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
    p.add_argument("--compare",
                   action="store_true",
                   help="show k = 0 next to --k on the same instance")
    args = p.parse_args()

    def fly(k):
        agents, targets = build(args.seed, args.agents, args.targets, k)
        mission = MissionData.fromTargets(targets)
        channel = Channel[Payload](complete_graph(
            [a.config.agent_id for a in agents]))
        result = runMission(agents, channel, mission)
        print(
            f"k={k}: converged={result.converged}  agreement round={result.rounds}  "
            f"completion={result.completion_time:.1f} s  filled={result.filled}  "
            f"total={result.total:.3f}")
        return result, agents, mission

    if args.compare:
        runs = [fly(0.0), fly(args.k)]
        anim = animateComparison(
            [r for r, _, _ in runs], [a for _, a, _ in runs],
            runs[0][2],
            titles=["scarcity off (k = 0)", f"scarcity on (k = {args.k})"],
            every=10,
            suptitle=
            f"seed {args.seed}: {args.agents} drones, {args.targets} targets")
    else:
        result, agents, mission = fly(args.k)
        anim = animateMission(
            result,
            agents,
            mission,
            title=
            f"seed {args.seed}: {args.agents} drones, {args.targets} targets")
    if args.save:
        saveAnimation(anim, args.save)
        print(f"wrote {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
