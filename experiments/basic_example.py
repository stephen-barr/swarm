"""
Vibecoded basic example to test if methodology works.

Scarcity weighting on one small instance.

Four drones, two targets. Every drone can do SURVEILLANCE; only drone 0 can
also do MAPPING. Target 0 needs one SURVEILLANCE drone and is close to
drone 0 and worth more; target 1 needs one MAPPING drone.

Note that k=2 is relatively high compared to the recommendation of k=0.1.
Note that k=0.1 does not work for this small example as the system is bottlenecked
by the discrepency between scores, and the distance between targets.

Three runs:

1. k = 0 (no weight): drone 0 takes the nearby, more valuable SURVEILLANCE
   slot, and the MAPPING slot, which nobody else can fill, stays empty.
2. k = 2, census learned during the auction: drones re-score as the census
   arrives; once drone 0 learns that MAPPING is scarce, it switches to it and
   a specialist covers SURVEILLANCE. Same result as run 3, two rounds later.
3. k = 2, roster known before launch: MAPPING has 1 capable drone per slot
   against a swarm average of 2, SURVEILLANCE has 4, so MAPPING is weighted up,
   drone 0 takes it, and a specialist covers SURVEILLANCE: both slots filled.

Run: python -m experiments.basic_example
"""

from __future__ import annotations

from swarm.agent import Agent
from swarm.baselines import solveHungarian
from swarm.models import Capability, Message, Payload, Target, role
from swarm.scenarios import at, make_agent, make_target, solve

S = Capability.SURVEILLANCE
M = Capability.MAPPING


def build(k: float) -> tuple[list[Agent], list[Target]]:
    """A fresh copy of the instance (agents are mutated by an auction)."""
    targets = [
        make_target(0, at(100.0, 0.0), 0.8,
                    {role(S): {
                         1: 1.0
                     }}),                             # common role, valuable, close to drone 0
        make_target(1, at(0.0, 300.0), 0.5,
                    {role(M): {
                         1: 1.0
                     }}),                             # scarce role: only drone 0 can fill it
    ]
    agents = [
        make_agent(0, {S, M}, 50.0, 0.0),             # the versatile drone
        make_agent(1, {S}, 400.0, 0.0),               # specialists, farther from target 0
        make_agent(2, {S}, 450.0, 50.0),
        make_agent(3, {S}, 450.0, -50.0),
    ]
    for a in agents:
        a.scarcity_k = k                              # switch to make_agent(..., scarcity_k=k) once it takes k
    return agents, targets


def preload_roster(agents: list[Agent]) -> None:
    """Give every drone the full census before the auction starts, as if the
    roster were loaded before launch."""
    roster = {a.config.agent_id: a.config for a in agents}
    for a in agents:
        a.updateCensus([
            Message(send_id=-1,
                    receiver_id=a.config.agent_id,
                    payload=Payload(winners={}, census=roster),
                    send_step=0)
        ])


def describe(slot) -> str:
    return f"target {slot.target_id} {'+'.join(sorted(c.name for c in slot.required))}"


def run(k: float, roster_known: bool) -> None:
    agents, targets = build(k)
    if roster_known:
        preload_roster(agents)
    targets_by_id = {t.target_id: t for t in targets}
    result = solve(agents, targets)

    label = "roster known before launch" if roster_known else "census learned during auction"
    print(f"k = {k}, {label}:  filled {result.filled} of 2 slots, "
          f"raw score {result.total:.3f}, {result.rounds} rounds")
    for a in agents:
        if a.slot is None:
            print(f"  drone {a.config.agent_id}: unassigned")
        else:
            raw = a.makeScore(targets_by_id[a.slot.target_id])
            print(
                f"  drone {a.config.agent_id}: {describe(a.slot):<24} raw score {raw:.3f}"
            )
    print()


def main() -> None:
    agents, targets = build(0.0)
    best_score, best_filled = solveHungarian(agents=agents, targets=targets)
    print(
        f"Optimal (Hungarian, raw score): filled {best_filled} of 2, raw score {best_score:.3f}\n"
    )
    run(0, roster_known=False)
    run(2, roster_known=False)
    run(2, roster_known=True)


if __name__ == "__main__":
    main()
