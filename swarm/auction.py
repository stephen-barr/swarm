import numpy as np

from swarm.agent import Agent
from swarm.comms import Channel
from swarm.models import (
    AuctionResult,
    MissionData,
    Payload,
    ThreeVector,
    Winners,
    raw_score,
)


def runRound(t: int, agents: list[Agent], channel: Channel[Payload],
             mission_data: MissionData) -> None:
    """
    All agents bid, then all send, then all receive.
    Each phase happens in the same time period for every agent, so message 
    reflects bids made this round. The assumption is synchronous rounds.
    """
    channel.beginStep(t)
    for a in agents:
        a.auctionPhase(mission_data)
    for a in agents:
        channel.transmitMessage(a.config.agent_id, a.outgoing())
    for a in agents:
        a.receieve(channel.deliverMessage(a.config.agent_id))
    channel.endStep()


def weightedTotal(
    winners: Winners
) -> tuple[float,
           int]:        # Returns (sum of winning scores, number of slots filled)
    return sum(bid.score for bid in winners.values()), len(winners)


def rawTotal(agents: list[Agent], mission_data: MissionData,
             positions: dict[int, ThreeVector]) -> tuple[float, int]:
    """Unweighted score of the current assignment, scored from the given positions."""
    score = 0.0
    filled = 0
    for a in agents:
        if a.slot is not None:
            filled += 1
            score += raw_score(a.config.platform.value,
                               positions[a.config.agent_id],
                               mission_data.targets_by_id[a.slot.target_id])
    return (score, filled)


def runAuction(agents: list[Agent],
               channel: Channel[Payload],
               mission_data: MissionData,
               max_rounds=500) -> AuctionResult:
    """
    Static auction. No agents move.
    Run synchronous rounds until no agent's winners, census or
    versions change (checked by an outside observer, not the agents).

    Args:
        agents: Fresh agents; mutated in place.
        channel: Delivers packets between neighbours; must be connected.
        slots_by_role: Slots grouped by required role.
        targets_by_id: Lookup from target_id to Target.
        max_rounds: Safety cap.

    Returns: AuctionResult (winners, assignment, total, filled, rounds).
    Raises: RuntimeError if still changing after max_rounds.
    """

    for t in range(1, max_rounds + 1):
        before = [(dict(a.winners), dict(a.census), dict(a.latest))
                  for a in agents]
        runRound(t, agents, channel, mission_data)
        if all((a.winners, a.census, a.latest) == b for a, b in zip(
                agents, before)):                                             # no changes were made this round
            slots_by_id = {a.config.agent_id: a.slot for a in agents}
            score = rawTotal(agents=agents,
                             mission_data=mission_data,
                             positions={
                                 a.config.agent_id: a.state.estimate.position
                                 for a in agents
                             })
            return AuctionResult(dynamics=False,
                                 converged=True,
                                 rounds=t,
                                 winners=before[0][0],
                                 assignment=slots_by_id,
                                 total=score[0],
                                 filled=score[1])
    raise RuntimeError("auction did not converge")


def runMission(agents: list[Agent],
               channel: Channel[Payload],
               mission_data: MissionData,
               dt=0.02,
               radio_period=0.1,
               max_time=600.0) -> AuctionResult:
    """
    Same as runAuction with added logic for dynamics. Auction ends when all agent's messages do not change
    AND every agent has reached target within tolerance (checked by an outside observer, not the agents).

    Args:
        agents: Fresh agents; mutated in place.
        channel: Delivers packets between neighbours; must be connected.
        mission_data: information about targets/slots, call 
        [MissionData].fromTargets(list[Target]) to create dataclass.
        dt: control loop period for dynamics
        radio_period: control loop period for RF transmissions (rounds)
        max_time: maximum amount of time you let the auction go.
    Returns: AuctionResult (converged, winners, assignment, total, filled, rounds).
    In any case, convergence or max_time reached discernable from converged bool.
    """
    ticks_per_round = round(radio_period /
                            dt)                                          # how many dt for one radio transmission
    rnd = 0
    round_convergence = 0                                                # increments until the auction algorithm stops
    start_positions = {
        a.config.agent_id: a.state.estimate.position
        for a in agents
    }                                                                    # start positions needed for accurate scoring with dynamics
    for tick in range(round(max_time / dt)):
        if tick % ticks_per_round == 0:                                  # if it is time for radio transmission
            rnd += 1                                                     # how many rounds
            before = [(dict(a.winners), dict(a.census), dict(a.latest))
                      for a in agents]
            runRound(rnd, agents, channel, mission_data)
            no_changes_to_payloads: bool = all(
                (a.winners, a.census, a.latest) == b
                for a, b in zip(agents, before)
            )                                                            # were changes made to census, winners, or latest
            agents_at_location = True
            for a in agents:
                if a.slot is not None:
                    agent_position = np.array(a.state.estimate.position[:2])
                    current_slot_position = np.array(
                        mission_data.slotPosition(a.slot)[:2])
                    if a.config.platform.value.can_hover:
                        if np.linalg.norm(
                                agent_position - current_slot_position
                        ) > 1:                                           # check if a multirotor is within a meter of target
                            agents_at_location = False
                    else:
                        if np.linalg.norm(
                                agent_position - current_slot_position
                        ) > 1.2 * a.config.platform.value.loiter_radius: # check if a fixed wing is within 1.2 (tolerance) of loiter radius to target
                            agents_at_location = False
            if not no_changes_to_payloads:
                round_convergence = rnd                                  # take down how many rounds it took for auction to converge
            if agents_at_location and no_changes_to_payloads:            # no changes made, return AuctionResult
                slots_by_id = {a.config.agent_id: a.slot for a in agents}
                score = rawTotal(agents=agents,
                                 mission_data=mission_data,
                                 positions=start_positions)
                return AuctionResult(dynamics=True,
                                     converged=True,
                                     rounds=round_convergence,
                                     winners=before[0][0],
                                     assignment=slots_by_id,
                                     total=score[0],
                                     filled=score[1],
                                     completion_time=tick * dt)

        ### advance all the agents
        for a in agents:
            a.step(dt, mission_data=mission_data)

    slots_by_id = {a.config.agent_id: a.slot for a in agents}
    score = rawTotal(agents, mission_data, positions=start_positions)
    return AuctionResult(dynamics=True,
                         converged=False,
                         rounds=round_convergence,
                         winners=dict(agents[0].winners),
                         assignment=slots_by_id,
                         total=score[0],
                         filled=score[1],
                         completion_time=max_time)
