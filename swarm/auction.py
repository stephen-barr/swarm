from swarm.agent import Agent
from swarm.comms import Channel
from swarm.models import (
    AuctionResult,
    MissionData,
    Payload,
    Target,
    Winners,
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


def rawTotal(agents: list[Agent],
             targets_by_id: dict[int, Target]) -> tuple[float, int]:
    """ 
    Unweighted score (does not utilize scarcity for score)
    to compare to baselines.
    """
    score = 0.0
    filled = 0
    for a in agents:
        if a.slot is not None:
            score += a.makeScore(targets_by_id[a.slot.target_id])
            filled += 1
    return score, filled


def runAuction(agents: list[Agent],
               channel: Channel[Payload],
               mission_data: MissionData,
               max_rounds=500) -> AuctionResult:
    """
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
                agents, before)):                                    # no changes were made this round
            slots_by_id = {a.config.agent_id: a.slot for a in agents}
            score = rawTotal(agents=agents,
                             targets_by_id=mission_data.targets_by_id)
            return AuctionResult(rounds=t,
                                 winners=before[0][0],
                                 assignment=slots_by_id,
                                 total=score[0],
                                 filled=score[1])
    raise RuntimeError("auction did not converge")
