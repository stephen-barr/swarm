from swarm.agent import Agent, Target
from swarm.comms import Channel
from swarm.models import AuctionResult, Payload, Role, Slot, Winners


def runRound(t: int, agents: list[Agent], channel: Channel[Payload],
             slots_by_role, targets_by_id) -> None:
    """All agents bid, then all send, then all receive.
    Each phase happens in the same time period for every agent, so message 
    reflects bids made this round. The assumption is synchronous rounds.
    """
    channel.beginStep(t)
    for a in agents:
        a.auctionPhase(slots_by_role, targets_by_id)
    for a in agents:
        channel.transmitMessage(a.config.agent_id, a.outgoing())
    for a in agents:
        a.receieve(channel.deliverMessage(a.config.agent_id))
    channel.endStep()


def auctionTotal(
    winners: Winners
) -> tuple[float,
           int]:        # Returns (sum of winning scores, number of slots filled)
    return sum(bid.score for bid in winners.values()), len(winners)


def runAuction(agents: list[Agent],
               channel: Channel[Payload],
               slots_by_role: dict[Role, list[Slot]],
               targets_by_id: dict[int, Target],
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
        runRound(t, agents, channel, slots_by_role, targets_by_id)
        if all((a.winners, a.census, a.latest) == b for a, b in zip(
                agents, before)):                                    # no changes were made this round
            slots_by_id = {a.config.agent_id: a.slot for a in agents}
            score = auctionTotal(before[0][0])
            return AuctionResult(rounds=t,
                                 winners=before[0][0],
                                 assignment=slots_by_id,
                                 total=score[0],
                                 filled=score[1])
    raise RuntimeError("auction did not converge")
