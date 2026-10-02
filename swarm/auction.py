from swarm.agent import Agent, Target
from swarm.comms import Channel
from swarm.models import AuctionResult, Role, Slot, Winners


def run_round(t, agents: list[Agent], channel: Channel, slots_by_capability,
              targets_by_id) -> None:
    """All agents bid, then all send, then all receive.
    Each phase happens in the same time period for every agent, so message 
    reflects bids made this round. The assumption is synchronous rounds.
    """
    channel.beginStep(t)
    for a in agents:
        a.auctionPhase(slots_by_capability, targets_by_id)
    for a in agents:
        channel.transmitMessage(a.config.agent_id, a.outgoing())
    for a in agents:
        a.receieve(channel.deliverMessage(a.config.agent_id))
    channel.endStep()


def auctionTotal(winners: Winners) -> tuple[float, int]:
    return sum((score for score, _ in winners.values())), len(winners)


def run_auction(agents: list[Agent],
                channel: Channel,
                slots_by_capability: dict[Role, list[Slot]],
                targets_by_id: dict[int, Target],
                max_rounds=200) -> AuctionResult:
    """
    Run synchronous auction rounds until no agent's table changes.

    Each round, every agent bids against its own table, broadcasts it to its
    neighbours, then merges what it received and releases any slot it lost.

    Args:
        agents: Agents with empty winners tables. Mutated in place; use fresh
            agents for each run.
        channel: Delivers tables between neighbours. Its topology must be
            connected, or agents in different components never agree.
        slots_by_role: Slots grouped by required role, shared by all agents.
        targets_by_id: Lookup from target_id to Target, used for scoring.
        max_rounds: Safety cap. Convergence normally takes at most
            (number of slots) x (network diameter) rounds.

    Returns:
        AuctionResult with the agreed winners table, each agent's slot,
        the total score, the number of slots filled, and the rounds taken.

    Raises:
        RuntimeError: If tables are still changing after max_rounds. Usually
            a bug (e.g. inconsistent tie-breaking) or a disconnected graph.
    """

    for t in range(1, max_rounds + 1):
        before = [dict(a.winners) for a in agents]
        run_round(t, agents, channel, slots_by_capability, targets_by_id)
        if all(a.winners == b for a, b in zip(
                agents, before)):                        # no changes were made this round
            slots_by_id = {a.config.agent_id: a.slot for a in agents}
            score = auctionTotal(before[0])
            return AuctionResult(rounds=t,
                                 winners=before[0],
                                 assignment=slots_by_id,
                                 total=score[0],
                                 filled=score[1])
    raise RuntimeError("auction did not converge")
