from swarm.agent import Agent, Target
from swarm.comms import Channel
from swarm.models import Role, Slot


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
