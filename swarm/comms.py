import copy
import random
from typing import Generic

from swarm.models import Message, P

EdgeList = dict[int, list[int]]   # agent_id : adjacent agents to the id


class Channel(Generic[P]):
    """Delivers each agent's message to its neighbours in the topology.

    message_delay: rounds between sending and delivery (0 = same round).
    Messages are delivered sorted by (sender, send_step) so runs are
    deterministic. endStep asserts every deliverable message was communicated,
    catching an agent that skipped its receive phase.
    """

    def __init__(self, edge_list: EdgeList, message_delay: int = 0) -> None:
        self.edge_list = edge_list
        self.time_step: int = 0
        self.message_delay: int = message_delay
        self.mailbox: dict[int, list[Message[P]]] = {
            agent_id: []
            for agent_id in edge_list
        }
        self.transmissions: int = 0         # number of transmissions

    def beginStep(self, t) -> None:
        assert t == self.time_step + 1, f"expected step {self.time_step + 1}, got {t}"
        self.time_step = t

    def messageDelay(
            self
    ) -> int:           # return the message delay, subclass modifies only this
        return self.message_delay

    def transmitMessage(
        self,
        sending_agent_id: int,
        payload: P,
    ) -> None:

        self.transmissions += 1                       # increment count of transmissions
        for neighbor_id in self.edge_list[sending_agent_id]:
            snapshot = copy.deepcopy(payload)         # create a copy of the payload
            message = Message(
                send_id=sending_agent_id,
                receiver_id=neighbor_id,
                payload=snapshot,
                send_step=self.time_step,
                message_delay=self.messageDelay(),
            )
            self.mailbox[neighbor_id].append(message)

    def deliverMessage(self, recipient_id: int) -> list[Message[P]]:
        box: list[Message] = self.mailbox[
            recipient_id]                                      # get the messages for the recipient
        current_messages: list[Message] = [
            m for m in box if m.deliveryStep <= self.time_step
        ]                                                      # only give out the current messages
        current_messages.sort(
            key=lambda m:
            (m.send_id, m.send_step))                          # sort by sender and send time
        self.mailbox[recipient_id] = [
            m for m in box if m.deliveryStep > self.time_step
        ]                                                      # only keep future messages
        return current_messages

    def endStep(self) -> None:
        leftover_messages: dict[int, int] = {
            i: len(box)
            for i, box in self.mailbox.items()
            if any(m.deliveryStep <= self.time_step for m in box)
        }
        assert not leftover_messages, f"undelivered at time {self.time_step} : {leftover_messages}"


class RandomDelayChannel(Channel[P]):
    """Channel with uniformly random delay between 0 and message_delay per message."""

    def __init__(self, edge_list: EdgeList, message_delay: int,
                 rng_seed: int) -> None:
        super().__init__(edge_list, message_delay)
        self.rng = random.Random(
            rng_seed)                       # make a random seed for reproducibility

    def messageDelay(self) -> int:
        return self.rng.randint(0, self.message_delay)
