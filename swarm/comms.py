from typing import Any

from swarm.models import Message, MessagesById

# Message is (send_id, receiver_id,
# payload, send_step, deliver step)

# Slot is (target_id, capability, index)
# Winners is Slot : [score, agent]

EdgeList = dict[int, list[int]]   # agent_id : adjacent agents to the id


class Channel:
    """Delivers each agent's message to its neighbours in the topology.

    message_delay: rounds between sending and delivery (0 = same round).
    Messages are delivered sorted by (sender, send_step) so runs are
    deterministic. end_step asserts every deliverable message was communicated,
    catching an agent that skipped its receive phase.
    """

    def __init__(self, edge_list: EdgeList) -> None:
        self.edge_list = edge_list
        self.time_step: int = 0
        self.message_delay: int = 0         # set the message delay to zero as default
        self.mailbox: MessagesById = {agent_id: [] for agent_id in edge_list}

    def beginStep(self, t) -> None:
        assert t == self.time_step + 1, f"expected step {self.time_step + 1}, got {t}"
        self.time_step = t

    def transmitMessage(self, sending_agent_id: int, payload: Any) -> None:
        for neighbor_id in self.edge_list[sending_agent_id]:
            message: Message = Message(
                send_id=sending_agent_id,
                receiver_id=neighbor_id,
                payload=dict(payload),
                send_step=self.time_step,
                message_delay=self.message_delay,
            )
            self.mailbox[neighbor_id].append(message)

    def deliverMessage(self, recipient_id: int) -> list[Message]:
        box: list[Message] = self.mailbox[
            recipient_id]                                      # get the messages for the recipent
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
