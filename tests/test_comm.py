from collections import defaultdict

import pytest

from swarm.comms import Channel, RandomDelayChannel
from swarm.scenarios import complete_edges, edges_to_topology, line_edges, ring_edges

LINE = (edges_to_topology(line_edges(5)))
COMPLETE = (edges_to_topology(complete_edges(5)))
RING = (edges_to_topology(ring_edges(5)))

SEEDS = (10, 14, 21)
DELAY = (0, 4, 8, 16)


def test_broadcast_reaches_only_neighbours():
    ch = Channel(LINE)
    ch.beginStep(1)
    ch.transmitMessage(0, {"from": 0})
    assert len(ch.deliverMessage(1)) == 1
    assert ch.deliverMessage(2) == []
    ch.endStep()


def test_broadcast_reaches_all_neighbours():
    ch = Channel(LINE)
    ch.beginStep(1)
    ch.transmitMessage(1, {"from": 1})
    assert len(ch.deliverMessage(0)) == 1
    assert len(ch.deliverMessage(2)) == 1
    ch.endStep()


def test_sender_does_not_hear_itself():
    ch = Channel(edges_to_topology(complete_edges(3)))
    ch.beginStep(1)
    ch.transmitMessage(0, {"from": 0})
    assert ch.deliverMessage(0) == []
    ch.deliverMessage(1)
    ch.deliverMessage(2)
    ch.endStep()


def test_mailbox_clears_after_delivery():
    ch = Channel(LINE)
    ch.beginStep(1)
    ch.transmitMessage(0, {"from": 0})
    assert len(ch.deliverMessage(1)) == 1
    assert ch.deliverMessage(1) == []       # not redelivered
    ch.endStep()


def test_payload_is_copied():
    ch = Channel(LINE)
    table = {"slot": 1.0}
    ch.beginStep(1)
    ch.transmitMessage(0, table)
    table["slot"] = 99.0          # sender mutates after sending
    msg = ch.deliverMessage(1)[0]
    assert msg.payload == {"slot": 1.0}
    ch.endStep()


def test_multiple_senders_accumulate():
    ch = Channel(edges_to_topology(line_edges(3)))
    ch.beginStep(1)
    ch.transmitMessage(0, {"from": 0})
    ch.transmitMessage(2, {"from": 2})
    received = ch.deliverMessage(1)
    assert len(received) == 2
    assert {m.send_id for m in received} == {0, 2}
    ch.endStep()


def test_delivery_order_is_deterministic():
    ch = Channel(edges_to_topology(line_edges(3)))
    ch.beginStep(1)
    ch.transmitMessage(2, {"from": 2})      # sent second-lowest id first
    ch.transmitMessage(0, {"from": 0})
    received = ch.deliverMessage(1)
    assert [m.send_id for m in received] == [0, 2]
    ch.endStep()


def test_steps_must_advance_by_one():
    ch = Channel(LINE)
    ch.beginStep(1)
    ch.endStep()
    with pytest.raises(AssertionError):
        ch.beginStep(3)


def test_undelivered_messages_are_caught():
    ch = Channel(LINE)
    ch.beginStep(1)
    ch.transmitMessage(0, {"from": 0})
    with pytest.raises(AssertionError):
        ch.endStep()    # agent 1 never called deliver


def test_delay_withholds_then_releases():
    ch = Channel(LINE, 2)
    ch.beginStep(1)
    ch.transmitMessage(0, {"from": 0})
    assert ch.deliverMessage(1) == []
    ch.endStep()

    ch.beginStep(2)
    assert ch.deliverMessage(1) == []
    ch.endStep()

    ch.beginStep(3)
    assert len(ch.deliverMessage(1)) == 1   # send_step 1 + delay 2
    ch.endStep()


def test_information_crosses_the_line_in_two_hops():
    """0 and 2 are not neighbours; 1 relays."""
    ch = Channel(LINE)
    heard = {0: {}, 1: {}, 2: {}}

    ch.beginStep(1)
    ch.transmitMessage(0, {"origin": 0})
    for i in (0, 1, 2):
        for m in ch.deliverMessage(i):
            heard[i].update(m.payload)
    ch.endStep()

    assert heard[1] == {"origin": 0}
    assert heard[2] == {}         # not yet

    ch.beginStep(2)
    ch.transmitMessage(1, heard[1])         # 1 passes on what it learned
    for i in (0, 1, 2):
        for m in ch.deliverMessage(i):
            heard[i].update(m.payload)
    ch.endStep()

    assert heard[2] == {"origin": 0}        # arrived on the second hop


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("delay", DELAY)
def test_random_delay_stays_between_zero_and_max_delay(delay, seed):
    ch = RandomDelayChannel(RING, message_delay=delay, rng_seed=seed)
    log = []
    N = 30
    for t in range(1, N + ch.message_delay +
                   1):                                # length of full communication plus delay
        ch.beginStep(t)
        if t <= N:
            for i in ch.mailbox:
                ch.transmitMessage(i, {"from": i, "sent": t})
        for i in ch.mailbox:
            for m in ch.deliverMessage(i):
                log.append((m.send_id, m.receiver_id, m.send_step, t))
        ch.endStep()
    assert all(0 <= l[3] - l[2] <= delay for l in log)
    messages_per_round = sum(len(v) for v in RING.values())
    assert len(log) == N * messages_per_round         # assert no messages were dropped

    by_sender_receiver = defaultdict(list)
    for sender, receiver, sent, delivered in log:
        by_sender_receiver[(sender, receiver)].append(
            (sent, delivered))                         # (sender, reciever) : (sent, delivered)

    messages_out_of_order = False
    for delay_info in by_sender_receiver.values():
        delay_info.sort(key=lambda e: (e[1], e[0]))
        sends = [sent for sent, _ in delay_info]
        if any(a > b for a, b in zip(
                sends, sends[1:])):         # assert there are messages out of order
            messages_out_of_order = True
    if delay > 0:
        assert messages_out_of_order
