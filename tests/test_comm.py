import pytest

from swarm.comms import Channel

LINE = {0: [1], 1: [0, 2], 2: [1]}          # 0 — 1 — 2
COMPLETE = {0: [1, 2], 1: [0, 2], 2: [0, 1]}


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
    ch = Channel(COMPLETE)
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
    ch = Channel(LINE)
    ch.beginStep(1)
    ch.transmitMessage(0, {"from": 0})
    ch.transmitMessage(2, {"from": 2})
    received = ch.deliverMessage(1)
    assert len(received) == 2
    assert {m.send_id for m in received} == {0, 2}
    ch.endStep()


def test_delivery_order_is_deterministic():
    ch = Channel(LINE)
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
    ch = Channel(LINE)
    ch.message_delay = 2
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
