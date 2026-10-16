from swarm.models import Bid, Capability, Message, Payload, Slot, role
from swarm.scenarios import make_agent

A = Capability.SURVEILLANCE
S = Slot(target_id=0, required=role(A), rank=0)
T = Slot(target_id=1, required=role(A), rank=0)


def msg(sender, winners, latest=None, receiver=0):
    """One delivered message from `sender` to agent `receiver`."""
    return Message(send_id=sender,
                   receiver_id=receiver,
                   payload=Payload(winners=winners, latest=latest),
                   send_step=1)


def fresh_agent():
    a = make_agent(0, [A], 0, 0)
    a.outgoing()        # clear the initial share flags
    return a


# ---- mergeLatest ----


def test_merge_latest_raises_version_and_flags_share():
    a = fresh_agent()
    changed = a.mergeLatest([msg(1, {}, latest={3: 2})])
    assert changed
    assert a.latest[3] == 2
    assert a.latest_to_share


def test_merge_latest_never_lowers():
    a = fresh_agent()
    a.latest[3] = 5
    changed = a.mergeLatest([msg(1, {}, latest={3: 2})])
    assert not changed
    assert a.latest[3] == 5


def test_merge_latest_reads_every_message():
    a = fresh_agent()
    a.mergeLatest([msg(1, {}, latest={3: 2}), msg(2, {}, latest={4: 7})])
    assert a.latest[3] == 2 and a.latest[4] == 7


# ---- pruning stored entries when latest rises ----


def test_raise_removes_dead_entry():
    a = fresh_agent()
    a.winners[S] = Bid(0.8, 3, 1)
    a.latest[3] = 1
    a.mergeLatest([msg(1, {}, latest={3: 2})])
    assert S not in a.winners


def test_raise_keeps_other_authors_entries():
    a = fresh_agent()
    a.winners[S] = Bid(0.8, 3, 1)
    a.winners[T] = Bid(0.5, 4, 1)
    a.latest.update({3: 1, 4: 1})
    a.mergeLatest([msg(1, {}, latest={3: 2})])
    assert S not in a.winners
    assert a.winners[T] == Bid(0.5, 4, 1)


# ---- filtering incoming bids ----


def test_dead_incoming_bid_is_not_stored():
    a = fresh_agent()
    a.latest[3] = 2
    a.updateWinners([msg(1, {S: Bid(0.9, 3, 1)})])
    assert S not in a.winners


def test_live_incoming_bid_is_stored_and_raises_latest():
    a = fresh_agent()
    a.updateWinners([msg(1, {S: Bid(0.9, 3, 4)})])
    assert a.winners[S] == Bid(0.9, 3, 4)
    assert a.latest[3] == 4
    assert a.latest_to_share


# ---- the hole from the trace ----


def test_correction_that_lost_elsewhere_still_kills_stale_bid():
    """Agent 0 holds drone 3's v1 at 0.8. Drone 3 has since moved to v2,
    but its v2 bid lost to drone 5's 0.7 upstream, so only drone 5's bid
    and the news 'drone 3 is at v2' arrive. The stale 0.8 must go and
    drone 5's 0.7 must be accepted."""
    a = fresh_agent()
    a.winners[S] = Bid(0.8, 3, 1)
    a.latest[3] = 1
    a.receieve([msg(1, {S: Bid(0.7, 5, 1)}, latest={3: 2, 5: 1})])
    assert a.winners[S] == Bid(0.7, 5, 1)


def test_stale_bid_echoed_back_is_rejected():
    """After the step above, a neighbour still holding drone 3's v1 sends it
    back. It must not return."""
    a = fresh_agent()
    a.winners[S] = Bid(0.7, 5, 1)
    a.latest.update({3: 2, 5: 1})
    a.receieve([msg(1, {S: Bid(0.8, 3, 1)})])
    assert a.winners[S] == Bid(0.7, 5, 1)
