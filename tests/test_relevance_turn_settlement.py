"""Deterministic lifecycle tests for the relevance turn-settlement DAG."""

from __future__ import annotations

import asyncio

import pytest

from kazusa_ai_chatbot.brain_service.turn_settlement import (
    PersistedChatFragment,
    TurnSettlementCoordinator,
)


class _FakeClock:
    """Controllable monotonic clock for deadline and heap tests."""

    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


def _fragment(
    sequence: int,
    *,
    author: str = "author-a",
    channel: str = "channel-1",
    body: str = "request",
    enqueue_monotonic: float = 0.0,
    targets: tuple[str, ...] = ("character",),
    reply_target: str = "character",
) -> PersistedChatFragment:
    """Build a persisted fragment with stable test metadata."""

    return PersistedChatFragment(
        arrival_sequence=sequence,
        scope=("discord", channel, "group"),
        author_platform_user_id=author,
        author_global_user_id=f"global-{author}",
        platform_message_id=f"message-{sequence}",
        conversation_row_id=f"row-{sequence}",
        storage_timestamp_utc=f"2026-07-16T00:00:{sequence:02d}+00:00",
        enqueue_monotonic=enqueue_monotonic,
        body_text=body,
        semantic_target_labels=targets,
        reply_target_label=reply_target,
        media_descriptions=(),
    )


def _coordinator(clock: _FakeClock, *, settled=None):
    """Build a coordinator with deterministic semantic stage doubles."""

    async def _frontline(_state):
        return {
            "intake_action": "start",
            "append_target": "none",
            "prelude_targets": [],
            "reason": "new candidate",
        }

    async def _settled(lease, state):
        del lease, state
        return settled or {
            "response_action": "proceed",
            "reason_to_respond": "grounded request",
            "use_reply_feature": True,
            "channel_topic": "request",
            "indirect_speech_context": "",
        }

    return TurnSettlementCoordinator(
        frontline_evaluator=_frontline,
        settled_evaluator=_settled,
        clock=clock,
    )


@pytest.mark.asyncio
async def test_group_turn_uses_six_second_quiet_window() -> None:
    """An admitted group turn becomes assessable after six seconds."""

    clock = _FakeClock()
    coordinator = _coordinator(clock)

    outcome = await coordinator.apply_frontline_decision(
        _fragment(1),
        {
            "intake_action": "start",
            "append_target": "none",
            "prelude_targets": [],
            "reason": "new candidate",
        },
    )

    assert outcome.action == "start"
    assert outcome.version == 1

    clock.advance(5.99)
    assert coordinator._ready_heap[0][0] > clock()

    clock.advance(0.01)
    lease = await coordinator.wait_for_assessment_ready()

    assert lease.turn_id == outcome.turn_id
    assert lease.version == 1
    assert lease.observation_status == "more_time_available"


@pytest.mark.asyncio
async def test_append_increments_version_and_clamps_to_hard_deadline() -> None:
    """A continuation extends quiet time but never moves past ten seconds."""

    clock = _FakeClock()
    coordinator = _coordinator(clock)
    start = await coordinator.apply_frontline_decision(
        _fragment(1),
        {
            "intake_action": "start",
            "append_target": "none",
            "prelude_targets": [],
            "reason": "new candidate",
        },
    )

    clock.advance(5.0)
    append = await coordinator.apply_frontline_decision(
        _fragment(2, body="clarification", enqueue_monotonic=5.0),
        {
            "intake_action": "append",
            "append_target": "open_1",
            "prelude_targets": [],
            "reason": "same topic",
        },
    )

    assert append.turn_id == start.turn_id
    assert append.version == 2

    clock.advance(4.99)
    assert any(
        token[0] > clock() and token[3] == append.version
        for token in coordinator._ready_heap
    )
    clock.advance(0.01)
    lease = await coordinator.wait_for_assessment_ready()

    assert lease.turn_id == start.turn_id
    assert lease.version == 2
    assert lease.observation_status == "observation_complete"


@pytest.mark.asyncio
async def test_three_open_turn_bound_freezes_oldest_before_fourth_start() -> None:
    """The fourth topic does not create an unbounded same-author comparison."""

    clock = _FakeClock()
    coordinator = _coordinator(clock)
    start_decision = {
        "intake_action": "start",
        "append_target": "none",
        "prelude_targets": [],
        "reason": "new topic",
    }

    outcomes = []
    for sequence in range(1, 5):
        outcomes.append(
            await coordinator.apply_frontline_decision(
                _fragment(sequence, body=f"topic-{sequence}"),
                start_decision,
            )
        )

    assert len(coordinator._pending_turns) == 4
    active_turns = [
        turn
        for turn in coordinator._pending_turns.values()
        if turn.status == "SETTLING"
    ]
    assert len(active_turns) == 3
    assert coordinator._pending_turns[outcomes[0].turn_id].status == (
        "ASSESSMENT_READY"
    )


@pytest.mark.asyncio
async def test_wait_uses_one_extension_and_reaches_complete_phase() -> None:
    """A semantic wait moves one turn to the hard deadline exactly once."""

    clock = _FakeClock()
    coordinator = _coordinator(
        clock,
        settled={
            "response_action": "wait",
            "reason_to_respond": "the request may continue",
            "use_reply_feature": False,
            "channel_topic": "",
            "indirect_speech_context": "",
        },
    )
    start = await coordinator.apply_frontline_decision(
        _fragment(1),
        {
            "intake_action": "start",
            "append_target": "none",
            "prelude_targets": [],
            "reason": "new candidate",
        },
    )
    clock.advance(6.0)
    first_lease = await coordinator.wait_for_assessment_ready()
    first_decision = await coordinator.evaluate_settled(first_lease, {})
    first_outcome = await coordinator.apply_settled_decision(
        first_lease,
        first_decision,
    )

    assert first_outcome.response_action == "wait"
    assert first_outcome.wait_used is True

    clock.advance(4.0)
    final_lease = await coordinator.wait_for_assessment_ready()
    assert final_lease.turn_id == start.turn_id
    assert final_lease.observation_status == "observation_complete"


@pytest.mark.asyncio
async def test_failed_assessment_closes_current_turn_without_semantic_ignore() -> None:
    """An operational settlement failure must release its pending turn."""

    clock = _FakeClock()
    coordinator = _coordinator(clock)
    start = await coordinator.apply_frontline_decision(
        _fragment(1),
        {
            "intake_action": "start",
            "append_target": "none",
            "prelude_targets": [],
            "reason": "new candidate",
        },
    )
    clock.advance(6.0)
    lease = await coordinator.wait_for_assessment_ready()

    closed = await coordinator.complete_failed_assessment(lease)

    assert closed is True
    assert start.turn_id not in coordinator._pending_turns
    next_state = await coordinator.build_frontline_state(
        _fragment(2, enqueue_monotonic=clock()),
    )
    assert next_state["open_turns"] == []


@pytest.mark.asyncio
async def test_failed_stale_assessment_preserves_newer_turn_version() -> None:
    """A stale operational failure cannot close a newer assembled turn."""

    clock = _FakeClock()
    coordinator = _coordinator(clock)
    start = await coordinator.apply_frontline_decision(
        _fragment(1),
        {
            "intake_action": "start",
            "append_target": "none",
            "prelude_targets": [],
            "reason": "new candidate",
        },
    )
    clock.advance(6.0)
    stale_lease = await coordinator.wait_for_assessment_ready()
    append = await coordinator.apply_frontline_decision(
        _fragment(2, body="newer intent", enqueue_monotonic=clock()),
        {
            "intake_action": "append",
            "append_target": "open_1",
            "prelude_targets": [],
            "reason": "same candidate",
        },
    )

    closed = await coordinator.complete_failed_assessment(stale_lease)

    assert closed is False
    assert coordinator._pending_turns[start.turn_id].version == append.version
    clock.advance(4.0)
    current_lease = await coordinator.wait_for_assessment_ready()
    assert current_lease.version == append.version
    assert current_lease.observation_status == "observation_complete"


@pytest.mark.asyncio
async def test_stale_assessment_cannot_claim_after_append() -> None:
    """A version change during assessment rejects the stale cognition entry."""

    clock = _FakeClock()
    coordinator = _coordinator(clock)
    start = await coordinator.apply_frontline_decision(
        _fragment(1),
        {
            "intake_action": "start",
            "append_target": "none",
            "prelude_targets": [],
            "reason": "new candidate",
        },
    )
    clock.advance(6.0)
    lease = await coordinator.wait_for_assessment_ready()
    decision = await coordinator.evaluate_settled(lease, {})
    append = await coordinator.apply_frontline_decision(
        _fragment(2, body="newer intent"),
        {
            "intake_action": "append",
            "append_target": "open_1",
            "prelude_targets": [],
            "reason": "same candidate",
        },
    )

    assert append.version == 2
    stale_outcome = await coordinator.apply_settled_decision(
        lease,
        decision,
    )
    assert stale_outcome.stale is True
    assert await coordinator.claim_for_cognition(
        start.turn_id,
        lease.version,
    ) is False
    clock.advance(4.0)
    final_lease = await coordinator.wait_for_assessment_ready()
    final_decision = await coordinator.evaluate_settled(final_lease, {})
    await coordinator.apply_settled_decision(final_lease, final_decision)
    assert await coordinator.claim_for_cognition(
        start.turn_id,
        append.version,
    ) is True


@pytest.mark.asyncio
async def test_ready_heap_orders_eligibility_then_leader_sequence() -> None:
    """Ready turns are globally ordered before the cognition lane claims one."""

    clock = _FakeClock()
    coordinator = _coordinator(clock)
    decision = {
        "intake_action": "start",
        "append_target": "none",
        "prelude_targets": [],
        "reason": "new candidate",
    }
    first = await coordinator.apply_frontline_decision(_fragment(1), decision)
    second = await coordinator.apply_frontline_decision(
        _fragment(2, author="author-b"),
        decision,
    )
    clock.advance(6.0)

    first_lease = await coordinator.wait_for_assessment_ready()
    assert first_lease.turn_id == first.turn_id
    first_decision = await coordinator.evaluate_settled(first_lease, {})
    first_outcome = await coordinator.apply_settled_decision(
        first_lease,
        first_decision,
    )
    assert first_outcome.response_action == "proceed"
    assert await coordinator.claim_for_cognition(
        first.turn_id,
        first_lease.version,
    ) is True

    second_lease = await coordinator.wait_for_assessment_ready()
    assert second_lease.turn_id == second.turn_id


@pytest.mark.asyncio
async def test_relevance_work_is_fifo_and_one_in_flight() -> None:
    """Frontline work shares one serialized relevance executor."""

    clock = _FakeClock()
    started = asyncio.Event()
    release = asyncio.Event()
    order: list[str] = []
    active = 0
    maximum_active = 0

    async def _frontline(state):
        nonlocal active, maximum_active
        active += 1
        maximum_active = max(maximum_active, active)
        order.append(state["label"])
        if state["label"] == "first":
            started.set()
            await release.wait()
        active -= 1
        return {
            "intake_action": "discard",
            "append_target": "none",
            "prelude_targets": [],
            "reason": "irrelevant",
        }

    async def _settled(_lease, _state):
        raise AssertionError("settled evaluator is not used in this test")

    coordinator = TurnSettlementCoordinator(
        frontline_evaluator=_frontline,
        settled_evaluator=_settled,
        clock=clock,
    )
    first_task = asyncio.create_task(
        coordinator.evaluate_frontline({"label": "first"})
    )
    await started.wait()
    second_task = asyncio.create_task(
        coordinator.evaluate_frontline({"label": "second"})
    )
    await asyncio.sleep(0)

    assert order == ["first"]
    assert maximum_active == 1

    release.set()
    await asyncio.gather(first_task, second_task)

    assert order == ["first", "second"]
    assert maximum_active == 1


@pytest.mark.asyncio
async def test_interleaved_authors_receive_only_their_own_open_turns() -> None:
    """A1, B1, A2 exposes A to A2 without exposing B as an append slot."""

    clock = _FakeClock()
    coordinator = _coordinator(clock)
    start = {
        "intake_action": "start",
        "append_target": "none",
        "prelude_targets": [],
        "reason": "new candidate",
    }
    await coordinator.apply_frontline_decision(_fragment(1), start)
    await coordinator.apply_frontline_decision(
        _fragment(2, author="author-b", body="B1"),
        start,
    )

    state = await coordinator.build_frontline_state(
        _fragment(3, body="A2", enqueue_monotonic=2.0),
    )

    assert len(state["open_turns"]) == 1
    assert state["open_turns"][0]["opening_excerpt"] == "request"


@pytest.mark.asyncio
async def test_discarded_prelude_is_promoted_by_supplied_slot() -> None:
    """A content-first message can join a later direct tag in chronology."""

    clock = _FakeClock()
    coordinator = _coordinator(clock)
    prelude = _fragment(
        1,
        body="The object is making this sound.",
        enqueue_monotonic=0.0,
        targets=("none",),
        reply_target="none",
    )
    await coordinator.apply_frontline_decision(
        prelude,
        {
            "intake_action": "discard",
            "append_target": "none",
            "prelude_targets": [],
            "reason": "not addressed yet",
        },
    )
    current = _fragment(
        2,
        body="Character, what do you think?",
        enqueue_monotonic=4.0,
    )
    state = await coordinator.build_frontline_state(current)

    assert state["recent_preludes"][0]["summary"].startswith("The object")

    outcome = await coordinator.apply_frontline_decision(
        current,
        {
            "intake_action": "start",
            "append_target": "none",
            "prelude_targets": ["prelude_1"],
            "reason": "the tag promotes the prior description",
        },
    )
    clock.advance(10.0)
    lease = await coordinator.wait_for_assessment_ready()

    assert outcome.turn_id == lease.turn_id
    assert [fragment.arrival_sequence for fragment in lease.fragments] == [1, 2]
    assert lease.leader_sequence == 1
    assert lease.response_owner_sequence == 2


@pytest.mark.asyncio
async def test_explicit_third_party_discard_is_not_a_prelude_candidate() -> None:
    """Typed traffic for another participant cannot join a later bot turn."""

    clock = _FakeClock()
    coordinator = _coordinator(clock)
    await coordinator.apply_frontline_decision(
        _fragment(
            1,
            body="A question only for another participant.",
            targets=("other_participant",),
            reply_target="other_participant",
        ),
        {
            "intake_action": "discard",
            "append_target": "none",
            "prelude_targets": [],
            "reason": "third-party traffic",
        },
    )
    clock.advance(3.0)

    state = await coordinator.build_frontline_state(_fragment(
        2,
        body="Character?",
        enqueue_monotonic=3.0,
    ))

    assert state["recent_preludes"] == []


@pytest.mark.asyncio
async def test_ingress_watermark_delays_claim_until_frontline_applies() -> None:
    """A pre-deadline queued follow-up blocks stale cognition entry."""

    clock = _FakeClock()
    coordinator = _coordinator(clock)
    start = await coordinator.apply_frontline_decision(
        _fragment(1),
        {
            "intake_action": "start",
            "append_target": "none",
            "prelude_targets": [],
            "reason": "new candidate",
        },
    )
    clock.advance(6.0)
    lease = await coordinator.wait_for_assessment_ready()
    decision = await coordinator.evaluate_settled(lease, {})
    await coordinator.apply_settled_decision(lease, decision)
    coordinator.register_ingress(
        sequence=2,
        scope=("discord", "channel-1", "group"),
        author_platform_user_id="author-a",
        enqueue_monotonic=9.5,
    )

    assert await coordinator.claim_for_cognition(
        start.turn_id,
        lease.version,
    ) is False

    append = await coordinator.apply_frontline_decision(
        _fragment(2, body="boundary follow-up", enqueue_monotonic=9.5),
        {
            "intake_action": "append",
            "append_target": "open_1",
            "prelude_targets": [],
            "reason": "same candidate",
        },
    )
    clock.advance(4.0)
    final_lease = await coordinator.wait_for_assessment_ready()

    assert append.version == 2
    assert final_lease.version == 2
    assert final_lease.observation_status == "observation_complete"


@pytest.mark.asyncio
async def test_private_turn_is_immediately_ready_and_collapsed_append_is_exact(
) -> None:
    """Private coalescing keeps immediate timing and exact survivor identity."""

    clock = _FakeClock()
    coordinator = _coordinator(clock)
    private = _fragment(1)
    private.scope = ("discord", "dm-1", "private")
    start = await coordinator.apply_frontline_decision(
        private,
        {
            "intake_action": "start",
            "append_target": "none",
            "prelude_targets": [],
            "reason": "private candidate",
        },
    )
    follower = _fragment(2, body="private follow-up")
    follower.scope = private.scope
    append = await coordinator.append_collapsed_private_fragment(
        follower,
        turn_id=start.turn_id,
    )
    lease = await coordinator.wait_for_assessment_ready()

    assert append.turn_id == start.turn_id
    assert lease.observation_status == "observation_complete"
    assert [fragment.arrival_sequence for fragment in lease.fragments] == [1, 2]


@pytest.mark.asyncio
async def test_completed_cognition_releases_pending_turn_state() -> None:
    """A claimed turn is removed after the single cognition lane finishes."""

    clock = _FakeClock()
    coordinator = _coordinator(clock)
    start = await coordinator.apply_frontline_decision(
        _fragment(1),
        {
            "intake_action": "start",
            "append_target": "none",
            "prelude_targets": [],
            "reason": "candidate",
        },
    )
    clock.advance(6.0)
    lease = await coordinator.wait_for_assessment_ready()
    decision = await coordinator.evaluate_settled(lease, {})
    await coordinator.apply_settled_decision(lease, decision)
    assert await coordinator.claim_for_cognition(
        start.turn_id,
        lease.version,
    ) is True

    await coordinator.complete_cognition(start.turn_id, lease.version)

    assert start.turn_id not in coordinator._pending_turns


@pytest.mark.asyncio
async def test_latest_bot_continuity_is_scoped_to_author_and_channel() -> None:
    """A completed dialog is visible only to its matching frontline scope."""

    clock = _FakeClock()
    coordinator = _coordinator(clock)
    await coordinator.record_bot_continuity(
        scope=("discord", "channel-1", "group"),
        author_platform_user_id="author-a",
        dialog_text="previous answer",
    )

    matching = await coordinator.build_frontline_state(_fragment(2))
    other_author = await coordinator.build_frontline_state(
        _fragment(3, author="author-b"),
    )
    other_channel = await coordinator.build_frontline_state(
        _fragment(4, channel="channel-2"),
    )

    assert matching["latest_bot_continuity"] == "previous answer"
    assert other_author["latest_bot_continuity"] == ""
    assert other_channel["latest_bot_continuity"] == ""

    await coordinator.apply_frontline_decision(
        _fragment(5),
        {
            "intake_action": "start",
            "append_target": "none",
            "prelude_targets": [],
            "reason": "new candidate",
        },
    )
    clock.advance(6.0)
    lease = await coordinator.wait_for_assessment_ready()

    assert lease.latest_bot_continuity == "previous answer"


@pytest.mark.asyncio
async def test_latest_bot_continuity_expires_outside_active_scene() -> None:
    """Old bot dialog cannot authorize a later unrelated group fragment."""

    clock = _FakeClock()
    coordinator = _coordinator(clock)
    await coordinator.record_bot_continuity(
        scope=("discord", "channel-1", "group"),
        author_platform_user_id="author-a",
        dialog_text="send the requested screenshot",
    )

    clock.advance(179.0)
    recent = await coordinator.build_frontline_state(_fragment(
        2,
        enqueue_monotonic=179.0,
    ))
    clock.advance(2.0)
    stale = await coordinator.build_frontline_state(_fragment(
        3,
        enqueue_monotonic=181.0,
    ))

    assert recent["latest_bot_continuity"] == "send the requested screenshot"
    assert stale["latest_bot_continuity"] == ""
    assert stale["conversation_scope"] == "group"
