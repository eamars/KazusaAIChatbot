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
        semantic_target_labels=("character",),
        reply_target_label="character",
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
        _fragment(2, body="clarification"),
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
    assert await coordinator.claim_for_cognition(start.turn_id, lease.version) is False
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
