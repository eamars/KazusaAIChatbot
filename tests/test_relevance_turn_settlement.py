"""Deterministic lifecycle tests for the relevance turn-settlement DAG."""

from __future__ import annotations

import asyncio
from importlib import import_module
from pathlib import Path

import pytest

from kazusa_ai_chatbot import relevance as relevance_public
from kazusa_ai_chatbot.brain_service.post_turn import settle_episode_trace
from kazusa_ai_chatbot.brain_service.turn_settlement import (
    PersistedChatFragment,
    TurnSettlementCoordinator,
)
from kazusa_ai_chatbot.relevance.contracts import (
    FrontlineDecision,
    RelevanceEvaluationEnvelope,
    SettledRelevanceDecision,
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


def _frontline_evaluation(
    decision: FrontlineDecision,
    *,
    attempt_diagnostics: list[dict[str, object]] | None = None,
) -> RelevanceEvaluationEnvelope:
    """Wrap one exact frontline decision for the coordinator boundary."""

    return {
        "decision": decision,
        "attempt_diagnostics": list(attempt_diagnostics or []),
    }


def _settled_evaluation(
    decision: SettledRelevanceDecision,
    *,
    attempt_diagnostics: list[dict[str, object]] | None = None,
) -> RelevanceEvaluationEnvelope:
    """Wrap one exact settled decision for the coordinator boundary."""

    return {
        "decision": decision,
        "attempt_diagnostics": list(attempt_diagnostics or []),
    }


def _start_evaluation(reason: str = "new candidate") -> RelevanceEvaluationEnvelope:
    """Build a common start evaluation for lifecycle tests."""

    return _frontline_evaluation({
        "intake_action": "start",
        "append_target": "none",
        "prelude_targets": [],
        "reason": reason,
    })


def _append_evaluation(reason: str = "same candidate") -> RelevanceEvaluationEnvelope:
    """Build a common append evaluation for lifecycle tests."""

    return _frontline_evaluation({
        "intake_action": "append",
        "append_target": "open_1",
        "prelude_targets": [],
        "reason": reason,
    })


def _discard_evaluation(reason: str = "discard") -> RelevanceEvaluationEnvelope:
    """Build a common discard evaluation for lifecycle tests."""

    return _frontline_evaluation({
        "intake_action": "discard",
        "append_target": "none",
        "prelude_targets": [],
        "reason": reason,
    })


def _diagnostic(error_code: str) -> dict[str, object]:
    """Build one valid bounded diagnostic row for carrier tests."""

    return {
        "schema_version": "episode_attempt_diagnostic.v1",
        "stage": "relevance",
        "error_code": error_code,
        "attempt_count": 2,
        "safe_checkpoint": "pre_state_commit",
        "retryable": False,
        "final_status": "accepted_degraded",
    }


def _coordinator(clock: _FakeClock, *, settled=None):
    """Build a coordinator with deterministic semantic stage doubles."""

    async def _frontline(_state):
        return _frontline_evaluation({
            "intake_action": "start",
            "append_target": "none",
            "prelude_targets": [],
            "reason": "new candidate",
        })

    async def _settled(lease, state):
        del lease, state
        return _settled_evaluation(settled or {
            "response_action": "proceed",
            "reason_to_respond": "grounded request",
            "use_reply_feature": True,
            "channel_topic": "request",
            "indirect_speech_context": "",
        })

    return TurnSettlementCoordinator(
        frontline_evaluator=_frontline,
        settled_evaluator=_settled,
        clock=clock,
    )


def test_relevance_exports_canonical_decision_types_without_producer_duplicates() -> None:
    """All relevance producers use the one canonical decision definitions."""

    contracts_module = import_module(
        "kazusa_ai_chatbot.relevance.contracts"
    )
    frontline_module = import_module(
        "kazusa_ai_chatbot.relevance.frontline_relevance_agent"
    )
    settled_module = import_module(
        "kazusa_ai_chatbot.relevance.persona_relevance_agent"
    )

    assert relevance_public.FrontlineDecision is contracts_module.FrontlineDecision
    assert relevance_public.SettledRelevanceDecision is (
        contracts_module.SettledRelevanceDecision
    )
    assert frontline_module.FrontlineDecision is contracts_module.FrontlineDecision
    assert settled_module.SettledRelevanceDecision is (
        contracts_module.SettledRelevanceDecision
    )
    assert "class FrontlineDecision" not in Path(
        frontline_module.__file__
    ).read_text(encoding="utf-8")
    assert "class SettledRelevanceDecision" not in Path(
        settled_module.__file__
    ).read_text(encoding="utf-8")


def test_relevance_evaluation_envelope_has_nested_decision_and_diagnostics_only() -> None:
    """The public carrier has exactly the nested decision and metadata keys."""

    envelope = _frontline_evaluation(
        {
            "intake_action": "discard",
            "append_target": "none",
            "prelude_targets": [],
            "reason": "not addressed",
        },
        attempt_diagnostics=[_diagnostic("frontline_relevance_deterministic_degraded")],
    )

    assert set(envelope) == {"decision", "attempt_diagnostics"}
    assert set(envelope["decision"]) == {
        "intake_action",
        "append_target",
        "prelude_targets",
        "reason",
    }
    assert len(envelope["attempt_diagnostics"]) == 1


@pytest.mark.asyncio
async def test_frontline_start_accumulates_relevance_diagnostics_in_order() -> None:
    """A start stores its diagnostic rows on the pending turn and lease."""

    clock = _FakeClock()
    coordinator = _coordinator(clock)
    first = _diagnostic("frontline-first")
    second = _diagnostic("frontline-second")
    start = await coordinator.apply_frontline_decision(
        _fragment(1),
        _frontline_evaluation(
            _start_evaluation()["decision"],
            attempt_diagnostics=[first, second],
        ),
    )

    assert coordinator._pending_turns[start.turn_id].attempt_diagnostics == [
        first,
        second,
    ]
    clock.advance(10.0)
    lease = await coordinator.wait_for_assessment_ready()
    assert list(lease.attempt_diagnostics) == [first, second]


@pytest.mark.asyncio
async def test_frontline_append_accumulates_relevance_diagnostics_in_order() -> None:
    """A valid append adds rows after the existing start rows."""

    clock = _FakeClock()
    coordinator = _coordinator(clock)
    first = _diagnostic("frontline-start")
    second = _diagnostic("frontline-append")
    start_decision = _start_evaluation()
    start = await coordinator.apply_frontline_decision(
        _fragment(1),
        _frontline_evaluation(
            start_decision["decision"],
            attempt_diagnostics=[first],
        ),
    )
    append_decision = _append_evaluation()
    await coordinator.apply_frontline_decision(
        _fragment(2, enqueue_monotonic=1.0),
        _frontline_evaluation(
            append_decision["decision"],
            attempt_diagnostics=[second],
        ),
    )

    assert coordinator._pending_turns[start.turn_id].attempt_diagnostics == [
        first,
        second,
    ]


@pytest.mark.asyncio
async def test_frontline_append_with_invalid_target_drops_diagnostics() -> None:
    """An append with no valid target discards its metadata with the fragment."""

    clock = _FakeClock()
    coordinator = _coordinator(clock)
    start = await coordinator.apply_frontline_decision(
        _fragment(1),
        _frontline_evaluation(
            _start_evaluation()["decision"],
            attempt_diagnostics=[_diagnostic("frontline-start")],
        ),
    )
    invalid_append = _frontline_evaluation(
        {
            "intake_action": "append",
            "append_target": "open_2",
            "prelude_targets": [],
            "reason": "slot unavailable",
        },
        attempt_diagnostics=[_diagnostic("frontline-invalid-append")],
    )

    outcome = await coordinator.apply_frontline_decision(
        _fragment(2, enqueue_monotonic=1.0),
        invalid_append,
    )

    assert outcome.action == "discard"
    assert coordinator._pending_turns[start.turn_id].attempt_diagnostics == [
        _diagnostic("frontline-start"),
    ]


@pytest.mark.asyncio
async def test_relevance_validators_receive_only_nested_decision(monkeypatch) -> None:
    """Coordinator validators never receive the outer metadata envelope."""

    import kazusa_ai_chatbot.brain_service.turn_settlement as settlement_module

    clock = _FakeClock()
    coordinator = _coordinator(clock)
    frontline_values = []
    settled_values = []
    original_frontline = settlement_module.validate_frontline_decision
    original_settled = settlement_module.validate_settled_relevance_decision

    def _frontline_validator(value):
        frontline_values.append(value)
        return original_frontline(value)

    def _settled_validator(value, *, observation_status):
        settled_values.append(value)
        return original_settled(
            value,
            observation_status=observation_status,
        )

    monkeypatch.setattr(
        settlement_module,
        "validate_frontline_decision",
        _frontline_validator,
    )
    monkeypatch.setattr(
        settlement_module,
        "validate_settled_relevance_decision",
        _settled_validator,
    )
    frontline_evaluation = await coordinator.evaluate_frontline({})
    await coordinator.apply_frontline_decision(
        _fragment(1),
        frontline_evaluation,
    )
    clock.advance(10.0)
    lease = await coordinator.wait_for_assessment_ready()
    settled_evaluation = await coordinator.evaluate_settled(lease, {})
    await coordinator.apply_settled_decision(lease, settled_evaluation)

    assert frontline_values
    assert settled_values
    assert all("attempt_diagnostics" not in value for value in frontline_values)
    assert all("attempt_diagnostics" not in value for value in settled_values)


@pytest.mark.asyncio
async def test_settled_relevance_diagnostics_append_after_frontline() -> None:
    """Settled degradation rows follow the retained frontline chronology."""

    clock = _FakeClock()
    settled_row = _diagnostic("settled-degraded")
    coordinator = _coordinator(clock)
    start = await coordinator.apply_frontline_decision(
        _fragment(1),
        _frontline_evaluation(
            _start_evaluation()["decision"],
            attempt_diagnostics=[_diagnostic("frontline-degraded")],
        ),
    )
    clock.advance(10.0)
    lease = await coordinator.wait_for_assessment_ready()
    settled = await coordinator.evaluate_settled(lease, {})
    settled["attempt_diagnostics"] = [settled_row]
    outcome = await coordinator.apply_settled_decision(lease, settled)

    assert outcome.claimable is True
    assert [row["error_code"] for row in outcome.attempt_diagnostics] == [
        "frontline-degraded",
        "settled-degraded",
    ]
    assert outcome.turn_id == start.turn_id


@pytest.mark.asyncio
async def test_stale_settled_lease_diagnostics_do_not_pollute_current_version() -> None:
    """Rows from a stale assessment never enter a newer turn version."""

    clock = _FakeClock()
    coordinator = _coordinator(clock)
    start_row = _diagnostic("frontline-start")
    append_row = _diagnostic("frontline-append")
    stale_row = _diagnostic("settled-stale")
    start = await coordinator.apply_frontline_decision(
        _fragment(1),
        _frontline_evaluation(
            _start_evaluation()["decision"],
            attempt_diagnostics=[start_row],
        ),
    )
    clock.advance(6.0)
    stale_lease = await coordinator.wait_for_assessment_ready()
    append_decision = _append_evaluation()
    await coordinator.apply_frontline_decision(
        _fragment(2, enqueue_monotonic=6.0),
        _frontline_evaluation(
            append_decision["decision"],
            attempt_diagnostics=[append_row],
        ),
    )
    stale_evaluation = _settled_evaluation(
        {
            "response_action": "proceed",
            "reason_to_respond": "stale",
            "use_reply_feature": False,
            "channel_topic": "",
            "indirect_speech_context": "",
        },
        attempt_diagnostics=[stale_row],
    )

    stale_outcome = await coordinator.apply_settled_decision(
        stale_lease,
        stale_evaluation,
    )
    assert stale_outcome.stale is True
    clock.advance(4.0)
    current_lease = await coordinator.wait_for_assessment_ready()

    assert [row["error_code"] for row in current_lease.attempt_diagnostics] == [
        "frontline-start",
        "frontline-append",
    ]
    assert "settled-stale" not in {
        row["error_code"] for row in current_lease.attempt_diagnostics
    }
    assert current_lease.turn_id == start.turn_id


@pytest.mark.asyncio
async def test_wait_diagnostics_carry_once_before_next_settled_append() -> None:
    """A wait carries its row into the next lease before later rows append."""

    clock = _FakeClock()
    responses = [
        _settled_evaluation(
            {
                "response_action": "wait",
                "reason_to_respond": "observe one more message",
                "use_reply_feature": False,
                "channel_topic": "",
                "indirect_speech_context": "",
            },
            attempt_diagnostics=[_diagnostic("settled-wait")],
        ),
        _settled_evaluation(
            {
                "response_action": "proceed",
                "reason_to_respond": "grounded request",
                "use_reply_feature": False,
                "channel_topic": "",
                "indirect_speech_context": "",
            },
            attempt_diagnostics=[_diagnostic("settled-final")],
        ),
    ]

    async def _settled(_lease, _state):
        return responses.pop(0)

    coordinator = TurnSettlementCoordinator(
        frontline_evaluator=_coordinator(clock)._frontline_evaluator,
        settled_evaluator=_settled,
        clock=clock,
    )
    start = await coordinator.apply_frontline_decision(
        _fragment(1),
        _frontline_evaluation(
            _start_evaluation()["decision"],
            attempt_diagnostics=[_diagnostic("frontline-start")],
        ),
    )
    clock.advance(6.0)
    first_lease = await coordinator.wait_for_assessment_ready()
    first_evaluation = await coordinator.evaluate_settled(first_lease, {})
    first_outcome = await coordinator.apply_settled_decision(
        first_lease,
        first_evaluation,
    )
    assert first_outcome.response_action == "wait"
    assert first_outcome.attempt_diagnostics == ()

    clock.advance(4.0)
    second_lease = await coordinator.wait_for_assessment_ready()
    assert [row["error_code"] for row in second_lease.attempt_diagnostics] == [
        "frontline-start",
        "settled-wait",
    ]
    second_evaluation = await coordinator.evaluate_settled(second_lease, {})
    second_outcome = await coordinator.apply_settled_decision(
        second_lease,
        second_evaluation,
    )

    assert second_outcome.turn_id == start.turn_id
    assert [row["error_code"] for row in second_outcome.attempt_diagnostics] == [
        "frontline-start",
        "settled-wait",
        "settled-final",
    ]


@pytest.mark.asyncio
async def test_relevance_diagnostics_are_bounded_to_sixteen_in_occurrence_order() -> None:
    """The carrier retains the newest sixteen rows chronologically."""

    clock = _FakeClock()
    coordinator = _coordinator(clock)
    start = await coordinator.apply_frontline_decision(
        _fragment(1),
        _frontline_evaluation(
            _start_evaluation()["decision"],
            attempt_diagnostics=[_diagnostic("row-01")],
        ),
    )
    append_decision = _append_evaluation()
    for sequence in range(2, 18):
        await coordinator.apply_frontline_decision(
            _fragment(sequence, enqueue_monotonic=1.0),
            _frontline_evaluation(
                append_decision["decision"],
                attempt_diagnostics=[_diagnostic(f"row-{sequence:02d}")],
            ),
        )

    clock.advance(10.0)
    lease = await coordinator.wait_for_assessment_ready()

    assert [row["error_code"] for row in lease.attempt_diagnostics] == [
        f"row-{sequence:02d}" for sequence in range(2, 18)
    ]
    assert lease.turn_id == start.turn_id


@pytest.mark.asyncio
async def test_discarded_turn_keeps_protected_trace_without_episode_diagnostics() -> None:
    """Discard removes the carrier while relevance trace ownership stays separate."""

    clock = _FakeClock()
    coordinator = _coordinator(clock)
    outcome = await coordinator.apply_frontline_decision(
        _fragment(1),
        _frontline_evaluation(
            _discard_evaluation()["decision"],
            attempt_diagnostics=[_diagnostic("frontline-discard")],
        ),
    )

    assert outcome.action == "discard"
    assert coordinator._pending_turns == {}
    assert not coordinator._ready_heap


@pytest.mark.asyncio
async def test_envelope_preserves_fifo_lease_and_claim_control_flow() -> None:
    """Envelope metadata does not alter FIFO lease or cognition claim order."""

    clock = _FakeClock()
    coordinator = _coordinator(clock)
    first = await coordinator.apply_frontline_decision(
        _fragment(1),
        _frontline_evaluation(
            _start_evaluation()["decision"],
            attempt_diagnostics=[_diagnostic("first")],
        ),
    )
    second = await coordinator.apply_frontline_decision(
        _fragment(2, author="author-b"),
        _frontline_evaluation(
            _start_evaluation()["decision"],
            attempt_diagnostics=[_diagnostic("second")],
        ),
    )
    clock.advance(6.0)
    first_lease = await coordinator.wait_for_assessment_ready()
    first_evaluation = await coordinator.evaluate_settled(first_lease, {})
    first_outcome = await coordinator.apply_settled_decision(
        first_lease,
        first_evaluation,
    )
    assert first_lease.turn_id == first.turn_id
    assert first_outcome.claimable is True
    assert await coordinator.claim_for_cognition(
        first.turn_id,
        first.version,
    ) is True
    second_lease = await coordinator.wait_for_assessment_ready()

    assert second_lease.turn_id == second.turn_id
    assert [row["error_code"] for row in second_lease.attempt_diagnostics] == [
        "second",
    ]


def test_service_consumes_only_settlement_outcome_diagnostics() -> None:
    """Service handoff reads metadata only from the settled outcome carrier."""

    service_source = Path(
        "src/kazusa_ai_chatbot/service.py"
    ).read_text(encoding="utf-8")
    lease_source = service_source[
        service_source.index("async def _process_settlement_lease"):
    ]

    assert "decision = evaluation[\"decision\"]" in lease_source
    assert (
        "settled_attempt_diagnostics=outcome.attempt_diagnostics"
        in lease_source
    )
    assert "settled_attempt_diagnostics=evaluation" not in lease_source


def test_settled_relevance_diagnostics_reach_episode_trace() -> None:
    """The existing episode trace carrier accepts the settled diagnostic row."""

    row = _diagnostic("settled_relevance_deterministic_degraded")
    trace = settle_episode_trace(
        episode={
            "episode_id": "episode-relevance-diagnostic",
            "trigger_source": "user_message",
            "created_at": "2026-07-16T00:00:00+00:00",
        },
        cognition_output=None,
        action_specs=[],
        action_results=[],
        surface_outputs=[],
        terminal_status="completed_private",
        attempt_diagnostics=[row],
        delivery_correlation={
            "schema_version": "delivery_correlation.v1",
            "delivery_intent": "deliver_now",
            "tracking_id": "",
            "receipt_status": "not_applicable",
            "receipt_ref": "",
        },
        settled_at="2026-07-16T00:00:01+00:00",
    )

    assert trace["attempt_diagnostics"] == [row]


@pytest.mark.asyncio
async def test_group_turn_uses_six_second_quiet_window() -> None:
    """An admitted group turn becomes assessable after six seconds."""

    clock = _FakeClock()
    coordinator = _coordinator(clock)

    outcome = await coordinator.apply_frontline_decision(
        _fragment(1),
        _start_evaluation(),
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
        _start_evaluation(),
    )

    clock.advance(5.0)
    append = await coordinator.apply_frontline_decision(
        _fragment(2, body="clarification", enqueue_monotonic=5.0),
        _append_evaluation("same topic"),
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
    start_decision = _start_evaluation("new topic")

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
        _start_evaluation(),
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
        _start_evaluation(),
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
        _start_evaluation(),
    )
    clock.advance(6.0)
    stale_lease = await coordinator.wait_for_assessment_ready()
    append = await coordinator.apply_frontline_decision(
        _fragment(2, body="newer intent", enqueue_monotonic=clock()),
        _append_evaluation(),
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
        _start_evaluation(),
    )
    clock.advance(6.0)
    lease = await coordinator.wait_for_assessment_ready()
    decision = await coordinator.evaluate_settled(lease, {})
    append = await coordinator.apply_frontline_decision(
        _fragment(2, body="newer intent"),
        _append_evaluation(),
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
    decision = _start_evaluation()
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
        return _discard_evaluation("irrelevant")

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
    start = _start_evaluation()
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
        _discard_evaluation("not addressed yet"),
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
        _frontline_evaluation({
            "intake_action": "start",
            "append_target": "none",
            "prelude_targets": ["prelude_1"],
            "reason": "the tag promotes the prior description",
        }),
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
        _discard_evaluation("third-party traffic"),
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
        _start_evaluation(),
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
        _append_evaluation(),
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
        _start_evaluation("private candidate"),
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
        _start_evaluation("candidate"),
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
        _start_evaluation(),
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
