"""Tests for future-cognition action materialization."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from kazusa_ai_chatbot.action_spec.models import ActionValidationError
from kazusa_ai_chatbot.action_spec.registry import (
    TRIGGER_FUTURE_COGNITION_CAPABILITY,
)


def _source_ref() -> dict:
    return {
        "schema_version": "action_source_ref.v1",
        "ref_kind": "cognitive_episode",
        "ref_id": "episode-123",
        "owner": "cognition",
        "relationship": "basis",
        "evidence_refs": [],
    }


def _continuation() -> dict:
    return {
        "schema_version": "action_continuation.v1",
        "mode": "scheduled_followup",
        "episode_type": "self_cognition",
        "max_depth": 1,
        "include_result_as": "scheduled_event",
    }


def _future_cognition_action_spec() -> dict:
    return {
        "schema_version": "action_spec.v1",
        "kind": TRIGGER_FUTURE_COGNITION_CAPABILITY,
        "cognition_mode": "deliberative",
        "source_refs": [_source_ref()],
        "target": {
            "schema_version": "action_target.v1",
            "target_kind": "cognitive_episode",
            "target_id": None,
            "owner": "orchestrator",
            "scope": {"episode_type": "self_cognition"},
        },
        "params": {
            "episode_type": "self_cognition",
            "trigger_at": "2026-05-16 10:00",
            "continuation_objective": "Re-check whether a natural pause appeared.",
        },
        "urgency": "scheduled",
        "visibility": "private",
        "deadline": None,
        "continuation": _continuation(),
        "reason": "The character wants a later private cognition cycle.",
    }


def test_build_future_cognition_event_uses_prompt_safe_scheduler_shape() -> None:
    """The scheduled event should carry a typed cognition request, not ids."""

    from kazusa_ai_chatbot.action_spec.handlers.future_cognition import (
        build_future_cognition_scheduled_event,
    )

    event = build_future_cognition_scheduled_event(
        _future_cognition_action_spec(),
        storage_timestamp_utc="2026-05-15T21:00:00+00:00",
        action_attempt_id="action_attempt:future-123",
    )
    serialized = json.dumps(event, sort_keys=True)

    assert event["tool"] == TRIGGER_FUTURE_COGNITION_CAPABILITY
    assert event["execute_at"] == "2026-05-15T22:00:00+00:00"
    assert event["status"] == "pending"
    assert event["args"]["episode_type"] == "self_cognition"
    assert event["args"]["source_action_attempt_id"] == (
        "action_attempt:future-123"
    )
    assert event["args"]["continuation_objective"] == (
        "Re-check whether a natural pause appeared."
    )
    assert "context_summary" not in event["args"]
    for forbidden in (
        "handler_id",
        "platform_channel_id",
        "raw_channel",
        "raw-message",
        "mongodb",
        "credential",
        '"params"',
    ):
        assert forbidden not in serialized


def test_future_cognition_event_copies_trusted_source_scope() -> None:
    """Code-bound source scope should survive into the scheduled trigger."""

    from kazusa_ai_chatbot.action_spec.handlers.future_cognition import (
        build_future_cognition_scheduled_event,
    )

    action_spec = _future_cognition_action_spec()
    action_spec["target"]["scope"].update(
        {
            "source_platform": "qq",
            "source_channel_id": "54369546",
            "source_channel_type": "group",
            "source_user_id": "673225019",
            "source_platform_bot_id": "bot-001",
            "source_character_name": "TestCharacter",
        }
    )

    event = build_future_cognition_scheduled_event(
        action_spec,
        storage_timestamp_utc="2026-05-15T21:00:00+00:00",
        action_attempt_id="action_attempt:future-123",
    )

    assert event["source_platform"] == "qq"
    assert event["source_channel_id"] == "54369546"
    assert event["source_channel_type"] == "group"
    assert event["source_user_id"] == "673225019"
    assert event["source_platform_bot_id"] == "bot-001"
    assert event["source_character_name"] == "TestCharacter"
    assert "source_channel_id" not in event["args"]
    assert "source_user_id" not in event["args"]


@pytest.mark.asyncio
async def test_execute_future_cognition_schedules_without_inline_cognition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The executor should persist the request and return action evidence."""

    from kazusa_ai_chatbot.action_spec.handlers import future_cognition

    scheduled_events: list[dict] = []

    async def fake_schedule_event(event: dict) -> str:
        scheduled_events.append(event)
        return_value = "scheduled-event-123"
        return return_value

    monkeypatch.setattr(
        future_cognition.scheduler,
        "schedule_event",
        fake_schedule_event,
    )

    result = await future_cognition.execute_future_cognition_action(
        _future_cognition_action_spec(),
        storage_timestamp_utc="2026-05-15T21:00:00+00:00",
        action_attempt_id="action_attempt:future-123",
    )

    assert result == {
        "status": "scheduled",
        "scheduled_event_ids": ["scheduled-event-123"],
        "episode_type": "self_cognition",
        "trigger_at": "2026-05-15T22:00:00+00:00",
        "reason": "The character wants a later private cognition cycle.",
    }
    assert len(scheduled_events) == 1
    assert scheduled_events[0]["tool"] == TRIGGER_FUTURE_COGNITION_CAPABILITY


def test_future_cognition_rejects_raw_target_id() -> None:
    """The LLM must not select scheduler or adapter identifiers."""

    from kazusa_ai_chatbot.action_spec.handlers.future_cognition import (
        build_future_cognition_scheduled_event,
    )

    action_spec = _future_cognition_action_spec()
    action_spec["target"]["target_id"] = "raw-channel-123"

    with pytest.raises(ActionValidationError, match="target_id"):
        build_future_cognition_scheduled_event(
            action_spec,
            storage_timestamp_utc="2026-05-15T21:00:00+00:00",
            action_attempt_id="action_attempt:future-123",
        )


def test_future_cognition_rejects_invalid_episode_type() -> None:
    """Only self-cognition follow-up slots are in scope for this plan."""

    from kazusa_ai_chatbot.action_spec.handlers.future_cognition import (
        build_future_cognition_scheduled_event,
    )

    action_spec = _future_cognition_action_spec()
    action_spec["params"]["episode_type"] = "reflection_signal"

    with pytest.raises(ActionValidationError, match="episode_type"):
        build_future_cognition_scheduled_event(
            action_spec,
            storage_timestamp_utc="2026-05-15T21:00:00+00:00",
            action_attempt_id="action_attempt:future-123",
        )


def test_future_cognition_rejects_unbounded_continuation() -> None:
    """A future cognition request must keep continuation depth bounded."""

    from kazusa_ai_chatbot.action_spec.handlers.future_cognition import (
        build_future_cognition_scheduled_event,
    )

    action_spec = _future_cognition_action_spec()
    action_spec["continuation"]["max_depth"] = 2

    with pytest.raises(ActionValidationError, match="max_depth"):
        build_future_cognition_scheduled_event(
            action_spec,
            storage_timestamp_utc="2026-05-15T21:00:00+00:00",
            action_attempt_id="action_attempt:future-123",
        )


def test_future_cognition_rejects_offset_aware_llm_trigger_time() -> None:
    """LLM-produced schedule times must be exact configured-local minutes."""

    from kazusa_ai_chatbot.action_spec.handlers.future_cognition import (
        build_future_cognition_scheduled_event,
    )

    action_spec = _future_cognition_action_spec()
    action_spec["params"]["trigger_at"] = "2026-05-16T10:00:00+12:00"

    with pytest.raises(ActionValidationError, match="trigger_at"):
        build_future_cognition_scheduled_event(
            action_spec,
            storage_timestamp_utc="2026-05-15T21:00:00+00:00",
            action_attempt_id="action_attempt:future-123",
        )
