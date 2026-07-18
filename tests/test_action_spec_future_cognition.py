"""Tests for future-cognition action materialization."""

from __future__ import annotations

import json

import pytest

from kazusa_ai_chatbot.calendar_scheduler import models as calendar_models
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


def test_build_future_cognition_calendar_docs_use_prompt_safe_shape() -> None:
    """Calendar rows should carry a typed cognition request, not visible text."""

    from kazusa_ai_chatbot.action_spec.handlers.future_cognition import (
        build_future_cognition_calendar_documents,
    )

    documents = build_future_cognition_calendar_documents(
        _future_cognition_action_spec(),
        storage_timestamp_utc="2026-05-15T21:00:00+00:00",
        action_attempt_id="action_attempt:future-123",
    )
    schedule = documents["schedule"]
    run = documents["run"]
    serialized_payload = json.dumps(schedule["payload"], sort_keys=True)

    assert schedule["trigger_kind"] == calendar_models.TRIGGER_FUTURE_COGNITION
    assert schedule["next_run_at"] == "2026-05-15T22:00:00+00:00"
    assert schedule["status"] == calendar_models.SCHEDULE_STATUS_ACTIVE
    assert schedule["payload"]["episode_type"] == "self_cognition"
    assert schedule["payload"]["source_action_attempt_id"] == (
        "action_attempt:future-123"
    )
    assert schedule["payload"]["continuation_objective"] == (
        "Re-check whether a natural pause appeared."
    )
    assert schedule["source_scope"]["source_platform"] == "orchestrator"
    assert run["trigger_kind"] == calendar_models.TRIGGER_FUTURE_COGNITION
    assert run["status"] == calendar_models.RUN_STATUS_PENDING
    assert run["payload"] == schedule["payload"]
    assert "context_summary" not in schedule["payload"]
    for forbidden in (
        "handler_id",
        "platform_channel_id",
        "raw_channel",
        "raw-message",
        "mongodb",
        "credential",
        '"params"',
        "visible_text",
        "send_message",
    ):
        assert forbidden not in serialized_payload


def test_future_cognition_calendar_docs_copy_trusted_source_scope() -> None:
    """Code-bound source scope should survive into calendar source scope."""

    from kazusa_ai_chatbot.action_spec.handlers.future_cognition import (
        build_future_cognition_calendar_documents,
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

    documents = build_future_cognition_calendar_documents(
        action_spec,
        storage_timestamp_utc="2026-05-15T21:00:00+00:00",
        action_attempt_id="action_attempt:future-123",
    )
    schedule = documents["schedule"]

    assert schedule["source_scope"]["source_platform"] == "qq"
    assert schedule["source_scope"]["source_channel_id"] == "54369546"
    assert schedule["source_scope"]["source_channel_type"] == "group"
    assert schedule["source_scope"]["source_user_id"] == "673225019"
    assert schedule["source_scope"]["source_platform_bot_id"] == "bot-001"
    assert schedule["source_scope"]["source_character_name"] == "TestCharacter"
    assert "source_channel_id" not in schedule["payload"]
    assert "source_user_id" not in schedule["payload"]


def test_future_cognition_missing_source_user_does_not_fabricate_identity() -> None:
    """Missing source user should stay targetless instead of becoming a user."""

    from kazusa_ai_chatbot.action_spec.handlers.future_cognition import (
        build_future_cognition_calendar_documents,
    )

    documents = build_future_cognition_calendar_documents(
        _future_cognition_action_spec(),
        storage_timestamp_utc="2026-05-15T21:00:00+00:00",
        action_attempt_id="action_attempt:future-123",
    )

    assert documents["schedule"]["source_scope"]["source_user_id"] == ""


@pytest.mark.asyncio
async def test_execute_future_cognition_schedules_without_inline_cognition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The executor should persist the request and return action evidence."""

    from kazusa_ai_chatbot.action_spec.handlers import future_cognition

    schedules: list[dict] = []
    runs: list[dict] = []

    async def upsert_schedule(schedule: dict) -> None:
        schedules.append(schedule)

    async def upsert_run(run: dict) -> None:
        runs.append(run)

    monkeypatch.setattr(
        future_cognition.calendar_repository,
        "upsert_calendar_schedule",
        upsert_schedule,
    )
    monkeypatch.setattr(
        future_cognition.calendar_repository,
        "upsert_calendar_run",
        upsert_run,
    )

    result = await future_cognition.execute_future_cognition_action(
        _future_cognition_action_spec(),
        storage_timestamp_utc="2026-05-15T21:00:00+00:00",
        action_attempt_id="action_attempt:future-123",
    )

    assert result["status"] == "scheduled"
    assert result["calendar_trigger_kind"] == (
        calendar_models.TRIGGER_FUTURE_COGNITION
    )
    assert result["scheduled_count"] == 1
    assert result["calendar_schedule_id"] == schedules[0]["schedule_id"]
    assert result["calendar_run_id"] == runs[0]["run_id"]
    assert result["episode_type"] == "self_cognition"
    assert result["trigger_at"] == "2026-05-15T22:00:00+00:00"
    assert result["reason"] == (
        "The character wants a later private cognition cycle."
    )
    assert "scheduled_event_ids" not in result
    assert len(schedules) == 1
    assert len(runs) == 1
    assert schedules[0]["trigger_kind"] == (
        calendar_models.TRIGGER_FUTURE_COGNITION
    )
    assert runs[0]["schedule_id"] == schedules[0]["schedule_id"]


@pytest.mark.asyncio
async def test_execute_future_cognition_trace_records_calendar_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Action execution should not keep the old scheduled-event result shape."""

    from kazusa_ai_chatbot.action_spec import execution as execution_module

    async def execute_future_cognition_action(
        action_spec: dict,
        *,
        storage_timestamp_utc: str,
        action_attempt_id: str,
    ) -> dict:
        del action_spec, storage_timestamp_utc, action_attempt_id
        return {
            "status": "scheduled",
            "calendar_trigger_kind": calendar_models.TRIGGER_FUTURE_COGNITION,
            "calendar_schedule_id": "calendar_schedule_123",
            "calendar_run_id": "calendar_run_123",
            "scheduled_count": 1,
            "episode_type": "self_cognition",
            "trigger_at": "2026-05-15T22:00:00+00:00",
            "reason": "The character wants a later private cognition cycle.",
        }

    monkeypatch.setattr(
        execution_module,
        "execute_future_cognition_action",
        execute_future_cognition_action,
    )

    recorded_attempts: list[dict] = []

    async def record_attempt(record: dict) -> None:
        recorded_attempts.append(record)

    results = await execution_module.execute_action_specs_for_trace(
        [_future_cognition_action_spec()],
        storage_timestamp_utc="2026-05-15T21:00:00+00:00",
        record_attempt_func=record_attempt,
    )

    assert results[0]["status"] == "scheduled"
    assert "scheduled self-cognition follow-up" in results[0]["result_summary"]
    assert "calendar" not in results[0]["result_summary"]
    assert "calendar_run_123" not in results[0]["result_summary"]
    execution_result = recorded_attempts[0]["execution_result"]
    assert execution_result["calendar_schedule_id"] == "calendar_schedule_123"
    assert execution_result["calendar_run_id"] == "calendar_run_123"
    assert "scheduled_event_ids" not in execution_result


def test_future_cognition_rejects_raw_target_id() -> None:
    """The LLM must not select scheduler or adapter identifiers."""

    from kazusa_ai_chatbot.action_spec.handlers.future_cognition import (
        build_future_cognition_calendar_documents,
    )

    action_spec = _future_cognition_action_spec()
    action_spec["target"]["target_id"] = "raw-channel-123"

    with pytest.raises(ActionValidationError, match="target_id"):
        build_future_cognition_calendar_documents(
            action_spec,
            storage_timestamp_utc="2026-05-15T21:00:00+00:00",
            action_attempt_id="action_attempt:future-123",
        )


def test_future_cognition_rejects_invalid_episode_type() -> None:
    """Only self-cognition follow-up slots are in scope for this plan."""

    from kazusa_ai_chatbot.action_spec.handlers.future_cognition import (
        build_future_cognition_calendar_documents,
    )

    action_spec = _future_cognition_action_spec()
    action_spec["params"]["episode_type"] = "scheduled_tick"

    with pytest.raises(ActionValidationError, match="episode_type"):
        build_future_cognition_calendar_documents(
            action_spec,
            storage_timestamp_utc="2026-05-15T21:00:00+00:00",
            action_attempt_id="action_attempt:future-123",
        )


def test_future_cognition_rejects_unbounded_continuation() -> None:
    """A future cognition request must keep continuation depth bounded."""

    from kazusa_ai_chatbot.action_spec.handlers.future_cognition import (
        build_future_cognition_calendar_documents,
    )

    action_spec = _future_cognition_action_spec()
    action_spec["continuation"]["max_depth"] = 2

    with pytest.raises(ActionValidationError, match="max_depth"):
        build_future_cognition_calendar_documents(
            action_spec,
            storage_timestamp_utc="2026-05-15T21:00:00+00:00",
            action_attempt_id="action_attempt:future-123",
        )


def test_future_cognition_rejects_offset_aware_llm_trigger_time() -> None:
    """LLM-produced schedule times must be exact configured-local minutes."""

    from kazusa_ai_chatbot.action_spec.handlers.future_cognition import (
        build_future_cognition_calendar_documents,
    )

    action_spec = _future_cognition_action_spec()
    action_spec["params"]["trigger_at"] = "2026-05-16T10:00:00+12:00"

    with pytest.raises(ActionValidationError, match="trigger_at"):
        build_future_cognition_calendar_documents(
            action_spec,
            storage_timestamp_utc="2026-05-15T21:00:00+00:00",
            action_attempt_id="action_attempt:future-123",
        )
