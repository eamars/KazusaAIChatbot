"""Tests for the public event logging interface."""

from __future__ import annotations

import asyncio
import inspect
import json
from datetime import datetime, timezone

import pytest

from kazusa_ai_chatbot import event_logging
import kazusa_ai_chatbot.event_logging.recording as recording_module


_RECORDER_NAMES = [
    "record_process_event",
    "record_worker_event",
    "record_llm_stage_event",
    "record_runtime_error_event",
    "record_pipeline_turn_event",
    "record_queue_intake_event",
    "record_rag_stage_event",
    "record_dialog_quality_event",
    "record_dispatcher_event",
    "record_database_operation_event",
    "record_self_cognition_event",
    "record_model_contract_event",
    "record_resource_health_event",
]


def test_public_recorders_are_async_keyword_only() -> None:
    """Public recorders should expose explicit async keyword-only contracts."""

    for name in _RECORDER_NAMES:
        recorder = getattr(event_logging, name)
        signature = inspect.signature(recorder)

        assert inspect.iscoroutinefunction(recorder)
        assert "payload" not in signature.parameters
        for parameter in signature.parameters.values():
            assert parameter.kind is inspect.Parameter.KEYWORD_ONLY


def test_public_module_does_not_export_generic_record_event() -> None:
    """Callers should not have access to a generic event recorder."""

    assert not hasattr(event_logging, "record_event")


@pytest.mark.asyncio
async def test_process_event_records_sanitized_scope(monkeypatch) -> None:
    """Recorder should hash channel scope and return the public result shape."""

    captured: dict[str, object] = {}

    async def write_event(document):
        captured.update(document)
        event_id = str(document["event_id"])
        return event_id

    monkeypatch.setattr(recording_module.repository, "write_event", write_event)

    result = await event_logging.record_queue_intake_event(
        component="service.chat_queue",
        correlation_id="corr-1",
        status="accepted",
        queue_depth=2,
        coalesced_count=0,
        dropped_count=0,
        protected_by_reply=False,
        listen_only=False,
        scope={
            "platform": "qq",
            "platform_channel_id": "raw-channel-1",
            "channel_type": "group",
        },
        occurred_at=datetime(2026, 5, 14, tzinfo=timezone.utc),
    )

    assert result["accepted"] is True
    assert result["status"] == "recorded"
    assert result["event_id"] == captured["event_id"]
    assert captured["event_family"] == "queue_intake"
    scope = captured["scope"]
    assert isinstance(scope, dict)
    assert scope["platform"] == "qq"
    assert scope["platform_channel_ref"].startswith("ch_")
    assert scope["platform_channel_ref"] != "raw-channel-1"
    serialized = json.dumps(captured, sort_keys=True)
    assert "raw-channel-1" not in serialized


@pytest.mark.asyncio
async def test_recorder_rejects_invalid_severity(monkeypatch) -> None:
    """Invalid severity should be rejected without calling the repository."""

    async def write_event(document):
        raise AssertionError("repository should not be called")

    monkeypatch.setattr(recording_module.repository, "write_event", write_event)

    result = await event_logging.record_worker_event(
        event_type="tick",
        component="reflection_cycle.worker",
        worker_name="reflection_cycle",
        enabled=True,
        dry_run=False,
        run_kind="hourly_slot",
        status="succeeded",
        severity="nope",  # type: ignore[arg-type]
    )

    assert result == {
        "accepted": False,
        "event_id": result["event_id"],
        "status": "rejected",
        "reason": "invalid severity",
    }


@pytest.mark.asyncio
async def test_self_cognition_budget_persists_only_approved_counters(
    monkeypatch,
) -> None:
    """Self-cognition recorder should drop unapproved budget keys."""

    captured: dict[str, object] = {}

    async def write_event(document):
        captured.update(document)
        event_id = str(document["event_id"])
        return event_id

    monkeypatch.setattr(recording_module.repository, "write_event", write_event)

    result = await event_logging.record_self_cognition_event(
        component="self_cognition.worker",
        case_id="case-1",
        trigger_kind="active_commitment",
        selected_route="send_message",
        output_mode="scheduled_action_request",
        budget={
            "rag_calls": 1,
            "cognition_calls": 1,
            "dialog_calls": 0,
            "topic_limit": 3,
            "unapproved_budget_key": "private source text",
        },  # type: ignore[typeddict-unknown-key]
        dispatch_status="accepted",
        status="succeeded",
    )

    assert result["status"] == "recorded"
    payload = captured["payload"]
    assert isinstance(payload, dict)
    assert payload["budget"] == {
        "rag_calls": 1,
        "cognition_calls": 1,
        "dialog_calls": 0,
        "topic_limit": 3,
    }
    serialized = json.dumps(captured, sort_keys=True)
    assert "private source text" not in serialized


@pytest.mark.asyncio
async def test_recorder_timeout_is_best_effort(monkeypatch) -> None:
    """A slow repository write should return failed instead of raising."""

    async def write_event(document):
        await asyncio.sleep(0.05)
        event_id = str(document["event_id"])
        return event_id

    monkeypatch.setattr(recording_module.repository, "write_event", write_event)
    monkeypatch.setattr(
        recording_module,
        "EVENT_LOG_WRITE_TIMEOUT_SECONDS",
        0.001,
    )

    result = await event_logging.record_resource_health_event(
        component="service",
        resource_name="mongodb",
        resource_kind="database",
        availability="unknown",
        latency_ms=0,
    )

    assert result["accepted"] is False
    assert result["status"] == "failed"
    assert "TimeoutError" in result["reason"]


def test_public_types_are_exported() -> None:
    """Public typed shapes should be importable from the package."""

    assert event_logging.EventLogWriteResult
    assert event_logging.EventScopeInput
    assert event_logging.SelfCognitionBudget
    assert event_logging.EventRefRecord
