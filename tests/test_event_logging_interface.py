"""Tests for the current public event logging interface."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

import kazusa_ai_chatbot.event_logging.recording as recording_module
from kazusa_ai_chatbot import event_logging

_RECORDER_NAMES = [
    "record_continuity_boundary_event",
    "record_character_identity_growth_event",
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






@pytest.mark.asyncio
async def test_continuity_boundary_payload_is_bounded_and_text_free(monkeypatch) -> None:
    captured: dict[str, object] = {}

    async def write_event(document):
        captured.update(document)
        return str(document["event_id"])

    monkeypatch.setattr(recording_module.repository, "write_event", write_event)
    result = await event_logging.record_continuity_boundary_event(
        component="conversation_progress.runtime",
        boundary="progress_record",
        status="succeeded",
        scope_kind="group_scene",
        candidate_count=1000,
        selected_count=1000,
        packet_turn_count=1000,
        protected_anchor_count=1000,
        rendered_chars=100000,
        packet_age="recent",
        source_age="fresh",
        recorder_disposition="append",
        write_disposition="written",
        cache_disposition="published",
        barrier_disposition="none",
        trace_ref="opaque-trace-ref",
        correlation_ref="opaque-correlation-ref",
        operation_ref="opaque-operation-ref",
    )
    assert result["accepted"] is True
    assert captured["payload"] == {
        "candidate_count": 1000,
        "selected_count": 1000,
        "packet_turn_count": 1000,
        "protected_anchor_count": 1000,
        "rendered_chars": 100000,
    }
    assert "prompt_text" not in json.dumps(captured, ensure_ascii=False)


@pytest.mark.asyncio
async def test_process_event_records_sanitized_scope(monkeypatch) -> None:
    captured: dict[str, object] = {}

    async def write_event(document):
        captured.update(document)
        return str(document["event_id"])

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
    assert result["status"] == "recorded"
    serialized = json.dumps(captured, sort_keys=True)
    assert "raw-channel-1" not in serialized




def test_public_types_are_exported() -> None:
    assert event_logging.EventLogWriteResult
    assert event_logging.EventScopeInput
    assert event_logging.SelfCognitionBudget
    assert event_logging.EventRefRecord
