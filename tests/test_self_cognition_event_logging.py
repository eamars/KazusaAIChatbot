"""Tests for self-cognition event-log mirroring."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock

import pytest

import kazusa_ai_chatbot.event_logging.recording as recording_module
from kazusa_ai_chatbot import event_logging
from kazusa_ai_chatbot.action_spec.registry import SPEAK_CAPABILITY
from kazusa_ai_chatbot.self_cognition import models, worker


def _target_scope() -> dict[str, str | None]:
    """Build a stable private target scope for test cases."""

    scope = {
        "platform": "qq",
        "platform_channel_id": "673225019",
        "channel_type": "private",
        "user_id": "673225019",
    }
    return scope


def _commitment_case() -> dict[str, Any]:
    """Build a self-cognition case that can choose outward contact."""

    case = {
        "case_name": models.CASE_COMMITMENT_PAST_DUE,
        "case_id": "commitment:promise-001",
        "idle_timestamp_utc": "2026-05-13T00:30:00+00:00",
        "last_evidence_timestamp_utc": "2026-05-13T00:00:00+00:00",
        "trigger_kind": models.TRIGGER_ACTIVE_COMMITMENT_DUE_CHECK,
        "semantic_due_state": models.DUE_STATE_PAST_DUE,
        "actionability": "past_due_commitment_contact_socially_available",
        "target_scope": _target_scope(),
        "source_refs": [
            {
                "source_kind": "user_memory_unit",
                "source_id": "promise-001",
                "due_at": "2026-05-13T00:00:00+00:00",
                "summary": "A promised follow-up is due.",
            }
        ],
        "visible_context": [
            {
                "role": "user",
                "body_text": "Please check back after the appointment.",
                "timestamp": "2026-05-12T23:50:00+00:00",
            }
        ],
    }
    return case


def _action_cognition_output() -> dict[str, Any]:
    """Build a cognition output that selects visible dialog through speak."""

    output = {
        "logical_stance": "CONFIRM",
        "character_intent": "PROVIDE",
        "action_directives": {
            "contextual_directives": {
                "social_distance": "friendly",
                "emotional_intensity": "low",
                "vibe_check": "focused",
                "relational_dynamic": "scheduled follow-up",
            },
            "linguistic_directives": {
                "rhetorical_strategy": "answer the scheduled follow-up",
                "linguistic_style": "brief",
                "accepted_user_preferences": [],
                "content_plan": {"semantic_content": "Checking in now."},
            },
        },
        "action_specs": [
            {
                "kind": SPEAK_CAPABILITY,
                "visibility": "user_visible",
            }
        ],
    }
    return output


def _case_runner_with_tracking(
    case: dict[str, Any],
) -> dict[str, Any]:
    """Build records that resemble a completed runner case."""

    trigger_record = {
        "trigger_id": "self_cognition_trigger:promise-001",
        "trigger_kind": case["trigger_kind"],
        "target_scope": case["target_scope"],
        "source_refs": case["source_refs"],
        "semantic_due_state": case["semantic_due_state"],
        "actionability": case["actionability"],
        "status": "accepted",
    }
    run_record = {
        "run_id": "self_cognition_run:promise-001",
        "trigger_id": trigger_record["trigger_id"],
        "idle_timestamp_utc": case["idle_timestamp_utc"],
        "output_mode": "scheduled_action_request",
        "selected_route": models.ROUTE_ACTION_CANDIDATE,
        "status": "completed",
        "evidence_refs": case["source_refs"],
        "budget": {
            "rag_calls": 0,
            "cognition_calls": 1,
            "dialog_calls": 0,
            "topic_limit": 1,
        },
    }
    action_attempt = {
        "attempt_id": "self_cognition_attempt:promise-001",
        "run_id": run_record["run_id"],
        "trigger_id": trigger_record["trigger_id"],
        "source_kind": "user_memory_unit",
        "source_id": "promise-001",
        "target_scope": case["target_scope"],
        "action_kind": models.ACTION_KIND_SEND_MESSAGE,
        "due_at": "2026-05-13T00:00:00+00:00",
        "idempotency_key": "sha256:test",
        "status": models.ACTION_ATTEMPT_STATUS_CANDIDATE,
    }
    action_candidate = {
        "attempt_id": action_attempt["attempt_id"],
        "target_platform": "qq",
        "target_channel": "673225019",
        "target_channel_type": "private",
        "text": "Checking in now.",
        "execute_at": None,
        "dispatch_shape": models.ACTION_KIND_SEND_MESSAGE,
        "production_handoff": False,
    }
    payloads = {
        models.ARTIFACT_TRIGGER_RECORD: trigger_record,
        models.ARTIFACT_RUN_RECORD: run_record,
        models.ARTIFACT_ACTION_ATTEMPT: action_attempt,
        models.ARTIFACT_ACTION_CANDIDATE: action_candidate,
        models.ARTIFACT_CONSOLIDATION_OUTCOME: {
            "consolidation_called": True,
            "write_success": {"character_state": True},
            "scheduled_event_count": 0,
            "cache_evicted_count": 1,
            "origin_trigger_source": "internal_thought",
            "origin_episode_id": "self_cognition:tracking:promise-001",
        },
    }
    return payloads








@pytest.mark.asyncio
async def test_self_cognition_event_logger_records_target_binding_failure(
    monkeypatch,
) -> None:
    """Target binding failure events should preserve audit fields only."""

    captured: dict[str, object] = {}

    async def write_event(document):
        captured.update(document)
        event_id = str(document["event_id"])
        return event_id

    monkeypatch.setattr(recording_module.repository, "write_event", write_event)

    await event_logging.record_self_cognition_event(
        component="self_cognition.worker",
        case_id="case-1",
        trigger_kind=models.TRIGGER_ACTIVE_COMMITMENT_DUE_CHECK,
        selected_route="not_started",
        output_mode="none",
        budget={
            "rag_calls": 0,
            "cognition_calls": 0,
            "dialog_calls": 0,
            "topic_limit": 0,
        },
        dispatch_status="target_binding_failed",
        status="target_binding_failed",
        target_binding_failure={
            "status": "target_binding_failed",
            "reason": "private_channel_unavailable_and_source_missing",
            "platform": "qq",
            "source_ref": "promise-001",
            "source_platform_channel_id": "",
            "source_channel_type": "internal",
            "target_global_user_id": "secret-global-user-id",
            "target_platform_user_id": None,
        },
    )

    payload = captured["payload"]
    assert isinstance(payload, dict)
    assert payload["target_binding_failure"] == {
        "reason": "private_channel_unavailable_and_source_missing",
        "platform": "qq",
        "source_ref": "promise-001",
        "source_platform_channel_id": "",
        "source_channel_type": "internal",
        "has_target_global_user_id": True,
        "has_target_platform_user_id": False,
    }
    assert "semantic_disposition" not in payload
    assert "policy_disposition" not in payload
    assert "execution_disposition" not in payload
    serialized = json.dumps(captured, ensure_ascii=False, sort_keys=True)
    assert "secret-global-user-id" not in serialized


@pytest.mark.asyncio
async def test_self_cognition_event_logger_sanitizes_consolidation_outcome(
    monkeypatch,
) -> None:
    """Consolidation event metadata should keep only approved outcome fields."""

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
        selected_route=models.ROUTE_AUDIT_ONLY,
        output_mode="silent",
        budget={
            "rag_calls": 0,
            "cognition_calls": 1,
            "dialog_calls": 0,
            "topic_limit": 1,
        },
        dispatch_status="not_requested",
        status="completed",
        consolidation_outcome={
            "consolidation_called": True,
            "write_success": {
                "character_state": True,
                "raw_output": "Internal consolidation note.",
            },
            "scheduled_event_count": 0,
            "cache_evicted_count": 1,
            "origin_trigger_source": "internal_thought",
            "origin_episode_id": "self_cognition:tracking:promise-001",
            "source_packet_text": "Please check back after the appointment.",
        },
    )

    assert result["status"] == "recorded"
    payload = captured["payload"]
    assert isinstance(payload, dict)
    assert payload["consolidation_outcome"] == {
        "consolidation_called": True,
        "write_success": {"character_state": True},
        "scheduled_event_count": 0,
        "cache_evicted_count": 1,
        "origin_trigger_source": "internal_thought",
        "origin_episode_id": "self_cognition:tracking:promise-001",
    }
    serialized = json.dumps(captured, ensure_ascii=False, sort_keys=True)
    assert "Internal consolidation note" not in serialized
    assert "Please check back" not in serialized
