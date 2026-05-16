"""Tests for self-cognition event-log mirroring."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from kazusa_ai_chatbot import event_logging
import kazusa_ai_chatbot.event_logging.recording as recording_module
from kazusa_ai_chatbot.self_cognition import models, runner, worker


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
        "idle_timestamp": "2026-05-13T00:30:00+00:00",
        "last_evidence_timestamp": "2026-05-13T00:00:00+00:00",
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
    """Build a cognition output that contains an action marker."""

    output = {
        "logical_stance": "outward contact is appropriate",
        "character_intent": "send a concise follow-up",
        "action_directives": {
            "linguistic_directives": {
                "content_anchors": [
                    "[ACTION_CANDIDATE] Checking in now.",
                ],
            },
        },
    }
    return output


def _case_runner_with_tracking(
    case: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    """Build records that resemble a completed runner case."""

    del output_dir
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
        "idle_timestamp": case["idle_timestamp"],
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
            "origin_episode_id": "self_cognition:dry_run:promise-001",
        },
    }
    return payloads


@pytest.mark.asyncio
async def test_runner_event_log_mirror_omits_candidate_text(
    monkeypatch,
    tmp_path,
) -> None:
    """Opt-in runner mirroring should store metadata but not candidate text."""

    record_self_cognition_event = AsyncMock()
    monkeypatch.setattr(
        runner.event_logging,
        "record_self_cognition_event",
        record_self_cognition_event,
    )

    await runner.run_self_cognition_case_async(
        _commitment_case(),
        tmp_path,
        cognition_client=lambda state: _action_cognition_output(),
        event_log_mirror=True,
    )

    record_self_cognition_event.assert_awaited_once()
    kwargs = record_self_cognition_event.await_args.kwargs
    assert kwargs["component"] == "self_cognition.runner"
    assert kwargs["case_id"] == "commitment:promise-001"
    assert kwargs["selected_route"] == models.ROUTE_ACTION_CANDIDATE
    assert kwargs["dispatch_status"] == "not_requested"
    assert "Checking in now." not in json.dumps(kwargs, ensure_ascii=False)


@pytest.mark.asyncio
async def test_worker_mirrors_production_run_without_dispatch_text(
    monkeypatch,
    tmp_path,
) -> None:
    """Production worker should mirror run metadata without delivery handoff."""

    record_self_cognition_event = AsyncMock()
    record_worker_event = AsyncMock()
    monkeypatch.setattr(
        worker.event_logging,
        "record_self_cognition_event",
        record_self_cognition_event,
    )
    monkeypatch.setattr(
        worker.event_logging,
        "record_worker_event",
        record_worker_event,
    )

    async def collect_cases(*, now: datetime, max_cases: int) -> list[dict[str, Any]]:
        del now, max_cases
        return [_commitment_case()]

    async def read_attempts(*, limit: int) -> list[dict[str, Any]]:
        assert limit > 0
        return []

    record_attempt = AsyncMock()

    result = await worker.run_self_cognition_worker_tick(
        output_root=tmp_path,
        now=datetime(2026, 5, 13, tzinfo=timezone.utc),
        is_primary_interaction_busy=lambda: False,
        collect_cases_func=collect_cases,
        run_case_func=_case_runner_with_tracking,
        read_attempts_func=read_attempts,
        record_attempt_func=record_attempt,
        max_cases=3,
    )

    assert result.processed_count == 1
    record_self_cognition_event.assert_awaited_once()
    self_kwargs = record_self_cognition_event.await_args.kwargs
    assert self_kwargs["component"] == "self_cognition.worker"
    assert self_kwargs["dispatch_status"] == "not_requested"
    assert self_kwargs["attempt_id"] == "self_cognition_attempt:promise-001"
    assert self_kwargs["consolidation_outcome"] == {
        "consolidation_called": True,
        "write_success": {"character_state": True},
        "scheduled_event_count": 0,
        "cache_evicted_count": 1,
        "origin_trigger_source": "internal_thought",
        "origin_episode_id": "self_cognition:dry_run:promise-001",
    }
    serialized = json.dumps({"self": self_kwargs}, ensure_ascii=False)
    assert "Checking in now." not in serialized
    record_worker_event.assert_awaited_once()
    record_attempt.assert_awaited_once()
    assert list(tmp_path.iterdir()) == []


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
            "dialog_calls": 1,
            "topic_limit": 1,
        },
        dispatch_status="not_requested",
        status="completed",
        consolidation_outcome={
            "consolidation_called": True,
            "write_success": {
                "character_state": True,
                "raw_output": "Private finalization for consolidation only.",
            },
            "scheduled_event_count": 0,
            "cache_evicted_count": 1,
            "origin_trigger_source": "internal_thought",
            "origin_episode_id": "self_cognition:dry_run:promise-001",
            "private_finalization": "Private finalization for consolidation only.",
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
        "origin_episode_id": "self_cognition:dry_run:promise-001",
    }
    serialized = json.dumps(captured, ensure_ascii=False, sort_keys=True)
    assert "Private finalization" not in serialized
    assert "Please check back" not in serialized
