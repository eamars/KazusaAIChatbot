"""Tests for the public event logging interface."""

from __future__ import annotations

import asyncio
import inspect
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pytest

from kazusa_ai_chatbot import event_logging
import kazusa_ai_chatbot.event_logging.recording as recording_module
import kazusa_ai_chatbot.event_logging.status as status_module
from kazusa_ai_chatbot.event_logging.sanitization import (
    sanitize_cognition_v2_event_fields,
)


_RECORDER_NAMES = [
    "record_cognition_v2_event",
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


def test_cognition_v2_sanitizer_redacts_state_ids_numbers_and_private_bids(
) -> None:
    """V2 diagnostics should retain only bounded semantic status fields."""

    fields = sanitize_cognition_v2_event_fields({
        "cognition_component": "goal_cognition",
        "selected_branch_id": "social_care",
        "state_scope": "user",
        "state_commit_status": "committed",
        "stage_status": "completed",
        "replacement_state": {"forbidden": True},
        "owner_key": "private-owner",
        "raw_intensity": 83,
        "prompt_text": "private prompt",
        "primary_bid": {"forbidden": True},
    })

    assert fields == {
        "cognition_component": "goal_cognition",
        "selected_branch_id": "social_care",
        "state_scope": "user",
        "state_commit_status": "committed",
        "stage_status": "completed",
    }


@pytest.mark.asyncio
async def test_cognition_v2_recorder_persists_only_bounded_fields(
    monkeypatch,
) -> None:
    """The V2 recorder should persist its registered bounded schema."""

    captured: dict[str, object] = {}

    async def write_event(document):
        captured.update(document)
        return str(document["event_id"])

    monkeypatch.setattr(recording_module.repository, "write_event", write_event)

    result = await event_logging.record_cognition_v2_event(
        component="nodes.persona_supervisor2",
        cognition_component="state_commit",
        status="committed",
        stage_status="completed",
        selected_branch_id="social_care",
        state_scope="user",
        state_commit_status="committed",
    )

    assert result["accepted"] is True
    assert captured["event_family"] == "cognition_v2"
    assert captured["payload"] == {}
    assert captured["cognition_v2"] == {
        "cognition_component": "state_commit",
        "selected_branch_id": "social_care",
        "state_scope": "user",
        "state_commit_status": "committed",
        "stage_status": "completed",
    }

_REPO_ROOT = Path(__file__).resolve().parents[1]
_STAGE7_TIME_BOUNDARY_DIRS = (
    "src/kazusa_ai_chatbot/event_logging",
    "src/kazusa_ai_chatbot/db",
    "src/kazusa_ai_chatbot/memory_evolution",
    "src/kazusa_ai_chatbot/global_character_growth",
    "src/kazusa_ai_chatbot/reflection_cycle",
    "src/scripts",
)
_STAGE7_FORBIDDEN_TIME_PATTERNS = (
    (
        re.compile(r"from kazusa_ai_chatbot\.time_context"),
        "legacy time_context import",
    ),
    (
        re.compile(r"build_character_" r"time_context"),
        "legacy time_context helper",
    ),
    (
        re.compile(r"datetime\.now\(timezone\.utc\)"),
        "direct UTC clock read",
    ),
    (
        re.compile(r"datetime\.datetime\.now\(datetime\.timezone\.utc\)"),
        "direct UTC clock read",
    ),
    (
        re.compile(r"datetime\.fromisoformat\("),
        "direct datetime parsing",
    ),
    (
        re.compile(r"datetime\.datetime\.fromisoformat\("),
        "direct datetime parsing",
    ),
    (
        re.compile(r"\.astimezone\((?!timezone\.utc\))"),
        "direct timezone conversion",
    ),
    (
        re.compile(r"ZoneInfo\("),
        "direct configured timezone loading",
    ),
)


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
async def test_recorder_uses_time_boundary_for_default_storage_timestamps(
    monkeypatch,
) -> None:
    """Recorder default event times should come from the canonical boundary."""

    captured: dict[str, object] = {}
    fixed_timestamp_utc = "2026-05-17T04:55:28+00:00"

    async def write_event(document):
        captured.update(document)
        event_id = str(document["event_id"])
        return event_id

    monkeypatch.setattr(recording_module.repository, "write_event", write_event)
    monkeypatch.setattr(
        recording_module,
        "storage_utc_now_iso",
        lambda: fixed_timestamp_utc,
    )

    result = await event_logging.record_process_event(
        event_type="startup",
        phase="lifespan",
        component="service",
        status="started",
        pid=123,
        host_label="host",
    )

    assert result["status"] == "recorded"
    assert captured["occurred_at"] == fixed_timestamp_utc
    assert captured["created_at"] == fixed_timestamp_utc


@pytest.mark.asyncio
async def test_status_builders_use_time_boundary_for_windows_and_generated_at(
    monkeypatch,
) -> None:
    """Ops status windows and generated timestamps should share one UTC clock."""

    captured_filters: list[dict[str, object]] = []
    fixed_now_utc = datetime(2026, 5, 17, 4, 55, 28, tzinfo=timezone.utc)

    async def count_events(filter_doc):
        captured_filters.append(filter_doc)
        return 0

    async def find_events(filter_doc, *, sort, limit):
        captured_filters.append(filter_doc)
        return []

    monkeypatch.setattr(status_module.repository, "count_events", count_events)
    monkeypatch.setattr(status_module.repository, "find_events", find_events)
    monkeypatch.setattr(status_module, "storage_utc_now", lambda: fixed_now_utc)

    result = await status_module.build_runtime_status(window_hours=1)

    assert result["generated_at"] == "2026-05-17T04:55:28+00:00"
    lower_bounds = [
        filter_doc["occurred_at"]["$gte"]
        for filter_doc in captured_filters
        if "occurred_at" in filter_doc
    ]
    assert lower_bounds
    assert set(lower_bounds) == {"2026-05-17T03:55:28+00:00"}


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
async def test_dialog_quality_event_persists_usage_mode(monkeypatch) -> None:
    """Dialog quality recorder should expose why dialog was invoked."""

    captured: dict[str, object] = {}

    async def write_event(document):
        captured.update(document)
        event_id = str(document["event_id"])
        return event_id

    monkeypatch.setattr(recording_module.repository, "write_event", write_event)

    result = await event_logging.record_dialog_quality_event(
        component="nodes.dialog_agent",
        correlation_id="corr-dialog",
        usage_mode="self_cognition_private_finalization",
        quality_status="passed",
        retry_count=1,
        failure_codes=[],
        content_plan_entry_count=3,
        status="succeeded",
    )

    assert result["status"] == "recorded"
    payload = captured["payload"]
    assert isinstance(payload, dict)
    assert payload["usage_mode"] == "self_cognition_private_finalization"
    assert payload["content_plan_entry_count"] == 3


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


def test_stage7_sources_import_time_boundary_for_time_conversion() -> None:
    """Stage 7 source files should not own UTC or local timezone conversion."""

    failures: list[str] = []
    for directory in _STAGE7_TIME_BOUNDARY_DIRS:
        source_root = _REPO_ROOT / directory
        for path in sorted(source_root.rglob("*.py")):
            text = path.read_text(encoding="utf-8")
            relative_path = path.relative_to(_REPO_ROOT).as_posix()
            for pattern, reason in _STAGE7_FORBIDDEN_TIME_PATTERNS:
                if pattern.search(text):
                    failures.append(f"{relative_path}: {reason}")

    assert failures == []
