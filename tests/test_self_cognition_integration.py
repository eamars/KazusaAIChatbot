"""Deterministic integration tests for the self-cognition runtime boundary."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from kazusa_ai_chatbot.calendar_scheduler import models as calendar_models
from kazusa_ai_chatbot.action_spec.registry import SPEAK_CAPABILITY
from kazusa_ai_chatbot.db import user_memory_units as memory_units_module
from kazusa_ai_chatbot.dispatcher import AdapterRegistry, SendResult
import kazusa_ai_chatbot.dispatcher.handlers as handlers_module
from kazusa_ai_chatbot.nodes.dialog_agent import StateContractError
from kazusa_ai_chatbot.self_cognition import models, projection, sources
from kazusa_ai_chatbot.self_cognition import tracking, worker


@pytest.fixture(autouse=True)
def _disable_event_log_writes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep deterministic self-cognition integration tests off MongoDB."""

    monkeypatch.setattr(
        worker.event_logging,
        "record_self_cognition_event",
        AsyncMock(),
    )
    monkeypatch.setattr(
        worker.event_logging,
        "record_worker_event",
        AsyncMock(),
    )
    monkeypatch.setattr(
        worker.event_logging,
        "record_runtime_error_event",
        AsyncMock(),
    )


def _target_scope() -> dict[str, str | None]:
    scope = {
        "platform": "qq",
        "platform_channel_id": "673225019",
        "channel_type": "private",
        "user_id": "673225019",
    }
    return scope


class _FakeMessagingAdapter:
    """Adapter double used by worker delivery integration tests."""

    platform = "qq"
    platform_bot_id = "bot-1"
    display_name = "Character"

    def __init__(self, *, fail: bool = False, can_send: bool = True) -> None:
        self.fail = fail
        self.can_send = can_send
        self.calls: list[dict[str, Any]] = []

    async def can_send_message(
        self,
        channel_id: str,
        *,
        channel_type: str,
    ) -> bool:
        """Return whether the fake adapter accepts the target channel."""

        del channel_id, channel_type
        return_value = self.can_send
        return return_value

    async def send_message(
        self,
        channel_id: str,
        text: str,
        *,
        channel_type: str,
        reply_to_msg_id: str | None = None,
        delivery_mentions: list[dict[str, Any]] | None = None,
    ) -> SendResult:
        """Capture one send request or raise a deterministic failure."""

        self.calls.append({
            "channel_id": channel_id,
            "text": text,
            "channel_type": channel_type,
            "reply_to_msg_id": reply_to_msg_id,
            "delivery_mentions": delivery_mentions,
        })
        if self.fail:
            raise RuntimeError("adapter failed")
        sent_at = datetime(2026, 5, 17, 5, 57, tzinfo=timezone.utc)
        result = SendResult(
            platform="qq",
            channel_id=channel_id,
            message_id="adapter-message-1",
            sent_at=sent_at,
        )
        return result


def _adapter_registry(adapter: _FakeMessagingAdapter) -> AdapterRegistry:
    """Build a registry containing one fake QQ adapter."""

    registry = AdapterRegistry()
    registry.register(adapter)
    return registry


def _commitment_case() -> dict[str, Any]:
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


def _delivery_target(
    *,
    channel_id: str = "dm-1",
    channel_type: str = "private",
    source_kind: str = "target_private_channel",
    fallback_reason: str = "",
) -> dict[str, Any]:
    """Build deterministic delivery target metadata for worker tests."""

    target = {
        "schema_version": "self_cognition_delivery_target.v1",
        "platform": "qq",
        "platform_channel_id": channel_id,
        "channel_type": channel_type,
        "target_global_user_id": "global-target",
        "target_platform_user_id": "qq-target",
        "source_kind": source_kind,
        "source_ref": "promise-001",
        "source_platform_channel_id": "group-1",
        "source_channel_type": "group",
        "source_message_id": "msg-1",
        "source_global_user_id": "global-target",
        "source_platform_bot_id": "bot-1",
        "source_character_name": "Character",
        "guild_id": "guild-1",
        "bot_permission_role": "user",
        "fallback_reason": fallback_reason,
    }
    return target


def _commitment_case_with_delivery_target(
    *,
    delivery_target: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a production worker case with bound delivery metadata."""

    case = _commitment_case()
    case["target_scope"] = {
        "platform": "qq",
        "platform_channel_id": "group-1",
        "channel_type": "group",
        "user_id": "global-target",
        "platform_user_id": "qq-target",
        "display_name": "Target User",
    }
    if delivery_target is None:
        delivery_target = _delivery_target()
    case["target_binding_status"] = "bound"
    case["delivery_target"] = delivery_target
    case["platform_bot_id"] = "bot-1"
    case["character_profile"] = {"name": "Character"}
    return case


def _target_binding_failed_case() -> dict[str, Any]:
    """Build a production worker case rejected before cognition."""

    case = _commitment_case()
    case["target_binding_status"] = "failed"
    case["target_binding_failure"] = {
        "status": "target_binding_failed",
        "reason": "private_channel_unavailable_and_source_missing",
        "platform": "qq",
        "source_ref": "promise-001",
        "source_platform_channel_id": "",
        "source_channel_type": "internal",
        "target_global_user_id": "global-target",
        "target_platform_user_id": "qq-target",
    }
    return case


def _future_cognition_run() -> dict[str, Any]:
    run = {
        "run_id": "calendar_run_future_123",
        "schedule_id": "calendar_schedule_future_123",
        "trigger_kind": calendar_models.TRIGGER_FUTURE_COGNITION,
        "due_at": "2026-05-16T10:00:00+00:00",
        "created_at": "2026-05-16T09:00:00+00:00",
        "status": calendar_models.RUN_STATUS_PENDING,
        "payload": {
            "episode_type": "self_cognition",
            "trigger_at": "2026-05-16T10:00:00+00:00",
            "continuation_objective": "Re-check whether a natural pause appeared.",
            "source_action_attempt_id": "action_attempt:future-123",
            "source_refs": [
                {
                    "ref_kind": "cognitive_episode",
                    "ref_id": "episode-123",
                    "owner": "cognition",
                    "relationship": "basis",
                    "evidence_refs": [],
                }
            ],
            "continuation": {
                "mode": "scheduled_followup",
                "episode_type": "self_cognition",
                "max_depth": 1,
                "include_result_as": "scheduled_event",
            },
        },
        "source_scope": {
            "source_platform": "orchestrator",
            "source_channel_id": "",
            "source_channel_type": "internal",
            "source_user_id": "self_cognition",
            "source_message_id": "action_attempt:future-123",
            "source_platform_bot_id": "",
            "source_character_name": "",
            "guild_id": None,
            "bot_role": "system",
        },
    }
    return run


def _commitment_due_run() -> dict[str, Any]:
    run = {
        "run_id": "calendar_run_commitment_123",
        "schedule_id": "calendar_schedule_commitment_123",
        "trigger_kind": calendar_models.TRIGGER_COMMITMENT_DUE_COGNITION,
        "due_at": "2026-05-13T00:00:00+00:00",
        "created_at": "2026-05-12T23:00:00+00:00",
        "status": calendar_models.RUN_STATUS_PENDING,
        "payload": {
            "unit_id": "promise-001",
            "global_user_id": "673225019",
            "due_at": "2026-05-13T00:00:00+00:00",
        },
        "source_scope": {},
    }
    return run


def _future_cognition_case() -> dict[str, Any]:
    case = {
        "case_name": models.CASE_SCHEDULED_FUTURE_COGNITION,
        "case_id": "scheduled_future_cognition_slot:2026-05-16T10:00:00+00:00",
        "idle_timestamp_utc": "2026-05-16T10:00:00+00:00",
        "last_evidence_timestamp_utc": "2026-05-16T10:00:00+00:00",
        "trigger_kind": models.TRIGGER_SCHEDULED_FUTURE_COGNITION,
        "semantic_due_state": models.DUE_STATE_DUE_NOW,
        "actionability": "scheduled_private_followup_ready_no_direct_contact",
        "target_scope": {
            "platform": "internal",
            "platform_channel_id": "",
            "channel_type": "internal",
            "user_id": None,
        },
        "source_refs": [
            {
                "source_kind": "scheduled_future_cognition_slot",
                "source_id": "scheduled_future_cognition_slot",
                "due_at": "2026-05-16T10:00:00+00:00",
                "summary": "Re-check whether a natural pause appeared.",
            }
        ],
        "visible_context": [],
        "source_calendar_run_id": "calendar_run_future_123",
        "target_binding_status": "bound",
        "delivery_target": _delivery_target(
            channel_id="group-1",
            channel_type="group",
            source_kind="self_cognition_source_channel",
            fallback_reason="",
        ),
    }
    return case


def _action_attempt(case: dict[str, Any], *, status: str) -> dict[str, Any]:
    source_ref = case["source_refs"][0]
    idempotency_key = tracking.build_idempotency_key(
        source_ref["source_kind"],
        source_ref["source_id"],
        source_ref["due_at"],
        case["target_scope"],
        models.ACTION_KIND_SEND_MESSAGE,
    )
    attempt = {
        "attempt_id": "self_cognition_attempt:promise-001",
        "run_id": "self_cognition_run:promise-001",
        "trigger_id": "self_cognition_trigger:promise-001",
        "source_kind": source_ref["source_kind"],
        "source_id": source_ref["source_id"],
        "target_scope": case["target_scope"],
        "action_kind": models.ACTION_KIND_SEND_MESSAGE,
        "due_at": source_ref["due_at"],
        "idempotency_key": idempotency_key,
        "status": status,
    }
    return attempt


def _action_candidate(attempt: dict[str, Any]) -> dict[str, Any]:
    candidate = {
        "attempt_id": attempt["attempt_id"],
        "target_platform": "qq",
        "target_channel": "673225019",
        "target_channel_type": "private",
        "text": "Checking in now.",
        "execute_at": None,
        "dispatch_shape": models.ACTION_KIND_SEND_MESSAGE,
        "production_handoff": False,
    }
    return candidate


def _selected_speak_artifacts(
    case: dict[str, Any],
    *,
    text: str = "Checking in now.",
    attempt_status: str = models.ACTION_ATTEMPT_STATUS_CANDIDATE,
) -> dict[str, Any]:
    """Build in-memory runner artifacts for a selected speak route."""

    trigger_record = tracking.build_trigger_record(case)
    run_record = tracking.build_run_record(
        case,
        trigger_record,
        selected_route=models.ROUTE_ACTION_CANDIDATE,
        budget={
            "rag_calls": 0,
            "cognition_calls": 1,
            "dialog_calls": 1,
            "topic_limit": 1,
        },
    )
    action_attempt = _action_attempt(case, status=attempt_status)
    action_candidate = tracking.build_action_candidate(
        case,
        action_attempt,
        text,
    )
    payloads: dict[str, Any] = {
        models.ARTIFACT_TRIGGER_RECORD: trigger_record,
        models.ARTIFACT_RUN_RECORD: run_record,
        models.ARTIFACT_ACTION_ATTEMPT: action_attempt,
    }
    if action_candidate is not None:
        payloads[models.ARTIFACT_ACTION_CANDIDATE] = action_candidate
    return payloads


def _patch_dispatcher_persistence(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep autonomous delivery tests off MongoDB persistence."""

    async def save_conversation(document: dict[str, Any]) -> str:
        assert document["body_text"]
        return "conversation-row-1"

    async def ensure_character_identity(**kwargs: Any) -> str:
        assert kwargs["platform"]
        return "character-global"

    async def apply_receipt(**kwargs: Any) -> None:
        assert kwargs["delivery_tracking_id"]

    monkeypatch.setattr(
        handlers_module,
        "save_conversation",
        save_conversation,
    )
    monkeypatch.setattr(
        handlers_module,
        "ensure_character_identity",
        ensure_character_identity,
    )
    monkeypatch.setattr(
        handlers_module,
        "apply_assistant_delivery_receipt",
        apply_receipt,
    )
    monkeypatch.setattr(
        handlers_module.event_logging,
        "record_dispatcher_event",
        AsyncMock(),
    )
    monkeypatch.setattr(
        handlers_module.event_logging,
        "record_runtime_error_event",
        AsyncMock(),
    )


def _progress_cognition_output() -> dict[str, Any]:
    """Build a cognition output that stays internal but affects state."""

    output = {
        "logical_stance": "OBSERVE",
        "character_intent": "WAIT",
        "self_cognition_route": models.ROUTE_PROGRESS_MAINTENANCE,
        "cognition_core_output": {
            "state_update": {"state_scope": "character"},
        },
        "cognition_state_committed": True,
    }
    return output


def _speak_action_spec() -> dict[str, Any]:
    """Build the selected visible action spec used by worker tests."""

    spec = {
        "kind": SPEAK_CAPABILITY,
        "visibility": "user_visible",
    }
    return spec


def _text_surface_output(content_plan: str = "Checking in now.") -> dict[str, Any]:
    """Build the canonical V2 surface result used by worker tests."""

    return {
        "schema_version": "text_surface_output.v2",
        "content_plan": content_plan,
        "content_requirements": ["Preserve the scheduled follow-up purpose."],
        "visible_boundaries": [],
        "addressee_plan": ["current user"],
        "style_guidance": "brief",
        "selected_surface_intent": "answer the scheduled follow-up",
        "permitted_action_results": [],
    }


def _action_cognition_output() -> dict[str, Any]:
    """Build a cognition output that selects visible dialog through speak."""

    output = {
        "logical_stance": "CONFIRM",
        "character_intent": "PROVIDE",
        "text_surface_output_v2": _text_surface_output(),
        "action_specs": [_speak_action_spec()],
        "cognition_core_output": {
            "state_update": {"state_scope": "character"},
        },
        "cognition_state_committed": True,
    }
    return output


def _consolidation_result() -> dict[str, Any]:
    """Build the shared consolidator metadata shape used by worker tests."""

    result = {
        "consolidation_metadata": {
            "write_success": {
                "character_state": True,
                "relationship_insight": True,
                "user_memory_units": False,
                "relationship_state": True,
                "character_image": False,
                "cache_invalidation": True,
            },
            "cache_evicted_count": 1,
        },
    }
    return result


def test_worker_v2_result_requires_character_scope_and_completed_commit() -> None:
    """Worker delivery must follow a committed character-scoped V2 result."""

    payloads = {
        models.ARTIFACT_COGNITION_OUTPUT: {
            "cognition_core_output": {
                "state_update": {"state_scope": "user"},
            },
            "cognition_state_committed": True,
        },
    }

    with pytest.raises(StateContractError, match="character scope"):
        worker._validate_worker_v2_cognition_result(
            payloads,
            required=True,
        )


def _case_runner_with_candidate(
    case: dict[str, Any],
) -> dict[str, Any]:
    attempt = _action_attempt(
        case,
        status=models.ACTION_ATTEMPT_STATUS_CANDIDATE,
    )
    candidate = _action_candidate(attempt)
    payloads = {
        models.ARTIFACT_ACTION_ATTEMPT: attempt,
        models.ARTIFACT_ACTION_CANDIDATE: candidate,
    }
    return payloads


def _case_runner_with_tracking(
    case: dict[str, Any],
) -> dict[str, Any]:
    """Build action artifacts using real tracking duplicate logic."""

    trigger_record = tracking.build_trigger_record(case)
    existing_attempts = case.get("existing_attempts")
    if not isinstance(existing_attempts, list):
        existing_attempts = []
    action_attempt = tracking.build_action_attempt(
        case,
        trigger_record,
        [
            attempt
            for attempt in existing_attempts
            if isinstance(attempt, dict)
        ],
    )
    action_candidate = tracking.build_action_candidate(
        case,
        action_attempt,
        "Checking in now.",
    )
    payloads = {models.ARTIFACT_ACTION_ATTEMPT: action_attempt}
    if action_candidate is not None:
        payloads[models.ARTIFACT_ACTION_CANDIDATE] = action_candidate
    return payloads


@pytest.mark.asyncio
async def test_collect_scheduled_future_cognition_cases_projects_due_slots() -> None:
    """Due future-cognition slots become normal prompt-safe trigger cases."""

    now = datetime(2026, 5, 16, 10, 0, tzinfo=timezone.utc)
    calls: list[dict[str, Any]] = []

    async def list_due_runs(**kwargs: Any) -> list[dict[str, Any]]:
        calls.append(dict(kwargs))
        return [_future_cognition_run()]

    async def no_private_channel(**kwargs: Any) -> None:
        del kwargs
        return None

    cases = await sources.collect_scheduled_future_cognition_cases(
        now=now,
        character_profile={"name": "TestCharacter"},
        max_cases=3,
        list_due_calendar_runs_func=list_due_runs,
        get_latest_private_channel_func=no_private_channel,
    )

    assert calls == [
        {
            "current_timestamp_utc": now.isoformat(),
            "trigger_kinds": [calendar_models.TRIGGER_FUTURE_COGNITION],
            "max_attempts": 3,
            "limit": 3,
        }
    ]
    assert len(cases) == 1
    case = cases[0]
    assert case["case_name"] == models.CASE_SCHEDULED_FUTURE_COGNITION
    assert case["trigger_kind"] == models.TRIGGER_SCHEDULED_FUTURE_COGNITION
    assert case["case_id"].startswith(
        "scheduled_future_cognition_slot:"
    )
    assert case["source_calendar_run_id"] == "calendar_run_future_123"
    assert case["source_refs"][0]["source_kind"] == (
        "scheduled_future_cognition_slot"
    )
    assert case["source_refs"][0]["source_id"].startswith(
        "scheduled_future_cognition_slot:"
    )
    assert case["source_refs"][0]["summary"] == (
        "Re-check whether a natural pause appeared."
    )
    assert case["conversation_progress"]["continuation_objective"] == (
        "Re-check whether a natural pause appeared."
    )
    assert "context_summary" not in case["conversation_progress"]
    source_packet = projection.build_source_packet(case)
    rendered_packet = projection.render_source_packet_text(source_packet)
    serialized = json.dumps(source_packet, ensure_ascii=False).lower()
    serialized = f"{serialized}\n{rendered_packet.lower()}"
    for forbidden in (
        "action_attempt:future-123",
        "episode-123",
        "future-123",
        "calendar_run",
        "calendar_schedule",
        "handler_id",
        "credential",
        "mongodb",
        "collection",
        "episode_type",
        "include_result_as",
        "max_depth",
        "raw_channel",
        "schema_version",
    ):
        assert forbidden not in serialized


@pytest.mark.asyncio
async def test_collect_scheduled_future_cognition_cases_keeps_same_due_runs_distinct() -> None:
    """Same-time future-cognition slots need unique prompt-safe identities."""

    now = datetime(2026, 5, 16, 10, 0, tzinfo=timezone.utc)
    first_run = _future_cognition_run()
    second_run = _future_cognition_run()
    second_run["run_id"] = "calendar_run_future_456"
    second_run["schedule_id"] = "calendar_schedule_future_456"
    second_run["idempotency_key"] = "future_cognition:second:2026-05-16"
    second_run["payload"] = dict(first_run["payload"])
    second_run["payload"]["source_action_attempt_id"] = (
        "action_attempt:future-456"
    )

    async def list_due_runs(**kwargs: Any) -> list[dict[str, Any]]:
        del kwargs
        return [first_run, second_run]

    async def no_private_channel(**kwargs: Any) -> None:
        del kwargs
        return None

    cases = await sources.collect_scheduled_future_cognition_cases(
        now=now,
        character_profile={"name": "TestCharacter"},
        max_cases=3,
        list_due_calendar_runs_func=list_due_runs,
        get_latest_private_channel_func=no_private_channel,
    )

    assert len(cases) == 2
    case_ids = [case["case_id"] for case in cases]
    assert len(set(case_ids)) == 2
    assert case_ids[0].startswith("scheduled_future_cognition_slot:")
    assert case_ids[1].startswith("scheduled_future_cognition_slot:")
    source_ids = [case["source_refs"][0]["source_id"] for case in cases]
    assert len(set(source_ids)) == 2
    assert source_ids[0].startswith("scheduled_future_cognition_slot:")
    assert source_ids[1].startswith("scheduled_future_cognition_slot:")

    action_attempts = []
    for case in cases:
        trigger_record = tracking.build_trigger_record(case)
        action_attempt = tracking.build_action_attempt(
            case,
            trigger_record,
            existing_attempts=[],
        )
        action_attempts.append(action_attempt)
        source_packet = projection.build_source_packet(case)
        rendered_packet = projection.render_source_packet_text(source_packet)
        serialized = json.dumps(source_packet, ensure_ascii=False).lower()
        serialized = f"{serialized}\n{rendered_packet.lower()}"
        for forbidden in (
            "action_attempt:future",
            "calendar_run",
            "calendar_schedule",
        ):
            assert forbidden not in serialized

    idempotency_keys = {
        action_attempt["idempotency_key"]
        for action_attempt in action_attempts
    }
    assert len(idempotency_keys) == 2


@pytest.mark.asyncio
async def test_collect_scheduled_future_cognition_cases_preserves_source_scope() -> None:
    """Scheduled future cognition should keep trusted scope for RAG/context."""

    now = datetime(2026, 5, 16, 10, 0, tzinfo=timezone.utc)
    run = _future_cognition_run()
    run["source_scope"].update(
        {
            "source_platform": "qq",
            "source_channel_id": "54369546",
            "source_channel_type": "group",
            "source_user_id": "673225019",
            "source_platform_bot_id": "bot-001",
            "source_character_name": "TestCharacter",
        }
    )

    async def list_due_runs(**kwargs: Any) -> list[dict[str, Any]]:
        del kwargs
        return [run]

    async def no_private_channel(**kwargs: Any) -> None:
        del kwargs
        return None

    async def user_profile(global_user_id: str) -> dict[str, Any]:
        return {"global_user_id": global_user_id, "relationship_state": 500}

    cases = await sources.collect_scheduled_future_cognition_cases(
        now=now,
        character_profile={"name": "TestCharacter"},
        max_cases=1,
        list_due_calendar_runs_func=list_due_runs,
        get_latest_private_channel_func=no_private_channel,
        get_user_profile_func=user_profile,
    )

    assert cases[0]["target_scope"] == {
        "platform": "qq",
        "platform_channel_id": "54369546",
        "channel_type": "group",
        "user_id": "673225019",
        "display_name": "673225019",
    }
    assert cases[0]["user_profile"]["global_user_id"] == "673225019"
    assert cases[0]["platform_bot_id"] == "bot-001"


@pytest.mark.asyncio
async def test_scheduled_future_cognition_real_user_missing_profile_is_not_defaulted() -> None:
    """A real scheduled source user must not receive a placeholder profile."""

    now = datetime(2026, 5, 16, 10, 0, tzinfo=timezone.utc)
    run = _future_cognition_run()
    run["source_scope"].update(
        {
            "source_platform": "qq",
            "source_channel_id": "54369546",
            "source_channel_type": "private",
            "source_user_id": "673225019",
        }
    )

    async def list_due_runs(**kwargs: Any) -> list[dict[str, Any]]:
        del kwargs
        return [run]

    async def no_private_channel(**kwargs: Any) -> None:
        del kwargs
        return None

    async def missing_user_profile(global_user_id: str) -> dict[str, Any]:
        del global_user_id
        return {}

    cases = await sources.collect_scheduled_future_cognition_cases(
        now=now,
        character_profile={"name": "TestCharacter"},
        max_cases=1,
        list_due_calendar_runs_func=list_due_runs,
        get_latest_private_channel_func=no_private_channel,
        get_user_profile_func=missing_user_profile,
    )

    assert cases[0]["target_scope"]["user_id"] == "673225019"
    assert cases[0]["user_profile"] == {}


@pytest.mark.asyncio
async def test_scheduled_future_cognition_synthetic_user_stays_targetless() -> None:
    """A stale synthetic scheduled user id must not become a user target."""

    now = datetime(2026, 5, 16, 10, 0, tzinfo=timezone.utc)
    run = _future_cognition_run()
    run["source_scope"].update(
        {
            "source_platform": "qq",
            "source_channel_id": "54369546",
            "source_channel_type": "group",
            "source_platform_bot_id": "bot-001",
            "source_character_name": "TestCharacter",
        }
    )

    async def list_due_runs(**kwargs: Any) -> list[dict[str, Any]]:
        del kwargs
        return [run]

    async def no_private_channel(**kwargs: Any) -> None:
        del kwargs
        return None

    async def no_user_profile(global_user_id: str) -> None:
        raise AssertionError(
            f"synthetic user id must not be profiled: {global_user_id}"
        )

    cases = await sources.collect_scheduled_future_cognition_cases(
        now=now,
        character_profile={"name": "TestCharacter"},
        max_cases=1,
        list_due_calendar_runs_func=list_due_runs,
        get_latest_private_channel_func=no_private_channel,
        get_user_profile_func=no_user_profile,
    )

    assert cases[0]["target_scope"] == {
        "platform": "qq",
        "platform_channel_id": "54369546",
        "channel_type": "group",
        "user_id": None,
        "display_name": "group audience",
    }
    assert cases[0]["delivery_target"]["target_global_user_id"] is None
    assert cases[0]["delivery_target"]["source_global_user_id"] is None
    assert cases[0]["user_profile"] == {}


@pytest.mark.asyncio
async def test_scheduled_future_cognition_without_user_keeps_group_targetless() -> None:
    """A group-origin scheduled slot must not fabricate a user target."""

    now = datetime(2026, 5, 16, 10, 0, tzinfo=timezone.utc)
    run = _future_cognition_run()
    run["source_scope"].update(
        {
            "source_platform": "qq",
            "source_channel_id": "54369546",
            "source_channel_type": "group",
            "source_user_id": "",
            "source_platform_bot_id": "bot-001",
            "source_character_name": "TestCharacter",
        }
    )

    async def list_due_runs(**kwargs: Any) -> list[dict[str, Any]]:
        del kwargs
        return [run]

    async def no_private_channel(**kwargs: Any) -> None:
        del kwargs
        return None

    cases = await sources.collect_scheduled_future_cognition_cases(
        now=now,
        character_profile={"name": "TestCharacter"},
        max_cases=1,
        list_due_calendar_runs_func=list_due_runs,
        get_latest_private_channel_func=no_private_channel,
    )

    assert cases[0]["target_scope"] == {
        "platform": "qq",
        "platform_channel_id": "54369546",
        "channel_type": "group",
        "user_id": None,
        "display_name": "group audience",
    }
    assert cases[0]["delivery_target"]["target_global_user_id"] is None


@pytest.mark.asyncio
async def test_collect_commitment_due_cognition_cases_projects_calendar_runs() -> None:
    """Due commitment calendar runs should become normal commitment cases."""

    run = _commitment_due_run()
    unit = {
        "unit_id": "promise-001",
        "global_user_id": "673225019",
        "unit_type": "active_commitment",
        "status": "active",
        "fact": "A promised follow-up is due.",
        "subjective_appraisal": "The user may expect a check-in.",
        "relationship_signal": "Following through matters.",
        "due_at": "2026-05-13T00:00:00+00:00",
        "last_seen_at": "2026-05-12T23:55:00+00:00",
        "updated_at": "2026-05-12T23:55:00+00:00",
    }
    rows = [
        {
            "platform": "qq",
            "platform_channel_id": "673225019",
            "channel_type": "private",
            "role": "user",
            "global_user_id": "673225019",
            "display_name": "User",
            "body_text": "Please check back after the appointment.",
            "timestamp": "2026-05-12T23:50:00+00:00",
        }
    ]

    async def list_due_runs(**kwargs: Any) -> list[dict[str, Any]]:
        assert kwargs["trigger_kinds"] == [
            calendar_models.TRIGGER_COMMITMENT_DUE_COGNITION,
        ]
        assert kwargs["limit"] == 2
        return [run]

    async def read_memory_unit(unit_id: str) -> dict[str, Any]:
        assert unit_id == "promise-001"
        return unit

    async def get_history(**kwargs: Any) -> list[dict[str, Any]]:
        assert kwargs["global_user_id"] == "673225019"
        return rows

    async def get_profile(global_user_id: str) -> dict[str, Any]:
        assert global_user_id == "673225019"
        return {"relationship_state": 600, "display_name": "User"}

    async def no_private_channel(**kwargs: Any) -> None:
        del kwargs
        return None

    cases = await sources.collect_commitment_due_cognition_cases(
        now=datetime(2026, 5, 13, 0, 30, tzinfo=timezone.utc),
        character_profile={"name": "Character", "mood": "focused"},
        max_cases=2,
        list_due_calendar_runs_func=list_due_runs,
        memory_unit_reader_func=read_memory_unit,
        get_conversation_history_func=get_history,
        get_user_profile_func=get_profile,
        get_latest_private_channel_func=no_private_channel,
    )

    assert len(cases) == 1
    assert cases[0]["case_name"] == models.CASE_COMMITMENT_PAST_DUE
    assert cases[0]["trigger_kind"] == models.TRIGGER_ACTIVE_COMMITMENT_DUE_CHECK
    assert cases[0]["source_calendar_run_id"] == "calendar_run_commitment_123"
    assert cases[0]["source_refs"][0] == {
        "source_kind": "user_memory_unit",
        "source_id": "promise-001",
        "due_at": "2026-05-13T00:00:00+00:00",
        "summary": "A promised follow-up is due.",
    }


@pytest.mark.asyncio
async def test_collect_commitment_due_cognition_cases_projects_stale_run_skip(
) -> None:
    """Stale due runs should reach the worker as terminal skip work."""

    run = _commitment_due_run()
    stale_unit = {
        "unit_id": "promise-001",
        "global_user_id": "673225019",
        "unit_type": "active_commitment",
        "status": "active",
        "fact": "A promised follow-up was rescheduled.",
        "subjective_appraisal": "The old due slot is stale.",
        "relationship_signal": "Use only the current due time.",
        "due_at": "2026-05-14T00:00:00+00:00",
        "updated_at": "2026-05-13T00:10:00+00:00",
    }

    async def list_due_runs(**kwargs: Any) -> list[dict[str, Any]]:
        del kwargs
        return [run]

    async def read_memory_unit(unit_id: str) -> dict[str, Any]:
        assert unit_id == "promise-001"
        return stale_unit

    cases = await sources.collect_commitment_due_cognition_cases(
        now=datetime(2026, 5, 13, 0, 30, tzinfo=timezone.utc),
        character_profile={"name": "Character", "mood": "focused"},
        max_cases=2,
        list_due_calendar_runs_func=list_due_runs,
        memory_unit_reader_func=read_memory_unit,
        get_conversation_history_func=lambda **kwargs: [],
        get_user_profile_func=lambda global_user_id: {},
        get_latest_private_channel_func=lambda **kwargs: None,
    )

    assert cases == [
        {
            "case_name": models.CASE_COMMITMENT_DUPLICATE_TICK,
            "case_id": "commitment_due_skip:calendar_run_commitment_123",
            "trigger_kind": models.TRIGGER_ACTIVE_COMMITMENT_DUE_CHECK,
            "source_calendar_run_id": "calendar_run_commitment_123",
            "source_calendar_skip_reason": "stale_active_commitment_due_at",
            "cognition_source": {
                "source_kind": "scheduler_event",
                "source_id": "calendar_run_commitment_123",
                "occurred_at": "2026-05-13T00:00:00+00:00",
                "semantic_summary": (
                    "scheduled commitment was skipped: "
                    "stale_active_commitment_due_at"
                ),
            },
        }
    ]


@pytest.mark.asyncio
async def test_collect_commitment_due_cognition_cases_skips_unbuildable_case(
) -> None:
    """Valid due runs should not stay pending when context cannot build a case."""

    run = _commitment_due_run()
    unit = {
        "unit_id": "promise-001",
        "global_user_id": "673225019",
        "unit_type": "active_commitment",
        "status": "active",
        "fact": "A promised follow-up is due.",
        "subjective_appraisal": "The user may expect a check-in.",
        "relationship_signal": "Following through matters.",
        "due_at": "2026-05-13T00:00:00+00:00",
        "updated_at": "2026-05-12T23:55:00+00:00",
    }

    async def list_due_runs(**kwargs: Any) -> list[dict[str, Any]]:
        del kwargs
        return [run]

    async def read_memory_unit(unit_id: str) -> dict[str, Any]:
        assert unit_id == "promise-001"
        return unit

    cases = await sources.collect_commitment_due_cognition_cases(
        now=datetime(2026, 5, 13, 0, 30, tzinfo=timezone.utc),
        character_profile={"name": "Character", "mood": "focused"},
        max_cases=2,
        list_due_calendar_runs_func=list_due_runs,
        memory_unit_reader_func=read_memory_unit,
        get_conversation_history_func=lambda **kwargs: [],
        get_user_profile_func=lambda global_user_id: {},
        get_latest_private_channel_func=lambda **kwargs: None,
    )

    assert cases == [
        {
            "case_name": models.CASE_COMMITMENT_DUPLICATE_TICK,
            "case_id": "commitment_due_skip:calendar_run_commitment_123",
            "trigger_kind": models.TRIGGER_ACTIVE_COMMITMENT_DUE_CHECK,
            "source_calendar_run_id": "calendar_run_commitment_123",
            "source_calendar_skip_reason": (
                "active_commitment_case_unavailable"
            ),
            "cognition_source": {
                "source_kind": "scheduler_event",
                "source_id": "calendar_run_commitment_123",
                "occurred_at": "2026-05-13T00:00:00+00:00",
                "semantic_summary": (
                    "scheduled commitment was skipped: "
                    "active_commitment_case_unavailable"
                ),
            },
        }
    ]


@pytest.mark.asyncio
async def test_collect_self_cognition_cases_includes_future_slots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The shared collector should include due scheduled cognition slots."""

    async def no_commitment_due(**kwargs: Any) -> list[dict[str, Any]]:
        del kwargs
        return []

    async def future_cases(**kwargs: Any) -> list[dict[str, Any]]:
        del kwargs
        return [_future_cognition_case()]

    monkeypatch.setattr(
        sources,
        "collect_scheduled_future_cognition_cases",
        future_cases,
    )
    monkeypatch.setattr(
        sources,
        "collect_commitment_due_cognition_cases",
        no_commitment_due,
        raising=False,
    )

    cases = await sources.collect_self_cognition_cases(
        now=datetime(2026, 5, 16, 10, 0, tzinfo=timezone.utc),
        character_profile={"name": "TestCharacter"},
        max_cases=3,
    )

    assert [case["trigger_kind"] for case in cases] == [
        models.TRIGGER_SCHEDULED_FUTURE_COGNITION,
    ]


@pytest.mark.asyncio
async def test_collect_self_cognition_cases_includes_calendar_commitment_due_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The default collector should read due commitments from calendar runs."""

    async def no_scheduled(**kwargs: Any) -> list[dict[str, Any]]:
        del kwargs
        return []

    async def commitment_due_cases(**kwargs: Any) -> list[dict[str, Any]]:
        del kwargs
        case = _commitment_case_with_delivery_target()
        case["source_calendar_run_id"] = "calendar_run_commitment_123"
        return [case]

    async def active_commitments(**kwargs: Any) -> list[dict[str, Any]]:
        del kwargs
        raise AssertionError("default collector should not poll commitments")

    monkeypatch.setattr(
        sources,
        "collect_scheduled_future_cognition_cases",
        no_scheduled,
    )
    monkeypatch.setattr(
        sources,
        "collect_commitment_due_cognition_cases",
        commitment_due_cases,
        raising=False,
    )
    monkeypatch.setattr(
        sources,
        "collect_active_commitment_cases",
        active_commitments,
    )
    monkeypatch.setattr(
        sources,
        "is_self_cognition_sleep_period",
        lambda now: False,
    )

    cases = await sources.collect_self_cognition_cases(
        now=datetime(2026, 5, 13, 0, 30, tzinfo=timezone.utc),
        character_profile={"name": "TestCharacter"},
        max_cases=3,
    )

    assert [case["source_calendar_run_id"] for case in cases] == [
        "calendar_run_commitment_123",
    ]


@pytest.mark.asyncio
async def test_collect_self_cognition_cases_does_not_poll_active_commitments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Production active-commitment due checks are calendar-run driven."""

    async def no_scheduled(**kwargs: Any) -> list[dict[str, Any]]:
        del kwargs
        return []

    async def no_commitment_due(**kwargs: Any) -> list[dict[str, Any]]:
        del kwargs
        return []

    async def active_commitments(**kwargs: Any) -> list[dict[str, Any]]:
        del kwargs
        raise AssertionError("default collector should not poll commitments")

    monkeypatch.setattr(
        sources,
        "collect_scheduled_future_cognition_cases",
        no_scheduled,
    )
    monkeypatch.setattr(
        sources,
        "collect_commitment_due_cognition_cases",
        no_commitment_due,
        raising=False,
    )
    monkeypatch.setattr(
        sources,
        "collect_active_commitment_cases",
        active_commitments,
    )
    monkeypatch.setattr(
        sources,
        "is_self_cognition_sleep_period",
        lambda now: False,
    )

    cases = await sources.collect_self_cognition_cases(
        now=datetime(2026, 5, 13, 0, 30, tzinfo=timezone.utc),
        character_profile={"name": "TestCharacter"},
        max_cases=3,
    )

    assert cases == []


@pytest.mark.asyncio
async def test_collect_self_cognition_cases_skips_active_commitments_during_sleep(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The production collector should not trigger promises during sleep."""

    async def no_scheduled(**kwargs: Any) -> list[dict[str, Any]]:
        del kwargs
        return []

    async def active_commitments(**kwargs: Any) -> list[dict[str, Any]]:
        del kwargs
        raise AssertionError("active commitments should sleep")

    monkeypatch.setattr(
        sources,
        "collect_scheduled_future_cognition_cases",
        no_scheduled,
    )
    monkeypatch.setattr(
        sources,
        "collect_active_commitment_cases",
        active_commitments,
    )
    monkeypatch.setattr(
        sources,
        "is_self_cognition_sleep_period",
        lambda now: True,
    )

    cases = await sources.collect_self_cognition_cases(
        now=datetime(2026, 5, 12, 14, 30, tzinfo=timezone.utc),
        character_profile={"name": "TestCharacter"},
        max_cases=3,
    )

    assert cases == []


@pytest.mark.asyncio
async def test_collect_self_cognition_cases_keeps_scheduled_future_slots_during_sleep(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sleep should not suppress explicitly scheduled future cognition."""

    async def future_cases(**kwargs: Any) -> list[dict[str, Any]]:
        del kwargs
        return [_future_cognition_case()]

    async def active_commitments(**kwargs: Any) -> list[dict[str, Any]]:
        del kwargs
        raise AssertionError("active commitments should sleep")

    monkeypatch.setattr(
        sources,
        "collect_scheduled_future_cognition_cases",
        future_cases,
    )
    monkeypatch.setattr(
        sources,
        "collect_active_commitment_cases",
        active_commitments,
    )
    monkeypatch.setattr(
        sources,
        "is_self_cognition_sleep_period",
        lambda now: True,
    )

    cases = await sources.collect_self_cognition_cases(
        now=datetime(2026, 5, 12, 14, 30, tzinfo=timezone.utc),
        character_profile={"name": "TestCharacter"},
        max_cases=3,
    )

    assert [case["trigger_kind"] for case in cases] == [
        models.TRIGGER_SCHEDULED_FUTURE_COGNITION,
    ]


@pytest.mark.asyncio
async def test_worker_tick_marks_future_cognition_run_completed(
    tmp_path: Path,
) -> None:
    """A processed calendar cognition run should not stay pending forever."""

    del tmp_path
    claimed_run_ids: list[str] = []
    completed_run_ids: list[str] = []
    published_artifacts: list[dict[str, Any]] = []

    async def collect_cases(**kwargs: Any) -> list[dict[str, Any]]:
        del kwargs
        return [_future_cognition_case()]

    async def run_case(case: dict[str, Any]) -> dict[str, Any]:
        assert case["trigger_kind"] == models.TRIGGER_SCHEDULED_FUTURE_COGNITION
        return {
            models.ARTIFACT_TRIGGER_RECORD: {
                "trigger_id": "self_cognition_trigger:future-123",
                "trigger_kind": models.TRIGGER_SCHEDULED_FUTURE_COGNITION,
            },
            models.ARTIFACT_RUN_RECORD: {
                "run_id": "self_cognition_run:future-123",
                "selected_route": models.ROUTE_AUDIT_ONLY,
                "output_mode": "audit_only",
                "status": "completed",
                "budget": {
                    "rag_calls": 0,
                    "cognition_calls": 1,
                    "dialog_calls": 0,
                    "topic_limit": 3,
                },
            },
            models.ARTIFACT_COGNITION_OUTPUT: {
                "internal_monologue": "bounded self-cognition reason",
                "logical_stance": "inspect due promise",
                "character_intent": "decide whether to act",
            },
        }

    async def publish_latest_graph(artifact_payloads: dict[str, Any]) -> None:
        published_artifacts.append(artifact_payloads)

    async def claim_run(run_id: str, **kwargs: Any) -> bool:
        claimed_run_ids.append(run_id)
        assert kwargs["lease_owner"] == "self_cognition_worker"
        return True

    async def mark_completed(run_id: str, **kwargs: Any) -> bool:
        completed_run_ids.append(run_id)
        assert kwargs["lease_owner"] == "self_cognition_worker"
        assert kwargs["result"]["status"] == "self_cognition_processed"
        return True

    result = await worker.run_self_cognition_worker_tick(
        now=datetime(2026, 5, 16, 10, 0, tzinfo=timezone.utc),
        is_primary_interaction_busy=lambda: False,
        collect_cases_func=collect_cases,
        run_case_func=run_case,
        read_attempts_func=lambda **kwargs: [],
        record_attempt_func=lambda attempt: None,
        claim_calendar_run_func=claim_run,
        complete_calendar_run_func=mark_completed,
        latest_cognition_graph_publisher=publish_latest_graph,
        max_cases=3,
    )

    assert result.processed_count == 1
    assert claimed_run_ids == ["calendar_run_future_123"]
    assert completed_run_ids == ["calendar_run_future_123"]
    assert published_artifacts
    assert (
        published_artifacts[0][models.ARTIFACT_RUN_RECORD]["run_id"]
        == "self_cognition_run:future-123"
    )


@pytest.mark.asyncio
async def test_worker_tick_marks_commitment_due_run_completed(
    tmp_path: Path,
) -> None:
    """Processed commitment due calendar runs should be marked terminal."""

    del tmp_path
    claimed_run_ids: list[str] = []
    completed_run_ids: list[str] = []
    case = _commitment_case_with_delivery_target()
    case["source_calendar_run_id"] = "calendar_run_commitment_123"

    async def collect_cases(**kwargs: Any) -> list[dict[str, Any]]:
        del kwargs
        return [case]

    async def run_case(case_arg: dict[str, Any]) -> dict[str, Any]:
        assert case_arg["trigger_kind"] == (
            models.TRIGGER_ACTIVE_COMMITMENT_DUE_CHECK
        )
        return {}

    async def claim_run(run_id: str, **kwargs: Any) -> bool:
        claimed_run_ids.append(run_id)
        assert kwargs["trigger_kind"] == (
            calendar_models.TRIGGER_COMMITMENT_DUE_COGNITION
        )
        assert kwargs["lease_owner"] == "self_cognition_worker"
        return True

    async def mark_completed(run_id: str, **kwargs: Any) -> bool:
        completed_run_ids.append(run_id)
        assert kwargs["lease_owner"] == "self_cognition_worker"
        assert kwargs["result"]["status"] == "self_cognition_processed"
        return True

    result = await worker.run_self_cognition_worker_tick(
        now=datetime(2026, 5, 13, 0, 30, tzinfo=timezone.utc),
        is_primary_interaction_busy=lambda: False,
        collect_cases_func=collect_cases,
        run_case_func=run_case,
        read_attempts_func=lambda **kwargs: [],
        record_attempt_func=lambda attempt: None,
        claim_calendar_run_func=claim_run,
        complete_calendar_run_func=mark_completed,
        max_cases=3,
    )

    assert result.processed_count == 1
    assert claimed_run_ids == ["calendar_run_commitment_123"]
    assert completed_run_ids == ["calendar_run_commitment_123"]


@pytest.mark.asyncio
async def test_worker_tick_skips_stale_commitment_due_run() -> None:
    """Stale commitment due calendar runs should be terminal after claim."""

    claimed_run_ids: list[str] = []
    skipped_run_ids: list[str] = []
    case = {
        "case_name": models.CASE_COMMITMENT_DUPLICATE_TICK,
        "case_id": "commitment_due_skip:calendar_run_commitment_123",
        "trigger_kind": models.TRIGGER_ACTIVE_COMMITMENT_DUE_CHECK,
        "source_calendar_run_id": "calendar_run_commitment_123",
        "source_calendar_skip_reason": "stale_active_commitment_due_at",
    }

    async def collect_cases(**kwargs: Any) -> list[dict[str, Any]]:
        del kwargs
        return [case]

    async def claim_run(run_id: str, **kwargs: Any) -> bool:
        claimed_run_ids.append(run_id)
        assert kwargs["trigger_kind"] == (
            calendar_models.TRIGGER_COMMITMENT_DUE_COGNITION
        )
        return True

    async def skip_run(run_id: str, **kwargs: Any) -> bool:
        skipped_run_ids.append(run_id)
        assert kwargs["reason"] == "stale_active_commitment_due_at"
        assert kwargs["lease_owner"] == "self_cognition_worker"
        return True

    run_case = AsyncMock()
    result = await worker.run_self_cognition_worker_tick(
        now=datetime(2026, 5, 13, 0, 30, tzinfo=timezone.utc),
        is_primary_interaction_busy=lambda: False,
        collect_cases_func=collect_cases,
        run_case_func=run_case,
        read_attempts_func=lambda **kwargs: [],
        record_attempt_func=lambda attempt: None,
        claim_calendar_run_func=claim_run,
        skip_calendar_run_func=skip_run,
        max_cases=3,
    )

    assert result.processed_count == 0
    assert result.skipped_count == 1
    assert claimed_run_ids == ["calendar_run_commitment_123"]
    assert skipped_run_ids == ["calendar_run_commitment_123"]
    run_case.assert_not_called()


@pytest.mark.asyncio
async def test_worker_tick_skips_future_cognition_slot_when_claim_fails(
    tmp_path: Path,
) -> None:
    """A due future-cognition slot should run only after an atomic claim."""

    processed_cases: list[dict[str, Any]] = []
    completed_run_ids: list[str] = []

    async def collect_cases(**kwargs: Any) -> list[dict[str, Any]]:
        del kwargs
        return [_future_cognition_case()]

    async def run_case(case: dict[str, Any]) -> dict[str, Any]:
        processed_cases.append(case)
        return {}

    async def mark_completed(run_id: str, **kwargs: Any) -> bool:
        del kwargs
        completed_run_ids.append(run_id)
        return True

    result = await worker.run_self_cognition_worker_tick(
        now=datetime(2026, 5, 16, 10, 0, tzinfo=timezone.utc),
        is_primary_interaction_busy=lambda: False,
        collect_cases_func=collect_cases,
        run_case_func=run_case,
        read_attempts_func=lambda **kwargs: [],
        record_attempt_func=lambda attempt: None,
        complete_calendar_run_func=mark_completed,
        claim_calendar_run_func=lambda run_id, **kwargs: False,
        max_cases=3,
    )

    assert result.processed_count == 0
    assert result.skipped_count == 1
    assert processed_cases == []
    assert completed_run_ids == []


@pytest.mark.asyncio
async def test_worker_tick_marks_state_contract_error_calendar_run_failed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Claimed source runs should be terminal when case contracts fail."""

    del tmp_path
    record_runtime_error_event = AsyncMock()
    monkeypatch.setattr(
        worker.event_logging,
        "record_runtime_error_event",
        record_runtime_error_event,
    )
    case = _future_cognition_case()
    failed_runs: list[dict[str, Any]] = []

    async def collect_cases(**kwargs: Any) -> list[dict[str, Any]]:
        del kwargs
        return [case]

    async def build_artifacts(next_case: dict[str, Any]) -> dict[str, Any]:
        assert next_case["source_calendar_run_id"] == "calendar_run_future_123"
        raise StateContractError(
            "usage_mode=self_cognition_action_candidate_render "
            "missing action_specs.speak"
        )

    async def claim_run(run_id: str, **kwargs: Any) -> dict[str, Any]:
        assert run_id == "calendar_run_future_123"
        assert kwargs["lease_owner"] == "self_cognition_worker"
        return {
            "run_id": run_id,
            "attempt_count": 3,
            "max_attempts": 3,
        }

    async def fail_run(run_id: str, **kwargs: Any) -> bool:
        failed_runs.append({
            "run_id": run_id,
            **kwargs,
        })
        return True

    result = await worker.run_self_cognition_worker_tick(
        now=datetime(2026, 5, 16, 10, 0, tzinfo=timezone.utc),
        is_primary_interaction_busy=lambda: False,
        collect_cases_func=collect_cases,
        run_case_func=build_artifacts,
        read_attempts_func=lambda **kwargs: [],
        record_attempt_func=lambda attempt: None,
        claim_calendar_run_func=claim_run,
        fail_calendar_run_func=fail_run,
        max_cases=3,
    )

    assert result.processed_count == 0
    assert result.failed_count == 1
    assert len(failed_runs) == 1
    failure = failed_runs[0]
    assert failure["run_id"] == "calendar_run_future_123"
    assert failure["lease_owner"] == "self_cognition_worker"
    assert failure["storage_timestamp_utc"] == (
        "2026-05-16T10:00:00+00:00"
    )
    assert failure["retryable"] is False
    assert "action_specs.speak" in failure["error"]
    record_runtime_error_event.assert_awaited_once()


@pytest.mark.asyncio
async def test_worker_tick_marks_unexpected_calendar_case_error_failed(
    tmp_path: Path,
) -> None:
    """Unexpected case crashes should release the source calendar lease."""

    del tmp_path
    case = _future_cognition_case()
    failed_runs: list[dict[str, Any]] = []

    async def collect_cases(**kwargs: Any) -> list[dict[str, Any]]:
        del kwargs
        return [case]

    async def build_artifacts(next_case: dict[str, Any]) -> dict[str, Any]:
        assert next_case["source_calendar_run_id"] == "calendar_run_future_123"
        raise RuntimeError("case runner crashed")

    async def claim_run(run_id: str, **kwargs: Any) -> dict[str, Any]:
        assert kwargs["lease_owner"] == "self_cognition_worker"
        return {
            "run_id": run_id,
            "attempt_count": 3,
            "max_attempts": 3,
        }

    async def fail_run(run_id: str, **kwargs: Any) -> bool:
        failed_runs.append({
            "run_id": run_id,
            **kwargs,
        })
        return True

    result = await worker.run_self_cognition_worker_tick(
        now=datetime(2026, 5, 16, 10, 0, tzinfo=timezone.utc),
        is_primary_interaction_busy=lambda: False,
        collect_cases_func=collect_cases,
        run_case_func=build_artifacts,
        read_attempts_func=lambda **kwargs: [],
        record_attempt_func=lambda attempt: None,
        claim_calendar_run_func=claim_run,
        fail_calendar_run_func=fail_run,
        max_cases=3,
    )

    assert result.processed_count == 0
    assert result.failed_count == 1
    assert len(failed_runs) == 1
    failure = failed_runs[0]
    assert failure["run_id"] == "calendar_run_future_123"
    assert failure["lease_owner"] == "self_cognition_worker"
    assert failure["retryable"] is False
    assert "case runner crashed" in failure["error"]


@pytest.mark.asyncio
async def test_worker_selected_speak_dispatches_to_private_channel(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Selected production speak should dispatch to a known private channel."""

    _patch_dispatcher_persistence(monkeypatch)
    adapter = _FakeMessagingAdapter()
    recorded_attempts: list[dict[str, Any]] = []

    async def collect_cases(**kwargs: Any) -> list[dict[str, Any]]:
        del kwargs
        return [_commitment_case_with_delivery_target()]

    async def run_case(case: dict[str, Any]) -> dict[str, Any]:
        return _selected_speak_artifacts(case)

    async def record_attempt(attempt: dict[str, Any]) -> None:
        recorded_attempts.append(dict(attempt))

    result = await worker.run_self_cognition_worker_tick(
        now=datetime(2026, 5, 17, 5, 57, tzinfo=timezone.utc),
        is_primary_interaction_busy=lambda: False,
        collect_cases_func=collect_cases,
        run_case_func=run_case,
        read_attempts_func=lambda **kwargs: [],
        record_attempt_func=record_attempt,
        adapter_registry_provider=lambda: _adapter_registry(adapter),
        max_cases=1,
    )

    assert result.processed_count == 1
    assert adapter.calls[0]["channel_id"] == "dm-1"
    assert adapter.calls[0]["channel_type"] == "private"
    assert recorded_attempts[-1]["status"] == models.ACTION_ATTEMPT_STATUS_SENT
    self_kwargs = worker.event_logging.record_self_cognition_event.await_args.kwargs
    assert self_kwargs["dispatch_status"] == "sent"


@pytest.mark.asyncio
async def test_worker_selected_speak_dispatches_to_bound_group_source_channel(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Selected group speak should dispatch to the bound source group."""

    _patch_dispatcher_persistence(monkeypatch)
    adapter = _FakeMessagingAdapter()
    source_group_target = _delivery_target(
        channel_id="group-1",
        channel_type="group",
        source_kind="self_cognition_source_channel",
        fallback_reason="",
    )

    async def collect_cases(**kwargs: Any) -> list[dict[str, Any]]:
        del kwargs
        return [
            _commitment_case_with_delivery_target(
                delivery_target=source_group_target,
            )
        ]

    async def run_case(case: dict[str, Any]) -> dict[str, Any]:
        return _selected_speak_artifacts(case)

    result = await worker.run_self_cognition_worker_tick(
        now=datetime(2026, 5, 17, 5, 57, tzinfo=timezone.utc),
        is_primary_interaction_busy=lambda: False,
        collect_cases_func=collect_cases,
        run_case_func=run_case,
        read_attempts_func=lambda **kwargs: [],
        record_attempt_func=lambda attempt: None,
        adapter_registry_provider=lambda: _adapter_registry(adapter),
        max_cases=1,
    )

    assert result.processed_count == 1
    assert adapter.calls[0]["channel_id"] == "group-1"
    assert adapter.calls[0]["channel_type"] == "group"


@pytest.mark.asyncio
async def test_worker_channel_capability_failure_blocks_before_history_write(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Unavailable source channels should fail before write-ahead persistence."""

    saved_documents: list[dict[str, Any]] = []
    recorded_attempts: list[dict[str, Any]] = []
    adapter = _FakeMessagingAdapter(can_send=False)

    async def save_conversation(document: dict[str, Any]) -> str:
        saved_documents.append(dict(document))
        return "conversation-row-1"

    async def ensure_character_identity(**kwargs: Any) -> str:
        del kwargs
        return "character-global"

    monkeypatch.setattr(
        handlers_module,
        "save_conversation",
        save_conversation,
    )
    monkeypatch.setattr(
        handlers_module,
        "ensure_character_identity",
        ensure_character_identity,
    )

    async def collect_cases(**kwargs: Any) -> list[dict[str, Any]]:
        del kwargs
        return [_commitment_case_with_delivery_target()]

    async def run_case(case: dict[str, Any]) -> dict[str, Any]:
        return _selected_speak_artifacts(case)

    async def record_attempt(attempt: dict[str, Any]) -> None:
        recorded_attempts.append(dict(attempt))

    result = await worker.run_self_cognition_worker_tick(
        now=datetime(2026, 5, 17, 5, 57, tzinfo=timezone.utc),
        is_primary_interaction_busy=lambda: False,
        collect_cases_func=collect_cases,
        run_case_func=run_case,
        read_attempts_func=lambda **kwargs: [],
        record_attempt_func=record_attempt,
        adapter_registry_provider=lambda: _adapter_registry(adapter),
        max_cases=1,
    )

    assert result.processed_count == 1
    assert adapter.calls == []
    assert saved_documents == []
    assert recorded_attempts[-1]["status"] == (
        models.ACTION_ATTEMPT_STATUS_DELIVERY_FAILED
    )
    assert recorded_attempts[-1]["failure_reason"] == (
        "adapter_channel_unavailable"
    )


@pytest.mark.asyncio
async def test_worker_missing_delivery_target_blocks_before_dialog(
    tmp_path: Path,
) -> None:
    """Production cases without target binding should stop before runner work."""

    run_case = AsyncMock()

    async def collect_cases(**kwargs: Any) -> list[dict[str, Any]]:
        del kwargs
        return [_commitment_case()]

    result = await worker.run_self_cognition_worker_tick(
        now=datetime(2026, 5, 17, 5, 57, tzinfo=timezone.utc),
        is_primary_interaction_busy=lambda: False,
        collect_cases_func=collect_cases,
        run_case_func=run_case,
        read_attempts_func=lambda **kwargs: [],
        record_attempt_func=lambda attempt: None,
        adapter_registry_provider=lambda: _adapter_registry(
            _FakeMessagingAdapter(),
        ),
        max_cases=1,
    )

    assert result.processed_count == 0
    assert result.skipped_count == 1
    run_case.assert_not_called()
    self_kwargs = worker.event_logging.record_self_cognition_event.await_args.kwargs
    assert self_kwargs["dispatch_status"] == "target_binding_failed"


@pytest.mark.asyncio
async def test_worker_missing_delivery_target_blocks_without_adapter_provider(
    tmp_path: Path,
) -> None:
    """Missing production target should fail closed even without adapters."""

    run_case = AsyncMock()
    record_attempt = AsyncMock()

    async def collect_cases(**kwargs: Any) -> list[dict[str, Any]]:
        del kwargs
        return [_commitment_case()]

    result = await worker.run_self_cognition_worker_tick(
        now=datetime(2026, 5, 17, 5, 57, tzinfo=timezone.utc),
        is_primary_interaction_busy=lambda: False,
        collect_cases_func=collect_cases,
        run_case_func=run_case,
        read_attempts_func=lambda **kwargs: [],
        record_attempt_func=record_attempt,
        max_cases=1,
    )

    assert result.processed_count == 0
    assert result.skipped_count == 1
    run_case.assert_not_called()
    record_attempt.assert_not_awaited()
    self_kwargs = worker.event_logging.record_self_cognition_event.await_args.kwargs
    assert self_kwargs["dispatch_status"] == "target_binding_failed"


@pytest.mark.asyncio
async def test_worker_records_target_binding_failed_and_skips_calendar_run(
    tmp_path: Path,
) -> None:
    """Invalid scheduled source rows should be recorded and marked terminal."""

    del tmp_path
    skipped_run_ids: list[str] = []
    case = _target_binding_failed_case()
    case["trigger_kind"] = models.TRIGGER_SCHEDULED_FUTURE_COGNITION
    case["source_calendar_run_id"] = "calendar_run_future_1"

    async def collect_cases(**kwargs: Any) -> list[dict[str, Any]]:
        del kwargs
        return [case]

    async def skip_run(run_id: str, **kwargs: Any) -> bool:
        skipped_run_ids.append(run_id)
        assert kwargs["lease_owner"] == "self_cognition_worker"
        assert kwargs["reason"] == "target_binding_failed"
        return True

    result = await worker.run_self_cognition_worker_tick(
        now=datetime(2026, 5, 17, 5, 57, tzinfo=timezone.utc),
        is_primary_interaction_busy=lambda: False,
        collect_cases_func=collect_cases,
        run_case_func=AsyncMock(),
        read_attempts_func=lambda **kwargs: [],
        record_attempt_func=lambda attempt: None,
        skip_calendar_run_func=skip_run,
        claim_calendar_run_func=lambda run_id, **kwargs: True,
        adapter_registry_provider=lambda: _adapter_registry(
            _FakeMessagingAdapter(),
        ),
        max_cases=1,
    )

    assert result.processed_count == 0
    assert result.skipped_count == 1
    assert skipped_run_ids == ["calendar_run_future_1"]
    self_kwargs = worker.event_logging.record_self_cognition_event.await_args.kwargs
    assert self_kwargs["status"] == "target_binding_failed"
    assert self_kwargs["dispatch_status"] == "target_binding_failed"


@pytest.mark.asyncio
async def test_worker_selected_speak_never_records_not_requested(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Selected speak should record a terminal delivery status."""

    _patch_dispatcher_persistence(monkeypatch)
    adapter = _FakeMessagingAdapter()

    async def collect_cases(**kwargs: Any) -> list[dict[str, Any]]:
        del kwargs
        return [_commitment_case_with_delivery_target()]

    async def run_case(case: dict[str, Any]) -> dict[str, Any]:
        return _selected_speak_artifacts(case)

    await worker.run_self_cognition_worker_tick(
        now=datetime(2026, 5, 17, 5, 57, tzinfo=timezone.utc),
        is_primary_interaction_busy=lambda: False,
        collect_cases_func=collect_cases,
        run_case_func=run_case,
        read_attempts_func=lambda **kwargs: [],
        record_attempt_func=lambda attempt: None,
        adapter_registry_provider=lambda: _adapter_registry(adapter),
        max_cases=1,
    )

    self_kwargs = worker.event_logging.record_self_cognition_event.await_args.kwargs
    assert self_kwargs["dispatch_status"] != "not_requested"


@pytest.mark.asyncio
async def test_worker_no_speak_does_not_dispatch(
    tmp_path: Path,
) -> None:
    """Non-speak artifact output should not call the adapter."""

    adapter = _FakeMessagingAdapter()

    async def collect_cases(**kwargs: Any) -> list[dict[str, Any]]:
        del kwargs
        return [_commitment_case_with_delivery_target()]

    async def run_case(case: dict[str, Any]) -> dict[str, Any]:
        del case
        return {}

    result = await worker.run_self_cognition_worker_tick(
        now=datetime(2026, 5, 17, 5, 57, tzinfo=timezone.utc),
        is_primary_interaction_busy=lambda: False,
        collect_cases_func=collect_cases,
        run_case_func=run_case,
        read_attempts_func=lambda **kwargs: [],
        record_attempt_func=lambda attempt: None,
        adapter_registry_provider=lambda: _adapter_registry(adapter),
        max_cases=1,
    )

    assert result.processed_count == 1
    assert adapter.calls == []


@pytest.mark.asyncio
async def test_worker_adapter_failure_marks_delivery_failed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Adapter send failures should persist a delivery_failed attempt."""

    _patch_dispatcher_persistence(monkeypatch)
    adapter = _FakeMessagingAdapter(fail=True)
    recorded_attempts: list[dict[str, Any]] = []

    async def collect_cases(**kwargs: Any) -> list[dict[str, Any]]:
        del kwargs
        return [_commitment_case_with_delivery_target()]

    async def run_case(case: dict[str, Any]) -> dict[str, Any]:
        return _selected_speak_artifacts(case)

    async def record_attempt(attempt: dict[str, Any]) -> None:
        recorded_attempts.append(dict(attempt))

    await worker.run_self_cognition_worker_tick(
        now=datetime(2026, 5, 17, 5, 57, tzinfo=timezone.utc),
        is_primary_interaction_busy=lambda: False,
        collect_cases_func=collect_cases,
        run_case_func=run_case,
        read_attempts_func=lambda **kwargs: [],
        record_attempt_func=record_attempt,
        adapter_registry_provider=lambda: _adapter_registry(adapter),
        max_cases=1,
    )

    assert recorded_attempts[-1]["status"] == (
        models.ACTION_ATTEMPT_STATUS_DELIVERY_FAILED
    )
    self_kwargs = worker.event_logging.record_self_cognition_event.await_args.kwargs
    assert self_kwargs["dispatch_status"] == "delivery_failed"


@pytest.mark.asyncio
async def test_worker_empty_dialog_text_marks_delivery_failed(
    tmp_path: Path,
) -> None:
    """Selected speak with empty rendered text must not persist candidate."""

    adapter = _FakeMessagingAdapter()
    recorded_attempts: list[dict[str, Any]] = []

    async def collect_cases(**kwargs: Any) -> list[dict[str, Any]]:
        del kwargs
        return [_commitment_case_with_delivery_target()]

    async def run_case(case: dict[str, Any]) -> dict[str, Any]:
        return _selected_speak_artifacts(case, text="")

    async def record_attempt(attempt: dict[str, Any]) -> None:
        recorded_attempts.append(dict(attempt))

    await worker.run_self_cognition_worker_tick(
        now=datetime(2026, 5, 17, 5, 57, tzinfo=timezone.utc),
        is_primary_interaction_busy=lambda: False,
        collect_cases_func=collect_cases,
        run_case_func=run_case,
        read_attempts_func=lambda **kwargs: [],
        record_attempt_func=record_attempt,
        adapter_registry_provider=lambda: _adapter_registry(adapter),
        max_cases=1,
    )

    assert adapter.calls == []
    assert recorded_attempts[-1]["status"] == (
        models.ACTION_ATTEMPT_STATUS_DELIVERY_FAILED
    )
    assert recorded_attempts[-1]["failure_reason"] == "empty_text"
    self_kwargs = worker.event_logging.record_self_cognition_event.await_args.kwargs
    assert self_kwargs["dispatch_status"] == "delivery_failed"


@pytest.mark.asyncio
async def test_worker_duplicate_suppression_marks_duplicate_suppressed(
    tmp_path: Path,
) -> None:
    """Duplicate suppression should not call adapter delivery."""

    adapter = _FakeMessagingAdapter()
    recorded_attempts: list[dict[str, Any]] = []

    async def collect_cases(**kwargs: Any) -> list[dict[str, Any]]:
        del kwargs
        return [_commitment_case_with_delivery_target()]

    async def run_case(case: dict[str, Any]) -> dict[str, Any]:
        return _selected_speak_artifacts(
            case,
            attempt_status=models.ACTION_ATTEMPT_STATUS_DUPLICATE,
        )

    async def record_attempt(attempt: dict[str, Any]) -> None:
        recorded_attempts.append(dict(attempt))

    await worker.run_self_cognition_worker_tick(
        now=datetime(2026, 5, 17, 5, 57, tzinfo=timezone.utc),
        is_primary_interaction_busy=lambda: False,
        collect_cases_func=collect_cases,
        run_case_func=run_case,
        read_attempts_func=lambda **kwargs: [],
        record_attempt_func=record_attempt,
        adapter_registry_provider=lambda: _adapter_registry(adapter),
        max_cases=1,
    )

    assert adapter.calls == []
    assert recorded_attempts[-1]["status"] == (
        models.ACTION_ATTEMPT_STATUS_DUPLICATE
    )
    self_kwargs = worker.event_logging.record_self_cognition_event.await_args.kwargs
    assert self_kwargs["dispatch_status"] == "duplicate_suppressed"


@pytest.mark.asyncio
async def test_worker_tick_records_state_contract_error_without_tick_failure(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """One malformed case should be recorded without failing the whole tick."""

    record_runtime_error_event = AsyncMock()
    record_worker_event = AsyncMock()
    monkeypatch.setattr(
        worker.event_logging,
        "record_runtime_error_event",
        record_runtime_error_event,
    )
    monkeypatch.setattr(
        worker.event_logging,
        "record_worker_event",
        record_worker_event,
    )

    async def collect_cases(*, now: datetime, max_cases: int) -> list[dict[str, Any]]:
        del now, max_cases
        return [_commitment_case_with_delivery_target()]

    async def read_attempts(*, limit: int) -> list[dict[str, Any]]:
        assert limit > 0
        return []

    async def build_artifacts(
        next_case: dict[str, Any],
    ) -> dict[str, Any]:
        del next_case
        raise StateContractError(
            "usage_mode=self_cognition_action_candidate_render "
            "missing action_specs.speak"
        )

    result = await worker.run_self_cognition_worker_tick(
        now=datetime(2026, 5, 13, tzinfo=timezone.utc),
        is_primary_interaction_busy=lambda: False,
        collect_cases_func=collect_cases,
        run_case_func=build_artifacts,
        read_attempts_func=read_attempts,
        max_cases=3,
    )

    assert result.processed_count == 0
    assert result.failed_count == 1
    record_runtime_error_event.assert_awaited_once()
    runtime_kwargs = record_runtime_error_event.await_args.kwargs
    assert runtime_kwargs["component"] == "self_cognition.worker"
    assert runtime_kwargs["error_class"] == "StateContractError"
    assert "action_specs.speak" in runtime_kwargs["error_preview"]
    assert runtime_kwargs["stack_fingerprint"] == (
        "self_cognition_case_state_contract"
    )
    assert runtime_kwargs["top_frame_module"] == worker.__name__
    assert runtime_kwargs["recovered"] is True
    record_worker_event.assert_awaited_once()
    worker_kwargs = record_worker_event.await_args.kwargs
    assert worker_kwargs["status"] == "failed"
    assert worker_kwargs["processed_count"] == 0
    assert worker_kwargs["failed_count"] == 1


class _AsyncCursor:
    def __init__(self, docs: list[dict[str, Any]]) -> None:
        self._docs = iter(docs)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            row = next(self._docs)
        except StopIteration as exc:
            raise StopAsyncIteration from exc
        return row


class _FakeUserMemoryUnitsCollection:
    def __init__(self) -> None:
        self.pipeline: list[dict[str, Any]] = []

    def aggregate(self, pipeline: list[dict[str, Any]]):
        self.pipeline = pipeline
        cursor = _AsyncCursor([{"unit_id": "promise-001"}])
        return cursor


@pytest.mark.asyncio
async def test_worker_default_path_requests_production_consolidation_without_files(
    monkeypatch,
    tmp_path,
) -> None:
    """Default worker runs should request consolidation and stay in memory."""

    case = _commitment_case_with_delivery_target()
    captured_kwargs: dict[str, Any] = {}

    async def collect_cases(*, now: datetime, max_cases: int) -> list[dict[str, Any]]:
        del now, max_cases
        return [case]

    async def read_attempts(*, limit: int) -> list[dict[str, Any]]:
        assert limit > 0
        return []

    async def build_artifacts(
        next_case: dict[str, Any],
        **kwargs: Any,
    ) -> dict[str, Any]:
        captured_kwargs.update(kwargs)
        trigger_record = tracking.build_trigger_record(next_case)
        run_record = tracking.build_run_record(
            next_case,
            trigger_record,
            models.ROUTE_AUDIT_ONLY,
            {
                "rag_calls": 0,
                "cognition_calls": 1,
                "dialog_calls": 1,
                "topic_limit": models.TOPIC_LIMIT,
            },
        )
        payloads = {
            models.ARTIFACT_TRIGGER_RECORD: trigger_record,
            models.ARTIFACT_RUN_RECORD: run_record,
            models.ARTIFACT_CONSOLIDATION_OUTCOME: {
                "consolidation_called": True,
                "write_success": {"character_state": True},
                "scheduled_event_count": 0,
                "cache_evicted_count": 0,
                "origin_trigger_source": "internal_thought",
                "origin_episode_id": "self_cognition:tracking:test",
            },
        }
        return payloads

    monkeypatch.setattr(
        worker.runner,
        "build_self_cognition_case_artifacts_async",
        build_artifacts,
    )

    result = await worker.run_self_cognition_worker_tick(
        now=datetime(2026, 5, 13, tzinfo=timezone.utc),
        is_primary_interaction_busy=lambda: False,
        collect_cases_func=collect_cases,
        read_attempts_func=read_attempts,
        max_cases=3,
    )

    assert result.processed_count == 1
    assert captured_kwargs["apply_consolidation"] is True
    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_worker_default_path_applies_consolidation_without_dispatch_or_files(
    monkeypatch,
    tmp_path,
) -> None:
    """Internal-only cognition should consolidate without outward delivery."""

    case = _commitment_case_with_delivery_target()
    captured_consolidation_state: dict[str, Any] = {}

    async def collect_cases(*, now: datetime, max_cases: int) -> list[dict[str, Any]]:
        del now, max_cases
        return [case]

    async def read_attempts(*, limit: int) -> list[dict[str, Any]]:
        assert limit > 0
        return []

    async def cognition_client(state: dict[str, Any]) -> dict[str, Any]:
        assert state["cognitive_episode"]["trigger_source"] == "scheduled_recall"
        return _progress_cognition_output()

    async def dialog_client(state: dict[str, Any]) -> dict[str, Any]:
        del state
        raise AssertionError("internal-only consolidation should not call dialog")

    async def consolidation_client(state: dict[str, Any]) -> dict[str, Any]:
        captured_consolidation_state.update(state)
        return _consolidation_result()

    monkeypatch.setattr(worker.runner, "_default_cognition_client", cognition_client)
    monkeypatch.setattr(worker.runner, "_default_dialog_client", dialog_client)
    monkeypatch.setattr(
        worker.runner,
        "_default_consolidation_client",
        consolidation_client,
    )

    result = await worker.run_self_cognition_worker_tick(
        now=datetime(2026, 5, 13, tzinfo=timezone.utc),
        is_primary_interaction_busy=lambda: False,
        collect_cases_func=collect_cases,
        read_attempts_func=read_attempts,
        max_cases=3,
    )

    assert result.processed_count == 1
    assert captured_consolidation_state["cognitive_episode"][
        "trigger_source"
    ] == "scheduled_recall"
    assert captured_consolidation_state["final_dialog"] == []
    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_worker_default_path_records_action_without_dispatch(
    monkeypatch,
    tmp_path,
) -> None:
    """Selected speak without a registry should persist delivery_failed."""

    case = _commitment_case_with_delivery_target()
    recorded_attempts: list[dict[str, Any]] = []
    captured_consolidation_state: dict[str, Any] = {}

    async def collect_cases(*, now: datetime, max_cases: int) -> list[dict[str, Any]]:
        del now, max_cases
        return [case]

    async def read_attempts(*, limit: int) -> list[dict[str, Any]]:
        assert limit > 0
        return list(recorded_attempts)

    async def record_attempt(attempt: dict[str, Any]) -> None:
        recorded_attempts.append(dict(attempt))

    async def cognition_client(state: dict[str, Any]) -> dict[str, Any]:
        assert state["cognitive_episode"]["trigger_source"] == "scheduled_recall"
        return _action_cognition_output()

    async def dialog_client(state: dict[str, Any]) -> dict[str, Any]:
        assert state["should_respond"] is False
        assert state["dialog_usage_mode"] == "self_cognition_action_candidate_render"
        return {"final_dialog": ["Checking in now."]}

    async def consolidation_client(state: dict[str, Any]) -> dict[str, Any]:
        captured_consolidation_state.update(state)
        return _consolidation_result()

    monkeypatch.setattr(worker.runner, "_default_cognition_client", cognition_client)
    monkeypatch.setattr(worker.runner, "_default_dialog_client", dialog_client)
    monkeypatch.setattr(
        worker.runner,
        "_default_consolidation_client",
        consolidation_client,
    )

    result = await worker.run_self_cognition_worker_tick(
        now=datetime(2026, 5, 13, tzinfo=timezone.utc),
        is_primary_interaction_busy=lambda: False,
        collect_cases_func=collect_cases,
        read_attempts_func=read_attempts,
        record_attempt_func=record_attempt,
        max_cases=3,
    )

    assert result.processed_count == 1
    assert recorded_attempts[0]["status"] == (
        models.ACTION_ATTEMPT_STATUS_DELIVERY_FAILED
    )
    assert recorded_attempts[0]["dispatch_status"] == "delivery_failed"
    assert recorded_attempts[0]["failure_reason"] == (
        "adapter_registry_unavailable"
    )
    assert captured_consolidation_state["final_dialog"] == ["Checking in now."]
    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_worker_tick_loads_prior_attempts_before_running_case(
    tmp_path,
) -> None:
    """Prior persisted attempts should enter the next case run."""

    case = _commitment_case_with_delivery_target()
    prior_attempt = _action_attempt(
        case,
        status=models.ACTION_ATTEMPT_STATUS_SCHEDULED,
    )
    captured_case: dict[str, Any] = {}

    async def collect_cases(*, now: datetime, max_cases: int) -> list[dict[str, Any]]:
        assert max_cases == 3
        return [case]

    async def read_attempts(*, limit: int) -> list[dict[str, Any]]:
        assert limit > 0
        return [prior_attempt]

    async def run_case(next_case: dict[str, Any]) -> dict[str, Any]:
        captured_case.update(next_case)
        return {}

    result = await worker.run_self_cognition_worker_tick(
        now=datetime(2026, 5, 13, tzinfo=timezone.utc),
        is_primary_interaction_busy=lambda: False,
        collect_cases_func=collect_cases,
        run_case_func=run_case,
        read_attempts_func=read_attempts,
        max_cases=3,
    )

    assert result.processed_count == 1
    assert captured_case["existing_attempts"][0]["idempotency_key"] == (
        prior_attempt["idempotency_key"]
    )


@pytest.mark.asyncio
async def test_worker_tick_blocks_unbound_case_before_candidate_render(
    tmp_path,
) -> None:
    """Unbound worker cases should not become private candidates."""

    case = _commitment_case()
    run_case = AsyncMock()
    record_attempt = AsyncMock()

    async def collect_cases(*, now: datetime, max_cases: int) -> list[dict[str, Any]]:
        del now, max_cases
        return [case]

    result = await worker.run_self_cognition_worker_tick(
        now=datetime(2026, 5, 13, tzinfo=timezone.utc),
        is_primary_interaction_busy=lambda: False,
        collect_cases_func=collect_cases,
        run_case_func=run_case,
        read_attempts_func=lambda **kwargs: [],
        record_attempt_func=record_attempt,
        max_cases=3,
    )

    assert result.processed_count == 0
    assert result.skipped_count == 1
    run_case.assert_not_called()
    record_attempt.assert_not_awaited()
    self_kwargs = worker.event_logging.record_self_cognition_event.await_args.kwargs
    assert self_kwargs["dispatch_status"] == "target_binding_failed"
    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_worker_tick_suppresses_duplicate_due_occurrence_from_prior_attempts(
    tmp_path,
) -> None:
    """A prior persisted attempt should prevent a repeated action attempt."""

    case = _commitment_case_with_delivery_target()
    prior_attempt = _action_attempt(
        case,
        status=models.ACTION_ATTEMPT_STATUS_SCHEDULED,
    )
    recorded_attempts = [prior_attempt]

    async def collect_cases(*, now: datetime, max_cases: int) -> list[dict[str, Any]]:
        del now, max_cases
        return [case]

    async def read_attempts(*, limit: int) -> list[dict[str, Any]]:
        assert limit > 0
        return list(recorded_attempts)

    async def record_attempt(attempt: dict[str, Any]) -> None:
        recorded_attempts.append(dict(attempt))

    def run_duplicate_case(
        next_case: dict[str, Any],
    ) -> dict[str, Any]:
        assert next_case["existing_attempts"][0]["idempotency_key"] == (
            prior_attempt["idempotency_key"]
        )
        duplicate = _action_attempt(
            next_case,
            status=models.ACTION_ATTEMPT_STATUS_DUPLICATE,
        )
        payloads = {models.ARTIFACT_ACTION_ATTEMPT: duplicate}
        return payloads

    await worker.run_self_cognition_worker_tick(
        now=datetime(2026, 5, 13, tzinfo=timezone.utc),
        is_primary_interaction_busy=lambda: False,
        collect_cases_func=collect_cases,
        run_case_func=run_duplicate_case,
        read_attempts_func=read_attempts,
        record_attempt_func=record_attempt,
        max_cases=3,
    )

    assert recorded_attempts[-1]["status"] == (
        models.ACTION_ATTEMPT_STATUS_DUPLICATE
    )
    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_worker_tick_uses_attempt_updates_between_cases(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """Same-tick duplicate cases should see persisted attempts recorded earlier."""

    _patch_dispatcher_persistence(monkeypatch)
    adapter = _FakeMessagingAdapter()
    case = _commitment_case_with_delivery_target()
    recorded_attempts: list[dict[str, Any]] = []

    async def collect_cases(
        *,
        now: datetime,
        max_cases: int,
    ) -> list[dict[str, Any]]:
        del now, max_cases
        return [case, dict(case)]

    async def read_attempts(*, limit: int) -> list[dict[str, Any]]:
        assert limit > 0
        return list(recorded_attempts)

    async def record_attempt(attempt: dict[str, Any]) -> None:
        recorded_attempts.append(dict(attempt))

    result = await worker.run_self_cognition_worker_tick(
        now=datetime(2026, 5, 13, tzinfo=timezone.utc),
        is_primary_interaction_busy=lambda: False,
        collect_cases_func=collect_cases,
        run_case_func=_case_runner_with_tracking,
        read_attempts_func=read_attempts,
        record_attempt_func=record_attempt,
        adapter_registry_provider=lambda: _adapter_registry(adapter),
        max_cases=3,
    )

    assert result.processed_count == 2
    assert adapter.calls == [
        {
            "channel_id": "dm-1",
            "text": "Checking in now.",
            "channel_type": "private",
            "reply_to_msg_id": None,
            "delivery_mentions": [],
        }
    ]
    assert recorded_attempts[0]["status"] == models.ACTION_ATTEMPT_STATUS_SENT
    assert recorded_attempts[1]["status"] == (
        models.ACTION_ATTEMPT_STATUS_DUPLICATE
    )
    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_worker_tick_defers_when_primary_interaction_is_busy(tmp_path) -> None:
    """The idle worker should not compete with active chat work."""

    async def collect_cases(*, now: datetime, max_cases: int) -> list[dict[str, Any]]:
        raise AssertionError("busy tick should not collect cases")

    result = await worker.run_self_cognition_worker_tick(
        now=datetime(2026, 5, 13, tzinfo=timezone.utc),
        is_primary_interaction_busy=lambda: True,
        collect_cases_func=collect_cases,
        max_cases=3,
    )

    assert result.deferred is True
    assert result.defer_reason == "primary interaction busy"
    assert result.processed_count == 0


@pytest.mark.asyncio
async def test_worker_tick_defers_pipeline_cancelled_case() -> None:
    """Cooperative cancellation should stop before successful action records."""

    from kazusa_ai_chatbot.runtime_coordination import (
        PipelineCancellation,
        PipelineCancelled,
        PipelineScope,
    )

    case = _commitment_case_with_delivery_target()
    recorded_attempts: list[dict[str, Any]] = []

    async def collect_cases(
        *,
        now: datetime,
        max_cases: int,
    ) -> list[dict[str, Any]]:
        assert now == datetime(2026, 5, 13, tzinfo=timezone.utc)
        assert max_cases == 3
        return [case]

    async def run_case(**_kwargs) -> dict[str, Any]:
        cancellation = PipelineCancellation(
            run_id="pipeline-run-1",
            scope=PipelineScope(
                platform="qq",
                platform_channel_id="group-1",
                channel_type="group",
            ),
            requested_by="service.chat_queue",
            reason="same_scope_foreground_pending",
            checkpoint="before_dispatch",
        )
        raise PipelineCancelled(cancellation)

    async def record_attempt(attempt: dict[str, Any]) -> None:
        recorded_attempts.append(dict(attempt))

    result = await worker.run_self_cognition_worker_tick(
        now=datetime(2026, 5, 13, tzinfo=timezone.utc),
        is_primary_interaction_busy=lambda: False,
        collect_cases_func=collect_cases,
        run_case_func=run_case,
        record_attempt_func=record_attempt,
        max_cases=3,
    )

    assert result.deferred is True
    assert result.defer_reason == "same_scope_foreground_pending"
    assert result.failed_count == 0
    assert result.processed_count == 0
    assert recorded_attempts == []


@pytest.mark.asyncio
async def test_worker_tick_defer_requeues_claimed_source_calendar_run() -> None:
    """Cancelled claimed source runs should not consume retry budget."""

    from kazusa_ai_chatbot.runtime_coordination import (
        PipelineCancellation,
        PipelineCancelled,
        PipelineScope,
    )

    case = _commitment_case_with_delivery_target()
    case["source_calendar_run_id"] = "calendar_run_commitment_123"
    claimed_runs: list[str] = []
    deferred_runs: list[dict[str, object]] = []
    completed_runs: list[str] = []
    failed_runs: list[str] = []

    async def collect_cases(
        *,
        now: datetime,
        max_cases: int,
    ) -> list[dict[str, Any]]:
        assert now == datetime(2026, 5, 13, tzinfo=timezone.utc)
        assert max_cases == 3
        return [case]

    async def claim_run(run_id: str, **kwargs) -> bool:
        assert kwargs["trigger_kind"] == (
            calendar_models.TRIGGER_COMMITMENT_DUE_COGNITION
        )
        claimed_runs.append(run_id)
        return True

    async def run_case(**_kwargs) -> dict[str, Any]:
        cancellation = PipelineCancellation(
            run_id="pipeline-run-1",
            scope=PipelineScope(
                platform="qq",
                platform_channel_id="group-1",
                channel_type="group",
            ),
            requested_by="service.chat_queue",
            reason="same_scope_foreground_pending",
            checkpoint="before_dispatch",
        )
        raise PipelineCancelled(cancellation)

    async def defer_run(run_id: str, **kwargs) -> bool:
        deferred_runs.append({"run_id": run_id, **kwargs})
        return True

    async def complete_run(run_id: str, **_kwargs) -> None:
        completed_runs.append(run_id)

    async def fail_run(run_id: str, **_kwargs) -> None:
        failed_runs.append(run_id)

    result = await worker.run_self_cognition_worker_tick(
        now=datetime(2026, 5, 13, tzinfo=timezone.utc),
        is_primary_interaction_busy=lambda: False,
        collect_cases_func=collect_cases,
        run_case_func=run_case,
        claim_calendar_run_func=claim_run,
        complete_calendar_run_func=complete_run,
        fail_calendar_run_func=fail_run,
        defer_calendar_run_func=defer_run,
        max_cases=3,
    )

    assert result.deferred is True
    assert result.defer_reason == "same_scope_foreground_pending"
    assert claimed_runs == ["calendar_run_commitment_123"]
    assert deferred_runs == [
        {
            "run_id": "calendar_run_commitment_123",
            "lease_owner": "self_cognition_worker",
            "storage_timestamp_utc": "2026-05-13T00:00:00+00:00",
            "reason": "same_scope_foreground_pending",
        }
    ]
    assert completed_runs == []
    assert failed_runs == []


@pytest.mark.asyncio
async def test_worker_tick_releases_pipeline_handle_when_claim_raises() -> None:
    """Coordinator handles must release when pre-run calendar claim fails."""

    from kazusa_ai_chatbot.runtime_coordination import (
        PipelineCoordinator,
        PipelineScope,
    )

    case = _commitment_case_with_delivery_target()
    case["source_calendar_run_id"] = "calendar_run_commitment_123"
    coordinator = PipelineCoordinator()
    scope = PipelineScope(
        platform="qq",
        platform_channel_id="group-1",
        channel_type="group",
    )

    async def collect_cases(
        *,
        now: datetime,
        max_cases: int,
    ) -> list[dict[str, Any]]:
        assert max_cases == 3
        return [case]

    async def claim_run(*_args, **_kwargs) -> bool:
        raise RuntimeError("claim failed")

    result = await worker.run_self_cognition_worker_tick(
        now=datetime(2026, 5, 13, tzinfo=timezone.utc),
        is_primary_interaction_busy=lambda: False,
        collect_cases_func=collect_cases,
        claim_calendar_run_func=claim_run,
        pipeline_coordinator=coordinator,
        max_cases=3,
    )

    assert result.failed_count == 1
    assert result.processed_count == 0
    assert coordinator.request_cancellation(
        scope=scope,
        requested_by="test",
        reason="probe",
    ) == []


@pytest.mark.asyncio
async def test_worker_tick_pauses_before_collection_for_affect_settling() -> None:
    """Pending daily affect settling should pause self-cognition collection."""

    collect_cases = AsyncMock(
        side_effect=AssertionError("paused tick should not collect cases"),
    )
    should_pause = AsyncMock(return_value=True)

    result = await worker.run_self_cognition_worker_tick(
        now=datetime(2026, 5, 13, tzinfo=timezone.utc),
        is_primary_interaction_busy=lambda: False,
        collect_cases_func=collect_cases,
        should_pause_for_affect_settling=should_pause,
        max_cases=3,
    )

    assert result.deferred is True
    assert result.defer_reason == "daily affect settling pending"
    assert result.processed_count == 0
    collect_cases.assert_not_awaited()
    should_pause.assert_awaited_once()


@pytest.mark.asyncio
async def test_active_commitment_source_builds_due_case_from_memory_unit() -> None:
    """Active commitment collection should build visible/actionable case input."""

    unit = {
        "unit_id": "promise-001",
        "global_user_id": "673225019",
        "unit_type": "active_commitment",
        "status": "active",
        "fact": "A promised follow-up is due.",
        "subjective_appraisal": "The user may expect a check-in.",
        "relationship_signal": "Following through matters.",
        "due_at": "2026-05-13T00:00:00+00:00",
        "last_seen_at": "2026-05-12T23:55:00+00:00",
        "updated_at": "2026-05-12T23:55:00+00:00",
    }
    rows = [
        {
            "platform": "qq",
            "platform_channel_id": "673225019",
            "channel_type": "private",
            "role": "user",
            "global_user_id": "673225019",
            "display_name": "User",
            "body_text": "Please check back after the appointment.",
            "timestamp": "2026-05-12T23:50:00+00:00",
        }
    ]

    async def list_commitments(*, current_timestamp_utc: str, limit: int):
        assert current_timestamp_utc == "2026-05-13T00:30:00+00:00"
        assert limit == 3
        return [unit]

    async def get_history(**kwargs):
        assert kwargs["global_user_id"] == "673225019"
        return rows

    async def get_profile(global_user_id: str):
        assert global_user_id == "673225019"
        return {"relationship_state": 600, "display_name": "User"}

    async def no_private_channel(**kwargs: Any) -> None:
        del kwargs
        return None

    cases = await sources.collect_active_commitment_cases(
        now=datetime(2026, 5, 13, 0, 30, tzinfo=timezone.utc),
        character_profile={"name": "Character", "mood": "focused"},
        max_cases=3,
        list_active_commitments_func=list_commitments,
        get_conversation_history_func=get_history,
        get_user_profile_func=get_profile,
        get_latest_private_channel_func=no_private_channel,
    )

    assert len(cases) == 1
    assert cases[0]["case_name"] == models.CASE_COMMITMENT_PAST_DUE
    assert cases[0]["target_scope"] == {
        **_target_scope(),
        "platform_user_id": "",
        "display_name": "User",
    }
    assert cases[0]["source_refs"][0]["source_id"] == "promise-001"
    assert cases[0]["visible_context"][0]["body_text"].startswith("Please")


@pytest.mark.asyncio
async def test_active_commitment_query_prioritizes_due_work(
    monkeypatch,
) -> None:
    """Active commitment reads should prioritize due items inside the tick cap."""

    collection = _FakeUserMemoryUnitsCollection()

    class FakeDatabase:
        user_memory_units = collection

    async def fake_get_db():
        database = FakeDatabase()
        return database

    monkeypatch.setattr(memory_units_module, "get_db", fake_get_db)

    rows = await memory_units_module.query_active_commitment_memory_units(
        current_timestamp_utc="2026-05-13T00:30:00+00:00",
        limit=3,
    )
    pipeline = collection.pipeline

    assert rows == [{"unit_id": "promise-001"}]
    assert pipeline[0]["$match"]["due_at"] == {"$type": "string", "$ne": ""}
    assert pipeline[1]["$addFields"]["_self_cognition_due_at"] == {
        "$dateFromString": {
            "dateString": {
                "$replaceOne": {
                    "input": "$due_at",
                    "find": " ",
                    "replacement": "T",
                }
            },
            "onError": None,
            "onNull": None,
        }
    }
    assert pipeline[2] == {"$match": {"_self_cognition_due_at": {"$ne": None}}}
    assert pipeline[3]["$addFields"]["_self_cognition_due_bucket"] == {
        "$cond": [
            {
                "$lte": [
                    "$_self_cognition_due_at",
                    datetime(2026, 5, 13, 0, 30, tzinfo=timezone.utc),
                ]
            },
            memory_units_module.ACTIVE_COMMITMENT_DUE_BUCKET_READY,
            memory_units_module.ACTIVE_COMMITMENT_DUE_BUCKET_FUTURE,
        ]
    }
    assert pipeline[4]["$sort"] == {
        "_self_cognition_due_bucket": 1,
        "_self_cognition_due_at": 1,
        "last_seen_at": -1,
        "updated_at": -1,
    }
    assert pipeline[5] == {"$limit": 3}
    assert pipeline[6]["$project"]["_self_cognition_due_at"] == 0
    assert pipeline[6]["$project"]["_self_cognition_due_bucket"] == 0
