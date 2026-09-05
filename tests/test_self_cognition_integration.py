"""Deterministic integration tests for the self-cognition runtime boundary."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

import kazusa_ai_chatbot.dispatcher.handlers as handlers_module
from kazusa_ai_chatbot import service
from kazusa_ai_chatbot.action_spec.registry import SPEAK_CAPABILITY
from kazusa_ai_chatbot.calendar_scheduler import models as calendar_models
from kazusa_ai_chatbot.cognition_core_v3.diagnostics import current_chain_scope
from kazusa_ai_chatbot.cognition_shared.contracts import (
    build_scheduled_future_speech_authority,
)
from kazusa_ai_chatbot.cognition_shared.state_models import (
    build_acquaintance_user_state,
)
from kazusa_ai_chatbot.db import user_memory_units as memory_units_module
from kazusa_ai_chatbot.dispatcher import AdapterRegistry, SendResult
from kazusa_ai_chatbot.nodes.dialog_agent import StateContractError
from kazusa_ai_chatbot.self_cognition import (
    models,
    projection,
    runner,
    sources,
    tracking,
    worker,
)
from tests.cognition_test_helpers import (
    canonical_cognition_output,
    canonical_service_character_profile,
)


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


@pytest.mark.asyncio
async def test_internal_latch_case_hydrates_bound_user_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An identity-bound internal latch retains its real user state."""

    user_id = "internal-latch-user"
    user_profile = {
        "global_user_id": user_id,
        "display_name": "Internal Latch User",
        "cognition_state": build_acquaintance_user_state(
            global_user_id=user_id,
            updated_at="2026-08-31T00:00:00Z",
        ),
    }
    profile_reader = AsyncMock(return_value=user_profile)
    monkeypatch.setattr(worker.db, "get_user_profile", profile_reader)

    case = await worker._case_from_internal_action_latch(
        {
            "claim_token": "claim-internal-latch",
            "latch": {
                "latch_id": "latch-internal-profile",
                "source_episode_id": "episode-internal-profile",
                "source_action_attempt_id": "attempt-internal-profile",
                "continuation_objective": "Continue the grounded task.",
                "evidence_refs": [],
                "target_scope": {
                    "platform": "debug",
                    "platform_channel_id": "private-internal-profile",
                    "channel_type": "private",
                    "current_global_user_id": user_id,
                    "current_platform_user_id": "platform-internal-user",
                    "current_display_name": "Internal Latch User",
                    "source_platform_bot_id": "bot-internal-profile",
                },
            },
        },
        character_profile={"name": "Test Character"},
        now=datetime(2026, 8, 31, tzinfo=timezone.utc),
    )

    profile_reader.assert_awaited_once_with(user_id)
    assert case["user_profile"] == user_profile
    assert case["user_profile"] is not user_profile


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


def _test_interaction_style_snapshot() -> dict[str, Any]:
    """Build the smallest immutable style snapshot accepted by self-cognition."""

    overlay = {
        "speech_guidelines": [],
        "social_guidelines": [],
        "pacing_guidelines": [],
        "engagement_guidelines": [],
        "confidence": "medium",
    }
    snapshot = {
        "schema_version": "interaction_style_turn_snapshot.v1",
        "sources": {},
        "relevance": {},
        "cognition": {},
        "surface": {
            "user": {"overlay": dict(overlay)},
            "group_channel": {"overlay": dict(overlay)},
        },
        "application_order": ["user", "group_channel"],
        "user_style": dict(overlay),
        "group_engagement_action_context": {
            "engagement_guidelines": ["Stay grounded in the visible scene."],
            "confidence": "medium",
        },
        "snapshot_digest": "test-style-snapshot",
    }
    return snapshot


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


def _future_cognition_run(
    *,
    platform: str = "qq",
    channel_id: str = "480386272",
    channel_type: str = "group",
    audience_kind: str = "group",
) -> dict[str, Any]:
    authority = build_scheduled_future_speech_authority(
        source_episode_id="episode-123",
        source_message_id="227312230",
        source_action_attempt_id="action_attempt:future-123",
        source_llm_trace_id="llmtrace_source-1",
        accepted_at_utc="2026-05-16T09:00:00+00:00",
        timezone="Pacific/Auckland",
        trigger_local="2026-05-16 22:00",
        platform=platform,
        channel_type=channel_type,
        audience_kind=audience_kind,
        semantic_objective="Re-check whether a natural pause appeared.",
        authorized_content_summary="在约定时间开始补偿考核。",
        authorized_detail_refs=[
            {
                "evidence_handle": "e1",
                "semantic_summary": "当前对话明确约定在该时间开始补偿考核。",
                "provenance_role": "current_event",
            }
        ],
    )
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
            "scheduled_future_speech_authority": dict(authority),
            "source_refs": [
                {
                    "ref_kind": "cognitive_episode",
                    "ref_id": calendar_models.FUTURE_SPEAK_SOURCE_REF_ID,
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
            "source_platform": platform,
            "source_channel_id": channel_id,
            "source_channel_type": channel_type,
            "source_user_id": "self_cognition",
            "source_message_id": "227312230",
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
    authority = build_scheduled_future_speech_authority(
        source_episode_id="episode-123",
        source_message_id="227312230",
        source_action_attempt_id="action_attempt:future-123",
        source_llm_trace_id="llmtrace_source-1",
        accepted_at_utc="2026-05-16T09:00:00+00:00",
        timezone="Pacific/Auckland",
        trigger_local="2026-05-16 22:00",
        platform="qq",
        channel_type="group",
        audience_kind="group",
        semantic_objective="Re-check whether a natural pause appeared.",
        authorized_content_summary="在约定时间开始补偿考核。",
        authorized_detail_refs=[
            {
                "evidence_handle": "e1",
                "semantic_summary": "当前对话明确约定在该时间开始补偿考核。",
                "provenance_role": "current_event",
            }
        ],
    )
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
        "conversation_progress": None,
        "source_context": {
            "schema_version": "self_cognition_scheduled_source_context.v1",
            "context_kind": "scheduled_future_cognition",
            "continuation_objective": "Re-check the open topic.",
            "continuation_mode": "observe_then_decide",
        },
        "source_calendar_run_id": "calendar_run_future_123",
        "source_calendar_run_due_at": "2026-05-16T10:00:00+00:00",
        "scheduled_future_speech_authority": dict(authority),
        "source_action_attempt_id": "action_attempt:future-123",
        "character_profile": {"name": "Character"},
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


def _self_cognition_core_output(
    *,
    state_scope: str = "user",
) -> dict[str, Any]:
    """Build one exact canonical self-cognition product for worker tests."""

    output = canonical_cognition_output(
        route="silence",
        state_scope=state_scope,
    )
    response_goal = "stay silent and retain internal progress"
    output["response_plan"].update({
        "goal_resolution": "answerable_now",
        "response_goal": response_goal,
        "self_cognition_response": {
            "decision": "stay_silent",
            "response_goal": response_goal,
            "reason": "the due source supports this bounded decision",
            "cause_summary": "the current self-cognition episode is due",
        },
    })
    return output


def _self_cognition_visible_core_output(
    *,
    state_scope: str = "user",
) -> dict[str, Any]:
    """Build one exact visible self-cognition product for worker tests."""

    output = canonical_cognition_output(
        route="speech",
        state_scope=state_scope,
    )
    response_goal = "send the grounded scheduled follow-up"
    output["response_plan"].update({
        "goal_resolution": "answerable_now",
        "response_goal": response_goal,
        "self_cognition_response": {
            "decision": "propose_visible_reply",
            "response_goal": response_goal,
            "reason": "the due source supports this bounded decision",
            "cause_summary": "the current self-cognition episode is due",
        },
    })
    return output


def _progress_cognition_output(
    *,
    state_scope: str = "user",
) -> dict[str, Any]:
    """Build a cognition output that stays internal but affects state."""

    output = {
        "logical_stance": "OBSERVE",
        "character_intent": "WAIT",
        "self_cognition_route": models.ROUTE_PROGRESS_MAINTENANCE,
        "cognition_core_output": _self_cognition_core_output(
            state_scope=state_scope,
        ),
        "cognition_state_committed": True,
    }
    return output


def _speak_action_spec() -> dict[str, Any]:
    """Build the selected visible action spec used by worker tests."""

    spec = {
        "schema_version": "action_spec.v1",
        "kind": SPEAK_CAPABILITY,
        "cognition_mode": "deliberative",
        "source_refs": [],
        "target": {
            "schema_version": "action_target.v1",
            "target_kind": "current_user",
            "target_id": None,
            "owner": "l3_text",
            "scope": {},
        },
        "params": {
            "delivery_mode": "visible_reply",
            "execute_at": None,
            "surface_requirements": {},
        },
        "urgency": "now",
        "visibility": "user_visible",
        "deadline": None,
        "continuation": {
            "schema_version": "action_continuation.v1",
            "mode": "none",
            "episode_type": None,
            "max_depth": 0,
            "include_result_as": None,
        },
        "surface_role": "ordinary",
        "goal_continuation_ref": None,
        "reason": "A bounded scheduled follow-up may be useful.",
    }
    return spec


def _text_surface_output(content_plan: str = "Checking in now.") -> dict[str, Any]:
    """Build the canonical V2 surface result used by worker tests."""

    return {
        "schema_version": "text_surface_output.v2",
        "content_plan": content_plan,
        "content_requirements": ["Preserve the scheduled follow-up purpose."],
        "epistemic_boundary": "Preserve the scheduled state exactly.",
        "visible_boundaries": [],
        "addressee_plan": [{
            "handle": "current_user",
            "display_name": "current user",
            "semantic_role": "direct_recipient",
            "wording_policy": "second_person_allowed",
        }],
        "delivery_profile": {
            "lexical_register": "plain",
            "sentence_shape": "brief",
            "rhythm": "steady",
            "hesitation": "minimal",
            "punctuation": "restrained",
        },
        "selected_surface_intent": "answer the scheduled follow-up",
        "permitted_action_results": [],
    }


def _action_cognition_output(
    *,
    state_scope: str = "user",
) -> dict[str, Any]:
    """Build a cognition output that selects visible dialog through speak."""

    output = {
        "logical_stance": "CONFIRM",
        "character_intent": "PROVIDE",
        "text_surface_output_v2": _text_surface_output(),
        "action_specs": [_speak_action_spec()],
        "cognition_core_output": _self_cognition_visible_core_output(
            state_scope=state_scope,
        ),
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
async def test_collect_scheduled_future_cognition_cases_projects_typed_source_context() -> None:
    """Due future-cognition slots use source context, not participant progress."""

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
    assert case["conversation_progress"] is None
    assert case["source_context"]["continuation_objective"] == (
        "Re-check whether a natural pause appeared."
    )
    assert case["source_context"]["continuation_mode"] == (
        "scheduled_followup"
    )
    assert "context_summary" not in case["source_context"]
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
    second_authority = build_scheduled_future_speech_authority(
        source_episode_id="episode-123",
        source_message_id="227312230",
        source_action_attempt_id="action_attempt:future-456",
        source_llm_trace_id="llmtrace_source-1",
        accepted_at_utc="2026-05-16T09:00:00+00:00",
        timezone="Pacific/Auckland",
        trigger_local="2026-05-16 22:00",
        platform="qq",
        channel_type="group",
        audience_kind="group",
        semantic_objective="Re-check whether a natural pause appeared.",
        authorized_content_summary="在约定时间开始补偿考核。",
        authorized_detail_refs=[
            {
                "evidence_handle": "e1",
                "semantic_summary": "当前对话明确约定在该时间开始补偿考核。",
                "provenance_role": "current_event",
            }
        ],
    )
    second_run["payload"]["scheduled_future_speech_authority"] = dict(
        second_authority
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
    run = _future_cognition_run(
        channel_type="private",
        audience_kind="private",
    )
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
        character_profile={
            "name": "Character",
            "mood": "focused",
            "platform_bot_id": "bot-001",
        },
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
    assert cases[0]["platform_bot_id"] == "bot-001"
    assert cases[0]["delivery_target"]["source_platform_bot_id"] == "bot-001"
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


@pytest.mark.asyncio
async def test_scheduled_case_carries_authority_and_source_identity() -> None:
    """Collected scheduled cases keep the authority and source identity."""

    now = datetime(2026, 5, 16, 10, 0, tzinfo=timezone.utc)
    run = _future_cognition_run()

    async def list_due_runs(**kwargs: Any) -> list[dict[str, Any]]:
        del kwargs
        return [run]

    cases = await sources.collect_scheduled_future_cognition_cases(
        now=now,
        character_profile={"name": "TestCharacter"},
        max_cases=3,
        list_due_calendar_runs_func=list_due_runs,
        get_latest_private_channel_func=lambda **kwargs: None,
    )

    assert len(cases) == 1
    case = cases[0]
    authority = case["scheduled_future_speech_authority"]
    assert authority["authority_id"].startswith("sha256-")
    assert authority["source"]["source_episode_id"] == "episode-123"
    assert authority["source"]["source_message_id"] == "227312230"
    assert authority["source"]["source_action_attempt_id"] == (
        "action_attempt:future-123"
    )
    assert case["source_calendar_run_due_at"] == run["due_at"]
    assert case["source_action_attempt_id"] == "action_attempt:future-123"


@pytest.mark.asyncio
async def test_scheduled_case_rejects_mismatched_source_attempt_objective_target_and_trigger() -> None:
    """Scheduled runs contradicting their authority become typed skip cases."""

    now = datetime(2026, 5, 16, 10, 0, tzinfo=timezone.utc)

    async def collect(run: dict[str, Any]) -> list[dict[str, Any]]:
        async def list_due_runs(**kwargs: Any) -> list[dict[str, Any]]:
            del kwargs
            return [run]

        cases = await sources.collect_scheduled_future_cognition_cases(
            now=now,
            character_profile={"name": "TestCharacter"},
            max_cases=3,
            list_due_calendar_runs_func=list_due_runs,
            get_latest_private_channel_func=lambda **kwargs: None,
        )
        return cases

    base_run = _future_cognition_run()
    cases = await collect(base_run)
    assert len(cases) == 1
    assert cases[0]["source_calendar_run_id"] == "calendar_run_future_123"

    mismatched_attempt = _future_cognition_run()
    mismatched_attempt["payload"]["source_action_attempt_id"] = (
        "action_attempt:other"
    )
    cases = await collect(mismatched_attempt)
    assert cases[0]["source_calendar_skip_reason"] == (
        "scheduled_authority_invalid"
    )
    assert "scheduled_future_speech_authority" not in cases[0]

    mismatched_objective = _future_cognition_run()
    mismatched_objective["payload"]["continuation_objective"] = "另一个目标。"
    cases = await collect(mismatched_objective)
    assert cases[0]["source_calendar_skip_reason"] == (
        "scheduled_authority_invalid"
    )

    mismatched_trigger = _future_cognition_run()
    mismatched_trigger["due_at"] = "2026-05-16T09:30:00+00:00"
    cases = await collect(mismatched_trigger)
    assert cases[0]["source_calendar_skip_reason"] == (
        "scheduled_authority_invalid"
    )

    mismatched_platform = _future_cognition_run()
    mismatched_platform["source_scope"]["source_platform"] = "debug"
    cases = await collect(mismatched_platform)
    assert cases[0]["source_calendar_skip_reason"] == (
        "scheduled_authority_invalid"
    )

    mismatched_channel = _future_cognition_run()
    mismatched_channel["source_scope"]["source_channel_type"] = "private"
    cases = await collect(mismatched_channel)
    assert cases[0]["source_calendar_skip_reason"] == (
        "scheduled_authority_invalid"
    )

    audience_mismatch = _future_cognition_run(audience_kind="private")
    cases = await collect(audience_mismatch)
    assert cases[0]["source_calendar_skip_reason"] == (
        "scheduled_authority_invalid"
    )








@pytest.mark.asyncio
def test_rejected_scheduled_candidate_is_removed_before_consolidation() -> None:
    """The worker scrub strips candidate text before consolidation input."""

    artifact_payloads = {
        models.ARTIFACT_ACTION_CANDIDATE: {
            "text": "厕所隔间的检查已经准备好了。",
        },
        models.ARTIFACT_COGNITION_OUTPUT: {
            "surface_outputs": [
                {
                    "schema_version": "surface_output.v1",
                    "visibility": "user_visible",
                    "delivery_intent": "deliver_now",
                    "text": "厕所隔间的检查已经准备好了。",
                },
                {
                    "schema_version": "surface_output.v1",
                    "visibility": "private",
                    "delivery_intent": "do_not_deliver",
                    "summary": "audit retained",
                },
            ]
        },
        models.RUNTIME_CONSOLIDATION_STATE: {
            "final_dialog": ["厕所隔间的检查已经准备好了。"],
            "surface_outputs": [
                {
                    "schema_version": "surface_output.v1",
                    "visibility": "user_visible",
                    "delivery_intent": "deliver_now",
                    "text": "厕所隔间的检查已经准备好了。",
                }
            ],
            "decontextualized_input": "来源：其他有效认知证据。",
        },
    }

    worker._scrub_scheduled_candidate_from_consolidation(artifact_payloads)

    consolidation_state = artifact_payloads[
        models.RUNTIME_CONSOLIDATION_STATE
    ]
    assert consolidation_state["final_dialog"] == []
    assert consolidation_state["surface_outputs"] == []
    assert (
        consolidation_state["decontextualized_input"]
        == "来源：其他有效认知证据。"
    )
    cognition_output = artifact_payloads[
        models.ARTIFACT_COGNITION_OUTPUT
    ]
    assert all(
        output.get("delivery_intent") != "deliver_now"
        for output in cognition_output["surface_outputs"]
    )
    assert models.ARTIFACT_ACTION_CANDIDATE not in artifact_payloads
