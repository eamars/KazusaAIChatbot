"""Deterministic self-cognition tracking and record contract tests."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from kazusa_ai_chatbot.action_spec.registry import (
    APPLY_MEMORY_LIFECYCLE_UPDATE_CAPABILITY,
    MEMORY_LIFECYCLE_UPDATE_CAPABILITY,
    SPEAK_CAPABILITY,
)
from kazusa_ai_chatbot.cognition_shared.contracts import (
    build_scheduled_future_speech_authority,
)
from kazusa_ai_chatbot.nodes.dialog_agent import StateContractError
from kazusa_ai_chatbot.self_cognition import (
    models,
    projection,
    runner,
    sources,
    tracking,
    worker,
)
from tests.cognition_test_helpers import canonical_cognition_output


@pytest.fixture(autouse=True)
def _disable_live_residue_recorder(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep deterministic tracking tests off the residue recorder LLM."""

    async def record_residue(**kwargs: Any) -> dict[str, Any]:
        del kwargs
        result = {
            "written": False,
            "skipped": True,
            "failure_reason": "deterministic_test_fixture",
        }
        return result

    monkeypatch.setattr(
        runner,
        "record_completed_episode_residue",
        record_residue,
    )


@pytest.mark.asyncio
async def test_worker_awaits_latest_character_profile_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The scheduler should resolve one fresh profile at each worker tick."""

    stop_event = asyncio.Event()
    received_profiles: list[dict[str, Any]] = []

    async def profile_provider() -> dict[str, Any]:
        return {"name": "revision-n"}

    async def run_tick(**kwargs: Any) -> worker.SelfCognitionWorkerResult:
        received_profiles.append(kwargs["character_profile"])
        stop_event.set()
        return worker.SelfCognitionWorkerResult(processed_count=1)

    monkeypatch.setattr(worker, "run_self_cognition_worker_tick", run_tick)

    await worker._self_cognition_worker_loop(
        stop_event=stop_event,
        is_primary_interaction_busy=lambda: False,
        character_profile_provider=profile_provider,
        adapter_registry_provider=None,
        latest_cognition_graph_publisher=None,
    )

    assert received_profiles == [{"name": "revision-n"}]


def _target_scope(channel_type: str = "private") -> dict[str, str | None]:
    platform_channel_id = "673225019"
    user_id = "673225019"
    if channel_type == "group":
        platform_channel_id = "54369546"
        user_id = None
    scope = {
        "platform": "qq",
        "platform_channel_id": platform_channel_id,
        "channel_type": channel_type,
        "user_id": user_id,
    }
    return scope


def _commitment_case(
    *,
    case_name: str = models.CASE_COMMITMENT_PAST_DUE,
    due_state: str = models.DUE_STATE_PAST_DUE,
) -> dict[str, Any]:
    case = {
        "case_name": case_name,
        "case_id": f"{case_name}:promise-001",
        "idle_timestamp_utc": "2026-05-10T00:30:00+00:00",
        "last_evidence_timestamp_utc": "2026-05-10T00:00:00+00:00",
        "trigger_kind": models.TRIGGER_ACTIVE_COMMITMENT_DUE_CHECK,
        "semantic_due_state": due_state,
        "actionability": "contact_is_socially_available",
        "target_scope": _target_scope(),
        "source_refs": [
            {
                "source_kind": "future_promise",
                "source_id": "promise-001",
                "due_at": "2026-05-10T00:00:00+00:00",
                "summary": "The user expected a follow-up by this time.",
            }
        ],
        "visible_context": [
            {
                "role": "user",
                "text": "Reminder was expected before this timestamp.",
                "timestamp": "2026-05-09T23:50:00+00:00",
            }
        ],
    }
    return case


def _duplicate_tick_case() -> dict[str, Any]:
    case = _commitment_case(case_name=models.CASE_COMMITMENT_DUPLICATE_TICK)
    source_ref = case["source_refs"][0]
    idempotency_key = tracking.build_idempotency_key(
        source_ref["source_kind"],
        source_ref["source_id"],
        source_ref["due_at"],
        case["target_scope"],
        models.ACTION_KIND_SEND_MESSAGE,
    )
    case["existing_attempts"] = [
        {
            "attempt_id": "self_cognition_attempt:existing",
            "idempotency_key": idempotency_key,
            "status": models.ACTION_ATTEMPT_STATUS_CANDIDATE,
        }
    ]
    return case




def _group_noise_case() -> dict[str, Any]:
    case = {
        "case_name": models.CASE_GROUP_NOISE_REJECTED,
        "case_id": "group-noise-001",
        "idle_timestamp_utc": "2026-05-10T00:30:00+00:00",
        "last_evidence_timestamp_utc": "2026-05-10T00:29:00+00:00",
        "trigger_kind": models.TRIGGER_GROUP_CHAT_REVIEW,
        "semantic_due_state": None,
        "actionability": "group_noise_no_clear_target",
        "target_scope": _target_scope(channel_type="group"),
        "source_refs": [
            {
                "source_kind": "conversation_window",
                "source_id": "group-window-001",
                "summary": "No direct mention or active commitment target.",
            }
        ],
        "visible_context": [
            {
                "role": "participant",
                "text": "Parallel group chatter without a clear target.",
                "timestamp": "2026-05-10T00:29:00+00:00",
            }
        ],
        "conversation_progress": None,
    }
    return case


def _group_chat_review_case() -> dict[str, Any]:
    """Build a group-review case with stable window identity."""

    case = {
        "case_name": models.CASE_GROUP_CHAT_REVIEW,
        "case_id": (
            "scope_group:2026-05-18T04:00:00+00:00:"
            "2026-05-18T04:15:00+00:00"
        ),
        "idle_timestamp_utc": "2026-05-18T04:15:00+00:00",
        "last_evidence_timestamp_utc": "2026-05-18T04:10:00+00:00",
        "trigger_kind": models.TRIGGER_GROUP_CHAT_REVIEW,
        "semantic_due_state": None,
        "actionability": "active_group_review_same_channel_no_fallback",
        "target_scope": _target_scope(channel_type="group"),
        "source_refs": [
            {
                "source_kind": "reflection_activity_window",
                "source_id": (
                    "scope_group:2026-05-18T04:00:00+00:00:"
                    "2026-05-18T04:15:00+00:00"
                ),
                "due_at": None,
                "summary": "quiet group activity, one speaker, risk low",
            }
        ],
        "visible_context": [
            {
                "role": "user",
                "body_text": "A recent group message.",
                "timestamp": "2026-05-18T04:10:00+00:00",
            }
        ],
        "conversation_progress": None,
        "source_context": {
            "schema_version": "self_cognition_group_source_context.v1",
            "context_kind": "group_chat_review",
            "group_activity_window": {
                "source": "reflection_activity_window",
                "window_start": "2026-05-18T04:00:00+00:00",
                "window_end": "2026-05-18T04:15:00+00:00",
                "semantic_labels": {
                    "assistant_presence": "present",
                    "bot_addressing": "directly_addressed",
                    "message_recency": "recent",
                    "response_risk": "low",
                },
            },
            "conversation_evidence": [],
        },
        "target_binding_status": "bound",
        "delivery_target": {
            "platform": "qq",
            "platform_channel_id": "54369546",
            "channel_type": "group",
            "user_id": None,
        },
    }
    return case


def _topic_followup_case() -> dict[str, Any]:
    case = {
        "case_name": models.CASE_TOPIC_RAG_FOLLOWUP,
        "case_id": "topic-followup-001",
        "idle_timestamp_utc": "2026-05-10T00:30:00+00:00",
        "last_evidence_timestamp_utc": "2026-05-10T00:00:00+00:00",
        "trigger_kind": models.TRIGGER_BOUNDED_FOLLOWUP_TOPIC,
        "semantic_due_state": models.DUE_STATE_FUTURE_DUE,
        "actionability": "bounded_topic_followup_requires_retrieval_before_contact",
        "target_scope": _target_scope(channel_type="group"),
        "source_refs": [
            {
                "source_kind": "conversation_episode_state",
                "source_id": "episode-001",
                "summary": "A technical follow-up topic remains open.",
            }
        ],
        "visible_context": [
            {
                "role": "user",
                "text": "Let's continue this architecture topic later.",
                "timestamp": "2026-05-10T00:00:00+00:00",
            }
        ],
    }
    return case


def _scheduled_future_cognition_case() -> dict[str, Any]:
    authority = build_scheduled_future_speech_authority(
        source_episode_id="episode-future-001",
        source_message_id="227312230",
        source_action_attempt_id="action_attempt:future-001",
        source_llm_trace_id="llmtrace_source-1",
        accepted_at_utc="2026-05-09T23:00:00+00:00",
        timezone="Pacific/Auckland",
        trigger_local="2026-05-10 12:30",
        platform="qq",
        channel_type="group",
        audience_kind="group",
        semantic_objective="Re-check the open topic.",
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
        "case_id": "future-cognition-001",
        "idle_timestamp_utc": "2026-05-10T00:30:00+00:00",
        "last_evidence_timestamp_utc": "2026-05-10T00:00:00+00:00",
        "trigger_kind": models.TRIGGER_SCHEDULED_FUTURE_COGNITION,
        "semantic_due_state": models.DUE_STATE_DUE_NOW,
        "actionability": "scheduled_private_followup_ready_no_direct_contact",
        "target_scope": _target_scope(channel_type="group"),
        "source_refs": [
            {
                "source_kind": "scheduled_future_cognition_slot",
                "source_id": "scheduled_future_cognition_slot",
                "due_at": "2026-05-10T00:30:00+00:00",
                "summary": "Re-check whether the open hardware topic changed.",
            }
        ],
        "visible_context": [
            {
                "role": "user",
                "text": "Let's check the GPU model topic later.",
                "timestamp": "2026-05-10T00:00:00+00:00",
            }
        ],
        "conversation_progress": None,
        "source_context": {
            "schema_version": (
                "self_cognition_scheduled_source_context.v1"
            ),
            "context_kind": "scheduled_future_cognition",
            "continuation_objective": "Re-check the open topic.",
            "continuation_mode": "observe_then_decide",
        },
        "source_calendar_run_id": "calendar_run_future_001",
        "source_calendar_run_due_at": "2026-05-10T00:30:00+00:00",
        "scheduled_future_speech_authority": dict(authority),
        "source_action_attempt_id": "action_attempt:future-001",
    }
    return case


def _action_cognition_output(text: str) -> dict[str, Any]:
    output = {
        "logical_stance": "CONFIRM",
        "character_intent": "PROVIDE",
        "internal_monologue": "The scheduled follow-up should be visible.",
        "text_surface_output_v2": {
            "schema_version": "text_surface_output.v2",
            "content_plan": text,
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
                "lexical_register": "direct",
                "sentence_shape": "brief",
                "rhythm": "steady",
                "hesitation": "minimal",
                "punctuation": "restrained",
            },
            "selected_surface_intent": "answer the scheduled follow-up",
            "permitted_action_results": [],
        },
        "action_specs": [_speak_action_spec()],
    }
    return output


def _progress_cognition_output() -> dict[str, Any]:
    output = {
        "logical_stance": "maintain awareness without outward contact",
        "character_intent": "keep progress internally visible",
        "self_cognition_route": models.ROUTE_PROGRESS_MAINTENANCE,
    }
    return output


def _audit_only_cognition_output_without_directives() -> dict[str, Any]:
    output = {
        "logical_stance": "DIVERGE",
        "character_intent": "SILENT_NO_WRITE",
        "self_cognition_route": models.ROUTE_AUDIT_ONLY,
        "action_specs": [],
    }
    return output


def _silent_cognition_output() -> dict[str, Any]:
    output = {
        "logical_stance": "no outward contact is warranted",
        "character_intent": "stay silent",
        "self_cognition_route": models.ROUTE_AUDIT_ONLY,
    }
    return output


def _speak_action_spec() -> dict[str, Any]:
    spec = {
        "schema_version": "action_spec.v1",
        "kind": SPEAK_CAPABILITY,
        "cognition_mode": "deliberative",
        "source_refs": [
            {
                "schema_version": "action_source_ref.v1",
                "ref_kind": "cognitive_episode",
                "ref_id": "self-cognition-episode",
                "owner": "cognition",
                "relationship": "basis",
                "evidence_refs": [],
            }
        ],
        "target": {
            "schema_version": "action_target.v1",
            "target_kind": "current_channel",
            "target_id": None,
            "owner": "l3_text",
            "scope": {"surface": "text"},
        },
        "params": {
            "delivery_mode": "visible_reply",
            "execute_at": None,
            "surface_requirements": {
                "intent": "answer the scheduled follow-up precisely",
            },
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
        "reason": "The scheduled self-cognition selected a visible reply.",
    }
    return spec


def _memory_lifecycle_action_spec() -> dict[str, Any]:
    spec = {
        "schema_version": "action_spec.v1",
        "kind": APPLY_MEMORY_LIFECYCLE_UPDATE_CAPABILITY,
        "cognition_mode": "deliberative",
        "source_refs": [
            {
                "schema_version": "action_source_ref.v1",
                "ref_kind": "cognitive_episode",
                "ref_id": "self-cognition-episode",
                "owner": "cognition_episode",
                "relationship": "basis",
                "evidence_refs": [],
            },
            {
                "schema_version": "action_source_ref.v1",
                "ref_kind": "memory_unit",
                "ref_id": "memory-unit-001",
                "owner": "user_memory_units",
                "relationship": "target",
                "evidence_refs": [],
            },
        ],
        "target": {
            "schema_version": "action_target.v1",
            "target_kind": "memory_unit",
            "target_id": "memory-unit-001",
            "owner": "user_memory_units",
            "scope": {"unit_type": "active_commitment"},
        },
        "params": {
            "memory_kind": "user_memory_unit",
            "unit_type": "active_commitment",
            "unit_id": "memory-unit-001",
            "lifecycle_decision": "abandoned",
            "due_at": "2026-05-10T00:00:00+00:00",
        },
        "urgency": "background",
        "visibility": "private",
        "deadline": None,
        "continuation": {
            "schema_version": "action_continuation.v1",
            "mode": "none",
            "episode_type": None,
            "max_depth": 0,
            "include_result_as": None,
        },
        "reason": "The character chose to abandon the stale commitment.",
    }
    return spec


def _memory_lifecycle_route_action_spec() -> dict[str, Any]:
    spec = {
        "schema_version": "action_spec.v1",
        "kind": MEMORY_LIFECYCLE_UPDATE_CAPABILITY,
        "cognition_mode": "deliberative",
        "source_refs": [
            {
                "schema_version": "action_source_ref.v1",
                "ref_kind": "cognitive_episode",
                "ref_id": "self-cognition-episode",
                "owner": "cognition_episode",
                "relationship": "basis",
                "evidence_refs": [],
            },
        ],
        "target": {
            "schema_version": "action_target.v1",
            "target_kind": "cognitive_episode",
            "target_id": None,
            "owner": "memory_lifecycle_specialist",
            "scope": {"unit_type": "active_commitment"},
        },
        "params": {
            "review_kind": "active_commitment_lifecycle",
            "detail": "Review the active commitment lifecycle.",
        },
        "urgency": "background",
        "visibility": "private",
        "deadline": None,
        "continuation": {
            "schema_version": "action_continuation.v1",
            "mode": "none",
            "episode_type": None,
            "max_depth": 0,
            "include_result_as": None,
        },
        "reason": "The self-cognition run selected lifecycle review.",
    }
    return spec


def _surface_output(content_plan: str = "Continue the GPU model topic.") -> dict[str, Any]:
    """Build the canonical V2 surface result used by dialog tests."""

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


def _self_cognition_core_output(
    *,
    state_scope: str = "user",
    visible_reply: bool,
) -> dict[str, Any]:
    """Build one exact canonical self-cognition product for tracking tests."""

    output = canonical_cognition_output(
        route="speech" if visible_reply else "silence",
        state_scope=state_scope,
    )
    response_goal = (
        "send the grounded scheduled follow-up"
        if visible_reply
        else "stay silent and retain internal progress"
    )
    output["response_plan"].update({
        "goal_resolution": "answerable_now",
        "response_goal": response_goal,
        "self_cognition_response": {
            "decision": (
                "propose_visible_reply" if visible_reply else "stay_silent"
            ),
            "response_goal": response_goal,
            "reason": "the due source supports this bounded decision",
            "cause_summary": "the current self-cognition episode is due",
        },
    })
    return output


def _speak_cognition_output_with_partial_directives() -> dict[str, Any]:
    output = {
        "logical_stance": "CONFIRM",
        "character_intent": "PROVIDE",
        "internal_monologue": "The scheduled follow-up should be answered.",
        "judgment_note": "A concise visible reply is appropriate.",
        "social_distance": "friendly",
        "emotional_intensity": "low",
        "vibe_check": "focused",
        "relational_dynamic": "scheduled follow-up",
        "action_specs": [_speak_action_spec()],
        "cognition_core_output": _self_cognition_core_output(
            visible_reply=True,
        ),
    }
    return output


def _build_tracking_records(
    case: dict[str, Any],
    unused_fixture_path: object | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Build self-cognition records for deterministic test cases."""

    del unused_fixture_path
    artifact_payloads = runner.build_self_cognition_case_artifacts(
        case,
        **kwargs,
    )
    return artifact_payloads


def _read_json(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        return_value = payload
        return return_value
    content = Path(payload).read_text(encoding="utf-8")
    data = json.loads(content)
    return data


def _dialog_client_with_text(
    text: str,
):
    """Build a deterministic dialog seam for selected speak tests."""

    async def dialog_client(state: dict[str, Any]) -> dict[str, Any]:
        del state
        result = {
            "final_dialog": [text],
        }
        return result

    return dialog_client


def test_build_idempotency_key_ignores_generated_text() -> None:
    case = _commitment_case()
    trigger_record = tracking.build_trigger_record(case)
    source_ref = case["source_refs"][0]
    idempotency_key = tracking.build_idempotency_key(
        source_ref["source_kind"],
        source_ref["source_id"],
        source_ref["due_at"],
        case["target_scope"],
        models.ACTION_KIND_SEND_MESSAGE,
    )
    action_attempt = tracking.build_action_attempt(
        case,
        trigger_record,
        existing_attempts=[],
    )

    first_candidate = tracking.build_action_candidate(
        case,
        action_attempt,
        "First possible message.",
    )
    second_candidate = tracking.build_action_candidate(
        case,
        action_attempt,
        "Different possible message.",
    )

    assert action_attempt["idempotency_key"] == idempotency_key
    assert first_candidate is not None
    assert second_candidate is not None
    assert first_candidate["attempt_id"] == second_candidate["attempt_id"]
    assert first_candidate["text"] != second_candidate["text"]


def test_build_idempotency_key_changes_when_due_occurrence_changes() -> None:
    case = _commitment_case()
    source_ref = case["source_refs"][0]
    first_key = tracking.build_idempotency_key(
        source_ref["source_kind"],
        source_ref["source_id"],
        "2026-05-10T00:00:00+00:00",
        case["target_scope"],
        models.ACTION_KIND_SEND_MESSAGE,
    )
    second_key = tracking.build_idempotency_key(
        source_ref["source_kind"],
        source_ref["source_id"],
        "2026-05-10T01:00:00+00:00",
        case["target_scope"],
        models.ACTION_KIND_SEND_MESSAGE,
    )

    assert first_key != second_key


def test_active_commitment_case_retains_target_platform_identity() -> None:
    unit = {
        "unit_id": "promise-001",
        "due_at": "2026-05-10T00:00:00+00:00",
        "fact": "The user promised a harder challenge.",
        "global_user_id": "global-target-1",
    }
    rows = [
        {
            "platform": "qq",
            "platform_channel_id": "54369546",
            "channel_type": "group",
            "platform_user_id": "qq-old",
            "display_name": "Old Name",
            "body_text": "Earlier message.",
            "timestamp": "2026-05-09T23:40:00+00:00",
        },
        {
            "platform": "qq",
            "platform_channel_id": "54369546",
            "channel_type": "group",
            "platform_user_id": "qq-target",
            "display_name": "Target User",
            "body_text": "Latest target message.",
            "timestamp": "2026-05-09T23:50:00+00:00",
        },
    ]

    case = sources._build_active_commitment_case(
        unit,
        rows,
        user_profile={"display_name": "Profile Name"},
        character_profile={},
        now=datetime(2026, 5, 10, 0, 30, tzinfo=timezone.utc),
        due_state=models.DUE_STATE_PAST_DUE,
    )

    assert case["target_scope"]["user_id"] == "global-target-1"
    assert case["target_scope"]["platform_user_id"] == "qq-target"
    assert case["target_scope"]["display_name"] == "Target User"


def test_group_action_candidate_omits_delivery_mention_without_inline_tag() -> None:
    case = _commitment_case()
    case["target_scope"] = {
        "platform": "qq",
        "platform_channel_id": "54369546",
        "channel_type": "group",
        "user_id": "global-target-1",
        "platform_user_id": "qq-target",
        "display_name": "Target User",
    }
    trigger_record = tracking.build_trigger_record(case)
    action_attempt = tracking.build_action_attempt(
        case,
        trigger_record,
        existing_attempts=[],
    )

    action_candidate = tracking.build_action_candidate(
        case,
        action_attempt,
        "Checking in now.",
    )

    assert action_candidate is not None
    assert "delivery_mentions" not in action_candidate


def test_group_action_candidate_carries_inline_delivery_mention() -> None:
    case = _commitment_case()
    case["target_scope"] = {
        "platform": "qq",
        "platform_channel_id": "54369546",
        "channel_type": "group",
        "user_id": "global-target-1",
        "platform_user_id": "qq-target",
        "display_name": "Target User",
    }
    trigger_record = tracking.build_trigger_record(case)
    action_attempt = tracking.build_action_attempt(
        case,
        trigger_record,
        existing_attempts=[],
    )

    action_candidate = tracking.build_action_candidate(
        case,
        action_attempt,
        "@Target User Checking in now.",
    )

    assert action_candidate is not None
    assert action_candidate["delivery_mentions"] == [
        {
            "entity_kind": "user",
            "platform_user_id": "qq-target",
            "display_name": "Target User",
        }
    ]


def test_group_review_action_candidate_uses_delivery_mention_users() -> None:
    case = _commitment_case(case_name=models.CASE_GROUP_CHAT_REVIEW)
    case["target_scope"] = {
        "platform": "qq",
        "platform_channel_id": "54369546",
        "channel_type": "group",
        "user_id": None,
    }
    case["delivery_mention_users"] = [
        {
            "global_user_id": "global-target-1",
            "platform_user_id": "qq-target",
            "display_name": "Target User",
        }
    ]
    trigger_record = tracking.build_trigger_record(case)
    action_attempt = tracking.build_action_attempt(
        case,
        trigger_record,
        existing_attempts=[],
    )

    action_candidate = tracking.build_action_candidate(
        case,
        action_attempt,
        "@Target User Checking this group thread.",
    )

    assert action_candidate is not None
    assert action_candidate["delivery_mentions"] == [
        {
            "entity_kind": "user",
            "platform_user_id": "qq-target",
            "display_name": "Target User",
        }
    ]


def test_private_action_candidate_keeps_inline_mention_for_adapter_noop(
) -> None:
    case = _commitment_case()
    case["target_scope"] = {
        "platform": "qq",
        "platform_channel_id": "673225019",
        "channel_type": "private",
        "user_id": "global-target-1",
        "platform_user_id": "qq-target",
        "display_name": "Target User",
    }
    trigger_record = tracking.build_trigger_record(case)
    action_attempt = tracking.build_action_attempt(
        case,
        trigger_record,
        existing_attempts=[],
    )

    action_candidate = tracking.build_action_candidate(
        case,
        action_attempt,
        "@Target User Checking in now.",
    )

    assert action_candidate is not None
    assert action_candidate["delivery_mentions"] == [
        {
            "entity_kind": "user",
            "platform_user_id": "qq-target",
            "display_name": "Target User",
        }
    ]


def test_build_idempotency_key_ignores_delivery_target_metadata() -> None:
    case = _commitment_case()
    source_ref = case["source_refs"][0]
    stable_key = tracking.build_idempotency_key(
        source_ref["source_kind"],
        source_ref["source_id"],
        source_ref["due_at"],
        case["target_scope"],
        models.ACTION_KIND_SEND_MESSAGE,
    )
    enriched_scope = dict(case["target_scope"])
    enriched_scope["platform_user_id"] = "qq-target"
    enriched_scope["display_name"] = "Target User"

    enriched_key = tracking.build_idempotency_key(
        source_ref["source_kind"],
        source_ref["source_id"],
        source_ref["due_at"],
        enriched_scope,
        models.ACTION_KIND_SEND_MESSAGE,
    )

    assert enriched_key == stable_key


def test_delivery_target_metadata_is_not_model_visible() -> None:
    case = _topic_followup_case()
    case["target_scope"] = {
        "platform": "qq",
        "platform_channel_id": "54369546",
        "channel_type": "group",
        "user_id": "global-target-1",
        "platform_user_id": "qq-target",
        "display_name": "Target User",
    }

    source_packet = projection.build_source_packet(case)
    rendered_packet = projection.render_source_packet_text(source_packet)

    assert "qq-target" not in rendered_packet
    assert "Target User" not in rendered_packet


def test_classify_route_returns_action_candidate_when_cognition_selects_contact() -> None:
    case = _commitment_case()
    route = tracking.classify_route(
        case,
        _action_cognition_output("Please send the reminder."),
    )

    assert route == models.ROUTE_ACTION_CANDIDATE


def test_classify_route_does_not_use_content_plan_without_speak_action() -> None:
    case = _commitment_case()
    route = tracking.classify_route(
        case,
        {
            "logical_stance": "CONFIRM",
            "character_intent": "PROVIDE",
            "text_surface_output_v2": {
                "schema_version": "text_surface_output.v2",
                "content_plan": "Check whether the user has started work.",
                "content_requirements": ["Ask whether the user has started work."],
                "epistemic_boundary": "Keep the user's current status unknown.",
                "visible_boundaries": [],
                "addressee_plan": [],
                "delivery_profile": {
                    "lexical_register": "plain",
                    "sentence_shape": "brief",
                    "rhythm": "steady",
                    "hesitation": "minimal",
                    "punctuation": "restrained",
                },
                "selected_surface_intent": "observe",
                "permitted_action_results": [],
            },
            "action_specs": [],
        },
    )

    assert route == models.ROUTE_AUDIT_ONLY


def test_classify_route_does_not_use_intent_label_without_speak_or_anchor() -> None:
    case = _commitment_case()
    route = tracking.classify_route(
        case,
        {
            "logical_stance": "CONFIRM",
            "character_intent": "PROVIDE",
            "action_specs": [],
        },
    )

    assert route == models.ROUTE_AUDIT_ONLY


def test_classify_route_does_not_render_private_only_action_specs() -> None:
    case = _commitment_case()
    route = tracking.classify_route(
        case,
        {
            "logical_stance": "CONFIRM",
            "character_intent": "PROVIDE",
            "action_specs": [
                {
                    "kind": "memory_lifecycle_update",
                    "visibility": "private",
                },
                {
                    "kind": "trigger_future_cognition",
                    "visibility": "private",
                },
            ],
        },
    )

    assert route == models.ROUTE_AUDIT_ONLY


def test_classify_route_uses_speak_action_spec_for_visible_candidate() -> None:
    case = _commitment_case()
    route = tracking.classify_route(
        case,
        {
            "logical_stance": "DIVERGE",
            "character_intent": "SILENT_NO_WRITE",
            "action_specs": [_speak_action_spec()],
        },
    )

    assert route == models.ROUTE_ACTION_CANDIDATE


def test_classify_route_does_not_force_action_for_past_due_silence() -> None:
    case = _commitment_case()
    route = tracking.classify_route(case, _silent_cognition_output())

    assert route == models.ROUTE_AUDIT_ONLY


def test_scheduled_canonical_proposal_reaches_action_candidate() -> None:
    case = _scheduled_future_cognition_case()
    output = {
        "schema_version": "cognition_output.v3",
        "response_plan": {
            "self_cognition_response": {
                "decision": "propose_visible_reply",
                "response_goal": "send the approved follow-up",
                "reason": "the due authority permits a grounded reply",
                "cause_summary": "the scheduled source is due",
            },
        },
    }

    assert tracking.classify_route(case, output) == models.ROUTE_ACTION_CANDIDATE


def test_scheduled_canonical_silence_remains_audit_only() -> None:
    case = _scheduled_future_cognition_case()
    output = {
        "schema_version": "cognition_output.v3",
        "response_plan": {
            "self_cognition_response": {
                "decision": "stay_silent",
                "response_goal": "",
                "reason": "the due source does not justify contact",
                "cause_summary": "the scheduled source is due",
            },
        },
    }

    assert tracking.classify_route(case, output) == models.ROUTE_AUDIT_ONLY


def test_scheduled_canonical_proposal_materializes_one_speak_spec(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[dict[str, Any]] = []

    def materialize(
        rows: list[dict[str, Any]],
        state: dict[str, Any],
    ) -> list[dict[str, Any]]:
        del state
        requests.extend(rows)
        return [{"kind": SPEAK_CAPABILITY}]

    monkeypatch.setattr(
        runner,
        "materialize_semantic_action_requests",
        materialize,
    )
    output = {
        "schema_version": "cognition_output.v3",
        "active_character_goal": {"reason": "the due source is grounded"},
        "response_plan": {
            "self_cognition_response": {
                "decision": "propose_visible_reply",
                "response_goal": "send the approved follow-up",
                "reason": "the due authority permits a grounded reply",
                "cause_summary": "the scheduled source is due",
            },
        },
    }
    result = runner._materialize_canonical_self_speak_action(
        {"cognitive_episode": {"trigger_source": "scheduled_tick"}},
        output,
        selected_route=models.ROUTE_ACTION_CANDIDATE,
    )

    assert result["action_specs"] == [{"kind": SPEAK_CAPABILITY}]
    assert requests[0]["capability"] == SPEAK_CAPABILITY


def test_scheduled_canonical_malformed_action_specs_fail_closed() -> None:
    output = {
        "schema_version": "cognition_output.v3",
        "response_plan": {
            "self_cognition_response": {
                "decision": "propose_visible_reply",
                "response_goal": "send the approved follow-up",
                "reason": "the due authority permits a grounded reply",
                "cause_summary": "the scheduled source is due",
            },
        },
        "action_specs": {"kind": SPEAK_CAPABILITY},
    }

    with pytest.raises(StateContractError, match="action_specs"):
        runner._materialize_canonical_self_speak_action(
            {"cognitive_episode": {"trigger_source": "scheduled_tick"}},
            output,
            selected_route=models.ROUTE_ACTION_CANDIDATE,
        )


def test_worker_canonical_result_requires_matching_input_scope() -> None:
    payloads = {
        models.ARTIFACT_COGNITION_INPUT: {"state_scope": "character"},
        models.ARTIFACT_COGNITION_OUTPUT: {
            "schema_version": "cognition_output.v3",
            "state_projection": {"state_scope": "user"},
            "cognition_state_committed": True,
        },
    }

    with pytest.raises(StateContractError, match="does not match cognition input"):
        worker._validate_worker_cognition_result(payloads, required=True)


def test_classify_route_honors_duplicate_action_attempt_state() -> None:
    case = _commitment_case()
    action_attempt = {"status": models.ACTION_ATTEMPT_STATUS_DUPLICATE}

    route = tracking.classify_route(
        case,
        _action_cognition_output("I should check in once."),
        action_attempt=action_attempt,
    )
    silent_route = tracking.classify_route(
        case,
        _silent_cognition_output(),
        action_attempt=action_attempt,
    )

    assert route == models.ROUTE_ACTION_CANDIDATE
    assert silent_route == models.ROUTE_AUDIT_ONLY












def test_consolidation_outcome_reports_incomplete_metadata_contract() -> None:
    """Self-cognition should diagnose incomplete producer metadata."""

    consolidation_state = {
        "cognitive_episode": {
            "trigger_source": "internal_thought",
            "episode_id": "self_cognition:diagnostic",
        },
    }
    consolidation_result = {
        "consolidation_metadata": {
            "write_success": {},
        },
    }

    with pytest.raises(ValueError, match="cache_evicted_count"):
        tracking.build_consolidation_outcome_record(
            consolidation_state,
            consolidation_result,
        )






























def test_duplicate_tick_fixture_supplies_prior_attempt_state() -> None:
    case = _duplicate_tick_case()
    source_ref = case["source_refs"][0]
    expected_key = tracking.build_idempotency_key(
        source_ref["source_kind"],
        source_ref["source_id"],
        source_ref["due_at"],
        case["target_scope"],
        models.ACTION_KIND_SEND_MESSAGE,
    )

    assert case["existing_attempts"][0]["idempotency_key"] == expected_key














def test_scheduled_gate_trace_contains_authority_and_disposition() -> None:
    """Run records bind authority, gate disposition, and dispatch outcome."""

    case = _scheduled_future_cognition_case()
    trigger_record = tracking.build_trigger_record(case)
    run_record = tracking.build_run_record(
        case,
        trigger_record,
        models.ROUTE_ACTION_CANDIDATE,
        {
            "rag_calls": 0,
            "cognition_calls": 1,
            "dialog_calls": 1,
            "topic_limit": models.TOPIC_LIMIT,
        },
    )

    authority = case["scheduled_future_speech_authority"]
    assert trigger_record["scheduled_authority_id"] == (
        authority["authority_id"]
    )
    assert run_record["scheduled_authority_id"] == authority["authority_id"]
    initial_trace = run_record["scheduled_gate_trace"]
    assert initial_trace["gate_disposition"] == "not_evaluated"
    assert initial_trace["dispatch_status"] == ""

    gate_result: models.SelfCognitionScheduledGateResult = {
        "schema_version": models.SCHEDULED_GATE_RESULT_SCHEMA_VERSION,
        "disposition": models.SCHEDULED_GATE_DISPOSITION_SUPPRESSED,
        "gate_codes": ["scheduled_due_not_reached"],
    }
    settled_trace = tracking.build_scheduled_gate_trace(
        case,
        gate_result=gate_result,
        dispatch_status="scheduled_content_suppressed",
    )

    assert settled_trace["authority_id"] == authority["authority_id"]
    assert settled_trace["source_episode_id"] == "episode-future-001"
    assert settled_trace["source_message_id"] == "227312230"
    assert settled_trace["source_action_attempt_id"] == (
        "action_attempt:future-001"
    )
    assert settled_trace["accepted_at_utc"] == authority["accepted_at"]["utc"]
    assert settled_trace["trigger_utc"] == authority["trigger"]["utc"]
    assert settled_trace["gate_disposition"] == "suppressed"
    assert settled_trace["gate_codes"] == ["scheduled_due_not_reached"]
    assert settled_trace["dispatch_status"] == "scheduled_content_suppressed"
