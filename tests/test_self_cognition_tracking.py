"""Deterministic self-cognition tracking and record contract tests."""

from __future__ import annotations

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
from kazusa_ai_chatbot.nodes.dialog_agent import StateContractError
from kazusa_ai_chatbot.self_cognition import models, projection
from kazusa_ai_chatbot.self_cognition import sources, tracking
from kazusa_ai_chatbot.self_cognition import runner


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


@pytest.mark.asyncio
async def test_default_self_cognition_client_uses_resolver_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Self-cognition should enter the same cognition resolver loop."""

    captured: dict[str, Any] = {}
    expected_result = {
        "internal_monologue": "resolver completed",
        "action_specs": [],
        "resolver_capability_requests": [],
    }

    async def direct_cognition(_state: dict[str, Any]) -> dict[str, Any]:
        raise AssertionError("self-cognition bypassed the resolver loop")

    async def resolver_loop(
        state: dict[str, Any],
        **kwargs: Any,
    ) -> dict[str, Any]:
        captured["state"] = state
        captured["kwargs"] = kwargs
        return expected_result

    monkeypatch.setattr(runner, "call_cognition_subgraph", direct_cognition)
    monkeypatch.setattr(
        runner,
        "call_cognition_resolver_loop",
        resolver_loop,
        raising=False,
    )
    state = {"cognitive_episode": {"trigger_source": "internal_thought"}}

    result = await runner._default_cognition_client(state)

    assert result == expected_result
    assert captured["state"] is state
    assert captured["kwargs"]["call_cognition_subgraph_func"] is (
        direct_cognition
    )
    assert callable(captured["kwargs"]["execute_capability_func"])
    assert captured["kwargs"]["upsert_pending_resume_func"] is (
        runner._non_persistent_pending_resume
    )
    assert captured["kwargs"]["apply_pending_resolution_func"] is (
        runner._non_persistent_pending_resolution
    )


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
    }
    return case


def _group_chat_review_case() -> dict[str, Any]:
    """Build a group-review case with stable window identity."""

    case = {
        "case_name": models.CASE_GROUP_CHAT_REVIEW,
        "case_id": "group_activity_window:scope_group:2026-05-18T04:00Z",
        "idle_timestamp_utc": "2026-05-18T04:15:00+00:00",
        "last_evidence_timestamp_utc": "2026-05-18T04:10:00+00:00",
        "trigger_kind": models.TRIGGER_GROUP_CHAT_REVIEW,
        "semantic_due_state": None,
        "actionability": "active_group_review_same_channel_no_fallback",
        "target_scope": _target_scope(channel_type="group"),
        "source_refs": [
            {
                "source_kind": "reflection_activity_window",
                "source_id": "scope_group:2026-05-18T04:00Z:2026-05-18T04:15Z",
                "due_at": None,
                "summary": "quiet group activity, one speaker, risk low",
            }
        ],
        "visible_context": [
            {
                "role": "user",
                "text": "A recent group message.",
                "timestamp": "2026-05-18T04:10:00+00:00",
            }
        ],
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
        "source_calendar_run_id": "calendar_run_future_001",
    }
    return case


def _action_cognition_output(text: str) -> dict[str, Any]:
    output = {
        "logical_stance": "CONFIRM",
        "character_intent": "PROVIDE",
        "internal_monologue": "The scheduled follow-up should be visible.",
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
                "content_plan": {
                    "semantic_content": text,
                    "rendering": "One ordinary text message; concise.",
                },
                "forbidden_phrases": [],
            },
        },
        "action_specs": [_speak_action_spec()],
    }
    return output


def _progress_cognition_output() -> dict[str, Any]:
    output = {
        "logical_stance": "maintain awareness without outward contact",
        "character_intent": "keep progress internally visible",
        "action_directives": {
            "linguistic_directives": {
                "content_plan": {
                    "semantic_content": "Track the commitment quietly.",
                },
            },
        },
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
        "action_directives": {"linguistic_directives": {"content_plan": {}}},
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


def _surface_action_directives() -> dict[str, Any]:
    directives = {
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
            "content_plan": {
                "semantic_content": "Continue the GPU model topic.",
                "rendering": "One ordinary text message; concise.",
            },
            "forbidden_phrases": [],
        },
        "visual_directives": {
            "facial_expression": [],
            "body_language": [],
            "gaze_direction": [],
            "visual_vibe": [],
        },
    }
    return directives


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
        "action_directives": {
            "linguistic_directives": {
                "content_plan": {
                    "semantic_content": "Continue the GPU model topic.",
                },
            },
        },
        "action_specs": [_speak_action_spec()],
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
            "action_directives": {
                "linguistic_directives": {
                    "content_plan": {
                        "semantic_content": (
                            "Check whether the user has started work."
                        ),
                    },
                },
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


def test_before_due_commitment_writes_progress_route_without_action_candidate(
    tmp_path,
) -> None:
    case = _commitment_case(
        case_name=models.CASE_COMMITMENT_BEFORE_DUE,
        due_state=models.DUE_STATE_FUTURE_DUE,
    )
    paths = _build_tracking_records(
        case,
        tmp_path,
        cognition_client=lambda state: _progress_cognition_output(),
    )
    route_effect = _read_json(paths[models.ARTIFACT_ROUTE_EFFECT])

    assert route_effect["route"] == models.ROUTE_PROGRESS_MAINTENANCE
    assert models.ARTIFACT_ACTION_ATTEMPT not in paths
    assert models.ARTIFACT_ACTION_CANDIDATE not in paths


def test_past_due_contact_decision_writes_action_attempt_and_candidate_without_handoff(
    tmp_path,
) -> None:
    case = _commitment_case()
    paths = _build_tracking_records(
        case,
        tmp_path,
        cognition_client=lambda state: _action_cognition_output(
            "I noticed the reminder is due; checking in now.",
        ),
        dialog_client=_dialog_client_with_text(
            "I noticed the reminder is due; checking in now.",
        ),
    )
    action_attempt = _read_json(paths[models.ARTIFACT_ACTION_ATTEMPT])
    action_candidate = _read_json(paths[models.ARTIFACT_ACTION_CANDIDATE])
    route_effect = _read_json(paths[models.ARTIFACT_ROUTE_EFFECT])

    assert action_attempt["status"] == models.ACTION_ATTEMPT_STATUS_CANDIDATE
    assert action_candidate["dispatch_shape"] == "send_message"
    assert action_candidate["production_handoff"] is False
    assert "checking in now" in action_candidate["text"]
    assert "without delivery" not in route_effect["effect_summary"]
    assert "runtime adapter bridge" in route_effect["effect_summary"]


def test_build_self_cognition_case_artifacts_does_not_write_files(tmp_path) -> None:
    """Production artifact construction should stay in memory."""

    case = _commitment_case()
    expected_missing_dir = tmp_path / "should_not_exist"

    artifact_payloads = runner.build_self_cognition_case_artifacts(
        case,
        cognition_client=lambda state: _action_cognition_output(
            "I noticed the reminder is due; checking in now.",
        ),
        dialog_client=_dialog_client_with_text(
            "I noticed the reminder is due; checking in now.",
        ),
    )

    assert not expected_missing_dir.exists()
    assert artifact_payloads[models.ARTIFACT_ACTION_ATTEMPT]["status"] == (
        models.ACTION_ATTEMPT_STATUS_CANDIDATE
    )
    assert "checking in now" in (
        artifact_payloads[models.ARTIFACT_ACTION_CANDIDATE]["text"]
    )


def test_runner_apply_consolidation_uses_empty_dialog_without_render() -> None:
    case = _commitment_case()
    captured_consolidation_state: dict[str, Any] = {}

    async def dialog_client(state: dict[str, Any]) -> dict[str, Any]:
        del state
        raise AssertionError("audit-only consolidation should not call dialog")

    async def consolidation_client(state: dict[str, Any]) -> dict[str, Any]:
        captured_consolidation_state.update(state)
        return_value = {
            "consolidation_metadata": {
                "write_success": {
                    "character_state": True,
                    "relationship_insight": True,
                    "user_memory_units": False,
                    "affinity": True,
                    "character_image": False,
                    "cache_invalidation": True,
                },
                "cache_evicted_count": 2,
            },
        }
        return return_value

    artifact_payloads = runner.build_self_cognition_case_artifacts(
        case,
        cognition_client=lambda state: _progress_cognition_output(),
        dialog_client=dialog_client,
        consolidation_client=consolidation_client,
        apply_consolidation=True,
    )

    assert captured_consolidation_state["cognitive_episode"]["trigger_source"] == (
        "internal_thought"
    )
    assert captured_consolidation_state["cognitive_episode"]["output_mode"] == (
        "preview"
    )
    assert captured_consolidation_state["final_dialog"] == []
    assert "The user expected a follow-up" in (
        captured_consolidation_state["decontexualized_input"]
    )
    assert "no_remember" not in captured_consolidation_state["debug_modes"]

    outcome = artifact_payloads[models.ARTIFACT_CONSOLIDATION_OUTCOME]
    assert outcome == {
        "consolidation_called": True,
        "write_success": {
            "character_state": True,
            "relationship_insight": True,
            "user_memory_units": False,
            "affinity": True,
            "character_image": False,
            "cache_invalidation": True,
        },
        "scheduled_event_count": 0,
        "cache_evicted_count": 2,
        "origin_trigger_source": "internal_thought",
        "origin_episode_id": "self_cognition:tracking:commitment_past_due:promise-001",
    }
    serialized = json.dumps(outcome, ensure_ascii=False)
    assert "Reminder was expected" not in serialized


def test_runner_consolidates_no_action_cognition_without_dialog() -> None:
    """Audit-only cognition should consolidate without fabricating dialog."""

    case = _scheduled_future_cognition_case()
    captured_consolidation_state: dict[str, Any] = {}

    async def dialog_client(state: dict[str, Any]) -> dict[str, Any]:
        del state
        raise AssertionError("no-action self-cognition should not call dialog")

    async def consolidation_client(state: dict[str, Any]) -> dict[str, Any]:
        captured_consolidation_state.update(state)
        return_value = {
            "consolidation_metadata": {
                "write_success": {"character_state": True},
                "scheduled_event_ids": [],
                "cache_evicted_count": 0,
            },
        }
        return return_value

    artifact_payloads = runner.build_self_cognition_case_artifacts(
        case,
        cognition_client=lambda state: (
            _audit_only_cognition_output_without_directives()
        ),
        dialog_client=dialog_client,
        consolidation_client=consolidation_client,
        apply_consolidation=True,
    )

    assert captured_consolidation_state["final_dialog"] == []
    assert captured_consolidation_state["logical_stance"] == "DIVERGE"
    assert artifact_payloads[models.ARTIFACT_RUN_RECORD]["budget"][
        "dialog_calls"
    ] == 0
    assert artifact_payloads[models.ARTIFACT_CONSOLIDATION_OUTCOME][
        "consolidation_called"
    ] is True


def test_runner_does_not_call_dialog_for_intent_only_no_speak() -> None:
    case = _commitment_case()
    captured_consolidation_state: dict[str, Any] = {}

    async def dialog_client(state: dict[str, Any]) -> dict[str, Any]:
        del state
        raise AssertionError("intent-only cognition should not call dialog")

    async def consolidation_client(state: dict[str, Any]) -> dict[str, Any]:
        captured_consolidation_state.update(state)
        return_value = {
            "consolidation_metadata": {
                "write_success": {"character_state": True},
                "scheduled_event_ids": [],
                "cache_evicted_count": 0,
            },
        }
        return return_value

    artifact_payloads = runner.build_self_cognition_case_artifacts(
        case,
        cognition_client=lambda state: {
            "logical_stance": "CONFIRM",
            "character_intent": "PROVIDE",
            "internal_monologue": (
                "The scheduled topic is remembered, but no visible reply "
                "was selected."
            ),
            "action_specs": [],
        },
        dialog_client=dialog_client,
        consolidation_client=consolidation_client,
        apply_consolidation=True,
    )

    assert captured_consolidation_state["final_dialog"] == []
    assert models.ARTIFACT_ACTION_ATTEMPT not in artifact_payloads
    assert models.ARTIFACT_ACTION_CANDIDATE not in artifact_payloads
    assert artifact_payloads[models.ARTIFACT_RUN_RECORD][
        "selected_route"
    ] == models.ROUTE_AUDIT_ONLY
    assert artifact_payloads[models.ARTIFACT_RUN_RECORD]["budget"][
        "dialog_calls"
    ] == 0


def test_runner_skips_dialog_for_private_only_actions_without_directives() -> None:
    case = _commitment_case()
    captured_consolidation_state: dict[str, Any] = {}

    async def dialog_client(state: dict[str, Any]) -> dict[str, Any]:
        del state
        raise AssertionError("private-only actions should not call dialog")

    async def consolidation_client(state: dict[str, Any]) -> dict[str, Any]:
        captured_consolidation_state.update(state)
        return_value = {
            "consolidation_metadata": {
                "write_success": {"character_state": True},
                "scheduled_event_ids": [],
                "cache_evicted_count": 0,
            },
        }
        return return_value

    artifact_payloads = runner.build_self_cognition_case_artifacts(
        case,
        cognition_client=lambda state: {
            "logical_stance": "CONFIRM",
            "character_intent": "PROVIDE",
            "internal_monologue": "Only private memory maintenance is selected.",
            "action_specs": [_memory_lifecycle_action_spec()],
        },
        dialog_client=dialog_client,
        consolidation_client=consolidation_client,
        apply_consolidation=True,
    )

    assert captured_consolidation_state["final_dialog"] == []
    assert artifact_payloads[models.ARTIFACT_RUN_RECORD][
        "selected_route"
    ] == models.ROUTE_AUDIT_ONLY
    assert artifact_payloads[models.ARTIFACT_RUN_RECORD]["budget"][
        "dialog_calls"
    ] == 0


def test_runner_rejects_explicit_visible_route_without_speak() -> None:
    case = _commitment_case()

    async def dialog_client(state: dict[str, Any]) -> dict[str, Any]:
        del state
        raise AssertionError("invalid visible route should not call dialog")

    with pytest.raises(StateContractError, match="action_specs.speak"):
        runner.build_self_cognition_case_artifacts(
            case,
            cognition_client=lambda state: {
                "logical_stance": "CONFIRM",
                "character_intent": "PROVIDE",
                "self_cognition_route": models.ROUTE_ACTION_CANDIDATE,
                "action_directives": _surface_action_directives(),
                "action_specs": [],
            },
            dialog_client=dialog_client,
        )


def test_runner_executes_private_lifecycle_action_for_consolidation(
    monkeypatch,
) -> None:
    case = _commitment_case()
    captured_consolidation_state: dict[str, Any] = {}
    captured_specs: list[dict[str, Any]] = []

    async def action_executor(
        action_specs: list[dict[str, Any]],
        *,
        storage_timestamp_utc: str,
        executed_action_attempt_ids: set[str] | None = None,
        record_attempt_func: Any = None,
    ) -> list[dict[str, Any]]:
        del storage_timestamp_utc, executed_action_attempt_ids
        del record_attempt_func
        captured_specs.extend(action_specs)
        action_results = [
            {
                "schema_version": "action_result.v1",
                "action_attempt_id": "action_attempt:memory-unit-001",
                "action_kind": APPLY_MEMORY_LIFECYCLE_UPDATE_CAPABILITY,
                "handler_owner": "memory_lifecycle",
                "status": "executed",
                "visibility": "private",
                "result_summary": (
                    "apply_memory_lifecycle_update executed: cancelled"
                ),
                "result_refs": [],
                "continuation": action_specs[0]["continuation"],
                "completed_at": "2026-05-10T00:30:00+00:00",
            }
        ]
        return action_results

    async def consolidation_client(state: dict[str, Any]) -> dict[str, Any]:
        captured_consolidation_state.update(state)
        return_value = {
            "consolidation_metadata": {
                "write_success": {"character_state": True},
                "scheduled_event_ids": [],
                "cache_evicted_count": 0,
            },
        }
        return return_value

    monkeypatch.setattr(
        runner,
        "execute_action_specs_for_trace",
        action_executor,
    )
    artifact_payloads = runner.build_self_cognition_case_artifacts(
        case,
        cognition_client=lambda state: {
            "logical_stance": "CONFIRM",
            "character_intent": "DISMISS",
            "internal_monologue": "Close the stale commitment privately.",
            "judgment_note": "The commitment should be abandoned.",
            "action_directives": {"linguistic_directives": {"content_plan": {}}},
            "action_specs": [_memory_lifecycle_action_spec()],
        },
        consolidation_client=consolidation_client,
        apply_consolidation=True,
        execute_private_actions=True,
    )
    cognition_output = artifact_payloads[models.ARTIFACT_COGNITION_OUTPUT]

    assert captured_specs[0]["kind"] == APPLY_MEMORY_LIFECYCLE_UPDATE_CAPABILITY
    assert cognition_output["action_results"][0]["status"] == "executed"
    assert cognition_output["episode_trace"]["action_results"][0][
        "action_kind"
    ] == APPLY_MEMORY_LIFECYCLE_UPDATE_CAPABILITY
    assert captured_consolidation_state["episode_trace"]["action_results"][0][
        "status"
    ] == "executed"


def test_runner_routes_lifecycle_intent_through_specialist_before_execution(
    monkeypatch,
) -> None:
    """Self-cognition route intents should become apply actions via specialist."""

    case = _commitment_case()
    captured_specialist_state: dict[str, Any] = {}
    captured_specs: list[dict[str, Any]] = []

    async def specialist_handler(state: dict[str, Any]) -> dict[str, Any]:
        captured_specialist_state.update(state)
        return_value = {
            "action_specs": [_memory_lifecycle_action_spec()],
            "memory_lifecycle_context": {
                "schema_version": "memory_lifecycle_context.v1",
                "source": "memory_lifecycle_specialist",
                "decision": "lifecycle_change",
                "visible_alias_count": 1,
                "omitted_alias_count": 0,
                "lifecycle_decisions": [
                    {
                        "target_alias": "commitment_1",
                        "decision": "abandoned",
                        "role": "avoid_reopening",
                        "evidence_anchor": "The commitment is stale.",
                    }
                ],
                "content_plan_roles": [
                    {
                        "role": "avoid_reopening",
                        "anchor": "Do not reopen the stale commitment.",
                    }
                ],
                "warnings": [],
            },
        }
        return return_value

    async def action_executor(
        action_specs: list[dict[str, Any]],
        *,
        storage_timestamp_utc: str,
        executed_action_attempt_ids: set[str] | None = None,
        record_attempt_func: Any = None,
    ) -> list[dict[str, Any]]:
        del storage_timestamp_utc, executed_action_attempt_ids
        del record_attempt_func
        captured_specs.extend(action_specs)
        action_results = [
            {
                "schema_version": "action_result.v1",
                "action_attempt_id": "action_attempt:memory-unit-001",
                "action_kind": APPLY_MEMORY_LIFECYCLE_UPDATE_CAPABILITY,
                "handler_owner": "memory_lifecycle",
                "status": "executed",
                "visibility": "private",
                "result_summary": (
                    "apply_memory_lifecycle_update executed: cancelled"
                ),
                "result_refs": [],
                "continuation": action_specs[0]["continuation"],
                "completed_at": "2026-05-10T00:30:00+00:00",
            }
        ]
        return action_results

    monkeypatch.setattr(
        runner,
        "call_memory_lifecycle_update_handler",
        specialist_handler,
    )
    monkeypatch.setattr(
        runner,
        "execute_action_specs_for_trace",
        action_executor,
    )
    artifact_payloads = runner.build_self_cognition_case_artifacts(
        case,
        cognition_client=lambda state: {
            "logical_stance": "CONFIRM",
            "character_intent": "DISMISS",
            "internal_monologue": "Review the stale commitment privately.",
            "judgment_note": "The commitment may need lifecycle review.",
            "action_directives": {"linguistic_directives": {"content_plan": {}}},
            "action_specs": [_memory_lifecycle_route_action_spec()],
        },
        apply_consolidation=False,
        execute_private_actions=True,
    )
    cognition_output = artifact_payloads[models.ARTIFACT_COGNITION_OUTPUT]

    assert captured_specialist_state["action_specs"][0][
        "kind"
    ] == MEMORY_LIFECYCLE_UPDATE_CAPABILITY
    assert captured_specs[0]["kind"] == APPLY_MEMORY_LIFECYCLE_UPDATE_CAPABILITY
    assert cognition_output["action_specs"][0][
        "kind"
    ] == APPLY_MEMORY_LIFECYCLE_UPDATE_CAPABILITY
    assert cognition_output["memory_lifecycle_context"]["decision"] == (
        "lifecycle_change"
    )
    assert cognition_output["episode_trace"]["action_specs"][0][
        "kind"
    ] == APPLY_MEMORY_LIFECYCLE_UPDATE_CAPABILITY


def test_runner_does_not_execute_private_actions_by_default(
    monkeypatch,
) -> None:
    """Record builds should not execute private actions unless requested."""

    case = _commitment_case()

    async def action_executor(
        action_specs: list[dict[str, Any]],
        *,
        storage_timestamp_utc: str,
        executed_action_attempt_ids: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        del action_specs, storage_timestamp_utc, executed_action_attempt_ids
        raise AssertionError("default builder must not execute private actions")

    async def consolidation_client(state: dict[str, Any]) -> dict[str, Any]:
        assert "action_results" not in state
        return_value = {
            "consolidation_metadata": {
                "write_success": {"character_state": True},
                "scheduled_event_ids": [],
                "cache_evicted_count": 0,
            },
        }
        return return_value

    monkeypatch.setattr(
        runner,
        "execute_action_specs_for_trace",
        action_executor,
    )
    artifact_payloads = runner.build_self_cognition_case_artifacts(
        case,
        cognition_client=lambda state: {
            "logical_stance": "CONFIRM",
            "character_intent": "DISMISS",
            "internal_monologue": "Close the stale commitment privately.",
            "judgment_note": "The commitment should be abandoned.",
            "action_directives": {"linguistic_directives": {"content_plan": {}}},
            "action_specs": [_memory_lifecycle_action_spec()],
        },
        consolidation_client=consolidation_client,
        apply_consolidation=True,
    )
    cognition_output = artifact_payloads[models.ARTIFACT_COGNITION_OUTPUT]

    assert "action_results" not in cognition_output


def test_runner_reuses_dialog_render_for_action_and_consolidation() -> None:
    case = _commitment_case()
    dialog_call_count = 0
    captured_dialog_state: dict[str, Any] = {}
    captured_consolidation_state: dict[str, Any] = {}

    async def dialog_client(state: dict[str, Any]) -> dict[str, Any]:
        nonlocal dialog_call_count
        dialog_call_count += 1
        captured_dialog_state.update(state)
        return_value = {
            "final_dialog": ["Checking in after the missed promise."],
            "target_addressed_user_ids": [state["global_user_id"]],
            "target_broadcast": False,
        }
        return return_value

    async def consolidation_client(state: dict[str, Any]) -> dict[str, Any]:
        captured_consolidation_state.update(state)
        return_value = {
            "consolidation_metadata": {
                "write_success": {"character_state": True},
                "scheduled_event_ids": [],
                "cache_evicted_count": 0,
            },
        }
        return return_value

    artifact_payloads = runner.build_self_cognition_case_artifacts(
        case,
        cognition_client=lambda state: _action_cognition_output(
            "The commitment is due now.",
        ),
        dialog_client=dialog_client,
        consolidation_client=consolidation_client,
        apply_consolidation=True,
    )

    assert dialog_call_count == 1
    assert captured_dialog_state["dialog_usage_mode"] == (
        "self_cognition_action_candidate_render"
    )
    assert artifact_payloads[models.ARTIFACT_ACTION_CANDIDATE]["text"] == (
        "Checking in after the missed promise."
    )
    assert captured_consolidation_state["final_dialog"] == [
        "Checking in after the missed promise.",
    ]
    assert artifact_payloads[models.ARTIFACT_RUN_RECORD]["budget"][
        "dialog_calls"
    ] == 1


def test_contact_decision_without_candidate_marker_uses_dialog_candidate(
    monkeypatch,
    tmp_path,
) -> None:
    case = _commitment_case()
    l3_states: list[dict[str, Any]] = []

    async def l3_text_surface_handler(state: dict[str, Any]) -> dict[str, Any]:
        l3_states.append(state)
        assert state["action_specs"][0]["kind"] == SPEAK_CAPABILITY
        result = {"action_directives": _surface_action_directives()}
        return result

    async def fake_dialog_client(state: dict[str, Any]) -> dict[str, Any]:
        return_value = {"final_dialog": ['我来确认一下，刚才那个时间已经到了哦。']}
        return return_value

    monkeypatch.setattr(
        runner,
        "call_l3_text_surface_handler",
        l3_text_surface_handler,
        raising=False,
    )
    monkeypatch.setattr(
        runner,
        "_default_dialog_client",
        fake_dialog_client,
    )
    paths = _build_tracking_records(
        case,
        tmp_path,
        cognition_client=lambda state: (
            _speak_cognition_output_with_partial_directives()
        ),
    )
    action_candidate = _read_json(paths[models.ARTIFACT_ACTION_CANDIDATE])
    run_record = _read_json(paths[models.ARTIFACT_RUN_RECORD])

    assert len(l3_states) == 1
    assert action_candidate["production_handoff"] is False
    assert action_candidate["text"] == '我来确认一下，刚才那个时间已经到了哦。'
    assert run_record["budget"]["dialog_calls"] == 1


def test_selected_speak_self_cognition_runs_l3_before_dialog(
    monkeypatch,
    tmp_path,
) -> None:
    case = _scheduled_future_cognition_case()
    l3_states: list[dict[str, Any]] = []
    dialog_states: list[dict[str, Any]] = []
    action_directives = _surface_action_directives()

    async def l3_text_surface_handler(state: dict[str, Any]) -> dict[str, Any]:
        l3_states.append(state)
        result = {"action_directives": action_directives}
        return result

    async def dialog_client(state: dict[str, Any]) -> dict[str, Any]:
        dialog_states.append(state)
        assert state["action_directives"] == action_directives
        result = {
            "final_dialog": ["Continuing the GPU model topic now."],
        }
        return result

    monkeypatch.setattr(
        runner,
        "call_l3_text_surface_handler",
        l3_text_surface_handler,
        raising=False,
    )
    paths = _build_tracking_records(
        case,
        tmp_path,
        cognition_client=lambda state: (
            _speak_cognition_output_with_partial_directives()
        ),
        dialog_client=dialog_client,
    )
    action_candidate = _read_json(paths[models.ARTIFACT_ACTION_CANDIDATE])
    run_record = _read_json(paths[models.ARTIFACT_RUN_RECORD])

    assert len(l3_states) == 1
    assert l3_states[0]["action_specs"][0]["kind"] == SPEAK_CAPABILITY
    assert len(dialog_states) == 1
    assert action_candidate["text"] == "Continuing the GPU model topic now."
    assert run_record["budget"]["dialog_calls"] == 1


def test_dialog_output_without_inline_tag_omits_group_action_delivery_mention(
    monkeypatch,
    tmp_path,
) -> None:
    case = _commitment_case()
    case["target_scope"] = {
        "platform": "qq",
        "platform_channel_id": "54369546",
        "channel_type": "group",
        "user_id": "global-target-1",
        "platform_user_id": "qq-target",
        "display_name": "Target User",
    }

    async def fake_dialog_client(state: dict[str, Any]) -> dict[str, Any]:
        return_value = {
            "final_dialog": ["Checking in after the missed promise."],
        }
        return return_value

    monkeypatch.setattr(
        runner,
        "_default_dialog_client",
        fake_dialog_client,
    )
    paths = _build_tracking_records(
        case,
        tmp_path,
        cognition_client=lambda state: _action_cognition_output(
            "The commitment is due now.",
        ),
    )
    action_candidate = _read_json(paths[models.ARTIFACT_ACTION_CANDIDATE])

    assert action_candidate["text"] == "Checking in after the missed promise."
    assert "delivery_mentions" not in action_candidate


def test_dialog_inline_tag_builds_group_action_delivery_mention(
    monkeypatch,
    tmp_path,
) -> None:
    case = _commitment_case()
    case["target_scope"] = {
        "platform": "qq",
        "platform_channel_id": "54369546",
        "channel_type": "group",
        "user_id": "global-target-1",
        "platform_user_id": "qq-target",
        "display_name": "Target User",
    }

    async def fake_dialog_client(state: dict[str, Any]) -> dict[str, Any]:
        return_value = {
            "final_dialog": ["@Target User Checking in after the missed promise."],
        }
        return return_value

    monkeypatch.setattr(
        runner,
        "_default_dialog_client",
        fake_dialog_client,
    )
    paths = _build_tracking_records(
        case,
        tmp_path,
        cognition_client=lambda state: _action_cognition_output(
            "The commitment is due now.",
        ),
    )
    action_candidate = _read_json(paths[models.ARTIFACT_ACTION_CANDIDATE])

    assert action_candidate["delivery_mentions"] == [
        {
            "entity_kind": "user",
            "platform_user_id": "qq-target",
            "display_name": "Target User",
        }
    ]


def test_duplicate_contact_decision_suppresses_same_due_occurrence(
    tmp_path,
) -> None:
    case = _duplicate_tick_case()
    paths = _build_tracking_records(
        case,
        tmp_path,
        cognition_client=lambda state: _action_cognition_output(
            "I should check again.",
        ),
    )
    action_attempt = _read_json(paths[models.ARTIFACT_ACTION_ATTEMPT])

    assert action_attempt["status"] == models.ACTION_ATTEMPT_STATUS_DUPLICATE
    assert models.ARTIFACT_ACTION_CANDIDATE not in paths


def test_duplicate_contact_decision_suppresses_active_prior_attempt_statuses(
    tmp_path,
) -> None:
    suppressing_statuses = [
        models.ACTION_ATTEMPT_STATUS_CANDIDATE,
        models.ACTION_ATTEMPT_STATUS_HELD,
        models.ACTION_ATTEMPT_STATUS_PENDING_HANDOFF,
        models.ACTION_ATTEMPT_STATUS_HANDOFF_ACCEPTED,
        models.ACTION_ATTEMPT_STATUS_SCHEDULED,
        models.ACTION_ATTEMPT_STATUS_SENT,
        models.ACTION_ATTEMPT_STATUS_DUPLICATE,
    ]

    for prior_status in suppressing_statuses:
        case = _commitment_case(
            case_name=f"{models.CASE_COMMITMENT_DUPLICATE_TICK}:{prior_status}",
        )
        case["case_name"] = models.CASE_COMMITMENT_DUPLICATE_TICK
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
                "attempt_id": f"self_cognition_attempt:{prior_status}",
                "idempotency_key": idempotency_key,
                "status": prior_status,
            }
        ]

        unused_fixture_path = tmp_path / prior_status
        paths = _build_tracking_records(
            case,
            unused_fixture_path,
            cognition_client=lambda state: _action_cognition_output(
                "I should check again.",
            ),
        )
        action_attempt = _read_json(paths[models.ARTIFACT_ACTION_ATTEMPT])

        assert action_attempt["status"] == models.ACTION_ATTEMPT_STATUS_DUPLICATE
        assert models.ARTIFACT_ACTION_CANDIDATE not in paths


def test_group_review_suppresses_prior_delivery_failed_attempt(
    tmp_path,
) -> None:
    """A failed send for one group window should not repeat visible speech."""

    case = _group_chat_review_case()
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
            "attempt_id": "self_cognition_attempt:group-delivery-failed",
            "idempotency_key": idempotency_key,
            "status": models.ACTION_ATTEMPT_STATUS_DELIVERY_FAILED,
        }
    ]

    paths = _build_tracking_records(
        case,
        tmp_path,
        cognition_client=lambda state: _action_cognition_output(
            "A visible group response.",
        ),
    )
    action_attempt = _read_json(paths[models.ARTIFACT_ACTION_ATTEMPT])

    assert action_attempt["status"] == models.ACTION_ATTEMPT_STATUS_DUPLICATE
    assert models.ARTIFACT_ACTION_CANDIDATE not in paths


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


def test_group_noise_rejected_without_rag_or_action(tmp_path) -> None:
    case = _group_noise_case()

    def reject_cognition(state: dict[str, Any]) -> dict[str, Any]:
        raise AssertionError("group noise should not call cognition")

    paths = _build_tracking_records(
        case,
        tmp_path,
        cognition_client=reject_cognition,
    )
    route_effect = _read_json(paths[models.ARTIFACT_ROUTE_EFFECT])

    assert route_effect["route"] == models.ROUTE_AUDIT_ONLY
    assert models.ARTIFACT_ACTION_ATTEMPT not in paths
    assert models.ARTIFACT_ACTION_CANDIDATE not in paths


def test_group_chat_review_starts_without_preloaded_rag(
    tmp_path,
) -> None:
    """Group review source hydration must not preload retrieval evidence."""

    case = _group_chat_review_case()

    paths = _build_tracking_records(
        case,
        tmp_path,
        cognition_client=lambda state: _silent_cognition_output(),
    )
    trigger_record = _read_json(paths[models.ARTIFACT_TRIGGER_RECORD])
    run_record = _read_json(paths[models.ARTIFACT_RUN_RECORD])

    assert run_record["budget"]["rag_calls"] == 0
    assert trigger_record["target_scope"]["channel_type"] == "group"
    assert trigger_record["target_scope"]["user_id"] is None


def test_topic_followup_contact_decision_writes_action_candidate(
    tmp_path,
) -> None:
    case = _topic_followup_case()

    paths = _build_tracking_records(
        case,
        tmp_path,
        cognition_client=lambda state: _action_cognition_output(
            "Want to continue the GraphRAG thread?",
        ),
        dialog_client=_dialog_client_with_text(
            "Want to continue the GraphRAG thread?",
        ),
    )
    action_attempt = _read_json(paths[models.ARTIFACT_ACTION_ATTEMPT])
    action_candidate = _read_json(paths[models.ARTIFACT_ACTION_CANDIDATE])

    assert action_attempt["status"] == models.ACTION_ATTEMPT_STATUS_CANDIDATE
    assert action_candidate["text"] == "Want to continue the GraphRAG thread?"


def test_scheduled_future_cognition_starts_without_preloaded_rag(
    tmp_path,
) -> None:
    """Scheduled follow-up cognition lets resolver decide retrieval needs."""

    case = _scheduled_future_cognition_case()

    def cognition_client(state: dict[str, Any]) -> dict[str, Any]:
        assert state["rag_result"]["answer"] == ""
        assert "user_image" in state["rag_result"]
        assert "character_image" in state["rag_result"]
        assert state["rag_result"]["user_image"]["user_memory_context"][
            "active_commitments"
        ] == []
        output = _silent_cognition_output()
        output["resolver_state"] = {
            "observations": [
                {
                    "capability_kind": "local_context_recall",
                }
            ]
        }
        return output

    paths = _build_tracking_records(
        case,
        tmp_path,
        cognition_client=cognition_client,
    )
    run_record = _read_json(paths[models.ARTIFACT_RUN_RECORD])

    assert run_record["budget"]["rag_calls"] == 1


def test_cognition_state_keeps_source_packet_inside_internal_percept(
    tmp_path,
) -> None:
    case = _commitment_case()
    captured: dict[str, Any] = {}

    def capture_state(state: dict[str, Any]) -> dict[str, Any]:
        captured.update(state)
        return _silent_cognition_output()

    paths = _build_tracking_records(
        case,
        tmp_path,
        cognition_client=capture_state,
    )
    cognition_input = _read_json(paths[models.ARTIFACT_COGNITION_INPUT])
    rendered_text = cognition_input["rendered_text"]
    percept_content = captured["cognitive_episode"]["percepts"][0]["content"]
    percept_payload = json.loads(percept_content)

    assert captured["prompt_message_context"]["body_text"] == (
        models.SELF_COGNITION_INPUT_TEXT
    )
    assert captured["decontexualized_input"] == models.SELF_COGNITION_INPUT_TEXT
    assert rendered_text not in captured["prompt_message_context"]["body_text"]
    assert rendered_text not in captured["decontexualized_input"]
    assert percept_payload["residue"]["internal_monologue"] == rendered_text


def test_cognition_state_disables_visual_and_does_not_suppress_memory(
    tmp_path,
) -> None:
    case = _commitment_case()
    captured: dict[str, Any] = {}

    def capture_state(state: dict[str, Any]) -> dict[str, Any]:
        captured.update(state)
        return _silent_cognition_output()

    _build_tracking_records(
        case,
        tmp_path,
        cognition_client=capture_state,
    )

    state_debug_modes = captured["debug_modes"]
    episode_debug_modes = captured["cognitive_episode"]["origin_metadata"][
        "debug_modes"
    ]

    assert state_debug_modes == {"no_visual_directives": True}
    assert episode_debug_modes == {"no_visual_directives": True}
    assert "no_remember" not in state_debug_modes
    assert "no_remember" not in episode_debug_modes
