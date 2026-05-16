"""Deterministic self-cognition tracking and artifact contract tests."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from kazusa_ai_chatbot.self_cognition import artifacts, models, projection
from kazusa_ai_chatbot.self_cognition import sources, tracking
from kazusa_ai_chatbot.self_cognition import runner
from kazusa_ai_chatbot.self_cognition.runner import run_self_cognition_case

from tests.test_config import _configured_subprocess_env_without_dotenv


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
        "idle_timestamp": "2026-05-10T00:30:00+00:00",
        "last_evidence_timestamp": "2026-05-10T00:00:00+00:00",
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
        "idle_timestamp": "2026-05-10T00:30:00+00:00",
        "last_evidence_timestamp": "2026-05-10T00:29:00+00:00",
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


def _topic_followup_case() -> dict[str, Any]:
    case = {
        "case_name": models.CASE_TOPIC_RAG_FOLLOWUP,
        "case_id": "topic-followup-001",
        "idle_timestamp": "2026-05-10T00:30:00+00:00",
        "last_evidence_timestamp": "2026-05-10T00:00:00+00:00",
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
        "rag_query": "Find evidence for the open technical follow-up topic.",
    }
    return case


def _action_cognition_output(text: str) -> dict[str, Any]:
    output = {
        "logical_stance": "outward contact is appropriate",
        "character_intent": "send a concise follow-up",
        "action_directives": {
            "linguistic_directives": {
                "content_anchors": [f"[ACTION_CANDIDATE] {text}"],
            },
        },
    }
    return output


def _progress_cognition_output() -> dict[str, Any]:
    output = {
        "logical_stance": "maintain awareness without outward contact",
        "character_intent": "keep progress internally visible",
        "action_directives": {
            "linguistic_directives": {
                "content_anchors": [
                    "[PROGRESS_MAINTENANCE] Track the commitment quietly.",
                ],
            },
        },
    }
    return output


def _silent_cognition_output() -> dict[str, Any]:
    output = {
        "logical_stance": "no outward contact is warranted",
        "character_intent": "stay silent",
        "self_cognition_route": models.ROUTE_AUDIT_ONLY,
        "action_directives": {"linguistic_directives": {"content_anchors": []}},
    }
    return output


def _read_json(path: str | Path) -> dict[str, Any]:
    content = Path(path).read_text(encoding="utf-8")
    data = json.loads(content)
    return data


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


def test_group_action_candidate_omits_delivery_mention_without_dialog_flag() -> None:
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


def test_group_action_candidate_carries_dialog_delivery_mention_request() -> None:
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
        mention_target_user=True,
    )

    assert action_candidate is not None
    assert action_candidate["delivery_mentions"] == [
        {
            "entity_kind": "user",
            "placement": "prefix",
            "platform_user_id": "qq-target",
            "global_user_id": "global-target-1",
            "display_name": "Target User",
            "requested_by": "dialog.mention_target_user",
        }
    ]


def test_private_action_candidate_keeps_dialog_mention_request_for_adapter_noop(
) -> None:
    case = _commitment_case()
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
        mention_target_user=True,
    )

    assert action_candidate is not None
    assert action_candidate["delivery_mentions"] == [
        {
            "entity_kind": "user",
            "placement": "prefix",
            "platform_user_id": None,
            "global_user_id": "673225019",
            "display_name": "",
            "requested_by": "dialog.mention_target_user",
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
    rag_request = projection.build_rag_request(case)
    rendered_rag_request = json.dumps(rag_request, ensure_ascii=False)

    assert "qq-target" not in rendered_packet
    assert "Target User" not in rendered_packet
    assert "qq-target" not in rendered_rag_request
    assert "Target User" not in rendered_rag_request


def test_classify_route_returns_action_candidate_when_cognition_selects_contact() -> None:
    case = _commitment_case()
    route = tracking.classify_route(
        case,
        _action_cognition_output("Please send the reminder."),
    )

    assert route == models.ROUTE_ACTION_CANDIDATE


def test_classify_route_returns_action_candidate_when_cognition_intent_provides() -> None:
    case = _commitment_case()
    route = tracking.classify_route(
        case,
        {
            "logical_stance": "CONFIRM",
            "character_intent": "PROVIDE",
            "action_directives": {
                "linguistic_directives": {
                    "content_anchors": [
                        "[ANSWER] Check whether the user has started work.",
                    ],
                },
            },
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
    paths = run_self_cognition_case(
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
    paths = run_self_cognition_case(
        case,
        tmp_path,
        cognition_client=lambda state: _action_cognition_output(
            "I noticed the reminder is due; checking in now.",
        ),
    )
    action_attempt = _read_json(paths[models.ARTIFACT_ACTION_ATTEMPT])
    action_candidate = _read_json(paths[models.ARTIFACT_ACTION_CANDIDATE])

    assert action_attempt["status"] == models.ACTION_ATTEMPT_STATUS_CANDIDATE
    assert action_candidate["dispatch_shape"] == "send_message"
    assert action_candidate["production_handoff"] is False
    assert "checking in now" in action_candidate["text"]


def test_build_self_cognition_case_artifacts_does_not_write_files(tmp_path) -> None:
    """Production artifact construction should stay in memory."""

    case = _commitment_case()
    output_dir = tmp_path / "should_not_exist"

    artifact_payloads = runner.build_self_cognition_case_artifacts(
        case,
        cognition_client=lambda state: _action_cognition_output(
            "I noticed the reminder is due; checking in now.",
        ),
    )

    assert not output_dir.exists()
    assert artifact_payloads[models.ARTIFACT_ACTION_ATTEMPT]["status"] == (
        models.ACTION_ATTEMPT_STATUS_CANDIDATE
    )
    assert "checking in now" in (
        artifact_payloads[models.ARTIFACT_ACTION_CANDIDATE]["text"]
    )


def test_runner_apply_consolidation_builds_private_finalization_state() -> None:
    case = _commitment_case()
    captured_dialog_state: dict[str, Any] = {}
    captured_consolidation_state: dict[str, Any] = {}

    async def dialog_client(state: dict[str, Any]) -> dict[str, Any]:
        captured_dialog_state.update(state)
        return_value = {
            "final_dialog": ["Private finalization for consolidation only."],
            "target_addressed_user_ids": [state["global_user_id"]],
            "target_broadcast": False,
        }
        return return_value

    async def consolidation_client(state: dict[str, Any]) -> dict[str, Any]:
        captured_consolidation_state.update(state)
        return_value = {
            "consolidation_metadata": {
                "write_success": {
                    "character_state": True,
                    "relationship_insight": True,
                    "user_memory_units": False,
                    "task_dispatch": True,
                    "affinity": True,
                    "character_image": False,
                    "cache_invalidation": True,
                },
                "scheduled_event_ids": ["scheduled-1"],
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

    assert captured_dialog_state["cognitive_episode"]["trigger_source"] == (
        "internal_thought"
    )
    assert captured_dialog_state["dialog_usage_mode"] == (
        "self_cognition_private_finalization"
    )
    assert captured_consolidation_state["cognitive_episode"]["trigger_source"] == (
        "internal_thought"
    )
    assert captured_consolidation_state["cognitive_episode"]["output_mode"] == (
        "preview"
    )
    assert captured_consolidation_state["final_dialog"] == [
        "Private finalization for consolidation only.",
    ]
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
            "task_dispatch": True,
            "affinity": True,
            "character_image": False,
            "cache_invalidation": True,
        },
        "scheduled_event_count": 1,
        "cache_evicted_count": 2,
        "origin_trigger_source": "internal_thought",
        "origin_episode_id": "self_cognition:dry_run:commitment_past_due:promise-001",
    }
    serialized = json.dumps(outcome, ensure_ascii=False)
    assert "Private finalization" not in serialized
    assert "Reminder was expected" not in serialized


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
        cognition_client=lambda state: {
            "logical_stance": "CONFIRM",
            "character_intent": "PROVIDE",
            "action_directives": {
                "contextual_directives": {
                },
                "linguistic_directives": {
                    "content_anchors": [
                        "[ANSWER] The commitment is due now.",
                    ],
                },
            },
        },
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

    async def fake_dialog_client(state: dict[str, Any]) -> dict[str, Any]:
        return_value = {"final_dialog": ['我来确认一下，刚才那个时间已经到了哦。']}
        return return_value

    monkeypatch.setattr(
        runner,
        "_default_dialog_client",
        fake_dialog_client,
    )
    paths = run_self_cognition_case(
        case,
        tmp_path,
        cognition_client=lambda state: {
            "logical_stance": "CONFIRM",
            "character_intent": "PROVIDE",
            "action_directives": {
                "linguistic_directives": {
                    "content_anchors": ["[ANSWER] The commitment is due now."],
                },
            },
        },
    )
    action_candidate = _read_json(paths[models.ARTIFACT_ACTION_CANDIDATE])
    run_record = _read_json(paths[models.ARTIFACT_RUN_RECORD])

    assert action_candidate["production_handoff"] is False
    assert action_candidate["text"] == '我来确认一下，刚才那个时间已经到了哦。'
    assert run_record["budget"]["dialog_calls"] == 1


def test_dialog_false_mention_flag_suppresses_group_action_delivery_mention(
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
            "mention_target_user": False,
        }
        return return_value

    monkeypatch.setattr(
        runner,
        "_default_dialog_client",
        fake_dialog_client,
    )
    paths = run_self_cognition_case(
        case,
        tmp_path,
        cognition_client=lambda state: {
            "logical_stance": "CONFIRM",
            "character_intent": "PROVIDE",
            "action_directives": {
                "linguistic_directives": {
                    "content_anchors": ["[ANSWER] The commitment is due now."],
                },
            },
        },
    )
    action_candidate = _read_json(paths[models.ARTIFACT_ACTION_CANDIDATE])

    assert action_candidate["text"] == "Checking in after the missed promise."
    assert "delivery_mentions" not in action_candidate


def test_dialog_true_mention_flag_builds_group_action_delivery_mention(
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
            "mention_target_user": True,
        }
        return return_value

    monkeypatch.setattr(
        runner,
        "_default_dialog_client",
        fake_dialog_client,
    )
    paths = run_self_cognition_case(
        case,
        tmp_path,
        cognition_client=lambda state: {
            "logical_stance": "CONFIRM",
            "character_intent": "PROVIDE",
            "action_directives": {
                "linguistic_directives": {
                    "content_anchors": ["[ANSWER] The commitment is due now."],
                },
            },
        },
    )
    action_candidate = _read_json(paths[models.ARTIFACT_ACTION_CANDIDATE])

    assert action_candidate["delivery_mentions"] == [
        {
            "entity_kind": "user",
            "placement": "prefix",
            "platform_user_id": "qq-target",
            "global_user_id": "global-target-1",
            "display_name": "Target User",
            "requested_by": "dialog.mention_target_user",
        }
    ]


def test_duplicate_contact_decision_suppresses_same_due_occurrence(
    tmp_path,
) -> None:
    case = _duplicate_tick_case()
    paths = run_self_cognition_case(
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

        output_dir = tmp_path / prior_status
        paths = run_self_cognition_case(
            case,
            output_dir,
            cognition_client=lambda state: _action_cognition_output(
                "I should check again.",
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

    def reject_rag(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise AssertionError("group noise should not call RAG")

    def reject_cognition(state: dict[str, Any]) -> dict[str, Any]:
        raise AssertionError("group noise should not call cognition")

    paths = run_self_cognition_case(
        case,
        tmp_path,
        rag_client=reject_rag,
        cognition_client=reject_cognition,
    )
    route_effect = _read_json(paths[models.ARTIFACT_ROUTE_EFFECT])

    assert route_effect["route"] == models.ROUTE_AUDIT_ONLY
    assert models.ARTIFACT_RAG_REQUEST not in paths
    assert models.ARTIFACT_RAG_OUTPUT not in paths
    assert models.ARTIFACT_ACTION_ATTEMPT not in paths
    assert models.ARTIFACT_ACTION_CANDIDATE not in paths


def test_topic_followup_contact_decision_writes_action_candidate(
    tmp_path,
) -> None:
    case = _topic_followup_case()

    def rag_client(
        _query: str,
        *,
        character_name: str,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        assert isinstance(character_name, str)
        assert "prompt_message_context" in context
        result = {
            "answer": "Relevant evidence supports a concise follow-up.",
            "known_facts": [],
            "unknown_slots": [],
            "loop_count": 0,
        }
        return result

    paths = run_self_cognition_case(
        case,
        tmp_path,
        rag_client=rag_client,
        cognition_client=lambda state: _action_cognition_output(
            "Want to continue the GraphRAG thread?",
        ),
    )
    action_attempt = _read_json(paths[models.ARTIFACT_ACTION_ATTEMPT])
    action_candidate = _read_json(paths[models.ARTIFACT_ACTION_CANDIDATE])

    assert action_attempt["status"] == models.ACTION_ATTEMPT_STATUS_CANDIDATE
    assert action_candidate["text"] == "Want to continue the GraphRAG thread?"


def test_cognition_state_keeps_source_packet_inside_internal_percept(
    tmp_path,
) -> None:
    case = _commitment_case()
    captured: dict[str, Any] = {}

    def capture_state(state: dict[str, Any]) -> dict[str, Any]:
        captured.update(state)
        return _silent_cognition_output()

    paths = run_self_cognition_case(
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

    run_self_cognition_case(
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


def test_artifact_writer_uses_expected_file_names(tmp_path) -> None:
    payloads = {
        models.ARTIFACT_TRIGGER_RECORD: {"trigger_id": "trigger-001"},
        models.ARTIFACT_LOOP_TRACE: "trace body",
    }

    paths = artifacts.write_tracking_artifacts(tmp_path, payloads)

    assert set(paths) == set(payloads)
    assert Path(paths[models.ARTIFACT_TRIGGER_RECORD]).name == (
        models.ARTIFACT_TRIGGER_RECORD
    )
    assert Path(paths[models.ARTIFACT_LOOP_TRACE]).read_text(
        encoding="utf-8",
    ) == "trace body"


def test_dry_run_command_rejects_unknown_case_name(tmp_path) -> None:
    case_file = tmp_path / "unknown_case.json"
    output_dir = tmp_path / "dry_run"
    case_file.write_text(
        json.dumps({"case_name": "unsupported_case"}),
        encoding="utf-8",
    )
    env = _configured_subprocess_env_without_dotenv()
    src_path = os.path.abspath("src")
    env["PYTHONPATH"] = src_path

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.run_self_cognition_dry_run",
            "--case-file",
            str(case_file),
            "--output-dir",
            str(output_dir),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "unsupported self-cognition case" in result.stderr
    assert not output_dir.exists()
