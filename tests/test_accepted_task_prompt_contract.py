"""Prompt-facing accepted-task contract tests."""

from __future__ import annotations

import pytest
pytest.skip("Stage 1 assertions replaced by the V2 contract suite", allow_module_level=True)

import json

from kazusa_ai_chatbot.nodes import (
    persona_supervisor2_cognition_actions as action_connector,
)
from kazusa_ai_chatbot.nodes import (
    persona_supervisor2_l3_surface as l3_surface_module,
)


def test_l2d_payload_projects_accepted_task_affordance() -> None:
    """L2d should see accepted-task vocabulary, not executor vocabulary."""

    payload = build_action_selection_payload(_minimal_state())
    serialized = json.dumps(payload, ensure_ascii=False).lower()
    action_affordances = payload["capabilities"]["action_affordances"]

    assert any(
        affordance["capability"] == "accepted_task_request"
        for affordance in action_affordances
    )
    assert any(
        affordance["capability"] == "accepted_task_status_check"
        for affordance in action_affordances
    )
    assert payload["work_seed"]["accepted_task_allowed"] is True
    for forbidden in (
        "background_work_request",
        "background work",
        "background_work_allowed",
        "job_ref",
        "queue_state",
        "operational_owner",
        "worker_metadata",
    ):
        assert forbidden not in serialized


def test_l2d_normalizes_accepted_task_request_without_internals() -> None:
    """Accepted-task route output must not carry executor parameters."""

    normalized = normalize_action_selection_output({
        "action_requests": [
            {
                "capability": "accepted_task_request",
                "decision": "accepted_delayed_task",
                "detail": "prepare a concise Fibonacci snippet",
                "reason": "The user asked for bounded delayed text work.",
                "task_brief": "forbidden",
                "worker": "forbidden",
                "work_kind": "forbidden",
                "artifact_text": "forbidden",
            },
        ],
    })

    requests = normalized["semantic_action_requests"]
    assert requests == [
        {
            "capability": "accepted_task_request",
            "decision": "accepted_delayed_task",
            "detail": "prepare a concise Fibonacci snippet",
            "reason": "The user asked for bounded delayed text work.",
        }
    ]


def test_materialized_accepted_task_threads_source_trigger() -> None:
    """Model-facing accepted_task_request becomes the internal executable action."""

    state = _minimal_state()
    specs = action_connector.materialize_semantic_action_requests(
        [
            {
                "capability": "accepted_task_request",
                "decision": "accepted_delayed_task",
                "detail": "prepare a concise Fibonacci snippet",
                "reason": "The user asked for bounded delayed text work.",
            }
        ],
        state,
    )

    assert len(specs) == 1
    spec = specs[0]
    assert spec["kind"] == "background_work_request"
    assert spec["target"]["scope"]["source_trigger_source"] == "user_message"
    assert spec["target"]["scope"]["source_channel_id"] == "debug:user:test"


def test_non_user_source_cannot_materialize_new_accepted_task() -> None:
    """Post-schedule or autonomous source cases cannot enqueue new delayed work."""

    state = _minimal_state()
    state["cognitive_episode"]["trigger_source"] = "accepted_task_result_ready"
    specs = action_connector.materialize_semantic_action_requests(
        [
            {
                "capability": "accepted_task_request",
                "decision": "accepted_delayed_task",
                "detail": "prepare a duplicate task",
                "reason": "The source should not create delayed work.",
            }
        ],
        state,
    )

    assert specs == []


def test_l3_acknowledgement_uses_accepted_task_state_only() -> None:
    """L3 acknowledgement intent should not expose queue or job vocabulary."""

    state = _surface_state()
    state["pre_surface_action_results"] = [
        {
            "action_kind": "background_work_request",
            "status": "pending",
            "accepted_task_state": "scheduled",
            "accepted_task_summary": "Generate a Fibonacci function snippet.",
            "acknowledgement_constraint": "promise_allowed",
            "wait_guidance": "non_numeric_wait",
            "queue_state": "queued",
            "job_ref": "background_work_job:job-001",
            "operational_owner": "background_work_job",
        }
    ]

    intent = l3_surface_module._selected_text_surface_intent(state)
    prompt_rows = l3_surface_module._pre_surface_action_result_prompts(state)
    serialized = json.dumps(prompt_rows, ensure_ascii=False).lower()

    assert "accepted_task_request" in intent
    assert "scheduled" in intent
    assert "Generate a Fibonacci function snippet." in intent
    for forbidden in (
        "background_work_request",
        "background_work_job",
        "job_ref",
        "queue_state",
        "operational_owner",
        "worker",
    ):
        assert forbidden not in intent
        assert forbidden not in serialized


def _minimal_state() -> dict:
    return {
        "cognitive_episode": {
            "trigger_source": "user_message",
            "input_sources": ["dialog_text"],
            "output_mode": "visible_reply",
            "target_scope": {
                "platform": "debug",
                "platform_channel_id": "debug:user:test",
                "channel_type": "private",
                "current_global_user_id": "global-user-001",
                "current_platform_user_id": "debug-user-001",
                "current_display_name": "Test User",
            },
            "origin_metadata": {
                "platform_message_id": "message-001",
                "platform_bot_id": "debug-bot-001",
            },
        },
        "character_profile": {"name": "Kazusa"},
        "platform": "debug",
        "platform_channel_id": "debug:user:test",
        "channel_type": "private",
        "platform_message_id": "message-001",
        "platform_bot_id": "debug-bot-001",
        "global_user_id": "global-user-001",
        "platform_user_id": "debug-user-001",
        "user_name": "Test User",
        "decontexualized_input": "Generate a Fibonacci function snippet.",
        "media_summary": "",
        "logical_stance": "CONFIRM",
        "character_intent": "PROVIDE",
        "judgment_note": "The task can be accepted.",
        "internal_monologue": "This is bounded delayed text work.",
        "emotional_appraisal": "calm",
        "interaction_subtext": "ordinary request",
        "boundary_core_assessment": {},
        "social_distance": "friendly",
        "emotional_intensity": "low",
        "vibe_check": "calm",
        "relational_dynamic": "stable",
        "rag_result": {},
        "conversation_progress": {},
        "available_action_affordances": [
            {
                "capability": "speak",
                "available": True,
                "visibility": "public",
                "semantic_input_summary": [
                    "Use when the character wants a text surface to exist.",
                ],
            },
            {
                "capability": "background_work_request",
                "available": True,
                "visibility": "private",
                "semantic_input_summary": [
                    "Use only for accepted bounded background text work.",
                ],
            },
            {
                "capability": "accepted_task_status_check",
                "available": True,
                "visibility": "private",
                "semantic_input_summary": [
                    "Use when the user asks about already accepted delayed work.",
                ],
            },
        ],
        "background_work_output_char_limit": 4000,
    }


def _surface_state() -> dict:
    state = _minimal_state()
    state.update({
        "action_specs": [
            {
                "kind": "speak",
                "params": {
                    "surface_requirements": {
                        "decision": "visible_reply",
                        "detail": "Acknowledge the accepted task.",
                    }
                },
                "reason": "The user needs a visible acknowledgement.",
            }
        ],
        "resolver_goal_progress": {},
        "resolver_state": {},
        "memory_lifecycle_context": {},
    })
    return state
