"""Executable tests for prompt-safe DSH task affordance boundaries."""

from __future__ import annotations

import pytest


def _episode() -> dict[str, object]:
    return {
        "trigger_source": "user_message",
        "target_scope": {"channel_type": "private"},
    }


def _state() -> dict[str, object]:
    return {
        "platform": "debug",
        "platform_channel_id": "channel-1",
        "channel_type": "private",
        "platform_user_id": "platform-user-1",
        "global_user_id": "user-1",
        "user_name": "User",
        "platform_bot_id": "bot-1",
        "cognitive_episode": _episode(),
        "rag_result": {},
        "action_availability_runtime": {
            "worker_status": {
                "accepted_task": "healthy",
                "background_work": "healthy",
            },
        },
        "action_selection_context": {
            "dsh_tasks": [{
                "schema_version": "dsh_accepted_task_affordance.v1",
                "accepted_task_ref": "accepted_task:task-1",
                "task_state": "terminal",
                "objective_summary": "Inspect the project",
                "latest_summary": "The task has a bounded result.",
                "allowed_next_actions": ["continue", "summarize", "cancel"],
                "followup_open": True,
                "updated_at": "2026-08-30T00:00:00Z",
            }],
        },
    }


@pytest.mark.asyncio
async def test_persona_graph_loads_only_scoped_active_or_open_followup_affordances(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The graph loads DSH rows through the exact current source scope."""

    from kazusa_ai_chatbot.nodes import persona_supervisor2 as owner

    calls: list[dict[str, object]] = []

    async def load_rows(**kwargs: object) -> list[dict[str, object]]:
        calls.append(dict(kwargs))
        return [{
            "schema_version": "dsh_accepted_task_affordance.v1",
            "accepted_task_ref": "accepted_task:task-1",
            "task_state": "terminal",
            "objective_summary": "Inspect the project",
            "latest_summary": "The task has a bounded result.",
            "allowed_next_actions": ["continue", "summarize", "cancel"],
            "followup_open": True,
            "updated_at": "2026-08-30T00:00:00Z",
        }]

    monkeypatch.setattr(owner, "load_open_dsh_task_affordances_for_scope", load_rows)
    projected = await owner._load_live_action_selection_context(_state())

    assert calls == [{
        "platform": "debug",
        "source_channel_id": "channel-1",
        "requester_global_user_id": "user-1",
        "limit": 3,
    }]
    context = projected["action_selection_context"]
    assert context == {
        "dsh_tasks": [{
            "schema_version": "dsh_accepted_task_affordance.v1",
            "accepted_task_ref": "accepted_task:task-1",
            "task_state": "terminal",
            "objective_summary": "Inspect the project",
            "latest_summary": "The task has a bounded result.",
            "allowed_next_actions": ["continue", "summarize", "cancel"],
            "followup_open": True,
            "updated_at": "2026-08-30T00:00:00Z",
        }],
    }


def test_action_selection_context_has_closed_accepted_task_shape() -> None:
    """The action-selection validator accepts only the closed DSH shape."""

    from kazusa_ai_chatbot.nodes import persona_supervisor2_schema as schema

    validator = getattr(schema, "validate_dsh_task_action_selection_context", None)
    assert callable(validator)
    task = {
        "schema_version": "dsh_accepted_task_affordance.v1",
        "accepted_task_ref": "accepted_task:task-1",
        "task_state": "active",
        "objective_summary": "Inspect the project",
        "latest_summary": "In progress.",
        "allowed_next_actions": ["cancel"],
        "followup_open": False,
        "updated_at": "2026-08-30T00:00:00Z",
    }
    context = {"dsh_tasks": [task]}
    assert validator(context) == context
    with pytest.raises((TypeError, ValueError)):
        validator({"dsh_tasks": [{**task, "session_id": "hidden"}]})


def test_model_selected_control_binds_only_trusted_advertised_task_ref() -> None:
    """The model selects a typed control; deterministic code never classifies text."""

    from kazusa_ai_chatbot.nodes import persona_supervisor2_cognition_actions as owner

    request = {
        "capability": "accepted_task_control",
        "decision": "continue",
        "detail": "Please cancel the run if the evidence is stale.",
        "reason": "The user asked for a bounded follow-up.",
        "context_ref": "accepted_task:task-1",
        "target_roles": [{"role": "target", "entity_kind": "user", "entity_id": "user-1"}],
        "evidence_handles": [],
        "surface_role": "task_acknowledgement",
        "goal_continuation_ref": None,
    }
    materialized = owner.materialize_semantic_action_requests([request], _state())

    assert len(materialized) == 1
    control = materialized[0]
    assert control["kind"] == "accepted_task_control"
    assert control["params"] == {
        "control": {
            "schema_version": "accepted_task_control.v1",
            "accepted_task_ref": "accepted_task:task-1",
            "operation": "continue",
            "instruction": request["detail"],
        },
    }
    assert "user_text" not in control
