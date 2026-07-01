"""Tests for core-owned semantic action selection contracts."""


def test_semantic_action_request_contract_matches_current_l2d_routes() -> None:
    from kazusa_ai_chatbot.cognition_chain_core.action_selection import (
        normalize_action_selection_output,
    )

    raw_output = {
        "resolver_capability_requests": [{
            "schema_version": "model-supplied",
            "capability_kind": "local_context_recall",
            "objective": "find prior context",
            "reason": "memory is needed",
            "priority": "now",
            "pending_row_id": "forbidden",
        }],
        "resolver_pending_resolution": {
            "schema_version": "model-supplied",
            "decision": "answered",
            "reason": "user provided the missing fact",
            "pending_row_id": "forbidden",
        },
        "resolver_goal_progress": {
            "schema_version": "model-supplied",
            "original_goal": "help with the plan",
            "current_focus": "answer",
        },
        "action_requests": [
            {
                "capability": "speak",
                "decision": "visible_reply",
                "detail": "answer the user",
                "reason": "enough context",
            },
            {
                "capability": "background_work_request",
                "decision": "background_task",
                "detail": "prepare a concise note",
                "reason": "accepted private work",
                "task_brief": "forbidden worker-local task",
                "worker": "forbidden worker",
            },
        ],
    }

    normalized = normalize_action_selection_output(raw_output)

    assert "action_requests" not in normalized
    assert normalized["semantic_action_requests"] == [
        {
            "capability": "speak",
            "decision": "visible_reply",
            "detail": "answer the user",
            "reason": "enough context",
        },
        {
            "capability": "background_work_request",
            "decision": "background_task",
            "detail": "prepare a concise note",
            "reason": "accepted private work",
        },
    ]
    resolver_request = normalized["resolver_capability_requests"][0]
    assert resolver_request["capability_kind"] == "local_context_recall"
    assert "pending_row_id" not in resolver_request
    assert "schema_version" in resolver_request
    assert "pending_row_id" not in normalized["resolver_pending_resolution"]
    assert "schema_version" not in normalized["resolver_goal_progress"]
