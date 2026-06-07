"""Tests for the action-router route-only output contract."""

from __future__ import annotations


def test_action_router_prompt_uses_runtime_affordance_roster() -> None:
    """The static prompt should read capability names from JSON affordances."""

    from kazusa_ai_chatbot.action_router.prompt import ACTION_ROUTER_PROMPT

    assert "capabilities.resolver_affordances" in ACTION_ROUTER_PROMPT
    assert "capabilities.action_affordances" in ACTION_ROUTER_PROMPT
    for hardcoded_capability in (
        "rag_evidence",
        "web_evidence",
        "human_clarification",
        "approval_preparation",
        "self_goal_resolution",
        "speak",
        "memory_lifecycle_update",
        "trigger_future_cognition",
        "background_work_request",
    ):
        assert hardcoded_capability not in ACTION_ROUTER_PROMPT


def test_action_router_normalizes_schema_free_resolver_requests() -> None:
    """Resolver requests should receive trusted schema metadata after routing."""

    from kazusa_ai_chatbot.action_router.contracts import (
        normalize_action_router_output,
    )

    raw_model_output = {
        "resolver_capability_requests": [
            {
                "capability_kind": "rag_evidence",
                "objective": "find Fibonacci background",
                "reason": "user asked about Fibonacci",
                "priority": "now",
                "schema_version": "resolver_request.v1",
                "pending_row_id": "abc-123",
            },
        ],
        "resolver_pending_resolution": None,
        "resolver_goal_progress": None,
        "action_requests": [
            {
                "capability": "speak",
                "decision": "visible_reply",
                "detail": "answer the question",
                "reason": "direct user question",
            },
        ],
    }

    normalized = normalize_action_router_output(raw_model_output)

    requests = normalized["resolver_capability_requests"]
    assert len(requests) == 1
    request = requests[0]
    assert request["schema_version"] == "resolver_capability_request.v1"
    assert request["capability_kind"] == "rag_evidence"
    assert request["objective"] == "find Fibonacci background"
    assert "pending_row_id" not in request


def test_action_router_background_work_route_rejects_task_brief() -> None:
    """A background_work_request action must not carry task_brief, worker,
    task_type, or other worker-facing fields from the router output."""

    from kazusa_ai_chatbot.action_router.contracts import (
        normalize_action_router_output,
    )

    raw_model_output = {
        "resolver_capability_requests": [],
        "resolver_pending_resolution": None,
        "resolver_goal_progress": None,
        "action_requests": [
            {
                "capability": "speak",
                "decision": "visible_reply",
                "detail": "acknowledge the request",
                "reason": "user expects a reply",
            },
            {
                "capability": "background_work_request",
                "decision": "queue_async_work",
                "detail": "accepted bounded text work",
                "reason": "user asked for a code snippet",
                "task_brief": "Generate a Fibonacci function.",
                "worker": "text_artifact",
                "task_type": "coding_snippet",
                "tool_args": {"language": "python"},
            },
        ],
    }

    normalized = normalize_action_router_output(raw_model_output)

    bw_requests = [
        r for r in normalized["action_requests"]
        if r["capability"] == "background_work_request"
    ]
    assert len(bw_requests) == 1

    bw = bw_requests[0]
    for forbidden in ("task_brief", "worker", "task_type", "tool_args"):
        assert forbidden not in bw, (
            f"worker-facing field '{forbidden}' must be stripped from "
            "background_work_request route output"
        )

    assert bw["capability"] == "background_work_request"
    assert bw["decision"] == "queue_async_work"
    assert "reason" in bw
