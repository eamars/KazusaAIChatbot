"""Tests for the action-selection route-only output contract."""

from __future__ import annotations


def test_action_selection_prompt_uses_runtime_affordance_roster() -> None:
    """The static prompt should read capability names from JSON affordances."""

    from kazusa_ai_chatbot.cognition_chain_core.action_selection_prompt import (
        ACTION_ROUTER_PROMPT,
    )

    assert "capabilities.resolver_affordances" in ACTION_ROUTER_PROMPT
    assert "capabilities.action_affordances" in ACTION_ROUTER_PROMPT
    for hardcoded_capability in (
        "public_answer_research",
        "local_context_recall",
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


def test_action_selection_prompt_explains_upstream_handoff() -> None:
    """L2d prompt should read upstream judgment instead of re-deciding it."""

    from kazusa_ai_chatbot.cognition_chain_core.action_selection_prompt import (
        ACTION_ROUTER_PROMPT,
    )

    required_explanations = (
        "前序 cognition 已经负责理解当前材料、形成立场、意图和边界判断",
        "本层不重新裁决事实、立场、关系压力、是否该回复",
        "本层对所有 trigger_source 都先读取上游判断",
        "只判断上游判断是否已经形成需要动作层处理的语义目标",
        "source、current_input、evidence、resolver 和可选上下文字段只用于解释",
        "如果上游判断表达旁观、保持距离、无需接话、只是观察",
        "不要用空字符串、空 deliverables 或占位 note 填充这个对象",
        "不适用的可选对象必须省略",
    )
    for explanation in required_explanations:
        assert explanation in ACTION_ROUTER_PROMPT

    assert "群聊自省" not in ACTION_ROUTER_PROMPT
    assert "group_engagement" not in ACTION_ROUTER_PROMPT
    assert "group_scene_digest" not in ACTION_ROUTER_PROMPT
    assert "群聊话题" not in ACTION_ROUTER_PROMPT


def test_action_selection_normalizes_schema_free_resolver_requests() -> None:
    """Resolver requests should receive trusted schema metadata after routing."""

    from kazusa_ai_chatbot.cognition_chain_core.action_selection import (
        normalize_action_selection_output,
    )

    raw_model_output = {
        "resolver_capability_requests": [
            {
                "capability_kind": "local_context_recall",
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

    normalized = normalize_action_selection_output(raw_model_output)

    requests = normalized["resolver_capability_requests"]
    assert len(requests) == 1
    request = requests[0]
    assert request["schema_version"] == "resolver_capability_request.v1"
    assert request["capability_kind"] == "local_context_recall"
    assert request["objective"] == "find Fibonacci background"
    assert "pending_row_id" not in request


def test_action_selection_background_work_route_rejects_task_brief() -> None:
    """A background_work_request action must not carry task_brief, worker,
    task_type, or other worker-facing fields from the router output."""

    from kazusa_ai_chatbot.cognition_chain_core.action_selection import (
        normalize_action_selection_output,
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

    normalized = normalize_action_selection_output(raw_model_output)

    bw_requests = [
        r for r in normalized["semantic_action_requests"]
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
