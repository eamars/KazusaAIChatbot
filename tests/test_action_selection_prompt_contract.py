"""Tests for the action-selection route-only output contract."""

from __future__ import annotations


def test_action_selection_prompt_uses_runtime_affordance_roster() -> None:
    """The static prompt should read capability names from JSON affordances."""

    from kazusa_ai_chatbot.cognition_chain_core.action_selection_prompt import (
        ACTION_ROUTER_PROMPT,
        ACTION_ROUTER_TASK_WILLINGNESS_PROMPT,
    )

    for prompt_text in (ACTION_ROUTER_PROMPT, ACTION_ROUTER_TASK_WILLINGNESS_PROMPT):
        assert "capabilities.resolver_affordances" in prompt_text
        assert "capabilities.action_affordances" in prompt_text
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
        "future_speak",
        "background_work_request",
    ):
        assert hardcoded_capability not in ACTION_ROUTER_PROMPT
        assert hardcoded_capability not in ACTION_ROUTER_TASK_WILLINGNESS_PROMPT


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


def test_action_selection_enabled_prompt_follows_task_refusal_outcome() -> None:
    """Enabled L2d prompt should route settled task refusal to visible speech."""

    from kazusa_ai_chatbot.cognition_chain_core.action_selection import (
        build_action_selection_messages,
    )
    from kazusa_ai_chatbot.cognition_chain_core.action_selection_prompt import (
        ACTION_ROUTER_PROMPT,
        ACTION_ROUTER_TASK_WILLINGNESS_PROMPT,
    )

    assert ACTION_ROUTER_TASK_WILLINGNESS_PROMPT != ACTION_ROUTER_PROMPT
    for required_text in (
        '任务承接意愿',
        '不重新判断关系、心情或场景是否允许接下任务',
        '上游已经拒绝、回避、打趣带过或只愿意给更小范围帮助',
        '通常选择可见表面动作',
        '不要选择私有、未来或延迟任务动作',
    ):
        assert required_text in ACTION_ROUTER_TASK_WILLINGNESS_PROMPT

    for forbidden_text in (
        'resource heavy',
        'tool cost',
        'background_work',
        'complex_task_resolution',
        'affinity threshold',
        'effort_score',
        'complexity_score',
        'willingness_score',
        'COGNITION_TASK_WILLINGNESS_BOUNDARY_ENABLED',
    ):
        assert forbidden_text not in ACTION_ROUTER_TASK_WILLINGNESS_PROMPT

    disabled_messages = build_action_selection_messages({})
    enabled_messages = build_action_selection_messages({
        "task_willingness_boundary_enabled": True,
    })

    assert disabled_messages[0].content == ACTION_ROUTER_PROMPT
    assert enabled_messages[0].content == ACTION_ROUTER_TASK_WILLINGNESS_PROMPT
    assert (
        "task_willingness_boundary_enabled"
        not in enabled_messages[1].content
    )


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


def test_action_selection_accepted_task_route_rejects_internals() -> None:
    """An accepted_task_request must not carry executor-facing fields."""

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
                "capability": "accepted_task_request",
                "decision": "accepted_delayed_task",
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

    task_requests = [
        r for r in normalized["semantic_action_requests"]
        if r["capability"] == "accepted_task_request"
    ]
    assert len(task_requests) == 1

    request = task_requests[0]
    for forbidden in ("task_brief", "worker", "task_type", "tool_args"):
        assert forbidden not in request, (
            f"executor-facing field '{forbidden}' must be stripped from "
            "accepted_task_request route output"
        )

    assert request["capability"] == "accepted_task_request"
    assert request["decision"] == "accepted_delayed_task"
    assert "reason" in request
