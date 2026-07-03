"""Tests for core-owned semantic action selection contracts."""

import json


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
                "capability": "accepted_task_request",
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
                "capability": "accepted_task_request",
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


def test_task_willingness_refusal_routes_visible_speak_only() -> None:
    """A settled upstream refusal should stay a visible route-only action."""

    from kazusa_ai_chatbot.cognition_chain_core.action_selection import (
        build_action_selection_messages,
        normalize_action_selection_output,
    )
    from kazusa_ai_chatbot.cognition_chain_core.action_selection_prompt import (
        ACTION_ROUTER_TASK_WILLINGNESS_PROMPT,
    )

    state = _task_willingness_refusal_state()
    messages = build_action_selection_messages(state)
    human_payload = json.loads(messages[1].content)

    assert messages[0].content == ACTION_ROUTER_TASK_WILLINGNESS_PROMPT
    assert human_payload["cognition"]["logical_stance"] == "REFUSE"
    assert human_payload["cognition"]["character_intent"] == "REJECT"
    assert "task_willingness_boundary_enabled" not in messages[1].content

    raw_output = {
        "resolver_capability_requests": [],
        "action_requests": [{
            "capability": "speak",
            "decision": "visible_refusal",
            "detail": "可见地拒绝接下这个持续请求，并给出更小范围回应",
            "reason": "上游已经决定现在不接下这个任务请求",
        }],
    }
    normalized = normalize_action_selection_output(raw_output)

    assert normalized["semantic_action_requests"] == [{
        "capability": "speak",
        "decision": "visible_refusal",
        "detail": "可见地拒绝接下这个持续请求，并给出更小范围回应",
        "reason": "上游已经决定现在不接下这个任务请求",
    }]
    assert all(
        request["capability"] != "accepted_task_request"
        for request in normalized["semantic_action_requests"]
    )


def _task_willingness_refusal_state() -> dict[str, object]:
    """Build a prompt-safe L2d state for an upstream task refusal."""

    return {
        "task_willingness_boundary_enabled": True,
        "cognitive_episode": {
            "trigger_source": "user_message",
            "input_sources": ["dialog_text"],
            "output_mode": "live_response",
        },
        "channel_type": "direct",
        "decontexualized_input": "帮我之后一直盯着这件事，整理好了再告诉我。",
        "media_summary": "",
        "logical_stance": "REFUSE",
        "character_intent": "REJECT",
        "judgment_note": "关系和当前气氛都不适合接下持续后续，只愿意给很小范围提醒。",
        "internal_monologue": "这要求太像把后续都丢给我了。",
        "emotional_appraisal": "被催着承担后续，有点抗拒。",
        "interaction_subtext": "对方想让我持续替他处理。",
        "boundary_core_assessment": {
            "boundary_issue": "mixed",
            "boundary_summary": "请求越过当前关系分寸。",
            "acceptance": "reject",
            "stance_bias": "refuse",
            "pressure_policy": "resist",
        },
        "social_distance": "distant",
        "emotional_intensity": "medium",
        "vibe_check": "有点紧绷",
        "relational_dynamic": "低熟悉度",
        "rag_result": {},
        "conversation_progress": {},
        "resolver_context": "",
        "background_work_output_char_limit": 4000,
        "available_action_affordances": [
            {
                "capability": "speak",
                "available": True,
                "visibility": "public",
                "semantic_input_summary": "可见回复当前用户。",
                "output_kind": "semantic_action_request",
            },
            {
                "capability": "accepted_task_request",
                "available": True,
                "visibility": "private",
                "semantic_input_summary": "已接受后才创建延迟文字任务。",
                "output_kind": "semantic_action_request",
            },
            {
                "capability": "future_speak",
                "available": True,
                "visibility": "private",
                "semantic_input_summary": "等待具体未来信息后再发言。",
                "output_kind": "semantic_action_request",
            },
        ],
    }
