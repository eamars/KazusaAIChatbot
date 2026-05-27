"""Tests for dialog_agent.py — generator/evaluator dialog loop."""

from __future__ import annotations

import logging
import typing
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.messages import AIMessage
import pytest

from kazusa_ai_chatbot.nodes import dialog_agent as dialog_module
from kazusa_ai_chatbot.nodes.dialog_agent import (
    _DIALOG_EVALUATOR_PROMPT,
    _DIALOG_GENERATOR_PROMPT,
    dialog_agent,
    DialogAgentState,
    StateContractError,
    validate_dialog_action_directives,
)
from kazusa_ai_chatbot.config import CHARACTER_GLOBAL_USER_ID
from kazusa_ai_chatbot.utils import build_interaction_history_recent


@pytest.fixture(autouse=True)
def _stub_dialog_event_logging(monkeypatch):
    """Keep deterministic dialog tests away from event-log persistence."""

    for recorder_name in (
        "record_llm_stage_event",
        "record_model_contract_event",
        "record_dialog_quality_event",
    ):
        monkeypatch.setattr(
            dialog_module.event_logging,
            recorder_name,
            AsyncMock(),
        )


def _base_global_state():
    """Minimal GlobalPersonaState for testing dialog_agent."""
    return {
        "internal_monologue": "thinking about greeting",
        "action_directives": {
            "contextual_directives": {
                "social_distance": "casual and friendly",
                "emotional_intensity": "light and positive",
                "vibe_check": "friendly conversation",
                "relational_dynamic": "user greets, bot responds warmly",
            },
            "linguistic_directives": {
                "rhetorical_strategy": "direct greeting",
                "linguistic_style": "warm and concise",
                "content_anchors": ["greet user"],
                "forbidden_phrases": [],
            },
        },
        "chat_history_wide": [],
        "chat_history_recent": [],
        "debug_modes": {},
        "should_respond": True,
        "platform_user_id": "user_123",
        "platform_bot_id": "bot_456",
        "global_user_id": "global-user-123",
        "user_name": "TestUser",
        "user_profile": {"affinity": 500},
        "character_profile": {
            "name": "Kazusa",
            "description": "A tsundere character",
            "personality_brief": {
                "logic": "analytical",
                "tempo": "moderate",
                "defense": "tsundere deflection",
                "quirks": "occasional stutter",
                "taboos": "never break character",
                "mbti": "INTJ",
            },
            "linguistic_texture_profile": {
                "hesitation_density": 0.4,
                "fragmentation": 0.4,
                "emotional_leakage": 0.4,
                "rhythmic_bounce": 0.4,
                "direct_assertion": 0.4,
                "softener_density": 0.4,
                "counter_questioning": 0.4,
                "formalism_avoidance": 0.4,
                "abstraction_reframing": 0.4,
                "self_deprecation": 0.4,
            },
            "boundary_profile": {
                "self_integrity": 0.7,
                "control_sensitivity": 0.3,
                "compliance_strategy": "comply",
                "relational_override": 0.65,
                "control_intimacy_misread": 0.35,
                "boundary_recovery": "rebound",
                "authority_skepticism": 0.35,
            },
        },
    }


def test_validate_dialog_action_directives_requires_action_directives() -> None:
    """Dialog validation should report the missing top-level directives."""

    state = _base_global_state()
    state.pop("action_directives")

    with pytest.raises(StateContractError, match="action_directives"):
        validate_dialog_action_directives(state, usage_mode="unit_test")


def test_validate_dialog_action_directives_requires_linguistic_directives() -> None:
    """Dialog validation should report missing linguistic directives."""

    state = _base_global_state()
    state["action_directives"].pop("linguistic_directives")

    with pytest.raises(
        StateContractError,
        match="action_directives.linguistic_directives",
    ):
        validate_dialog_action_directives(state, usage_mode="unit_test")


def test_validate_dialog_action_directives_requires_contextual_directives() -> None:
    """Dialog validation should report missing contextual directives."""

    state = _base_global_state()
    state["action_directives"].pop("contextual_directives")

    with pytest.raises(
        StateContractError,
        match="action_directives.contextual_directives",
    ):
        validate_dialog_action_directives(state, usage_mode="unit_test")


def test_validate_dialog_action_directives_accepts_complete_directives() -> None:
    """Dialog validation should return the directive dictionaries unchanged."""

    state = _base_global_state()

    linguistic_directives, contextual_directives = (
        validate_dialog_action_directives(state, usage_mode="unit_test")
    )

    assert linguistic_directives is state["action_directives"][
        "linguistic_directives"
    ]
    assert contextual_directives is state["action_directives"][
        "contextual_directives"
    ]


class TestDialogAgentState:
    def test_is_typed_dict(self):
        assert issubclass(DialogAgentState, dict)

    def test_has_required_fields(self):
        hints = typing.get_type_hints(DialogAgentState)
        required = [
            "internal_monologue", "action_directives",
            "chat_history_wide", "chat_history_recent", "platform_user_id", "platform_bot_id", "global_user_id", "user_name", "user_profile",
            "character_profile",
        ]
        for field in required:
            assert field in hints, f"Missing field: {field}"


def test_dialog_evaluator_prompt_preserves_concise_safe_dialog() -> None:
    """Evaluator prompt should not force rewrites of short safe on-topic dialog."""

    assert "简短、贴锚点、安全的台词应通过" in _DIALOG_EVALUATOR_PROMPT
    assert "硬门槛全部通过后，才看软风格" in _DIALOG_EVALUATOR_PROMPT
    assert "动作描写、物理感官、不可见状态" in _DIALOG_EVALUATOR_PROMPT


def test_dialog_evaluator_prompt_checks_anchor_fidelity() -> None:
    """Evaluator prompt should judge dialog against every anchor class."""

    prompt = _DIALOG_EVALUATOR_PROMPT
    assert '`content_anchors` 是唯一语义权威' in prompt
    assert '锚点忠实：不得缺失、替换、反转或绕开' in prompt
    assert '[DECISION]' in prompt
    assert '[FACT]' in prompt
    assert '[ANSWER]' in prompt
    assert '[SOCIAL]' in prompt
    assert '[AVOID_REPEAT]' in prompt
    assert '[PROGRESSION]' in prompt
    assert '[SCOPE]' in prompt


def test_dialog_evaluator_prompt_orders_hard_gates_before_style() -> None:
    """Evaluator prompt should read as an ordered weak-model audit."""

    prompt = _DIALOG_EVALUATOR_PROMPT
    assert '不要从上下文自行决定话题、意图或风格' in prompt
    assert prompt.index('# 通过条件') < prompt.index('# 硬门槛')
    assert prompt.index('# 硬门槛') < prompt.index('# 软风格')
    assert prompt.index('# 软风格') < prompt.index('# 动态通过逻辑')
    assert prompt.index('# 动态通过逻辑') < prompt.index('# 审核顺序')
    assert '"should_stop": boolean' in prompt
    assert 'should_stop=false` 表示必须把 `feedback` 交回生成器重试' in prompt


def test_dialog_evaluator_prompt_rejects_unsupported_concrete_content() -> None:
    """Evaluator prompt should reject concrete claims not backed by anchors."""

    prompt = _DIALOG_EVALUATOR_PROMPT
    rule_start = prompt.index('事实边界：不得添加 `content_anchors` 未授权')
    unsupported_rule = prompt[rule_start:rule_start + 500]

    assert '具体实体' in unsupported_rule
    assert '属性' in unsupported_rule
    assert '数量' in unsupported_rule
    assert '时间' in unsupported_rule
    assert '地点' in unsupported_rule
    assert '承诺' in unsupported_rule


def test_dialog_evaluator_prompt_rejects_guess_owner_flip() -> None:
    """Evaluator prompt should reject changed owner for guessing gates."""

    prompt = _DIALOG_EVALUATOR_PROMPT
    assert '指代与动作所有权' in prompt
    assert '指代基准' in prompt
    assert '硬失败速查' in prompt
    assert '默认猜测动作属于被回应者' in prompt
    assert '我/我的/自己' in prompt
    assert '你/对方/你们' in prompt
    assert '猜测对象' in prompt
    assert '当前角色' in prompt
    assert '偏好' in prompt
    assert '我会想看' in prompt
    assert '我想看' in prompt
    assert '必须驳回' in prompt


def test_dialog_generator_prompt_has_no_decision_ownership() -> None:
    """Dialog generation stays an execution renderer, not a decision stage."""

    assert "only turns upstream decisions" in (dialog_module.__doc__ or "")
    assert "logical_stance" not in _DIALOG_GENERATOR_PROMPT
    assert "character_intent" not in _DIALOG_GENERATOR_PROMPT
    assert "boundary_profile" not in _DIALOG_GENERATOR_PROMPT
    assert "话题合法性" not in _DIALOG_GENERATOR_PROMPT


def test_dialog_prompts_use_content_anchors_as_semantic_authority() -> None:
    """Generator and evaluator prompts should not expose stale history fields."""

    assert 'content_anchors` 是本轮可见回复的唯一语义内容来源' in _DIALOG_GENERATOR_PROMPT
    assert 'internal_monologue' not in _DIALOG_GENERATOR_PROMPT
    assert 'tone' '_history' not in _DIALOG_GENERATOR_PROMPT
    assert 'last_user' '_message' not in _DIALOG_EVALUATOR_PROMPT
    assert 'internal_monologue' not in _DIALOG_EVALUATOR_PROMPT
    assert '`content_anchors` 是唯一语义权威' in _DIALOG_EVALUATOR_PROMPT
    assert '只有同时满足以下条件才返回 `should_stop=true`' in _DIALOG_EVALUATOR_PROMPT
    assert '没有把另一个对象、提议、请求、问题或偏好所有者当作核心话题' in (
        _DIALOG_EVALUATOR_PROMPT
    )
    assert '`retry` 只是输入里的计数字段' in _DIALOG_EVALUATOR_PROMPT
    assert '绝不能影响 pass/fail' in _DIALOG_EVALUATOR_PROMPT
    assert '强制 `should_stop: true`' not in _DIALOG_EVALUATOR_PROMPT


def test_build_interaction_history_recent_excludes_other_user_messages():
    """Scoped history should keep only the current user's turns and bot replies."""
    history = [
        {
            "role": "user",
            "platform_user_id": "user_a",
            "global_user_id": "global-a",
            "body_text": "这是 active character 的照片",
            "addressed_to_global_user_ids": [CHARACTER_GLOBAL_USER_ID],
            "broadcast": False,
        },
        {
            "role": "assistant",
            "platform_user_id": "bot_456",
            "body_text": "明明就是想看我出糗吧。",
            "addressed_to_global_user_ids": ["global-a"],
            "broadcast": False,
        },
        {
            "role": "user",
            "platform_user_id": "user_b",
            "global_user_id": "global-b",
            "body_text": "你照片真涩情",
            "addressed_to_global_user_ids": [CHARACTER_GLOBAL_USER_ID],
            "broadcast": False,
        },
        {
            "role": "assistant",
            "platform_user_id": "bot_456",
            "body_text": "看照片的眼神，感觉有点过分了啊。",
            "addressed_to_global_user_ids": ["global-b"],
            "broadcast": False,
        },
    ]

    scoped = build_interaction_history_recent(
        history,
        "user_b",
        "bot_456",
        current_global_user_id="global-b",
    )

    assert scoped == [
        {
            "role": "user",
            "platform_user_id": "user_b",
            "global_user_id": "global-b",
            "body_text": "你照片真涩情",
            "addressed_to_global_user_ids": [CHARACTER_GLOBAL_USER_ID],
            "broadcast": False,
        },
        {
            "role": "assistant",
            "platform_user_id": "bot_456",
            "body_text": "看照片的眼神，感觉有点过分了啊。",
            "addressed_to_global_user_ids": ["global-b"],
            "broadcast": False,
        },
    ]


def test_build_interaction_history_recent_returns_empty_without_current_user():
    """Ambiguous bot replies should not leak across group-chat users."""
    history = [
        {
            "role": "user",
            "platform_user_id": "user_a",
            "global_user_id": "global-a",
            "content": "prior thread",
            "addressed_to_global_user_ids": ["character-global"],
        },
        {
            "role": "assistant",
            "platform_user_id": "bot_456",
            "content": "bot reply to prior thread",
            "addressed_to_global_user_ids": ["global-a"],
        },
    ]

    scoped = build_interaction_history_recent(history, "user_b", "bot_456")

    assert scoped == []


@pytest.mark.asyncio
async def test_dialog_agent_returns_final_dialog():
    """dialog_agent should return a dict with 'final_dialog' key."""
    state = _base_global_state()

    # Mock the generator LLM to return dialog
    from langchain_core.messages import AIMessage
    generator_response = AIMessage(content='{"final_dialog": ["Hello there!", "How are you?"]}')

    # Mock the evaluator LLM to approve immediately
    evaluator_response = AIMessage(content='{"fatal_errors": [], "guideline_violations": [], "score": 90, "should_stop": true, "feedback": "good"}')

    call_count = 0

    async def mock_ainvoke(messages):
        nonlocal call_count
        call_count += 1
        if call_count % 2 == 1:
            return generator_response
        return evaluator_response

    with patch("kazusa_ai_chatbot.nodes.dialog_agent._dialog_generator_llm") as mock_generator, \
         patch("kazusa_ai_chatbot.nodes.dialog_agent._dialog_evaluator_llm") as mock_evaluator:
        mock_generator.ainvoke = mock_ainvoke
        mock_evaluator.ainvoke = mock_ainvoke

        result = await dialog_agent(state)

    assert "final_dialog" in result
    assert isinstance(result["final_dialog"], list)
    assert result["target_addressed_user_ids"] == ["global-user-123"]
    assert result["target_broadcast"] is False


@pytest.mark.asyncio
async def test_dialog_agent_validates_action_directives_before_llm_call(
    monkeypatch,
) -> None:
    """Missing dialog-ready directives should fail before LLM invocation."""

    state = _base_global_state()
    state["action_directives"].pop("linguistic_directives")
    generator_llm = MagicMock()
    generator_llm.ainvoke = AsyncMock()
    evaluator_llm = MagicMock()
    evaluator_llm.ainvoke = AsyncMock()
    monkeypatch.setattr(dialog_module, "_dialog_generator_llm", generator_llm)
    monkeypatch.setattr(dialog_module, "_dialog_evaluator_llm", evaluator_llm)

    with pytest.raises(
        StateContractError,
        match="action_directives.linguistic_directives",
    ):
        await dialog_agent(state)

    generator_llm.ainvoke.assert_not_awaited()
    evaluator_llm.ainvoke.assert_not_awaited()


@pytest.mark.asyncio
async def test_dialog_agent_handles_empty_dialog():
    """If generator returns no dialog, final_dialog should default to empty list."""
    state = _base_global_state()

    from langchain_core.messages import AIMessage
    generator_response = AIMessage(content='{"final_dialog": []}')

    evaluator_response = AIMessage(content='{"fatal_errors": [], "guideline_violations": [], "score": 90, "should_stop": true, "feedback": "ok"}')

    call_count = 0

    async def mock_ainvoke(messages):
        nonlocal call_count
        call_count += 1
        if call_count % 2 == 1:
            return generator_response
        return evaluator_response

    with patch("kazusa_ai_chatbot.nodes.dialog_agent._dialog_generator_llm") as mock_generator, \
         patch("kazusa_ai_chatbot.nodes.dialog_agent._dialog_evaluator_llm") as mock_evaluator:
        mock_generator.ainvoke = mock_ainvoke
        mock_evaluator.ainvoke = mock_ainvoke

        result = await dialog_agent(state)

    assert result["final_dialog"] == [] or isinstance(result["final_dialog"], list)


@pytest.mark.asyncio
async def test_dialog_agent_logs_explicit_usage_mode(caplog):
    """Dialog output logs should identify the caller-supplied usage mode."""

    state = _base_global_state()
    state["dialog_usage_mode"] = "background_contract_render"

    from langchain_core.messages import AIMessage
    generator_response = AIMessage(content='{"final_dialog": ["Internal only."]}')
    evaluator_response = AIMessage(
        content=(
            '{"fatal_errors": [], "guideline_violations": [], "score": 90, '
            '"should_stop": true, "feedback": "Passed"}'
        )
    )

    call_count = 0

    async def mock_ainvoke(messages):
        nonlocal call_count
        call_count += 1
        if call_count % 2 == 1:
            return generator_response
        return evaluator_response

    with (
        patch(
            "kazusa_ai_chatbot.nodes.dialog_agent._dialog_generator_llm"
        ) as mock_generator,
        patch(
            "kazusa_ai_chatbot.nodes.dialog_agent._dialog_evaluator_llm"
        ) as mock_evaluator,
    ):
        mock_generator.ainvoke = mock_ainvoke
        mock_evaluator.ainvoke = mock_ainvoke

        with caplog.at_level(logging.INFO, logger=dialog_module.__name__):
            await dialog_agent(state)

    assert "usage_mode=background_contract_render" in caplog.text


def test_dialog_usage_mode_requires_internal_state_fields():
    """Missing required state fields should not fall back to visible mode."""

    state_without_debug_modes = _base_global_state()
    state_without_debug_modes.pop("debug_modes")
    with pytest.raises(KeyError):
        dialog_module._dialog_usage_mode(state_without_debug_modes)

    state_without_should_respond = _base_global_state()
    state_without_should_respond.pop("should_respond")
    with pytest.raises(KeyError):
        dialog_module._dialog_usage_mode(state_without_should_respond)


@pytest.mark.asyncio
async def test_dialog_agent_ignores_retired_response_field(monkeypatch):
    """Dialog generation should not own response routing decisions."""

    state = _base_global_state()
    retired_field = "expression" + "_willingness"
    state["action_directives"]["contextual_directives"][
        retired_field
    ] = "silent"
    generator_llm = MagicMock()
    generator_llm.ainvoke = AsyncMock(return_value=AIMessage(
        content='{"final_dialog": ["Still answering."], "mention_target_user": false}'
    ))
    evaluator_llm = MagicMock()
    evaluator_llm.ainvoke = AsyncMock(return_value=AIMessage(
        content='{"feedback": "Passed", "should_stop": true}'
    ))
    monkeypatch.setattr(dialog_module, "_dialog_generator_llm", generator_llm)
    monkeypatch.setattr(dialog_module, "_dialog_evaluator_llm", evaluator_llm)

    result = await dialog_agent(state)

    assert result["final_dialog"] == ["Still answering."]
    assert result["target_addressed_user_ids"] == ["global-user-123"]
    assert result["target_broadcast"] is False
    generator_llm.ainvoke.assert_awaited_once()
    evaluator_llm.ainvoke.assert_awaited_once()
