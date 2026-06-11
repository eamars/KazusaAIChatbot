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
                "content_plan": {
                    "semantic_content": "Greet the user warmly.",
                    "rendering": "One visible chat bubble; concise.",
                },
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
    """Dialog validation should return normalized directive dictionaries."""

    state = _base_global_state()

    linguistic_directives, contextual_directives = (
        validate_dialog_action_directives(state, usage_mode="unit_test")
    )

    assert linguistic_directives["content_plan"] == {
        "semantic_content": "Greet the user warmly.",
        "rendering": "One visible chat bubble; concise.",
    }
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

    assert "简短、贴计划、安全的台词应通过" in _DIALOG_EVALUATOR_PROMPT
    assert "硬门槛全部通过后，才看软风格" in _DIALOG_EVALUATOR_PROMPT
    assert "动作描写、物理感官、不可见状态" in _DIALOG_EVALUATOR_PROMPT


def test_dialog_evaluator_prompt_checks_content_plan_fidelity() -> None:
    """Evaluator prompt should judge dialog against one content plan."""

    prompt = _DIALOG_EVALUATOR_PROMPT
    assert '`content_plan` 是本轮可见回复的语义计划' in prompt
    assert '`semantic_content` 如果存在' in prompt
    assert '计划忠实：不得缺失、替换、反转或绕开' in prompt
    assert '不要把每个字段机械当成独立可见段落' in prompt
    assert '[DECISION]' not in prompt
    assert '[ANSWER]' not in prompt
    assert '[SCOPE]' not in prompt


def test_dialog_prompts_preserve_multi_part_deliverables() -> None:
    """Dialog prompts should preserve L3 content without literal overfitting."""

    generator_prompt = _DIALOG_GENERATOR_PROMPT
    evaluator_prompt = _DIALOG_EVALUATOR_PROMPT

    for required_text in (
        '# 生成流程',
        '# 核心转换',
        '# 角色表达依据',
        '这些字段只决定“怎么说”',
        '性格底色怎样影响台词',
        '声纹质感怎样影响台词',
        '最终文字采用当前角色对当前用户说话的视角',
        '先把叙述句、分析句和泛称感受句转成角色台词',
        '角色表达依据只能作用在已经重锚定好的台词上',
        '建立内容骨架',
        '重锚定说话视角',
        '按内容类型渲染',
        '选择角色表达',
        '组织单气泡布局',
        '多项步骤、候选、风险、对比或时间段',
        '覆盖主要项',
        '不能改变事实或造成歧义',
        '不丢失主要信息、不制造计划外结论',
        '固定格式块必须作为 `final_dialog` 数组里的字符串元素输出',
        '外层回复仍然必须是合法 JSON 对象',
        '保持同义且不矛盾',
        '裸 JSON 对象',
    ):
        assert required_text in generator_prompt

    for forbidden_text in (
        '# 角色底色',
        '# 角色声纹约束',
        '**hesitation_density:**',
        '**fragmentation:**',
        '**direct_assertion:**',
        'GB300',
        'Pro6000',
    ):
        assert forbidden_text not in generator_prompt

    for required_text in (
        '# 审核流程',
        '建立语义计划',
        '对照可见气泡',
        '审核单气泡布局和固定格式块',
        '审核表达安全',
        '最后看软风格',
        '语义是否忠实',
        '允许自然改写、轻微解释和布局调整',
        '轻微解释不矛盾时可以通过',
        '压缩表达可以通过',
        '自然表达可以通过',
        '说话视角忠实',
        '先扫说话视角',
        '泛称感受扫描',
        '只有同时满足以下条件才返回 `should_stop=true`',
        'should_stop=false',
    ):
        assert required_text in evaluator_prompt


def test_dialog_generator_prompt_describes_one_bubble_layout_contract() -> None:
    """Generator prompt should describe final_dialog as one visible bubble."""

    prompt = _DIALOG_GENERATOR_PROMPT

    assert '一个聊天气泡' in prompt
    assert '运行时会用换行连接 `final_dialog`' in prompt
    assert '布局单位' in prompt
    assert '每个元素写可见文字或一个完整固定格式块' in prompt
    assert '停顿用标点和句子节奏表达' in prompt
    assert '回答从左花括号开始，以右花括号结束' in prompt
    assert '输出前自检' in prompt
    assert '平台分别发送' not in prompt
    assert '打一段、发一段' not in prompt
    assert '要发送的台词片段' not in prompt
    assert '使用 6-12 个短字符串片段是允许的' not in prompt


def test_dialog_prompts_preserve_fixed_format_blocks() -> None:
    """Dialog prompts should preserve code and fixed-format block layout."""

    generator_prompt = _DIALOG_GENERATOR_PROMPT
    evaluator_prompt = _DIALOG_EVALUATOR_PROMPT

    for required_text in (
        '固定格式块',
        '代码',
        'JSON',
        '配置',
        '日志',
        '命令',
        '缩进',
        '空行',
        'fenced code block',
        '角色语气只放在块外',
    ):
        assert required_text in generator_prompt

    for required_text in (
        '固定格式块',
        '代码块',
        'JSON 示例',
        '缩进',
        'fenced code block',
        '不得因为必要代码围栏而驳回',
    ):
        assert required_text in evaluator_prompt


def test_dialog_evaluator_prompt_audits_layout_without_line_budget() -> None:
    """Evaluator should audit one-bubble layout without line-count caps."""

    prompt = _DIALOG_EVALUATOR_PROMPT

    assert '单个可见聊天气泡' in prompt
    assert '布局可读性' in prompt
    assert '技术对比、参数列表和多候选推荐应使用普通聊天行' in prompt
    assert '只有当 `content_plan` 中的固定格式内容已经是表格时才保留表格' in prompt
    assert '不得仅因技术交付使用多行而驳回' in prompt
    assert '不得按固定行数、固定段数或固定字数判定失败' in prompt


@pytest.mark.asyncio
async def test_dialog_agent_preserves_evaluator_accepted_fragment_text(
    monkeypatch,
) -> None:
    """Dialog should not rewrite string fragments after evaluator acceptance."""

    state = _base_global_state()
    generator_llm = MagicMock()
    generator_llm.ainvoke = AsyncMock(return_value=AIMessage(
        content=(
            '{"final_dialog": ["raw<br>fragment"], '
            '"mention_target_user": false}'
        )
    ))
    evaluator_llm = MagicMock()
    evaluator_llm.ainvoke = AsyncMock(return_value=AIMessage(
        content='{"feedback": "Passed", "should_stop": true}'
    ))
    monkeypatch.setattr(dialog_module, "_dialog_generator_llm", generator_llm)
    monkeypatch.setattr(dialog_module, "_dialog_evaluator_llm", evaluator_llm)

    result = await dialog_agent(state)

    assert result["final_dialog"] == ["raw<br>fragment"]


def test_dialog_evaluator_prompt_orders_hard_gates_before_style() -> None:
    """Evaluator prompt should read as an ordered weak-model audit."""

    prompt = _DIALOG_EVALUATOR_PROMPT
    assert '不重新决定话题、意图、是否回答或角色立场' in prompt
    assert prompt.index('1. **建立语义计划**') < prompt.index(
        '2. **对照可见气泡**',
    )
    assert prompt.index('2. **对照可见气泡**') < prompt.index(
        '3. **审核单气泡布局和固定格式块**',
    )
    assert prompt.index('3. **审核单气泡布局和固定格式块**') < prompt.index(
        '4. **审核表达安全**',
    )
    assert prompt.index('4. **审核表达安全**') < prompt.index(
        '5. **最后看软风格**',
    )
    assert prompt.index('5. **最后看软风格**') < prompt.index('# 通过逻辑')
    assert '"should_stop": boolean' in prompt
    assert 'should_stop=false` 表示必须把 `feedback` 交回生成器重试' in prompt


def test_dialog_evaluator_prompt_rejects_unsupported_concrete_content() -> None:
    """Evaluator prompt should reject concrete claims not backed by the plan."""

    prompt = _DIALOG_EVALUATOR_PROMPT
    rule_start = prompt.index('事实边界：不得添加会改变结论')
    unsupported_rule = prompt[rule_start:rule_start + 500]

    assert '具体实体' in unsupported_rule
    assert '属性' in unsupported_rule
    assert '数量' in unsupported_rule
    assert '时间' in unsupported_rule
    assert '地点' in unsupported_rule
    assert '承诺' in unsupported_rule
    assert '轻微解释不矛盾时可以通过' in unsupported_rule
    assert '具体对象边界' in prompt
    assert '只能保留计划允许的泛化类别、行动骨架、筛选标准和核实清单' in prompt


def test_dialog_evaluator_prompt_rejects_guess_owner_flip() -> None:
    """Evaluator prompt should reject changed owner for guessing gates."""

    prompt = _DIALOG_EVALUATOR_PROMPT
    assert '指代与动作所有权' in prompt
    assert '审核对象' in prompt
    assert '先确认猜测动作和偏好所有者是谁' in prompt
    assert '我/我的/自己' in prompt
    assert '你/对方/你们' in prompt
    assert '猜测动作' in prompt
    assert '当前角色' in prompt
    assert '偏好' in prompt
    assert '我想看' in prompt
    assert '台词不得改成猜当前角色想看' in prompt


def test_dialog_generator_prompt_has_no_decision_ownership() -> None:
    """Dialog generation stays an execution renderer, not a decision stage."""

    assert "turns the upstream content plan into natural chat text" in (
        dialog_module.__doc__ or ""
    )
    assert "logical_stance" not in _DIALOG_GENERATOR_PROMPT
    assert "character_intent" not in _DIALOG_GENERATOR_PROMPT
    assert "boundary_profile" not in _DIALOG_GENERATOR_PROMPT
    assert "话题合法性" not in _DIALOG_GENERATOR_PROMPT


def test_dialog_prompts_use_content_plan_as_semantic_authority() -> None:
    """Generator and evaluator prompts should use content_plan authority."""

    generator_prompt = _DIALOG_GENERATOR_PROMPT
    evaluator_prompt = _DIALOG_EVALUATOR_PROMPT

    for required_text in (
        'content_plan` 是本轮已经决定好的可见语义计划',
        '# 角色表达依据',
        '这些字段只决定“怎么说”',
        '性格底色怎样影响台词',
        '声纹质感怎样影响台词',
        '先把叙述句、分析句和泛称感受句转成角色台词',
        '角色表达依据只能作用在已经重锚定好的台词上',
        '保持同义且不矛盾',
        '不丢失主要信息、不制造计划外结论',
        '固定格式块必须作为 `final_dialog` 数组里的字符串元素输出',
        '外层回复仍然必须是合法 JSON 对象',
        '对照角色表达依据',
    ):
        assert required_text in generator_prompt

    for forbidden_text in (
        '# 角色底色',
        '# 角色声纹约束',
        '**hesitation_density:**',
        '**fragmentation:**',
        '**direct_assertion:**',
        'GB300',
        'Pro6000',
    ):
        assert forbidden_text not in generator_prompt

    for required_text in (
        '`content_plan` 是本轮可见回复的语义计划',
        '语义是否忠实',
        '允许自然改写、轻微解释和布局调整',
        '轻微解释不矛盾时可以通过',
        '压缩表达可以通过',
        '自然表达可以通过',
        '说话视角忠实',
        '先扫说话视角',
        '泛称感受扫描',
        '只有同时满足以下条件才返回 `should_stop=true`',
    ):
        assert required_text in evaluator_prompt

    assert 'last_user' '_message' not in evaluator_prompt
    assert 'internal_monologue' not in evaluator_prompt
    assert 'internal_monologue' not in generator_prompt
    assert 'tone' '_history' not in generator_prompt
    assert '强制 `should_stop: true`' not in evaluator_prompt

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

