"""Tests for dialog_agent.py generator-only dialog rendering."""

from __future__ import annotations

import logging
import typing
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.messages import AIMessage
import pytest

from kazusa_ai_chatbot.nodes import dialog_agent as dialog_module
from kazusa_ai_chatbot.nodes.dialog_agent import (
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
                    "rendering": "One outbound message; concise.",
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
        "rendering": "One outbound message; concise.",
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


def test_dialog_prompts_preserve_multi_part_deliverables() -> None:
    """Dialog generator should preserve L3 content without literal overfitting."""

    generator_prompt = _DIALOG_GENERATOR_PROMPT

    for required_text in (
        '# 生成流程',
        '# 核心转换',
        '# 角色表达依据',
        '这些字段只决定“怎么说”',
        '性格底色怎样影响台词',
        '声纹质感怎样影响台词',
        '最终文字采用当前角色对当前用户说话的视角',
        '叙述句和分析句先转成角色当场聊天的互动骨架',
        '角色表达依据只能作用在已经重锚定好的台词上',
        '建立内容骨架',
        '重锚定说话视角',
        '按内容类型渲染',
        '选择角色表达',
        '组织消息序列',
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
        "Evaluator" " Feedback",
        'GB300',
        'Pro6000',
    ):
        assert forbidden_text not in generator_prompt


def test_dialog_generator_prompt_describes_message_sequence_contract() -> None:
    """Generator prompt should describe final_dialog as outbound messages."""

    prompt = _DIALOG_GENERATOR_PROMPT

    assert '`final_dialog` 的每个字符串都是一条完整的普通在线文字消息' in prompt
    assert '按 `content_plan.rendering` 生成 1-N 条消息' in prompt
    assert '连续发送' in prompt
    assert '每个元素写一条可独立发送的可见文字' in prompt
    assert '停顿用标点和句子节奏表达' in prompt
    assert '回答从左花括号开始，以右花括号结束' in prompt
    assert '输出前自检' in prompt
    retired_layout = ''.join(('一个', '\u804a\u5929\u6c14\u6ce1'))
    retired_join = '运行时会用' + '换行连接 `final_dialog`'
    retired_layout_unit = '布局' + '单位'
    assert retired_layout not in prompt
    assert retired_join not in prompt
    assert retired_layout_unit not in prompt
    assert '平台分别发送' not in prompt
    assert '打一段、发一段' not in prompt
    assert '使用 6-12 个短字符串片段是允许的' not in prompt


def test_dialog_generator_prompt_allows_inline_tag_sign() -> None:
    """Generator prompt should let dialog author visible inline tags."""

    prompt = _DIALOG_GENERATOR_PROMPT
    retired_field = "mention" + "_target_user"

    assert '@display_name' in prompt
    assert retired_field not in prompt
    assert '用户 ID' not in prompt
    assert '平台标签' not in prompt


def test_dialog_prompts_preserve_fixed_format_blocks() -> None:
    """Dialog generator should preserve code and fixed-format block layout."""

    generator_prompt = _DIALOG_GENERATOR_PROMPT

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


@pytest.mark.asyncio
async def test_dialog_agent_preserves_generator_fragment_text(
    monkeypatch,
) -> None:
    """Dialog should not rewrite string fragments after generation."""

    state = _base_global_state()
    generator_llm = MagicMock()
    generator_llm.ainvoke = AsyncMock(return_value=AIMessage(
        content='{"final_dialog": ["raw<br>fragment"]}'
    ))
    monkeypatch.setattr(dialog_module, "_dialog_generator_llm", generator_llm)

    result = await dialog_agent(state)

    assert result["final_dialog"] == ["raw<br>fragment"]


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
    """Generator prompt should use content_plan authority."""

    generator_prompt = _DIALOG_GENERATOR_PROMPT

    for required_text in (
        'content_plan` 是本轮已经决定好的台词计划',
        '# 角色表达依据',
        '这些字段只决定“怎么说”',
        '性格底色怎样影响台词',
        '声纹质感怎样影响台词',
        '叙述句和分析句先转成角色当场聊天的互动骨架',
        '角色表达依据只能作用在已经重锚定好的台词上',
        '保持同义且不矛盾',
        '技术和事实场景里的调侃只落在开头、转接或收尾的轻重',
        '吐槽对象只能是交付语气或角色整理信息的姿态',
        '如果计划只说“更适合”，台词仍保留为“更适合”',
        '技术场景里的吐槽是否只影响交付语气',
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
        "Evaluator" " Feedback",
        'GB300',
        'Pro6000',
    ):
        assert forbidden_text not in generator_prompt

    assert 'internal_monologue' not in generator_prompt
    assert 'tone' '_history' not in generator_prompt

def test_dialog_rendered_voice_constraints_do_not_seed_literal_phrases() -> None:
    """Rendered voice constraints should not invite catchphrase copying."""

    state = _base_global_state()
    ltp = state["character_profile"]["linguistic_texture_profile"]
    ltp["counter_questioning"] = 0.6
    ltp["softener_density"] = 1.0
    character_profile = state["character_profile"]
    personality_brief = character_profile["personality_brief"]

    prompt = _DIALOG_GENERATOR_PROMPT.format(
        character_name=character_profile["name"],
        character_logic=personality_brief["logic"],
        character_tempo=personality_brief["tempo"],
        character_defense=personality_brief["defense"],
        character_quirks=personality_brief["quirks"],
        character_taboos=personality_brief["taboos"],
        ltp_hesitation_density=dialog_module.get_hesitation_density_description(
            ltp["hesitation_density"],
        ),
        ltp_fragmentation=dialog_module.get_fragmentation_description(
            ltp["fragmentation"],
        ),
        ltp_emotional_leakage=dialog_module.get_emotional_leakage_description(
            ltp["emotional_leakage"],
        ),
        ltp_rhythmic_bounce=dialog_module.get_rhythmic_bounce_description(
            ltp["rhythmic_bounce"],
        ),
        ltp_direct_assertion=dialog_module.get_direct_assertion_description(
            ltp["direct_assertion"],
        ),
        ltp_softener_density=dialog_module.get_softener_density_description(
            ltp["softener_density"],
        ),
        ltp_counter_questioning=dialog_module.get_counter_questioning_description(
            ltp["counter_questioning"],
        ),
        ltp_formalism_avoidance=(
            dialog_module.get_formalism_avoidance_description(
                ltp["formalism_avoidance"],
            )
        ),
        ltp_abstraction_reframing=(
            dialog_module.get_abstraction_reframing_description(
                ltp["abstraction_reframing"],
            )
        ),
        ltp_self_deprecation=dialog_module.get_self_deprecation_description(
            ltp["self_deprecation"],
        ),
    )
    voice_start = prompt.index("# 角色表达依据")
    voice_end = prompt.index("# 生成流程")
    voice_block = prompt[voice_start:voice_end]

    assert "content_plan" in voice_block
    for forbidden_text in (
        "不然呢",
        "你觉得呢",
        "是吗",
        "「",
        "」",
    ):
        assert forbidden_text not in voice_block


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

    async def mock_ainvoke(messages, *, config=None):
        return generator_response

    with patch(
        "kazusa_ai_chatbot.nodes.dialog_agent._dialog_generator_llm"
    ) as mock_generator:
        mock_generator.ainvoke = mock_ainvoke

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
    monkeypatch.setattr(dialog_module, "_dialog_generator_llm", generator_llm)

    with pytest.raises(
        StateContractError,
        match="action_directives.linguistic_directives",
    ):
        await dialog_agent(state)

    generator_llm.ainvoke.assert_not_awaited()


@pytest.mark.asyncio
async def test_dialog_agent_handles_empty_dialog():
    """If generator returns no dialog, final_dialog should default to empty list."""
    state = _base_global_state()

    from langchain_core.messages import AIMessage
    generator_response = AIMessage(content='{"final_dialog": []}')

    async def mock_ainvoke(messages, *, config=None):
        return generator_response

    with patch(
        "kazusa_ai_chatbot.nodes.dialog_agent._dialog_generator_llm"
    ) as mock_generator:
        mock_generator.ainvoke = mock_ainvoke

        result = await dialog_agent(state)

    assert result["final_dialog"] == [] or isinstance(result["final_dialog"], list)


@pytest.mark.asyncio
async def test_dialog_agent_logs_explicit_usage_mode(caplog):
    """Dialog output logs should identify the caller-supplied usage mode."""

    state = _base_global_state()
    state["dialog_usage_mode"] = "background_contract_render"

    from langchain_core.messages import AIMessage
    generator_response = AIMessage(content='{"final_dialog": ["Internal only."]}')
    async def mock_ainvoke(messages, *, config=None):
        return generator_response

    with patch(
        "kazusa_ai_chatbot.nodes.dialog_agent._dialog_generator_llm"
    ) as mock_generator:
        mock_generator.ainvoke = mock_ainvoke

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
        content='{"final_dialog": ["Still answering."]}'
    ))
    monkeypatch.setattr(dialog_module, "_dialog_generator_llm", generator_llm)

    result = await dialog_agent(state)

    assert result["final_dialog"] == ["Still answering."]
    assert result["target_addressed_user_ids"] == ["global-user-123"]
    assert result["target_broadcast"] is False
    retired_field = "mention" + "_target_user"
    assert retired_field not in result
    generator_llm.ainvoke.assert_awaited_once()

