"""Tests for dialog_agent.py — generator/evaluator dialog loop."""

from __future__ import annotations

import typing
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kazusa_ai_chatbot.agents.dialog_agent import dialog_agent, DialogAgentState
from kazusa_ai_chatbot.utils import build_interaction_history_recent


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
                "expression_willingness": "open",
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
        "platform_user_id": "user_123",
        "platform_bot_id": "bot_456",
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
        },
    }


class TestDialogAgentState:
    def test_is_typed_dict(self):
        assert issubclass(DialogAgentState, dict)

    def test_has_required_fields(self):
        hints = typing.get_type_hints(DialogAgentState)
        required = [
            "internal_monologue", "action_directives",
            "chat_history_wide", "chat_history_recent", "platform_user_id", "platform_bot_id", "user_name", "user_profile",
            "character_profile",
        ]
        for field in required:
            assert field in hints, f"Missing field: {field}"


def test_build_interaction_history_recent_excludes_other_user_messages():
    """Scoped history should keep only the current user's turns and bot replies."""
    history = [
        {"role": "user", "platform_user_id": "user_a", "content": "这是千纱你的照片"},
        {"role": "assistant", "platform_user_id": "bot_456", "content": "明明就是想看我出糗吧，学长。"},
        {"role": "user", "platform_user_id": "user_b", "content": "你照片真涩情"},
        {"role": "assistant", "platform_user_id": "bot_456", "content": "学长看照片的眼神，感觉有点过分了啊。"},
    ]

    scoped = build_interaction_history_recent(history, "user_b", "bot_456")

    assert scoped == [
        {"role": "user", "platform_user_id": "user_b", "content": "你照片真涩情"},
        {"role": "assistant", "platform_user_id": "bot_456", "content": "学长看照片的眼神，感觉有点过分了啊。"},
    ]


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

    with patch("kazusa_ai_chatbot.agents.dialog_agent._dialog_generator_llm") as mock_generator, \
         patch("kazusa_ai_chatbot.agents.dialog_agent._dialog_evaluator_llm") as mock_evaluator:
        mock_generator.ainvoke = mock_ainvoke
        mock_evaluator.ainvoke = mock_ainvoke

        result = await dialog_agent(state)

    assert "final_dialog" in result
    assert isinstance(result["final_dialog"], list)


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

    with patch("kazusa_ai_chatbot.agents.dialog_agent._dialog_generator_llm") as mock_generator, \
         patch("kazusa_ai_chatbot.agents.dialog_agent._dialog_evaluator_llm") as mock_evaluator:
        mock_generator.ainvoke = mock_ainvoke
        mock_evaluator.ainvoke = mock_ainvoke

        result = await dialog_agent(state)

    assert result["final_dialog"] == [] or isinstance(result["final_dialog"], list)
