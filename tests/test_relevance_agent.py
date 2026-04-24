"""Tests for relevance_agent.py — relevance gate logic."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kazusa_ai_chatbot.nodes.relevance_agent import relevance_agent


def _base_state():
    """Minimal IMProcessState for testing relevance_agent."""
    return {
        "timestamp": "2024-01-01T00:00:00Z",
        "platform": "discord",
        "platform_message_id": "msg_123",
        "platform_user_id": "user_123",
        "global_user_id": "uuid-123",
        "user_name": "TestUser",
        "user_input": "Hello bot!",
        "user_multimedia_input": [],
        "user_profile": {"affinity": 500, "last_relationship_insight": ""},
        "platform_bot_id": "bot_456",
        "bot_name": "TestBot",
        "character_profile": {
            "name": "Kazusa",
            "mood": "neutral",
            "global_vibe": "calm",
            "reflection_summary": "nothing notable",
        },
        "platform_channel_id": "chan_1",
        "channel_name": "general",
        "chat_history_wide": [],
        "chat_history_recent": [],
        "reply_context": {},
        "debug_modes": {},
    }


@pytest.mark.asyncio
async def test_relevance_agent_returns_should_respond():
    """LLM says should_respond=true → agent forwards that decision."""
    llm_response = MagicMock()
    llm_response.content = '{"should_respond": true, "reason_to_respond": "user greeted", "use_reply_feature": false, "channel_topic": "greetings", "indirect_speech_context": ""}'

    with patch("kazusa_ai_chatbot.nodes.relevance_agent._relevance_agent_llm") as mock_llm:
        mock_llm.ainvoke = AsyncMock(return_value=llm_response)
        result = await relevance_agent(_base_state())

    assert result["should_respond"] is True
    assert result["channel_topic"] == "greetings"
    assert result["indirect_speech_context"] == ""


@pytest.mark.asyncio
async def test_relevance_agent_should_not_respond():
    """LLM says should_respond=false → agent forwards that decision."""
    llm_response = MagicMock()
    llm_response.content = '{"should_respond": false, "reason_to_respond": "third party conversation", "use_reply_feature": false, "channel_topic": "sports", "indirect_speech_context": ""}'

    with patch("kazusa_ai_chatbot.nodes.relevance_agent._relevance_agent_llm") as mock_llm:
        mock_llm.ainvoke = AsyncMock(return_value=llm_response)
        result = await relevance_agent(_base_state())

    assert result["should_respond"] is False
    assert result["reason_to_respond"] == "third party conversation"


@pytest.mark.asyncio
async def test_relevance_agent_malformed_json_defaults_to_not_respond():
    """If LLM returns garbage JSON, parse_llm_json_output returns {} and should_respond defaults to False."""
    llm_response = MagicMock()
    llm_response.content = "this is not json at all"

    with patch("kazusa_ai_chatbot.nodes.relevance_agent._relevance_agent_llm") as mock_llm:
        mock_llm.ainvoke = AsyncMock(return_value=llm_response)
        result = await relevance_agent(_base_state())

    assert result["should_respond"] is False
    assert result["channel_topic"] == ""
    assert result["indirect_speech_context"] == ""


@pytest.mark.asyncio
async def test_relevance_agent_use_reply_feature():
    """LLM says use_reply_feature=true → agent forwards it."""
    llm_response = MagicMock()
    llm_response.content = '{"should_respond": true, "reason_to_respond": "reply needed", "use_reply_feature": true, "channel_topic": "topic", "indirect_speech_context": ""}'

    with patch("kazusa_ai_chatbot.nodes.relevance_agent._relevance_agent_llm") as mock_llm:
        mock_llm.ainvoke = AsyncMock(return_value=llm_response)
        result = await relevance_agent(_base_state())

    assert result["use_reply_feature"] is True


@pytest.mark.asyncio
async def test_relevance_agent_short_circuits_structured_third_party_reply_in_group() -> None:
    """Structured reply metadata to another participant should skip the LLM in noisy channels."""
    state = _base_state()
    state["user_input"] = '[Reply to message] <@someone-else> 我同事上下班是不用加油的'
    state["reply_context"] = {
        "reply_to_message_id": "other-msg",
        "reply_to_platform_user_id": "someone-else",
        "reply_to_current_bot": False,
    }

    with patch("kazusa_ai_chatbot.nodes.relevance_agent._relevance_agent_llm") as mock_llm:
        mock_llm.ainvoke = AsyncMock()
        result = await relevance_agent(state)

    assert result["should_respond"] is False
    assert "another participant" in result["reason_to_respond"]
    mock_llm.ainvoke.assert_not_called()


@pytest.mark.asyncio
async def test_relevance_agent_allows_structured_third_party_reply_when_bot_explicitly_addressed() -> None:
    """Explicit bot address should override the third-party reply short-circuit."""
    state = _base_state()
    state["user_input"] = '<@bot_456> 你怎么看他刚才那句？'
    state["reply_context"] = {
        "reply_to_message_id": "other-msg",
        "reply_to_platform_user_id": "someone-else",
        "reply_to_current_bot": False,
    }
    llm_response = MagicMock()
    llm_response.content = '{"should_respond": true, "reason_to_respond": "bot explicitly addressed", "use_reply_feature": true, "channel_topic": "discussion", "indirect_speech_context": ""}'

    with patch("kazusa_ai_chatbot.nodes.relevance_agent._relevance_agent_llm") as mock_llm:
        mock_llm.ainvoke = AsyncMock(return_value=llm_response)
        result = await relevance_agent(state)

    assert result["should_respond"] is True
    mock_llm.ainvoke.assert_called_once()


@pytest.mark.asyncio
async def test_relevance_agent_legacy_reply_marker_still_blocks_third_party_reply() -> None:
    """Legacy text-only reply markers should still gate obvious third-party replies."""
    state = _base_state()
    state["user_input"] = '[Reply to message] <@someone-else> 我同事上下班是不用加油的'

    with patch("kazusa_ai_chatbot.nodes.relevance_agent._relevance_agent_llm") as mock_llm:
        mock_llm.ainvoke = AsyncMock()
        result = await relevance_agent(state)

    assert result["should_respond"] is False
    mock_llm.ainvoke.assert_not_called()


@pytest.mark.asyncio
async def test_relevance_agent_does_not_expose_reflection_summary_in_prompt() -> None:
    """Global reflection summaries should not be injected into relevance reasoning."""
    llm_response = MagicMock()
    llm_response.content = '{"should_respond": true, "reason_to_respond": "user greeted", "use_reply_feature": false, "channel_topic": "greetings", "indirect_speech_context": ""}'

    with patch("kazusa_ai_chatbot.nodes.relevance_agent._relevance_agent_llm") as mock_llm:
        mock_llm.ainvoke = AsyncMock(return_value=llm_response)
        await relevance_agent(_base_state())

    rendered_prompt = mock_llm.ainvoke.await_args.args[0][0].content
    assert "自我反思" not in rendered_prompt
    assert "nothing notable" not in rendered_prompt
