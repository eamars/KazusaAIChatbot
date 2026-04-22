"""Tests for persona_supervisor2_msg_decontexualizer.py — message decontextualization."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kazusa_ai_chatbot.nodes.persona_supervisor2_msg_decontexualizer import call_msg_decontexualizer


def _base_state():
    """Minimal GlobalPersonaState for testing call_msg_decontexualizer."""
    return {
        "user_input": "他在干啥？",
        "user_name": "TestUser",
        "platform_user_id": "user_123",
        "platform_bot_id": "bot_456",
        "chat_history_recent": [
            {"name": "Alice", "user_id": "u1", "content": "Bob is cooking", "role": "user", "timestamp": "t1"},
        ],
        "channel_topic": "general chat",
        "indirect_speech_context": "",
    }


@pytest.mark.asyncio
async def test_decontexualizer_returns_modified_input():
    """When LLM says is_modified=true, output should be the decontextualized text."""
    llm_response = MagicMock()
    llm_response.content = '{"output": "Bob在干啥？", "reasoning": "resolved pronoun", "is_modified": true}'

    with patch("kazusa_ai_chatbot.nodes.persona_supervisor2_msg_decontexualizer._msg_decontexualizer_llm") as mock_llm:
        mock_llm.ainvoke = AsyncMock(return_value=llm_response)

        result = await call_msg_decontexualizer(_base_state())

    assert result["decontexualized_input"] == "Bob在干啥？"


@pytest.mark.asyncio
async def test_decontexualizer_returns_original_when_not_modified():
    """When LLM says is_modified=false, output should be the original user_input."""
    llm_response = MagicMock()
    llm_response.content = '{"output": "他在干啥？", "reasoning": "already clear", "is_modified": false}'

    with patch("kazusa_ai_chatbot.nodes.persona_supervisor2_msg_decontexualizer._msg_decontexualizer_llm") as mock_llm:
        mock_llm.ainvoke = AsyncMock(return_value=llm_response)

        result = await call_msg_decontexualizer(_base_state())

    assert result["decontexualized_input"] == "他在干啥？"


@pytest.mark.asyncio
async def test_decontexualizer_fallback_on_llm_error():
    """If LLM call raises, output falls back to original user_input."""
    with patch("kazusa_ai_chatbot.nodes.persona_supervisor2_msg_decontexualizer._msg_decontexualizer_llm") as mock_llm:
        mock_llm.ainvoke = AsyncMock(side_effect=RuntimeError("LLM down"))

        result = await call_msg_decontexualizer(_base_state())

    assert result["decontexualized_input"] == "他在干啥？"


@pytest.mark.asyncio
async def test_decontexualizer_fallback_on_malformed_json():
    """If LLM returns garbage, output falls back to original user_input."""
    llm_response = MagicMock()
    llm_response.content = "not json at all"

    with patch("kazusa_ai_chatbot.nodes.persona_supervisor2_msg_decontexualizer._msg_decontexualizer_llm") as mock_llm:
        mock_llm.ainvoke = AsyncMock(return_value=llm_response)

        result = await call_msg_decontexualizer(_base_state())

    # parse_llm_json_output returns {} for garbage → is_modified defaults to False
    assert result["decontexualized_input"] == "他在干啥？"
