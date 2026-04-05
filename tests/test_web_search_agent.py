from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage

from kazusa_ai_chatbot.agents.web_search_agent import WebSearchAgent


@pytest.mark.asyncio
async def test_web_search_agent_calls_search_tool_and_returns_summary():
    agent = WebSearchAgent()

    # First call: LLM returns an AIMessage with tool_calls
    tool_call_msg = MagicMock(spec=AIMessage)
    tool_call_msg.tool_calls = [{
        "name": "searxng_web_search",
        "args": {"query": "Tokyo weather today"},
        "id": "call_123",
    }]
    tool_call_msg.content = ""

    # Second call: LLM returns final JSON (no tool calls)
    final_msg = MagicMock(spec=AIMessage)
    final_msg.tool_calls = []
    final_msg.content = '{"status": "success", "summary": "Tokyo weather today is mild with light clouds."}'

    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(side_effect=[tool_call_msg, final_msg])

    # Mock the searxng_web_search tool
    mock_search_tool = AsyncMock()
    mock_search_tool.name = "searxng_web_search"
    mock_search_tool.ainvoke = AsyncMock(return_value="Tokyo weather today: mild, light clouds.")

    with patch("kazusa_ai_chatbot.agents.web_search_agent._get_llm_with_tools", return_value=mock_llm), \
         patch("kazusa_ai_chatbot.agents.web_search_agent._get_langchain_tools", return_value=[mock_search_tool]):
        result = await agent.run(
            {"user_id": "user_123", "channel_id": "chan_456"},
            "What's the weather in Tokyo today?",
            "Search for Tokyo weather today.",
            "Return a concise factual weather summary.",
        )

    assert result["agent"] == "web_search_agent"
    assert result["status"] == "success"
    assert "tokyo weather" in result["summary"].lower()
    assert result["tool_history"][0]["tool"] == "searxng_web_search"
    assert result["tool_history"][0]["args"] == {"query": "Tokyo weather today"}
    mock_search_tool.ainvoke.assert_awaited_once()


@pytest.mark.asyncio
async def test_web_search_agent_returns_error_when_no_tools_available():
    agent = WebSearchAgent()

    with patch("kazusa_ai_chatbot.agents.web_search_agent._get_langchain_tools", return_value=[]):
        result = await agent.run(
            {"user_id": "user_123", "channel_id": "chan_456"},
            "Search for current weather.",
            "Search for current weather.",
        )

    assert result["agent"] == "web_search_agent"
    assert result["status"] == "error"
    assert result["summary"] == "No search tools available."
    assert result["tool_history"] == []

