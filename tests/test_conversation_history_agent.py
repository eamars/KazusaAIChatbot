from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import AIMessage

from kazusa_ai_chatbot.agents.conversation_history_agent import ConversationHistoryAgent


@pytest.mark.asyncio
async def test_conversation_history_agent_searches_history_and_returns_summary():
    agent = ConversationHistoryAgent()
    mock_llm = AsyncMock()
    mock_llm.ainvoke = AsyncMock(side_effect=[
        AIMessage(
            content='I will search the conversation history for mentions of the northern gate.',
            tool_calls=[{
                "name": "search_conversation_history",
                "args": {"query": "northern gate", "method": "vector", "limit": 2},
                "id": "tool_call_1",
                "type": "tool_call"
            }]
        ),
        AIMessage(content='''{"status": "success", "summary": "Earlier in the conversation, the user said the northern gate was under pressure."}'''),
    ])

    with patch("kazusa_ai_chatbot.agents.conversation_history_agent._get_llm_with_tools", return_value=mock_llm), \
         patch("kazusa_ai_chatbot.agents.conversation_history_agent.search_conversation_history", new_callable=AsyncMock, return_value=[
             (0.91, {
                 "user_id": "user_123",
                 "name": "Commander",
                 "role": "user",
                 "content": "The northern gate is under pressure.",
                 "timestamp": "2026-04-01T12:00:00Z",
             })
         ]) as mock_search:
        result = await agent.run(
            {
                "user_id": "user_123",
                "channel_id": "chan_456",
                "message_text": "Do you remember what I said about the northern gate?",
            },
            "Search conversation history for the user's earlier mention of the northern gate.\n\nExpected response: Return a concise continuity summary only.",
        )

    assert result["agent"] == "conversation_history_agent"
    assert result["status"] == "success"
    assert "northern gate" in result["summary"].lower()
    assert result["tool_history"][0]["tool"] == "search_conversation_history"
    assert result["tool_history"][0]["args"]["query"] == "northern gate"


@pytest.mark.asyncio
async def test_conversation_history_agent_returns_error_on_llm_failure():
    agent = ConversationHistoryAgent()
    mock_llm = AsyncMock()
    mock_llm.ainvoke = AsyncMock(side_effect=RuntimeError("LLM unavailable"))

    with patch("kazusa_ai_chatbot.agents.conversation_history_agent._get_llm_with_tools", return_value=mock_llm):
        result = await agent.run(
            {"user_id": "user_123", "channel_id": "chan_456"},
            "What did we talk about before?",
            "Look up the previous conversation topic.",
        )

    assert result["agent"] == "conversation_history_agent"
    assert result["status"] == "error"
    assert "LLM unavailable" in result["summary"]
    assert result["tool_history"] == []
