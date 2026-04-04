from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import AIMessage

from kazusa_ai_chatbot.agents.db_lookup_agent import DBLookupAgent
from kazusa_ai_chatbot.db import close_db


live_llm = pytest.mark.live_llm
live_db = pytest.mark.live_db


@pytest.mark.asyncio
async def test_db_lookup_agent_searches_conversation_history_with_supervisor_instruction():
    agent = DBLookupAgent()
    mock_llm = AsyncMock()
    mock_llm.ainvoke = AsyncMock(side_effect=[
        AIMessage(content='''<tool_call>{"name": "search_conversation_history", "args": {"query": "northern gate", "method": "keyword", "limit": 3}}</tool_call>'''),
        AIMessage(content="Earlier in this channel, the user said the northern gate was under pressure."),
    ])

    state = {
        "user_id": "user_123",
        "channel_id": "chan_456",
    }

    with patch("kazusa_ai_chatbot.agents.db_lookup_agent._get_llm", return_value=mock_llm), \
         patch("kazusa_ai_chatbot.agents.db_lookup_agent.search_conversation_history", new_callable=AsyncMock, return_value=[
             (-1.0, {
                 "user_id": "user_123",
                 "name": "Commander",
                 "role": "user",
                 "content": "The northern gate was under pressure.",
                 "timestamp": "2026-03-01T00:00:00Z",
             })
         ]):
        result = await agent.run(
            state,
            "Do you remember what I said about the northern gate?",
            "Search recent conversation history for prior mentions of the northern gate.",
        )

    assert result["status"] == "success"
    assert "northern gate" in result["summary"]
    assert result["tool_history"][0]["tool"] == "search_conversation_history"
    assert result["tool_history"][0]["args"]["query"] == "northern gate"


@pytest.mark.asyncio
async def test_db_lookup_agent_can_fetch_user_facts():
    agent = DBLookupAgent()
    mock_llm = AsyncMock()
    mock_llm.ainvoke = AsyncMock(side_effect=[
        AIMessage(content='''<tool_call>{"name": "get_user_facts", "args": {}}</tool_call>'''),
        AIMessage(content="The user prefers to be called Commander."),
    ])

    with patch("kazusa_ai_chatbot.agents.db_lookup_agent._get_llm", return_value=mock_llm), \
         patch("kazusa_ai_chatbot.agents.db_lookup_agent.get_user_facts", new_callable=AsyncMock, return_value=["Prefers to be called Commander"]):
        result = await agent.run(
            {"user_id": "user_123", "channel_id": "chan_456"},
            "What should you call me?",
            "Check remembered user preferences for naming.",
        )

    assert result["status"] == "success"
    assert "Commander" in result["summary"]
    assert result["tool_history"][0]["tool"] == "get_user_facts"


@pytest.mark.asyncio
async def test_db_lookup_agent_failure_returns_error_result():
    agent = DBLookupAgent()
    mock_llm = AsyncMock()
    mock_llm.ainvoke = AsyncMock(side_effect=RuntimeError("LLM unavailable"))

    with patch("kazusa_ai_chatbot.agents.db_lookup_agent._get_llm", return_value=mock_llm):
        result = await agent.run(
            {"user_id": "user_123", "channel_id": "chan_456"},
            "Do you remember me?",
            "Look up any remembered user details.",
        )

    assert result["status"] == "error"
    assert "LLM unavailable" in result["summary"]
    assert result["tool_history"] == []


@live_llm
@live_db
@pytest.mark.asyncio
async def test_live_db_lookup_agent_finds_matcha_chat():
    agent = DBLookupAgent()
    await close_db()
    try:
        result = await agent.run(
            {},
            "Find specific prior chat about 抹茶 in the current production database.",
            "Use search_conversation_history without channel or user filters to find exact mentions of 抹茶 in the existing configured database. Do not use user facts. Summarize the matched chat precisely.",
        )

        combined_text = result["summary"] + "\n" + "\n".join(call["result"] for call in result["tool_history"])

        assert result["status"] == "success"
        assert result["tool_history"]
        assert any(call["tool"] == "search_conversation_history" for call in result["tool_history"])
        assert "抹茶" in combined_text
    finally:
        await close_db()
