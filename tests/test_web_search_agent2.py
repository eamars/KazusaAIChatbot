"""Tests for web_search_agent2.py — web search tool and agent orchestration."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kazusa_ai_chatbot.agents.web_search_agent2 import web_search, web_search_agent


@pytest.mark.asyncio
async def test_web_search_tool_delegates_to_mcp():
    """web_search tool should call mcp_manager.call_tool with correct arguments."""
    with patch("kazusa_ai_chatbot.agents.web_search_agent2.mcp_manager") as mock_mcp:
        mock_mcp.call_tool = AsyncMock(return_value="search results")
        result = await web_search.ainvoke({"query": "test query"})

    mock_mcp.call_tool.assert_called_once()
    call_args = mock_mcp.call_tool.call_args
    assert call_args[0][0] == "mcp-searxng__searxng_web_search"
    assert call_args[0][1]["query"] == "test query"
    assert result == "search results"


@pytest.mark.asyncio
async def test_web_search_agent_returns_expected_keys():
    """web_search_agent entry point should return dict with status/reason/response keys."""
    # Mock the full subgraph by patching the StateGraph compile and ainvoke
    mock_result = {
        "final_status": "complete",
        "final_reason": "found info",
        "final_response": "Here are the results",
        "knowledge_metadata": {},
    }

    with patch("kazusa_ai_chatbot.agents.web_search_agent2.StateGraph") as MockSG:
        mock_graph = MagicMock()
        mock_graph.compile.return_value.ainvoke = AsyncMock(return_value=mock_result)
        MockSG.return_value = mock_graph

        result = await web_search_agent(
            task="search something",
            context={},
            expected_response="relevant results",
        )

    assert result["status"] == "complete"
    assert result["reason"] == "found info"
    assert result["response"] == "Here are the results"


@pytest.mark.asyncio
async def test_web_search_agent_default_timestamp():
    """web_search_agent should generate a timestamp if none provided."""
    mock_result = {
        "final_status": "error",
        "final_reason": "no results",
        "final_response": "",
    }

    with patch("kazusa_ai_chatbot.agents.web_search_agent2.StateGraph") as MockSG:
        mock_graph = MagicMock()
        mock_graph.compile.return_value.ainvoke = AsyncMock(return_value=mock_result)
        MockSG.return_value = mock_graph

        result = await web_search_agent(
            task="query",
            context={},
            expected_response="something",
            timestamp=None,
        )

    assert result["status"] == "error"
