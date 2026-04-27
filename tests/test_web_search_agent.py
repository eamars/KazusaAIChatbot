"""Tests for the RAG2 web search helper agent."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kazusa_ai_chatbot.rag.web_search_agent import WebSearchAgent, _run_subgraph, web_search, web_url_read


@pytest.mark.asyncio
async def test_web_search_tool_delegates_to_mcp() -> None:
    """web_search should call the configured SearXNG MCP search tool."""
    with patch("kazusa_ai_chatbot.rag.web_search_agent.mcp_manager") as mock_mcp:
        mock_mcp.call_tool = AsyncMock(return_value="search results")
        result = await web_search.ainvoke({"query": "test query"})

    mock_mcp.call_tool.assert_awaited_once()
    call_args = mock_mcp.call_tool.await_args
    assert call_args.args[0] == "mcp-searxng__searxng_web_search"
    assert call_args.args[1]["query"] == "test query"
    assert call_args.args[1]["safesearch"] == 0
    assert result == "search results"


@pytest.mark.asyncio
async def test_web_url_read_tool_delegates_to_mcp() -> None:
    """web_url_read should call the configured MCP URL reader tool."""
    with patch("kazusa_ai_chatbot.rag.web_search_agent.mcp_manager") as mock_mcp:
        mock_mcp.call_tool = AsyncMock(return_value="page body")
        result = await web_url_read.ainvoke({"url": "https://example.test", "maxLength": 120})

    mock_mcp.call_tool.assert_awaited_once()
    call_args = mock_mcp.call_tool.await_args
    assert call_args.args[0] == "mcp-searxng__web_url_read"
    assert call_args.args[1]["url"] == "https://example.test"
    assert call_args.args[1]["maxLength"] == 120
    assert result == "page body"


@pytest.mark.asyncio
async def test_run_subgraph_returns_expected_keys() -> None:
    """_run_subgraph should map compiled graph state to the public result shape."""
    mock_result = {
        "final_status": "success",
        "final_reason": "found info",
        "final_response": "Here are the results",
        "final_is_empty_result": False,
        "knowledge_metadata": {"tool": "web_search"},
    }

    with patch("kazusa_ai_chatbot.rag.web_search_agent.StateGraph") as state_graph:
        graph_builder = MagicMock()
        graph_builder.compile.return_value.ainvoke = AsyncMock(return_value=mock_result)
        state_graph.return_value = graph_builder

        result = await _run_subgraph(
            task="search something",
            context={},
            expected_response="relevant results",
            timestamp="2026-04-27T00:00:00+00:00",
        )

    assert result == {
        "status": "success",
        "reason": "found info",
        "response": "Here are the results",
        "is_empty_result": False,
        "knowledge_metadata": {"tool": "web_search"},
    }


@pytest.mark.asyncio
async def test_web_search_agent_run_wraps_subgraph_result() -> None:
    """WebSearchAgent.run should expose the BaseRAG helper contract."""
    with patch(
        "kazusa_ai_chatbot.rag.web_search_agent._run_subgraph",
        new_callable=AsyncMock,
        return_value={
            "status": "success",
            "reason": "found info",
            "response": "evidence package",
            "is_empty_result": False,
            "knowledge_metadata": {},
        },
    ) as run_subgraph:
        result = await WebSearchAgent().run(
            task="search current weather",
            context={"current_timestamp": "2026-04-27T00:00:00+00:00"},
        )

    run_subgraph.assert_awaited_once()
    assert result == {
        "resolved": True,
        "result": "evidence package",
        "attempts": 1,
        "cache": {
            "enabled": False,
            "hit": False,
            "cache_name": "",
            "reason": "agent_not_cacheable",
        },
    }
