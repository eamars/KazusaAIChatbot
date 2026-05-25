"""Tests for the RAG2 web search helper agent."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from kazusa_ai_chatbot.rag import web_search_agent as web_module
from kazusa_ai_chatbot.rag.web_search_agent import (
    WebSearchAgent,
    _run_subgraph,
    web_search,
    web_url_read,
)
from kazusa_ai_chatbot.time_boundary import build_turn_clock_from_storage_utc


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
            local_prompt_timestamp="2026-04-27 12:00",
        )

    sub_state = graph_builder.compile.return_value.ainvoke.await_args.args[0]
    assert sub_state["prompt_timestamp"] == "2026-04-27 12:00"
    assert result == {
        "status": "success",
        "reason": "found info",
        "response": "Here are the results",
        "is_empty_result": False,
        "knowledge_metadata": {"tool": "web_search"},
    }


@pytest.mark.asyncio
async def test_tool_call_generator_passes_reference_time_to_human_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Generator prompt keeps current time in the late Human JSON payload."""

    fake_llm = SimpleNamespace(
        ainvoke=AsyncMock(return_value=AIMessage(content="")),
    )
    monkeypatch.setattr(web_module, "_generator_llm", fake_llm)
    state = {
        "task": "Find the latest release status.",
        "context": {"platform": "debug"},
        "messages": [HumanMessage(content="start")],
        "prompt_timestamp": "2026-05-25 21:30 (Monday)",
    }

    await web_module._tool_call_generator(state)

    messages = fake_llm.ainvoke.await_args.args[0]
    system_prompt = messages[0].content
    payload = json.loads(messages[1].content)
    assert "# 可用工具" not in system_prompt
    assert payload["reference_time"] == "2026-05-25 21:30 (Monday)"


@pytest.mark.asyncio
async def test_tool_call_evaluator_passes_reference_time_to_human_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Evaluator prompt keeps current time in the late Human JSON payload."""

    fake_llm = SimpleNamespace(
        ainvoke=AsyncMock(return_value=AIMessage(content='{"should_stop": true}')),
    )
    monkeypatch.setattr(web_module, "_evaluator_llm", fake_llm)
    state = {
        "task": "Find the latest release status.",
        "expected_response": "official status",
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "web_search",
                        "args": {"query": "Python release"},
                        "id": "call-1",
                    }
                ],
            ),
            ToolMessage(content="search result", tool_call_id="call-1"),
        ],
        "retry": 0,
        "prompt_timestamp": "2026-05-25 21:30 (Monday)",
    }

    await web_module._tool_call_evaluator(state)

    messages = fake_llm.ainvoke.await_args.args[0]
    payload = json.loads(messages[1].content)
    assert payload["reference_time"] == "2026-05-25 21:30 (Monday)"
    assert payload["call_history"][0]["tool"] == "web_search"


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
        turn_clock = build_turn_clock_from_storage_utc(
            "2026-04-27T00:00:00+00:00"
        )
        result = await WebSearchAgent().run(
            task="search current weather",
            context={"local_time_context": turn_clock["local_time_context"]},
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
