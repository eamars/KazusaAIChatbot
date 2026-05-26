"""Deterministic routing edge-case tests for web_agent3."""

from __future__ import annotations

import importlib
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import AIMessage

from kazusa_ai_chatbot.rag import web_agent3 as web_module
from kazusa_ai_chatbot.rag.web_agent3 import agent as agent_module


def test_web_agent3_router_normalizes_edge_case_decisions() -> None:
    """Router normalization should stay bounded under malformed LLM output."""
    cases = [
        {
            "raw": {
                "action": " READ ",
                "source": " YouTube ",
                "query": " https://www.youtube.com/watch?v=abc123 ",
            },
            "fallback_query": "fallback youtube task",
            "expected": web_module._RouterDecision(
                action="read",
                source="youtube",
                query="https://www.youtube.com/watch?v=abc123",
            ),
        },
        {
            "raw": {
                "action": "open",
                "source": "archive",
                "query": "SearXNG JSON API docs",
            },
            "fallback_query": "fallback search task",
            "expected": web_module._RouterDecision(
                action="search",
                source="generic",
                query="SearXNG JSON API docs",
            ),
        },
        {
            "raw": {
                "action": "read",
                "source": "nhentai",
                "query": "",
            },
            "fallback_query": "Web-evidence: lookup gallery 652244",
            "expected": web_module._RouterDecision(
                action="search",
                source="generic",
                query="Web-evidence: lookup gallery 652244",
            ),
        },
        {
            "raw": {
                "action": "stop",
                "source": "bilibili",
                "query": "BV1example",
            },
            "fallback_query": "fallback bilibili task",
            "expected": web_module._RouterDecision(
                action="stop",
                source="bilibili",
                query="",
            ),
        },
    ]

    for case in cases:
        decision = web_module._normalize_router_decision(
            case["raw"],
            fallback_query=case["fallback_query"],
            valid_sources=("generic", "bilibili", "youtube", "nhentai"),
        )

        assert decision == case["expected"]


@pytest.mark.asyncio
async def test_web_agent3_generator_accepts_edge_source_routes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Generator should preserve valid edge-source route decisions."""
    fake_llm = SimpleNamespace(
        ainvoke=AsyncMock(side_effect=[
            AIMessage(
                content=json.dumps({
                    "action": "read",
                    "source": "youtube",
                    "query": "https://www.youtube.com/watch?v=abc123",
                }),
            ),
            AIMessage(
                content=json.dumps({
                    "action": "read",
                    "source": "bilibili",
                    "query": "BV1example",
                }),
            ),
            AIMessage(
                content=json.dumps({
                    "action": "read",
                    "source": "nhentai",
                    "query": "652244",
                }),
            ),
            AIMessage(
                content=json.dumps({
                    "action": "read",
                    "source": "generic",
                    "query": "https://example.test/docs",
                }),
            ),
        ]),
    )
    monkeypatch.setattr(agent_module, "_generator_llm", fake_llm)
    cases = [
        (
            "Web-evidence: summarize https://www.youtube.com/watch?v=abc123",
            {
                "action": "read",
                "source": "youtube",
                "query": "https://www.youtube.com/watch?v=abc123",
            },
        ),
        (
            "Web-evidence: summarize bilibili BV1example",
            {"action": "read", "source": "bilibili", "query": "BV1example"},
        ),
        (
            "Web-evidence: lookup nhentai 652244 metadata",
            {"action": "read", "source": "nhentai", "query": "652244"},
        ),
        (
            "Web-evidence: summarize https://example.test/docs",
            {
                "action": "read",
                "source": "generic",
                "query": "https://example.test/docs",
            },
        ),
    ]

    for task, expected_decision in cases:
        state = {
            "task": task,
            "context": {"platform": "debug"},
            "messages": [],
            "observations": [],
            "evaluator_feedback": "",
            "prompt_timestamp": "2026-05-27 12:00 (Wednesday)",
        }

        update = await agent_module._tool_call_generator(state)

        assert update["router_decision"] == expected_decision


@pytest.mark.asyncio
async def test_web_agent3_generator_falls_back_for_invalid_edge_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Invalid router output should degrade to a generic search decision."""
    fake_llm = SimpleNamespace(
        ainvoke=AsyncMock(return_value=AIMessage(
            content=json.dumps({
                "action": "browse",
                "source": "forum",
                "query": "",
            }),
        )),
    )
    monkeypatch.setattr(agent_module, "_generator_llm", fake_llm)
    state = {
        "task": "Web-evidence: search public docs for current API behavior",
        "context": {"platform": "debug"},
        "messages": [],
        "observations": [],
        "evaluator_feedback": "try a broader search",
        "prompt_timestamp": "2026-05-27 12:00 (Wednesday)",
    }

    update = await agent_module._tool_call_generator(state)

    assert update["router_decision"] == {
        "action": "search",
        "source": "generic",
        "query": "Web-evidence: search public docs for current API behavior",
    }


@pytest.mark.asyncio
async def test_web_agent3_executor_dispatches_edge_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Executor should preserve source dispatch and source-local workarounds."""
    generic_subagent = importlib.import_module(
        "kazusa_ai_chatbot.rag.web_agent3.subagent.generic"
    )
    fake_search = SimpleNamespace(ainvoke=AsyncMock(return_value="search body"))
    fake_read = SimpleNamespace(ainvoke=AsyncMock(return_value="page body"))
    monkeypatch.setattr(generic_subagent.searxng_tools, "web_search", fake_search)
    monkeypatch.setattr(generic_subagent.searxng_tools, "web_url_read", fake_read)

    generic_search_result = await web_module._execute_source_decision(
        web_module._RouterDecision(
            action="search",
            source="generic",
            query="official docs latest release",
        )
    )
    generic_read_result = await web_module._execute_source_decision(
        web_module._RouterDecision(
            action="read",
            source="generic",
            query="https://example.test/docs",
        )
    )
    specialized_result = await web_module._execute_source_decision(
        web_module._RouterDecision(
            action="read",
            source="youtube",
            query="https://www.youtube.com/watch?v=abc123",
        )
    )

    fake_search.ainvoke.assert_awaited_once_with({
        "query": "official docs latest release",
    })
    assert fake_read.ainvoke.await_args_list[0].args[0] == {
        "url": "https://example.test/docs",
    }
    assert generic_search_result == "search body"
    assert generic_read_result == "page body"
    assert fake_read.ainvoke.await_count == 2
    assert fake_read.ainvoke.await_args_list[1].args[0] == {
        "url": "https://www.youtube.com/watch?v=abc123",
    }
    assert specialized_result == "page body"
