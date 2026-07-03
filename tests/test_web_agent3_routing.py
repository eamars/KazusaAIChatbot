"""Deterministic routing edge-case tests for web_agent3."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import AIMessage

from kazusa_ai_chatbot.rag import web_agent3 as web_module
from kazusa_ai_chatbot.rag.web_agent3 import agent as agent_module
from kazusa_ai_chatbot.rag.web_agent3 import providers as provider_module
from kazusa_ai_chatbot.rag.web_agent3.subagent import web_read as web_read_subagent
from kazusa_ai_chatbot.rag.web_agent3.subagent import web_search as web_search_subagent


def test_web_agent3_router_normalizes_edge_case_decisions() -> None:
    """Router normalization should stay bounded under malformed LLM output."""
    source_actions = {
        "bilibili": ("read", "search"),
        "web_read": ("read",),
        "web_search": ("search",),
        "nhentai": ("read", "search"),
    }
    cases = [
        {
            "raw": {
                "action": " READ ",
                "source": " web_read ",
                "query": " https://example.test/page ",
            },
            "fallback_query": "fallback read task",
            "expected": web_module._RouterDecision(
                action="read",
                source="web_read",
                query="https://example.test/page",
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
                source="web_search",
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
                source="web_search",
                query="Web-evidence: lookup gallery 652244",
            ),
        },
        {
            "raw": {
                "action": "stop",
                "source": "web_search",
                "query": "ignored",
            },
            "fallback_query": "fallback task",
            "expected": web_module._RouterDecision(
                action="stop",
                source="web_read",
                query="",
            ),
        },
    ]

    for case in cases:
        decision = web_module._normalize_router_decision(
            case["raw"],
            fallback_query=case["fallback_query"],
            valid_sources=("web_read", "web_search", "nhentai"),
            source_actions=source_actions,
        )

        assert decision == case["expected"]


def test_web_agent3_router_normalizes_final_source_action_matrix() -> None:
    """Router normalization should enforce the final source/action matrix."""
    source_actions = {
        "bilibili": ("read", "search"),
        "web_read": ("read",),
        "web_search": ("search",),
        "nhentai": ("read", "search"),
    }
    enabled_sources = ("web_read", "web_search", "nhentai", "bilibili")
    cases = [
        (
            {"action": "stop", "source": "nhentai", "query": "652244"},
            enabled_sources,
            web_module._RouterDecision("stop", "web_read", ""),
        ),
        (
            {"action": "search", "source": "nhentai", "query": "tag:demo"},
            enabled_sources,
            web_module._RouterDecision("search", "nhentai", "tag:demo"),
        ),
        (
            {"action": "search", "source": "nhentai", "query": "tag:demo"},
            ("web_read", "web_search"),
            web_module._RouterDecision("stop", "web_read", ""),
        ),
        (
            {
                "action": "search",
                "source": "bilibili",
                "query": "vibe coding",
            },
            enabled_sources,
            web_module._RouterDecision("search", "bilibili", "vibe coding"),
        ),
        (
            {
                "action": "search",
                "source": "bilibili",
                "query": "vibe coding",
            },
            ("web_read", "web_search"),
            web_module._RouterDecision("stop", "web_read", ""),
        ),
        (
            {"action": "search", "source": "forum", "query": "api docs"},
            ("web_read", "web_search"),
            web_module._RouterDecision("search", "web_search", "api docs"),
        ),
        (
            {"action": "search", "source": "forum", "query": "api docs"},
            ("web_read",),
            web_module._RouterDecision("stop", "web_read", ""),
        ),
        (
            {"action": "read", "source": "forum", "query": "https://example.test"},
            ("web_read",),
            web_module._RouterDecision("read", "web_read", "https://example.test"),
        ),
        (
            {"action": "read", "source": "forum", "query": "https://example.test"},
            (),
            web_module._RouterDecision("stop", "web_read", ""),
        ),
        (
            {"action": "read", "source": "forum", "query": "not-a-url"},
            ("web_read", "web_search"),
            web_module._RouterDecision("stop", "web_read", ""),
        ),
        (
            {"action": "read", "source": "nhentai", "query": "652244"},
            enabled_sources,
            web_module._RouterDecision("read", "nhentai", "652244"),
        ),
        (
            {"action": "read", "source": "nhentai", "query": "652244"},
            ("web_read", "web_search"),
            web_module._RouterDecision("stop", "web_read", ""),
        ),
        (
            {
                "action": "read",
                "source": "bilibili",
                "query": "https://www.bilibili.com/video/BV1CqV266EJY/",
            },
            enabled_sources,
            web_module._RouterDecision(
                "read",
                "bilibili",
                "https://www.bilibili.com/video/BV1CqV266EJY/",
            ),
        ),
        (
            {
                "action": "read",
                "source": "bilibili",
                "query": "https://www.bilibili.com/video/BV1CqV266EJY/",
            },
            ("web_read", "web_search"),
            web_module._RouterDecision("stop", "web_read", ""),
        ),
        (
            {"action": "read", "source": "web_read", "query": ""},
            ("web_read", "web_search"),
            web_module._RouterDecision("search", "web_search", "fallback query"),
        ),
        (
            {"action": "read", "source": "web_read", "query": ""},
            ("web_read",),
            web_module._RouterDecision("stop", "web_read", ""),
        ),
    ]

    for raw_decision, valid_sources, expected_decision in cases:
        decision = web_module._normalize_router_decision(
            raw_decision,
            fallback_query="fallback query",
            valid_sources=valid_sources,
            source_actions=source_actions,
        )

        assert decision == expected_decision

    decision_without_search_action = web_module._normalize_router_decision(
        {"action": "search", "source": "forum", "query": "api docs"},
        fallback_query="fallback query",
        valid_sources=("web_read", "web_search"),
        source_actions={"web_read": ("read",)},
    )

    assert decision_without_search_action == web_module._RouterDecision(
        "stop",
        "web_read",
        "",
    )


@pytest.mark.asyncio
async def test_web_agent3_generator_uses_final_enabled_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Generator should preserve valid final source route decisions."""
    fake_llm = SimpleNamespace(
        ainvoke=AsyncMock(side_effect=[
            AIMessage(
                content=json.dumps({
                    "action": "read",
                    "source": "web_read",
                    "query": "https://example.test/page",
                }),
            ),
            AIMessage(
                content=json.dumps({
                    "action": "search",
                    "source": "web_search",
                    "query": "api docs",
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
                    "action": "stop",
                    "source": "web_search",
                    "query": "ignored",
                }),
            ),
        ]),
    )
    monkeypatch.setattr(agent_module, "_generator_llm", fake_llm)
    monkeypatch.setattr(
        agent_module,
        "_SUBAGENT_NAMES",
        ("web_read", "web_search", "nhentai"),
    )
    monkeypatch.setattr(
        agent_module,
        "_SUBAGENT_SUPPORTED_ACTIONS",
        {
            "web_read": ("read",),
            "web_search": ("search",),
            "nhentai": ("read", "search"),
        },
    )
    cases = [
        (
            "Web-evidence: summarize https://example.test/page",
            {
                "action": "read",
                "source": "web_read",
                "query": "https://example.test/page",
            },
        ),
        (
            "Web-evidence: search public API docs",
            {"action": "search", "source": "web_search", "query": "api docs"},
        ),
        (
            "Web-evidence: lookup nhentai 652244 metadata",
            {"action": "read", "source": "nhentai", "query": "652244"},
        ),
        (
            "Web-evidence: stop when done",
            {
                "action": "stop",
                "source": "web_read",
                "query": "",
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
    """Invalid router output should degrade to enabled web_search when possible."""
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
    monkeypatch.setattr(agent_module, "_SUBAGENT_NAMES", ("web_read", "web_search"))
    monkeypatch.setattr(
        agent_module,
        "_SUBAGENT_SUPPORTED_ACTIONS",
        {"web_read": ("read",), "web_search": ("search",)},
    )
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
        "source": "web_search",
        "query": "Web-evidence: search public docs for current API behavior",
    }


@pytest.mark.asyncio
async def test_web_agent3_executor_dispatches_edge_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Executor should preserve final source dispatch."""
    fake_search = SimpleNamespace(ainvoke=AsyncMock(return_value="search body"))
    fake_read = SimpleNamespace(ainvoke=AsyncMock(return_value="page body"))
    monkeypatch.setitem(
        provider_module._source_subagent_package._SUBAGENTS,
        "web_search",
        web_search_subagent,
    )
    monkeypatch.setitem(
        provider_module._source_subagent_package._SUBAGENTS,
        "web_read",
        web_read_subagent,
    )
    monkeypatch.setattr(web_search_subagent.searxng_tools, "web_search", fake_search)
    monkeypatch.setattr(web_read_subagent.searxng_tools, "web_url_read", fake_read)

    search_result = await web_module._execute_source_decision(
        web_module._RouterDecision(
            action="search",
            source="web_search",
            query="official docs latest release",
        )
    )
    read_result = await web_module._execute_source_decision(
        web_module._RouterDecision(
            action="read",
            source="web_read",
            query="https://example.test/docs",
        )
    )

    fake_search.ainvoke.assert_awaited_once_with({
        "query": "official docs latest release",
    })
    assert fake_read.ainvoke.await_args_list[0].args[0] == {
        "url": "https://example.test/docs",
    }
    assert search_result == "search body"
    assert read_result == "page body"


@pytest.mark.asyncio
async def test_web_agent3_stop_bypasses_source_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stop decisions should be handled by the graph executor itself."""
    execute_decision = AsyncMock(return_value="unexpected")
    monkeypatch.setattr(agent_module, "_execute_source_decision", execute_decision)
    state = {
        "router_decision": {
            "action": "stop",
            "source": "web_read",
            "query": "",
        },
        "observations": [],
    }

    update = await agent_module._tool_call_executor(state)

    execute_decision.assert_not_awaited()
    record = json.loads(update["messages"][0].content)
    assert record == {
        "action": "stop",
        "source": "web_read",
        "query": "",
        "result": {
            "status": "stopped",
            "source": "web_read",
            "action": "stop",
            "query": "",
            "message": "Router stopped without another web action.",
        },
    }
    assert update["observations"] == [record]
