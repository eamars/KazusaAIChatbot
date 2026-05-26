"""Focused deterministic tests for the RAG2 web_agent3 helper."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from kazusa_ai_chatbot.rag import web_agent3 as web_module
from kazusa_ai_chatbot.rag.web_agent3 import WebAgent3
from kazusa_ai_chatbot.rag.web_agent3 import agent as agent_module
from kazusa_ai_chatbot.rag.web_agent3 import providers as provider_module
from kazusa_ai_chatbot.rag.web_agent3 import searxng_tools as searxng_module
from kazusa_ai_chatbot.time_boundary import build_turn_clock_from_storage_utc


def test_web_agent3_is_subpackage_with_icd() -> None:
    """web_agent3 should be an importable package with a local ICD."""
    package_path = Path(web_module.__file__).parent

    assert package_path.name == "web_agent3"
    assert (package_path / "README.md").is_file()


@pytest.mark.asyncio
async def test_web_agent3_search_tool_delegates_to_existing_searxng_facility() -> None:
    """web_search should call the existing MCP-backed SearXNG search tool."""
    with patch("kazusa_ai_chatbot.rag.web_agent3.searxng_tools.mcp_manager") as mock_mcp:
        mock_mcp.call_tool = AsyncMock(return_value="search results")
        result = await searxng_module.web_search.ainvoke({
            "query": "test query",
            "pageno": 2,
            "time_range": "month",
            "language": "en",
        })

    mock_mcp.call_tool.assert_awaited_once()
    call_args = mock_mcp.call_tool.await_args
    assert call_args.args[0] == "mcp-searxng__searxng_web_search"
    assert call_args.args[1] == {
        "query": "test query",
        "pageno": 2,
        "time_range": "month",
        "language": "en",
        "safesearch": 0,
    }
    assert result == "search results"


@pytest.mark.asyncio
async def test_web_agent3_url_read_tool_delegates_to_existing_searxng_facility() -> None:
    """web_url_read should call the existing MCP-backed URL reader tool."""
    with patch("kazusa_ai_chatbot.rag.web_agent3.searxng_tools.mcp_manager") as mock_mcp:
        mock_mcp.call_tool = AsyncMock(return_value="page body")
        result = await searxng_module.web_url_read.ainvoke({
            "url": "https://example.test",
            "startChar": 10,
            "maxLength": 120,
            "section": "Usage",
            "paragraphRange": "1-3",
            "readHeadings": True,
        })

    mock_mcp.call_tool.assert_awaited_once()
    call_args = mock_mcp.call_tool.await_args
    assert call_args.args[0] == "mcp-searxng__web_url_read"
    assert call_args.args[1] == {
        "url": "https://example.test",
        "startChar": 10,
        "maxLength": 120,
        "section": "Usage",
        "paragraphRange": "1-3",
        "readHeadings": True,
    }
    assert result == "page body"


@pytest.mark.asyncio
async def test_web_agent3_url_read_omits_zero_max_length() -> None:
    """A zero maxLength should preserve the old no-explicit-limit behavior."""
    with patch("kazusa_ai_chatbot.rag.web_agent3.searxng_tools.mcp_manager") as mock_mcp:
        mock_mcp.call_tool = AsyncMock(return_value="page body")
        await searxng_module.web_url_read.ainvoke({
            "url": "https://example.test",
            "maxLength": 0,
        })

    call_args = mock_mcp.call_tool.await_args
    assert "maxLength" not in call_args.args[1]


def test_web_agent3_router_output_parsing_is_minimal() -> None:
    """Router parsing should keep only action, source, and query semantics."""
    decision = web_module._normalize_router_decision(
        {
            "action": "read",
            "source": "nhentai",
            "query": "652244",
            "reason": "ignored",
            "api_params": {"id": 652244},
        },
        fallback_query="fallback search",
    )

    assert decision == web_module._RouterDecision(
        action="read",
        source="nhentai",
        query="652244",
    )


def test_web_agent3_router_prompt_omits_input_format_headers() -> None:
    """Router prompt should not use the retired input-format headers."""
    forbidden_headers = [
        "# " + "\u8f93\u5165\u683c\u5f0f",
        "# " + "Input Format",
    ]

    for header in forbidden_headers:
        assert header not in agent_module._WEB_AGENT3_GENERATOR_PROMPT


def test_web_agent3_router_prompt_uses_project_prompt_style() -> None:
    """Router prompt should use the static project prompt style."""
    prompt = agent_module._WEB_AGENT3_GENERATOR_PROMPT

    assert "# 来源原则" in prompt
    assert "# 审计步骤" in prompt
    assert "# 输出格式" in prompt
    assert "# 输出契约" not in prompt
    assert "source adapter roster" not in prompt
    assert agent_module._WEB_AGENT3_SOURCE_TOOLS_TEXT in prompt
    assert '"source": "string"' in prompt
    hardcoded_source_schema = '"source": "{}"'.format(
        "|".join(["generic", "bilibili", "youtube", "nhentai"])
    )
    assert hardcoded_source_schema not in prompt


def test_web_agent3_router_uses_subagent_generation_rules() -> None:
    """Router query guidance should come from source subagent descriptions."""
    generic_subagent = importlib.import_module(
        "kazusa_ai_chatbot.rag.web_agent3.subagent.generic"
    )

    assert "search:" in generic_subagent.DESCRIPTION
    assert "read:" in generic_subagent.DESCRIPTION
    assert "site:" in generic_subagent.DESCRIPTION
    assert "snippet" in generic_subagent.DESCRIPTION.lower()
    assert generic_subagent.DESCRIPTION in agent_module._WEB_AGENT3_SOURCE_TOOLS_TEXT
    assert "遵循所选来源描述" in agent_module._WEB_AGENT3_GENERATOR_PROMPT


def test_web_agent3_router_source_text_omits_execution_details() -> None:
    """Router source descriptions should expose capability, not implementation."""
    source_text = agent_module._WEB_AGENT3_SOURCE_TOOLS_TEXT
    forbidden_terms = [
        'SearXNG',
        '当前占位',
        '不会调用',
        '回退',
        'fallback',
    ]

    for forbidden_term in forbidden_terms:
        assert forbidden_term not in source_text


def test_web_agent3_source_subagents_are_discovered_from_subagent_package() -> None:
    """Source subagents should be discovered from per-source modules."""
    package_path = Path(web_module.__file__).parent
    subagent_path = package_path / "subagent"
    retired_source_file = package_path / "source_subagents.py"

    assert subagent_path.is_dir()
    assert not retired_source_file.exists()

    source_module = importlib.import_module(
        "kazusa_ai_chatbot.rag.web_agent3.subagent"
    )
    expected_sources = {"generic", "bilibili", "youtube", "nhentai"}

    assert Path(source_module.__file__).name == "__init__.py"
    assert set(source_module._SUBAGENTS) == expected_sources
    assert set(source_module._SUBAGENT_DESCRIPTIONS) == expected_sources
    assert list(source_module._SUBAGENT_DESCRIPTIONS) == sorted(expected_sources)

    for source in expected_sources:
        module_path = subagent_path / f"{source}.py"
        source_subagent = importlib.import_module(
            f"kazusa_ai_chatbot.rag.web_agent3.subagent.{source}"
        )

        assert module_path.is_file()
        assert source_subagent.SOURCE == source
        assert source_subagent.DESCRIPTION
        assert callable(source_subagent.execute)

    assert not hasattr(provider_module, "_SOURCE_SUBAGENTS")
    assert not hasattr(provider_module, "_SOURCE_ADAPTER_DESCRIPTIONS")
    assert not hasattr(provider_module, "_SOURCE_ADAPTERS")


@pytest.mark.asyncio
async def test_web_agent3_generator_outputs_router_decision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Generator should parse a strict action/source/query router decision."""
    fake_llm = SimpleNamespace(
        ainvoke=AsyncMock(return_value=AIMessage(
            content='{"action": "search", "source": "youtube", "query": "demo"}',
        )),
    )
    monkeypatch.setattr(agent_module, "_generator_llm", fake_llm)
    state = {
        "task": "Find a YouTube demo.",
        "context": {"platform": "debug"},
        "messages": [HumanMessage(content="start")],
        "observations": [{"action": "search", "source": "generic"}],
        "evaluator_feedback": "read a YouTube result next",
        "prompt_timestamp": "2026-05-25 21:30 (Monday)",
    }

    update = await agent_module._tool_call_generator(state)

    messages = fake_llm.ainvoke.await_args.args[0]
    system_prompt = messages[0].content
    payload = json.loads(messages[1].content)
    assert "2026-05-25" not in system_prompt
    assert payload["reference_time"] == "2026-05-25 21:30 (Monday)"
    assert payload["evaluator_feedback"] == "read a YouTube result next"
    assert update["router_decision"] == {
        "action": "search",
        "source": "youtube",
        "query": "demo",
    }


@pytest.mark.asyncio
async def test_web_agent3_generic_search_receives_query_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Generic search should pass router query directly to SearXNG."""
    generic_subagent = importlib.import_module(
        "kazusa_ai_chatbot.rag.web_agent3.subagent.generic"
    )
    fake_search = SimpleNamespace(ainvoke=AsyncMock(return_value="search body"))
    monkeypatch.setattr(generic_subagent.searxng_tools, "web_search", fake_search)
    decision = web_module._RouterDecision(
        action="search",
        source="generic",
        query="local tool router demo web agent architecture",
    )

    result = await web_module._execute_source_decision(decision)

    fake_search.ainvoke.assert_awaited_once_with({
        "query": "local tool router demo web agent architecture",
    })
    assert result == "search body"


@pytest.mark.asyncio
async def test_web_agent3_placeholder_sources_are_dedicated_no_result_subagents() -> None:
    """Bilibili and YouTube should remain separate no-result subagents."""
    source_module = importlib.import_module(
        "kazusa_ai_chatbot.rag.web_agent3.subagent"
    )
    source_subagents = source_module._SUBAGENTS
    generic_subagent = source_subagents["generic"]

    assert source_subagents["nhentai"] is not generic_subagent

    for source in ("bilibili", "youtube"):
        decision = web_module._RouterDecision(
            action="search",
            source=source,
            query="raw-source-target",
        )

        result = await web_module._execute_source_decision(decision)

        assert result == {
            "status": "no_search_data",
            "source": source,
            "action": "search",
            "query": "raw-source-target",
            "message": "Source subagent placeholder has no search data.",
        }


@pytest.mark.asyncio
async def test_web_agent3_specialized_adapters_return_no_search_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Placeholder source adapters should not fall back to generic search data."""
    generic_subagent = importlib.import_module(
        "kazusa_ai_chatbot.rag.web_agent3.subagent.generic"
    )
    fake_read = SimpleNamespace(ainvoke=AsyncMock(return_value="page body"))
    fake_search = SimpleNamespace(ainvoke=AsyncMock(return_value="search body"))
    monkeypatch.setattr(generic_subagent.searxng_tools, "web_url_read", fake_read)
    monkeypatch.setattr(generic_subagent.searxng_tools, "web_search", fake_search)

    for source in ("bilibili", "youtube"):
        fake_read.ainvoke.reset_mock()
        fake_search.ainvoke.reset_mock()
        decision = web_module._RouterDecision(
            action="read",
            source=source,
            query="652244",
        )

        result = await web_module._execute_source_decision(decision)

        fake_read.ainvoke.assert_not_awaited()
        fake_search.ainvoke.assert_not_awaited()
        assert result == {
            "status": "no_search_data",
            "source": source,
            "action": "read",
            "query": "652244",
            "message": "Source subagent placeholder has no search data.",
        }

    assert web_module._DUMMY_PROVIDER_FIXME.startswith("FIXME(web_agent3)")


@pytest.mark.asyncio
async def test_web_agent3_executor_records_minimal_observation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Executor should keep prompt-facing state minimal."""
    execute_decision = AsyncMock(return_value="generic page body")
    monkeypatch.setattr(agent_module, "_execute_source_decision", execute_decision)
    state = {
        "router_decision": {
            "action": "read",
            "source": "nhentai",
            "query": "652244",
        },
        "observations": [],
    }

    update = await agent_module._tool_call_executor(state)

    execute_decision.assert_awaited_once_with(web_module._RouterDecision(
        action="read",
        source="nhentai",
        query="652244",
    ))
    record = json.loads(update["messages"][0].content)
    assert record == {
        "action": "read",
        "source": "nhentai",
        "query": "652244",
        "result": "generic page body",
    }
    assert update["observations"] == [record]


@pytest.mark.asyncio
async def test_web_agent3_evaluator_continues_with_feedback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Evaluator feedback should feed the next router iteration."""
    fake_llm = SimpleNamespace(
        ainvoke=AsyncMock(return_value=AIMessage(
            content='{"should_stop": false, "feedback": "read the first URL"}',
        )),
    )
    monkeypatch.setattr(agent_module, "_evaluator_llm", fake_llm)
    state = {
        "task": "Find current docs.",
        "expected_response": "official status",
        "messages": [],
        "observations": [
            {
                "action": "search",
                "source": "generic",
                "query": "current docs",
                "result": "search result",
            }
        ],
        "retry": 0,
        "prompt_timestamp": "2026-05-25 21:30 (Monday)",
    }

    update = await agent_module._tool_call_evaluator(state)

    messages = fake_llm.ainvoke.await_args.args[0]
    system_prompt = messages[0].content
    payload = json.loads(messages[1].content)
    assert "`retry`" not in system_prompt
    assert payload["reference_time"] == "2026-05-25 21:30 (Monday)"
    assert payload["call_history"][0]["query"] == "current docs"
    assert "retry" not in payload
    assert update["should_stop"] is False
    assert update["evaluator_feedback"] == "read the first URL"
    assert update["retry"] == 1


@pytest.mark.asyncio
async def test_web_agent3_run_subgraph_returns_expected_keys() -> None:
    """_run_subgraph should map compiled graph state to the public result shape."""
    mock_result = {
        "final_status": "success",
        "final_reason": "found info",
        "final_response": "Here are the results",
        "final_is_empty_result": False,
        "knowledge_metadata": {},
    }

    with patch("kazusa_ai_chatbot.rag.web_agent3.agent.StateGraph") as state_graph:
        graph_builder = MagicMock()
        graph_builder.compile.return_value.ainvoke = AsyncMock(return_value=mock_result)
        state_graph.return_value = graph_builder

        result = await agent_module._run_subgraph(
            task="search something",
            context={},
            expected_response="relevant results",
            local_prompt_timestamp="2026-04-27 12:00",
        )

    sub_state = graph_builder.compile.return_value.ainvoke.await_args.args[0]
    assert sub_state["prompt_timestamp"] == "2026-04-27 12:00"
    assert sub_state["router_decision"] == {
        "action": "stop",
        "source": "generic",
        "query": "",
    }
    assert result == {
        "status": "success",
        "reason": "found info",
        "response": "Here are the results",
        "is_empty_result": False,
        "knowledge_metadata": {},
    }


@pytest.mark.asyncio
async def test_web_agent3_run_preserves_base_helper_contract() -> None:
    """WebAgent3.run should expose the BaseRAG helper contract."""
    with patch(
        "kazusa_ai_chatbot.rag.web_agent3.agent._run_subgraph",
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
            "2026-04-27T00:00:00+00:00",
        )
        result = await WebAgent3().run(
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


@pytest.mark.asyncio
async def test_web_agent3_finalizer_comparison_helper_returns_public_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Comparison finalizer helper should preserve the web_agent3 result shape."""
    fake_finalizer = AsyncMock(return_value={
        "final_status": "success",
        "final_reason": "enough",
        "final_response": "证据：example",
        "final_is_empty_result": False,
    })
    monkeypatch.setattr(agent_module, "_tool_call_finalizer", fake_finalizer)
    tool_result = web_module._WebToolResult(
        resolved=True,
        operation="search",
        query="example query",
        url=None,
        title=None,
        description=None,
        content="",
        items=[
            web_module._WebSearchItem(
                title="Example",
                url="https://example.test",
                snippet="snippet",
                source="fixture",
            )
        ],
        delegation_reason=None,
        missing_context=[],
        error=None,
    )

    result = await web_module._finalize_web_agent3_result(
        task="Web-evidence: example",
        context={},
        local_prompt_timestamp="2026-05-26 12:00",
        tool_result=tool_result,
        evaluator_feedback="snippet-only evidence",
        evidence_limitations=["snippet_only"],
        max_status="partial",
    )

    assert result == {
        "status": "partial",
        "reason": "enough",
        "response": "证据：example",
        "is_empty_result": False,
        "knowledge_metadata": {
            "evidence_limitations": ["snippet_only"],
            "max_status": "partial",
        },
    }


@pytest.mark.asyncio
async def test_web_agent3_finalizer_payload_uses_clean_feedback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Finalizer should not receive evaluator message wrapper metadata."""
    fake_llm = SimpleNamespace(
        ainvoke=AsyncMock(return_value=AIMessage(
            content=(
                '{"response": "证据：example", "score": 90, '
                '"reason": "enough", "is_empty_result": false}'
            ),
        )),
    )
    monkeypatch.setattr(agent_module, "_finalizer_llm", fake_llm)
    feedback_payload = {
        "feedback": "read official page",
        "source": "evaluator",
        "evidence_limitations": ["snippet_only"],
    }
    state = {
        "task": "Web-evidence: example",
        "expected_response": "official evidence",
        "messages": [
            ToolMessage(
                content='{"result": "official page"}',
                tool_call_id="tool-1",
            ),
            HumanMessage(
                content=json.dumps(feedback_payload, ensure_ascii=False),
                name="evaluator",
            ),
        ],
        "evaluator_feedback": "read official page",
    }

    await agent_module._tool_call_finalizer(state)

    messages = fake_llm.ainvoke.await_args.args[0]
    finalizer_payload = json.loads(messages[1].content)
    assert finalizer_payload["evaluator_feedback"] == "read official page"
    assert "source" not in messages[1].content
    assert "evidence_limitations" not in messages[1].content
