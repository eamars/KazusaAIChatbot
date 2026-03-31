"""Tests for the Speech Agent."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agents.speech_agent import _build_agent_context, speech_agent
from state import AgentResult, SupervisorPlan


# ── _build_agent_context unit tests ─────────────────────────────────


class TestBuildAgentContext:
    def test_empty_results_and_directive(self):
        assert _build_agent_context([], "") == ""

    def test_directive_only(self):
        ctx = _build_agent_context([], "Respond casually.")
        assert "[Supervisor directive]" in ctx
        assert "Respond casually." in ctx

    def test_single_success_result(self):
        results = [AgentResult(
            agent="web_search_agent",
            status="success",
            summary="Tokyo is 18°C.",
            tool_history=[],
        )]
        ctx = _build_agent_context(results, "Mention the weather.")
        assert "web_search_agent (success)" in ctx
        assert "Tokyo is 18°C." in ctx
        assert "Mention the weather." in ctx

    def test_error_result(self):
        results = [AgentResult(
            agent="web_search_agent",
            status="error",
            summary="Search failed: timeout",
            tool_history=[],
        )]
        ctx = _build_agent_context(results, "Apologize for the failure.")
        assert "web_search_agent (FAILED)" in ctx
        assert "Search failed: timeout" in ctx

    def test_multiple_results(self):
        results = [
            AgentResult(agent="a", status="success", summary="Result A", tool_history=[]),
            AgentResult(agent="b", status="error", summary="Error B", tool_history=[]),
        ]
        ctx = _build_agent_context(results, "Combine both.")
        assert "a (success)" in ctx
        assert "b (FAILED)" in ctx


# ── speech_agent integration tests ──────────────────────────────────


@pytest.mark.asyncio
async def test_speech_agent_basic_response():
    """Speech agent generates a reply from assembled messages."""
    state = {
        "llm_messages": [
            SystemMessage(content="You are Zara."),
            HumanMessage(content="Hello!"),
        ],
        "supervisor_plan": SupervisorPlan(agents=[], speech_directive="Respond casually."),
        "agent_results": [],
    }

    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(
        return_value=AIMessage(content="Hey there, Commander.")
    )

    with patch("agents.speech_agent._get_llm", return_value=mock_llm):
        result = await speech_agent(state)

    assert result["response"] == "Hey there, Commander."
    mock_llm.ainvoke.assert_called_once()


@pytest.mark.asyncio
async def test_speech_agent_incorporates_agent_results():
    """Agent results are injected into the system prompt."""
    state = {
        "llm_messages": [
            SystemMessage(content="You are Zara."),
            HumanMessage(content="What's the weather?"),
        ],
        "supervisor_plan": SupervisorPlan(
            agents=["web_search_agent"],
            speech_directive="Mention the weather casually.",
        ),
        "agent_results": [AgentResult(
            agent="web_search_agent",
            status="success",
            summary="Tokyo is 18°C, partly cloudy.",
            tool_history=[],
        )],
    }

    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(
        return_value=AIMessage(content="Hmm, 18 degrees in Tokyo. Not bad.")
    )

    with patch("agents.speech_agent._get_llm", return_value=mock_llm):
        result = await speech_agent(state)

    assert result["response"] == "Hmm, 18 degrees in Tokyo. Not bad."

    # Verify the system prompt was enriched with agent context
    call_args = mock_llm.ainvoke.call_args[0][0]
    # After _prepare_messages folds system into first human, check the merged content
    merged_content = call_args[0].content
    assert "web_search_agent (success)" in merged_content
    assert "Tokyo is 18°C" in merged_content
    assert "Mention the weather casually" in merged_content


@pytest.mark.asyncio
async def test_speech_agent_handles_error_result():
    """Speech agent should still generate a reply when an agent failed."""
    state = {
        "llm_messages": [
            SystemMessage(content="You are Zara."),
            HumanMessage(content="Search for news."),
        ],
        "supervisor_plan": SupervisorPlan(
            agents=["web_search_agent"],
            speech_directive="Apologize if the search failed.",
        ),
        "agent_results": [AgentResult(
            agent="web_search_agent",
            status="error",
            summary="Web search failed: context limit exceeded",
            tool_history=[],
        )],
    }

    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(
        return_value=AIMessage(content="I tried looking that up, but something went wrong. Sorry, Commander.")
    )

    with patch("agents.speech_agent._get_llm", return_value=mock_llm):
        result = await speech_agent(state)

    assert "sorry" in result["response"].lower()


@pytest.mark.asyncio
async def test_speech_agent_empty_messages():
    state = {"llm_messages": []}
    result = await speech_agent(state)
    assert result["response"] == "..."


@pytest.mark.asyncio
async def test_speech_agent_llm_failure():
    """LLM crash should result in a fallback response."""
    state = {
        "llm_messages": [
            SystemMessage(content="You are Zara."),
            HumanMessage(content="Hello!"),
        ],
    }

    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(side_effect=Exception("LLM down"))

    with patch("agents.speech_agent._get_llm", return_value=mock_llm):
        result = await speech_agent(state)

    assert result["response"] == "*stays silent*"


@pytest.mark.asyncio
async def test_speech_agent_no_supervisor_plan():
    """Speech agent works even without a supervisor plan (backward compat)."""
    state = {
        "llm_messages": [
            SystemMessage(content="You are Zara."),
            HumanMessage(content="Hello!"),
        ],
    }

    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(
        return_value=AIMessage(content="Greetings.")
    )

    with patch("agents.speech_agent._get_llm", return_value=mock_llm):
        result = await speech_agent(state)

    assert result["response"] == "Greetings."
