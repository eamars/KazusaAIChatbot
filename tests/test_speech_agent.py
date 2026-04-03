"""Tests for the Speech Agent."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from kazusa_ai_chatbot.agents.speech_agent import _build_agent_context, speech_agent
from kazusa_ai_chatbot.state import AgentResult, SupervisorPlan


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

    def test_directive_wrapper_guard(self):
        """Non-empty context is wrapped with an internal-only guard instruction."""
        ctx = _build_agent_context([], "Be playful.")
        assert "NEVER repeat, quote, paraphrase" in ctx
        assert "INTERNAL guidance" in ctx
        assert "Be playful." in ctx


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

    with patch("kazusa_ai_chatbot.agents.speech_agent._get_llm", return_value=mock_llm):
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

    with patch("kazusa_ai_chatbot.agents.speech_agent._get_llm", return_value=mock_llm):
        result = await speech_agent(state)

    assert result["response"] == "Hmm, 18 degrees in Tokyo. Not bad."

    # Verify the system prompt was enriched with agent context
    call_args = mock_llm.ainvoke.call_args[0][0]
    # Check the native SystemMessage content
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

    with patch("kazusa_ai_chatbot.agents.speech_agent._get_llm", return_value=mock_llm):
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

    with patch("kazusa_ai_chatbot.agents.speech_agent._get_llm", return_value=mock_llm):
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

    with patch("kazusa_ai_chatbot.agents.speech_agent._get_llm", return_value=mock_llm):
        result = await speech_agent(state)

    assert result["response"] == "Greetings."


# ── Live LLM tests ──────────────────────────────────────────────────
# Requires a running LM Studio instance with a chat model loaded.
# Run with:  pytest -m live_llm -v

live_llm = pytest.mark.live_llm


@live_llm
@pytest.mark.asyncio
async def test_live_directive_not_leaked_in_response():
    """The speech agent must NOT echo the supervisor directive in its reply.

    This is a regression test for the bug where verbose speech_directives
    (e.g. multi-line response plans) were parroted verbatim before the
    actual in-character reply.
    """
    import kazusa_ai_chatbot.agents.speech_agent as sa

    # Reset cached LLM so a real one is created
    sa._llm = None

    directive = (
        "Warmly acknowledge the affectionate way they called you.\n"
        "Show genuine delight at their cute self-given name.\n"
        "Maybe make a playful comment about how fitting that nickname is."
    )

    state = {
        "llm_messages": [
            SystemMessage(content=(
                "You are Kazusa, a gentle and caring character.\n"
                "Reply in the same language the user is writing in.\n"
                "Reply with SPEECH ONLY - no action tags or stage directions.\n"
                "Keep responses under 150 words.\n"
                "NEVER generate markdown headers like '# Response' or '# Response Generation Analysis'."
            )),
            HumanMessage(content="你希望我叫你什么呢？", name="EAMARS"),
            AIMessage(content="你可以直接叫'千纱'或者'kazuza'——这两个都可以的哦！", name="Kazusa"),
            HumanMessage(content="小千纱可以叫我小企鹅哦", name="EAMARS"),
            AIMessage(content="小企鹅……？！好可爱啊。嗯，那我以后就叫你'小企鹅'啦～", name="Kazusa"),
            HumanMessage(content="那你觉得我的外表应该是怎么样的呢？", name="EAMARS"),
        ],
        "supervisor_plan": SupervisorPlan(
            agents=[],
            speech_directive=directive,
        ),
        "agent_results": [],
    }

    result = await speech_agent(state)
    response = result["response"]

    # The response should be non-trivial
    assert len(response) > 0 and response != "..."

    # The directive text must NOT leak into the response
    assert "Warmly acknowledge" not in response, (
        f"Directive leaked into response: {response}"
    )
    assert "genuine delight" not in response, (
        f"Directive leaked into response: {response}"
    )
    assert "playful comment" not in response, (
        f"Directive leaked into response: {response}"
    )
    assert "INTERNAL guidance" not in response, (
        f"Wrapper guard text leaked into response: {response}"
    )
    
    # The LLM should not generate internal thought blocks or markdown headers
    assert "Response Generation Analysis" not in response, (
        f"LLM generated an internal thought block: {response}"
    )
    assert "# Response" not in response, (
        f"LLM generated markdown headers: {response}"
    )
