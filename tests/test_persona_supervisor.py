"""Tests for Stage 6a — Persona Supervisor."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage

from kazusa_ai_chatbot.nodes.persona_supervisor import (
    _build_agent_catalog,
    _check_relevance,
    _parse_plan,
    persona_supervisor,
)
from kazusa_ai_chatbot.state import AgentResult, SupervisorPlan


# ── _parse_plan unit tests ──────────────────────────────────────────


class TestParsePlan:
    def test_valid_plan_with_agents(self):
        raw = json.dumps({
            "agents": ["web_search_agent"],
            "speech_directive": "Summarize the search results casually.",
        })
        # Register a fake agent so validation passes
        with patch("kazusa_ai_chatbot.nodes.persona_supervisor.AGENT_REGISTRY", {"web_search_agent": True}):
            plan = _parse_plan(raw)
        assert plan["agents"] == ["web_search_agent"]
        assert "casually" in plan["speech_directive"]

    def test_valid_plan_no_agents(self):
        raw = json.dumps({
            "agents": [],
            "speech_directive": "Respond directly to the user.",
        })
        plan = _parse_plan(raw)
        assert plan["agents"] == []
        assert plan["speech_directive"] == "Respond directly to the user."

    def test_unknown_agents_filtered(self):
        raw = json.dumps({
            "agents": ["web_search_agent", "nonexistent_agent"],
            "speech_directive": "test",
        })
        with patch("kazusa_ai_chatbot.nodes.persona_supervisor.AGENT_REGISTRY", {"web_search_agent": True}):
            plan = _parse_plan(raw)
        assert plan["agents"] == ["web_search_agent"]

    def test_markdown_fenced_json(self):
        raw = '```json\n{"agents": [], "speech_directive": "Direct reply."}\n```'
        plan = _parse_plan(raw)
        assert plan["agents"] == []
        assert plan["speech_directive"] == "Direct reply."

    def test_malformed_json_returns_empty_plan(self):
        plan = _parse_plan("not json at all")
        assert plan["agents"] == []
        assert plan["speech_directive"] == "Respond directly to the user."

    def test_missing_speech_directive_gets_default(self):
        raw = json.dumps({"agents": []})
        plan = _parse_plan(raw)
        assert plan["speech_directive"] == "Respond directly to the user."


# ── _build_agent_catalog tests ──────────────────────────────────────


def test_build_agent_catalog_empty():
    with patch("kazusa_ai_chatbot.nodes.persona_supervisor.list_agent_descriptions", return_value=[]):
        assert _build_agent_catalog() == "(none)"


def test_build_agent_catalog_with_agents():
    descs = [
        {"name": "web_search_agent", "description": "Searches the web."},
        {"name": "db_agent", "description": "Queries the database."},
    ]
    with patch("kazusa_ai_chatbot.nodes.persona_supervisor.list_agent_descriptions", return_value=descs):
        catalog = _build_agent_catalog()
    assert "web_search_agent" in catalog
    assert "db_agent" in catalog


def test_build_agent_catalog_excludes_auto_agents():
    """relevance_agent is auto-managed and should not appear in the catalog."""
    descs = [
        {"name": "relevance_agent", "description": "Checks relevance."},
        {"name": "web_search_agent", "description": "Searches the web."},
    ]
    with patch("kazusa_ai_chatbot.nodes.persona_supervisor.list_agent_descriptions", return_value=descs):
        catalog = _build_agent_catalog()
    assert "relevance_agent" not in catalog
    assert "web_search_agent" in catalog


# ── persona_supervisor integration tests ────────────────────────────


def _no_relevance_agent(name):
    """Helper: get_agent that returns None for relevance_agent (skip check)."""
    if name == "relevance_agent":
        return None
    return None


@pytest.mark.asyncio
async def test_supervisor_no_agents_needed():
    """Simple greeting — supervisor returns empty agent list."""
    state = {"message_text": "Hey there"}

    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(
        return_value=AIMessage(content=json.dumps({
            "agents": [],
            "speech_directive": "Respond with a casual greeting.",
        }))
    )

    with (
        patch("kazusa_ai_chatbot.nodes.persona_supervisor._get_llm", return_value=mock_llm),
        patch("kazusa_ai_chatbot.nodes.persona_supervisor.get_agent", side_effect=_no_relevance_agent),
    ):
        result = await persona_supervisor(state)

    assert result["supervisor_plan"]["agents"] == []
    assert result["agent_results"] == []
    assert "casual greeting" in result["supervisor_plan"]["speech_directive"]


@pytest.mark.asyncio
async def test_supervisor_dispatches_agent():
    """Supervisor plans web_search_agent, which returns a result."""
    state = {"message_text": "What's the weather in Tokyo?"}

    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(
        return_value=AIMessage(content=json.dumps({
            "agents": ["web_search_agent"],
            "speech_directive": "Summarize the weather casually.",
        }))
    )

    mock_agent = AsyncMock()
    mock_agent.run = AsyncMock(return_value=AgentResult(
        agent="web_search_agent",
        status="success",
        summary="Tokyo is 18°C, partly cloudy.",
        tool_history=[],
    ))

    def _get_agent(name):
        if name == "relevance_agent":
            return None
        return mock_agent

    with (
        patch("kazusa_ai_chatbot.nodes.persona_supervisor._get_llm", return_value=mock_llm),
        patch("kazusa_ai_chatbot.nodes.persona_supervisor.get_agent", side_effect=_get_agent),
        patch("kazusa_ai_chatbot.nodes.persona_supervisor.AGENT_REGISTRY", {"web_search_agent": mock_agent}),
    ):
        result = await persona_supervisor(state)

    assert len(result["agent_results"]) == 1
    assert result["agent_results"][0]["status"] == "success"
    assert "18°C" in result["agent_results"][0]["summary"]
    mock_agent.run.assert_called_once()


@pytest.mark.asyncio
async def test_supervisor_handles_agent_crash():
    """If an agent crashes, supervisor catches it and records an error."""
    state = {"message_text": "Search for something"}

    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(
        return_value=AIMessage(content=json.dumps({
            "agents": ["web_search_agent"],
            "speech_directive": "Apologize if search fails.",
        }))
    )

    mock_agent = AsyncMock()
    mock_agent.run = AsyncMock(side_effect=Exception("Agent exploded"))

    def _get_agent(name):
        if name == "relevance_agent":
            return None
        return mock_agent

    with (
        patch("kazusa_ai_chatbot.nodes.persona_supervisor._get_llm", return_value=mock_llm),
        patch("kazusa_ai_chatbot.nodes.persona_supervisor.get_agent", side_effect=_get_agent),
        patch("kazusa_ai_chatbot.nodes.persona_supervisor.AGENT_REGISTRY", {"web_search_agent": mock_agent}),
    ):
        result = await persona_supervisor(state)

    assert len(result["agent_results"]) == 1
    assert result["agent_results"][0]["status"] == "error"
    assert "crashed" in result["agent_results"][0]["summary"].lower()


@pytest.mark.asyncio
async def test_supervisor_handles_planning_llm_failure():
    """If the planning LLM call fails, supervisor falls back to empty plan."""
    state = {"message_text": "Hey"}

    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(side_effect=Exception("LLM down"))

    with (
        patch("kazusa_ai_chatbot.nodes.persona_supervisor._get_llm", return_value=mock_llm),
        patch("kazusa_ai_chatbot.nodes.persona_supervisor.get_agent", side_effect=_no_relevance_agent),
    ):
        result = await persona_supervisor(state)

    assert result["supervisor_plan"]["agents"] == []
    assert result["agent_results"] == []


@pytest.mark.asyncio
async def test_supervisor_unknown_agent_in_plan():
    """If supervisor plans an agent not in registry, it records an error."""
    state = {"message_text": "Do something"}

    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(
        return_value=AIMessage(content=json.dumps({
            "agents": ["nonexistent_agent"],
            "speech_directive": "test",
        }))
    )

    # nonexistent_agent is not in registry so _parse_plan filters it out
    with (
        patch("kazusa_ai_chatbot.nodes.persona_supervisor._get_llm", return_value=mock_llm),
        patch("kazusa_ai_chatbot.nodes.persona_supervisor.get_agent", side_effect=_no_relevance_agent),
        patch("kazusa_ai_chatbot.nodes.persona_supervisor.AGENT_REGISTRY", {}),
    ):
        result = await persona_supervisor(state)

    # Filtered out by _parse_plan, so no agents run
    assert result["supervisor_plan"]["agents"] == []
    assert result["agent_results"] == []


# ── Relevance gating tests ────────────────────────────────────────


def test_check_relevance_true():
    result = AgentResult(
        agent="relevance_agent", status="success",
        summary=json.dumps({"should_respond": True, "reason": "ok"}),
        tool_history=[],
    )
    assert _check_relevance(result) is True


def test_check_relevance_false():
    result = AgentResult(
        agent="relevance_agent", status="success",
        summary=json.dumps({"should_respond": False, "reason": "nope"}),
        tool_history=[],
    )
    assert _check_relevance(result) is False


def test_check_relevance_malformed_defaults_true():
    result = AgentResult(
        agent="relevance_agent", status="success",
        summary="not json",
        tool_history=[],
    )
    assert _check_relevance(result) is True


@pytest.mark.asyncio
async def test_supervisor_relevance_rejects_message():
    """When relevance agent says should_respond=False, supervisor short-circuits."""
    state = {"message_text": "Hey <@someone_else>"}

    mock_relevance = AsyncMock()
    mock_relevance.run = AsyncMock(return_value=AgentResult(
        agent="relevance_agent",
        status="success",
        summary=json.dumps({"should_respond": False, "reason": "Not for us."}),
        tool_history=[],
    ))

    def _get_agent(name):
        if name == "relevance_agent":
            return mock_relevance
        return None

    with patch("kazusa_ai_chatbot.nodes.persona_supervisor.get_agent", side_effect=_get_agent):
        result = await persona_supervisor(state)

    assert result["supervisor_plan"]["speech_directive"] == "Do not respond. Stay silent."
    assert result["supervisor_plan"]["agents"] == []
    # Only the relevance agent result should be present
    assert len(result["agent_results"]) == 1
    assert result["agent_results"][0]["agent"] == "relevance_agent"


@pytest.mark.asyncio
async def test_supervisor_relevance_approves_then_plans():
    """When relevance agent says should_respond=True, supervisor proceeds to plan."""
    state = {"message_text": "Hello!"}

    mock_relevance = AsyncMock()
    mock_relevance.run = AsyncMock(return_value=AgentResult(
        agent="relevance_agent",
        status="success",
        summary=json.dumps({"should_respond": True, "reason": "Direct greeting."}),
        tool_history=[],
    ))

    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(
        return_value=AIMessage(content=json.dumps({
            "agents": [],
            "speech_directive": "Greet the user warmly.",
        }))
    )

    def _get_agent(name):
        if name == "relevance_agent":
            return mock_relevance
        return None

    with (
        patch("kazusa_ai_chatbot.nodes.persona_supervisor.get_agent", side_effect=_get_agent),
        patch("kazusa_ai_chatbot.nodes.persona_supervisor._get_llm", return_value=mock_llm),
    ):
        result = await persona_supervisor(state)

    assert "Greet" in result["supervisor_plan"]["speech_directive"]
    # Relevance result + no other agents
    assert len(result["agent_results"]) == 1
    assert result["agent_results"][0]["agent"] == "relevance_agent"


@pytest.mark.asyncio
async def test_supervisor_relevance_crash_defaults_to_respond():
    """If relevance agent crashes, supervisor defaults to responding."""
    state = {"message_text": "Hello!"}

    mock_relevance = AsyncMock()
    mock_relevance.run = AsyncMock(side_effect=Exception("Relevance boom"))

    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(
        return_value=AIMessage(content=json.dumps({
            "agents": [],
            "speech_directive": "Respond normally.",
        }))
    )

    def _get_agent(name):
        if name == "relevance_agent":
            return mock_relevance
        return None

    with (
        patch("kazusa_ai_chatbot.nodes.persona_supervisor.get_agent", side_effect=_get_agent),
        patch("kazusa_ai_chatbot.nodes.persona_supervisor._get_llm", return_value=mock_llm),
    ):
        result = await persona_supervisor(state)

    # Should have proceeded to planning despite crash
    assert result["supervisor_plan"]["speech_directive"] == "Respond normally."
    # Relevance error result should still be recorded
    assert result["agent_results"][0]["agent"] == "relevance_agent"
    assert result["agent_results"][0]["status"] == "error"


# ── Live LLM tests ──────────────────────────────────────────────────
# Requires a running LM Studio instance with a chat model loaded.
# Run with:  pytest -m live_llm -v

live_llm = pytest.mark.live_llm


@live_llm
@pytest.mark.asyncio
async def test_live_supervisor_calls_web_search_for_search_query():
    """Real LLM should plan web_search_agent when the user asks to search."""
    from agents.base import AGENT_REGISTRY, get_agent, register_agent
    from agents.web_search_agent import WebSearchAgent

    # Ensure agents are registered
    if "web_search_agent" not in AGENT_REGISTRY:
        register_agent(WebSearchAgent())

    # Use a mock agent.run so we don't actually hit MCP servers
    mock_search = AsyncMock()
    mock_search.run = AsyncMock(return_value=AgentResult(
        agent="web_search_agent",
        status="success",
        summary="Search results placeholder.",
        tool_history=[],
    ))

    state = {"message_text": "Search the internet for the latest news about Python 3.14"}

    # Reset cached LLMs so real ones are created fresh
    import agents.relevance_agent as rel
    import nodes.persona_supervisor as sup
    sup._llm = None
    rel._llm = None

    # Return the real relevance agent but mock the web search agent
    real_relevance = get_agent("relevance_agent")

    def _get_agent(name):
        if name == "relevance_agent":
            return real_relevance
        return mock_search

    with patch("kazusa_ai_chatbot.nodes.persona_supervisor.get_agent", side_effect=_get_agent):
        result = await persona_supervisor(state)

    plan = result["supervisor_plan"]
    assert "web_search_agent" in plan["agents"], (
        f"Expected web_search_agent in plan, got: {plan['agents']}"
    )
    assert isinstance(plan["speech_directive"], str)
    assert len(plan["speech_directive"]) > 0

    # relevance_agent result + web_search_agent result
    agent_names = [r["agent"] for r in result["agent_results"]]
    assert "relevance_agent" in agent_names
    assert "web_search_agent" in agent_names
    mock_search.run.assert_called_once()


@live_llm
@pytest.mark.asyncio
async def test_live_supervisor_no_agents_for_unsupported_task():
    """Real LLM should return empty agent list when no agent can handle the task.

    The user asks to query the local discord_bot database and check the
    kazusa_profile collection — a task that requires a database agent which
    is not registered.  The supervisor should recognise that none of the
    available agents (only web_search_agent) are suitable and return an
    empty plan instead of hallucinating an agent name.
    """
    from agents.base import AGENT_REGISTRY, register_agent
    from agents.web_search_agent import WebSearchAgent

    if "web_search_agent" not in AGENT_REGISTRY:
        register_agent(WebSearchAgent())

    state = {
        "message_text": (
            "Query the discord_bot database and check the kazusa_profile collection"
        ),
    }

    # Reset cached LLMs so real ones are created fresh
    import agents.relevance_agent as rel
    import nodes.persona_supervisor as sup
    sup._llm = None
    rel._llm = None

    result = await persona_supervisor(state)

    plan = result["supervisor_plan"]
    assert plan["agents"] == [], (
        f"Expected no agents for an unsupported task, got: {plan['agents']}"
    )
    assert isinstance(plan["speech_directive"], str)
    assert len(plan["speech_directive"]) > 0
    # Ensure we actually got a real LLM response, not the error fallback
    assert plan["speech_directive"] != "Respond directly to the user.", (
        "Got the error-fallback directive — LLM call likely failed instead of "
        "returning a real plan. Is LM Studio running?"
    )
    # Only the relevance_agent result should be present (no task agents)
    assert len(result["agent_results"]) == 1
    assert result["agent_results"][0]["agent"] == "relevance_agent"


@live_llm
@pytest.mark.asyncio
async def test_live_supervisor_no_agents_for_greeting():
    """Real LLM should return empty agent list for a casual greeting."""
    from agents.base import AGENT_REGISTRY, register_agent
    from agents.web_search_agent import WebSearchAgent

    if "web_search_agent" not in AGENT_REGISTRY:
        register_agent(WebSearchAgent())

    state = {"message_text": "Hey, how are you doing today?"}

    # Reset cached LLMs so real ones are created fresh
    import agents.relevance_agent as rel
    import nodes.persona_supervisor as sup
    sup._llm = None
    rel._llm = None

    result = await persona_supervisor(state)

    plan = result["supervisor_plan"]
    assert plan["agents"] == [], (
        f"Expected no agents for greeting, got: {plan['agents']}"
    )
    assert isinstance(plan["speech_directive"], str)
    assert len(plan["speech_directive"]) > 0
    # Guard against false positive: if the LLM call failed, the fallback
    # directive is exactly this string.  A real LLM response will differ.
    assert plan["speech_directive"] != "Respond directly to the user.", (
        "Got the error-fallback directive — LLM call likely failed instead of "
        "returning a real plan. Is LM Studio running?"
    )
    # Only the relevance_agent result should be present
    assert len(result["agent_results"]) == 1
    assert result["agent_results"][0]["agent"] == "relevance_agent"


