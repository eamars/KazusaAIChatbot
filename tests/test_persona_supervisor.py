"""Tests for Stage 6a — Persona Supervisor."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage

from kazusa_ai_chatbot.nodes.persona_supervisor import (
    _build_agent_catalog,
    _build_personality_block,
    _build_character_state_block,
    _build_affinity_block,
    _parse_plan,
    persona_supervisor,
)
from kazusa_ai_chatbot.state import AgentResult, AssemblerOutput, SupervisorPlan


# ── _parse_plan unit tests ──────────────────────────────────────────


class TestBuildPersonalityBlock:
    def test_empty_personality(self):
        result = _build_personality_block({})
        assert result["description"] == "You are a helpful role-play character."

    def test_with_name(self):
        result = _build_personality_block({"name": "Zara"})
        assert result["name"] == "Zara"

    def test_with_traits(self):
        result = _build_personality_block({
            "name": "Zara",
            "age": "20",
            "tone": "sarcastic"
        })
        assert result["name"] == "Zara"
        assert result["age"] == "20"
        assert result["tone"] == "sarcastic"

    def test_with_custom_fields(self):
        result = _build_personality_block({
            "name": "Zara",
            "likes": ["apples", "swords"],
            "dislikes": "rain"
        })
        assert result["name"] == "Zara"
        assert "likes" in result["extra_traits"]
        assert "dislikes" in result["extra_traits"]


class TestBuildCharacterStateBlock:
    def test_empty_state(self):
        assert _build_character_state_block({}) == {}

    def test_with_mood_and_tone(self, sample_character_state):
        result = _build_character_state_block(sample_character_state)
        assert result["mood"] == "alert"
        assert result["emotional_tone"] == "guarded"

    def test_with_recent_events(self, sample_character_state):
        result = _build_character_state_block(sample_character_state)
        assert "Shadow wolves attacked the northern gate" in result["recent_events"]

    def test_mood_only(self):
        result = _build_character_state_block({"mood": "happy"})
        assert result["mood"] == "happy"


class TestBuildAffinityBlock:
    def test_hostile(self):
        result = _build_affinity_block(100)
        assert result["level"] == "Hostile"
        assert "contempt" in result["instruction"] or "dismissive" in result["instruction"]

    def test_cold(self):
        result = _build_affinity_block(300)
        assert result["level"] == "Cold"
        assert "curt" in result["instruction"] or "short" in result["instruction"]

    def test_neutral(self):
        result = _build_affinity_block(500)
        assert result["level"] == "Neutral"

    def test_friendly(self):
        result = _build_affinity_block(700)
        assert result["level"] == "Friendly"
        assert "warmer" in result["instruction"]

    def test_devoted(self):
        result = _build_affinity_block(900)
        assert result["level"] == "Devoted"
        assert "deeply loyal" in result["instruction"]


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


# ── persona_supervisor integration tests ────────────────────────────


@pytest.fixture
def sample_speech_state():
    return {
        "supervisor_plan": {
            "agents": [],
            "content_directive": "Acknowledge the user.",
            "emotion_directive": "Warm and friendly."
        },
        "agent_results": [],
        "speech_human_data": {
            "current_message": {
                "speaker": "Commander",
                "speaker_id": "user_123",
                "message": "Hello"
            },
            "context": {
                "personality": {"name": "Zara"},
            }
        }
    }


@pytest.fixture
def mock_assembler_state():
    return {
        "message_text": "Hello bot",
        "assembler_output": AssemblerOutput(
            channel_topic="General",
            user_topic="Greeting",
            should_respond=True
        )
    }


@pytest.fixture
def mock_assembler_ignore_state():
    return {
        "message_text": "ignore this",
        "assembler_output": AssemblerOutput(
            channel_topic="Random",
            user_topic="Noise",
            should_respond=False,
        )
    }

@pytest.mark.asyncio
async def test_supervisor_no_agents_needed(mock_assembler_state):
    """Supervisor calls LLM and executes no agents."""
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content=json.dumps({
        "agents": [],
        "content_directive": "Say hello back.",
        "emotion_directive": "Warm."
    })))

    with patch("kazusa_ai_chatbot.nodes.persona_supervisor._get_llm", return_value=mock_llm):
        result = await persona_supervisor(mock_assembler_state)

    plan = result["supervisor_plan"]
    assert plan["agents"] == []
    assert plan["content_directive"] == "Say hello back."
    assert len(result["agent_results"]) == 0
    mock_llm.ainvoke.assert_called_once()


@pytest.mark.asyncio
async def test_supervisor_dispatches_agent(mock_assembler_state):
    """Supervisor parses plan and invokes the requested agent."""
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content=json.dumps({
        "agents": ["web_search_agent"],
        "content_directive": "Report weather.",
        "emotion_directive": "Neutral."
    })))

    mock_agent = AsyncMock()
    mock_agent.run = AsyncMock(return_value=AgentResult(
        agent="web_search_agent", status="success", summary="It is sunny.", tool_history=[]
    ))

    def _get_agent(name):
        return mock_agent if name == "web_search_agent" else None

    with patch("kazusa_ai_chatbot.nodes.persona_supervisor._get_llm", return_value=mock_llm), \
         patch("kazusa_ai_chatbot.nodes.persona_supervisor.get_agent", side_effect=_get_agent), \
         patch("kazusa_ai_chatbot.nodes.persona_supervisor.AGENT_REGISTRY", {"web_search_agent": True}):

        result = await persona_supervisor(mock_assembler_state)

    plan = result["supervisor_plan"]
    assert plan["agents"] == ["web_search_agent"]

    assert len(result["agent_results"]) == 1
    assert result["agent_results"][0]["agent"] == "web_search_agent"
    assert result["agent_results"][0]["status"] == "success"


@pytest.mark.asyncio
async def test_supervisor_handles_agent_crash(mock_assembler_state):
    """If a dispatched agent raises an exception, it is recorded as an error."""
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content=json.dumps({
        "agents": ["web_search_agent"],
        "content_directive": "Report weather.",
        "emotion_directive": "Neutral."
    })))

    mock_agent = AsyncMock()
    mock_agent.run = AsyncMock(side_effect=ValueError("Timeout error"))

    def _get_agent(name):
        return mock_agent if name == "web_search_agent" else None

    with patch("kazusa_ai_chatbot.nodes.persona_supervisor._get_llm", return_value=mock_llm), \
         patch("kazusa_ai_chatbot.nodes.persona_supervisor.get_agent", side_effect=_get_agent), \
         patch("kazusa_ai_chatbot.nodes.persona_supervisor.AGENT_REGISTRY", {"web_search_agent": True}):

        result = await persona_supervisor(mock_assembler_state)

    assert len(result["agent_results"]) == 1
    error_result = result["agent_results"][0]
    assert error_result["agent"] == "web_search_agent"
    assert error_result["status"] == "error"
    assert "Timeout error" in error_result["summary"]


@pytest.mark.asyncio
async def test_supervisor_handles_planning_llm_failure(mock_assembler_state):
    """If the planning LLM call fails, default to no agents and direct response."""
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(side_effect=Exception("API down"))

    with patch("kazusa_ai_chatbot.nodes.persona_supervisor._get_llm", return_value=mock_llm):
        result = await persona_supervisor(mock_assembler_state)

    plan = result["supervisor_plan"]
    assert plan["agents"] == []
    assert "Respond directly" in plan["content_directive"]


@pytest.mark.asyncio
async def test_supervisor_unknown_agent_in_plan(mock_assembler_state):
    """If the LLM hallucinates an agent, it is stripped by _parse_plan."""
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content=json.dumps({
        "agents": ["web_search_agent", "hallucinated_agent"],
        "content_directive": "Report weather.",
        "emotion_directive": "Neutral."
    })))

    with patch("kazusa_ai_chatbot.nodes.persona_supervisor._get_llm", return_value=mock_llm):
        result = await persona_supervisor(mock_assembler_state)

    plan = result["supervisor_plan"]
    assert "hallucinated_agent" not in plan["agents"]


@pytest.mark.asyncio
async def test_supervisor_short_circuits_if_not_should_respond(mock_assembler_ignore_state):
    """If the assembler says not to respond, supervisor stays silent without planning."""
    mock_llm = MagicMock()

    with patch("kazusa_ai_chatbot.nodes.persona_supervisor._get_llm", return_value=mock_llm):
        result = await persona_supervisor(mock_assembler_ignore_state)

    plan = result["supervisor_plan"]
    assert plan["agents"] == []
    assert plan["content_directive"] == "Do not respond. Stay silent."
    mock_llm.ainvoke.assert_not_called()  # No planning LLM call made


# ── Live LLM tests ──────────────────────────────────────────────────
# Requires a running LM Studio instance with a chat model loaded.
# Run with:  pytest -m live_llm -v

live_llm = pytest.mark.live_llm


@live_llm
@pytest.mark.asyncio
async def test_live_supervisor_calls_web_search_for_search_query():
    """Real LLM should plan web_search_agent when the user asks to search."""
    from kazusa_ai_chatbot.agents.base import AGENT_REGISTRY, get_agent, register_agent
    from kazusa_ai_chatbot.agents.web_search_agent import WebSearchAgent

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

    state = {
        "message_text": "Search the internet for the latest news about Python 3.14",
        "assembler_output": AssemblerOutput(
            channel_topic="Python",
            user_topic="Search request",
            should_respond=True
        )
    }

    # Reset cached LLMs so real ones are created fresh
    import kazusa_ai_chatbot.nodes.persona_supervisor as sup
    sup._llm = None

    def _get_agent(name):
        return mock_search

    with patch("kazusa_ai_chatbot.nodes.persona_supervisor.get_agent", side_effect=_get_agent):
        result = await persona_supervisor(state)

    plan = result["supervisor_plan"]
    assert "web_search_agent" in plan["agents"], (
        f"Expected web_search_agent in plan, got: {plan['agents']}"
    )
    assert isinstance(plan["content_directive"], str)
    assert len(plan["content_directive"]) > 0

    agent_names = [r["agent"] for r in result["agent_results"]]
    assert "web_search_agent" in agent_names
    mock_search.run.assert_called_once()


@live_llm
@pytest.mark.asyncio
async def test_live_supervisor_no_agents_for_unsupported_task():
    """Real LLM should return empty agent list when no agent can handle the task."""
    from kazusa_ai_chatbot.agents.base import AGENT_REGISTRY, register_agent
    from kazusa_ai_chatbot.agents.web_search_agent import WebSearchAgent

    if "web_search_agent" not in AGENT_REGISTRY:
        register_agent(WebSearchAgent())

    state = {
        "message_text": "Query the discord_bot database and check the kazusa_profile collection",
        "assembler_output": AssemblerOutput(
            channel_topic="Database",
            user_topic="DB Query",
            should_respond=True
        )
    }

    import kazusa_ai_chatbot.nodes.persona_supervisor as sup
    sup._llm = None

    result = await persona_supervisor(state)

    plan = result["supervisor_plan"]
    assert plan["agents"] == [], (
        f"Expected no agents for an unsupported task, got: {plan['agents']}"
    )
    assert isinstance(plan["content_directive"], str)
    assert len(plan["content_directive"]) > 0
    assert len(result["agent_results"]) == 0


@live_llm
@pytest.mark.asyncio
async def test_live_supervisor_no_agents_for_greeting():
    """Real LLM should return empty agent list for a casual greeting."""
    from kazusa_ai_chatbot.agents.base import AGENT_REGISTRY, register_agent
    from kazusa_ai_chatbot.agents.web_search_agent import WebSearchAgent

    if "web_search_agent" not in AGENT_REGISTRY:
        register_agent(WebSearchAgent())

    state = {
        "message_text": "Hey, how are you doing today?",
        "assembler_output": AssemblerOutput(
            channel_topic="General",
            user_topic="Greeting",
            should_respond=True
        )
    }

    import kazusa_ai_chatbot.nodes.persona_supervisor as sup
    sup._llm = None

    result = await persona_supervisor(state)

    plan = result["supervisor_plan"]
    assert plan["agents"] == [], (
        f"Expected no agents for greeting, got: {plan['agents']}"
    )
    assert isinstance(plan["content_directive"], str)
    assert len(plan["content_directive"]) > 0
    assert len(result["agent_results"]) == 0


