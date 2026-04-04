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
def mock_assembler_state():
    return {
        "message_text": "Hello bot",
        "user_name": "Commander",
        "bot_id": "bot_001",
        "personality": {"name": "Zara", "description": "A calm strategist."},
        "user_memory": ["The user prefers to be called Commander"],
        "character_state": {"mood": "alert", "emotional_tone": "warm", "recent_events": ["Discussed patrol routes"]},
        "affinity": 650,
        "conversation_history": [
            {"role": "user", "user_id": "user_123", "name": "Commander", "content": "How are the patrols going?"},
            {"role": "assistant", "user_id": "bot_001", "name": "Zara", "content": "They are holding for now."},
        ],
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
    assert plan["instructions"] == {}
    assert plan["content_directive"] == "Say hello back."
    assert len(result["agent_results"]) == 0
    speech_brief = result["speech_brief"]
    assert speech_brief["personality"]["name"] == "Zara"
    assert speech_brief["user_input_brief"]["user_topic"] == "Greeting"
    assert speech_brief["response_brief"]["response_goal"] == "Say hello back."
    assert speech_brief["response_brief"]["tone_guidance"] == "Warm."
    assert "current_message" not in speech_brief
    mock_llm.ainvoke.assert_called_once()


@pytest.mark.asyncio
async def test_supervisor_dispatches_agent(mock_assembler_state):
    """Supervisor parses plan and invokes the requested agent."""
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content=json.dumps({
        "agents": ["web_search_agent"],
        "instructions": {
            "web_search_agent": {
                "command": "Search the web for the current weather.",
                "expected_response": "Return a short factual summary with the current conditions.",
            }
        },
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
    assert plan["instructions"]["web_search_agent"]["command"] == "Search the web for the current weather."
    assert plan["instructions"]["web_search_agent"]["expected_response"] == "Return a short factual summary with the current conditions."

    assert len(result["agent_results"]) == 1
    assert result["agent_results"][0]["agent"] == "web_search_agent"
    assert result["agent_results"][0]["status"] == "success"
    assert "It is sunny." in result["speech_brief"]["response_brief"]["key_points_to_cover"]
    mock_agent.run.assert_awaited_once_with(
        mock_assembler_state,
        "Hello bot",
        "Search the web for the current weather.",
        "Return a short factual summary with the current conditions.",
    )


@pytest.mark.asyncio
async def test_supervisor_handles_agent_crash(mock_assembler_state):
    """If a dispatched agent raises an exception, it is recorded as an error."""
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content=json.dumps({
        "agents": ["web_search_agent"],
        "instructions": {
            "web_search_agent": {
                "command": "Find the relevant weather lookup.",
                "expected_response": "Return the key weather facts only.",
            }
        },
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
    assert result["speech_brief"]["response_brief"]["response_goal"].startswith("Respond directly")


@pytest.mark.asyncio
async def test_supervisor_unknown_agent_in_plan(mock_assembler_state):
    """If the LLM hallucinates an agent, it is stripped by _parse_plan."""
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content=json.dumps({
        "agents": ["web_search_agent", "hallucinated_agent"],
        "instructions": {
            "web_search_agent": {
                "command": "Search the weather.",
                "expected_response": "Return a concise weather brief.",
            },
            "hallucinated_agent": {
                "command": "Do something impossible.",
                "expected_response": "Return whatever.",
            },
        },
        "content_directive": "Report weather.",
        "emotion_directive": "Neutral."
    })))

    with patch("kazusa_ai_chatbot.nodes.persona_supervisor._get_llm", return_value=mock_llm):
        result = await persona_supervisor(mock_assembler_state)

    plan = result["supervisor_plan"]
    assert "hallucinated_agent" not in plan["agents"]
    assert "hallucinated_agent" not in plan["instructions"]


def test_parse_plan_preserves_instructions():
    raw = json.dumps({
        "agents": ["db_lookup_agent"],
        "instructions": {
            "db_lookup_agent": {
                "command": "Look up any remembered nickname for this user.",
                "expected_response": "Return only a concise remembered nickname summary.",
            }
        },
        "content_directive": "Use the remembered nickname if available.",
        "emotion_directive": "Warm.",
    })

    with patch("kazusa_ai_chatbot.nodes.persona_supervisor.AGENT_REGISTRY", {"db_lookup_agent": True}):
        plan = _parse_plan(raw)

    assert plan["agents"] == ["db_lookup_agent"]
    assert plan["instructions"] == {
        "db_lookup_agent": {
            "command": "Look up any remembered nickname for this user.",
            "expected_response": "Return only a concise remembered nickname summary.",
        }
    }


@pytest.mark.asyncio
async def test_supervisor_dispatches_db_lookup_agent(mock_assembler_state):
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content=json.dumps({
        "agents": ["db_lookup_agent"],
        "instructions": {
            "db_lookup_agent": {
                "command": "Look up recent conversation context about patrol routes and any remembered user preferences.",
                "expected_response": "Return a short memory-oriented brief without raw transcripts.",
            }
        },
        "content_directive": "Answer using relevant remembered details if any are found.",
        "emotion_directive": "Thoughtful.",
    })))

    mock_agent = AsyncMock()
    mock_agent.run = AsyncMock(return_value=AgentResult(
        agent="db_lookup_agent", status="success", summary="The user prefers to be called Commander.", tool_history=[]
    ))

    def _get_agent(name):
        return mock_agent if name == "db_lookup_agent" else None

    with patch("kazusa_ai_chatbot.nodes.persona_supervisor._get_llm", return_value=mock_llm), \
         patch("kazusa_ai_chatbot.nodes.persona_supervisor.get_agent", side_effect=_get_agent), \
         patch("kazusa_ai_chatbot.nodes.persona_supervisor.AGENT_REGISTRY", {"db_lookup_agent": True}):

        result = await persona_supervisor(mock_assembler_state)

    plan = result["supervisor_plan"]
    assert plan["agents"] == ["db_lookup_agent"]
    assert "db_lookup_agent" in plan["instructions"]
    assert result["agent_results"][0]["agent"] == "db_lookup_agent"
    assert "Commander" in result["speech_brief"]["response_brief"]["key_points_to_cover"][-1]
    mock_agent.run.assert_awaited_once_with(
        mock_assembler_state,
        "Hello bot",
        "Look up recent conversation context about patrol routes and any remembered user preferences.",
        "Return a short memory-oriented brief without raw transcripts.",
    )


@pytest.mark.asyncio
async def test_supervisor_short_circuits_if_not_should_respond(mock_assembler_ignore_state):
    """If the assembler says not to respond, supervisor stays silent without planning."""
    mock_llm = MagicMock()

    with patch("kazusa_ai_chatbot.nodes.persona_supervisor._get_llm", return_value=mock_llm):
        result = await persona_supervisor(mock_assembler_ignore_state)

    plan = result["supervisor_plan"]
    assert plan["agents"] == []
    assert plan["content_directive"] == "Do not respond. Stay silent."
    assert result["speech_brief"]["response_brief"]["should_respond"] is False
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
    assert isinstance(plan["instructions"], dict)

    agent_names = [r["agent"] for r in result["agent_results"]]
    assert "web_search_agent" in agent_names
    mock_search.run.assert_called_once()


@live_llm
@pytest.mark.asyncio
async def test_live_supervisor_calls_db_lookup_for_database_task():
    """Real LLM should plan db_lookup_agent for a memory/database lookup request."""
    from kazusa_ai_chatbot.agents.base import AGENT_REGISTRY, register_agent
    from kazusa_ai_chatbot.agents.db_lookup_agent import DBLookupAgent
    from kazusa_ai_chatbot.agents.web_search_agent import WebSearchAgent

    if "web_search_agent" not in AGENT_REGISTRY:
        register_agent(WebSearchAgent())
    if "db_lookup_agent" not in AGENT_REGISTRY:
        register_agent(DBLookupAgent())

    mock_db = AsyncMock()
    mock_db.run = AsyncMock(return_value=AgentResult(
        agent="db_lookup_agent",
        status="success",
        summary="Database lookup placeholder.",
        tool_history=[],
    ))

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

    def _get_agent(name):
        return mock_db

    with patch("kazusa_ai_chatbot.nodes.persona_supervisor.get_agent", side_effect=_get_agent):
        result = await persona_supervisor(state)

    plan = result["supervisor_plan"]
    assert "db_lookup_agent" in plan["agents"], (
        f"Expected db_lookup_agent in plan, got: {plan['agents']}"
    )
    assert isinstance(plan["content_directive"], str)
    assert len(plan["content_directive"]) > 0
    assert len(result["agent_results"]) == 1


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


