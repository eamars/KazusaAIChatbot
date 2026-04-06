"""Tests for Stage 6a — Persona Supervisor."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage

from kazusa_ai_chatbot.nodes.persona_supervisor import (
    _build_agent_catalog,
    _build_personality_block,
    _build_affinity_block,
    _parse_action,
    _parse_memory_check,
    _parse_plan,
    _parse_synthesis,
    persona_supervisor,
)
from kazusa_ai_chatbot.state import AgentInstruction, AgentResult, AssemblerOutput, SupervisorAction, SupervisorPlan, BotState
from kazusa_ai_chatbot.db import AFFINITY_DEFAULT


def _finish_action_json(**overrides):
    """Return a JSON string for a 'finish' evaluate response."""
    data = {
        "action": "finish",
        "agent": "",
        "instruction": {"command": "", "expected_response": ""},
        "reason": "Results are satisfactory.",
    }
    data.update(overrides)
    return json.dumps(data)


def _synthesis_json(topics=None, facts=None, emotion="Neutral."):
    """Return a JSON string for a synthesis response."""
    return json.dumps({
        "topics_to_cover": topics or [],
        "facts_to_cover": facts or [],
        "emotion_directive": emotion,
    })


def _no_store_json():
    """Return a JSON string for a memory-check response that declines to store."""
    return json.dumps({
        "should_store": False,
        "command": "",
        "expected_response": "",
        "reason": "Nothing worth remembering.",
    })


def _make_plan_eval_synth_llm(plan_json: dict, synth_topics=None, synth_facts=None, synth_emotion="Neutral."):
    """Create a mock LLM: plan → finish → memory_check(no) → synthesis."""
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(side_effect=[
        AIMessage(content=json.dumps(plan_json)),
        AIMessage(content=_finish_action_json()),
        AIMessage(content=_no_store_json()),
        AIMessage(content=_synthesis_json(synth_topics, synth_facts, synth_emotion)),
    ])
    return mock_llm


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


class TestBuildAffinityBlock:
    def test_hostile(self):
        result = _build_affinity_block(100)
        assert result["level"] == "Scornful"
        assert "contempt" in result["instruction"] or "dismissive" in result["instruction"]

    def test_cold(self):
        result = _build_affinity_block(300)
        assert result["level"] == "Reserved"
        assert "brief" in result["instruction"] or "professional" in result["instruction"]

    def test_neutral(self):
        result = _build_affinity_block(AFFINITY_DEFAULT)
        assert result["level"] == "Antagonistic"

    def test_friendly(self):
        result = _build_affinity_block(700)
        assert result["level"] == "Warm"
        assert "warmth" in result["instruction"] or "enthusiasm" in result["instruction"]

    def test_devoted(self):
        result = _build_affinity_block(900)
        assert result["level"] == "Protective"
        assert "protective" in result["instruction"] or "loyalty" in result["instruction"]


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
            should_respond=True,
            reason_to_respond="User greeted the bot",
            use_reply_feature=False
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
            reason_to_respond="Message is noise and should be ignored",
            use_reply_feature=False
        )
    }

@pytest.mark.asyncio
async def test_supervisor_no_agents_needed(mock_assembler_state):
    """Supervisor calls LLM and executes no agents."""
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(side_effect=[
        # 1) Plan: no agents
        AIMessage(content=json.dumps({
            "agents": [],
            "response_language": "English",
            "topics_to_cover": ["Say hello back."],
            "facts_to_cover": ["The user prefers to be called Commander."],
            "emotion_directive": "Warm."
        })),
        # 2) Memory check: nothing to store
        AIMessage(content=_no_store_json()),
    ])

    with patch("kazusa_ai_chatbot.nodes.persona_supervisor._get_llm", return_value=mock_llm):
        result = await persona_supervisor(mock_assembler_state)

    plan = result["supervisor_plan"]
    assert plan["agents"] == []
    assert plan["instructions"] == {}
    assert plan["response_language"] == "English"
    assert plan["topics_to_cover"] == ["Say hello back."]
    assert plan["facts_to_cover"] == ["The user prefers to be called Commander."]
    assert len(result["agent_results"]) == 0
    speech_brief = result["speech_brief"]
    assert speech_brief["personality"]["name"] == "Zara"
    assert speech_brief["user_input_brief"]["user_topic"] == "Greeting"
    assert speech_brief["response_brief"]["response_language"] == "English"
    assert speech_brief["response_brief"]["topics_to_cover"] == ["Say hello back."]
    assert speech_brief["response_brief"]["facts_to_cover"] == ["The user prefers to be called Commander."]
    assert speech_brief["response_brief"]["tone_guidance"] == "Warm."
    assert "current_message" not in speech_brief
    # plan + memory_check = 2 LLM calls
    assert mock_llm.ainvoke.await_count == 2


@pytest.mark.asyncio
async def test_supervisor_passes_user_facts_into_planning_context(mock_assembler_state):
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(side_effect=[
        # 1) Plan
        AIMessage(content=json.dumps({
            "agents": [],
            "response_language": "English",
            "topics_to_cover": ["Reply normally."],
            "facts_to_cover": [],
            "emotion_directive": "Neutral.",
        })),
        # 2) Memory check
        AIMessage(content=_no_store_json()),
    ])

    with patch("kazusa_ai_chatbot.nodes.persona_supervisor._get_llm", return_value=mock_llm):
        await persona_supervisor(mock_assembler_state)

    # First call is the planning call
    planning_messages = mock_llm.ainvoke.await_args_list[0].args[0]
    planning_payload = json.loads(planning_messages[1].content)
    assert planning_payload["user_context"]["user_memory"] == [
        "The user prefers to be called Commander"
    ]


@pytest.mark.asyncio
async def test_supervisor_dispatches_agent(mock_assembler_state):
    """Supervisor parses plan, invokes the requested agent, evaluates, synthesizes, and finishes."""
    mock_llm = _make_plan_eval_synth_llm(
        {
            "agents": ["web_search_agent"],
            "instructions": {
                "web_search_agent": {
                    "command": "Search the web for the current weather.",
                    "expected_response": "Return a short factual summary with the current conditions.",
                }
            },
            "response_language": "English",
            "topics_to_cover": ["Report the current weather."],
            "facts_to_cover": ["State the current conditions clearly."],
            "emotion_directive": "Neutral."
        },
        synth_topics=["Report the current weather."],
        synth_facts=["It is currently sunny."],
    )

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
    # Facts come from synthesis, not raw agent output
    assert result["speech_brief"]["response_brief"]["topics_to_cover"] == ["Report the current weather."]
    assert "It is currently sunny." in result["speech_brief"]["response_brief"]["facts_to_cover"]
    mock_agent.run.assert_awaited_once_with(
        mock_assembler_state,
        "Search the web for the current weather.",
        "Return a short factual summary with the current conditions.",
    )
    # plan + evaluate + memory_check + synthesis = 4 LLM calls
    assert mock_llm.ainvoke.await_count == 4


@pytest.mark.asyncio
async def test_supervisor_handles_agent_crash(mock_assembler_state):
    """If a dispatched agent raises an exception, it is recorded as an error."""
    mock_llm = _make_plan_eval_synth_llm(
        {
            "agents": ["web_search_agent"],
            "instructions": {
                "web_search_agent": {
                    "command": "Find the relevant weather lookup.",
                    "expected_response": "Return the key weather facts only.",
                }
            },
            "response_language": "English",
            "topics_to_cover": ["Report the current weather."],
            "facts_to_cover": [],
            "emotion_directive": "Neutral."
        },
        synth_topics=["Report the current weather."],
        synth_facts=[],
    )

    mock_agent = AsyncMock()
    mock_agent.run = AsyncMock(side_effect=ValueError("Timeout error"))

    def _get_agent(name):
        return mock_agent if name == "web_search_agent" else None

    with patch("kazusa_ai_chatbot.nodes.persona_supervisor._get_llm", return_value=mock_llm), \
         patch("kazusa_ai_chatbot.nodes.persona_supervisor.get_agent", side_effect=_get_agent), \
         patch("kazusa_ai_chatbot.nodes.persona_supervisor.AGENT_REGISTRY", {"web_search_agent": True}):

        result = await persona_supervisor(mock_assembler_state)

    # The crashed agent result is present (evaluate still runs and finishes)
    error_results = [r for r in result["agent_results"] if r["status"] == "error"]
    assert len(error_results) >= 1
    assert error_results[0]["agent"] == "web_search_agent"
    assert "Timeout error" in error_results[0]["summary"]


@pytest.mark.asyncio
async def test_supervisor_handles_planning_llm_failure(mock_assembler_state):
    """If the planning LLM call fails, default to no agents and direct response."""
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(side_effect=Exception("API down"))

    with patch("kazusa_ai_chatbot.nodes.persona_supervisor._get_llm", return_value=mock_llm):
        result = await persona_supervisor(mock_assembler_state)

    plan = result["supervisor_plan"]
    assert plan["agents"] == []
    assert plan["topics_to_cover"]
    assert result["speech_brief"]["response_brief"]["response_goal"].startswith("Respond directly")


@pytest.mark.asyncio
async def test_supervisor_unknown_agent_in_plan(mock_assembler_state):
    """If the LLM hallucinates an agent, it is stripped by _parse_plan."""
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(side_effect=[
        # 1) Plan with hallucinated agent
        AIMessage(content=json.dumps({
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
            "response_language": "English",
            "topics_to_cover": ["Report the current weather."],
            "facts_to_cover": [],
            "emotion_directive": "Neutral."
        })),
        # 2) Memory check
        AIMessage(content=_no_store_json()),
    ])

    # Mock the web_search_agent to prevent real LLM calls
    mock_agent = AsyncMock()
    mock_agent.run = AsyncMock(return_value=AgentResult(
        agent="web_search_agent",
        status="success",
        summary="Weather search completed.",
        tool_history=[]
    ))

    def _get_agent(name):
        return mock_agent if name == "web_search_agent" else None

    with patch("kazusa_ai_chatbot.nodes.persona_supervisor._get_llm", return_value=mock_llm), \
         patch("kazusa_ai_chatbot.nodes.persona_supervisor.get_agent", side_effect=_get_agent), \
         patch("kazusa_ai_chatbot.nodes.persona_supervisor.AGENT_REGISTRY", {"web_search_agent": True}):
        result = await persona_supervisor(mock_assembler_state)

    plan = result["supervisor_plan"]
    assert "hallucinated_agent" not in plan["agents"]
    assert "hallucinated_agent" not in plan["instructions"]


# ── _parse_action unit tests ───────────────────────────────────────


class TestParseAction:
    def _make_plan(self):
        return SupervisorPlan(
            agents=["web_search_agent"],
            instructions={},
            response_language="English",
            topics_to_cover=["Original topic"],
            facts_to_cover=[],
            emotion_directive="Neutral.",
        )

    def test_parse_finish(self):
        raw = json.dumps({"action": "finish", "agent": "", "reason": "All good."})
        with patch("kazusa_ai_chatbot.nodes.persona_supervisor.AGENT_REGISTRY", {"web_search_agent": True}):
            action = _parse_action(raw, self._make_plan())
        assert action["action"] == "finish"

    def test_parse_retry(self):
        raw = json.dumps({
            "action": "retry",
            "agent": "web_search_agent",
            "instruction": {"command": "Try again", "expected_response": "Better results"},
            "reason": "First attempt was poor.",
        })
        with patch("kazusa_ai_chatbot.nodes.persona_supervisor.AGENT_REGISTRY", {"web_search_agent": True}):
            action = _parse_action(raw, self._make_plan())
        assert action["action"] == "retry"
        assert action["agent"] == "web_search_agent"
        assert action["instruction"]["command"] == "Try again"

    def test_parse_escalate(self):
        raw = json.dumps({
            "action": "escalate",
            "agent": "memory_agent",
            "instruction": {"command": "Recall memory", "expected_response": "Memory details"},
            "reason": "Need stored memory.",
        })
        with patch("kazusa_ai_chatbot.nodes.persona_supervisor.AGENT_REGISTRY", {"web_search_agent": True, "memory_agent": True}):
            action = _parse_action(raw, self._make_plan())
        assert action["action"] == "escalate"
        assert action["agent"] == "memory_agent"

    def test_invalid_agent_falls_back_to_finish(self):
        raw = json.dumps({"action": "retry", "agent": "nonexistent_agent", "reason": "Bad ref."})
        with patch("kazusa_ai_chatbot.nodes.persona_supervisor.AGENT_REGISTRY", {"web_search_agent": True}):
            action = _parse_action(raw, self._make_plan())
        assert action["action"] == "finish"

    def test_unknown_action_falls_back_to_finish(self):
        raw = json.dumps({"action": "dance", "agent": "", "reason": "???"})
        with patch("kazusa_ai_chatbot.nodes.persona_supervisor.AGENT_REGISTRY", {"web_search_agent": True}):
            action = _parse_action(raw, self._make_plan())
        assert action["action"] == "finish"

    def test_malformed_json_falls_back_to_finish(self):
        with patch("kazusa_ai_chatbot.nodes.persona_supervisor.AGENT_REGISTRY", {"web_search_agent": True}):
            action = _parse_action("not json at all", self._make_plan())
        assert action["action"] == "finish"

    def test_evaluate_updates_plan_directives(self):
        plan = self._make_plan()
        raw = json.dumps({
            "action": "finish",
            "agent": "",
            "reason": "Done.",
            "topics_to_cover": ["Updated topic"],
            "facts_to_cover": ["New fact"],
            "emotion_directive": "Warm.",
        })
        with patch("kazusa_ai_chatbot.nodes.persona_supervisor.AGENT_REGISTRY", {"web_search_agent": True}):
            _parse_action(raw, plan)
        assert plan["topics_to_cover"] == ["Updated topic"]
        assert plan["facts_to_cover"] == ["New fact"]
        assert plan["emotion_directive"] == "Warm."


# ── _parse_synthesis unit tests ────────────────────────────────────


class TestParseSynthesis:
    def _make_plan(self):
        return SupervisorPlan(
            agents=["web_search_agent"],
            instructions={},
            response_language="English",
            topics_to_cover=["Original topic"],
            facts_to_cover=["Original fact"],
            emotion_directive="Neutral.",
        )

    def test_updates_all_fields(self):
        plan = self._make_plan()
        raw = json.dumps({
            "topics_to_cover": ["Synthesized topic"],
            "facts_to_cover": ["Distilled fact A", "Distilled fact B"],
            "emotion_directive": "Warm.",
        })
        _parse_synthesis(raw, plan)
        assert plan["topics_to_cover"] == ["Synthesized topic"]
        assert plan["facts_to_cover"] == ["Distilled fact A", "Distilled fact B"]
        assert plan["emotion_directive"] == "Warm."

    def test_partial_update_keeps_other_fields(self):
        plan = self._make_plan()
        raw = json.dumps({"facts_to_cover": ["Only facts updated"]})
        _parse_synthesis(raw, plan)
        assert plan["topics_to_cover"] == ["Original topic"]
        assert plan["facts_to_cover"] == ["Only facts updated"]
        assert plan["emotion_directive"] == "Neutral."

    def test_malformed_json_keeps_plan_unchanged(self):
        plan = self._make_plan()
        _parse_synthesis("not json", plan)
        assert plan["topics_to_cover"] == ["Original topic"]
        assert plan["facts_to_cover"] == ["Original fact"]

    def test_empty_facts_clears_them(self):
        plan = self._make_plan()
        raw = json.dumps({"facts_to_cover": []})
        _parse_synthesis(raw, plan)
        assert plan["facts_to_cover"] == []

    def test_strips_markdown_fence(self):
        plan = self._make_plan()
        raw = '```json\n{"facts_to_cover": ["Clean fact"]}\n```'
        _parse_synthesis(raw, plan)
        assert plan["facts_to_cover"] == ["Clean fact"]


class TestParseMemoryCheck:
    def test_should_store_true(self):
        raw = json.dumps({
            "should_store": True,
            "command": "Store the user's favorite color as blue.",
            "expected_response": "Confirm the memory was saved.",
            "reason": "User explicitly asked to remember.",
        })
        result = _parse_memory_check(raw)
        assert result["should_store"] is True
        assert result["command"] == "Store the user's favorite color as blue."
        assert result["expected_response"] == "Confirm the memory was saved."
        assert result["reason"] == "User explicitly asked to remember."

    def test_should_store_false(self):
        raw = json.dumps({
            "should_store": False,
            "command": "",
            "expected_response": "",
            "reason": "Casual greeting, nothing to remember.",
        })
        result = _parse_memory_check(raw)
        assert result["should_store"] is False
        assert result["command"] == ""

    def test_malformed_json_returns_no_store(self):
        result = _parse_memory_check("not valid json {{{")
        assert result["should_store"] is False
        assert result["reason"] == ""  # Updated since our utility returns empty dict on failure

    def test_missing_fields_default_safely(self):
        raw = json.dumps({"should_store": True})
        result = _parse_memory_check(raw)
        assert result["should_store"] is False  # Updated: validation sets to False when command is empty
        assert result["command"] == ""
        assert result["expected_response"] == ""
        assert "[ERROR: Empty command provided]" in result["reason"]  # Check for error message

    def test_strips_markdown_fence(self):
        raw = '```json\n{"should_store": true, "command": "Save it.", "expected_response": "Done.", "reason": "Important."}\n```'
        result = _parse_memory_check(raw)
        assert result["should_store"] is True
        assert result["command"] == "Save it."


def test_parse_plan_preserves_instructions():
    raw = json.dumps({
        "agents": ["conversation_history_agent"],
        "instructions": {
            "conversation_history_agent": {
                "command": "Search prior chat history for the user's earlier mention of the northern gate.",
                "expected_response": "Return only a concise continuity summary from past chat history.",
            }
        },
        "response_language": "English",
        "topics_to_cover": ["Use prior chat continuity if available."],
        "facts_to_cover": ["The user may have mentioned the northern gate earlier."],
        "emotion_directive": "Warm.",
    })

    with patch("kazusa_ai_chatbot.nodes.persona_supervisor.AGENT_REGISTRY", {"conversation_history_agent": True}):
        plan = _parse_plan(raw)

    assert plan["agents"] == ["conversation_history_agent"]
    assert plan["instructions"] == {
        "conversation_history_agent": {
            "command": "Search prior chat history for the user's earlier mention of the northern gate.",
            "expected_response": "Return only a concise continuity summary from past chat history.",
        }
    }
    assert plan["response_language"] == "English"
    assert plan["topics_to_cover"] == ["Use prior chat continuity if available."]
    assert plan["facts_to_cover"] == ["The user may have mentioned the northern gate earlier."]


@pytest.mark.asyncio
async def test_supervisor_dispatches_conversation_history_agent(mock_assembler_state):
    mock_llm = _make_plan_eval_synth_llm(
        {
            "agents": ["conversation_history_agent"],
            "instructions": {
                "conversation_history_agent": {
                    "command": "Look up recent conversation context about patrol routes from past chat history.",
                    "expected_response": "Return a short memory-oriented brief without raw transcripts.",
                }
            },
            "response_language": "English",
            "topics_to_cover": ["Answer using remembered details if any are found."],
            "facts_to_cover": ["Use remembered prior chat details if they are confirmed."],
            "emotion_directive": "Thoughtful.",
        },
        synth_topics=["Answer using remembered details if any are found."],
        synth_facts=["The user previously asked about patrol routes and was told they were holding."],
        synth_emotion="Thoughtful.",
    )

    mock_agent = AsyncMock()
    mock_agent.run = AsyncMock(return_value=AgentResult(
        agent="conversation_history_agent", status="success", summary="Earlier in this channel, the user asked about patrol routes and the bot said they were holding.", tool_history=[]
    ))

    def _get_agent(name):
        return mock_agent if name == "conversation_history_agent" else None

    with patch("kazusa_ai_chatbot.nodes.persona_supervisor._get_llm", return_value=mock_llm), \
         patch("kazusa_ai_chatbot.nodes.persona_supervisor.get_agent", side_effect=_get_agent), \
         patch("kazusa_ai_chatbot.nodes.persona_supervisor.AGENT_REGISTRY", {"conversation_history_agent": True}):

        result = await persona_supervisor(mock_assembler_state)

    plan = result["supervisor_plan"]
    assert plan["agents"] == ["conversation_history_agent"]
    assert "conversation_history_agent" in plan["instructions"]
    assert result["agent_results"][0]["agent"] == "conversation_history_agent"
    assert result["speech_brief"]["response_brief"]["topics_to_cover"] == ["Answer using remembered details if any are found."]
    assert "patrol routes" in result["speech_brief"]["response_brief"]["facts_to_cover"][-1]
    mock_agent.run.assert_awaited_once_with(
        mock_assembler_state,
        "Look up recent conversation context about patrol routes from past chat history.",
        "Return a short memory-oriented brief without raw transcripts.",
    )


@pytest.mark.asyncio
async def test_supervisor_dispatches_memory_agent(mock_assembler_state):
    mock_llm = _make_plan_eval_synth_llm(
        {
            "agents": ["memory_agent"],
            "instructions": {
                "memory_agent": {
                    "command": "Recall any stored memory about the previously shared embedding guide and save new details only if the current message adds better information.",
                    "expected_response": "Return a concise memory brief describing what was recalled or saved without raw database fields.",
                }
            },
            "response_language": "English",
            "topics_to_cover": ["Answer using the remembered embedding guide details."],
            "facts_to_cover": ["Use the recalled guide details if they are relevant."],
            "emotion_directive": "Helpful.",
        },
        synth_topics=["Answer using the remembered embedding guide details."],
        synth_facts=["The embedding guide recommends vector similarity for semantic recall."],
        synth_emotion="Helpful.",
    )

    mock_agent = AsyncMock()
    mock_agent.run = AsyncMock(return_value=AgentResult(
        agent="memory_agent",
        status="success",
        summary="Stored memory says the embedding guide recommends vector similarity for semantic recall.",
        tool_history=[],
    ))

    def _get_agent(name):
        return mock_agent if name == "memory_agent" else None

    with patch("kazusa_ai_chatbot.nodes.persona_supervisor._get_llm", return_value=mock_llm), \
         patch("kazusa_ai_chatbot.nodes.persona_supervisor.get_agent", side_effect=_get_agent), \
         patch("kazusa_ai_chatbot.nodes.persona_supervisor.AGENT_REGISTRY", {"memory_agent": True}):

        result = await persona_supervisor(mock_assembler_state)

    plan = result["supervisor_plan"]
    assert plan["agents"] == ["memory_agent"]
    assert "memory_agent" in plan["instructions"]
    assert result["agent_results"][0]["agent"] == "memory_agent"
    assert result["speech_brief"]["response_brief"]["topics_to_cover"] == ["Answer using the remembered embedding guide details."]
    assert "vector similarity" in result["speech_brief"]["response_brief"]["facts_to_cover"][-1]
    mock_agent.run.assert_awaited_once_with(
        mock_assembler_state,
        "Recall any stored memory about the previously shared embedding guide and save new details only if the current message adds better information.",
        "Return a concise memory brief describing what was recalled or saved without raw database fields.",
    )


@pytest.mark.asyncio
async def test_supervisor_does_not_forward_nonfinal_agent_summary_to_facts():
    state = {
        "message_text": "千纱能把这个清单记一下么我们",
        "user_name": "Commander",
        "bot_id": "bot_001",
        "personality": {"name": "Zara", "description": "A calm strategist."},
        "user_memory": [],
        "character_state": {"mood": "alert", "emotional_tone": "warm", "recent_events": []},
        "affinity": 650,
        "conversation_history": [],
        "assembler_output": AssemblerOutput(
            channel_topic="Planning",
            user_topic="Checklist memory",
            should_respond=True,
            reason_to_respond="User is asking about checklist memory",
            use_reply_feature=False
        ),
    }

    mock_llm = _make_plan_eval_synth_llm(
        {
            "agents": ["memory_agent"],
            "instructions": {
                "memory_agent": {
                    "command": "Remember the referenced checklist for later.",
                    "expected_response": "Return a concise memory brief.",
                }
            },
            "response_language": "Chinese",
            "topics_to_cover": ["Handle the checklist memory request."],
            "facts_to_cover": [],
            "emotion_directive": "Warm.",
        },
        synth_topics=["Handle the checklist memory request."],
        synth_facts=[],
        synth_emotion="Warm.",
    )

    mock_memory_agent = AsyncMock()
    mock_memory_agent.run = AsyncMock(return_value=AgentResult(
        agent="memory_agent",
        status="needs_clarification",
        summary="您好！我注意到您提到要让我记一个清单，但消息中似乎没有包含具体的内容呢。请问这个清单具体是什么？您可以把需要记录的内容发给我吗？",
        tool_history=[],
    ))

    mock_history_agent = AsyncMock()
    mock_history_agent.run = AsyncMock(return_value=AgentResult(
        agent="conversation_history_agent",
        status="needs_context",
        summary="Could not resolve which checklist the user meant from recent chat history.",
        tool_history=[],
    ))

    def _get_agent(name):
        if name == "memory_agent":
            return mock_memory_agent
        if name == "conversation_history_agent":
            return mock_history_agent
        return None

    with patch("kazusa_ai_chatbot.nodes.persona_supervisor._get_llm", return_value=mock_llm), \
         patch("kazusa_ai_chatbot.nodes.persona_supervisor.get_agent", side_effect=_get_agent), \
         patch("kazusa_ai_chatbot.nodes.persona_supervisor.AGENT_REGISTRY", {"memory_agent": True, "conversation_history_agent": True}):
        result = await persona_supervisor(state)

    assert result["speech_brief"]["response_brief"]["topics_to_cover"] == [
        "Handle the checklist memory request."
    ]
    assert result["speech_brief"]["response_brief"]["facts_to_cover"] == []
    assert "没有包含具体" not in "\n".join(result["speech_brief"]["response_brief"]["facts_to_cover"])
    mock_memory_agent.run.assert_awaited_once()
    mock_history_agent.run.assert_not_awaited()


@pytest.mark.asyncio
async def test_supervisor_evaluate_escalates_to_another_agent():
    """Evaluate step can escalate from memory_agent to conversation_history_agent."""
    state = {
        "message_text": "千纱能把这个清单记一下么我们",
        "user_name": "Commander",
        "bot_id": "bot_001",
        "personality": {"name": "Zara", "description": "A calm strategist."},
        "user_memory": [],
        "character_state": {"mood": "alert", "emotional_tone": "warm", "recent_events": []},
        "affinity": 650,
        "conversation_history": [
            {"role": "user", "user_id": "user_123", "name": "Commander", "content": "清单是鸡蛋、奶油和糖。"},
        ],
        "assembler_output": AssemblerOutput(
            channel_topic="Planning",
            user_topic="Checklist memory",
            should_respond=True,
            reason_to_respond="User is asking about checklist memory",
            use_reply_feature=False
        ),
    }

    # LLM call sequence: plan → escalate → finish → memory_check → synthesis
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(side_effect=[
        # 1) Planning: dispatch memory_agent
        AIMessage(content=json.dumps({
            "agents": ["memory_agent"],
            "instructions": {
                "memory_agent": {
                    "command": "Remember the referenced checklist for later.",
                    "expected_response": "Return a concise memory brief.",
                }
            },
            "response_language": "Chinese",
            "topics_to_cover": ["Handle the checklist memory request."],
            "facts_to_cover": [],
            "emotion_directive": "Warm.",
        })),
        # 2) Evaluate: memory_agent needs_clarification → escalate to conversation_history_agent
        AIMessage(content=json.dumps({
            "action": "escalate",
            "agent": "conversation_history_agent",
            "instruction": {
                "command": "Search past chat for the checklist the user mentioned earlier.",
                "expected_response": "Return the checklist items.",
            },
            "reason": "memory_agent could not find the checklist, trying conversation history.",
        })),
        # 3) Evaluate: conversation_history_agent succeeded → finish
        AIMessage(content=_finish_action_json()),
        # 4) Memory check: nothing extra to store
        AIMessage(content=_no_store_json()),
        # 5) Synthesis: distill agent results into speech-ready facts
        AIMessage(content=_synthesis_json(
            topics=["Handle the checklist memory request."],
            facts=["The checklist contains eggs, cream, and sugar."],
            emotion="Warm.",
        )),
    ])

    mock_memory_agent = AsyncMock()
    mock_memory_agent.run = AsyncMock(return_value=AgentResult(
        agent="memory_agent",
        status="needs_clarification",
        summary="请问这个清单具体是什么？",
        tool_history=[],
    ))

    mock_history_agent = AsyncMock()
    mock_history_agent.run = AsyncMock(return_value=AgentResult(
        agent="conversation_history_agent",
        status="success",
        summary="The referenced checklist is eggs, cream, and sugar.",
        tool_history=[],
    ))

    def _get_agent(name):
        if name == "memory_agent":
            return mock_memory_agent
        if name == "conversation_history_agent":
            return mock_history_agent
        return None

    with patch("kazusa_ai_chatbot.nodes.persona_supervisor._get_llm", return_value=mock_llm), \
         patch("kazusa_ai_chatbot.nodes.persona_supervisor.get_agent", side_effect=_get_agent), \
         patch("kazusa_ai_chatbot.nodes.persona_supervisor.AGENT_REGISTRY", {"memory_agent": True, "conversation_history_agent": True}):
        result = await persona_supervisor(state)

    # Both agents were called
    mock_memory_agent.run.assert_awaited_once()
    mock_history_agent.run.assert_awaited_once()
    # Agent results contain both
    agent_names = [r["agent"] for r in result["agent_results"]]
    assert "memory_agent" in agent_names
    assert "conversation_history_agent" in agent_names
    # The successful result should appear in synthesized facts
    assert any("eggs, cream, and sugar" in f for f in result["speech_brief"]["response_brief"]["facts_to_cover"])
    # 5 LLM calls: plan + evaluate + evaluate + memory_check + synthesis
    assert mock_llm.ainvoke.await_count == 5


@pytest.mark.asyncio
async def test_supervisor_evaluate_retries_agent():
    """Evaluate step can retry the same agent with refined instructions."""
    state = {
        "message_text": "Search for Python 3.14 release notes",
        "user_name": "Commander",
        "bot_id": "bot_001",
        "personality": {"name": "Zara", "description": "A calm strategist."},
        "user_memory": [],
        "character_state": {},
        "affinity": AFFINITY_DEFAULT,
        "conversation_history": [],
        "assembler_output": AssemblerOutput(
            channel_topic="Python",
            user_topic="Release notes",
            should_respond=True,
            reason_to_respond="User is asking about Python release notes",
            use_reply_feature=False
        ),
    }

    # LLM call sequence: plan → retry → finish → memory_check → synthesis
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(side_effect=[
        # 1) Planning: dispatch web_search_agent
        AIMessage(content=json.dumps({
            "agents": ["web_search_agent"],
            "instructions": {
                "web_search_agent": {
                    "command": "Search for Python 3.14 info.",
                    "expected_response": "Return release notes.",
                }
            },
            "response_language": "English",
            "topics_to_cover": ["Python 3.14 release"],
            "facts_to_cover": [],
            "emotion_directive": "Neutral.",
        })),
        # 2) Evaluate: needs_context → retry with better search terms
        AIMessage(content=json.dumps({
            "action": "retry",
            "agent": "web_search_agent",
            "instruction": {
                "command": "Search specifically for 'Python 3.14 release notes changelog'.",
                "expected_response": "Return the key new features and changes.",
            },
            "reason": "First search was too vague.",
        })),
        # 3) Evaluate: success → finish
        AIMessage(content=_finish_action_json()),
        # 4) Memory check: nothing to store
        AIMessage(content=_no_store_json()),
        # 5) Synthesis: distill results
        AIMessage(content=_synthesis_json(
            topics=["Python 3.14 release"],
            facts=["Python 3.14 adds pattern matching improvements and JIT."],
        )),
    ])

    call_count = 0
    mock_agent = AsyncMock()
    async def _agent_run(state, task, expected_response=""):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return AgentResult(
                    agent="web_search_agent", status="needs_context",
                    summary="No specific results found.", tool_history=[],
                )
            return AgentResult(
                agent="web_search_agent", status="success",
                summary="Python 3.14 adds pattern matching improvements and JIT.", tool_history=[],
            )
    mock_agent.run = AsyncMock(side_effect=_agent_run)

    def _get_agent(name):
        return mock_agent if name == "web_search_agent" else None

    with patch("kazusa_ai_chatbot.nodes.persona_supervisor._get_llm", return_value=mock_llm), \
         patch("kazusa_ai_chatbot.nodes.persona_supervisor.get_agent", side_effect=_get_agent), \
         patch("kazusa_ai_chatbot.nodes.persona_supervisor.AGENT_REGISTRY", {"web_search_agent": True}):
        result = await persona_supervisor(state)

    # Agent was called twice (initial + retry)
    assert mock_agent.run.await_count == 2
    # Agent results contain both calls
    assert len(result["agent_results"]) == 2
    assert result["agent_results"][0]["status"] == "needs_context"
    assert result["agent_results"][1]["status"] == "success"
    # The retry instruction was used
    second_call_args = mock_agent.run.await_args_list[1]
    assert "changelog" in second_call_args.args[1]  # task arg
    # 5 LLM calls: plan + evaluate + evaluate + memory_check + synthesis
    assert mock_llm.ainvoke.await_count == 5


@pytest.mark.asyncio
async def test_supervisor_memory_check_triggers_store():
    """When the memory check LLM says should_store=True, memory_agent is dispatched."""
    state = {
        "message_text": "Remember that my favorite color is blue.",
        "user_name": "Commander",
        "bot_id": "bot_001",
        "personality": {"name": "Zara", "description": "A calm strategist."},
        "user_memory": [],
        "character_state": {},
        "affinity": AFFINITY_DEFAULT,
        "conversation_history": [],
        "assembler_output": AssemblerOutput(
            channel_topic="General",
            user_topic="Memory request",
            should_respond=True,
            reason_to_respond="User is requesting memory retrieval",
            use_reply_feature=False
        ),
    }

    # LLM call sequence: plan (no agents) → memory_check (yes) → synthesis
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(side_effect=[
        # 1) Plan: no agents needed for the reply itself
        AIMessage(content=json.dumps({
            "agents": [],
            "instructions": {},
            "response_language": "English",
            "topics_to_cover": ["Confirm the color preference was noted."],
            "facts_to_cover": [],
            "emotion_directive": "Warm.",
        })),
        # 2) Memory check: yes, store the preference
        AIMessage(content=json.dumps({
            "should_store": True,
            "command": "Store the user's favorite color as blue.",
            "expected_response": "Confirm the memory was saved.",
            "reason": "User explicitly asked to remember their favorite color.",
        })),
        # 3) Synthesis: distill (now includes memory_agent result)
        AIMessage(content=_synthesis_json(
            topics=["Confirm the color preference was noted."],
            facts=["The user's favorite color (blue) has been saved to memory."],
            emotion="Warm.",
        )),
    ])

    mock_memory_agent = AsyncMock()
    mock_memory_agent.run = AsyncMock(return_value=AgentResult(
        agent="memory_agent",
        status="success",
        summary="Saved: user's favorite color is blue.",
        tool_history=[],
    ))

    def _get_agent(name):
        return mock_memory_agent if name == "memory_agent" else None

    with patch("kazusa_ai_chatbot.nodes.persona_supervisor._get_llm", return_value=mock_llm), \
         patch("kazusa_ai_chatbot.nodes.persona_supervisor.get_agent", side_effect=_get_agent), \
         patch("kazusa_ai_chatbot.nodes.persona_supervisor.AGENT_REGISTRY", {"memory_agent": True}):
        result = await persona_supervisor(state)

    # Memory agent was dispatched by the memory check step
    mock_memory_agent.run.assert_awaited_once()
    call_args = mock_memory_agent.run.await_args
    assert "favorite color" in call_args.args[1]  # task arg

    # The memory agent result appears in agent_results
    assert len(result["agent_results"]) == 1
    assert result["agent_results"][0]["agent"] == "memory_agent"
    assert result["agent_results"][0]["status"] == "success"

    # Synthesized facts reflect the memory store
    assert any("blue" in f for f in result["speech_brief"]["response_brief"]["facts_to_cover"])

    # 3 LLM calls: plan + memory_check + synthesis
    assert mock_llm.ainvoke.await_count == 3


@pytest.mark.asyncio
async def test_supervisor_short_circuits_if_not_should_respond(mock_assembler_ignore_state):
    """If the assembler says not to respond, supervisor stays silent without planning."""
    mock_llm = MagicMock()

    with patch("kazusa_ai_chatbot.nodes.persona_supervisor._get_llm", return_value=mock_llm):
        result = await persona_supervisor(mock_assembler_ignore_state)

    plan = result["supervisor_plan"]
    assert plan["agents"] == []
    assert plan["topics_to_cover"] == []
    assert plan["facts_to_cover"] == []
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
            should_respond=True,
            reason_to_respond="User wants to search for Python news",
            use_reply_feature=False
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
    assert isinstance(plan["response_language"], str)
    assert isinstance(plan["topics_to_cover"], list)
    assert len(plan["topics_to_cover"]) > 0
    assert isinstance(plan["instructions"], dict)

    agent_names = [r["agent"] for r in result["agent_results"]]
    assert "web_search_agent" in agent_names
    mock_search.run.assert_called_once()


@live_llm
@pytest.mark.asyncio
async def test_live_supervisor_calls_all_three_sub_agents_in_single_plan():
    """Real LLM should plan and call conversation history, memory, and web search agents in one run."""
    from kazusa_ai_chatbot.agents.base import AGENT_REGISTRY, register_agent
    from kazusa_ai_chatbot.agents.conversation_history_agent import ConversationHistoryAgent
    from kazusa_ai_chatbot.agents.memory_agent import MemoryAgent
    from kazusa_ai_chatbot.agents.web_search_agent import WebSearchAgent

    if "web_search_agent" not in AGENT_REGISTRY:
        register_agent(WebSearchAgent())
    if "conversation_history_agent" not in AGENT_REGISTRY:
        register_agent(ConversationHistoryAgent())
    if "memory_agent" not in AGENT_REGISTRY:
        register_agent(MemoryAgent())

    mock_history = AsyncMock()
    mock_history.run = AsyncMock(return_value=AgentResult(
        agent="conversation_history_agent",
        status="success",
        summary="Conversation history placeholder.",
        tool_history=[],
    ))

    mock_memory = AsyncMock()
    mock_memory.run = AsyncMock(return_value=AgentResult(
        agent="memory_agent",
        status="success",
        summary="Memory placeholder.",
        tool_history=[],
    ))

    mock_search = AsyncMock()
    mock_search.run = AsyncMock(return_value=AgentResult(
        agent="web_search_agent",
        status="success",
        summary="Search results placeholder.",
        tool_history=[],
    ))

    state = {
        "message_text": (
            "Do three things in one reply: "
            "first, search our past chat and remind me what I said earlier about the northern gate; "
            "second, recall the saved memory note about the embedding guide and use its main points; "
            "third, search the internet right now for the latest Python 3.14 news. "
            "You will need past chat history, saved memory, and current internet information."
        ),
        "assembler_output": AssemblerOutput(
            channel_topic="Multi-source request",
            user_topic="History, memory, and web lookup",
            should_respond=True,
            reason_to_respond="User needs comprehensive information from multiple sources",
            use_reply_feature=False
        )
    }

    import kazusa_ai_chatbot.nodes.persona_supervisor as sup
    sup._llm = None

    def _get_agent(name):
        if name == "conversation_history_agent":
            return mock_history
        if name == "memory_agent":
            return mock_memory
        if name == "web_search_agent":
            return mock_search
        return None

    with patch("kazusa_ai_chatbot.nodes.persona_supervisor.get_agent", side_effect=_get_agent):
        result = await persona_supervisor(state)

    plan = result["supervisor_plan"]
    assert "conversation_history_agent" in plan["agents"], (
        f"Expected conversation_history_agent in plan, got: {plan['agents']}"
    )
    assert "memory_agent" in plan["agents"], (
        f"Expected memory_agent in plan, got: {plan['agents']}"
    )
    assert "web_search_agent" in plan["agents"], (
        f"Expected web_search_agent in plan, got: {plan['agents']}"
    )
    assert isinstance(plan["response_language"], str)
    assert isinstance(plan["topics_to_cover"], list)
    assert len(plan["topics_to_cover"]) > 0
    assert isinstance(plan["instructions"], dict)

    agent_names = [r["agent"] for r in result["agent_results"]]
    assert "conversation_history_agent" in agent_names
    assert "memory_agent" in agent_names
    assert "web_search_agent" in agent_names
    mock_history.run.assert_called_once()
    mock_memory.run.assert_called_once()
    mock_search.run.assert_called_once()


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
            should_respond=True,
            reason_to_respond="User greeted the bot",
            use_reply_feature=False
        )
    }

    import kazusa_ai_chatbot.nodes.persona_supervisor as sup
    sup._llm = None

    result = await persona_supervisor(state)

    plan = result["supervisor_plan"]
    assert plan["agents"] == [], (
        f"Expected no agents for greeting, got: {plan['agents']}"
    )
    assert isinstance(plan["response_language"], str)
    assert isinstance(plan["topics_to_cover"], list)
    assert len(plan["topics_to_cover"]) > 0
    assert len(result["agent_results"]) == 0


