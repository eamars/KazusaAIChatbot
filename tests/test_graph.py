"""Integration test — full graph execution with all external calls mocked."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage

from kazusa_ai_chatbot.graph import build_graph
from kazusa_ai_chatbot.state import BotState, SupervisorPlan
from kazusa_ai_chatbot.db import AFFINITY_DEFAULT


@pytest.mark.asyncio
async def test_full_graph_question_flow(sample_personality):
    """End-to-end: a question message goes through all stages and produces a response."""
    state: BotState = {
        "user_id": "user_123",
        "user_name": "TestUser",
        "channel_id": "chan_456",
        "guild_id": "guild_789",
        "bot_id": "999888777",
        "message_text": "What happened at the northern gate last night?",
        "timestamp": "2026-03-30T20:00:00Z",
        "should_respond": True,
        "personality": sample_personality,
    }

    # Supervisor LLM: no agents needed (plan + memory_check)
    _no_store = json.dumps({"should_store": False, "command": "", "expected_response": "", "reason": "Nothing to store."})
    mock_supervisor_llm = MagicMock()
    mock_supervisor_llm.ainvoke = AsyncMock(side_effect=[
        AIMessage(content=json.dumps({
            "agents": [],
            "response_language": "English",
            "topics_to_cover": ["Answer the lore question using context."],
            "facts_to_cover": ["The gate held."],
            "emotion_directive": "Neutral"
        })),
        AIMessage(content=_no_store),
    ])

    mock_speech_llm = MagicMock()
    mock_speech_llm.ainvoke = AsyncMock(
        return_value=AIMessage(content="The gate held, Commander.")
    )

    mock_history = [
        {"role": "user", "user_id": "user_123", "name": "TestUser", "content": "Is it safe?"},
    ]
    mock_char_state = {"mood": "alert", "emotional_tone": "guarded", "recent_events": []}

    graph = build_graph()

    # Relevance Agent LLM: analyze topics and decide to respond
    mock_relevance_llm = MagicMock()
    mock_relevance_llm.ainvoke = AsyncMock(
        return_value=AIMessage(content=json.dumps({
            "channel_topic": "General",
            "user_topic": "Question",
            "should_respond": True
        }))
    )

    with (
        patch("kazusa_ai_chatbot.nodes.relevance_agent.get_conversation_history", new_callable=AsyncMock, return_value=mock_history),
        patch("kazusa_ai_chatbot.nodes.relevance_agent.get_user_facts", new_callable=AsyncMock, return_value=["User goes by Commander"]),
        patch("kazusa_ai_chatbot.nodes.relevance_agent.get_character_state", new_callable=AsyncMock, return_value=mock_char_state),
        patch("kazusa_ai_chatbot.nodes.relevance_agent.get_affinity", new_callable=AsyncMock, return_value=AFFINITY_DEFAULT),
        patch("kazusa_ai_chatbot.nodes.relevance_agent._get_llm", return_value=mock_relevance_llm),
        patch("kazusa_ai_chatbot.nodes.persona_supervisor._get_llm", return_value=mock_supervisor_llm),
        patch("kazusa_ai_chatbot.nodes.speech_agent._get_llm", return_value=mock_speech_llm),
    ):
        result = await graph.ainvoke(state)

    assert result["response"] == "The gate held, Commander."
    assert result["should_respond"] is True


@pytest.mark.asyncio
async def test_full_graph_casual_greeting(sample_personality):
    """A casual greeting fetches context through relevance_agent."""
    state: BotState = {
        "user_id": "user_123",
        "user_name": "TestUser",
        "channel_id": "chan_456",
        "guild_id": "guild_789",
        "bot_id": "999888777",
        "message_text": "Hey",
        "timestamp": "2026-03-30T20:00:00Z",
        "should_respond": True,
        "personality": sample_personality,
    }

    _no_store = json.dumps({"should_store": False, "command": "", "expected_response": "", "reason": "Nothing to store."})
    mock_supervisor_llm = MagicMock()
    mock_supervisor_llm.ainvoke = AsyncMock(side_effect=[
        AIMessage(content=json.dumps({
            "agents": [],
            "response_language": "English",
            "topics_to_cover": ["Respond with a casual greeting."],
            "facts_to_cover": [],
            "emotion_directive": "Casual"
        })),
        AIMessage(content=_no_store),
    ])

    mock_speech_llm = MagicMock()
    mock_speech_llm.ainvoke = AsyncMock(
        return_value=AIMessage(content="Hey there.")
    )

    graph = build_graph()

    mock_relevance_llm = MagicMock()
    mock_relevance_llm.ainvoke = AsyncMock(
        return_value=AIMessage(content=json.dumps({
            "channel_topic": "General",
            "user_topic": "Greeting",
            "should_respond": True
        }))
    )

    with (
        patch("kazusa_ai_chatbot.nodes.relevance_agent.get_conversation_history", new_callable=AsyncMock, return_value=[]),
        patch("kazusa_ai_chatbot.nodes.relevance_agent.get_user_facts", new_callable=AsyncMock, return_value=[]),
        patch("kazusa_ai_chatbot.nodes.relevance_agent.get_character_state", new_callable=AsyncMock, return_value={}),
        patch("kazusa_ai_chatbot.nodes.relevance_agent.get_affinity", new_callable=AsyncMock, return_value=AFFINITY_DEFAULT),
        patch("kazusa_ai_chatbot.nodes.relevance_agent._get_llm", return_value=mock_relevance_llm),
        patch("kazusa_ai_chatbot.nodes.persona_supervisor._get_llm", return_value=mock_supervisor_llm),
        patch("kazusa_ai_chatbot.nodes.speech_agent._get_llm", return_value=mock_speech_llm),
    ):
        result = await graph.ainvoke(state)

    assert result["response"] == "Hey there."


@pytest.mark.asyncio
async def test_full_graph_conversation_history_flow(sample_personality):
    """End-to-end: supervisor can dispatch the conversation history agent and still produce a reply."""
    state: BotState = {
        "user_id": "user_123",
        "user_name": "TestUser",
        "channel_id": "chan_456",
        "guild_id": "guild_789",
        "bot_id": "999888777",
        "message_text": "Do you remember what I said about the northern gate?",
        "timestamp": "2026-03-30T20:00:00Z",
        "should_respond": True,
        "personality": sample_personality,
    }

    mock_supervisor_llm = MagicMock()
    mock_supervisor_llm.ainvoke = AsyncMock(side_effect=[
        # 1) Plan
        AIMessage(content=json.dumps({
            "agents": ["conversation_history_agent"],
            "instructions": {
                "conversation_history_agent": {
                    "command": "Search recent conversation history for prior mentions of the northern gate and summarize the relevant continuity.",
                    "expected_response": "Return a short memory brief without raw transcript formatting.",
                }
            },
            "response_language": "English",
            "topics_to_cover": ["Answer using prior remembered conversation if found."],
            "facts_to_cover": ["If a prior mention is found, state it explicitly."],
            "emotion_directive": "Thoughtful",
        })),
        # 2) Evaluate: finish
        AIMessage(content=json.dumps({
            "action": "finish", "agent": "",
            "instruction": {"command": "", "expected_response": ""},
            "reason": "Results are satisfactory.",
        })),
        # 3) Memory check: nothing to store
        AIMessage(content=json.dumps({"should_store": False, "command": "", "expected_response": "", "reason": "Nothing to store."})),
        # 4) Synthesis
        AIMessage(content=json.dumps({
            "topics_to_cover": ["Answer using prior remembered conversation if found."],
            "facts_to_cover": ["The user previously said the northern gate was under pressure."],
            "emotion_directive": "Thoughtful",
        })),
    ])

    mock_speech_llm = MagicMock()
    mock_speech_llm.ainvoke = AsyncMock(
        return_value=AIMessage(content="You mentioned the northern gate was under pressure, and I still remember that.")
    )

    mock_relevance_llm = MagicMock()
    mock_relevance_llm.ainvoke = AsyncMock(
        return_value=AIMessage(content=json.dumps({
            "channel_topic": "General",
            "user_topic": "Memory check",
            "should_respond": True
        }))
    )

    mock_history_agent = AsyncMock()
    mock_history_agent.run = AsyncMock(return_value={
        "agent": "conversation_history_agent",
        "status": "success",
        "summary": "Earlier in this channel, the user said the northern gate was under pressure.",
        "tool_history": [],
    })

    graph = build_graph()

    with (
        patch("kazusa_ai_chatbot.nodes.relevance_agent.get_conversation_history", new_callable=AsyncMock, return_value=[]),
        patch("kazusa_ai_chatbot.nodes.relevance_agent.get_user_facts", new_callable=AsyncMock, return_value=[]),
        patch("kazusa_ai_chatbot.nodes.relevance_agent.get_character_state", new_callable=AsyncMock, return_value={}),
        patch("kazusa_ai_chatbot.nodes.relevance_agent.get_affinity", new_callable=AsyncMock, return_value=AFFINITY_DEFAULT),
        patch("kazusa_ai_chatbot.nodes.relevance_agent._get_llm", return_value=mock_relevance_llm),
        patch("kazusa_ai_chatbot.nodes.persona_supervisor._get_llm", return_value=mock_supervisor_llm),
        patch("kazusa_ai_chatbot.nodes.persona_supervisor.get_agent", return_value=mock_history_agent),
        patch("kazusa_ai_chatbot.nodes.speech_agent._get_llm", return_value=mock_speech_llm),
    ):
        result = await graph.ainvoke(state)

    assert result["response"] == "You mentioned the northern gate was under pressure, and I still remember that."
    assert result["agent_results"][0]["agent"] == "conversation_history_agent"
    assert result["speech_brief"]["response_brief"]["topics_to_cover"] == ["Answer using prior remembered conversation if found."]
    assert "northern gate" in result["speech_brief"]["response_brief"]["facts_to_cover"][-1]
    assert mock_supervisor_llm.ainvoke.await_count == 4  # plan + evaluate + memory_check + synthesis


@pytest.mark.asyncio
async def test_full_graph_memory_agent_flow(sample_personality):
    """End-to-end: supervisor can dispatch the memory agent to recall stored detailed memory."""
    state: BotState = {
        "user_id": "user_123",
        "user_name": "TestUser",
        "channel_id": "chan_456",
        "guild_id": "guild_789",
        "bot_id": "999888777",
        "message_text": "Do you remember the article I shared about embedding models?",
        "timestamp": "2026-03-30T20:00:00Z",
        "should_respond": True,
        "personality": sample_personality,
    }

    mock_supervisor_llm = MagicMock()
    mock_supervisor_llm.ainvoke = AsyncMock(side_effect=[
        # 1) Plan
        AIMessage(content=json.dumps({
            "agents": ["memory_agent"],
            "instructions": {
                "memory_agent": {
                    "command": "Recall any stored memory about the previously shared embedding-model article and summarize the relevant remembered details.",
                    "expected_response": "Return a short memory brief without raw database formatting.",
                }
            },
            "response_language": "English",
            "topics_to_cover": ["Answer using the remembered article details if found."],
            "facts_to_cover": ["If relevant memory exists, state the article's main takeaway explicitly."],
            "emotion_directive": "Helpful",
        })),
        # 2) Evaluate: finish
        AIMessage(content=json.dumps({
            "action": "finish", "agent": "",
            "instruction": {"command": "", "expected_response": ""},
            "reason": "Results are satisfactory.",
        })),
        # 3) Memory check: nothing to store
        AIMessage(content=json.dumps({"should_store": False, "command": "", "expected_response": "", "reason": "Nothing to store."})),
        # 4) Synthesis
        AIMessage(content=json.dumps({
            "topics_to_cover": ["Answer using the remembered article details if found."],
            "facts_to_cover": ["The article recommended using vector search for semantic recall."],
            "emotion_directive": "Helpful",
        })),
    ])

    mock_speech_llm = MagicMock()
    mock_speech_llm.ainvoke = AsyncMock(
        return_value=AIMessage(content="I remember that article; it emphasized vector search for semantic recall.")
    )

    mock_relevance_llm = MagicMock()
    mock_relevance_llm.ainvoke = AsyncMock(
        return_value=AIMessage(content=json.dumps({
            "channel_topic": "General",
            "user_topic": "Memory recall",
            "should_respond": True
        }))
    )

    mock_memory_agent = AsyncMock()
    mock_memory_agent.run = AsyncMock(return_value={
        "agent": "memory_agent",
        "status": "success",
        "summary": "Stored memory says the article's main takeaway was to use vector search for semantic recall.",
        "tool_history": [],
    })

    graph = build_graph()

    with (
        patch("kazusa_ai_chatbot.nodes.relevance_agent.get_conversation_history", new_callable=AsyncMock, return_value=[]),
        patch("kazusa_ai_chatbot.nodes.relevance_agent.get_user_facts", new_callable=AsyncMock, return_value=[]),
        patch("kazusa_ai_chatbot.nodes.relevance_agent.get_character_state", new_callable=AsyncMock, return_value={}),
        patch("kazusa_ai_chatbot.nodes.relevance_agent.get_affinity", new_callable=AsyncMock, return_value=AFFINITY_DEFAULT),
        patch("kazusa_ai_chatbot.nodes.relevance_agent._get_llm", return_value=mock_relevance_llm),
        patch("kazusa_ai_chatbot.nodes.persona_supervisor._get_llm", return_value=mock_supervisor_llm),
        patch("kazusa_ai_chatbot.nodes.persona_supervisor.get_agent", return_value=mock_memory_agent),
        patch("kazusa_ai_chatbot.nodes.speech_agent._get_llm", return_value=mock_speech_llm),
    ):
        result = await graph.ainvoke(state)

    assert result["response"] == "I remember that article; it emphasized vector search for semantic recall."
    assert result["agent_results"][0]["agent"] == "memory_agent"
    assert result["speech_brief"]["response_brief"]["topics_to_cover"] == ["Answer using the remembered article details if found."]
    assert "vector search" in result["speech_brief"]["response_brief"]["facts_to_cover"][-1]
    assert mock_supervisor_llm.ainvoke.await_count == 4  # plan + evaluate + memory_check + synthesis


def test_should_respond_after_intake():
    from kazusa_ai_chatbot.graph import _should_respond_after_intake
    from kazusa_ai_chatbot.state import BotState
    from langgraph.graph import END

    # False -> END
    state: BotState = {"should_respond": False}
    assert _should_respond_after_intake(state) == [END]

    # True -> Relevance Agent
    state = {"should_respond": True}
    assert _should_respond_after_intake(state) == ["relevance_agent"]


@pytest.mark.asyncio
async def test_full_graph_empty_message(sample_personality):
    """An empty message after intake stripping should exit early."""
    state: BotState = {
        "user_id": "user_123",
        "user_name": "TestUser",
        "channel_id": "chan_456",
        "guild_id": "guild_789",
        "bot_id": "999888777",
        "message_text": "<@999888777>",
        "timestamp": "2026-03-30T20:00:00Z",
        "should_respond": True,
        "personality": sample_personality,
    }

    graph = build_graph()

    result = await graph.ainvoke(state)

    assert result["should_respond"] is False
    assert "response" not in result or result.get("response", "") == ""
