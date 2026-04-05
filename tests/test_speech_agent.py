"""Tests for the Speech Agent."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from kazusa_ai_chatbot.nodes.speech_agent import speech_agent


# ── speech_agent node tests ─────────────────────────────────────────

@pytest.fixture
def sample_speech_state():
    return {
        "speech_brief": {
            "personality": {"name": "Zara", "tone": "warm"},
            "user_input_brief": {
                "channel_topic": "Greeting",
                "user_topic": "Hello",
                "intent_summary": "The user is greeting Zara.",
            },
            "response_brief": {
                "should_respond": True,
                "response_goal": "Acknowledge the user.",
                "response_language": "English",
                "tone_guidance": "Warm and friendly.",
                "relationship_guidance": "Be a little open and welcoming.",
                "state_guidance": "Maintain a light and cheerful demeanor.",
                "continuity_summary": "No additional recent continuity context is required.",
                "topics_to_cover": ["Greet the user back."],
                "facts_to_cover": ["The user prefers to be called Commander."],
                "unknowns_or_limits": [],
            },
        }
    }


@pytest.mark.asyncio
async def test_speech_agent_basic_response(sample_speech_state):
    """Speech agent should pass the sanitized speech brief to the LLM and return the reply."""
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="Hey there, Commander."))

    with patch("kazusa_ai_chatbot.nodes.speech_agent._get_llm", return_value=mock_llm):
        result = await speech_agent(sample_speech_state)

    assert result["response"] == "Hey there, Commander."

    # Check what was sent to LLM
    args, _ = mock_llm.ainvoke.call_args
    messages = args[0]

    assert len(messages) == 2
    assert isinstance(messages[0], SystemMessage)
    assert isinstance(messages[1], HumanMessage)

    human_json = json.loads(messages[1].content)
    assert human_json["personality"]["name"] == "Zara"
    assert human_json["response_brief"]["response_goal"] == "Acknowledge the user."
    assert human_json["response_brief"]["response_language"] == "English"
    assert human_json["response_brief"]["topics_to_cover"] == ["Greet the user back."]
    assert human_json["response_brief"]["facts_to_cover"] == ["The user prefers to be called Commander."]


@pytest.mark.asyncio
async def test_speech_agent_receives_no_raw_message_or_internal_state(sample_speech_state):
    """The speech brief should stay sanitized and exclude raw message/internal state fields."""

    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="Hello there."))

    with patch("kazusa_ai_chatbot.nodes.speech_agent._get_llm", return_value=mock_llm):
        await speech_agent(sample_speech_state)

    args, _ = mock_llm.ainvoke.call_args
    messages = args[0]
    human_json = json.loads(messages[1].content)

    assert "current_message" not in human_json
    assert "conversation_history" not in human_json
    assert "character_state" not in human_json
    assert "affinity" not in human_json


@pytest.mark.asyncio
async def test_speech_agent_includes_unknowns_and_limits(sample_speech_state):
    """Speech agent should preserve supervisor-provided limits in the sanitized brief."""
    sample_speech_state["speech_brief"]["response_brief"]["unknowns_or_limits"] = [
        "Do not claim certainty about the weather forecast.",
    ]

    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="I may be missing a detail, but here is what I can say."))

    with patch("kazusa_ai_chatbot.nodes.speech_agent._get_llm", return_value=mock_llm):
        result = await speech_agent(sample_speech_state)

    assert len(result["response"]) > 0

    args, _ = mock_llm.ainvoke.call_args
    messages = args[0]
    human_json = json.loads(messages[1].content)

    assert human_json["response_brief"]["unknowns_or_limits"] == [
        "Do not claim certainty about the weather forecast.",
    ]


@pytest.mark.asyncio
async def test_speech_agent_short_circuits_silence():
    """If the supervisor marks should_respond false, return empty string."""
    state = {
        "speech_brief": {
            "response_brief": {
                "should_respond": False,
            }
        }
    }

    # Should not even try to call the LLM
    mock_llm = MagicMock()
    with patch("kazusa_ai_chatbot.nodes.speech_agent._get_llm", return_value=mock_llm):
        result = await speech_agent(state)

    assert result["response"] == ""
    mock_llm.ainvoke.assert_not_called()


@pytest.mark.asyncio
async def test_speech_agent_llm_failure(sample_speech_state):
    """If LLM fails, return fallback '*stays silent*' string."""
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(side_effect=Exception("API limit"))

    with patch("kazusa_ai_chatbot.nodes.speech_agent._get_llm", return_value=mock_llm):
        result = await speech_agent(sample_speech_state)

    assert result["response"] == "*stays silent*"


@pytest.mark.asyncio
async def test_speech_agent_empty_messages():
    """If speech_brief is missing, it should just return ..."""
    state = {
        "speech_brief": {},
    }

    result = await speech_agent(state)
    assert result["response"] == "..."


@pytest.mark.asyncio
async def test_speech_agent_without_supervisor_plan():
    """Speech agent only needs speech_brief and should not depend on supervisor_plan."""
    state = {
        "speech_brief": {
            "personality": {"name": "Zara"},
            "user_input_brief": {"intent_summary": "The user is greeting Zara."},
            "response_brief": {
                "should_respond": True,
                "response_goal": "Reply to the greeting.",
                "response_language": "English",
                "tone_guidance": "Warm.",
                "relationship_guidance": "Friendly.",
                "state_guidance": "Calm.",
                "continuity_summary": "No additional recent continuity context is required.",
                "topics_to_cover": ["Reply to the greeting."],
                "facts_to_cover": [],
                "unknowns_or_limits": [],
            }
        }
    }

    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="Greetings."))

    with patch("kazusa_ai_chatbot.nodes.speech_agent._get_llm", return_value=mock_llm):
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
    import kazusa_ai_chatbot.nodes.speech_agent as sa

    # Reset cached LLM so a real one is created
    sa._llm = None

    content_directive = (
        "Warmly acknowledge the affectionate way they called you.\n"
        "Show genuine delight at their cute self-given name.\n"
        "Maybe make a playful comment about how fitting that nickname is."
    )

    state = {
        "speech_brief": {
            "personality": {
                "name": "Kazusa",
                "description": "A gentle and caring character.",
                "tone": "warm and playful",
            },
            "user_input_brief": {
                "channel_topic": "Appearance",
                "user_topic": "How Kazusa imagines the user's appearance",
                "intent_summary": "The user wants an affectionate, imaginative answer about how Kazusa pictures their appearance.",
            },
            "response_brief": {
                "should_respond": True,
                "response_goal": content_directive,
                "response_language": "English",
                "tone_guidance": "Warm and playful",
                "relationship_guidance": "Be openly fond and affectionate.",
                "state_guidance": "Maintain a delighted and gentle demeanor.",
                "continuity_summary": "Kazusa and the user have already established cute nicknames and a warm rapport.",
                "topics_to_cover": [
                    "Answer how Kazusa imagines the user's appearance.",
                    "Keep the answer affectionate and playful.",
                ],
                "facts_to_cover": [
                    "The user and Kazusa have already established cute nicknames.",
                ],
                "unknowns_or_limits": [],
            },
        },
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
