"""Tests for the Speech Agent."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from kazusa_ai_chatbot.agents.speech_agent import _build_agent_context, speech_agent
from kazusa_ai_chatbot.state import AgentResult, SupervisorPlan


# ── _build_agent_context unit tests ─────────────────────────────────


class TestBuildAgentContext:
    def test_empty_results_and_directive(self):
        assert _build_agent_context([], "", "") == {}

    def test_directive_only(self):
        ctx = _build_agent_context([], "Respond casually.", "Happy")
        assert ctx["supervisor_directives"]["content"] == "Respond casually."
        assert ctx["supervisor_directives"]["emotion_tone"] == "Happy"

    def test_agent_results_success_and_error(self):
        results = [
            AgentResult(agent="a", status="success", summary="Result A", tool_history=[]),
            AgentResult(agent="b", status="error", summary="Error B", tool_history=[]),
        ]
        ctx = _build_agent_context(results, "Combine both.", "")

        assert ctx["supervisor_directives"]["content"] == "Combine both."
        assert len(ctx["agent_results"]) == 2
        assert ctx["agent_results"][0]["agent"] == "a"
        assert ctx["agent_results"][0]["status"] == "success"
        assert ctx["agent_results"][1]["agent"] == "b"
        assert ctx["agent_results"][1]["status"] == "FAILED"


# ── speech_agent node tests ─────────────────────────────────────────

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
                "message": "Hello"
            },
            "context": {
                "personality": {"name": "Zara"},
            }
        }
    }


@pytest.mark.asyncio
async def test_speech_agent_basic_response(sample_speech_state):
    """Speech agent should append agent context to the system prompt and return response."""
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="Hey there, Commander."))

    with patch("kazusa_ai_chatbot.agents.speech_agent._get_llm", return_value=mock_llm):
        result = await speech_agent(sample_speech_state)

    assert result["response"] == "Hey there, Commander."

    # Check what was sent to LLM
    args, _ = mock_llm.ainvoke.call_args
    messages = args[0]

    assert len(messages) == 2
    assert isinstance(messages[0], SystemMessage)
    assert isinstance(messages[1], HumanMessage)

    human_json = json.loads(messages[1].content)
    assert "Acknowledge the user." in human_json["context"]["supervisor_directives"]["content"]


@pytest.mark.asyncio
async def test_speech_agent_incorporates_agent_results(sample_speech_state):
    """Speech agent should include agent results in the human context."""
    sample_speech_state["supervisor_plan"]["content_directive"] = "Tell them the weather."
    sample_speech_state["agent_results"] = [
        AgentResult(
            agent="web_search_agent",
            status="success",
            summary="It is 18 degrees in Tokyo.",
            tool_history=[]
        )
    ]

    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="Hmm, 18 degrees in Tokyo. Not bad."))

    with patch("kazusa_ai_chatbot.agents.speech_agent._get_llm", return_value=mock_llm):
        result = await speech_agent(sample_speech_state)

    assert result["response"] == "Hmm, 18 degrees in Tokyo. Not bad."

    args, _ = mock_llm.ainvoke.call_args
    messages = args[0]
    human_json = json.loads(messages[1].content)

    assert human_json["context"]["agent_results"][0]["summary"] == "It is 18 degrees in Tokyo."


@pytest.mark.asyncio
async def test_speech_agent_handles_error_result(sample_speech_state):
    """Speech agent should handle FAILED agent states properly."""
    sample_speech_state["agent_results"] = [
        AgentResult(
            agent="db_agent",
            status="error",
            summary="Database timeout.",
            tool_history=[]
        )
    ]

    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="I am sorry, my memory failed me."))

    with patch("kazusa_ai_chatbot.agents.speech_agent._get_llm", return_value=mock_llm):
        result = await speech_agent(sample_speech_state)

    assert "sorry" in result["response"].lower()

    args, _ = mock_llm.ainvoke.call_args
    messages = args[0]
    human_json = json.loads(messages[1].content)

    assert human_json["context"]["agent_results"][0]["status"] == "FAILED"


@pytest.mark.asyncio
async def test_speech_agent_short_circuits_silence():
    """If the supervisor says 'Do not respond. Stay silent.', return empty string."""
    state = {
        "supervisor_plan": {
            "content_directive": "Do not respond. Stay silent."
        }
    }

    # Should not even try to call the LLM
    mock_llm = MagicMock()
    with patch("kazusa_ai_chatbot.agents.speech_agent._get_llm", return_value=mock_llm):
        result = await speech_agent(state)

    assert result["response"] == ""
    mock_llm.ainvoke.assert_not_called()


@pytest.mark.asyncio
async def test_speech_agent_llm_failure(sample_speech_state):
    """If LLM fails, return fallback '*stays silent*' string."""
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(side_effect=Exception("API limit"))

    with patch("kazusa_ai_chatbot.agents.speech_agent._get_llm", return_value=mock_llm):
        result = await speech_agent(sample_speech_state)

    assert result["response"] == "*stays silent*"


@pytest.mark.asyncio
async def test_speech_agent_empty_messages():
    """If speech_human_data is missing, it should just return ..."""
    state = {
        "speech_human_data": {},
        "supervisor_plan": {}
    }

    result = await speech_agent(state)
    assert result["response"] == "..."


@pytest.mark.asyncio
async def test_speech_agent_no_supervisor_plan():
    """Works fine if supervisor plan is missing but human_data exists."""
    state = {
        "speech_human_data": {
            "current_message": {"speaker": "u", "message": "hi"}
        }
    }

    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="Greetings."))

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

    content_directive = (
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
            content_directive=content_directive,
            emotion_directive="Warm and playful",
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
