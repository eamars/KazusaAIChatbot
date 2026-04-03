"""Tests for the Relevance Agent."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage

from kazusa_ai_chatbot.agents.relevance_agent import RelevanceAgent, _parse_relevance
from kazusa_ai_chatbot.state import ChatMessage
from kazusa_ai_chatbot.utils import format_history_text


# ── _parse_relevance unit tests ──────────────────────────────────────


class TestParseRelevance:
    def test_valid_should_respond_true(self):
        raw = json.dumps({"should_respond": True, "reason": "Direct greeting."})
        result = _parse_relevance(raw)
        assert result["should_respond"] is True
        assert "greeting" in result["reason"].lower()

    def test_valid_should_respond_false(self):
        raw = json.dumps({"should_respond": False, "reason": "Talking to someone else."})
        result = _parse_relevance(raw)
        assert result["should_respond"] is False

    def test_markdown_fenced_json(self):
        raw = '```json\n{"should_respond": false, "reason": "Not relevant."}\n```'
        result = _parse_relevance(raw)
        assert result["should_respond"] is False

    def test_malformed_json_defaults_to_true(self):
        result = _parse_relevance("not json at all")
        assert result["should_respond"] is True

    def test_missing_should_respond_defaults_to_true(self):
        raw = json.dumps({"reason": "some reason"})
        result = _parse_relevance(raw)
        assert result["should_respond"] is True

    def test_missing_reason_defaults_to_empty(self):
        raw = json.dumps({"should_respond": False})
        result = _parse_relevance(raw)
        assert result["should_respond"] is False
        assert result["reason"] == ""


# ── _format_history unit tests ───────────────────────────────────────


class TestFormatHistory:
    def test_empty_history(self):
        assert format_history_text([], "Kazusa") == ""

    def test_single_user_message(self):
        history = [
            ChatMessage(role="user", user_id="u1", name="Alice", content="Hello!"),
        ]
        result = format_history_text(history, "Kazusa")
        assert result == "[Alice]: Hello!"

    def test_assistant_uses_persona_name(self):
        history = [
            ChatMessage(role="user", user_id="u1", name="Alice", content="Hi"),
            ChatMessage(role="assistant", user_id="bot", name="bot", content="Hey there!"),
        ]
        result = format_history_text(history, "Kazusa")
        assert "[Alice]: Hi" in result
        assert "[Kazusa]: Hey there!" in result

    def test_respects_limit(self):
        history = [
            ChatMessage(role="user", user_id="u1", name="Alice", content=f"msg{i}")
            for i in range(10)
        ]
        result = format_history_text(history, "Kazusa", limit=3)
        lines = result.strip().split("\n")
        assert len(lines) == 3
        assert "msg7" in lines[0]
        assert "msg9" in lines[2]

    def test_multiple_users(self):
        history = [
            ChatMessage(role="user", user_id="u1", name="Alice", content="Hey"),
            ChatMessage(role="user", user_id="u2", name="Bob", content="Yo"),
            ChatMessage(role="assistant", user_id="bot", name="bot", content="Hi both"),
        ]
        result = format_history_text(history, "Kazusa")
        assert "[Alice]" in result
        assert "[Bob]" in result
        assert "[Kazusa]" in result


# ── RelevanceAgent.run tests ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_relevance_agent_returns_should_respond_true():
    agent = RelevanceAgent()
    state = {"message_text": "Hey there!", "personality": {"name": "Zara"}}

    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(
        return_value=AIMessage(content=json.dumps({
            "should_respond": True,
            "reason": "Direct greeting to the bot.",
        }))
    )

    with patch("kazusa_ai_chatbot.agents.relevance_agent._get_llm", return_value=mock_llm):
        result = await agent.run(state, "Hey there!")

    assert result["agent"] == "relevance_agent"
    assert result["status"] == "success"
    data = json.loads(result["summary"])
    assert data["should_respond"] is True


@pytest.mark.asyncio
async def test_relevance_agent_returns_should_respond_false():
    agent = RelevanceAgent()
    state = {"message_text": "random noise", "personality": {"name": "Zara"}}

    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(
        return_value=AIMessage(content=json.dumps({
            "should_respond": False,
            "reason": "Irrelevant message.",
        }))
    )

    with patch("kazusa_ai_chatbot.agents.relevance_agent._get_llm", return_value=mock_llm):
        result = await agent.run(state, "random noise")

    data = json.loads(result["summary"])
    assert data["should_respond"] is False


@pytest.mark.asyncio
async def test_relevance_agent_llm_failure_defaults_to_respond():
    agent = RelevanceAgent()
    state = {"message_text": "Hi", "personality": {"name": "Zara"}}

    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(side_effect=Exception("LLM down"))

    with patch("kazusa_ai_chatbot.agents.relevance_agent._get_llm", return_value=mock_llm):
        result = await agent.run(state, "Hi")

    data = json.loads(result["summary"])
    assert data["should_respond"] is True  # fail-open


@pytest.mark.asyncio
async def test_relevance_agent_name_and_description():
    agent = RelevanceAgent()
    assert agent.name == "relevance_agent"
    assert "respond" in agent.description.lower()


# ── Live LLM tests ──────────────────────────────────────────────────
# Requires a running LM Studio instance with a chat model loaded.
# Run with:  pytest -m live_llm -v

live_llm = pytest.mark.live_llm


def _make_state(
    persona_name: str = "Kazusa",
    history: list[ChatMessage] | None = None,
) -> dict:
    state = {"message_text": "", "personality": {"name": persona_name}}
    if history is not None:
        state["conversation_history"] = history
    return state


async def _run_live(
    query: str,
    persona_name: str = "Kazusa",
    history: list[ChatMessage] | None = None,
) -> dict:
    """Run the relevance agent against the real LLM and return the decision."""
    import agents.relevance_agent as rel
    rel._llm = None  # reset cached LLM

    agent = RelevanceAgent()
    state = _make_state(persona_name, history=history)
    result = await agent.run(state, query)

    assert result["agent"] == "relevance_agent"
    assert result["status"] == "success"
    return json.loads(result["summary"])


@live_llm
@pytest.mark.asyncio
async def test_live_direct_greeting_should_respond():
    """A friendly greeting directed at the bot should trigger a response."""
    decision = await _run_live("Hey Kazusa, how are you doing today?")
    assert decision["should_respond"] is True, (
        f"Expected should_respond=True for direct greeting, reason: {decision['reason']}"
    )


@live_llm
@pytest.mark.asyncio
async def test_live_question_for_bot_should_respond():
    """A question clearly aimed at the bot should trigger a response."""
    decision = await _run_live("What do you think about the new park downtown?")
    assert decision["should_respond"] is True, (
        f"Expected should_respond=True for question, reason: {decision['reason']}"
    )


@live_llm
@pytest.mark.asyncio
async def test_live_conversation_between_others_should_not_respond():
    """A message that is clearly between two other people should not respond."""
    decision = await _run_live(
        "Hey Jake, are we still meeting at 5pm for the gym session?"
    )
    assert decision["should_respond"] is False, (
        f"Expected should_respond=False for other-user conversation, reason: {decision['reason']}"
    )


@live_llm
@pytest.mark.asyncio
async def test_live_emoji_spam_should_not_respond():
    """Random emoji spam should not trigger a response."""
    decision = await _run_live("😂😂😂😂😂🔥🔥🔥")
    assert decision["should_respond"] is False, (
        f"Expected should_respond=False for emoji spam, reason: {decision['reason']}"
    )


@live_llm
@pytest.mark.asyncio
async def test_live_bot_command_not_for_us_should_not_respond():
    """A bot command intended for a different bot should not respond."""
    decision = await _run_live("!play despacito")
    assert decision["should_respond"] is False, (
        f"Expected should_respond=False for other-bot command, reason: {decision['reason']}"
    )


@live_llm
@pytest.mark.asyncio
async def test_live_followup_in_active_conversation_should_respond():
    """An ambiguous follow-up like 'yeah' should respond when the bot was
    just talking to the user (history shows active exchange)."""
    history = [
        ChatMessage(role="user", user_id="u1", name="Alice", content="Kazusa, do you like rain?"),
        ChatMessage(role="assistant", user_id="bot", name="Kazusa", content="I find the sound of rain calming."),
        ChatMessage(role="user", user_id="u1", name="Alice", content="Me too honestly"),
        ChatMessage(role="assistant", user_id="bot", name="Kazusa", content="It makes everything feel quieter."),
    ]
    decision = await _run_live("yeah that sounds nice", history=history)
    assert decision["should_respond"] is True, (
        f"Expected should_respond=True for follow-up in active conversation, reason: {decision['reason']}"
    )


@live_llm
@pytest.mark.asyncio
async def test_live_other_users_chatting_should_not_respond():
    """An ambiguous message like 'yeah' should NOT respond when the history
    shows a conversation between two other users, not involving the bot."""
    history = [
        ChatMessage(role="user", user_id="u1", name="Alice", content="Did you finish the report?"),
        ChatMessage(role="user", user_id="u2", name="Bob", content="Almost, just need the graphs."),
        ChatMessage(role="user", user_id="u1", name="Alice", content="Send it when you're done"),
        ChatMessage(role="user", user_id="u2", name="Bob", content="Will do"),
    ]
    decision = await _run_live("yeah sounds good", history=history)
    assert decision["should_respond"] is False, (
        f"Expected should_respond=False for conversation between others, reason: {decision['reason']}"
    )
