"""Tests for the Relevance Agent."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage

from agents.relevance_agent import RelevanceAgent, _parse_relevance


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

    with patch("agents.relevance_agent._get_llm", return_value=mock_llm):
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

    with patch("agents.relevance_agent._get_llm", return_value=mock_llm):
        result = await agent.run(state, "random noise")

    data = json.loads(result["summary"])
    assert data["should_respond"] is False


@pytest.mark.asyncio
async def test_relevance_agent_llm_failure_defaults_to_respond():
    agent = RelevanceAgent()
    state = {"message_text": "Hi", "personality": {"name": "Zara"}}

    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(side_effect=Exception("LLM down"))

    with patch("agents.relevance_agent._get_llm", return_value=mock_llm):
        result = await agent.run(state, "Hi")

    data = json.loads(result["summary"])
    assert data["should_respond"] is True  # fail-open


@pytest.mark.asyncio
async def test_relevance_agent_name_and_description():
    agent = RelevanceAgent()
    assert agent.name == "relevance_agent"
    assert "respond" in agent.description.lower()
