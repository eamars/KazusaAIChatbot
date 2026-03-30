"""Tests for Stage 7 — Memory Writer."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage

from bot.nodes.memory_writer import memory_writer


def _mock_llm(content: str) -> MagicMock:
    """Create a mock ChatOpenAI whose ainvoke returns an AIMessage."""
    llm = MagicMock()
    llm.ainvoke = AsyncMock(return_value=AIMessage(content=content))
    return llm


def _make_state(**overrides) -> dict:
    base = {
        "user_id": "user_123",
        "message_text": "Call me Commander from now on",
        "response": "Understood, Commander.",
        "timestamp": "2026-03-30T20:00:00Z",
    }
    base.update(overrides)
    return base


@pytest.mark.asyncio
async def test_writer_extracts_facts_and_mood():
    llm_output = json.dumps({
        "user_facts": ["User prefers to be called Commander"],
        "character_state": {
            "mood": "respectful",
            "emotional_tone": "formal",
            "event_summary": "User asked to be called Commander",
        },
        "affinity_delta": 5,
    })

    with (
        patch("bot.nodes.memory_writer._get_llm", return_value=_mock_llm(llm_output)),
        patch("bot.nodes.memory_writer.upsert_user_facts", new_callable=AsyncMock) as mock_facts,
        patch("bot.nodes.memory_writer.upsert_character_state", new_callable=AsyncMock) as mock_char,
        patch("bot.nodes.memory_writer.update_affinity", new_callable=AsyncMock, return_value=505) as mock_aff,
    ):
        result = await memory_writer(_make_state())

    assert result["new_facts"] == ["User prefers to be called Commander"]
    mock_facts.assert_called_once_with("user_123", ["User prefers to be called Commander"])
    mock_char.assert_called_once()
    call_args = mock_char.call_args
    assert call_args[0][0] == "respectful"  # mood
    assert call_args[0][1] == "formal"  # emotional_tone
    mock_aff.assert_called_once_with("user_123", 5)


@pytest.mark.asyncio
async def test_writer_no_facts():
    llm_output = json.dumps({
        "user_facts": [],
        "character_state": {
            "mood": "neutral",
            "emotional_tone": "balanced",
            "event_summary": "",
        },
        "affinity_delta": 3,
    })

    with (
        patch("bot.nodes.memory_writer._get_llm", return_value=_mock_llm(llm_output)),
        patch("bot.nodes.memory_writer.upsert_user_facts", new_callable=AsyncMock) as mock_facts,
        patch("bot.nodes.memory_writer.upsert_character_state", new_callable=AsyncMock) as mock_char,
        patch("bot.nodes.memory_writer.update_affinity", new_callable=AsyncMock, return_value=503),
    ):
        result = await memory_writer(_make_state())

    assert result["new_facts"] == []
    mock_facts.assert_not_called()
    mock_char.assert_called_once()


@pytest.mark.asyncio
async def test_writer_handles_markdown_fenced_json():
    llm_output = "```json\n" + json.dumps({
        "user_facts": ["Likes cats"],
        "character_state": {"mood": "amused", "emotional_tone": "warm", "event_summary": ""},
    }) + "\n```"

    with (
        patch("bot.nodes.memory_writer._get_llm", return_value=_mock_llm(llm_output)),
        patch("bot.nodes.memory_writer.upsert_user_facts", new_callable=AsyncMock) as mock_facts,
        patch("bot.nodes.memory_writer.upsert_character_state", new_callable=AsyncMock),
        patch("bot.nodes.memory_writer.update_affinity", new_callable=AsyncMock, return_value=503),
    ):
        result = await memory_writer(_make_state())

    assert result["new_facts"] == ["Likes cats"]
    mock_facts.assert_called_once()


@pytest.mark.asyncio
async def test_writer_handles_malformed_json():
    with (
        patch("bot.nodes.memory_writer._get_llm", return_value=_mock_llm("not valid json at all")),
        patch("bot.nodes.memory_writer.upsert_user_facts", new_callable=AsyncMock) as mock_facts,
        patch("bot.nodes.memory_writer.upsert_character_state", new_callable=AsyncMock) as mock_char,
        patch("bot.nodes.memory_writer.update_affinity", new_callable=AsyncMock),
    ):
        result = await memory_writer(_make_state())

    assert result["new_facts"] == []
    mock_facts.assert_not_called()
    mock_char.assert_not_called()


@pytest.mark.asyncio
async def test_writer_handles_llm_failure():
    mock = MagicMock()
    mock.ainvoke = AsyncMock(side_effect=Exception("LLM down"))

    with (
        patch("bot.nodes.memory_writer._get_llm", return_value=mock),
        patch("bot.nodes.memory_writer.upsert_user_facts", new_callable=AsyncMock) as mock_facts,
    ):
        result = await memory_writer(_make_state())

    assert result["new_facts"] == []
    mock_facts.assert_not_called()


@pytest.mark.asyncio
async def test_writer_skips_when_no_user_id():
    result = await memory_writer(_make_state(user_id=""))
    assert result["new_facts"] == []


@pytest.mark.asyncio
async def test_writer_skips_when_no_message():
    result = await memory_writer(_make_state(message_text=""))
    assert result["new_facts"] == []


@pytest.mark.asyncio
async def test_writer_clamps_affinity_delta():
    """affinity_delta from LLM is clamped to [-20, +10]."""
    llm_output = json.dumps({
        "user_facts": [],
        "character_state": {"mood": "angry", "emotional_tone": "hostile", "event_summary": ""},
        "affinity_delta": -50,  # LLM returns out-of-range value
    })

    with (
        patch("bot.nodes.memory_writer._get_llm", return_value=_mock_llm(llm_output)),
        patch("bot.nodes.memory_writer.upsert_user_facts", new_callable=AsyncMock),
        patch("bot.nodes.memory_writer.upsert_character_state", new_callable=AsyncMock),
        patch("bot.nodes.memory_writer.update_affinity", new_callable=AsyncMock, return_value=480) as mock_aff,
    ):
        await memory_writer(_make_state())

    # Should be clamped to -20, not -50
    mock_aff.assert_called_once_with("user_123", -20)


# ── Live LLM test ────────────────────────────────────────────────────
# Requires a running LM Studio instance with a chat model loaded.
# Run with:  pytest -m live_llm -v

live_llm = pytest.mark.live_llm


@live_llm
@pytest.mark.asyncio
async def test_live_writer_extracts_valid_json():
    """Call the real LLM and verify it returns parseable extraction JSON."""
    state = _make_state(
        message_text="Please call me Commander from now on.",
        response="As you wish, Commander. I shall address you accordingly.",
    )

    with (
        patch("bot.nodes.memory_writer.upsert_user_facts", new_callable=AsyncMock) as mock_facts,
        patch("bot.nodes.memory_writer.upsert_character_state", new_callable=AsyncMock) as mock_char,
        patch("bot.nodes.memory_writer.update_affinity", new_callable=AsyncMock, return_value=505) as mock_aff,
    ):
        result = await memory_writer(state)

    # The LLM should extract at least one fact about the name preference
    assert isinstance(result["new_facts"], list)
    mock_facts.assert_called_once()
    facts_arg = mock_facts.call_args[0][1]
    assert len(facts_arg) > 0, "LLM should extract at least one user fact"

    # Character state should have been updated
    mock_char.assert_called_once()

    # Affinity delta should have been called with an int in [-20, +10]
    mock_aff.assert_called_once()
    delta = mock_aff.call_args[0][1]
    assert isinstance(delta, int)
    assert -20 <= delta <= 10
