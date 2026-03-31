"""Shared fixtures for all tests."""

from __future__ import annotations

import pytest

from state import BotState, CharacterState, ChatMessage, RagResult


@pytest.fixture
def sample_personality() -> dict:
    return {
        "name": "Zara",
        "description": "A battle-scarred elven sentinel.",
        "tone": "sardonic, loyal, terse",
        "speech_patterns": "Short sentences. Military jargon.",
        "backstory": "Lost her company at the Siege of Dawnhollow.",
    }


@pytest.fixture
def base_state(sample_personality) -> BotState:
    """A minimal valid BotState with all Stage-1 fields populated."""
    return BotState(
        user_id="user_123",
        user_name="TestUser",
        channel_id="chan_456",
        guild_id="guild_789",
        bot_id="999888777",
        message_text="Hello Zara, what happened at the northern gate?",
        timestamp="2026-03-30T20:00:00Z",
        should_respond=True,
        personality=sample_personality,
    )


@pytest.fixture
def routed_state(base_state) -> BotState:
    """State after the router has run — both retrieval flags set."""
    return {
        **base_state,
        "retrieve_rag": True,
        "retrieve_memory": True,
        "rag_query": "what happened at the northern gate",
    }


@pytest.fixture
def sample_rag_results() -> list[RagResult]:
    return [
        RagResult(text="The northern gate was breached by shadow wolves.", source="lore/events", score=0.87),
        RagResult(text="Captain Voss reported three casualties.", source="lore/npcs", score=0.82),
    ]


@pytest.fixture
def sample_history() -> list[ChatMessage]:
    return [
        ChatMessage(role="user", user_id="user_123", name="TestUser", content="Is the gate safe?"),
        ChatMessage(role="assistant", user_id="bot_001", name="bot", content="For now, Commander."),
    ]


@pytest.fixture
def sample_character_state() -> CharacterState:
    return CharacterState(
        mood="alert",
        emotional_tone="guarded",
        recent_events=["Shadow wolves attacked the northern gate"],
        updated_at="2026-03-30T19:00:00Z",
    )


@pytest.fixture
def assembled_state(
    routed_state,
    sample_rag_results,
    sample_history,
    sample_character_state,
) -> BotState:
    """State ready for the assembler — all retrieval results populated."""
    return {
        **routed_state,
        "rag_results": sample_rag_results,
        "conversation_history": sample_history,
        "user_memory": ["User goes by Commander"],
        "character_state": sample_character_state,
    }
