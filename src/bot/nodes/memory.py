"""Stage 4 — Memory Retriever (no LLM).

Fetches two things from MongoDB:
  a) Recent conversation history for the channel
  b) Long-term user facts (extracted by the Memory Writer on prior turns)
"""

from __future__ import annotations

import logging

from bot.config import CONVERSATION_HISTORY_LIMIT
from bot.db import (
    AFFINITY_DEFAULT,
    get_affinity,
    get_character_state,
    get_conversation_history,
    get_user_facts,
)
from bot.state import BotState, CharacterState, ChatMessage

logger = logging.getLogger(__name__)


async def memory_retriever(state: BotState) -> BotState:
    """Load conversation history and user-specific long-term memory."""
    if not state.get("retrieve_memory"):
        return {
            "conversation_history": [],
            "user_memory": [],
            "character_state": {},
            "affinity": AFFINITY_DEFAULT,
        }

    channel_id = state.get("channel_id", "")
    user_id = state.get("user_id", "")

    # ── 4a: conversation history ────────────────────────────────────
    try:
        raw_history = await get_conversation_history(
            channel_id, limit=CONVERSATION_HISTORY_LIMIT
        )
        history: list[ChatMessage] = [
            ChatMessage(
                role=doc.get("role", "user"),
                user_id=doc.get("user_id", ""),
                name=doc.get("name", "unknown"),
                content=doc.get("content", ""),
            )
            for doc in raw_history
        ]
    except Exception:
        logger.exception("Failed to fetch conversation history")
        history = []

    # ── 4b: long-term user facts ────────────────────────────────────
    try:
        user_mem = await get_user_facts(user_id) if user_id else []
    except Exception:
        logger.exception("Failed to fetch user facts")
        user_mem = []

    # ── 4c: global character state (mood, tone) ─────────────────────
    try:
        raw_state = await get_character_state()
        char_state = CharacterState(
            mood=raw_state.get("mood", "neutral"),
            emotional_tone=raw_state.get("emotional_tone", "balanced"),
            recent_events=raw_state.get("recent_events", []),
            updated_at=raw_state.get("updated_at", ""),
        )
    except Exception:
        logger.exception("Failed to fetch character state")
        char_state = {}

    # ── 4d: user affinity score ───────────────────────────────────
    try:
        affinity = await get_affinity(user_id) if user_id else AFFINITY_DEFAULT
    except Exception:
        logger.exception("Failed to fetch affinity")
        affinity = AFFINITY_DEFAULT

    return {
        "conversation_history": history,
        "user_memory": user_mem,
        "character_state": char_state,
        "affinity": affinity,
    }
