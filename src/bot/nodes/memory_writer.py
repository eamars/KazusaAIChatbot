"""Stage 7 — Memory Writer (LLM call, async / fire-and-forget).

Extracts notable facts about the user from the latest exchange and
persists them to MongoDB.  Runs AFTER the reply has been sent, so
the user does not wait for this.
"""

from __future__ import annotations

import json
import logging

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from bot.config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL
from bot.db import update_affinity, upsert_character_state, upsert_user_facts
from bot.state import BotState

logger = logging.getLogger(__name__)

_llm: ChatOpenAI | None = None

EXTRACTION_PROMPT = """\
Analyse this exchange between a user and a role-play character.
Return ONLY a JSON object with three keys:

1. "user_facts": array of short fact strings about the user. Return [] if nothing notable.
   Do NOT include facts about the bot character.
2. "character_state": object with keys "mood", "emotional_tone", and "event_summary".
   - "mood": the character's current mood after this exchange (e.g. "playful", "melancholic", "irritated", "content")
   - "emotional_tone": how the character is expressing themselves (e.g. "warm", "guarded", "teasing", "affectionate")
   - "event_summary": one short sentence summarising what happened in this exchange, or "" if nothing notable
3. "affinity_delta": integer from -20 to +10 indicating how this exchange changes the character's feeling toward the user.
   - Friendly, respectful, or engaging conversation: +3 to +10
   - Neutral small-talk: +1 to +3
   - Rude, hostile, or dismissive behaviour from the user: -5 to -20
   - Default to +0 if the exchange is unremarkable

Examples:
{{
  "user_facts": ["User prefers to be called Commander"],
  "character_state": {{"mood": "amused", "emotional_tone": "teasing", "event_summary": "User asked about the northern gate incident"}},
  "affinity_delta": 5
}}

Exchange:
User: {user_message}
Bot: {bot_response}

JSON:"""


def _get_llm() -> ChatOpenAI:
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model=LLM_MODEL,
            temperature=0.0,
            base_url=LLM_BASE_URL,
            api_key=LLM_API_KEY,
        )
    return _llm


async def memory_writer(state: BotState) -> BotState:
    """Extract user facts and character state update. Best-effort — failures are silent."""
    user_id = state.get("user_id", "")
    message_text = state.get("message_text", "")
    response = state.get("response", "")
    timestamp = state.get("timestamp", "")
    tool_history = state.get("tool_history", [])

    if not user_id or not message_text:
        return {**state, "new_facts": []}

    try:
        llm = _get_llm()

        # Build exchange text, including tool calls if any
        exchange = f"User: {message_text}\n"
        for t in tool_history:
            exchange += f"[Tool: {t['tool']}({t['args']}) → {t['result']}]\n"
        exchange += f"Bot: {response}"

        prompt = EXTRACTION_PROMPT.format(
            user_message=exchange,
            bot_response=response,
        )
        result = await llm.ainvoke([HumanMessage(content=prompt)])
        raw = result.content or "{}"

        # Parse JSON — tolerate markdown fences
        raw = raw.strip().strip("`").strip()
        if raw.startswith("json"):
            raw = raw[4:].strip()
        parsed = json.loads(raw)

        if not isinstance(parsed, dict):
            parsed = {}

        # ── User facts ──────────────────────────────────────────────
        facts = parsed.get("user_facts", [])
        if not isinstance(facts, list):
            facts = []
        facts = [f for f in facts if isinstance(f, str) and f.strip()]

        if facts:
            await upsert_user_facts(user_id, facts)
            logger.info("Stored %d new facts for user %s", len(facts), user_id)

        # ── Character state ──────────────────────────────────────────
        char_state = parsed.get("character_state", {})
        if isinstance(char_state, dict):
            mood = char_state.get("mood", "")
            tone = char_state.get("emotional_tone", "")
            event = char_state.get("event_summary", "")
            if mood or tone:
                recent = [event] if event else []
                await upsert_character_state(mood, tone, recent, timestamp)
                logger.info("Updated character state: mood=%s tone=%s", mood, tone)

        # ── Affinity delta ────────────────────────────────────────────
        raw_delta = parsed.get("affinity_delta", 3)
        if isinstance(raw_delta, (int, float)):
            delta = max(-20, min(10, int(raw_delta)))  # clamp to safe range
            new_affinity = await update_affinity(user_id, delta)
            logger.info("Affinity for %s: delta=%+d → %d", user_id, delta, new_affinity)

    except Exception:
        logger.exception("Memory writer failed — skipping")
        facts = []

    return {**state, "new_facts": facts}
