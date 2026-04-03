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

from kazusa_ai_chatbot.config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL
from kazusa_ai_chatbot.db import update_affinity, upsert_character_state, upsert_user_facts
from kazusa_ai_chatbot.state import BotState
from kazusa_ai_chatbot.utils import format_history_text

logger = logging.getLogger(__name__)

_llm: ChatOpenAI | None = None

EXTRACTION_PROMPT = """\
Analyse this exchange between {user_name} and {persona_name}.
Return ONLY a JSON object with three keys:

1. "user_facts": array of short fact strings about the user. Return [] if nothing notable.
   Do NOT include facts about the bot character.
2. "character_state": object with keys "mood", "emotional_tone", and "event_summary".
   - "mood": the character's current mood after this exchange (e.g. "playful", "melancholic", "irritated", "content")
   - "emotional_tone": how the character is expressing themselves (e.g. "warm", "guarded", "teasing", "affectionate")
   - "event_summary": one short sentence summarising what happened in this exchange, or "" if nothing notable.
     Use the actual names ({user_name} and {persona_name}), NOT generic words like "user" or "bot".
3. "affinity_delta": integer from -20 to +10 indicating how this exchange changes the character's feeling toward the user.
   - Actively engaging, emotionally warm, or thoughtful conversation: +5 to +10
   - Friendly and respectful conversation with substance: +3 to +5
   - Neutral small-talk or simple acknowledgements: +1 to +2
   - Polite but disengaging (declining invitations, brushing off topics, ending conversation): 0
   - Cold, dismissive, or indifferent behaviour from the user: -3 to -5
   - Rude, hostile, or deliberately hurtful behaviour from the user: -5 to -20
   - Default to 0 if the exchange is unremarkable

Examples:
{{
  "user_facts": ["User prefers to be called Commander"],
  "character_state": {{"mood": "amused", "emotional_tone": "teasing", "event_summary": "{user_name} asked {persona_name} about the northern gate incident"}},

  "affinity_delta": 5
}}

Recent conversation (for context):
{history}

Latest exchange (analyse THIS):
{user_name}: {user_message}
{persona_name}: {bot_response}

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
    user_name = state.get("user_name", "User")
    personality = state.get("personality", {})
    persona_name = personality.get("name", "Bot")
    message_text = state.get("message_text", "")
    response = state.get("response", "")
    timestamp = state.get("timestamp", "")
    agent_results = state.get("agent_results", [])
    conversation_history = state.get("conversation_history", [])

    if not user_id or not message_text:
        return {**state, "new_facts": []}

    try:
        llm = _get_llm()

        # Build exchange text, including agent results if any
        exchange = f"{user_name}: {message_text}\n"
        for ar in agent_results:
            status = ar.get("status", "unknown")
            summary = ar.get("summary", "")
            exchange += f"[Agent: {ar['agent']} ({status}) → {summary}]\n"
        exchange += f"{persona_name}: {response}"

        history_text = format_history_text(
            conversation_history, persona_name, limit=10,
        )

        prompt = EXTRACTION_PROMPT.format(
            user_name=user_name,
            persona_name=persona_name,
            history=history_text or "(no prior conversation)",
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
        raw_delta = parsed.get("affinity_delta", 0)
        if isinstance(raw_delta, (int, float)):
            delta = max(-20, min(10, int(raw_delta)))  # clamp to safe range
            new_affinity = await update_affinity(user_id, delta)
            logger.info("Affinity for %s: delta=%+d → %d", user_id, delta, new_affinity)

    except Exception:
        logger.exception("Memory writer failed — skipping")
        facts = []

    return {**state, "new_facts": facts}
