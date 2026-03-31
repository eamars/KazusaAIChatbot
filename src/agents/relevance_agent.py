"""Relevance Agent — LLM-based check on whether the bot should reply.

The supervisor always runs this agent first (before any other planned
agents).  It makes a lightweight LLM call to decide if the user's
message warrants a response from the bot, considering:

- Whether the message is directed at the bot or is general chatter.
- Whether the topic is something the bot's persona would engage with.
- Whether responding would be appropriate (e.g. not interrupting a
  conversation between other users).

The agent returns ``should_respond: true`` or ``false`` in its summary,
which the speech agent uses to decide whether to stay silent.
"""

from __future__ import annotations

import json
import logging

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from agents.base import BaseAgent
from config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL
from state import AgentResult, BotState

logger = logging.getLogger(__name__)

_llm: ChatOpenAI | None = None

_RELEVANCE_PROMPT = """\
You are a relevance filter for a role-play chatbot.  Your ONLY job is to
decide whether the bot should respond to the user's message.

The bot should respond when:
- The message is clearly directed at the bot (greeting, question, conversation).
- The message is a continuation of an ongoing conversation with the bot.
- The message is casual chat in a channel where the bot is expected to participate.

The bot should NOT respond when:
- The message is clearly part of a conversation between other users.
- The message is a system/bot command not intended for this bot.
- The message is irrelevant noise (random emoji, spam, etc.).

Personality context: {persona_name}

Respond with ONLY valid JSON (no markdown fences):
{{"should_respond": true/false, "reason": "brief explanation"}}
"""


def _get_llm() -> ChatOpenAI:
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model=LLM_MODEL,
            temperature=0.1,
            base_url=LLM_BASE_URL,
            api_key=LLM_API_KEY,
        )
    return _llm


def _parse_relevance(raw: str) -> dict:
    """Parse the LLM's JSON response into a relevance decision.

    Returns {"should_respond": bool, "reason": str}.
    Falls back to should_respond=True on parse failure (fail-open).
    """
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
    if text.endswith("```"):
        text = text.rsplit("```", 1)[0]
    text = text.strip()

    try:
        data = json.loads(text)
        should_respond = bool(data.get("should_respond", True))
        reason = str(data.get("reason", ""))
        return {"should_respond": should_respond, "reason": reason}
    except (json.JSONDecodeError, TypeError, AttributeError):
        logger.warning("Failed to parse relevance response: %s", raw[:200])
        return {"should_respond": True, "reason": "Parse failure — defaulting to respond."}


class RelevanceAgent(BaseAgent):
    """Lightweight LLM check on whether the bot should reply."""

    @property
    def name(self) -> str:
        return "relevance_agent"

    @property
    def description(self) -> str:
        return (
            "Determines whether the bot should respond to the current message. "
            "Always runs first. Returns should_respond true/false with a reason."
        )

    async def run(self, state: BotState, user_query: str) -> AgentResult:
        """Evaluate whether the bot should respond to this message."""
        personality = state.get("personality", {})
        persona_name = personality.get("name", "the bot")

        prompt = _RELEVANCE_PROMPT.format(persona_name=persona_name)

        try:
            llm = _get_llm()
            result = await llm.ainvoke([
                HumanMessage(
                    content=f"{prompt}\n\n---\n\nUser message: \"{user_query}\""
                ),
            ])
            decision = _parse_relevance(result.content or "")
        except Exception:
            logger.exception("Relevance agent LLM call failed — defaulting to respond")
            decision = {"should_respond": True, "reason": "LLM call failed — defaulting to respond."}

        logger.info(
            "Relevance decision: should_respond=%s reason=%s",
            decision["should_respond"],
            decision["reason"],
        )

        return AgentResult(
            agent=self.name,
            status="success",
            summary=json.dumps(decision),
            tool_history=[],
        )
