"""Stage 6 — Persona Agent (LLM call).

Single LLM call to generate the in-character role-play response.
No tool use, no structured output — just text generation.
"""

from __future__ import annotations

import logging

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from bot.config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, LLM_TEMPERATURE
from bot.state import BotState

logger = logging.getLogger(__name__)

_llm: ChatOpenAI | None = None


def _get_llm() -> ChatOpenAI:
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            base_url=LLM_BASE_URL,
            api_key=LLM_API_KEY,
        )
    return _llm


def _prepare_messages(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Fold SystemMessage content into the first HumanMessage.

    Many local models (e.g. Qwen via LM Studio) have Jinja templates
    that reject or mishandle the ``system`` role.  Merging it into the
    first HumanMessage keeps compatibility with every model template.
    """
    system_parts: list[str] = []
    other: list[BaseMessage] = []
    for msg in messages:
        if isinstance(msg, SystemMessage):
            system_parts.append(msg.content)
        else:
            other.append(msg)

    if not system_parts or not other:
        return other or messages

    system_text = "\n\n".join(system_parts)
    first = other[0]
    other[0] = HumanMessage(content=f"{system_text}\n\n---\n\n{first.content}")
    return other


async def persona_agent(state: BotState) -> BotState:
    """Generate an in-character reply using the assembled prompt."""
    messages = state.get("llm_messages", [])
    if not messages:
        return {**state, "response": "..."}

    try:
        llm = _get_llm()
        prepared = _prepare_messages(messages)
        result = await llm.ainvoke(prepared)
        response = result.content or "..."
    except Exception:
        logger.exception("Persona agent LLM call failed")
        response = "*stays silent*"

    return {**state, "response": response}
