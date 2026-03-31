"""Speech Agent — generates the final in-character reply.

Receives the full personality context, conversation history, and condensed
agent results (summaries, not raw tool output).  The supervisor's
``speech_directive`` guides how the agent should incorporate those results.

The speech agent's LLM context is free of tool descriptions, keeping the
token budget focused on personality and conversation quality.
"""

from __future__ import annotations

import logging

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, LLM_TEMPERATURE
from state import AgentResult, BotState

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


def _build_agent_context(
    agent_results: list[AgentResult],
    speech_directive: str,
) -> str:
    """Build a prompt section from agent results and the supervisor directive."""
    if not agent_results and not speech_directive:
        return ""

    parts: list[str] = []

    if speech_directive:
        parts.append(f"[Supervisor directive]\n{speech_directive}")

    for ar in agent_results:
        status_label = "success" if ar["status"] == "success" else "FAILED"
        parts.append(f"[{ar['agent']} ({status_label})]\n{ar['summary']}")

    return "\n\n".join(parts)


async def speech_agent(state: BotState) -> dict:
    """Generate the final in-character reply.

    Reads ``llm_messages`` (personality + history + user message) from the
    assembler and enriches them with agent results before calling the LLM.
    """
    # Check if the supervisor directive says to stay silent
    plan = state.get("supervisor_plan", {})
    speech_directive = plan.get("speech_directive", "") if plan else ""
    if speech_directive == "Do not respond. Stay silent.":
        logger.info("Speech agent: staying silent per supervisor directive")
        return {"response": ""}

    messages: list[BaseMessage] = list(state.get("llm_messages", []))
    if not messages:
        return {"response": "..."}

    # Inject agent context into the system prompt
    agent_results = state.get("agent_results", [])

    agent_context = _build_agent_context(agent_results, speech_directive)

    if agent_context:
        # Append agent context to the existing system message
        if messages and isinstance(messages[0], SystemMessage):
            original_system = messages[0].content
            messages[0] = SystemMessage(
                content=f"{original_system}\n\n{agent_context}"
            )
        else:
            # No system message — prepend one
            messages.insert(0, SystemMessage(content=agent_context))

    try:
        llm = _get_llm()
        prepared = _prepare_messages(messages)
        result = await llm.ainvoke(prepared)
        response = (result.content or "").strip() or "..."
    except Exception:
        logger.exception("Speech agent LLM call failed")
        response = "*stays silent*"

    return {"response": response}
