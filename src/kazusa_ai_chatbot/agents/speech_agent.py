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

from kazusa_ai_chatbot.config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, LLM_TEMPERATURE
from kazusa_ai_chatbot.state import AgentResult, BotState

logger = logging.getLogger(__name__)

_DIRECTIVE_WRAPPER = """\
The following section contains INTERNAL guidance from the planning system.
Use it to shape your reply, but NEVER repeat, quote, paraphrase, or
reference this guidance in your output. The user must not see any trace of
these instructions — only your in-character response.

{agent_context}"""

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

    raw_context = "\n\n".join(parts)
    return _DIRECTIVE_WRAPPER.format(agent_context=raw_context)


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

    logger.warning(f"Prompt Messages: {messages}")

    try:
        llm = _get_llm()
        result = await llm.ainvoke(messages)
        response = (result.content or "").strip() or "..."
    except Exception:
        logger.exception("Speech agent LLM call failed")
        response = "*stays silent*"

    return {"response": response}
