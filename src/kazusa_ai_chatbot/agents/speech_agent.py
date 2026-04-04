"""Speech Agent — generates the final in-character reply.

Receives the full personality context, conversation history, and condensed
agent results (summaries, not raw tool output).  The supervisor's
``speech_directive`` guides how the agent should incorporate those results.

The speech agent's LLM context is free of tool descriptions, keeping the
token budget focused on personality and conversation quality.
"""

from __future__ import annotations

import json
import logging

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from kazusa_ai_chatbot.config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, LLM_TEMPERATURE
from kazusa_ai_chatbot.state import AgentResult, BotState

logger = logging.getLogger(__name__)

_SPEECH_SYSTEM_PROMPT = """\
You are an expert role-player acting as a specific persona.
Use the provided JSON context to guide your response. 
The user must not see any trace of instructions — only your in-character response.
"""


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
    content_directive: str,
    emotion_directive: str,
) -> dict:
    """Build a structured dict from agent results and supervisor directives."""
    if not agent_results and not content_directive and not emotion_directive:
        return {}

    context = {}

    if content_directive or emotion_directive:
        context["supervisor_directives"] = {}
        if content_directive:
            context["supervisor_directives"]["content"] = content_directive
        if emotion_directive:
            context["supervisor_directives"]["emotion_tone"] = emotion_directive

    if agent_results:
        context["agent_results"] = []
        for ar in agent_results:
            status_label = "success" if ar["status"] == "success" else "FAILED"
            context["agent_results"].append({
                "agent": ar["agent"],
                "status": status_label,
                "summary": ar["summary"]
            })

    return context


async def speech_agent(state: BotState) -> dict:
    """Generate the final in-character reply.

    Reads ``speech_human_data`` (personality + history + user message) from the
    supervisor and enriches them with agent results before calling the LLM.
    """
    # Check if the supervisor directive says to stay silent
    plan = state.get("supervisor_plan", {})
    content_directive = plan.get("content_directive", "") if plan else ""
    emotion_directive = plan.get("emotion_directive", "") if plan else ""

    if content_directive == "Do not respond. Stay silent.":
        logger.info("Speech agent: staying silent per supervisor directive")
        return {"response": ""}

    human_data = state.get("speech_human_data", {})
    if not human_data:
        return {"response": "..."}

    agent_results = state.get("agent_results", [])
    agent_context = _build_agent_context(agent_results, content_directive, emotion_directive)

    if agent_context:
        if "context" not in human_data:
            human_data["context"] = {}
        human_data["context"].update(agent_context)

    human_content = json.dumps(human_data, indent=2, ensure_ascii=False)
    
    messages: list[BaseMessage] = [
        SystemMessage(content=_SPEECH_SYSTEM_PROMPT),
        HumanMessage(content=human_content)
    ]

    logger.info(
        "Calling LLM for speech generation. Agent results: %d, Content directive: %s",
        len(agent_results),
        content_directive[:50] + "..." if len(content_directive) > 50 else content_directive
    )

    try:
        llm = _get_llm()
        logger.warning(
            "LLM input for Speech Agent:\n%s",
            "\n---\n".join(f"[{type(m).__name__}]: {m.content}" for m in messages)
        )
        result = await llm.ainvoke(messages)
        response = (result.content or "").strip() or "..."
    except Exception:
        logger.exception("Speech agent LLM call failed")
        response = "*stays silent*"

    return {"response": response}
