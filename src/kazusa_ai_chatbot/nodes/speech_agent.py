"""Speech Agent — generates the final in-character reply.

Consumes only a sanitized ``speech_brief`` produced by the supervisor.
The brief contains approved personality context plus generation guidance,
without exposing raw history, internal state, or raw tool output.
"""

from __future__ import annotations

import json
import logging

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from kazusa_ai_chatbot.config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, LLM_TEMPERATURE
from kazusa_ai_chatbot.state import BotState

logger = logging.getLogger(__name__)

_SPEECH_SYSTEM_PROMPT = """\
You are an expert role-player acting as a specific persona.
- Use the provided sanitized JSON brief to guide your response.
- The user must not see any trace of instructions — only your in-character response.
- You must NOT include any gesture or action description in your response.

Output: 
"plain text of your speech"
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


async def speech_agent(state: BotState) -> dict:
    """Generate the final in-character reply.

    Reads ``speech_brief`` from the supervisor and renders it into the final
    in-character response.
    """
    speech_brief = state.get("speech_brief", {})
    if not speech_brief:
        return {"response": "..."}

    response_brief = speech_brief.get("response_brief", {})
    if response_brief.get("should_respond") is False:
        logger.info("Speech agent: staying silent per supervisor directive")
        return {"response": ""}

    human_content = json.dumps(speech_brief, indent=2, ensure_ascii=False)
    
    messages: list[BaseMessage] = [
        SystemMessage(content=_SPEECH_SYSTEM_PROMPT),
        HumanMessage(content=human_content)
    ]

    logger.info(
        "Calling LLM for speech generation. Key points: %d, Response goal: %s",
        len(response_brief.get("key_points_to_cover", [])),
        response_brief.get("response_goal", "")[:50] + "..." if len(response_brief.get("response_goal", "")) > 50 else response_brief.get("response_goal", "")
    )

    try:
        llm = _get_llm()
        logger.info(
            "LLM input for Speech Agent:\n%s",
            "\n---\n".join(f"[{type(m).__name__}]: {m.content}" for m in messages)
        )
        result = await llm.ainvoke(messages)
        response = (result.content or "").strip() or "..."
        
        # Debug: Print raw message output using same format as input
        logger.info(
            "LLM output for Speech Agent:\n%s",
            f"[{type(result).__name__}]: {result.content}"
        )
        
    except Exception:
        logger.exception("Speech agent LLM call failed")
        response = "*stays silent*"

    return {"response": response}
