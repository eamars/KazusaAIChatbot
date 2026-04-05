"""Speech Agent — generates the final in-character reply.

Consumes only a sanitized ``speech_brief`` produced by the supervisor.
The brief contains approved personality context plus generation guidance,
without exposing raw history, internal state, or raw tool output.
"""

from __future__ import annotations

import json
import logging

from json_repair import repair_json
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from kazusa_ai_chatbot.config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, LLM_TEMPERATURE
from kazusa_ai_chatbot.state import BotState

logger = logging.getLogger(__name__)

_SPEECH_SYSTEM_PROMPT = """\
# You are an expert role-player acting as a specific persona.

Role: 
- Your goal is to interact with the user while maintaining 100% consistency with the personality, history, and physical traits defined in your memory. 
- You are this person; you are not an AI.
- Always use "I," "me," and "my." Never describe yourself in the third person.
- Use the provided sanitized JSON brief to guide your response.
- You must reply in the language specified by `response_brief.response_language`.
- You must type speech only - no gesture or descripitive language.
- You must reply to the `user_input_brief` based on `response_brief` acting based on `personality`.
- You must NOT include thought process in the output
- You shall respond on the JSON format as demonstrated. 
- You do not ask follow up question if not explicitly requested so. 

# Output Format (raw JSON text — no markdown wrapping):
{
  "speech": "Your speech here"
}
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
        "Calling LLM for speech generation. Topics: %d, Facts: %d, Response goal: %s",
        len(response_brief.get("topics_to_cover", [])),
        len(response_brief.get("facts_to_cover", [])),
        response_brief.get("response_goal", "")[:50] + "..." if len(response_brief.get("response_goal", "")) > 50 else response_brief.get("response_goal", "")
    )

    try:
        llm = _get_llm()
        logger.info(
            "LLM input for Speech Agent:\n%s",
            "\n---\n".join(f"[{type(m).__name__}]: {m.content}" for m in messages)
        )
        result = await llm.ainvoke(messages)
        raw_response = (result.content or "").strip() or "..."
        
        # Debug: Print raw message output using same format as input
        logger.info(
            "LLM output for Speech Agent:\n%s",
            f"[{type(result).__name__}]: {result.content}"
        )

        # Strip markdown fence
        raw_response = raw_response.strip("```").strip("json")
        
        # Extract speech from JSON output
        try:
            # Use repair_json which handles both valid and broken JSON
            # repair_json returns a Python object, not a string
            parsed = repair_json(raw_response, return_objects=True)
            
            if isinstance(parsed, dict) and "speech" in parsed:
                response = str(parsed["speech"]).strip()
                logger.info(f"Successfully parsed speech from JSON: {response[:100]}...")
            else:
                logger.warning(f"LLM output missing 'speech' field, using raw response. Parsed type: {type(parsed)}, Parsed: {parsed}")
                response = raw_response
        except Exception as e:
            logger.warning(f"Failed to parse LLM output as JSON, using raw response. Error: {e}. Raw: {raw_response[:200]}...")
            response = raw_response
        
        # Fallback if response is empty
        if not response:
            response = "..."
        
        logger.info(f"Final speech agent response: {response[:100]}...")
        
    except Exception:
        logger.exception("Speech agent LLM call failed")
        response = "*stays silent*"

    return {"response": response}
