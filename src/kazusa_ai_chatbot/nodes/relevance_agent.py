"""Stage 5 — Context Relevance Agent.

Analyzes the conversation history, RAG results, and user memory
to determine the current topics and whether the bot should respond at all.
Outputs a structured JSON decision.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from kazusa_ai_chatbot.config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, TOKEN_BUDGET
from kazusa_ai_chatbot.state import AssemblerOutput, BotState
from kazusa_ai_chatbot.utils import format_history_lines

logger = logging.getLogger(__name__)

_llm: ChatOpenAI | None = None

# Rough estimate: 1 token ≈ 4 characters for English text
CHARS_PER_TOKEN = 4

_RELEVANCE_PROMPT = """\
You are a context analysis engine. Your job is to analyze the conversation history, RAG context, and user memory, then output ONLY a JSON object.

You represent a Discord bot roleplaying as the character '{persona_name}'.
Your Discord user ID is '{bot_id}'.

DO NOT write explanations, analysis, or commentary. Output ONLY the JSON object.

Required fields:
- channel_topic: General topic being discussed in the channel
- user_topic: Specific topic/intent of user's latest message
- latest_message: Concise summary of what user just said
- should_respond: true if bot should reply, false otherwise

Bot should respond when:
- Message directed at bot (greeting, question, conversation)
- Continuation of ongoing conversation with bot
- Casual chat where bot is expected to participate

Bot should NOT respond when:
- Conversation between other users
- System/bot command not for this bot
- Irrelevant noise (random emoji, spam)

Output format (ONLY this, nothing else):
{{
    "channel_topic": "string",
    "user_topic": "string",
    "should_respond": true
}}
"""


def _get_llm() -> ChatOpenAI:
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model=LLM_MODEL,
            temperature=0.1,  # Low temp for structured analysis
            base_url=LLM_BASE_URL,
            api_key=LLM_API_KEY,
        )
    return _llm


def _parse_relevance_output(raw: str) -> AssemblerOutput:
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
    if text.endswith("```"):
        text = text.rsplit("```", 1)[0]
    text = text.strip()

    try:
        data = json.loads(text)
        return AssemblerOutput(
            channel_topic=str(data.get("channel_topic", "Unknown")),
            user_topic=str(data.get("user_topic", "Unknown")),
            should_respond=bool(data.get("should_respond", True))
        )
    except Exception:
        logger.exception("Failed to parse relevance LLM output: %s", raw[:200])
        # Fail-open
        return AssemblerOutput(
            channel_topic="Unknown",
            user_topic="Unknown",
            should_respond=True
        )


def _build_history_json(
    history: list[dict], persona_name: str = "assistant", bot_id: str = "unknown_bot_id"
) -> list[dict[str, str]]:
    """Convert conversation history into a JSON-native list of objects."""
    lines = []
    for name, content, role, speaker_id in format_history_lines(history, persona_name, bot_id):
        lines.append({"speaker": name, "speaker_id": speaker_id, "message": content})
    return lines


async def relevance_agent(state: BotState) -> BotState:
    """Analyze context and determine relevance using LLM."""
    personality = state.get("personality", {})
    rag_results = state.get("rag_results", [])
    user_memory = state.get("user_memory", [])
    history = state.get("conversation_history", [])
    message_text = state.get("message_text", "")
    user_name = state.get("user_name", "user")
    user_id = state.get("user_id", "unknown_user_id")
    persona_name = personality.get("name", "assistant")
    bot_id = state.get("bot_id", "unknown_bot_id")

    # ── Build Context Data ──────────────────────────────────────────
    
    # 1. Format RAG
    formatted_rag = []
    for r in rag_results:
        formatted_rag.append({
            "text": r.get('text', ''),
            "source": r.get('source', 'unknown')
        })
        
    formatted_history = _build_history_json(history, persona_name, bot_id)

    # ── Build Human Message Data ───────────────────────────────────
    human_data = {
        "current_message": {
            "speaker": user_name,
            "speaker_id": user_id,
            "message": message_text
        },
        "context": {
            "rag": formatted_rag,
            "user_memory": user_memory,
            "conversation_history": formatted_history
        }
    }

    human_content = json.dumps(human_data, indent=2, ensure_ascii=False)

    # ── Analyze Context ─────────────────────────────────────────────
    try:
        llm = _get_llm()
        bot_id = state.get("bot_id", "unknown_bot_id")
        
        formatted_prompt = _RELEVANCE_PROMPT.format(
            persona_name=persona_name,
            bot_id=bot_id
        )
        
        analysis_prompt = SystemMessage(content=formatted_prompt)
        current_human_msg = HumanMessage(content=human_content)
        
        analysis_messages = [analysis_prompt, current_human_msg]
        
        logger.warning(
            "LLM input for Relevance Agent analysis:\n%s",
            "\n---\n".join(f"[{type(m).__name__}]: {m.content}" for m in analysis_messages)
        )
        result = await llm.ainvoke(analysis_messages)
        assembler_output = _parse_relevance_output(result.content or "")
    except Exception:
        logger.exception("Relevance Agent analysis LLM call failed")
        assembler_output = AssemblerOutput(
            channel_topic="Unknown",
            user_topic="Unknown",
            latest_message=message_text,
            should_respond=True
        )

    logger.info("Relevance Agent output: %s", assembler_output)

    return {
        **state, 
        "assembler_output": assembler_output
    }

