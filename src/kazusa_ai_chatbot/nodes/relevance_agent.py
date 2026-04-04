"""Stage 5 — Context Relevance Agent.

Loads conversational context from MongoDB, then analyzes that context
to determine the current topics and whether the bot should respond at all.
Outputs a structured JSON decision.
"""

from __future__ import annotations

import asyncio
import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from kazusa_ai_chatbot.config import CONVERSATION_HISTORY_LIMIT, LLM_API_KEY, LLM_BASE_URL, LLM_MODEL
from kazusa_ai_chatbot.db import AFFINITY_DEFAULT, get_affinity, get_character_state, get_conversation_history, get_user_facts
from kazusa_ai_chatbot.state import AssemblerOutput, BotState, CharacterState, ChatMessage
from kazusa_ai_chatbot.utils import format_history_lines

logger = logging.getLogger(__name__)

_llm: ChatOpenAI | None = None

_RELEVANCE_PROMPT = """\
You are a context analysis engine. Your job is to analyze the conversation history, user memory, character state, and current message, then output ONLY a JSON object.

You represent a Discord bot roleplaying as the character '{persona_name}'.
Your Discord user ID is '{bot_id}'.

DO NOT write explanations, analysis, or commentary. Output ONLY the JSON object.

Required fields:
- channel_topic: General topic being discussed in the channel
- user_topic: Specific topic/intent of user's latest message
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


async def _load_context(state: BotState) -> tuple[list[ChatMessage], list[str], CharacterState, int]:
    channel_id = state.get("channel_id", "")
    user_id = state.get("user_id", "")

    history_task = get_conversation_history(channel_id, limit=CONVERSATION_HISTORY_LIMIT) if channel_id else asyncio.sleep(0, result=[])
    facts_task = get_user_facts(user_id) if user_id else asyncio.sleep(0, result=[])
    character_state_task = get_character_state()
    affinity_task = get_affinity(user_id) if user_id else asyncio.sleep(0, result=AFFINITY_DEFAULT)

    raw_history, raw_facts, raw_character_state, raw_affinity = await asyncio.gather(
        history_task,
        facts_task,
        character_state_task,
        affinity_task,
        return_exceptions=True,
    )

    if isinstance(raw_history, Exception):
        logger.exception("Failed to fetch conversation history", exc_info=raw_history)
        history = []
    else:
        history = [
            ChatMessage(
                role=doc.get("role", "user"),
                user_id=doc.get("user_id", ""),
                name=doc.get("name", "unknown"),
                content=doc.get("content", ""),
            )
            for doc in raw_history
        ]

    if isinstance(raw_facts, Exception):
        logger.exception("Failed to fetch user facts", exc_info=raw_facts)
        user_memory = []
    else:
        user_memory = [str(fact) for fact in raw_facts]

    if isinstance(raw_character_state, Exception):
        logger.exception("Failed to fetch character state", exc_info=raw_character_state)
        character_state = CharacterState()
    else:
        character_state = CharacterState(
            mood=raw_character_state.get("mood", "neutral"),
            emotional_tone=raw_character_state.get("emotional_tone", "balanced"),
            recent_events=raw_character_state.get("recent_events", []),
            updated_at=raw_character_state.get("updated_at", ""),
        )

    if isinstance(raw_affinity, Exception):
        logger.exception("Failed to fetch affinity", exc_info=raw_affinity)
        affinity = AFFINITY_DEFAULT
    else:
        affinity = int(raw_affinity)

    return history, user_memory, character_state, affinity


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
    message_text = state.get("message_text", "")
    user_name = state.get("user_name", "user")
    user_id = state.get("user_id", "unknown_user_id")
    persona_name = personality.get("name", "assistant")
    bot_id = state.get("bot_id", "unknown_bot_id")
    history, user_memory, character_state, affinity = await _load_context(state)

    # ── Build Context Data ──────────────────────────────────────────
    formatted_history = _build_history_json(history, persona_name, bot_id)

    # ── Build Human Message Data ───────────────────────────────────
    human_data = {
        "current_message": {
            "speaker": user_name,
            "speaker_id": user_id,
            "message": message_text
        },
        "context": {
            "user_memory": user_memory,
            "conversation_history": formatted_history,
            "character_state": character_state,
            "affinity": affinity,
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
            should_respond=True
        )

    logger.info("Relevance Agent output: %s", assembler_output)

    return {
        "conversation_history": history,
        "user_memory": user_memory,
        "character_state": character_state,
        "affinity": affinity,
        "assembler_output": assembler_output
    }

