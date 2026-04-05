from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from kazusa_ai_chatbot.agents.base import AGENT_REGISTRY, register_agent
from kazusa_ai_chatbot.agents.conversation_history_agent import ConversationHistoryAgent
from kazusa_ai_chatbot.agents.memory_agent import MemoryAgent
from kazusa_ai_chatbot.agents.web_search_agent import WebSearchAgent
from kazusa_ai_chatbot.config import CONVERSATION_HISTORY_LIMIT
from kazusa_ai_chatbot.db import close_db, get_affinity, get_character_state, get_conversation_history, get_db, get_user_facts
from kazusa_ai_chatbot.mcp_client import mcp_manager
from kazusa_ai_chatbot.nodes.persona_supervisor import persona_supervisor
from kazusa_ai_chatbot.nodes.speech_agent import speech_agent
from kazusa_ai_chatbot.state import AssemblerOutput, BotState

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PERSONALITY_PATH = os.getenv("DEBUG_PERSONALITY_PATH", "")
DEFAULT_USER_ID = os.getenv("DEBUG_USER_ID", "320899931776745483")
DEFAULT_USER_NAME = os.getenv("DEBUG_USER_NAME", "EAMARS")
DEFAULT_CHANNEL_ID = os.getenv("DEBUG_CHANNEL_ID", "1485606207069880361")
DEFAULT_GUILD_ID = os.getenv("DEBUG_GUILD_ID", "")
DEFAULT_BOT_ID = os.getenv("DEBUG_BOT_ID", "1485169644888395817")
DEFAULT_CHANNEL_TOPIC = os.getenv("DEBUG_CHANNEL_TOPIC", "课间交流")
DEFAULT_USER_TOPIC = os.getenv("DEBUG_USER_TOPIC", "交流作业")


def _resolve_personality_path() -> Path:
    if DEFAULT_PERSONALITY_PATH:
        return Path(DEFAULT_PERSONALITY_PATH)
    kazusa = ROOT / "personalities" / "kazusa.json"
    if kazusa.exists():
        return kazusa
    example = ROOT / "personalities" / "example.json"
    return example


def _load_personality(path: Path) -> dict[str, Any]:
    if not path.exists():
        logger.warning("Personality file %s not found, using empty personality", path)
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _prompt(label: str, default: str) -> str:
    value = input(f"{label} [{default}]: ").strip()
    return value or default


def _register_agents() -> None:
    if "conversation_history_agent" not in AGENT_REGISTRY:
        register_agent(ConversationHistoryAgent())
    if "memory_agent" not in AGENT_REGISTRY:
        register_agent(MemoryAgent())
    if "web_search_agent" not in AGENT_REGISTRY:
        register_agent(WebSearchAgent())


async def _build_base_state(
    *,
    user_id: str,
    user_name: str,
    channel_id: str,
    guild_id: str,
    bot_id: str,
    personality: dict[str, Any],
    channel_topic: str,
    user_topic: str,
) -> BotState:
    conversation_history = await get_conversation_history(channel_id, CONVERSATION_HISTORY_LIMIT) if channel_id else []
    user_memory = await get_user_facts(user_id) if user_id else []
    affinity = await get_affinity(user_id) if user_id else 500
    character_state = await get_character_state()

    state: BotState = {
        "user_id": user_id,
        "user_name": user_name,
        "channel_id": channel_id,
        "guild_id": guild_id,
        "bot_id": bot_id,
        "message_text": "",
        "timestamp": "",
        "should_respond": True,
        "personality": personality,
        "conversation_history": conversation_history,
        "user_memory": user_memory,
        "character_state": character_state,
        "affinity": affinity,
        "assembler_output": AssemblerOutput(
            channel_topic=channel_topic,
            user_topic=user_topic,
            should_respond=True,
        ),
    }
    return state


def _with_message(base_state: BotState, message_text: str) -> BotState:
    return {
        **base_state,
        "message_text": message_text,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _print_section(title: str, payload: Any) -> None:
    print(f"\n===== {title} =====")
    if isinstance(payload, (dict, list)):
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    print(payload)


async def main() -> None:
    personality_path = _resolve_personality_path()
    personality = _load_personality(personality_path)
    user_id = DEFAULT_USER_ID
    user_name = DEFAULT_USER_NAME
    channel_id = DEFAULT_CHANNEL_ID
    guild_id = DEFAULT_GUILD_ID
    bot_id = DEFAULT_BOT_ID
    channel_topic = DEFAULT_CHANNEL_TOPIC
    user_topic = DEFAULT_USER_TOPIC

    print("Supervisor + Speech debugger")
    print("All debug context is preloaded. Type quit to exit.")

    logger.info("Using personality file: %s", personality_path)
    logger.info("Loaded personality name: %s", personality.get("name", "(unknown)"))

    await get_db()
    try:
        await mcp_manager.start()
    except Exception:
        logger.exception("MCP manager failed to start; web search tools may be unavailable")

    _register_agents()
    base_state = await _build_base_state(
        user_id=user_id,
        user_name=user_name,
        channel_id=channel_id,
        guild_id=guild_id,
        bot_id=bot_id,
        personality=personality,
        channel_topic=channel_topic,
        user_topic=user_topic,
    )

    logger.info(
        "Preloaded debug context: user_id=%s, channel_id=%s, history=%d, user_memory=%d, affinity=%d, has_character_state=%s",
        user_id,
        channel_id,
        len(base_state.get("conversation_history", [])),
        len(base_state.get("user_memory", [])),
        int(base_state.get("affinity", 500)),
        bool(base_state.get("character_state", {})),
    )
    _print_section(
        "preloaded_context",
        {
            "personality_path": str(personality_path),
            "user_id": user_id,
            "user_name": user_name,
            "channel_id": channel_id,
            "guild_id": guild_id,
            "bot_id": bot_id,
            "channel_topic": channel_topic,
            "user_topic": user_topic,
            "conversation_history_count": len(base_state.get("conversation_history", [])),
            "user_memory_count": len(base_state.get("user_memory", [])),
            "affinity": int(base_state.get("affinity", 500)),
            "has_character_state": bool(base_state.get("character_state", {})),
        },
    )

    try:
        while True:
            message_text = input("\nmessage_text: ").strip()
            if message_text.lower() in {"quit", "exit", ":q"}:
                break
            if not message_text:
                continue

            state = _with_message(base_state, message_text)

            supervisor_result = await persona_supervisor(state)
            merged_state: BotState = {**state, **supervisor_result}
            speech_result = await speech_agent(merged_state)

            _print_section("supervisor_plan", merged_state.get("supervisor_plan", {}))
            _print_section("agent_results", merged_state.get("agent_results", []))
            _print_section("speech_brief", merged_state.get("speech_brief", {}))
            _print_section("response", speech_result.get("response", ""))
    finally:
        await mcp_manager.stop()
        await close_db()


if __name__ == "__main__":
    asyncio.run(main())


def async_main() -> None:
    asyncio.run(main())
