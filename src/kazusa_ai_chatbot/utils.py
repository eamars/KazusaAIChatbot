"""Shared utility functions for the Kazusa AI chatbot."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from json_repair import repair_json

if TYPE_CHECKING:
    from kazusa_ai_chatbot.state import ChatMessage


def format_history_lines(
    history: list[ChatMessage],
    persona_name: str = "assistant",
    bot_id: str = "unknown_bot_id",
    limit: int | None = None,
) -> list[tuple[str, str, str, str]]:
    """Return a list of ``(name, content, role, speaker_id)`` tuples from chat history.

    *   User messages use the sender's display name.
    *   Assistant messages use *persona_name* so the LLM sees the character name.
    *   When *limit* is given, only the last *limit* messages are returned.
    """
    msgs = history[-limit:] if limit else history
    result: list[tuple[str, str, str, str]] = []
    for msg in msgs:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "assistant":
            name = persona_name
            speaker_id = bot_id
        else:
            name = msg.get("name", "user")
            speaker_id = msg.get("user_id", "unknown_user_id")
            
        # Clean the name to be alphanumeric/underscore only (OpenAI name requirement)
        clean_name = re.sub(r'[^a-zA-Z0-9_-]', '', name)
        if not clean_name:
            clean_name = "user" if role == "user" else "assistant"
            
        result.append((clean_name, content, role, speaker_id))
    return result


def format_history_text(
    history: list[ChatMessage],
    persona_name: str = "assistant",
    bot_id: str = "unknown_bot_id",
    limit: int | None = None,
) -> str:
    """Format chat history into a plain-text block.

    Each line is ``[Name]: message content``.
    Returns an empty string when *history* is empty.
    """
    lines = format_history_lines(history, persona_name, bot_id, limit)
    if not lines:
        return ""
    return "\n".join(f"[{label}]: {content}" for label, content, _, _ in lines)


def parse_llm_json_output(raw_output: str) -> dict:
    """Parse LLM JSON output, handling markdown fences and malformed JSON.
    
    Args:
        raw_output: Raw string output from LLM
        
    Returns:
        Parsed JSON object as dict, or empty dict if parsing fails
    """
    if not raw_output:
        return {}
    
    try:
        # Strip markdown fences and clean up
        raw = raw_output.strip().strip("`").strip()
        if raw.startswith("json"):
            raw = raw[4:].strip()
        
        # Use repair_json which handles both valid and broken JSON
        parsed = repair_json(raw, return_objects=True)
        
        if isinstance(parsed, dict):
            return parsed
        else:
            return {}
            
    except Exception:
        return {}
