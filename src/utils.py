"""Shared utility helpers used across agents and nodes."""

from __future__ import annotations

from state import ChatMessage


def format_history_lines(
    history: list[ChatMessage],
    persona_name: str = "assistant",
    limit: int | None = None,
) -> list[tuple[str, str, str]]:
    """Return a list of ``(label, content, role)`` tuples from chat history.

    *   User messages are labelled with the sender's display name.
    *   Assistant messages are labelled with *persona_name* so the LLM sees
        the character name rather than a generic "bot".
    *   When *limit* is given, only the last *limit* messages are returned.
    """
    msgs = history[-limit:] if limit else history
    result: list[tuple[str, str, str]] = []
    for msg in msgs:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "assistant":
            label = persona_name
        else:
            label = msg.get("name", "user")
        result.append((label, content, role))
    return result


def format_history_text(
    history: list[ChatMessage],
    persona_name: str = "assistant",
    limit: int | None = None,
) -> str:
    """Format chat history into a plain-text block.

    Each line is ``[Name]: message content``.
    Returns an empty string when *history* is empty.
    """
    lines = format_history_lines(history, persona_name, limit)
    if not lines:
        return ""
    return "\n".join(f"[{label}]: {content}" for label, content, _ in lines)
