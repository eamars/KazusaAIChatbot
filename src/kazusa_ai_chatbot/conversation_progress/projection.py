"""Projection from stored episode state to prompt-facing progress."""

from __future__ import annotations

import logging
from datetime import timedelta

from kazusa_ai_chatbot.conversation_progress.models import ConversationProgressPromptDoc
from kazusa_ai_chatbot.conversation_progress.policy import (
    ASSISTANT_MOVES_LIMIT,
    AVOID_REOPENING_LIMIT,
    MAX_ENTRY_CHARS,
    MAX_GUIDANCE_CHARS,
    MAX_LABEL_CHARS,
    MAX_MOVE_CHARS,
    MAX_THREAD_CHARS,
    NEXT_AFFORDANCES_LIMIT,
    OPEN_LOOPS_LIMIT,
    OVERUSED_MOVES_LIMIT,
    RESOLVED_THREADS_LIMIT,
    USER_STATE_UPDATES_LIMIT,
    cap_text,
    empty_progress_prompt_doc,
    enforce_progress_prompt_budget,
    parse_iso_datetime,
)
from kazusa_ai_chatbot.db.schemas import ConversationEpisodeEntryDoc, ConversationEpisodeStateDoc

logger = logging.getLogger(__name__)


def age_hint(*, first_seen_at: str, current_timestamp: str) -> str:
    """Convert a first-seen timestamp into a compact relative-age label.

    Args:
        first_seen_at: ISO-8601 timestamp when the entry first appeared.
        current_timestamp: ISO-8601 timestamp for the current turn.

    Returns:
        Human-facing relative age such as ``"just now"`` or ``"~3h ago"``.
    """

    first_seen = parse_iso_datetime(first_seen_at)
    current = parse_iso_datetime(current_timestamp)
    delta = current - first_seen
    if delta < timedelta(minutes=5):
        return "just now"
    if delta < timedelta(hours=1):
        minutes = max(5, round(delta.total_seconds() / 60 / 5) * 5)
        return_value = f"~{minutes}m ago"
        return return_value
    if delta < timedelta(hours=8):
        hours = max(1, round(delta.total_seconds() / 3600))
        return_value = f"~{hours}h ago"
        return return_value
    if first_seen.date() == current.date():
        return "earlier today"
    if (current.date() - first_seen.date()).days == 1:
        return "yesterday"
    return "earlier in this episode"


def _project_entry(
    *,
    entry: ConversationEpisodeEntryDoc,
    current_timestamp: str,
) -> dict[str, str] | None:
    """Project one stored entry when its shape is prompt-safe.

    Args:
        entry: Stored episode entry.
        current_timestamp: Current turn timestamp for age hints.

    Returns:
        Prompt-facing entry, or ``None`` for malformed legacy data.
    """

    text = entry.get("text")
    first_seen_at = entry.get("first_seen_at")
    if not isinstance(text, str) or not text.strip() or not isinstance(first_seen_at, str) or not first_seen_at.strip():
        return None
    try:
        relative_age = age_hint(
            first_seen_at=first_seen_at,
            current_timestamp=current_timestamp,
        )
    except ValueError as exc:
        logger.debug(f"Dropping progress entry with invalid first_seen_at: {exc}")
        return None
    return_value = {
        "text": cap_text(text, MAX_ENTRY_CHARS),
        "age_hint": relative_age,
    }
    return return_value


def _project_entries(
    *,
    entries: list[ConversationEpisodeEntryDoc],
    current_timestamp: str,
    limit: int,
) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    for entry in entries:
        projected = _project_entry(entry=entry, current_timestamp=current_timestamp)
        if projected is None:
            continue
        result.append(projected)
        if len(result) >= limit:
            break
    return result


def _project_string_list(*, values: list, limit: int, max_chars: int) -> list[str]:
    """Project only clean string list items.

    Args:
        values: Stored list value.
        limit: Maximum number of items to project.
        max_chars: Maximum characters per projected item.

    Returns:
        Prompt-safe clean string list.
    """

    result: list[str] = []
    for text_value in values:
        if not isinstance(text_value, str) or not text_value.strip():
            continue
        result.append(cap_text(text_value, max_chars))
        if len(result) >= limit:
            break
    return result


def project_prompt_doc(
    *,
    document: ConversationEpisodeStateDoc | None,
    current_timestamp: str,
) -> ConversationProgressPromptDoc:
    """Project a stored episode document into the prompt-facing shape.

    Args:
        document: Stored episode-state document or ``None``.
        current_timestamp: Current turn timestamp for age hints.

    Returns:
        Compact progress payload safe to place in a HumanMessage.
    """

    if document is None:
        return_value = empty_progress_prompt_doc()
        return return_value

    if document["continuity"] == "sharp_transition":
        return_value = {
            "status": "new_episode",
            "episode_label": cap_text(document.get("episode_label", ""), MAX_LABEL_CHARS),
            "continuity": "sharp_transition",
            "turn_count": int(document["turn_count"]),
            "conversation_mode": "",
            "episode_phase": "",
            "topic_momentum": "sharp_break",
            "current_thread": "",
            "user_goal": "",
            "current_blocker": "",
            "user_state_updates": [],
            "assistant_moves": [],
            "overused_moves": [],
            "open_loops": [],
            "resolved_threads": [],
            "avoid_reopening": [],
            "emotional_trajectory": "",
            "next_affordances": [],
            "progression_guidance": "",
        }
        return return_value

    prompt_doc: ConversationProgressPromptDoc = {
        "status": cap_text(document["status"], MAX_LABEL_CHARS),
        "episode_label": cap_text(document["episode_label"], MAX_LABEL_CHARS),
        "continuity": cap_text(document["continuity"], MAX_LABEL_CHARS),
        "turn_count": int(document["turn_count"]),
        "conversation_mode": cap_text(document.get("conversation_mode", ""), MAX_LABEL_CHARS),
        "episode_phase": cap_text(document.get("episode_phase", ""), MAX_LABEL_CHARS),
        "topic_momentum": cap_text(document.get("topic_momentum", ""), MAX_LABEL_CHARS),
        "current_thread": cap_text(document.get("current_thread", ""), MAX_THREAD_CHARS),
        "user_goal": cap_text(document.get("user_goal", ""), MAX_THREAD_CHARS),
        "current_blocker": cap_text(document.get("current_blocker", ""), MAX_THREAD_CHARS),
        "user_state_updates": _project_entries(
            entries=document.get("user_state_updates", []),
            current_timestamp=current_timestamp,
            limit=USER_STATE_UPDATES_LIMIT,
        ),
        "assistant_moves": _project_string_list(
            values=document.get("assistant_moves", []),
            limit=ASSISTANT_MOVES_LIMIT,
            max_chars=MAX_MOVE_CHARS,
        ),
        "overused_moves": _project_string_list(
            values=document.get("overused_moves", []),
            limit=OVERUSED_MOVES_LIMIT,
            max_chars=MAX_MOVE_CHARS,
        ),
        "open_loops": _project_entries(
            entries=document.get("open_loops", []),
            current_timestamp=current_timestamp,
            limit=OPEN_LOOPS_LIMIT,
        ),
        "resolved_threads": _project_entries(
            entries=document.get("resolved_threads", []),
            current_timestamp=current_timestamp,
            limit=RESOLVED_THREADS_LIMIT,
        ),
        "avoid_reopening": _project_entries(
            entries=document.get("avoid_reopening", []),
            current_timestamp=current_timestamp,
            limit=AVOID_REOPENING_LIMIT,
        ),
        "emotional_trajectory": cap_text(document.get("emotional_trajectory", ""), MAX_THREAD_CHARS),
        "next_affordances": _project_string_list(
            values=document.get("next_affordances", []),
            limit=NEXT_AFFORDANCES_LIMIT,
            max_chars=MAX_ENTRY_CHARS,
        ),
        "progression_guidance": cap_text(document.get("progression_guidance", ""), MAX_GUIDANCE_CHARS),
    }
    return_value = enforce_progress_prompt_budget(prompt_doc)
    return return_value
