"""Projection from stored episode state to prompt-facing progress."""

from __future__ import annotations

from datetime import timedelta

from kazusa_ai_chatbot.conversation_progress.models import ConversationProgressPromptDoc
from kazusa_ai_chatbot.conversation_progress.policy import empty_progress_prompt_doc, parse_iso_datetime
from kazusa_ai_chatbot.db.schemas import ConversationEpisodeEntryDoc, ConversationEpisodeStateDoc


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
        return f"~{minutes}m ago"
    if delta < timedelta(hours=8):
        hours = max(1, round(delta.total_seconds() / 3600))
        return f"~{hours}h ago"
    if first_seen.date() == current.date():
        return "earlier today"
    if (current.date() - first_seen.date()).days == 1:
        return "yesterday"
    return "earlier in this episode"


def _project_entries(
    *,
    entries: list[ConversationEpisodeEntryDoc],
    current_timestamp: str,
) -> list[dict[str, str]]:
    return [
        {
            "text": str(entry["text"]),
            "age_hint": age_hint(
                first_seen_at=str(entry["first_seen_at"]),
                current_timestamp=current_timestamp,
            ),
        }
        for entry in entries
        if str(entry.get("text", "")).strip() and str(entry.get("first_seen_at", "")).strip()
    ]


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
        return empty_progress_prompt_doc()

    if document["continuity"] == "sharp_transition":
        return {
            "status": "new_episode",
            "episode_label": str(document.get("episode_label", "")),
            "continuity": "sharp_transition",
            "turn_count": int(document["turn_count"]),
            "user_state_updates": [],
            "assistant_moves": [],
            "overused_moves": [],
            "open_loops": [],
            "progression_guidance": "",
        }

    return {
        "status": str(document["status"]),
        "episode_label": str(document["episode_label"]),
        "continuity": str(document["continuity"]),
        "turn_count": int(document["turn_count"]),
        "user_state_updates": _project_entries(
            entries=document.get("user_state_updates", []),
            current_timestamp=current_timestamp,
        ),
        "assistant_moves": [str(item) for item in document.get("assistant_moves", []) if str(item).strip()],
        "overused_moves": [str(item) for item in document.get("overused_moves", []) if str(item).strip()],
        "open_loops": _project_entries(
            entries=document.get("open_loops", []),
            current_timestamp=current_timestamp,
        ),
        "progression_guidance": str(document.get("progression_guidance", "")),
    }
