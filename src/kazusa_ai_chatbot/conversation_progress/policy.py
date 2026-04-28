"""Constants and small policy helpers for conversation progress."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from kazusa_ai_chatbot.conversation_progress.models import ConversationProgressPromptDoc

COLLECTION_NAME = "conversation_episode_state"
EPISODE_TTL = timedelta(hours=48)
CACHE_TTL_SECONDS = 60 * 60

USER_STATE_UPDATES_LIMIT = 8
ASSISTANT_MOVES_LIMIT = 8
OVERUSED_MOVES_LIMIT = 5
OPEN_LOOPS_LIMIT = 5

VALID_CONTINUITY = {"same_episode", "related_shift", "sharp_transition"}
VALID_STATUS = {"active", "suspended", "closed"}


def empty_progress_prompt_doc() -> ConversationProgressPromptDoc:
    """Return the stable empty prompt-facing progress payload.

    Returns:
        Empty progress state used when no prior episode document exists.
    """

    return {
        "status": "new_episode",
        "episode_label": "",
        "continuity": "sharp_transition",
        "turn_count": 0,
        "user_state_updates": [],
        "assistant_moves": [],
        "overused_moves": [],
        "open_loops": [],
        "progression_guidance": "",
    }


def expires_at_for(timestamp: str) -> str:
    """Compute the TTL expiry timestamp for an episode document.

    Args:
        timestamp: Current turn timestamp in ISO-8601 format.

    Returns:
        ISO-8601 timestamp when the episode state should expire.
    """

    current = parse_iso_datetime(timestamp)
    return (current + EPISODE_TTL).isoformat()


def parse_iso_datetime(value: str) -> datetime:
    """Parse an ISO-8601 timestamp into an aware UTC datetime.

    Args:
        value: ISO-8601 timestamp. A trailing ``Z`` is accepted.

    Returns:
        Timezone-aware datetime normalized to UTC.
    """

    normalized = value.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)
