"""Constants and small policy helpers for conversation progress."""

from __future__ import annotations

import json
from datetime import timedelta

from kazusa_ai_chatbot.conversation_progress.models import ConversationProgressPromptDoc
from kazusa_ai_chatbot.time_boundary import parse_storage_utc_datetime

COLLECTION_NAME = "conversation_episode_state"
EPISODE_TTL = timedelta(hours=48)
CACHE_TTL_SECONDS = 60 * 60

USER_STATE_UPDATES_LIMIT = 8
ASSISTANT_MOVES_LIMIT = 8
OVERUSED_MOVES_LIMIT = 5
OPEN_LOOPS_LIMIT = 5
RESOLVED_THREADS_LIMIT = 5
AVOID_REOPENING_LIMIT = 5
NEXT_AFFORDANCES_LIMIT = 4

MAX_ENTRY_CHARS = 160
MAX_MOVE_CHARS = 120
MAX_LABEL_CHARS = 80
MAX_THREAD_CHARS = 180
MAX_GUIDANCE_CHARS = 240
MAX_PROGRESS_PROMPT_CHARS = 5000

VALID_CONTINUITY = {"same_episode", "related_shift", "sharp_transition"}
VALID_STATUS = {"active", "suspended", "closed"}


def empty_progress_prompt_doc() -> ConversationProgressPromptDoc:
    """Return the stable empty prompt-facing progress payload.

    Returns:
        Empty progress state used when no prior episode document exists.
    """

    return_value = {
        "status": "new_episode",
        "episode_label": "",
        "continuity": "sharp_transition",
        "turn_count": 0,
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


def cap_text(value: str, max_chars: int) -> str:
    """Return stripped text capped to the provided character budget.

    Args:
        value: Source text from storage, recorder output, or projection.
        max_chars: Maximum characters to keep.

    Returns:
        Stripped and capped text.

    Raises:
        TypeError: If ``value`` is not a string.
    """

    if not isinstance(value, str):
        raise TypeError("cap_text value must be a string")
    text = value.strip()
    if len(text) <= max_chars:
        return text
    return_value = text[:max_chars].rstrip()
    return return_value


def prompt_payload_chars(payload: ConversationProgressPromptDoc) -> int:
    """Return the serialized prompt payload size in characters.

    Args:
        payload: Prompt-facing conversation progress payload.

    Returns:
        Character count after JSON serialization with CJK preserved.
    """

    return_value = len(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    return return_value


def enforce_progress_prompt_budget(payload: ConversationProgressPromptDoc) -> ConversationProgressPromptDoc:
    """Drop low-priority prompt fields until the progress payload fits budget.

    Args:
        payload: Per-field-capped prompt-facing progress document.

    Returns:
        Payload capped to ``MAX_PROGRESS_PROMPT_CHARS``.
    """

    result: ConversationProgressPromptDoc = dict(payload)
    if prompt_payload_chars(result) <= MAX_PROGRESS_PROMPT_CHARS:
        return result

    result["resolved_threads"] = []
    if prompt_payload_chars(result) <= MAX_PROGRESS_PROMPT_CHARS:
        return result

    result["assistant_moves"] = []
    if prompt_payload_chars(result) <= MAX_PROGRESS_PROMPT_CHARS:
        return result

    result["user_state_updates"] = result["user_state_updates"][:4]
    if prompt_payload_chars(result) <= MAX_PROGRESS_PROMPT_CHARS:
        return result

    result["open_loops"] = result["open_loops"][:3]
    if prompt_payload_chars(result) <= MAX_PROGRESS_PROMPT_CHARS:
        return result

    result["avoid_reopening"] = []
    return result


def expires_at_for(storage_timestamp_utc: str) -> str:
    """Compute the TTL expiry timestamp for an episode document.

    Args:
        storage_timestamp_utc: Current turn storage UTC timestamp.

    Returns:
        ISO-8601 timestamp when the episode state should expire.
    """

    current = parse_storage_utc_datetime(storage_timestamp_utc)
    return_value = (current + EPISODE_TTL).isoformat()
    return return_value
