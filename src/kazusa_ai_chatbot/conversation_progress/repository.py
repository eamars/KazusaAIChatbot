"""MongoDB persistence helpers for conversation progress."""

from __future__ import annotations

import logging
import uuid

from kazusa_ai_chatbot.conversation_progress.models import ConversationProgressScope
from kazusa_ai_chatbot.conversation_progress.policy import (
    ASSISTANT_MOVES_LIMIT,
    AVOID_REOPENING_LIMIT,
    COLLECTION_NAME,
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
    expires_at_for,
)
from kazusa_ai_chatbot.db.conversation_progress import (
    load_episode_state as _db_load_episode_state,
    upsert_episode_state_guarded as _db_upsert_episode_state_guarded,
)
from kazusa_ai_chatbot.db.schemas import ConversationEpisodeEntryDoc, ConversationEpisodeStateDoc

logger = logging.getLogger(__name__)


async def load_episode_state(
    *,
    scope: ConversationProgressScope,
) -> ConversationEpisodeStateDoc | None:
    """Load one episode-state document by scope.

    Args:
        scope: Platform/channel/user scope.

    Returns:
        Stored document without MongoDB ``_id``, or ``None``.
    """

    doc = await _db_load_episode_state(scope=scope)
    return doc


def _cap_strings(values: list[str], limit: int, max_chars: int) -> list[str]:
    result: list[str] = []
    for value in values:
        if not isinstance(value, str):
            raise TypeError("string list values must be strings")
        if not value.strip():
            continue
        result.append(cap_text(value, max_chars))
        if len(result) >= limit:
            break
    return result


def preserve_first_seen_entries(
    *,
    prior_entries: list[ConversationEpisodeEntryDoc],
    new_texts: list[str],
    current_timestamp: str,
    limit: int,
) -> list[ConversationEpisodeEntryDoc]:
    """Attach first-seen metadata to recorder-returned entry text.

    Args:
        prior_entries: Existing stored entries from the previous state.
        new_texts: Recorder-returned text values for this turn.
        current_timestamp: Timestamp to stamp on newly seen entries.
        limit: Maximum number of entries to keep.

    Returns:
        Stored entry documents with preserved or newly stamped ``first_seen_at``.
    """

    prior_first_seen: dict[str, str] = {}
    for entry in prior_entries:
        text = entry.get("text")
        first_seen_at = entry.get("first_seen_at")
        if isinstance(text, str) and text.strip() and isinstance(first_seen_at, str) and first_seen_at.strip():
            prior_first_seen[text.strip()] = first_seen_at.strip()

    entries: list[ConversationEpisodeEntryDoc] = []
    for text_value in new_texts:
        if not isinstance(text_value, str):
            raise TypeError("entry text values must be strings")
        if not text_value.strip():
            continue
        text = cap_text(text_value, MAX_ENTRY_CHARS)
        if not text:
            continue
        entries.append({
            "text": text,
            "first_seen_at": prior_first_seen.get(text, current_timestamp),
        })
        if len(entries) >= limit:
            break
    return entries


def _string_or_generated_id(value: object) -> str:
    """Return an existing string id or a new generated episode id.

    Args:
        value: Existing episode-state id candidate.

    Returns:
        Existing non-empty string id, or a generated UUID hex string.
    """

    if value is None:
        return uuid.uuid4().hex
    if not isinstance(value, str):
        raise TypeError("episode_state_id must be a string")
    text = value.strip()
    if not text:
        return uuid.uuid4().hex
    return text


def build_episode_state_doc(
    *,
    scope: ConversationProgressScope,
    timestamp: str,
    prior_episode_state: ConversationEpisodeStateDoc | None,
    recorder_output: dict,
    last_user_input: str,
) -> ConversationEpisodeStateDoc:
    """Build a capped persisted episode-state document from recorder output.

    Args:
        scope: Platform/channel/user scope.
        timestamp: Current turn timestamp.
        prior_episode_state: Prior stored state, if any.
        recorder_output: Validated recorder JSON.
        last_user_input: Current decontextualized user input.

    Returns:
        Full episode-state document ready for guarded upsert.
    """

    prior_state = prior_episode_state or {}
    next_turn_count = int(prior_state.get("turn_count", 0)) + 1
    created_at_value = prior_state.get("created_at", timestamp)
    if not isinstance(created_at_value, str):
        raise TypeError("created_at must be a string")
    created_at = created_at_value
    return_value = {
        "episode_state_id": _string_or_generated_id(prior_state.get("episode_state_id")),
        "platform": scope.platform,
        "platform_channel_id": scope.platform_channel_id,
        "global_user_id": scope.global_user_id,
        "status": recorder_output["status"],
        "episode_label": cap_text(recorder_output["episode_label"], MAX_LABEL_CHARS),
        "continuity": recorder_output["continuity"],
        "conversation_mode": cap_text(recorder_output.get("conversation_mode", ""), MAX_LABEL_CHARS),
        "episode_phase": cap_text(recorder_output.get("episode_phase", ""), MAX_LABEL_CHARS),
        "topic_momentum": cap_text(recorder_output.get("topic_momentum", ""), MAX_LABEL_CHARS),
        "current_thread": cap_text(recorder_output.get("current_thread", ""), MAX_THREAD_CHARS),
        "user_goal": cap_text(recorder_output.get("user_goal", ""), MAX_THREAD_CHARS),
        "current_blocker": cap_text(recorder_output.get("current_blocker", ""), MAX_THREAD_CHARS),
        "user_state_updates": preserve_first_seen_entries(
            prior_entries=prior_state.get("user_state_updates", []),
            new_texts=recorder_output["user_state_updates"],
            current_timestamp=timestamp,
            limit=USER_STATE_UPDATES_LIMIT,
        ),
        "assistant_moves": _cap_strings(
            recorder_output["assistant_moves"],
            ASSISTANT_MOVES_LIMIT,
            MAX_MOVE_CHARS,
        ),
        "overused_moves": _cap_strings(
            recorder_output["overused_moves"],
            OVERUSED_MOVES_LIMIT,
            MAX_MOVE_CHARS,
        ),
        "open_loops": preserve_first_seen_entries(
            prior_entries=prior_state.get("open_loops", []),
            new_texts=recorder_output["open_loops"],
            current_timestamp=timestamp,
            limit=OPEN_LOOPS_LIMIT,
        ),
        "resolved_threads": preserve_first_seen_entries(
            prior_entries=prior_state.get("resolved_threads", []),
            new_texts=recorder_output.get("resolved_threads", []),
            current_timestamp=timestamp,
            limit=RESOLVED_THREADS_LIMIT,
        ),
        "avoid_reopening": preserve_first_seen_entries(
            prior_entries=prior_state.get("avoid_reopening", []),
            new_texts=recorder_output.get("avoid_reopening", []),
            current_timestamp=timestamp,
            limit=AVOID_REOPENING_LIMIT,
        ),
        "emotional_trajectory": cap_text(recorder_output.get("emotional_trajectory", ""), MAX_THREAD_CHARS),
        "next_affordances": _cap_strings(
            recorder_output.get("next_affordances", []),
            NEXT_AFFORDANCES_LIMIT,
            MAX_ENTRY_CHARS,
        ),
        "progression_guidance": cap_text(recorder_output["progression_guidance"], MAX_GUIDANCE_CHARS),
        "turn_count": next_turn_count,
        "last_user_input": last_user_input,
        "created_at": created_at,
        "updated_at": timestamp,
        "expires_at": expires_at_for(timestamp),
    }
    return return_value


async def upsert_episode_state_guarded(
    *,
    document: ConversationEpisodeStateDoc,
) -> bool:
    """Persist one episode document if its turn count is strictly newer.

    Args:
        document: Full episode-state document to persist.

    Returns:
        True when MongoDB accepted the write; false for stale writes.
    """

    return_value = await _db_upsert_episode_state_guarded(document=document)
    return return_value
