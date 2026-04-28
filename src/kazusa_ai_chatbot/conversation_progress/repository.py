"""MongoDB persistence helpers for conversation progress."""

from __future__ import annotations

import uuid

from pymongo.errors import DuplicateKeyError

from kazusa_ai_chatbot.conversation_progress.models import ConversationProgressScope
from kazusa_ai_chatbot.conversation_progress.policy import (
    ASSISTANT_MOVES_LIMIT,
    COLLECTION_NAME,
    OPEN_LOOPS_LIMIT,
    OVERUSED_MOVES_LIMIT,
    USER_STATE_UPDATES_LIMIT,
    expires_at_for,
)
from kazusa_ai_chatbot.db._client import get_db
from kazusa_ai_chatbot.db.schemas import ConversationEpisodeEntryDoc, ConversationEpisodeStateDoc


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

    db = await get_db()
    doc = await db[COLLECTION_NAME].find_one(
        {
            "platform": scope.platform,
            "platform_channel_id": scope.platform_channel_id,
            "global_user_id": scope.global_user_id,
        },
        projection={"_id": 0},
    )
    return doc


def _cap_strings(values: list[str], limit: int) -> list[str]:
    return [str(value) for value in values if str(value).strip()][:limit]


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

    prior_first_seen = {
        str(entry["text"]): str(entry["first_seen_at"])
        for entry in prior_entries
        if str(entry.get("text", "")).strip() and str(entry.get("first_seen_at", "")).strip()
    }
    entries: list[ConversationEpisodeEntryDoc] = []
    for raw_text in new_texts:
        text = str(raw_text).strip()
        if not text:
            continue
        entries.append({
            "text": text,
            "first_seen_at": prior_first_seen.get(text, current_timestamp),
        })
        if len(entries) >= limit:
            break
    return entries


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
    created_at = str(prior_state.get("created_at", timestamp))
    return {
        "episode_state_id": str(prior_state.get("episode_state_id", uuid.uuid4().hex)),
        "platform": scope.platform,
        "platform_channel_id": scope.platform_channel_id,
        "global_user_id": scope.global_user_id,
        "status": str(recorder_output["status"]),
        "episode_label": str(recorder_output["episode_label"]),
        "continuity": str(recorder_output["continuity"]),
        "user_state_updates": preserve_first_seen_entries(
            prior_entries=prior_state.get("user_state_updates", []),
            new_texts=recorder_output["user_state_updates"],
            current_timestamp=timestamp,
            limit=USER_STATE_UPDATES_LIMIT,
        ),
        "assistant_moves": _cap_strings(recorder_output["assistant_moves"], ASSISTANT_MOVES_LIMIT),
        "overused_moves": _cap_strings(recorder_output["overused_moves"], OVERUSED_MOVES_LIMIT),
        "open_loops": preserve_first_seen_entries(
            prior_entries=prior_state.get("open_loops", []),
            new_texts=recorder_output["open_loops"],
            current_timestamp=timestamp,
            limit=OPEN_LOOPS_LIMIT,
        ),
        "progression_guidance": str(recorder_output["progression_guidance"]),
        "turn_count": next_turn_count,
        "last_user_input": last_user_input,
        "created_at": created_at,
        "updated_at": timestamp,
        "expires_at": expires_at_for(timestamp),
    }


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

    db = await get_db()
    scope_filter = {
        "platform": document["platform"],
        "platform_channel_id": document["platform_channel_id"],
        "global_user_id": document["global_user_id"],
    }
    guarded_filter = {
        **scope_filter,
        "$or": [
            {"turn_count": {"$lt": int(document["turn_count"])}},
            {"turn_count": {"$exists": False}},
        ],
    }
    update = {"$set": dict(document)}
    try:
        result = await db[COLLECTION_NAME].update_one(guarded_filter, update, upsert=True)
    except DuplicateKeyError:
        return False
    return bool(result.upserted_id is not None or result.modified_count)
