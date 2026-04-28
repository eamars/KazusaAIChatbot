"""Process-local last-completed cache for conversation progress."""

from __future__ import annotations

import time

from kazusa_ai_chatbot.conversation_progress.models import ConversationProgressScope
from kazusa_ai_chatbot.conversation_progress.policy import CACHE_TTL_SECONDS
from kazusa_ai_chatbot.db.schemas import ConversationEpisodeStateDoc

_CacheKey = tuple[str, str, str]

_last_completed_cache: dict[_CacheKey, dict] = {}


def _cache_key(scope: ConversationProgressScope) -> _CacheKey:
    return (scope.platform, scope.platform_channel_id, scope.global_user_id)


def clear_cache() -> None:
    """Clear the process-local cache.

    Returns:
        None.
    """

    _last_completed_cache.clear()


def store_completed_document(
    *,
    scope: ConversationProgressScope,
    document: ConversationEpisodeStateDoc,
    completed_at: float | None = None,
) -> None:
    """Store the most recently written document for one scope.

    Args:
        scope: Conversation scope for the cached document.
        document: Full episode-state document just written.
        completed_at: Optional monotonic-like timestamp for tests.

    Returns:
        None.
    """

    _last_completed_cache[_cache_key(scope)] = {
        "turn_count": int(document["turn_count"]),
        "document": dict(document),
        "completed_at": completed_at if completed_at is not None else time.time(),
    }


def select_latest_document(
    *,
    scope: ConversationProgressScope,
    db_document: ConversationEpisodeStateDoc | None,
    current_time: float | None = None,
) -> tuple[ConversationEpisodeStateDoc | None, bool]:
    """Choose the cached document only when it is fresher than MongoDB.

    Args:
        scope: Conversation scope to load.
        db_document: Document loaded from MongoDB, if any.
        current_time: Optional current epoch seconds for deterministic tests.

    Returns:
        ``(document, used_cache)`` where ``used_cache`` is true only when the
        process-local document has a strictly higher turn count.
    """

    now = current_time if current_time is not None else time.time()
    key = _cache_key(scope)
    cached = _last_completed_cache.get(key)
    if cached is None:
        return db_document, False

    completed_at = float(cached["completed_at"])
    if now - completed_at > CACHE_TTL_SECONDS:
        del _last_completed_cache[key]
        return db_document, False

    cached_turn_count = int(cached["turn_count"])
    db_turn_count = int(db_document["turn_count"]) if db_document is not None else 0
    if cached_turn_count > db_turn_count:
        return dict(cached["document"]), True
    return db_document, False
