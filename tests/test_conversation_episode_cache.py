"""Tests for the conversation progress last-completed cache."""

from __future__ import annotations

from kazusa_ai_chatbot.conversation_progress import cache
from kazusa_ai_chatbot.conversation_progress.models import ConversationProgressScope


def test_cache_document_wins_when_turn_count_is_higher() -> None:
    """A fresher completed write supplies the next-turn load."""

    cache.clear_cache()
    scope = ConversationProgressScope("qq", "channel-1", "user-1")
    cache.store_completed_document(
        scope=scope,
        document={
            "turn_count": 3,
            "platform": "qq",
            "platform_channel_id": "channel-1",
            "global_user_id": "user-1",
            "status": "active",
            "episode_label": "essay_help",
            "continuity": "same_episode",
        },
        completed_at=100.0,
    )

    document, used_cache = cache.select_latest_document(
        scope=scope,
        db_document={"turn_count": 2},
        current_time=110.0,
    )

    assert used_cache is True
    assert document["turn_count"] == 3


def test_db_document_wins_when_cache_is_not_strictly_newer() -> None:
    """Equal turn counts do not use cache fallback."""

    cache.clear_cache()
    scope = ConversationProgressScope("qq", "channel-1", "user-1")
    cache.store_completed_document(
        scope=scope,
        document={"turn_count": 2},
        completed_at=100.0,
    )

    document, used_cache = cache.select_latest_document(
        scope=scope,
        db_document={"turn_count": 2, "source": "db"},
        current_time=110.0,
    )

    assert used_cache is False
    assert document["source"] == "db"


def test_stale_cache_entry_is_evicted() -> None:
    """Expired process-local entries are ignored."""

    cache.clear_cache()
    scope = ConversationProgressScope("qq", "channel-1", "user-1")
    cache.store_completed_document(
        scope=scope,
        document={"turn_count": 9},
        completed_at=100.0,
    )

    document, used_cache = cache.select_latest_document(
        scope=scope,
        db_document={"turn_count": 1},
        current_time=3701.0,
    )

    assert used_cache is False
    assert document["turn_count"] == 1
