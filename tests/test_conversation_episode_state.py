"""Tests for conversation episode state repository helpers."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from kazusa_ai_chatbot.conversation_progress import repository
from kazusa_ai_chatbot.conversation_progress.models import ConversationProgressScope


def test_preserve_first_seen_entries_keeps_llm_returned_text_timestamp() -> None:
    """Exact recorder-returned text keeps its original first_seen_at."""

    entries = repository.preserve_first_seen_entries(
        prior_entries=[
            {"text": "user has an unresolved draft outline", "first_seen_at": "2026-04-28T01:00:00+00:00"},
        ],
        new_texts=[
            "user has an unresolved draft outline",
            "user is worried about the introduction",
        ],
        current_timestamp="2026-04-28T04:00:00+00:00",
        limit=8,
    )

    assert entries == [
        {"text": "user has an unresolved draft outline", "first_seen_at": "2026-04-28T01:00:00+00:00"},
        {"text": "user is worried about the introduction", "first_seen_at": "2026-04-28T04:00:00+00:00"},
    ]


def test_build_episode_state_doc_caps_lists_and_increments_turn_count() -> None:
    """Built documents are capped and advance from the prior turn count."""

    scope = ConversationProgressScope("qq", "channel-1", "user-1")
    document = repository.build_episode_state_doc(
        scope=scope,
        timestamp="2026-04-28T04:00:00+00:00",
        prior_episode_state={
            "episode_state_id": "episode-1",
            "turn_count": 3,
            "created_at": "2026-04-28T01:00:00+00:00",
            "user_state_updates": [],
            "open_loops": [],
        },
        recorder_output={
            "status": "active",
            "episode_label": "draft_help",
            "continuity": "same_episode",
            "user_state_updates": [f"state {index}" for index in range(12)],
            "assistant_moves": [f"move {index}" for index in range(12)],
            "overused_moves": [f"overused {index}" for index in range(12)],
            "open_loops": [f"loop {index}" for index in range(12)],
            "progression_guidance": "answer the missing outline point",
        },
        last_user_input="what about the third point?",
    )

    assert document["turn_count"] == 4
    assert document["episode_state_id"] == "episode-1"
    assert document["created_at"] == "2026-04-28T01:00:00+00:00"
    assert len(document["user_state_updates"]) == 8
    assert len(document["assistant_moves"]) == 8
    assert len(document["overused_moves"]) == 5
    assert len(document["open_loops"]) == 5
    assert document["expires_at"] == "2026-04-30T04:00:00+00:00"


@pytest.mark.asyncio
async def test_load_episode_state_queries_by_scope_without_mongo_id(monkeypatch) -> None:
    """The repository loads by platform/channel/user and excludes _id."""

    collection = MagicMock()
    collection.find_one = AsyncMock(return_value=None)
    db = {repository.COLLECTION_NAME: collection}
    monkeypatch.setattr(repository, "get_db", AsyncMock(return_value=db))

    await repository.load_episode_state(
        scope=ConversationProgressScope("qq", "channel-1", "user-1"),
    )

    collection.find_one.assert_awaited_once_with(
        {
            "platform": "qq",
            "platform_channel_id": "channel-1",
            "global_user_id": "user-1",
        },
        projection={"_id": 0},
    )


@pytest.mark.asyncio
async def test_upsert_episode_state_uses_turn_count_guard(monkeypatch) -> None:
    """Writes include a strict turn_count freshness guard."""

    update_result = MagicMock(upserted_id=None, modified_count=1)
    collection = MagicMock()
    collection.update_one = AsyncMock(return_value=update_result)
    db = {repository.COLLECTION_NAME: collection}
    monkeypatch.setattr(repository, "get_db", AsyncMock(return_value=db))

    written = await repository.upsert_episode_state_guarded(
        document={
            "episode_state_id": "episode-1",
            "platform": "qq",
            "platform_channel_id": "channel-1",
            "global_user_id": "user-1",
            "status": "active",
            "episode_label": "draft_help",
            "continuity": "same_episode",
            "user_state_updates": [],
            "assistant_moves": [],
            "overused_moves": [],
            "open_loops": [],
            "progression_guidance": "",
            "turn_count": 2,
            "last_user_input": "hello",
            "created_at": "2026-04-28T04:00:00+00:00",
            "updated_at": "2026-04-28T04:00:00+00:00",
            "expires_at": "2026-04-30T04:00:00+00:00",
        },
    )

    assert written is True
    filter_arg = collection.update_one.call_args[0][0]
    assert filter_arg["platform"] == "qq"
    assert filter_arg["$or"] == [
        {"turn_count": {"$lt": 2}},
        {"turn_count": {"$exists": False}},
    ]
