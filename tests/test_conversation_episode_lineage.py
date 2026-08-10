"""Persistence tests for settled cognitive-episode lineage."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from kazusa_ai_chatbot.db import conversation as conversation_module


@pytest.mark.asyncio
async def test_conversation_rows_receive_one_settled_episode_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """All persisted user fragments can be linked to one settled episode."""

    collection = SimpleNamespace()
    collection.update_many = AsyncMock(
        return_value=SimpleNamespace(matched_count=2),
    )
    database = SimpleNamespace(conversation_history=collection)
    monkeypatch.setattr(
        conversation_module,
        "get_db",
        AsyncMock(return_value=database),
    )

    matched = await conversation_module.set_conversation_source_episode_id(
        row_ids=["row-1", "row-2", "row-1"],
        source_episode_id="episode-settled-1",
    )

    assert matched == 2
    collection.update_many.assert_awaited_once_with(
        {
            "$or": [{
                "conversation_row_id": {"$in": ["row-1", "row-2"]},
            }],
        },
        {"$set": {"source_episode_id": "episode-settled-1"}},
    )
