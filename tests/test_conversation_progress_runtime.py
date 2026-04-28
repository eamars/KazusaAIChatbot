"""Facade/runtime tests for conversation progress."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from kazusa_ai_chatbot.conversation_progress import runtime
from kazusa_ai_chatbot.conversation_progress.models import ConversationProgressScope


@pytest.mark.asyncio
async def test_runtime_load_projects_empty_state(monkeypatch) -> None:
    """Missing DB/cache state returns the empty prompt projection."""

    monkeypatch.setattr(runtime.repository, "load_episode_state", AsyncMock(return_value=None))
    monkeypatch.setattr(
        runtime.cache,
        "select_latest_document",
        lambda *, scope, db_document: (db_document, False),
    )
    progress_runtime = runtime.ConversationProgressRuntime(
        recorder_callable=AsyncMock(return_value={}),
    )

    result = await progress_runtime.load_progress_context(
        scope=ConversationProgressScope("qq", "channel-1", "user-1"),
        current_timestamp="2026-04-28T04:00:00+00:00",
    )

    assert result["episode_state"] is None
    assert result["source"] == "empty"
    assert result["conversation_progress"]["continuity"] == "sharp_transition"
    assert result["conversation_progress"]["turn_count"] == 0


@pytest.mark.asyncio
async def test_runtime_record_writes_and_updates_cache(monkeypatch) -> None:
    """Recorder output is persisted and cached through the runtime facade."""

    recorder_callable = AsyncMock(return_value={
        "status": "active",
        "episode_label": "essay_help",
        "continuity": "same_episode",
        "user_state_updates": ["user still needs the third point"],
        "assistant_moves": ["reassurance"],
        "overused_moves": [],
        "open_loops": ["third point missing"],
        "progression_guidance": "address the missing third point",
    })
    monkeypatch.setattr(runtime.repository, "upsert_episode_state_guarded", AsyncMock(return_value=True))
    store_completed_document = MagicMock()
    monkeypatch.setattr(runtime.cache, "store_completed_document", store_completed_document)

    progress_runtime = runtime.ConversationProgressRuntime(recorder_callable=recorder_callable)
    scope = ConversationProgressScope("qq", "channel-1", "user-1")
    result = await progress_runtime.record_turn_progress(
        record_input={
            "scope": scope,
            "timestamp": "2026-04-28T04:00:00+00:00",
            "prior_episode_state": None,
            "decontexualized_input": "what is the third point?",
            "chat_history_recent": [],
            "content_anchors": ["[DECISION] answer"],
            "logical_stance": "CONFIRM",
            "character_intent": "PROVIDE",
            "final_dialog": ["The third point is about scope."],
        },
    )

    assert result == {
        "written": True,
        "turn_count": 1,
        "continuity": "same_episode",
        "status": "active",
        "cache_updated": True,
    }
    recorder_callable.assert_awaited_once()
    runtime.repository.upsert_episode_state_guarded.assert_awaited_once()
    store_completed_document.assert_called_once()


@pytest.mark.asyncio
async def test_runtime_record_does_not_cache_stale_write(monkeypatch) -> None:
    """A rejected guarded write does not refresh the process-local cache."""

    recorder_callable = AsyncMock(return_value={
        "status": "active",
        "episode_label": "essay_help",
        "continuity": "same_episode",
        "user_state_updates": [],
        "assistant_moves": [],
        "overused_moves": [],
        "open_loops": [],
        "progression_guidance": "",
    })
    monkeypatch.setattr(runtime.repository, "upsert_episode_state_guarded", AsyncMock(return_value=False))
    store_completed_document = MagicMock()
    monkeypatch.setattr(runtime.cache, "store_completed_document", store_completed_document)

    progress_runtime = runtime.ConversationProgressRuntime(recorder_callable=recorder_callable)
    result = await progress_runtime.record_turn_progress(
        record_input={
            "scope": ConversationProgressScope("qq", "channel-1", "user-1"),
            "timestamp": "2026-04-28T04:00:00+00:00",
            "prior_episode_state": None,
            "decontexualized_input": "hello",
            "chat_history_recent": [],
            "content_anchors": [],
            "logical_stance": "CONFIRM",
            "character_intent": "PROVIDE",
            "final_dialog": ["hello"],
        },
    )

    assert result["written"] is False
    assert result["cache_updated"] is False
    store_completed_document.assert_not_called()
