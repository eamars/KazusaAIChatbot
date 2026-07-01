"""Tests for DB helpers — mocked unit tests + live MongoDB integration tests."""

from __future__ import annotations

import asyncio
import logging
import re
from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import kazusa_ai_chatbot.db as db_module
import kazusa_ai_chatbot.db._client as db_client_module
import kazusa_ai_chatbot.db.bootstrap as db_bootstrap_module
import kazusa_ai_chatbot.db.character as db_character_module
import kazusa_ai_chatbot.db.conversation as db_conversation_module
import kazusa_ai_chatbot.db.global_character_growth as db_global_growth_module
import kazusa_ai_chatbot.db.memory as db_memory_module
import kazusa_ai_chatbot.db.self_cognition as db_self_cognition_module
import kazusa_ai_chatbot.db.script_operations as db_script_operations_module
import kazusa_ai_chatbot.db.users as db_users_module
from kazusa_ai_chatbot.db import (
    AFFINITY_DEFAULT,
    RUNTIME_CHARACTER_STATE_FIELDS,
    build_memory_doc,
    close_db,
    compose_character_profile,
    get_affinity,
    get_character_profile,
    get_character_runtime_state,
    get_character_state,
    get_conversation_history,
    get_user_profile,
    save_character_profile,
    save_conversation,
    save_memory,
    search_memory,
    split_character_profile_runtime_state,
    update_affinity,
    update_last_relationship_insight,
    upsert_character_state,
)
from kazusa_ai_chatbot.db._client import get_db

# Mark for tests that require a running MongoDB instance.
# Run with:  pytest -m live_db -v
live_db = pytest.mark.live_db


def _mock_db():
    """Create a mock database object with collection access."""
    db = MagicMock()
    return db


class _BootstrapCollection:
    """Small async collection fake for bootstrap index tests."""

    def __init__(self) -> None:
        """Create a collection fake that records indexes."""

        self.indexes: list[dict] = []
        self.find_one = AsyncMock(return_value={"_id": "global"})
        self.insert_one = AsyncMock()

    async def create_index(self, keys, **kwargs) -> None:
        """Record one requested index."""

        self.indexes.append({"keys": keys, "kwargs": kwargs})


class _BootstrapDb:
    """Small DB fake containing the collections touched by bootstrap."""

    def __init__(self) -> None:
        """Create all collections expected by db_bootstrap."""

        collection_names = [
            "conversation_history",
            "user_profiles",
            "character_state",
            "memory",
            "user_memory_units",
            "scheduled_events",
            "calendar_schedules",
            "calendar_runs",
            "self_cognition_action_attempts",
            "self_cognition_group_review_windows",
            "conversation_episode_state",
            "character_reflection_runs",
            "interaction_style_images",
            "global_character_growth_traits",
            "global_character_growth_runs",
            "rag_cache2_persistent",
            "event_log_events",
            "event_log_snapshots",
        ]
        self.collections = {
            name: _BootstrapCollection()
            for name in collection_names
        }
        for name, collection in self.collections.items():
            setattr(self, name, collection)

    async def list_collection_names(self) -> list[str]:
        """Return all expected collections so bootstrap skips creation."""

        return_value = list(self.collections)
        return return_value

    async def create_collection(self, name: str) -> None:
        """Create a collection in the fake DB."""

        collection = _BootstrapCollection()
        self.collections[name] = collection
        setattr(self, name, collection)

    def __getitem__(self, name: str) -> _BootstrapCollection:
        """Return a collection by Mongo-style item access."""

        return_value = self.collections[name]
        return return_value


@contextmanager
def _patched_get_db(db):
    """Patch every DB submodule binding used by re-exported helper functions."""
    mock_get_db = AsyncMock(return_value=db)
    with patch.object(db_client_module, "get_db", mock_get_db), \
         patch.object(db_character_module, "get_db", mock_get_db), \
         patch.object(db_conversation_module, "get_db", mock_get_db), \
         patch.object(db_memory_module, "get_db", mock_get_db), \
         patch.object(db_self_cognition_module, "get_db", mock_get_db), \
         patch.object(db_users_module, "get_db", mock_get_db):
        yield mock_get_db


@contextmanager
def _patched_embedding(return_value=None, mock=None):
    """Patch document embedding bindings used by DB helpers under test."""
    mock_embed = mock or AsyncMock(return_value=return_value)
    with patch.object(db_module, "get_text_embedding", mock_embed), \
         patch.object(db_module, "get_document_text_embedding", mock_embed), \
         patch.object(db_client_module, "get_text_embedding", mock_embed), \
         patch.object(db_client_module, "get_document_text_embedding", mock_embed), \
         patch.object(db_conversation_module, "get_document_text_embedding", mock_embed):
        yield mock_embed


@contextmanager
def _patched_query_embedding(return_value=None, mock=None):
    """Patch every query embedding binding used by vector search helpers."""
    mock_embed = mock or AsyncMock(return_value=return_value)
    with patch.object(db_module, "get_query_text_embedding", mock_embed), \
         patch.object(db_client_module, "get_query_text_embedding", mock_embed), \
         patch.object(db_conversation_module, "get_query_text_embedding", mock_embed), \
         patch.object(db_memory_module, "get_query_text_embedding", mock_embed):
        yield mock_embed


def _group_review_window_doc(
    *,
    source_id: str,
    status: str,
    case_id: str | None = None,
    skip_reason: str | None = None,
) -> dict:
    """Build a minimal group-review window ledger document for DB tests."""

    doc = {
        "source_id": source_id,
        "case_id": case_id,
        "scope_ref": "scope_group",
        "platform": "qq",
        "platform_channel_id": "group-1",
        "channel_type": "group",
        "window_start": "2026-05-18T04:00:00+00:00",
        "window_end": "2026-05-18T04:15:00+00:00",
        "status": status,
        "reviewed_at": "2026-05-18T04:20:00+00:00",
        "selected_route": None,
        "dispatch_status": None,
        "skip_reason": skip_reason,
    }
    return doc


@pytest.mark.asyncio
async def test_get_conversation_history():
    mock_docs = [
        {"role": "assistant", "platform_user_id": "bot_001", "display_name": "bot", "body_text": "Hello", "timestamp": "t2"},
        {"role": "user", "platform_user_id": "user_001", "display_name": "User", "body_text": "Hi", "timestamp": "t1"},
    ]
    db = _mock_db()
    cursor = AsyncMock()
    cursor.to_list = AsyncMock(return_value=mock_docs)
    db.conversation_history.find.return_value.sort.return_value.limit.return_value = cursor

    with _patched_get_db(db):
        result = await get_conversation_history(platform="discord", platform_channel_id="chan_1", limit=10)
        
        # Verify basic find parameters
        db.conversation_history.find.assert_called_with({"platform": "discord", "platform_channel_id": "chan_1"})

        # Verify sorting and limit
        db.conversation_history.find.return_value.sort.assert_called_with("timestamp", -1)
        db.conversation_history.find.return_value.sort.return_value.limit.assert_called_with(10)

    # Should be reversed (oldest first)
    assert result[0]["body_text"] == "Hi"
    assert result[1]["body_text"] == "Hello"


@pytest.mark.asyncio
async def test_get_conversation_history_filters():
    db = _mock_db()
    cursor = AsyncMock()
    cursor.to_list = AsyncMock(return_value=[])
    db.conversation_history.find.return_value.sort.return_value.limit.return_value = cursor

    with _patched_get_db(db):
        # test global_user_id filter
        await get_conversation_history(platform="discord", platform_channel_id="chan_1", global_user_id="user_123")
        db.conversation_history.find.assert_called_with({"platform": "discord", "platform_channel_id": "chan_1", "global_user_id": "user_123"})
        
        # test display_name filter (should be used when global_user_id is missing)
        await get_conversation_history(platform="discord", platform_channel_id="chan_1", display_name="Kira")
        db.conversation_history.find.assert_called_with({"platform": "discord", "platform_channel_id": "chan_1", "display_name": "Kira"})
        
        # test priority: global_user_id should be prioritized over display_name
        await get_conversation_history(platform="discord", platform_channel_id="chan_1", global_user_id="user_123", display_name="Kira")
        db.conversation_history.find.assert_called_with({"platform": "discord", "platform_channel_id": "chan_1", "global_user_id": "user_123"})
        
        # test timestamp filter
        await get_conversation_history(
            platform="discord",
            platform_channel_id="chan_1",
            from_timestamp="2025-01-01T00:00:00Z", 
            to_timestamp="2025-01-02T00:00:00Z"
        )
        db.conversation_history.find.assert_called_with({
            "platform": "discord",
            "platform_channel_id": "chan_1",
            "timestamp": {
                "$gte": "2025-01-01T00:00:00Z",
                "$lte": "2025-01-02T00:00:00Z"
            }
        })
        
        # test all combined
        await get_conversation_history(
            platform="discord",
            platform_channel_id="chan_1",
            display_name="Kira", 
            from_timestamp="2025-01-01T00:00:00Z"
        )
        db.conversation_history.find.assert_called_with({
            "platform": "discord",
            "platform_channel_id": "chan_1",
            "display_name": "Kira",
            "timestamp": {
                "$gte": "2025-01-01T00:00:00Z"
            }
        })
        
        # test without platform_channel_id
        await get_conversation_history(
            global_user_id="user_123"
        )
        db.conversation_history.find.assert_called_with({
            "global_user_id": "user_123"
        })


@pytest.mark.asyncio
async def test_get_character_state_found():
    db = _mock_db()
    db.character_state.find_one = AsyncMock(return_value={
        "_id": "global",
        "mood": "calm",
        "emotional_tone": "warm",
        "recent_events": [],
        "updated_at": "t1",
    })

    with _patched_get_db(db):
        result = await get_character_state()

    assert result["mood"] == "calm"
    assert "_id" not in result


@pytest.mark.asyncio
async def test_get_character_state_not_found():
    db = _mock_db()
    db.character_state.find_one = AsyncMock(return_value=None)

    with _patched_get_db(db):
        result = await get_character_state()

    assert result == {}


@pytest.mark.asyncio
async def test_upsert_character_state():
    db = _mock_db()
    db.character_state.find_one = AsyncMock(return_value={
        "_id": "global",
        "mood": "old",
        "global_vibe": "old_vibe",
        "reflection_summary": "old_summary",
    })
    db.character_state.update_one = AsyncMock()

    with _patched_get_db(db):
        await upsert_character_state("happy", "relaxed", "feeling good", "t2")

    call_args = db.character_state.update_one.call_args
    set_payload = call_args[0][1]["$set"]
    assert set_payload["mood"] == "happy"
    assert set_payload["global_vibe"] == "relaxed"
    assert set_payload["reflection_summary"] == "feeling good"


# ── User profile ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_user_profile_found():
    db = _mock_db()
    db.user_profiles.find_one = AsyncMock(return_value={
        "_id": "abc",
        "global_user_id": "u1",
        "facts": ["fact1"],
        "affinity": 600,
        "last_relationship_insight": "friendly",
    })

    with _patched_get_db(db):
        result = await get_user_profile("u1")

    assert result["global_user_id"] == "u1"
    assert result["affinity"] == 600
    assert "_id" not in result


@pytest.mark.asyncio
async def test_get_user_profile_not_found():
    db = _mock_db()
    db.user_profiles.find_one = AsyncMock(return_value=None)

    with _patched_get_db(db):
        result = await get_user_profile("unknown")

    assert result == {}


# ── Relationship insight ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_update_last_relationship_insight():
    db = _mock_db()
    db.user_profiles.update_one = AsyncMock()

    with _patched_get_db(db):
        await update_last_relationship_insight("u1", "very friendly")

    db.user_profiles.update_one.assert_called_once_with(
        {"global_user_id": "u1"},
        {"$set": {"last_relationship_insight": "very friendly"}},
        upsert=True,
    )


# ── Save conversation ──────────────────────────────────────────────


def _conversation_save_doc() -> dict:
    """Build a valid conversation row for save-conversation tests."""

    doc = {
        "platform": "qq",
        "platform_channel_id": "c1",
        "channel_type": "group",
        "role": "user",
        "platform_message_id": "m1",
        "platform_user_id": "u1",
        "global_user_id": "global-u1",
        "display_name": "Alice",
        "body_text": "Hello!",
        "raw_wire_text": "Hello!",
        "content_type": "text",
        "addressed_to_global_user_ids": ["character-global"],
        "mentions": [],
        "broadcast": False,
        "attachments": [],
        "timestamp": "t1",
        "embedding": [0.1, 0.2],
    }
    return doc


@pytest.mark.asyncio
async def test_save_conversation_generates_embedding():
    """save_conversation should generate an embedding when not provided."""
    db = _mock_db()
    insert_result = MagicMock()
    insert_result.inserted_id = "row-1"
    db.conversation_history.insert_one = AsyncMock(return_value=insert_result)

    doc = {
        "platform": "qq",
        "platform_channel_id": "c1",
        "channel_type": "group",
        "role": "user",
        "platform_message_id": "m1",
        "platform_user_id": "u1",
        "global_user_id": "global-u1",
        "display_name": "Alice",
        "body_text": "Hello!",
        "raw_wire_text": "Hello!",
        "content_type": "text",
        "addressed_to_global_user_ids": ["character-global"],
        "mentions": [],
        "broadcast": False,
        "attachments": [],
        "timestamp": "t1",
    }

    with _patched_get_db(db), _patched_embedding(return_value=[0.1, 0.2]):
        await save_conversation(doc)

    assert doc["embedding"] == [0.1, 0.2]
    db.conversation_history.insert_one.assert_called_once_with(doc)


@pytest.mark.asyncio
async def test_save_conversation_preserves_existing_embedding():
    """save_conversation should not regenerate embedding if already present."""
    db = _mock_db()
    insert_result = MagicMock()
    insert_result.inserted_id = "row-1"
    db.conversation_history.insert_one = AsyncMock(return_value=insert_result)

    doc = {
        "platform": "qq",
        "platform_channel_id": "c1",
        "channel_type": "group",
        "role": "user",
        "platform_message_id": "m1",
        "platform_user_id": "u1",
        "global_user_id": "global-u1",
        "display_name": "Alice",
        "body_text": "Hello!",
        "raw_wire_text": "Hello!",
        "content_type": "text",
        "addressed_to_global_user_ids": ["character-global"],
        "mentions": [],
        "broadcast": False,
        "attachments": [],
        "timestamp": "t1",
        "embedding": [0.5, 0.6],
    }

    with _patched_get_db(db), _patched_embedding(mock=AsyncMock()) as mock_embed:
        await save_conversation(doc)

    mock_embed.assert_not_called()
    assert doc["embedding"] == [0.5, 0.6]


@pytest.mark.asyncio
async def test_save_conversation_returns_row_id_after_cache_invalidation(
    monkeypatch,
) -> None:
    """save_conversation should return Mongo row identity after invalidation."""
    db = _mock_db()
    call_order = []
    insert_result = MagicMock()
    insert_result.inserted_id = "row-after-invalidation"

    async def _insert_one(doc):
        call_order.append("insert")
        return_value = insert_result
        return return_value

    async def _invalidate(event):
        call_order.append("invalidate")
        return_value = 0
        return return_value

    runtime = MagicMock()
    runtime.invalidate = AsyncMock(side_effect=_invalidate)
    db.conversation_history.insert_one = AsyncMock(side_effect=_insert_one)
    monkeypatch.setattr(
        "kazusa_ai_chatbot.rag.cache2_runtime.get_rag_cache2_runtime",
        MagicMock(return_value=runtime),
    )

    with _patched_get_db(db):
        row_id = await save_conversation(_conversation_save_doc())

    assert row_id == "row-after-invalidation"
    assert call_order == ["insert", "invalidate"]


@pytest.mark.asyncio
async def test_save_conversation_returns_row_id_when_cache_invalidation_fails(
    monkeypatch,
    caplog,
) -> None:
    """Cache invalidation failure should not discard a committed row ID."""
    db = _mock_db()
    insert_result = MagicMock()
    insert_result.inserted_id = "row-with-cache-warning"
    db.conversation_history.insert_one = AsyncMock(return_value=insert_result)
    runtime = MagicMock()
    runtime.invalidate = AsyncMock(side_effect=RuntimeError("cache offline"))
    monkeypatch.setattr(
        "kazusa_ai_chatbot.rag.cache2_runtime.get_rag_cache2_runtime",
        MagicMock(return_value=runtime),
    )

    with _patched_get_db(db), caplog.at_level(
        logging.WARNING,
        logger="kazusa_ai_chatbot.db.conversation",
    ):
        row_id = await save_conversation(_conversation_save_doc())

    assert row_id == "row-with-cache-warning"
    assert "Cache2 invalidation after save_conversation failed: cache offline" in (
        caplog.text
    )


@pytest.mark.asyncio
async def test_save_conversation_insert_failure_still_propagates(
    monkeypatch,
) -> None:
    """Insert failures should stay exceptional instead of returning row IDs."""
    db = _mock_db()
    db.conversation_history.insert_one = AsyncMock(
        side_effect=RuntimeError("insert failed")
    )
    runtime = MagicMock()
    runtime.invalidate = AsyncMock()
    monkeypatch.setattr(
        "kazusa_ai_chatbot.rag.cache2_runtime.get_rag_cache2_runtime",
        MagicMock(return_value=runtime),
    )

    with _patched_get_db(db), pytest.raises(RuntimeError, match="insert failed"):
        await save_conversation(_conversation_save_doc())

    runtime.invalidate.assert_not_awaited()


@pytest.mark.asyncio
async def test_apply_assistant_delivery_receipt_updates_tracking_row() -> None:
    """Delivery receipts should update one logical assistant row."""
    db = _mock_db()
    update_result = MagicMock()
    update_result.matched_count = 1
    update_result.modified_count = 1
    db.conversation_history.update_one = AsyncMock(
        return_value=update_result,
    )

    with _patched_get_db(db):
        updated = await db_conversation_module.apply_assistant_delivery_receipt(
            platform="qq",
            platform_channel_id="chan-1",
            delivery_tracking_id="delivery-1",
            logical_message_index=1,
            platform_message_id="platform-123",
            delivered_at="2026-05-07T11:00:00+00:00",
            adapter="napcat",
        )

    assert updated is True
    db.conversation_history.update_one.assert_awaited_once_with(
        {
            "platform": "qq",
            "platform_channel_id": "chan-1",
            "role": "assistant",
            "delivery_tracking_id": "delivery-1",
            "logical_message_index": 1,
        },
        {
            "$set": {
                "platform_message_id": "platform-123",
                "delivery_status": "delivered",
                "delivered_at": "2026-05-07T11:00:00+00:00",
                "delivery_adapter": "napcat",
            }
        },
    )


@pytest.mark.asyncio
async def test_apply_assistant_delivery_receipt_allows_empty_channel_scope() -> None:
    """Empty channel ids should match by platform and tracking id only."""
    db = _mock_db()
    update_result = MagicMock()
    update_result.matched_count = 1
    update_result.modified_count = 1
    db.conversation_history.update_one = AsyncMock(
        return_value=update_result,
    )

    with _patched_get_db(db):
        updated = await db_conversation_module.apply_assistant_delivery_receipt(
            platform="debug",
            platform_channel_id="",
            delivery_tracking_id="delivery-1",
            logical_message_index=0,
            platform_message_id="platform-123",
            delivered_at="2026-05-07T11:00:00+00:00",
            adapter="debug",
        )

    assert updated is True
    update_query = db.conversation_history.update_one.await_args.args[0]
    assert update_query == {
        "platform": "debug",
        "role": "assistant",
        "delivery_tracking_id": "delivery-1",
        "logical_message_index": 0,
    }


@pytest.mark.asyncio
async def test_apply_assistant_delivery_receipt_uses_matched_count() -> None:
    """Idempotent re-receipts should still count as updated rows."""
    db = _mock_db()
    update_result = MagicMock()
    update_result.matched_count = 1
    update_result.modified_count = 0
    db.conversation_history.update_one = AsyncMock(
        return_value=update_result,
    )

    with _patched_get_db(db):
        updated = await db_conversation_module.apply_assistant_delivery_receipt(
            platform="qq",
            platform_channel_id="chan-1",
            delivery_tracking_id="delivery-1",
            logical_message_index=0,
            platform_message_id="platform-123",
            delivered_at="2026-05-07T11:00:00+00:00",
            adapter="napcat",
        )

    assert updated is True


@pytest.mark.asyncio
async def test_apply_assistant_delivery_receipt_has_no_embedding_or_cache_side_effects(
    monkeypatch,
) -> None:
    """Receipt metadata updates must not touch embeddings or Cache2."""
    db = _mock_db()
    update_result = MagicMock()
    update_result.matched_count = 0
    update_result.modified_count = 0
    db.conversation_history.update_one = AsyncMock(
        return_value=update_result,
    )
    runtime = MagicMock()
    runtime.invalidate = AsyncMock()

    monkeypatch.setattr(
        "kazusa_ai_chatbot.rag.cache2_runtime.get_rag_cache2_runtime",
        MagicMock(return_value=runtime),
    )
    with _patched_get_db(db), _patched_embedding(mock=AsyncMock()) as mock_embed:
        updated = await db_conversation_module.apply_assistant_delivery_receipt(
            platform="qq",
            platform_channel_id="chan-1",
            delivery_tracking_id="delivery-1",
            logical_message_index=0,
            platform_message_id="platform-123",
            delivered_at="2026-05-07T11:00:00+00:00",
            adapter="napcat",
        )

    assert updated is False
    mock_embed.assert_not_awaited()
    runtime.invalidate.assert_not_awaited()


@pytest.mark.asyncio
async def test_get_conversation_by_platform_message_id_uses_exact_scope() -> None:
    """Reply fallback lookup should use exact platform/channel/message id."""
    expected = {
        "platform": "qq",
        "platform_channel_id": "chan-1",
        "platform_message_id": "platform-123",
        "role": "assistant",
        "body_text": "prior bot text",
    }
    db = _mock_db()
    db.conversation_history.find_one = AsyncMock(return_value=expected)

    with _patched_get_db(db):
        result = await db_conversation_module.get_conversation_by_platform_message_id(
            platform="qq",
            platform_channel_id="chan-1",
            platform_message_id="platform-123",
        )

    assert result == expected
    db.conversation_history.find_one.assert_awaited_once_with({
        "platform": "qq",
        "platform_channel_id": "chan-1",
        "platform_message_id": "platform-123",
    })


@pytest.mark.asyncio
async def test_upsert_self_cognition_action_attempt_uses_idempotency_key() -> None:
    """Self-cognition attempts should be stored by stable action identity."""

    attempt = {
        "attempt_id": "self_cognition_attempt:abc",
        "run_id": "self_cognition_run:abc",
        "trigger_id": "self_cognition_trigger:abc",
        "source_kind": "user_memory_unit",
        "source_id": "promise-001",
        "target_scope": {
            "platform": "qq",
            "platform_channel_id": "673225019",
            "channel_type": "private",
            "user_id": "user-001",
        },
        "action_kind": "send_message",
        "due_at": "2026-05-13T00:00:00+00:00",
        "idempotency_key": "sha256:abc",
        "status": "scheduled",
        "recorded_at": "2026-05-13T00:01:00+00:00",
    }
    db = _mock_db()
    db.self_cognition_action_attempts.replace_one = AsyncMock()

    with _patched_get_db(db):
        await db_self_cognition_module.upsert_self_cognition_action_attempt(attempt)

    db.self_cognition_action_attempts.replace_one.assert_awaited_once_with(
        {"idempotency_key": "sha256:abc"},
        attempt,
        upsert=True,
    )


@pytest.mark.asyncio
async def test_list_self_cognition_action_attempts_returns_recent_rows() -> None:
    """Self-cognition attempt reads should stay behind the DB facade."""

    rows = [{"idempotency_key": "sha256:abc", "status": "scheduled"}]
    cursor = MagicMock()
    cursor.to_list = AsyncMock(return_value=rows)
    db = _mock_db()
    db.self_cognition_action_attempts.find.return_value.sort.return_value.limit.return_value = (
        cursor
    )

    with _patched_get_db(db):
        result = await db_self_cognition_module.list_self_cognition_action_attempts(
            limit=25,
        )

    assert result == rows
    db.self_cognition_action_attempts.find.assert_called_once_with({}, {"_id": 0})
    db.self_cognition_action_attempts.find.return_value.sort.assert_called_once_with(
        "recorded_at",
        -1,
    )
    db.self_cognition_action_attempts.find.return_value.sort.return_value.limit.assert_called_once_with(
        25,
    )
    cursor.to_list.assert_awaited_once_with(length=25)


@pytest.mark.asyncio
async def test_upsert_group_review_window_inserts_by_source_id() -> None:
    """Group review window terminal rows should be keyed by source identity."""

    window_doc = _group_review_window_doc(
        source_id="scope_group:window-1",
        status="reviewed",
        case_id="case-1",
    )
    db = _mock_db()
    db.self_cognition_group_review_windows.find_one = AsyncMock(return_value=None)
    db.self_cognition_group_review_windows.insert_one = AsyncMock()

    with _patched_get_db(db):
        result = await (
            db_self_cognition_module.upsert_self_cognition_group_review_window(
                window_doc,
            )
        )

    assert result == window_doc
    db.self_cognition_group_review_windows.find_one.assert_awaited_once_with(
        {"source_id": "scope_group:window-1"},
        {"_id": 0},
    )
    db.self_cognition_group_review_windows.insert_one.assert_awaited_once_with(
        window_doc,
    )


@pytest.mark.asyncio
async def test_upsert_group_review_window_returns_existing_terminal_row() -> None:
    """Existing terminal ledger rows must not be overwritten."""

    existing_doc = _group_review_window_doc(
        source_id="scope_group:window-1",
        status="coalesced_skipped",
        skip_reason="older unreviewed window coalesced",
    )
    later_doc = _group_review_window_doc(
        source_id="scope_group:window-1",
        status="reviewed",
        case_id="case-later",
    )
    db = _mock_db()
    db.self_cognition_group_review_windows.find_one = AsyncMock(
        return_value=existing_doc,
    )
    db.self_cognition_group_review_windows.insert_one = AsyncMock()
    db.self_cognition_group_review_windows.replace_one = AsyncMock()

    with _patched_get_db(db):
        result = await (
            db_self_cognition_module.upsert_self_cognition_group_review_window(
                later_doc,
            )
        )

    assert result == existing_doc
    db.self_cognition_group_review_windows.insert_one.assert_not_awaited()
    db.self_cognition_group_review_windows.replace_one.assert_not_awaited()


@pytest.mark.asyncio
async def test_find_group_review_window_reads_by_source_id() -> None:
    """Reviewed-window lookups should use the durable source identity."""

    expected_doc = _group_review_window_doc(
        source_id="scope_group:window-1",
        status="reviewed",
        case_id="case-1",
    )
    db = _mock_db()
    db.self_cognition_group_review_windows.find_one = AsyncMock(
        return_value=expected_doc,
    )

    with _patched_get_db(db):
        result = await (
            db_self_cognition_module.find_self_cognition_group_review_window(
                source_id="scope_group:window-1",
            )
        )

    assert result == expected_doc
    db.self_cognition_group_review_windows.find_one.assert_awaited_once_with(
        {"source_id": "scope_group:window-1"},
        {"_id": 0},
    )


@pytest.mark.asyncio
async def test_group_review_skipped_rows_require_skip_reason() -> None:
    """Skipped terminal statuses should carry no case or dispatch metadata."""

    db = _mock_db()
    db.self_cognition_group_review_windows.find_one = AsyncMock(return_value=None)
    db.self_cognition_group_review_windows.insert_one = AsyncMock()
    valid_doc = _group_review_window_doc(
        source_id="scope_group:coalesced",
        status="coalesced_skipped",
        skip_reason="older unreviewed window coalesced",
    )

    with _patched_get_db(db):
        await db_self_cognition_module.upsert_self_cognition_group_review_window(
            valid_doc,
        )

    invalid_doc = _group_review_window_doc(
        source_id="scope_group:invalid",
        status="stale_skipped",
    )
    with pytest.raises(ValueError, match="skip_reason"):
        await db_self_cognition_module.upsert_self_cognition_group_review_window(
            invalid_doc,
        )

    invalid_case_doc = _group_review_window_doc(
        source_id="scope_group:invalid-case",
        status="coalesced_skipped",
        case_id="case-should-not-exist",
        skip_reason="older unreviewed window coalesced",
    )
    with pytest.raises(ValueError, match="case_id"):
        await db_self_cognition_module.upsert_self_cognition_group_review_window(
            invalid_case_doc,
        )


@pytest.mark.asyncio
async def test_group_review_reviewed_and_failed_rows_validate_fields() -> None:
    """Reviewed and failed terminal rows should enforce status-specific fields."""

    reviewed_doc = _group_review_window_doc(
        source_id="scope_group:reviewed",
        status="reviewed",
        case_id="case-1",
    )
    target_failed_doc = _group_review_window_doc(
        source_id="scope_group:target-failed",
        status="target_binding_failed",
        case_id="case-2",
        skip_reason="delivery target missing",
    )
    review_failed_doc = _group_review_window_doc(
        source_id="scope_group:review-failed",
        status="review_failed",
        case_id="case-3",
        skip_reason="self-cognition worker failed",
    )
    db = _mock_db()
    db.self_cognition_group_review_windows.find_one = AsyncMock(return_value=None)
    db.self_cognition_group_review_windows.insert_one = AsyncMock()

    with _patched_get_db(db):
        await db_self_cognition_module.upsert_self_cognition_group_review_window(
            reviewed_doc,
        )
        await db_self_cognition_module.upsert_self_cognition_group_review_window(
            target_failed_doc,
        )
        await db_self_cognition_module.upsert_self_cognition_group_review_window(
            review_failed_doc,
        )

    missing_case_doc = _group_review_window_doc(
        source_id="scope_group:missing-case",
        status="reviewed",
    )
    with pytest.raises(ValueError, match="case_id"):
        await db_self_cognition_module.upsert_self_cognition_group_review_window(
            missing_case_doc,
        )

    reviewed_with_skip_reason = _group_review_window_doc(
        source_id="scope_group:reviewed-with-reason",
        status="reviewed",
        case_id="case-4",
        skip_reason="not allowed for reviewed rows",
    )
    with pytest.raises(ValueError, match="skip_reason"):
        await db_self_cognition_module.upsert_self_cognition_group_review_window(
            reviewed_with_skip_reason,
        )


@pytest.mark.asyncio
async def test_db_bootstrap_creates_platform_message_lookup_index(monkeypatch) -> None:
    """Bootstrap should index exact reply-target platform message lookups."""
    db = _BootstrapDb()
    monkeypatch.setattr(db_bootstrap_module, "get_db", AsyncMock(return_value=db))
    monkeypatch.setattr(db_bootstrap_module, "enable_vector_index", AsyncMock())
    monkeypatch.setattr(
        db_bootstrap_module,
        "ensure_reflection_run_indexes",
        AsyncMock(),
    )
    monkeypatch.setattr(
        db_bootstrap_module,
        "ensure_interaction_style_image_indexes",
        AsyncMock(),
    )
    monkeypatch.setattr(
        db_bootstrap_module,
        "ensure_global_character_growth_indexes",
        AsyncMock(),
    )
    monkeypatch.setattr(
        db_bootstrap_module,
        "ensure_event_log_indexes",
        AsyncMock(),
    )
    monkeypatch.setattr(
        db_bootstrap_module,
        "purge_stale_initializer_entries",
        AsyncMock(),
    )
    monkeypatch.setattr(
        db_bootstrap_module,
        "prune_persistent_entries",
        AsyncMock(),
    )

    await db_bootstrap_module.db_bootstrap()

    index_specs = db.conversation_history.indexes
    assert {
        "keys": [
            ("platform", 1),
            ("platform_channel_id", 1),
            ("platform_message_id", 1),
        ],
        "kwargs": {"name": "conv_platform_channel_message_id"},
    } in index_specs


@pytest.mark.asyncio
async def test_db_bootstrap_creates_calendar_collections_and_indexes(
    monkeypatch,
) -> None:
    """Bootstrap should prepare calendar schedules and due-run indexes."""

    db = _BootstrapDb()
    monkeypatch.setattr(db_bootstrap_module, "get_db", AsyncMock(return_value=db))
    monkeypatch.setattr(db_bootstrap_module, "enable_vector_index", AsyncMock())
    monkeypatch.setattr(
        db_bootstrap_module,
        "ensure_reflection_run_indexes",
        AsyncMock(),
    )
    monkeypatch.setattr(
        db_bootstrap_module,
        "ensure_interaction_style_image_indexes",
        AsyncMock(),
    )
    monkeypatch.setattr(
        db_bootstrap_module,
        "ensure_global_character_growth_indexes",
        AsyncMock(),
    )
    monkeypatch.setattr(
        db_bootstrap_module,
        "ensure_event_log_indexes",
        AsyncMock(),
    )
    monkeypatch.setattr(
        db_bootstrap_module,
        "ensure_internal_monologue_residue_indexes",
        AsyncMock(),
    )
    monkeypatch.setattr(
        db_bootstrap_module,
        "purge_stale_initializer_entries",
        AsyncMock(),
    )
    monkeypatch.setattr(
        db_bootstrap_module,
        "prune_persistent_entries",
        AsyncMock(),
    )

    await db_bootstrap_module.db_bootstrap()

    schedule_indexes = db.calendar_schedules.indexes
    assert {
        "keys": "idempotency_key",
        "kwargs": {
            "unique": True,
            "name": "calendar_schedule_idempotency_unique",
        },
    } in schedule_indexes

    run_indexes = db.calendar_runs.indexes
    assert {
        "keys": "idempotency_key",
        "kwargs": {
            "unique": True,
            "name": "calendar_run_idempotency_unique",
        },
    } in run_indexes
    assert {
        "keys": [("status", 1), ("due_at", 1), ("trigger_kind", 1)],
        "kwargs": {"name": "calendar_run_status_due_trigger"},
    } in run_indexes
    assert {
        "keys": [("trigger_kind", 1), ("period_start_utc", 1), ("run_id", 1)],
        "kwargs": {"name": "calendar_run_reflection_phase_period"},
    } in run_indexes


@pytest.mark.asyncio
async def test_db_bootstrap_creates_background_artifact_collection_and_indexes(
    monkeypatch,
) -> None:
    """Bootstrap should prepare durable background artifact job storage."""

    db = _BootstrapDb()
    ensure_background_artifact = AsyncMock()
    monkeypatch.setattr(db_bootstrap_module, "get_db", AsyncMock(return_value=db))
    monkeypatch.setattr(db_bootstrap_module, "enable_vector_index", AsyncMock())
    monkeypatch.setattr(
        db_bootstrap_module,
        "ensure_reflection_run_indexes",
        AsyncMock(),
    )
    monkeypatch.setattr(
        db_bootstrap_module,
        "ensure_interaction_style_image_indexes",
        AsyncMock(),
    )
    monkeypatch.setattr(
        db_bootstrap_module,
        "ensure_global_character_growth_indexes",
        AsyncMock(),
    )
    monkeypatch.setattr(
        db_bootstrap_module,
        "ensure_event_log_indexes",
        AsyncMock(),
    )
    monkeypatch.setattr(
        db_bootstrap_module,
        "ensure_internal_monologue_residue_indexes",
        AsyncMock(),
    )
    monkeypatch.setattr(
        db_bootstrap_module,
        "ensure_background_artifact_job_indexes",
        ensure_background_artifact,
    )
    monkeypatch.setattr(
        db_bootstrap_module,
        "purge_stale_initializer_entries",
        AsyncMock(),
    )
    monkeypatch.setattr(
        db_bootstrap_module,
        "prune_persistent_entries",
        AsyncMock(),
    )

    await db_bootstrap_module.db_bootstrap()

    assert "background_artifact_jobs" in db.collections
    ensure_background_artifact.assert_awaited_once()


def test_db_facade_exports_calendar_schema_docs() -> None:
    """Calendar schedule and run schemas should be public facade exports."""

    assert db_module.CalendarScheduleDoc.__name__ == "CalendarScheduleDoc"
    assert db_module.CalendarRunDoc.__name__ == "CalendarRunDoc"
    assert "CalendarScheduleDoc" in db_module.__all__
    assert "CalendarRunDoc" in db_module.__all__


@pytest.mark.asyncio
async def test_script_operations_loads_calendar_migration_events(
    monkeypatch,
) -> None:
    """Calendar migration scripts should read legacy rows via maintenance DB."""

    cursor = MagicMock()
    cursor.to_list = AsyncMock(return_value=[{"event_id": "legacy-1"}])
    db = _mock_db()
    db.scheduled_events.find.return_value = cursor
    monkeypatch.setattr(
        db_script_operations_module,
        "get_db",
        AsyncMock(return_value=db),
    )

    rows = (
        await db_script_operations_module
        .list_scheduled_events_for_calendar_migration()
    )

    assert rows == [{"event_id": "legacy-1"}]
    db.scheduled_events.find.assert_called_once_with({})
    cursor.to_list.assert_awaited_once_with(length=None)


@pytest.mark.asyncio
async def test_script_operations_mutates_calendar_migration_legacy_status(
    monkeypatch,
) -> None:
    """Calendar migration legacy writes should expose boolean match status."""

    db = _mock_db()
    db.scheduled_events.update_one = AsyncMock(
        side_effect=[
            MagicMock(matched_count=1),
            MagicMock(matched_count=0),
        ],
    )
    monkeypatch.setattr(
        db_script_operations_module,
        "get_db",
        AsyncMock(return_value=db),
    )

    cancelled = await (
        db_script_operations_module
        .cancel_pending_send_message_for_calendar_migration("send-1")
    )
    migrated = await (
        db_script_operations_module
        .mark_pending_future_cognition_migrated_for_calendar_migration(
            "future-1",
        )
    )

    assert cancelled is True
    assert migrated is False
    cancel_call = db.scheduled_events.update_one.await_args_list[0]
    migrated_call = db.scheduled_events.update_one.await_args_list[1]
    assert cancel_call.args[0] == {
        "event_id": "send-1",
        "status": "pending",
        "tool": "send_message",
    }
    assert cancel_call.args[1] == {"$set": {"status": "cancelled"}}
    assert migrated_call.args[0] == {
        "event_id": "future-1",
        "status": "pending",
        "tool": "trigger_future_cognition",
    }
    assert migrated_call.args[1] == {"$set": {"status": "migrated"}}


@pytest.mark.asyncio
async def test_db_bootstrap_creates_self_cognition_attempt_indexes(
    monkeypatch,
) -> None:
    """Bootstrap should index self-cognition attempt state."""

    db = _BootstrapDb()
    monkeypatch.setattr(db_bootstrap_module, "get_db", AsyncMock(return_value=db))
    monkeypatch.setattr(db_bootstrap_module, "enable_vector_index", AsyncMock())
    monkeypatch.setattr(
        db_bootstrap_module,
        "ensure_reflection_run_indexes",
        AsyncMock(),
    )
    monkeypatch.setattr(
        db_bootstrap_module,
        "ensure_interaction_style_image_indexes",
        AsyncMock(),
    )
    monkeypatch.setattr(
        db_bootstrap_module,
        "ensure_global_character_growth_indexes",
        AsyncMock(),
    )
    monkeypatch.setattr(
        db_bootstrap_module,
        "ensure_event_log_indexes",
        AsyncMock(),
    )
    monkeypatch.setattr(
        db_bootstrap_module,
        "ensure_internal_monologue_residue_indexes",
        AsyncMock(),
    )
    monkeypatch.setattr(
        db_bootstrap_module,
        "purge_stale_initializer_entries",
        AsyncMock(),
    )
    monkeypatch.setattr(
        db_bootstrap_module,
        "prune_persistent_entries",
        AsyncMock(),
    )

    await db_bootstrap_module.db_bootstrap()

    index_specs = db.self_cognition_action_attempts.indexes
    assert {
        "keys": "idempotency_key",
        "kwargs": {
            "unique": True,
            "name": "self_cognition_attempt_idempotency_unique",
        },
    } in index_specs
    assert {
        "keys": [("status", 1), ("recorded_at", -1)],
        "kwargs": {"name": "self_cognition_attempt_status_recorded"},
    } in index_specs


@pytest.mark.asyncio
async def test_db_bootstrap_creates_group_review_window_indexes(
    monkeypatch,
) -> None:
    """Bootstrap should index reviewed group-window ledger state."""

    db = _BootstrapDb()
    monkeypatch.setattr(db_bootstrap_module, "get_db", AsyncMock(return_value=db))
    monkeypatch.setattr(db_bootstrap_module, "enable_vector_index", AsyncMock())
    monkeypatch.setattr(
        db_bootstrap_module,
        "ensure_reflection_run_indexes",
        AsyncMock(),
    )
    monkeypatch.setattr(
        db_bootstrap_module,
        "ensure_interaction_style_image_indexes",
        AsyncMock(),
    )
    monkeypatch.setattr(
        db_bootstrap_module,
        "ensure_global_character_growth_indexes",
        AsyncMock(),
    )
    monkeypatch.setattr(
        db_bootstrap_module,
        "ensure_event_log_indexes",
        AsyncMock(),
    )
    monkeypatch.setattr(
        db_bootstrap_module,
        "ensure_internal_monologue_residue_indexes",
        AsyncMock(),
    )
    monkeypatch.setattr(
        db_bootstrap_module,
        "purge_stale_initializer_entries",
        AsyncMock(),
    )
    monkeypatch.setattr(
        db_bootstrap_module,
        "prune_persistent_entries",
        AsyncMock(),
    )

    await db_bootstrap_module.db_bootstrap()

    index_specs = db.self_cognition_group_review_windows.indexes
    assert {
        "keys": "source_id",
        "kwargs": {
            "unique": True,
            "name": "self_cognition_group_review_window_source_unique",
        },
    } in index_specs
    assert {
        "keys": [("scope_ref", 1), ("status", 1), ("window_start", 1)],
        "kwargs": {"name": "self_cognition_group_review_window_scope_status"},
    } in index_specs
    assert {
        "keys": "reviewed_at",
        "kwargs": {"name": "self_cognition_group_review_window_reviewed_at"},
    } in index_specs


@pytest.mark.asyncio
async def test_db_bootstrap_delegates_global_character_growth_indexes(
    monkeypatch,
) -> None:
    """Bootstrap should prepare global character-growth storage."""

    db = _BootstrapDb()
    ensure_global_growth = AsyncMock()
    monkeypatch.setattr(db_bootstrap_module, "get_db", AsyncMock(return_value=db))
    monkeypatch.setattr(db_bootstrap_module, "enable_vector_index", AsyncMock())
    monkeypatch.setattr(
        db_bootstrap_module,
        "ensure_reflection_run_indexes",
        AsyncMock(),
    )
    monkeypatch.setattr(
        db_bootstrap_module,
        "ensure_interaction_style_image_indexes",
        AsyncMock(),
    )
    monkeypatch.setattr(
        db_bootstrap_module,
        "ensure_global_character_growth_indexes",
        ensure_global_growth,
    )
    monkeypatch.setattr(
        db_bootstrap_module,
        "ensure_event_log_indexes",
        AsyncMock(),
    )
    monkeypatch.setattr(
        db_bootstrap_module,
        "purge_stale_initializer_entries",
        AsyncMock(),
    )
    monkeypatch.setattr(
        db_bootstrap_module,
        "prune_persistent_entries",
        AsyncMock(),
    )

    await db_bootstrap_module.db_bootstrap()

    ensure_global_growth.assert_awaited_once()


@pytest.mark.asyncio
async def test_db_bootstrap_delegates_event_log_indexes(
    monkeypatch,
) -> None:
    """Bootstrap should prepare event-log storage through its DB owner."""

    db = _BootstrapDb()
    ensure_event_log = AsyncMock()
    monkeypatch.setattr(db_bootstrap_module, "get_db", AsyncMock(return_value=db))
    monkeypatch.setattr(db_bootstrap_module, "enable_vector_index", AsyncMock())
    monkeypatch.setattr(
        db_bootstrap_module,
        "ensure_reflection_run_indexes",
        AsyncMock(),
    )
    monkeypatch.setattr(
        db_bootstrap_module,
        "ensure_interaction_style_image_indexes",
        AsyncMock(),
    )
    monkeypatch.setattr(
        db_bootstrap_module,
        "ensure_global_character_growth_indexes",
        AsyncMock(),
    )
    monkeypatch.setattr(
        db_bootstrap_module,
        "ensure_event_log_indexes",
        ensure_event_log,
    )
    monkeypatch.setattr(
        db_bootstrap_module,
        "purge_stale_initializer_entries",
        AsyncMock(),
    )
    monkeypatch.setattr(
        db_bootstrap_module,
        "prune_persistent_entries",
        AsyncMock(),
    )

    await db_bootstrap_module.db_bootstrap()

    ensure_event_log.assert_awaited_once()


@pytest.mark.asyncio
async def test_db_bootstrap_configures_conversation_vector_filter_paths(
    monkeypatch,
) -> None:
    """Bootstrap should request filterable fields for conversation vector search."""

    db = _BootstrapDb()
    enable_vector_index = AsyncMock()
    monkeypatch.setattr(db_bootstrap_module, "get_db", AsyncMock(return_value=db))
    monkeypatch.setattr(
        db_bootstrap_module,
        "enable_vector_index",
        enable_vector_index,
    )
    monkeypatch.setattr(
        db_bootstrap_module,
        "ensure_reflection_run_indexes",
        AsyncMock(),
    )
    monkeypatch.setattr(
        db_bootstrap_module,
        "ensure_interaction_style_image_indexes",
        AsyncMock(),
    )
    monkeypatch.setattr(
        db_bootstrap_module,
        "ensure_global_character_growth_indexes",
        AsyncMock(),
    )
    monkeypatch.setattr(
        db_bootstrap_module,
        "ensure_event_log_indexes",
        AsyncMock(),
    )
    monkeypatch.setattr(
        db_bootstrap_module,
        "purge_stale_initializer_entries",
        AsyncMock(),
    )
    monkeypatch.setattr(
        db_bootstrap_module,
        "prune_persistent_entries",
        AsyncMock(),
    )

    await db_bootstrap_module.db_bootstrap()

    enable_vector_index.assert_any_await(
        "conversation_history",
        "conversation_history_vector_index",
        path="embedding",
        filter_paths=[
            "platform",
            "platform_channel_id",
            "global_user_id",
            "role",
            "timestamp",
        ],
    )


@pytest.mark.asyncio
async def test_global_character_growth_index_bootstrap(monkeypatch) -> None:
    """Global growth DB interface should create all required indexes."""

    db = _BootstrapDb()
    db.collections.pop("global_character_growth_traits")
    db.collections.pop("global_character_growth_runs")
    delattr(db, "global_character_growth_traits")
    delattr(db, "global_character_growth_runs")
    monkeypatch.setattr(
        db_global_growth_module,
        "get_db",
        AsyncMock(return_value=db),
    )

    await db_global_growth_module.ensure_global_character_growth_indexes()

    traits = db["global_character_growth_traits"]
    runs = db["global_character_growth_runs"]
    trait_index_names = {
        index["kwargs"]["name"]
        for index in traits.indexes
    }
    run_index_names = {
        index["kwargs"]["name"]
        for index in runs.indexes
    }
    assert trait_index_names == {
        "global_growth_trait_id_unique",
        "global_growth_trait_status_maturity",
        "global_growth_trait_axis_status",
        "global_growth_trait_source_memory",
    }
    assert run_index_names == {
        "global_growth_run_id_unique",
        "global_growth_run_status_updated",
        "global_growth_run_source_memory",
        "global_growth_run_source_reflection",
    }


# ── Character state edge cases ─────────────────────────────────────


@pytest.mark.asyncio
async def test_upsert_character_state_preserves_on_empty_string():
    """When mood/global_vibe/reflection_summary is empty string, preserve existing value."""
    db = _mock_db()
    db.character_state.find_one = AsyncMock(return_value={
        "_id": "global",
        "mood": "old_mood",
        "global_vibe": "old_vibe",
        "reflection_summary": "old_summary",
    })
    db.character_state.update_one = AsyncMock()

    with _patched_get_db(db):
        await upsert_character_state("", "", "", "t2")

    call_args = db.character_state.update_one.call_args
    set_payload = call_args[0][1]["$set"]
    assert set_payload["mood"] == "old_mood"
    assert set_payload["global_vibe"] == "old_vibe"
    assert set_payload["reflection_summary"] == "old_summary"


# ── Character profile ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_character_profile_found():
    """All top-level fields (minus _id) are returned."""
    db = _mock_db()
    db.character_state.find_one = AsyncMock(return_value={
        "_id": "global",
        "mood": "calm",
        "name": "Kazusa",
        "age": 15,
    })

    with _patched_get_db(db):
        result = await get_character_profile()

    assert result == {"mood": "calm", "name": "Kazusa", "age": 15}
    assert "_id" not in result


@pytest.mark.asyncio
async def test_get_character_profile_not_found():
    """Returns empty dict when no global doc exists."""
    db = _mock_db()
    db.character_state.find_one = AsyncMock(return_value=None)

    with _patched_get_db(db):
        result = await get_character_profile()

    assert result == {}


@pytest.mark.asyncio
async def test_save_character_profile():
    """Each profile key is $set at the top level."""
    db = _mock_db()
    db.character_state.update_one = AsyncMock()

    profile = {"name": "Kazusa", "age": 15, "tone": "warm"}

    with _patched_get_db(db):
        await save_character_profile(profile)

    db.character_state.update_one.assert_called_once_with(
        {"_id": "global"},
        {"$set": profile},
        upsert=True,
    )


@pytest.mark.asyncio
async def test_get_character_state_returns_same_as_profile():
    """get_character_state is an alias for get_character_profile."""
    db = _mock_db()
    db.character_state.find_one = AsyncMock(return_value={
        "_id": "global",
        "mood": "happy",
        "name": "Kazusa",
        "global_vibe": "warm",
    })

    with _patched_get_db(db):
        result = await get_character_state()

    assert result == {"mood": "happy", "name": "Kazusa", "global_vibe": "warm"}
    assert "_id" not in result


def test_split_character_profile_runtime_state_separates_runtime_fields():
    """Static profile and runtime state should be split without DB shape changes."""
    profile = {
        "_id": "global",
        "name": "Kazusa",
        "personality_brief": "sharp but kind",
        "mood": "focused",
        "global_vibe": "warm",
        "reflection_summary": "recent chat was calm",
        "self_image": {"core": "steady"},
        "updated_at": "t1",
    }

    static_profile, runtime_state = split_character_profile_runtime_state(profile)

    assert static_profile == {
        "name": "Kazusa",
        "personality_brief": "sharp but kind",
    }
    assert runtime_state == {
        "mood": "focused",
        "global_vibe": "warm",
        "reflection_summary": "recent chat was calm",
        "self_image": {"core": "steady"},
        "updated_at": "t1",
    }


def test_compose_character_profile_merges_static_runtime_and_identity():
    """Graph-facing profile should preserve static fields and fresh runtime state."""
    static_profile = {
        "name": "Kazusa",
        "personality_brief": "sharp but kind",
    }
    runtime_state = {
        "mood": "focused",
        "global_vibe": "warm",
        "reflection_summary": "recent chat was calm",
    }

    result = compose_character_profile(
        static_profile,
        runtime_state,
        "character-global-id",
    )

    assert result == {
        "name": "Kazusa",
        "personality_brief": "sharp but kind",
        "mood": "focused",
        "global_vibe": "warm",
        "reflection_summary": "recent chat was calm",
        "global_user_id": "character-global-id",
    }


@pytest.mark.asyncio
async def test_get_character_runtime_state_uses_runtime_projection():
    """Runtime reader should only request mutable runtime character fields."""
    db = _mock_db()
    db.character_state.find_one = AsyncMock(return_value={
        "_id": "global",
        "mood": "calm",
        "global_vibe": "warm",
        "reflection_summary": "recent chat was calm",
        "self_image": {"core": "steady"},
        "updated_at": "t1",
    })

    with _patched_get_db(db):
        result = await get_character_runtime_state()

    expected_projection = {
        field_name: 1 for field_name in RUNTIME_CHARACTER_STATE_FIELDS
    }
    db.character_state.find_one.assert_awaited_once_with(
        {"_id": "global"},
        expected_projection,
    )
    assert result == {
        "mood": "calm",
        "global_vibe": "warm",
        "reflection_summary": "recent chat was calm",
        "self_image": {"core": "steady"},
        "updated_at": "t1",
    }


# ── Save memory ────────────────────────────────────────────────────


def test_build_memory_doc_defaults():
    doc = build_memory_doc(
        memory_name="test_mem",
        content="some content",
        source_global_user_id="user-123",
        memory_type="fact",
        source_kind="conversation_extracted",
        confidence_note="stable fact",
    )
    assert doc["memory_name"] == "test_mem"
    assert doc["content"] == "some content"
    assert doc["source_global_user_id"] == "user-123"
    assert doc["memory_type"] == "fact"
    assert doc["source_kind"] == "conversation_extracted"
    assert doc["confidence_note"] == "stable fact"
    assert doc["status"] == "active"
    assert doc["expiry_timestamp"] is None


def test_build_memory_doc_with_expiry():
    doc = build_memory_doc(
        memory_name="promise",
        content="will do X",
        source_global_user_id="user-456",
        memory_type="promise",
        source_kind="conversation_extracted",
        confidence_note="pending",
        status="active",
        expiry_timestamp="2026-04-20T10:00:00Z",
    )
    assert doc["expiry_timestamp"] == "2026-04-20T10:00:00Z"
    assert doc["status"] == "active"


@pytest.mark.asyncio
async def test_save_memory_wraps_evolving_insert_api():
    """Legacy save_memory should delegate to the evolving memory API."""
    doc = build_memory_doc(
        memory_name="test_mem",
        content="some content",
        source_global_user_id="",
        memory_type="fact",
        source_kind="conversation_extracted",
        confidence_note="stable fact",
    )

    insert_memory_unit = AsyncMock(return_value={})
    with patch(
        "kazusa_ai_chatbot.db.memory.insert_memory_unit",
        insert_memory_unit,
    ):
        await save_memory(doc, "2024-01-01T00:00:00Z")

    insert_memory_unit.assert_awaited_once()
    payload = insert_memory_unit.await_args.kwargs["document"]
    assert payload["memory_unit_id"].startswith("manual_")
    assert payload["lineage_id"] == payload["memory_unit_id"]
    assert payload["version"] == 1
    assert payload["memory_name"] == "test_mem"
    assert payload["content"] == "some content"
    assert payload["memory_type"] == "fact"
    assert payload["source_kind"] == "conversation_extracted"
    assert payload["status"] == "active"
    assert payload["timestamp"] == "2024-01-01T00:00:00Z"


# ── Search memory ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_search_memory_keyword():
    """Keyword search uses $or regex on memory_name and content."""
    db = _mock_db()
    cursor = AsyncMock()
    cursor.to_list = AsyncMock(return_value=[
        {"memory_name": "test", "content": "data", "timestamp": "t1", "embedding": [0.1]},
    ])
    db.memory.find.return_value.limit.return_value = cursor

    with _patched_get_db(db):
        results = await search_memory("test", method="keyword", limit=5)

    assert len(results) == 1
    assert results[0][0] == -1.0
    # Embedding should be removed
    assert "embedding" not in results[0][1]


@pytest.mark.asyncio
async def test_search_memory_keyword_escapes_regex_metacharacters():
    """Keyword memory search should treat user text as literal text."""
    db = _mock_db()
    cursor = AsyncMock()
    cursor.to_list = AsyncMock(return_value=[])
    db.memory.find.return_value.limit.return_value = cursor

    query = r"\p{Han}"
    with _patched_get_db(db):
        await search_memory(query, method="keyword", limit=5)

    call_filter = db.memory.find.call_args[0][0]
    expected_pattern = re.escape(query)
    assert call_filter["$or"][0]["memory_name"]["$regex"] == expected_pattern
    assert call_filter["$or"][1]["content"]["$regex"] == expected_pattern


@pytest.mark.asyncio
async def test_search_memory_vector():
    """Vector search uses $vectorSearch aggregation pipeline."""
    db = _mock_db()
    cursor = AsyncMock()
    cursor.to_list = AsyncMock(return_value=[
        {"memory_name": "test", "content": "data", "timestamp": "t1", "score": 0.88},
    ])
    db.memory.aggregate = MagicMock(return_value=cursor)

    with _patched_get_db(db), _patched_query_embedding(return_value=[0.1, 0.2]):
        results = await search_memory("test query", method="vector", limit=3)

    assert len(results) == 1
    assert results[0][0] == 0.88
    assert results[0][1]["memory_name"] == "test"
    # Verify aggregate was called with $vectorSearch
    pipeline = db.memory.aggregate.call_args[0][0]
    assert "$vectorSearch" in pipeline[0]
    vector_search = pipeline[0]["$vectorSearch"]
    assert "filter" not in vector_search
    assert vector_search["limit"] > 3
    assert {"status": "active"} in pipeline[2]["$match"]["$and"]
    assert pipeline[-1] == {"$limit": 3}


# ── close_db ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_close_db_resets_globals():
    """close_db should set _client and _db to None."""
    mock_client = MagicMock()

    db_client_module._client = mock_client
    db_client_module._db = MagicMock()
    db_client_module._db_loop = None

    await close_db()

    assert db_client_module._client is None
    assert db_client_module._db is None
    assert db_client_module._db_loop is None
    mock_client.close.assert_called_once()


@pytest.mark.asyncio
async def test_close_db_noop_when_not_connected():
    """close_db should be safe to call when not connected."""
    db_client_module._client = None
    db_client_module._db = None
    db_client_module._db_loop = None

    await close_db()  # Should not raise

    assert db_client_module._client is None


# ═══════════════════════════════════════════════════════════════════════
# Live MongoDB integration tests
# Require a running MongoDB instance.
# Run with:  pytest -m live_db -v
# ═══════════════════════════════════════════════════════════════════════

TEST_DB_NAME = "_test_roleplay_bot"


@pytest.fixture
async def live_test_db():
    """Connect to the real MongoDB, use a dedicated test database, and clean up.

    Overrides the module-level MONGODB_DB_NAME in bot.db so that get_db()
    creates a client pointing at the test database instead of production.
    All collections are dropped after each test.
    """
    # Reset the db module singleton so get_db() creates a fresh connection
    db_client_module._client = None
    db_client_module._db = None
    db_client_module._db_loop = None

    # Patch the module-level binding that get_db() reads on line 22 of db.py
    original_db_name = db_client_module.MONGODB_DB_NAME
    db_client_module.MONGODB_DB_NAME = TEST_DB_NAME

    # Force connection to the test database
    db = await get_db()

    # Clean up any stale data from previous runs
    for coll_name in await db.list_collection_names():
        await db.drop_collection(coll_name)

    yield db

    # Cleanup: drop all test collections
    for coll_name in await db.list_collection_names():
        await db.drop_collection(coll_name)

    # Close and restore original DB name
    await close_db()
    db_client_module.MONGODB_DB_NAME = original_db_name


# ── Conversation history ──────────────────────────────────────────────

@live_db
@pytest.mark.asyncio
async def test_live_save_and_get_conversation_history(live_test_db):
    """Save messages and retrieve them in chronological order."""
    await save_conversation({
        "channel_id": "test_chan",
        "role": "user",
        "user_id": "user_100",
        "name": "Alice",
        "content": "Hello!",
        "timestamp": "2026-01-01T00:00:01Z"
    })
    await save_conversation({
        "channel_id": "test_chan",
        "role": "assistant",
        "user_id": "bot_001",
        "name": "bot",
        "content": "Hi Alice!",
        "timestamp": "2026-01-01T00:00:02Z"
    })
    await save_conversation({
        "channel_id": "test_chan",
        "role": "user",
        "user_id": "user_100",
        "name": "Alice",
        "content": "How are you?",
        "timestamp": "2026-01-01T00:00:03Z"
    })

    history = await get_conversation_history("test_chan", limit=10)

    assert len(history) == 3
    # Oldest first
    assert history[0]["content"] == "Hello!"
    assert history[0]["role"] == "user"
    assert history[0]["user_id"] == "user_100"
    assert history[1]["content"] == "Hi Alice!"
    assert history[1]["role"] == "assistant"
    assert history[1]["user_id"] == "bot_001"
    assert history[2]["content"] == "How are you?"
    assert history[2]["user_id"] == "user_100"


@live_db
@pytest.mark.asyncio
async def test_live_conversation_history_respects_limit(live_test_db):
    """Only the most recent N messages are returned."""
    for i in range(10):
        await save_conversation({
            "channel_id": "test_chan",
            "role": "user",
            "user_id": "user_100",
            "name": "Alice",
            "content": f"msg_{i}",
            "timestamp": f"2026-01-01T00:00:{i:02d}Z"
        })

    history = await get_conversation_history("test_chan", limit=3)

    assert len(history) == 3
    # Should be the last 3 messages, oldest first
    assert history[0]["content"] == "msg_7"
    assert history[1]["content"] == "msg_8"
    assert history[2]["content"] == "msg_9"


@live_db
@pytest.mark.asyncio
async def test_live_conversation_history_channel_isolation(live_test_db):
    """Messages from different channels don't mix."""
    await save_conversation({
        "channel_id": "chan_a",
        "role": "user",
        "user_id": "user_100",
        "name": "Alice",
        "content": "In channel A",
        "timestamp": "2026-01-01T00:00:01Z"
    })
    await save_conversation({
        "channel_id": "chan_b",
        "role": "user",
        "user_id": "user_200",
        "name": "Bob",
        "content": "In channel B",
        "timestamp": "2026-01-01T00:00:02Z"
    })

    history_a = await get_conversation_history("chan_a", limit=10)
    history_b = await get_conversation_history("chan_b", limit=10)

    assert len(history_a) == 1
    assert history_a[0]["content"] == "In channel A"
    assert len(history_b) == 1
    assert history_b[0]["content"] == "In channel B"


# ── Character state ──────────────────────────────────────────────────


@live_db
@pytest.mark.asyncio
async def test_live_character_state_empty(live_test_db):
    """No character state initially."""
    state = await get_character_state()
    assert state == {}


@live_db
@pytest.mark.asyncio
async def test_live_upsert_and_get_character_state(live_test_db):
    """Store character state and retrieve it."""
    await upsert_character_state("happy", "warm vibe", "Met an old friend today", "2026-01-01T00:00:00Z")

    state = await get_character_state()
    assert state["mood"] == "happy"
    assert state["global_vibe"] == "warm vibe"
    assert state["reflection_summary"] == "Met an old friend today"
    assert state["updated_at"] == "2026-01-01T00:00:00Z"
    assert "_id" not in state


@live_db
@pytest.mark.asyncio
async def test_live_character_state_update_overwrites_mood(live_test_db):
    """Updating character state replaces mood and global_vibe."""
    await upsert_character_state("happy", "warm", "reflection 1", "t1")
    await upsert_character_state("sad", "guarded", "reflection 2", "t2")

    state = await get_character_state()
    assert state["mood"] == "sad"
    assert state["global_vibe"] == "guarded"
    assert state["reflection_summary"] == "reflection 2"


# ── Affinity (mocked) ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_affinity_default_for_unknown_user():
    db = _mock_db()
    db.user_profiles.find_one = AsyncMock(return_value=None)

    with _patched_get_db(db):
        result = await get_affinity("unknown_user")

    assert result == AFFINITY_DEFAULT


@pytest.mark.asyncio
async def test_get_affinity_returns_stored_value():
    db = _mock_db()
    db.user_profiles.find_one = AsyncMock(return_value={"user_id": "u1", "affinity": 750})

    with _patched_get_db(db):
        result = await get_affinity("u1")

    assert result == 750


@pytest.mark.asyncio
async def test_update_affinity_clamps_to_max():
    db = _mock_db()
    db.user_profiles.find_one = AsyncMock(return_value={"user_id": "u1", "affinity": 995})
    db.user_profiles.update_one = AsyncMock()

    with _patched_get_db(db):
        result = await update_affinity("u1", 10)

    assert result == 1000


@pytest.mark.asyncio
async def test_update_affinity_clamps_to_min():
    db = _mock_db()
    db.user_profiles.find_one = AsyncMock(return_value={"user_id": "u1", "affinity": 5})
    db.user_profiles.update_one = AsyncMock()

    with _patched_get_db(db):
        result = await update_affinity("u1", -20)

    assert result == 0


@pytest.mark.asyncio
async def test_enable_vector_index_mocked():
    db = _mock_db()
    
    # Create a proper AsyncMock for the collection
    collection = AsyncMock()
    
    # Mock list_search_indexes to simulate no existing index
    async def mock_list_indexes():
        return
        yield  # This makes it an async generator that yields nothing
    
    collection.list_search_indexes = MagicMock(return_value=mock_list_indexes())
    collection.create_search_index = AsyncMock()
    db.__getitem__.return_value = collection

    with _patched_get_db(db), _patched_embedding(return_value=[0.1, 0.2, 0.3]):
        await db_module.enable_vector_index("conversation_history", "conversation_history_vector_index")
    
    collection.create_search_index.assert_called_once()
    # Check the model definition
    model = collection.create_search_index.call_args[0][0]
    assert model.document["name"] == "conversation_history_vector_index"
    assert model.document["type"] == "vectorSearch"
    assert model.document["definition"]["fields"][0]["numDimensions"] == 3


@pytest.mark.asyncio
async def test_search_conversation_history_keyword_mocked():
    """Keyword search uses regex and returns score -1.0."""
    db = _mock_db()
    cursor = AsyncMock()
    cursor.to_list = AsyncMock(return_value=[
        {"channel_id": "c1", "timestamp": "t1", "body_text": "keyword matched"},
    ])
    db.conversation_history.find.return_value.sort.return_value.limit.return_value = cursor

    with _patched_get_db(db):
        results = await db_module.search_conversation_history("keyword", method="keyword", limit=2)

    assert len(results) == 1
    assert results[0][0] == -1.0
    assert results[0][1]["body_text"] == "keyword matched"
    # Verify the regex filter was passed
    call_filter = db.conversation_history.find.call_args[0][0]
    assert call_filter["$or"][0]["body_text"]["$regex"] == "keyword"
    assert call_filter["$or"][1]["attachments.description"]["$regex"] == "keyword"


@pytest.mark.asyncio
async def test_search_conversation_history_keyword_escapes_regex_metacharacters():
    """Keyword conversation search should treat user text as literal text."""
    db = _mock_db()
    cursor = AsyncMock()
    cursor.to_list = AsyncMock(return_value=[])
    db.conversation_history.find.return_value.sort.return_value.limit.return_value = (
        cursor
    )

    query = r"\p{Han}"
    with _patched_get_db(db):
        await db_module.search_conversation_history(
            query,
            method="keyword",
            limit=2,
        )

    call_filter = db.conversation_history.find.call_args[0][0]
    expected_pattern = re.escape(query)
    assert call_filter["$or"][0]["body_text"]["$regex"] == expected_pattern
    assert (
        call_filter["$or"][1]["attachments.description"]["$regex"]
        == expected_pattern
    )


@pytest.mark.asyncio
async def test_search_conversation_history_keyword_with_filters_mocked():
    """Keyword search applies platform_channel_id and global_user_id filters."""
    db = _mock_db()
    cursor = AsyncMock()
    cursor.to_list = AsyncMock(return_value=[])
    db.conversation_history.find.return_value.sort.return_value.limit.return_value = cursor

    with _patched_get_db(db):
        await db_module.search_conversation_history(
            "test", method="keyword", platform_channel_id="ch1", global_user_id="u1"
        )

    call_filter = db.conversation_history.find.call_args[0][0]
    assert call_filter["platform_channel_id"] == "ch1"
    assert call_filter["global_user_id"] == "u1"
    assert call_filter["$or"][0]["body_text"]["$regex"] == "test"
    assert call_filter["$or"][1]["attachments.description"]["$regex"] == "test"


@pytest.mark.asyncio
async def test_search_conversation_history_vector_mocked(
    monkeypatch: pytest.MonkeyPatch,
):
    """Vector search uses $vectorSearch aggregation pipeline."""
    db = _mock_db()
    cursor = AsyncMock()
    cursor.to_list = AsyncMock(return_value=[
        {"channel_id": "c1", "body_text": "vector matched", "score": 0.95},
    ])
    db.conversation_history.aggregate = MagicMock(return_value=cursor)
    monkeypatch.setattr(
        db_conversation_module,
        "_conversation_vector_prefilter_supported",
        AsyncMock(return_value=False),
        raising=False,
    )

    with _patched_get_db(db), _patched_query_embedding(
        return_value=[0.1, 0.2, 0.3],
    ):
        results = await db_module.search_conversation_history("test query", method="vector", limit=2)

    assert len(results) == 1
    assert results[0][0] == 0.95
    assert results[0][1]["body_text"] == "vector matched"
    # Verify aggregate was called with $vectorSearch pipeline
    pipeline = db.conversation_history.aggregate.call_args[0][0]
    assert "$vectorSearch" in pipeline[0]
    vs = pipeline[0]["$vectorSearch"]
    assert vs["queryVector"] == [0.1, 0.2, 0.3]
    assert vs["index"] == "conversation_history_vector_index"
    assert vs["numCandidates"] == db_conversation_module.RAG_VECTOR_MIN_CANDIDATES


@pytest.mark.asyncio
async def test_search_conversation_history_vector_with_filters_mocked(
    monkeypatch: pytest.MonkeyPatch,
):
    """Vector search adds $match stage for platform_channel_id/global_user_id post-filtering."""
    db = _mock_db()
    cursor = AsyncMock()
    cursor.to_list = AsyncMock(return_value=[])
    db.conversation_history.aggregate = MagicMock(return_value=cursor)
    monkeypatch.setattr(
        db_conversation_module,
        "_conversation_vector_prefilter_supported",
        AsyncMock(return_value=False),
        raising=False,
    )

    with _patched_get_db(db), _patched_query_embedding(
        return_value=[0.1, 0.2, 0.3],
    ):
        await db_module.search_conversation_history(
            "test", method="vector", platform_channel_id="ch1", global_user_id="u1", limit=3
        )

    pipeline = db.conversation_history.aggregate.call_args[0][0]
    # Should contain a $match stage with both filters
    match_stages = [s for s in pipeline if "$match" in s]
    assert len(match_stages) == 1
    assert match_stages[0]["$match"]["platform_channel_id"] == "ch1"
    assert match_stages[0]["$match"]["global_user_id"] == "u1"
    assert "filter" not in pipeline[0]["$vectorSearch"]


@pytest.mark.asyncio
async def test_search_conversation_history_vector_uses_prefilter_when_supported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Vector search should push supported filters into $vectorSearch."""

    db = _mock_db()
    cursor = AsyncMock()
    cursor.to_list = AsyncMock(return_value=[])
    db.conversation_history.aggregate = MagicMock(return_value=cursor)
    monkeypatch.setattr(
        db_conversation_module,
        "_conversation_vector_prefilter_supported",
        AsyncMock(return_value=True),
        raising=False,
    )

    with _patched_get_db(db), _patched_query_embedding(
        return_value=[0.1, 0.2, 0.3],
    ):
        await db_module.search_conversation_history(
            "test",
            method="vector",
            platform="qq",
            platform_channel_id="ch1",
            global_user_id="u1",
            from_timestamp="2026-05-11T00:00:00Z",
            to_timestamp="2026-05-12T00:00:00Z",
            limit=3,
        )

    pipeline = db.conversation_history.aggregate.call_args[0][0]
    vector_filter = pipeline[0]["$vectorSearch"]["filter"]
    assert vector_filter == {
        "platform": "qq",
        "platform_channel_id": "ch1",
        "global_user_id": "u1",
        "timestamp": {
            "$gte": "2026-05-11T00:00:00Z",
            "$lte": "2026-05-12T00:00:00Z",
        },
    }
    assert [stage for stage in pipeline if "$match" in stage] == []


@pytest.mark.asyncio
async def test_search_conversation_history_vector_post_filters_when_prefilter_not_supported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Vector search should keep post-filtering when index filters are absent."""

    db = _mock_db()
    cursor = AsyncMock()
    cursor.to_list = AsyncMock(return_value=[])
    db.conversation_history.aggregate = MagicMock(return_value=cursor)
    monkeypatch.setattr(
        db_conversation_module,
        "_conversation_vector_prefilter_supported",
        AsyncMock(return_value=False),
        raising=False,
    )

    with _patched_get_db(db), _patched_query_embedding(
        return_value=[0.1, 0.2, 0.3],
    ):
        await db_module.search_conversation_history(
            "test",
            method="vector",
            platform="qq",
            platform_channel_id="ch1",
            from_timestamp="2026-05-11T00:00:00Z",
            limit=20,
        )

    pipeline = db.conversation_history.aggregate.call_args[0][0]
    assert "filter" not in pipeline[0]["$vectorSearch"]
    match_stages = [stage for stage in pipeline if "$match" in stage]
    assert match_stages == [
        {
            "$match": {
                "platform": "qq",
                "platform_channel_id": "ch1",
                "timestamp": {"$gte": "2026-05-11T00:00:00Z"},
            }
        }
    ]


def test_conversation_vector_prefilter_support_detects_missing_fields() -> None:
    """Index definition helpers should report missing required filter paths."""

    index_document = {
        "latestDefinition": {
            "fields": [
                {"type": "vector", "path": "embedding", "numDimensions": 768},
                {"type": "filter", "path": "platform"},
            ]
        }
    }
    required_paths = ["platform", "platform_channel_id"]

    missing_paths = db_client_module.vector_index_missing_filter_paths(
        index_document,
        required_paths,
    )

    assert missing_paths == ["platform_channel_id"]
    assert not db_client_module.vector_index_has_filter_paths(
        index_document,
        required_paths,
    )


def test_vector_index_definition_issues_detect_vector_mismatch() -> None:
    """Index definition helpers should report vector-field mismatches."""

    index_document = {
        "latestDefinition": {
            "fields": [
                {
                    "type": "vector",
                    "path": "wrong_embedding",
                    "numDimensions": 512,
                    "similarity": "dotProduct",
                },
                {"type": "filter", "path": "platform"},
            ]
        }
    }

    issues = db_client_module.vector_index_definition_issues(
        index_document,
        path="embedding",
        num_dimensions=768,
        required_filter_paths=["platform", "platform_channel_id"],
    )

    assert issues == [
        "vector_path",
        "num_dimensions",
        "similarity",
        "missing_filter_path:platform_channel_id",
    ]


def test_vector_num_candidates_is_capped_to_atlas_limit() -> None:
    """Conversation vector search should not exceed the Atlas candidate cap."""

    candidate_count = db_conversation_module._vector_num_candidates(1000)

    assert candidate_count == 10000


@pytest.mark.asyncio
async def test_conversation_vector_prefilter_support_rechecks_cached_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stale negative capability cache should not survive index recreation."""

    index_document = {
        "latestDefinition": {
            "fields": [
                {
                    "type": "vector",
                    "path": "embedding",
                    "numDimensions": 768,
                    "similarity": "cosine",
                },
                {"type": "filter", "path": "platform"},
                {"type": "filter", "path": "platform_channel_id"},
                {"type": "filter", "path": "global_user_id"},
                {"type": "filter", "path": "role"},
                {"type": "filter", "path": "timestamp"},
            ]
        }
    }
    get_definition = AsyncMock(return_value=index_document)
    monkeypatch.setattr(
        db_conversation_module,
        "_conversation_vector_prefilter_support_cache",
        False,
        raising=False,
    )
    monkeypatch.setattr(
        db_conversation_module,
        "get_search_index_definition",
        get_definition,
    )

    supported = await db_conversation_module._conversation_vector_prefilter_supported()

    assert supported is True
    get_definition.assert_awaited_once()


# NOTE: test_search_lore_mocked removed due to async mock complexity
# The search_lore functionality is tested in the RAG integration tests


# ── Affinity (live) ──────────────────────────────────────────────────


@live_db
@pytest.mark.asyncio
async def test_live_affinity_default_for_new_user(live_test_db):
    """New users start at AFFINITY_DEFAULT (200)."""
    score = await get_affinity("new_user")
    assert score == AFFINITY_DEFAULT


@live_db
@pytest.mark.asyncio
async def test_live_update_affinity_positive(live_test_db):
    """Positive delta increases affinity from default."""
    new_val = await update_affinity("user_aff", 10)
    assert new_val == AFFINITY_DEFAULT + 10
    assert await get_affinity("user_aff") == AFFINITY_DEFAULT + 10


@live_db
@pytest.mark.asyncio
async def test_live_update_affinity_negative(live_test_db):
    """Negative delta decreases affinity."""
    await update_affinity("user_aff", 10)  # 510
    new_val = await update_affinity("user_aff", -20)  # 490
    assert new_val == 490


@live_db
@pytest.mark.asyncio
async def test_live_affinity_clamps_at_bounds(live_test_db):
    """Affinity cannot exceed 0–1000."""
    # Clamp at 0
    await update_affinity("user_low", -600)  # AFFINITY_DEFAULT - 600 → 0
    assert await get_affinity("user_low") == 0

    # Clamp at 1000
    await update_affinity("user_high", 600)  # AFFINITY_DEFAULT + 600 → 1000
    assert await get_affinity("user_high") == 1000


@live_db
@pytest.mark.asyncio
async def test_live_enable_conversation_history_vector_index(live_test_db):
    """Ensure vector index creation works and doesn't fail on existing index.
    
    This actually hits the local embedding LLM to get the vector dimensions.
    """
    # Insert a dummy document to ensure the collection exists
    await db_module.save_conversation({
        "channel_id": "test_chan",
        "role": "user",
        "user_id": "user_100",
        "name": "Alice",
        "content": "test message for index creation",
        "timestamp": "2026-01-01T00:00:01Z"
    })
    
    # 1. Create the index
    await db_module.enable_vector_index("conversation_history", "conversation_history_vector_index")
    
    # 2. Check that it exists
    indexes = []
    async for idx in live_test_db.conversation_history.list_search_indexes():
        indexes.append(idx["name"])
    
    assert "conversation_history_vector_index" in indexes
    
    # 3. Running again shouldn't fail (it should log that it exists and return)
    await db_module.enable_vector_index("conversation_history", "conversation_history_vector_index")


@live_db
@pytest.mark.asyncio
async def test_live_search_conversation_history_keyword(live_test_db):
    """Keyword search returns regex-matched messages with score -1."""
    await db_module.save_conversation({
        "channel_id": "test_search",
        "role": "user",
        "user_id": "user_100",
        "name": "Alice",
        "content": "I like the unique term FLUMMOXED today.",
        "timestamp": "2026-01-01T00:00:01Z"
    })
    await db_module.save_conversation({
        "channel_id": "test_search",
        "role": "user",
        "user_id": "user_200",
        "name": "Bob",
        "content": "Let's eat some pizza and watch a movie.",
        "timestamp": "2026-01-01T00:00:02Z"
    })

    results = await db_module.search_conversation_history(
        "FLUMMOXED", channel_id="test_search", method="keyword", limit=5
    )

    assert len(results) == 1
    assert results[0][0] == -1.0
    assert "FLUMMOXED" in results[0][1]["content"]


@live_db
@pytest.mark.asyncio
async def test_live_search_conversation_history_vector(live_test_db):
    """Vector search returns semantically similar messages via $vectorSearch."""
    await db_module.save_conversation({
        "channel_id": "test_search",
        "role": "user",
        "user_id": "user_100",
        "name": "Alice",
        "content": "The northern gate was attacked by shadow wolves last night.",
        "timestamp": "2026-01-01T00:00:01Z"
    })
    await db_module.save_conversation({
        "channel_id": "test_search",
        "role": "user",
        "user_id": "user_200",
        "name": "Bob",
        "content": "I would like chocolate cake with strawberries please.",
        "timestamp": "2026-01-01T00:00:02Z"
    })
    await db_module.enable_vector_index("conversation_history", "conversation_history_vector_index")

    # Wait for the vector search index to become queryable (builds asynchronously)
    for _ in range(30):
        ready = False
        async for idx in live_test_db.conversation_history.list_search_indexes():
            if idx.get("name") == "conversation_history_vector_index" and idx.get("status") == "READY":
                ready = True
                break
        if ready:
            break
        await asyncio.sleep(1)

    results = await db_module.search_conversation_history(
        "wolves breached the gate", channel_id="test_search", method="vector", limit=2
    )

    # Should return results scored by vector similarity
    assert len(results) >= 1
    for score, doc in results:
        assert isinstance(score, float)
        assert score > 0.0
        assert "embedding" not in doc  # embeddings should be $unset
