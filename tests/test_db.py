"""Tests for DB helpers — mocked unit tests + live MongoDB integration tests."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import kazusa_ai_chatbot.db as db_module
from kazusa_ai_chatbot.db import (
    AFFINITY_DEFAULT,
    build_memory_doc,
    close_db,
    get_affinity,
    get_character_profile,
    get_character_state,
    get_conversation_history,
    get_db,
    get_user_profile,
    save_character_profile,
    save_conversation,
    save_memory,
    search_memory,
    update_affinity,
    update_last_relationship_insight,
    upsert_character_state,
)

# Mark for tests that require a running MongoDB instance.
# Run with:  pytest -m live_db -v
live_db = pytest.mark.live_db


def _mock_db():
    """Create a mock database object with collection access."""
    db = MagicMock()
    return db


@pytest.mark.asyncio
async def test_get_conversation_history():
    mock_docs = [
        {"role": "assistant", "user_id": "bot_001", "name": "bot", "content": "Hello", "timestamp": "t2"},
        {"role": "user", "user_id": "user_001", "name": "User", "content": "Hi", "timestamp": "t1"},
    ]
    db = _mock_db()
    cursor = AsyncMock()
    cursor.to_list = AsyncMock(return_value=mock_docs)
    db.conversation_history.find.return_value.sort.return_value.limit.return_value = cursor

    with patch("kazusa_ai_chatbot.db.get_db", new_callable=AsyncMock, return_value=db):
        result = await get_conversation_history(platform="discord", platform_channel_id="chan_1", limit=10)
        
        # Verify basic find parameters
        db.conversation_history.find.assert_called_with({"platform": "discord", "platform_channel_id": "chan_1"})

        # Verify sorting and limit
        db.conversation_history.find.return_value.sort.assert_called_with("timestamp", -1)
        db.conversation_history.find.return_value.sort.return_value.limit.assert_called_with(10)

    # Should be reversed (oldest first)
    assert result[0]["content"] == "Hi"
    assert result[1]["content"] == "Hello"


@pytest.mark.asyncio
async def test_get_conversation_history_filters():
    db = _mock_db()
    cursor = AsyncMock()
    cursor.to_list = AsyncMock(return_value=[])
    db.conversation_history.find.return_value.sort.return_value.limit.return_value = cursor

    with patch("kazusa_ai_chatbot.db.get_db", new_callable=AsyncMock, return_value=db):
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

    with patch("kazusa_ai_chatbot.db.get_db", new_callable=AsyncMock, return_value=db):
        result = await get_character_state()

    assert result["mood"] == "calm"
    assert "_id" not in result


@pytest.mark.asyncio
async def test_get_character_state_not_found():
    db = _mock_db()
    db.character_state.find_one = AsyncMock(return_value=None)

    with patch("kazusa_ai_chatbot.db.get_db", new_callable=AsyncMock, return_value=db):
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

    with patch("kazusa_ai_chatbot.db.get_db", new_callable=AsyncMock, return_value=db):
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
        "embedding": [0.1, 0.2],
    })

    with patch("kazusa_ai_chatbot.db.get_db", new_callable=AsyncMock, return_value=db):
        result = await get_user_profile("u1")

    assert result["global_user_id"] == "u1"
    assert result["affinity"] == 600
    assert "_id" not in result
    assert "embedding" not in result


@pytest.mark.asyncio
async def test_get_user_profile_not_found():
    db = _mock_db()
    db.user_profiles.find_one = AsyncMock(return_value=None)

    with patch("kazusa_ai_chatbot.db.get_db", new_callable=AsyncMock, return_value=db):
        result = await get_user_profile("unknown")

    assert result == {}


# ── Relationship insight ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_update_last_relationship_insight():
    db = _mock_db()
    db.user_profiles.update_one = AsyncMock()

    with patch("kazusa_ai_chatbot.db.get_db", new_callable=AsyncMock, return_value=db):
        await update_last_relationship_insight("u1", "very friendly")

    db.user_profiles.update_one.assert_called_once_with(
        {"global_user_id": "u1"},
        {"$set": {"last_relationship_insight": "very friendly"}},
        upsert=True,
    )


# ── Save conversation ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_save_conversation_generates_embedding():
    """save_conversation should generate an embedding when not provided."""
    db = _mock_db()
    db.conversation_history.insert_one = AsyncMock()

    doc = {
        "channel_id": "c1",
        "role": "user",
        "user_id": "u1",
        "name": "Alice",
        "content": "Hello!",
        "timestamp": "t1",
    }

    with patch("kazusa_ai_chatbot.db.get_db", new_callable=AsyncMock, return_value=db), \
         patch("kazusa_ai_chatbot.db.get_text_embedding", new_callable=AsyncMock, return_value=[0.1, 0.2]):
        await save_conversation(doc)

    assert doc["embedding"] == [0.1, 0.2]
    db.conversation_history.insert_one.assert_called_once_with(doc)


@pytest.mark.asyncio
async def test_save_conversation_preserves_existing_embedding():
    """save_conversation should not regenerate embedding if already present."""
    db = _mock_db()
    db.conversation_history.insert_one = AsyncMock()

    doc = {
        "channel_id": "c1",
        "role": "user",
        "user_id": "u1",
        "name": "Alice",
        "content": "Hello!",
        "timestamp": "t1",
        "embedding": [0.5, 0.6],
    }

    with patch("kazusa_ai_chatbot.db.get_db", new_callable=AsyncMock, return_value=db), \
         patch("kazusa_ai_chatbot.db.get_text_embedding", new_callable=AsyncMock) as mock_embed:
        await save_conversation(doc)

    mock_embed.assert_not_called()
    assert doc["embedding"] == [0.5, 0.6]


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

    with patch("kazusa_ai_chatbot.db.get_db", new_callable=AsyncMock, return_value=db):
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

    with patch("kazusa_ai_chatbot.db.get_db", new_callable=AsyncMock, return_value=db):
        result = await get_character_profile()

    assert result == {"mood": "calm", "name": "Kazusa", "age": 15}
    assert "_id" not in result


@pytest.mark.asyncio
async def test_get_character_profile_not_found():
    """Returns empty dict when no global doc exists."""
    db = _mock_db()
    db.character_state.find_one = AsyncMock(return_value=None)

    with patch("kazusa_ai_chatbot.db.get_db", new_callable=AsyncMock, return_value=db):
        result = await get_character_profile()

    assert result == {}


@pytest.mark.asyncio
async def test_save_character_profile():
    """Each profile key is $set at the top level."""
    db = _mock_db()
    db.character_state.update_one = AsyncMock()

    profile = {"name": "Kazusa", "age": 15, "tone": "warm"}

    with patch("kazusa_ai_chatbot.db.get_db", new_callable=AsyncMock, return_value=db):
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

    with patch("kazusa_ai_chatbot.db.get_db", new_callable=AsyncMock, return_value=db):
        result = await get_character_state()

    assert result == {"mood": "happy", "name": "Kazusa", "global_vibe": "warm"}
    assert "_id" not in result


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
async def test_save_memory_creates_embedding_and_inserts():
    db = _mock_db()
    db.memory.insert_one = AsyncMock()

    doc = build_memory_doc(
        memory_name="test_mem",
        content="some content",
        source_global_user_id="",
        memory_type="fact",
        source_kind="conversation_extracted",
        confidence_note="stable fact",
    )

    mock_embed = AsyncMock(return_value=[0.1, 0.2])
    with patch("kazusa_ai_chatbot.db.get_db", new_callable=AsyncMock, return_value=db), \
         patch("kazusa_ai_chatbot.db.get_text_embedding", mock_embed):
        await save_memory(doc, "2024-01-01T00:00:00Z")

    # Verify structured embedding text
    expected_text = "type:fact\nsource:conversation_extracted\ntitle:test_mem\ncontent:some content"
    mock_embed.assert_called_once_with(expected_text)

    db.memory.insert_one.assert_called_once()
    payload = db.memory.insert_one.call_args[0][0]
    assert payload["memory_name"] == "test_mem"
    assert payload["content"] == "some content"
    assert payload["embedding"] == [0.1, 0.2]
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

    with patch("kazusa_ai_chatbot.db.get_db", new_callable=AsyncMock, return_value=db):
        results = await search_memory("test", method="keyword", limit=5)

    assert len(results) == 1
    assert results[0][0] == -1.0
    # Embedding should be removed
    assert "embedding" not in results[0][1]


@pytest.mark.asyncio
async def test_search_memory_vector():
    """Vector search uses $vectorSearch aggregation pipeline."""
    db = _mock_db()
    cursor = AsyncMock()
    cursor.to_list = AsyncMock(return_value=[
        {"memory_name": "test", "content": "data", "timestamp": "t1", "score": 0.88},
    ])
    db.memory.aggregate = MagicMock(return_value=cursor)

    with patch("kazusa_ai_chatbot.db.get_db", new_callable=AsyncMock, return_value=db), \
         patch("kazusa_ai_chatbot.db.get_text_embedding", new_callable=AsyncMock, return_value=[0.1, 0.2]):
        results = await search_memory("test query", method="vector", limit=3)

    assert len(results) == 1
    assert results[0][0] == 0.88
    assert results[0][1]["memory_name"] == "test"
    # Verify aggregate was called with $vectorSearch
    pipeline = db.memory.aggregate.call_args[0][0]
    assert "$vectorSearch" in pipeline[0]


# ── close_db ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_close_db_resets_globals():
    """close_db should set _client and _db to None."""
    mock_client = MagicMock()

    db_module._client = mock_client
    db_module._db = MagicMock()

    await close_db()

    assert db_module._client is None
    assert db_module._db is None
    mock_client.close.assert_called_once()


@pytest.mark.asyncio
async def test_close_db_noop_when_not_connected():
    """close_db should be safe to call when not connected."""
    db_module._client = None
    db_module._db = None

    await close_db()  # Should not raise

    assert db_module._client is None


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
    db_module._client = None
    db_module._db = None

    # Patch the module-level binding that get_db() reads on line 22 of db.py
    original_db_name = db_module.MONGODB_DB_NAME
    db_module.MONGODB_DB_NAME = TEST_DB_NAME

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
    db_module.MONGODB_DB_NAME = original_db_name


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

    with patch("kazusa_ai_chatbot.db.get_db", new_callable=AsyncMock, return_value=db):
        result = await get_affinity("unknown_user")

    assert result == AFFINITY_DEFAULT


@pytest.mark.asyncio
async def test_get_affinity_returns_stored_value():
    db = _mock_db()
    db.user_profiles.find_one = AsyncMock(return_value={"user_id": "u1", "affinity": 750})

    with patch("kazusa_ai_chatbot.db.get_db", new_callable=AsyncMock, return_value=db):
        result = await get_affinity("u1")

    assert result == 750


@pytest.mark.asyncio
async def test_update_affinity_clamps_to_max():
    db = _mock_db()
    db.user_profiles.find_one = AsyncMock(return_value={"user_id": "u1", "affinity": 995})
    db.user_profiles.update_one = AsyncMock()

    with patch("kazusa_ai_chatbot.db.get_db", new_callable=AsyncMock, return_value=db):
        result = await update_affinity("u1", 10)

    assert result == 1000


@pytest.mark.asyncio
async def test_update_affinity_clamps_to_min():
    db = _mock_db()
    db.user_profiles.find_one = AsyncMock(return_value={"user_id": "u1", "affinity": 5})
    db.user_profiles.update_one = AsyncMock()

    with patch("kazusa_ai_chatbot.db.get_db", new_callable=AsyncMock, return_value=db):
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

    with patch("kazusa_ai_chatbot.db.get_db", new_callable=AsyncMock, return_value=db), \
         patch("kazusa_ai_chatbot.db.get_text_embedding", new_callable=AsyncMock, return_value=[0.1, 0.2, 0.3]):
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
        {"channel_id": "c1", "timestamp": "t1", "content": "keyword matched"},
    ])
    db.conversation_history.find.return_value.sort.return_value.limit.return_value = cursor

    with patch("kazusa_ai_chatbot.db.get_db", new_callable=AsyncMock, return_value=db):
        results = await db_module.search_conversation_history("keyword", method="keyword", limit=2)

    assert len(results) == 1
    assert results[0][0] == -1.0
    assert results[0][1]["content"] == "keyword matched"
    # Verify the regex filter was passed
    call_filter = db.conversation_history.find.call_args[0][0]
    assert "$regex" in call_filter["content"]


@pytest.mark.asyncio
async def test_search_conversation_history_keyword_with_filters_mocked():
    """Keyword search applies platform_channel_id and global_user_id filters."""
    db = _mock_db()
    cursor = AsyncMock()
    cursor.to_list = AsyncMock(return_value=[])
    db.conversation_history.find.return_value.sort.return_value.limit.return_value = cursor

    with patch("kazusa_ai_chatbot.db.get_db", new_callable=AsyncMock, return_value=db):
        await db_module.search_conversation_history(
            "test", method="keyword", platform_channel_id="ch1", global_user_id="u1"
        )

    call_filter = db.conversation_history.find.call_args[0][0]
    assert call_filter["platform_channel_id"] == "ch1"
    assert call_filter["global_user_id"] == "u1"


@pytest.mark.asyncio
async def test_search_conversation_history_vector_mocked():
    """Vector search uses $vectorSearch aggregation pipeline."""
    db = _mock_db()
    cursor = AsyncMock()
    cursor.to_list = AsyncMock(return_value=[
        {"channel_id": "c1", "content": "vector matched", "score": 0.95},
    ])
    db.conversation_history.aggregate = MagicMock(return_value=cursor)

    with patch("kazusa_ai_chatbot.db.get_db", new_callable=AsyncMock, return_value=db), \
         patch("kazusa_ai_chatbot.db.get_text_embedding", new_callable=AsyncMock, return_value=[0.1, 0.2, 0.3]):
        results = await db_module.search_conversation_history("test query", method="vector", limit=2)

    assert len(results) == 1
    assert results[0][0] == 0.95
    assert results[0][1]["content"] == "vector matched"
    # Verify aggregate was called with $vectorSearch pipeline
    pipeline = db.conversation_history.aggregate.call_args[0][0]
    assert "$vectorSearch" in pipeline[0]
    vs = pipeline[0]["$vectorSearch"]
    assert vs["queryVector"] == [0.1, 0.2, 0.3]
    assert vs["index"] == "conversation_history_vector_index"


@pytest.mark.asyncio
async def test_search_conversation_history_vector_with_filters_mocked():
    """Vector search adds $match stage for platform_channel_id/global_user_id post-filtering."""
    db = _mock_db()
    cursor = AsyncMock()
    cursor.to_list = AsyncMock(return_value=[])
    db.conversation_history.aggregate = MagicMock(return_value=cursor)

    with patch("kazusa_ai_chatbot.db.get_db", new_callable=AsyncMock, return_value=db), \
         patch("kazusa_ai_chatbot.db.get_text_embedding", new_callable=AsyncMock, return_value=[0.1, 0.2, 0.3]):
        await db_module.search_conversation_history(
            "test", method="vector", platform_channel_id="ch1", global_user_id="u1", limit=3
        )

    pipeline = db.conversation_history.aggregate.call_args[0][0]
    # Should contain a $match stage with both filters
    match_stages = [s for s in pipeline if "$match" in s]
    assert len(match_stages) == 1
    assert match_stages[0]["$match"]["platform_channel_id"] == "ch1"
    assert match_stages[0]["$match"]["global_user_id"] == "u1"


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
