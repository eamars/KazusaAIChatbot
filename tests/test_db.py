"""Tests for DB helpers — mocked unit tests + live MongoDB integration tests."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import db as db_module
from db import (
    AFFINITY_DEFAULT,
    close_db,
    get_affinity,
    get_character_state,
    get_conversation_history,
    get_db,
    get_user_facts,
    save_message,
    update_affinity,
    upsert_character_state,
    upsert_user_facts,
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

    with patch("db.get_db", new_callable=AsyncMock, return_value=db):
        result = await get_conversation_history("chan_1", limit=10)

    # Should be reversed (oldest first)
    assert result[0]["content"] == "Hi"
    assert result[1]["content"] == "Hello"


@pytest.mark.asyncio
async def test_get_user_facts_found():
    db = _mock_db()
    db.user_facts.find_one = AsyncMock(return_value={"user_id": "u1", "facts": ["fact1", "fact2"]})

    with patch("db.get_db", new_callable=AsyncMock, return_value=db):
        result = await get_user_facts("u1")

    assert result == ["fact1", "fact2"]


@pytest.mark.asyncio
async def test_get_user_facts_not_found():
    db = _mock_db()
    db.user_facts.find_one = AsyncMock(return_value=None)

    with patch("db.get_db", new_callable=AsyncMock, return_value=db):
        result = await get_user_facts("u1")

    assert result == []


@pytest.mark.asyncio
async def test_upsert_user_facts_deduplicates():
    db = _mock_db()
    db.user_facts.find_one = AsyncMock(return_value={"user_id": "u1", "facts": ["fact1", "fact2"]})
    db.user_facts.update_one = AsyncMock()

    with patch("db.get_db", new_callable=AsyncMock, return_value=db):
        await upsert_user_facts("u1", ["fact2", "fact3"])

    call_args = db.user_facts.update_one.call_args
    merged = call_args[1]["upsert"] if "upsert" in call_args[1] else None
    # Check the $set payload
    set_payload = call_args[0][1]["$set"]
    assert set_payload["facts"] == ["fact1", "fact2", "fact3"]


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

    with patch("db.get_db", new_callable=AsyncMock, return_value=db):
        result = await get_character_state()

    assert result["mood"] == "calm"
    assert "_id" not in result


@pytest.mark.asyncio
async def test_get_character_state_not_found():
    db = _mock_db()
    db.character_state.find_one = AsyncMock(return_value=None)

    with patch("db.get_db", new_callable=AsyncMock, return_value=db):
        result = await get_character_state()

    assert result == {}


@pytest.mark.asyncio
async def test_upsert_character_state_merges_events():
    db = _mock_db()
    # Existing state has 2 events
    db.character_state.find_one = AsyncMock(return_value={
        "_id": "global",
        "mood": "old",
        "recent_events": ["event1", "event2"],
    })
    db.character_state.update_one = AsyncMock()

    with patch("db.get_db", new_callable=AsyncMock, return_value=db):
        await upsert_character_state("happy", "warm", ["event3"], "t2")

    call_args = db.character_state.update_one.call_args
    set_payload = call_args[0][1]["$set"]
    assert set_payload["mood"] == "happy"
    assert set_payload["emotional_tone"] == "warm"
    assert set_payload["recent_events"] == ["event1", "event2", "event3"]


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
    await save_message("test_chan", "user", "user_100", "Alice", "Hello!", "2026-01-01T00:00:01Z")
    await save_message("test_chan", "assistant", "bot_001", "bot", "Hi Alice!", "2026-01-01T00:00:02Z")
    await save_message("test_chan", "user", "user_100", "Alice", "How are you?", "2026-01-01T00:00:03Z")

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
        await save_message("test_chan", "user", "user_100", "Alice", f"msg_{i}", f"2026-01-01T00:00:{i:02d}Z")

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
    await save_message("chan_a", "user", "user_100", "Alice", "In channel A", "2026-01-01T00:00:01Z")
    await save_message("chan_b", "user", "user_200", "Bob", "In channel B", "2026-01-01T00:00:02Z")

    history_a = await get_conversation_history("chan_a", limit=10)
    history_b = await get_conversation_history("chan_b", limit=10)

    assert len(history_a) == 1
    assert history_a[0]["content"] == "In channel A"
    assert len(history_b) == 1
    assert history_b[0]["content"] == "In channel B"


# ── User facts ────────────────────────────────────────────────────────


@live_db
@pytest.mark.asyncio
async def test_live_user_facts_empty(live_test_db):
    """No facts for a new user."""
    facts = await get_user_facts("nonexistent_user")
    assert facts == []


@live_db
@pytest.mark.asyncio
async def test_live_upsert_and_get_user_facts(live_test_db):
    """Store facts and retrieve them."""
    await upsert_user_facts("user_1", ["Likes swords", "Goes by Commander"])
    facts = await get_user_facts("user_1")

    assert "Likes swords" in facts
    assert "Goes by Commander" in facts
    assert len(facts) == 2


@live_db
@pytest.mark.asyncio
async def test_live_user_facts_deduplication(live_test_db):
    """Upserting duplicate facts should not create duplicates."""
    await upsert_user_facts("user_1", ["Likes swords", "Goes by Commander"])
    await upsert_user_facts("user_1", ["Goes by Commander", "Allied with North"])

    facts = await get_user_facts("user_1")
    assert facts == ["Likes swords", "Goes by Commander", "Allied with North"]


@live_db
@pytest.mark.asyncio
async def test_live_user_facts_isolation(live_test_db):
    """Different users have separate fact stores."""
    await upsert_user_facts("user_a", ["Fact A"])
    await upsert_user_facts("user_b", ["Fact B"])

    assert await get_user_facts("user_a") == ["Fact A"]
    assert await get_user_facts("user_b") == ["Fact B"]


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
    await upsert_character_state("happy", "warm", ["Met an old friend"], "2026-01-01T00:00:00Z")

    state = await get_character_state()
    assert state["mood"] == "happy"
    assert state["emotional_tone"] == "warm"
    assert state["recent_events"] == ["Met an old friend"]
    assert state["updated_at"] == "2026-01-01T00:00:00Z"
    assert "_id" not in state


@live_db
@pytest.mark.asyncio
async def test_live_character_state_update_overwrites_mood(live_test_db):
    """Updating character state replaces mood and tone."""
    await upsert_character_state("happy", "warm", ["event1"], "t1")
    await upsert_character_state("sad", "guarded", ["event2"], "t2")

    state = await get_character_state()
    assert state["mood"] == "sad"
    assert state["emotional_tone"] == "guarded"


@live_db
@pytest.mark.asyncio
async def test_live_character_state_merges_recent_events(live_test_db):
    """Recent events accumulate across updates."""
    await upsert_character_state("calm", "neutral", ["event1", "event2"], "t1")
    await upsert_character_state("alert", "tense", ["event3"], "t2")

    state = await get_character_state()
    assert state["recent_events"] == ["event1", "event2", "event3"]


@live_db
@pytest.mark.asyncio
async def test_live_character_state_sliding_window(live_test_db):
    """Recent events are capped at 10 (sliding window)."""
    events_batch1 = [f"old_event_{i}" for i in range(8)]
    await upsert_character_state("calm", "neutral", events_batch1, "t1")

    events_batch2 = [f"new_event_{i}" for i in range(5)]
    await upsert_character_state("alert", "tense", events_batch2, "t2")

    state = await get_character_state()
    assert len(state["recent_events"]) == 10
    # Oldest events should have been dropped
    assert state["recent_events"][0] == "old_event_3"
    assert state["recent_events"][-1] == "new_event_4"


# ── Affinity (mocked) ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_affinity_default_for_unknown_user():
    db = _mock_db()
    db.user_facts.find_one = AsyncMock(return_value=None)

    with patch("db.get_db", new_callable=AsyncMock, return_value=db):
        result = await get_affinity("unknown_user")

    assert result == AFFINITY_DEFAULT


@pytest.mark.asyncio
async def test_get_affinity_returns_stored_value():
    db = _mock_db()
    db.user_facts.find_one = AsyncMock(return_value={"user_id": "u1", "affinity": 750})

    with patch("db.get_db", new_callable=AsyncMock, return_value=db):
        result = await get_affinity("u1")

    assert result == 750


@pytest.mark.asyncio
async def test_update_affinity_clamps_to_max():
    db = _mock_db()
    db.user_facts.find_one = AsyncMock(return_value={"user_id": "u1", "affinity": 995})
    db.user_facts.update_one = AsyncMock()

    with patch("db.get_db", new_callable=AsyncMock, return_value=db):
        result = await update_affinity("u1", 10)

    assert result == 1000


@pytest.mark.asyncio
async def test_update_affinity_clamps_to_min():
    db = _mock_db()
    db.user_facts.find_one = AsyncMock(return_value={"user_id": "u1", "affinity": 5})
    db.user_facts.update_one = AsyncMock()

    with patch("db.get_db", new_callable=AsyncMock, return_value=db):
        result = await update_affinity("u1", -20)

    assert result == 0


# ── Affinity (live) ──────────────────────────────────────────────────


@live_db
@pytest.mark.asyncio
async def test_live_affinity_default_for_new_user(live_test_db):
    """New users start at AFFINITY_DEFAULT (500)."""
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
    await update_affinity("user_low", -600)  # 500 - 600 → 0
    assert await get_affinity("user_low") == 0

    # Clamp at 1000
    await update_affinity("user_high", 600)  # 500 + 600 → 1000
    assert await get_affinity("user_high") == 1000


@live_db
@pytest.mark.asyncio
async def test_live_affinity_coexists_with_facts(live_test_db):
    """Affinity and facts live on the same user_facts doc without clobbering."""
    await upsert_user_facts("user_combo", ["Likes swords"])
    await update_affinity("user_combo", 50)

    facts = await get_user_facts("user_combo")
    affinity = await get_affinity("user_combo")

    assert facts == ["Likes swords"]
    assert affinity == AFFINITY_DEFAULT + 50
