"""MongoDB helpers and document schemas.

Each TypedDict below mirrors exactly one document shape stored in MongoDB.
They serve as a static reference for the database layout and are used as
type annotations in function signatures throughout this module.

Collections
-----------
conversation_history  → ConversationMessageDoc
user_facts            → UserFactsDoc
character_state       → CharacterStateDoc
lore                  → LoreDoc  (used by vector search)
"""

from __future__ import annotations

import logging
from typing import Any, TypedDict

import numpy as np

from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.errors import ConnectionFailure

from bot.config import MONGODB_URI, MONGODB_DB_NAME

logger = logging.getLogger(__name__)


# ── Document schemas (TypedDict) ──────────────────────────────────────


class ConversationMessageDoc(TypedDict):
    """A single chat message in the ``conversation_history`` collection.

    Indexed by ``channel_id`` + ``timestamp`` (descending) for efficient
    retrieval of the most recent messages in a channel.
    """

    channel_id: str      # Discord channel ID
    role: str            # "user" | "assistant"
    user_id: str         # Discord user/bot ID — unique message source
    name: str            # display name (for prompt formatting)
    content: str         # message text
    timestamp: str       # ISO-8601 UTC timestamp


class UserFactsDoc(TypedDict):
    """Long-term memory about a single user in the ``user_facts`` collection.

    Keyed by ``user_id``.  Facts are deduplicated on upsert.
    """

    user_id: str         # Discord user ID (unique key)
    facts: list[str]     # extracted facts about this user
    affinity: int        # 0–1000 affinity score (default 500)


class CharacterStateDoc(TypedDict):
    """Global character mood/state in the ``character_state`` collection.

    There is exactly **one** document with ``_id: "global"``.
    Recent events are capped at 10 (sliding window).
    """

    mood: str               # e.g. "melancholic", "playful", "irritated"
    emotional_tone: str     # e.g. "warm", "guarded", "teasing"
    recent_events: list[str]  # short summaries, max 10
    updated_at: str         # ISO-8601 UTC timestamp of last update


class LoreDoc(TypedDict):
    """A lore entry in the ``lore`` collection (used by vector search).

    The ``embedding`` field is indexed by a MongoDB Atlas vector search index.
    """

    text: str                  # lore content
    source: str                # provenance tag, e.g. "lore/events", "lore/npcs"
    embedding: list[float]     # dense vector for similarity search

_client: AsyncIOMotorClient | None = None
_db = None


async def get_db():
    """Return the async MongoDB database handle, creating the client on first call."""
    global _client, _db
    if _db is None:
        _client = AsyncIOMotorClient(MONGODB_URI)
        _db = _client[MONGODB_DB_NAME]
        # Verify connectivity
        try:
            await _client.admin.command("ping")
            logger.info("Connected to MongoDB at %s", MONGODB_URI)
        except ConnectionFailure:
            logger.error("Failed to connect to MongoDB at %s", MONGODB_URI)
            raise
    return _db


async def close_db():
    """Close the MongoDB client connection."""
    global _client, _db
    if _client is not None:
        _client.close()
        _client = None
        _db = None


# ── Collection helpers ──────────────────────────────────────────────

async def get_conversation_history(
    channel_id: str, limit: int = 20
) -> list[ConversationMessageDoc]:
    """Fetch the last `limit` messages for a channel, oldest first."""
    db = await get_db()
    cursor = (
        db.conversation_history
        .find({"channel_id": channel_id})
        .sort("timestamp", -1)
        .limit(limit)
    )
    docs = await cursor.to_list(length=limit)
    docs.reverse()  # oldest first
    return docs


async def save_message(
    channel_id: str,
    role: str,
    user_id: str,
    name: str,
    content: str,
    timestamp: str,
) -> None:
    """Persist a single message to conversation history."""
    db = await get_db()
    doc: ConversationMessageDoc = {
        "channel_id": channel_id,
        "role": role,
        "user_id": user_id,
        "name": name,
        "content": content,
        "timestamp": timestamp,
    }
    await db.conversation_history.insert_one(doc)


async def get_user_facts(user_id: str) -> list[str]:
    """Retrieve long-term memory facts for a user."""
    db = await get_db()
    doc = await db.user_facts.find_one({"user_id": user_id})
    if doc is None:
        return []
    return doc.get("facts", [])


AFFINITY_DEFAULT = 500
AFFINITY_MIN = 0
AFFINITY_MAX = 1000


async def get_affinity(user_id: str) -> int:
    """Return the affinity score for a user (0–1000, default 500)."""
    db = await get_db()
    doc = await db.user_facts.find_one({"user_id": user_id})
    if doc is None:
        return AFFINITY_DEFAULT
    return doc.get("affinity", AFFINITY_DEFAULT)


async def update_affinity(user_id: str, delta: int) -> int:
    """Apply a delta to the user's affinity score, clamped to 0–1000.

    Creates the user_facts doc if it doesn't exist yet.
    Returns the new affinity value.
    """
    current = await get_affinity(user_id)
    new_value = max(AFFINITY_MIN, min(AFFINITY_MAX, current + delta))
    db = await get_db()
    await db.user_facts.update_one(
        {"user_id": user_id},
        {"$set": {"affinity": new_value}},
        upsert=True,
    )
    return new_value


async def upsert_user_facts(user_id: str, new_facts: list[str]) -> None:
    """Add new facts to a user's memory, deduplicating."""
    db = await get_db()
    existing = await get_user_facts(user_id)
    merged = list(dict.fromkeys(existing + new_facts))  # deduplicate, preserve order
    await db.user_facts.update_one(
        {"user_id": user_id},
        {"$set": {"user_id": user_id, "facts": merged}},
        upsert=True,
    )


async def get_character_state() -> CharacterStateDoc | dict:
    """Retrieve the global character state (mood, tone, recent events).

    There is only ONE character state doc — it is shared across all channels.
    """
    db = await get_db()
    doc = await db.character_state.find_one({"_id": "global"})
    if doc is None:
        return {}
    doc.pop("_id", None)
    return doc


async def upsert_character_state(
    mood: str,
    emotional_tone: str,
    recent_events: list[str],
    timestamp: str,
) -> None:
    """Update the global character state. Keeps last 10 recent events."""
    db = await get_db()
    # Fetch existing events and append, keeping a sliding window
    existing = await get_character_state()
    old_events = existing.get("recent_events", [])
    merged_events = (old_events + recent_events)[-10:]  # keep last 10

    await db.character_state.update_one(
        {"_id": "global"},
        {
            "$set": {
                "mood": mood,
                "emotional_tone": emotional_tone,
                "recent_events": merged_events,
                "updated_at": timestamp,
            }
        },
        upsert=True,
    )


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    va, vb = np.array(a), np.array(b)
    norm_a, norm_b = np.linalg.norm(va), np.linalg.norm(vb)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(va, vb) / (norm_a * norm_b))


async def vector_search(
    collection_name: str,
    query_embedding: list[float],
    top_k: int = 3,
) -> list[dict[str, Any]]:
    """Code-side vector search: fetch all docs with embeddings and rank by cosine similarity.

    NOTE: This works on vanilla MongoDB (no Atlas required) but does not
    scale well beyond a few thousand documents.
    """
    db = await get_db()
    cursor = db[collection_name].find(
        {"embedding": {"$exists": True}},
        {"text": 1, "source": 1, "embedding": 1, "_id": 0},
    )
    docs = await cursor.to_list(length=None)

    scored = []
    for doc in docs:
        emb = doc.get("embedding")
        if not emb:
            continue
        score = _cosine_similarity(query_embedding, emb)
        scored.append({"text": doc.get("text", ""), "source": doc.get("source", "unknown"), "score": score})

    scored.sort(key=lambda d: d["score"], reverse=True)
    return scored[:top_k]
