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
from pymongo.operations import SearchIndexModel

from kazusa_ai_chatbot.config import MONGODB_URI, MONGODB_DB_NAME, EMBEDDING_BASE_URL, EMBEDDING_MODEL, LLM_API_KEY, RAG_TOP_K
from openai import AsyncOpenAI


logger = logging.getLogger(__name__)


# Lazily initialised embedding client
_embed_client: AsyncOpenAI | None = None


def _get_embed_client() -> AsyncOpenAI:
    global _embed_client
    if _embed_client is None:
        _embed_client = AsyncOpenAI(
            base_url=EMBEDDING_BASE_URL,
            api_key=LLM_API_KEY,
        )
    return _embed_client


async def get_text_embedding(text: str) -> list[float]:
    """Get embedding vector for a single text string."""
    client = _get_embed_client()
    resp = await client.embeddings.create(input=[text], model=EMBEDDING_MODEL)
    return resp.data[0].embedding


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
    embedding: list[float]  # dense vector for similarity search


class UserFactsDoc(TypedDict):
    """Long-term memory about a single user in the ``user_facts`` collection.

    Keyed by ``user_id``.  Facts are deduplicated on upsert.
    """

    user_id: str         # Discord user ID (unique key)
    facts: list[str]     # extracted facts about this user
    affinity: int        # 0–1000 affinity score (default 500)
    embedding: list[float]  # dense vector for similarity search


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


async def vector_search(
    collection_name: str,
    query_embedding: list[float],
    top_k: int = 3,
    index_name: str = "default",
) -> list[dict[str, Any]]:
    """Atlas $vectorSearch: run a vector similarity query via aggregation pipeline.

    Requires a vectorSearch index on the target collection's ``embedding`` field.
    Returns up to ``top_k`` documents with ``text``, ``source``, and ``score`` keys.
    """
    db = await get_db()
    collection = db[collection_name]

    pipeline: list[dict[str, Any]] = [
        {
            "$vectorSearch": {
                "index": index_name,
                "path": "embedding",
                "queryVector": query_embedding,
                "numCandidates": top_k * 10,
                "limit": top_k,
            }
        },
        {"$addFields": {"score": {"$meta": "vectorSearchScore"}}},
        {"$project": {"text": 1, "source": 1, "score": 1, "_id": 0}},
    ]

    cursor = collection.aggregate(pipeline)
    docs = await cursor.to_list(length=top_k)
    return docs


async def enable_vector_index(collection_name: str, index_name: str) -> None:
    """Create a vector search index on the specified collection for semantic search."""
    db = await get_db()
    collection = db[collection_name]

    # Check if index exists
    try:
        async for index in collection.list_search_indexes():
            if index.get("name") == index_name:
                logger.info("Vector search index '%s' already exists.", index_name)
                return
    except Exception as e:
        logger.debug("Could not list search indexes (might not exist yet or not supported): %s", e)

    logger.info("Vector search index '%s' not found. Creating...", index_name)

    # Determine dimension from the current text embedding model
    sample_embedding = await get_text_embedding("test")
    num_dimensions = len(sample_embedding)

    search_index_model = SearchIndexModel(
        definition={
            "fields": [
                {
                    "type": "vector",
                    "path": "embedding",
                    "numDimensions": num_dimensions,
                    "similarity": "cosine",
                }
            ]
        },
        name=index_name,
        type="vectorSearch",
    )

    try:
        await collection.create_search_index(search_index_model)
        logger.info("Successfully created vector search index '%s' with %d dimensions.", index_name, num_dimensions)
    except Exception as e:
        logger.error("Failed to create vector search index '%s': %s", index_name, e)
        raise



# ----------------------------------------------------------------------------------------
# Conversation history
# Collection: conversation_history

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


async def search_conversation_history(
    query: str, 
    channel_id: str | None = None,
    user_id: str | None = None,
    limit: int = 5,
    method: str = "vector"  # "keyword", "vector"
) -> list[tuple[float, ConversationMessageDoc]]:
    """
    Search conversation history using keyword or vector relevance.
    
    Args:
        query: The search query string.
        channel_id: Optional channel filter.
        user_id: Optional user filter.
        limit: Maximum number of results.
        method: "keyword" for regex text search, "vector" for semantic search.
        
    Returns a list of tuples (similarity_score, message_doc).
    Keyword search results always have similarity_score of -1.
    """
    db = await get_db()
    collection = db.conversation_history

    if method == "keyword":
        base_filter: dict[str, Any] = {"content": {"$regex": query, "$options": "i"}}
        if channel_id:
            base_filter["channel_id"] = channel_id
        if user_id:
            base_filter["user_id"] = user_id

        cursor = collection.find(base_filter).sort("timestamp", -1).limit(limit)
        docs = await cursor.to_list(length=limit)
        return [(-1.0, doc) for doc in docs]

    # method == "vector"
    query_embedding = await get_text_embedding(query)
    index_name = "conversation_history_vector_index"

    pipeline: list[dict[str, Any]] = [
        {
            "$vectorSearch": {
                "index": index_name,
                "path": "embedding",
                "queryVector": query_embedding,
                "numCandidates": limit * 10,
                "limit": limit,
            }
        },
        {"$addFields": {"score": {"$meta": "vectorSearchScore"}}},
    ]

    # Apply post-filters for channel_id / user_id
    match_filter: dict[str, Any] = {}
    if channel_id:
        match_filter["channel_id"] = channel_id
    if user_id:
        match_filter["user_id"] = user_id
    if match_filter:
        pipeline.append({"$match": match_filter})

    pipeline.append({"$limit": limit})
    pipeline.append({"$unset": "embedding"})

    cursor = collection.aggregate(pipeline)
    docs = await cursor.to_list(length=limit)
    return [(doc.pop("score", 0.0), doc) for doc in docs]


async def save_conversation(doc: ConversationMessageDoc) -> None:
    """Persist a single message to conversation history, generating its embedding."""
    db = await get_db()
    
    if "embedding" not in doc or not doc.get("embedding"):
        doc["embedding"] = await get_text_embedding(doc["content"])
        
    await db.conversation_history.insert_one(doc)


# ----------------------------------------------------------------------------------------
# User Fact
# Collection: user_facts

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



# ----------------------------------------------------------------------------------------


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
