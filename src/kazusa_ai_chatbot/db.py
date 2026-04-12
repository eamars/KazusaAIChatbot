"""MongoDB helpers and document schemas.

Each TypedDict below mirrors exactly one document shape stored in MongoDB.
They serve as a static reference for the database layout and are used as
type annotations in function signatures throughout this module.

Collections
-----------
conversation_history  → ConversationMessageDoc
user_facts            → UserFactsDoc
character_state       → CharacterStateDoc
"""

from __future__ import annotations

import logging
from typing import Any, TypedDict

from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.errors import ConnectionFailure
from pymongo.operations import SearchIndexModel

from kazusa_ai_chatbot.config import EMBEDDING_BASE_URL, EMBEDDING_MODEL, LLM_API_KEY, MONGODB_URI, MONGODB_DB_NAME
from kazusa_ai_chatbot.config import AFFINITY_DEFAULT, AFFINITY_MAX, AFFINITY_MIN
from kazusa_ai_chatbot.config import CONVERSATION_HISTORY_LIMIT
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
    facts: list[str]     # Character's diary about this user
    affinity: int        # 0–1000 affinity score (default 500)
    last_relationship_insight: str  # Character's instantaneous impression of the user
    embedding: list[float]  # dense vector for similarity search


class CharacterStateDoc(TypedDict):
    """Global character mood/state in the ``character_state`` collection.

    There is exactly **one** document with ``_id: "global"``.
    Recent events accumulate without limit.
    """

    mood: str               # e.g. "melancholic", "playful", "irritated"
    global_vibe: str        # See Cognition Layer
    reflection_summary: str # See Cognition Layer
    updated_at: str         # ISO-8601 UTC timestamp of last update


class MemoryDoc(TypedDict):
    """Memory base in the ``memory`` collection.
    """
    memory_name: str         # Name of the memory
    content: str     # memory content
    timestamp: str   # ISO-8601 UTC timestamp of when memory was created/updated
    embedding: list[float]  # dense vector for similarity search

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
    channel_id: str | None = None,
    limit: int = CONVERSATION_HISTORY_LIMIT,
    user_id: str | None = None,
    name: str | None = None,
    from_timestamp: str | None = None,
    to_timestamp: str | None = None,
) -> list[ConversationMessageDoc]:
    """Fetch the last `limit` messages for a channel (or all channels if omitted), oldest first."""
    db = await get_db()
    
    query: dict[str, Any] = {}
    if channel_id:
        query["channel_id"] = channel_id
        
    if user_id:
        query["user_id"] = user_id
    elif name:
        query["name"] = name
        
    if from_timestamp or to_timestamp:
        query["timestamp"] = {}
        if from_timestamp:
            query["timestamp"]["$gte"] = from_timestamp
        if to_timestamp:
            query["timestamp"]["$lte"] = to_timestamp

    cursor = (
        db.conversation_history
        .find(query)
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

async def get_user_profile(user_id: str) -> UserFactsDoc:
    """Retrieve user profile for a user."""
    db = await get_db()
    doc = await db.user_facts.find_one({"user_id": user_id})
    if doc is None:
        return {}
    doc.pop("_id", None)
    doc.pop("embedding", None)
    return doc


async def get_user_facts(user_id: str) -> list[str]:
    """Retrieve long-term memory facts for a user."""
    db = await get_db()
    doc = await db.user_facts.find_one({"user_id": user_id})
    if doc is None:
        return []
    return doc.get("facts", [])


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


async def update_last_relationship_insight(user_id: str, insight: str) -> None:
    """Update the last relationship insight for a user."""
    db = await get_db()
    await db.user_facts.update_one(
        {"user_id": user_id},
        {"$set": {"last_relationship_insight": insight}},
        upsert=True,
    )


async def upsert_user_facts(user_id: str, new_facts: list[str]) -> None:
    """Insert one character's view to the user to the top"""
    db = await get_db()
    existing = await get_user_facts(user_id)
    merged = list(dict.fromkeys(existing + new_facts))

    if merged:
        combined_facts_text = "\n".join(merged)
        embedding = await get_text_embedding(combined_facts_text)
    else:
        embedding = []

    await db.user_facts.update_one(
        {"user_id": user_id},
        {"$set": {"user_id": user_id, "facts": merged, "embedding": embedding}},
        upsert=True,
    )


async def overwrite_user_facts(user_id: str, facts: list[str]) -> None:
    """Overwrite all user facts with a new list, replacing existing ones."""
    db = await get_db()
    
    if facts:
        combined_facts_text = "\n".join(facts)
        embedding = await get_text_embedding(combined_facts_text)
    else:
        embedding = []

    await db.user_facts.update_one(
        {"user_id": user_id},
        {"$set": {"user_id": user_id, "facts": facts, "embedding": embedding}},
        upsert=True,
    )


async def enable_user_facts_vector_index() -> None:
    """Create a vector search index on the user_facts collection for semantic user search."""
    await enable_vector_index("user_facts", "user_facts_vector_index")


async def search_users_by_facts(
    query: str,
    limit: int = 5,
) -> list[tuple[float, UserFactsDoc]]:
    """Search for users based on semantic similarity of their accumulated facts.
    
    Args:
        query: Search query text
        limit: Maximum number of results to return
        
    Returns:
        List of (score, user_doc) tuples sorted by similarity (highest first)
    """
    query_embedding = await get_text_embedding(query)
    db = await get_db()
    collection = db.user_facts

    pipeline: list[dict[str, Any]] = [
        {
            "$vectorSearch": {
                "index": "user_facts_vector_index",
                "path": "embedding",
                "queryVector": query_embedding,
                "numCandidates": limit * 10,
                "limit": limit,
            }
        },
        {"$addFields": {"score": {"$meta": "vectorSearchScore"}}},
        {"$project": {"user_id": 1, "facts": 1, "affinity": 1, "score": 1, "_id": 0}},
    ]

    cursor = collection.aggregate(pipeline)
    docs = await cursor.to_list(length=limit)

    # Convert to (score, doc) tuples
    results = []
    for doc in docs:
        score = doc.pop("score")
        results.append((score, doc))

    return results


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
    global_vibe: str,
    reflection_summary: str,
    timestamp: str,
) -> None:
    """Update the global character state."""
    db = await get_db()
    # Fetch existing events and append
    existing = await get_character_state()
    
    # Don't update if the value is empty string
    if mood == "":
        mood = existing.get("mood", "")
    if global_vibe == "":
        global_vibe = existing.get("global_vibe", "")
    if reflection_summary == "":
        reflection_summary = existing.get("reflection_summary", "")

    await db.character_state.update_one(
        {"_id": "global"},
        {
            "$set": {
                "mood": mood,
                "global_vibe": global_vibe,
                "reflection_summary": reflection_summary,
                "updated_at": timestamp,
            }
        },
        upsert=True,
    )


# ----------------------------------------------------------------------------------------
# Memory
# Collection: memory


async def enable_memory_vector_index() -> None:
    """Create a vector search index on the memory collection for semantic memory search."""
    await enable_vector_index("memory", "memory_vector_index")


async def save_memory(
    memory_name: str,
    content: str,
    timestamp: str,
) -> None:
    """Save a memory entry with embedding to the memory collection.
    
    Args:
        memory_name: Name/identifier for the memory
        content: The memory content/text
        timestamp: ISO-8601 UTC timestamp for when the memory was created or updated
    """
    db = await get_db()
    
    # Create embedding based on "memory_name: content"
    combined_text = f"{memory_name}: {content}"
    embedding = await get_text_embedding(combined_text)
    
    await db.memory.update_one(
        {"memory_name": memory_name},
        {
            "$set": {
                "memory_name": memory_name,
                "content": content,
                "timestamp": timestamp,
                "embedding": embedding,
            }
        },
        upsert=True,
    )


async def search_memory(
    query: str,
    limit: int = 5,
    method: str = "vector",  # "keyword", "vector"
) -> list[tuple[float, MemoryDoc]]:
    """
    Search memory collection using keyword or vector relevance.
    
    Args:
        query: The search query string.
        limit: Maximum number of results.
        method: "keyword" for regex text search, "vector" for semantic search.
        
    Returns a list of tuples (similarity_score, memory_doc).
    Keyword search results always have similarity_score of -1.
    """
    db = await get_db()
    collection = db.memory

    if method == "keyword":
        base_filter: dict[str, Any] = {
            "$or": [
                {"memory_name": {"$regex": query, "$options": "i"}},
                {"content": {"$regex": query, "$options": "i"}}
            ]
        }

        cursor = collection.find(base_filter).limit(limit)
        docs = await cursor.to_list(length=limit)
        # Remove embedding field from results
        for doc in docs:
            doc.pop("embedding", None)
        return [(-1.0, doc) for doc in docs]

    # method == "vector"
    query_embedding = await get_text_embedding(query)
    index_name = "memory_vector_index"

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
        {"$unset": "embedding"},
    ]

    cursor = collection.aggregate(pipeline)
    docs = await cursor.to_list(length=limit)
    return [(doc.pop("score", 0.0), doc) for doc in docs]

