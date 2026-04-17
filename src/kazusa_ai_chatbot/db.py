"""MongoDB helpers and document schemas.

Each TypedDict below mirrors exactly one document shape stored in MongoDB.
They serve as a static reference for the database layout and are used as
type annotations in function signatures throughout this module.

Collections
-----------
conversation_history  → ConversationMessageDoc
user_profiles         → UserProfileDoc
character_state       → CharacterProfileDoc
memory                → MemoryDoc
"""

from __future__ import annotations

import logging
import uuid
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


class AttachmentDoc(TypedDict, total=False):
    """Multimedia attachment embedded in a conversation message."""
    media_type: str       # MIME type: "image/png", "audio/ogg", etc.
    url: str              # External URL (CDN, S3, etc.) — preferred for large files
    base64_data: str      # Inline base64 — for small attachments only
    description: str      # Alt-text / transcription / OCR summary
    size_bytes: int       # File size


class ConversationMessageDoc(TypedDict, total=False):
    """A single chat message in the ``conversation_history`` collection.

    Indexed by ``(platform, platform_channel_id, timestamp)`` (descending)
    for efficient retrieval of the most recent messages in a channel.
    """

    platform: str              # "discord" | "qq" | "wechat" | "whatsapp" | "telegram" | "system"
    platform_channel_id: str   # Original channel/group ID from the platform
    channel_type: str          # "group" | "private" | "system"
    role: str                  # "user" | "assistant"
    platform_user_id: str      # Original user/bot ID from the platform
    global_user_id: str        # Our internal UUID key
    display_name: str          # Display name at time of message
    content: str               # Text content
    content_type: str          # "text" | "image" | "voice" | "mixed"
    attachments: list[AttachmentDoc]  # Images, voice, files
    timestamp: str             # ISO-8601 UTC timestamp
    embedding: list[float]     # Dense vector (on text content only)


class PlatformAccountDoc(TypedDict, total=False):
    """A linked platform account within a UserProfileDoc."""
    platform: str             # "discord" | "qq" | ...
    platform_user_id: str     # Original ID on that platform
    display_name: str         # Last known display name
    linked_at: str            # ISO-8601 when this account was linked


class UserProfileDoc(TypedDict, total=False):
    """Long-term memory about a single user in the ``user_profiles`` collection.

    Keyed by ``global_user_id`` (UUID4).  Facts are deduplicated on upsert.
    """

    global_user_id: str                    # UUID4 — our internal unique key
    platform_accounts: list[PlatformAccountDoc]  # All linked accounts
    suspected_aliases: list[str]           # Other global_user_ids suspected to be same person
    facts: list[str]                       # Character's diary about this user
    affinity: int                          # 0–1000 affinity score (default 500)
    last_relationship_insight: str         # Character's instantaneous impression of the user
    embedding: list[float]                 # Dense vector for similarity search


class CharacterProfileDoc(TypedDict, total=False):
    """All fields of the singleton ``_id: "global"`` document in
    the ``character_state`` collection.

    Both personality profile fields **and** runtime state fields live
    at the top level.  The schema is intentionally open-ended
    (``total=False``) so new fields can be added without migration.
    """

    # ── personality profile ────────────────────────────────────────
    name: str
    description: str
    gender: str
    age: int
    birthday: str
    tone: str
    speech_patterns: str
    backstory: str
    personality_brief: dict
    boundary_profile: dict

    # ── runtime state ─────────────────────────────────────────────
    mood: str               # e.g. "melancholic", "playful", "irritated"
    global_vibe: str        # See Cognition Layer
    reflection_summary: str # See Cognition Layer
    updated_at: str         # ISO-8601 UTC timestamp of last update


class MemoryDoc(TypedDict, total=False):
    """Memory base in the ``memory`` collection.
    """
    memory_name: str                # Name of the memory
    content: str                    # memory content
    source_global_user_id: str      # UUID4 of the user who triggered this memory (empty for non-user-specific)
    timestamp: str                  # ISO-8601 UTC timestamp of when memory was created/updated
    embedding: list[float]          # dense vector for similarity search


class ScheduledEventDoc(TypedDict, total=False):
    """A scheduled future event in the ``scheduled_events`` collection.

    Used by the scheduler to persist pending jobs across restarts.
    """
    event_id: str               # UUID4
    event_type: str             # "followup_message" | "mood_decay" | "reflection" | ...
    target_platform: str        # Platform to deliver on
    target_channel_id: str      # Channel/group to deliver to
    target_global_user_id: str  # User the event relates to
    payload: dict               # Event-specific data (message text, etc.)
    scheduled_at: str           # ISO-8601 when to fire
    created_at: str             # ISO-8601 when the event was created
    status: str                 # "pending" | "running" | "completed" | "failed"


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


async def db_bootstrap() -> None:
    """Ensure all required collections and indexes exist.

    Called once at service startup.  Safe to call repeatedly — every
    operation is idempotent (create-if-not-exists).
    """
    db = await get_db()
    existing = set(await db.list_collection_names())

    required_collections = [
        "conversation_history",
        "user_profiles",
        "character_state",
        "memory",
        "scheduled_events",
    ]
    for name in required_collections:
        if name not in existing:
            await db.create_collection(name)
            logger.info("Created collection '%s'", name)
        else:
            logger.debug("Collection '%s' already exists", name)

    # Seed mandatory singleton documents
    existing_state = await db.character_state.find_one({"_id": "global"})
    if existing_state is None:
        from datetime import datetime, timezone
        await db.character_state.insert_one({
            "_id": "global",
            "mood": "neutral",
            "global_vibe": "",
            "reflection_summary": "",
            "updated_at": datetime.now(timezone.utc).isoformat(),
        })
        logger.info("Seeded default character_state document")

    # Standard indexes (idempotent — MongoDB ignores duplicates)
    await db.conversation_history.create_index(
        [("platform", 1), ("platform_channel_id", 1), ("timestamp", -1)],
        name="conv_platform_channel_ts",
    )
    await db.user_profiles.create_index(
        "global_user_id", unique=True, name="user_global_id_unique",
    )
    await db.scheduled_events.create_index(
        "event_id", unique=True, name="event_id_unique",
    )
    await db.scheduled_events.create_index(
        [("status", 1), ("scheduled_at", 1)], name="event_status_scheduled",
    )
    await db.memory.create_index(
        "memory_name", name="memory_name_idx",
    )
    await db.memory.create_index(
        "source_global_user_id", name="memory_source_user_idx",
    )

    # Vector search indexes (best-effort — requires Atlas)
    try:
        await enable_vector_index("conversation_history", "conversation_history_vector_index")
    except Exception:
        logger.warning("Could not create conversation_history vector index (requires Atlas)")
    try:
        await enable_vector_index("user_profiles", "user_facts_vector_index")
    except Exception:
        logger.warning("Could not create user_profiles vector index (requires Atlas)")
    try:
        await enable_vector_index("memory", "memory_vector_index")
    except Exception:
        logger.warning("Could not create memory vector index (requires Atlas)")

    logger.info("Database bootstrap complete")


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
    platform: str | None = None,
    platform_channel_id: str | None = None,
    limit: int = CONVERSATION_HISTORY_LIMIT,
    global_user_id: str | None = None,
    display_name: str | None = None,
    from_timestamp: str | None = None,
    to_timestamp: str | None = None,
) -> list[ConversationMessageDoc]:
    """Fetch the last `limit` messages for a channel (or all channels if omitted), oldest first."""
    db = await get_db()
    
    query: dict[str, Any] = {}
    if platform:
        query["platform"] = platform
    if platform_channel_id:
        query["platform_channel_id"] = platform_channel_id
        
    if global_user_id:
        query["global_user_id"] = global_user_id
    elif display_name:
        query["display_name"] = display_name
        
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
    platform: str | None = None,
    platform_channel_id: str | None = None,
    global_user_id: str | None = None,
    limit: int = 5,
    method: str = "vector"  # "keyword", "vector"
) -> list[tuple[float, ConversationMessageDoc]]:
    """
    Search conversation history using keyword or vector relevance.
    
    Args:
        query: The search query string.
        platform: Optional platform filter ("discord", "qq", etc.).
        platform_channel_id: Optional channel filter.
        global_user_id: Optional user filter (internal UUID).
        limit: Maximum number of results.
        method: "keyword" for regex text search, "vector" for semantic search.
        
    Returns a list of tuples (similarity_score, message_doc).
    Keyword search results always have similarity_score of -1.
    """
    db = await get_db()
    collection = db.conversation_history

    if method == "keyword":
        base_filter: dict[str, Any] = {"content": {"$regex": query, "$options": "i"}}
        if platform:
            base_filter["platform"] = platform
        if platform_channel_id:
            base_filter["platform_channel_id"] = platform_channel_id
        if global_user_id:
            base_filter["global_user_id"] = global_user_id

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

    # Apply post-filters
    match_filter: dict[str, Any] = {}
    if platform:
        match_filter["platform"] = platform
    if platform_channel_id:
        match_filter["platform_channel_id"] = platform_channel_id
    if global_user_id:
        match_filter["global_user_id"] = global_user_id
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
    
    # Default optional fields
    doc.setdefault("content_type", "text")
    doc.setdefault("attachments", [])
    doc.setdefault("channel_type", "group")
    
    if "embedding" not in doc or not doc.get("embedding"):
        doc["embedding"] = await get_text_embedding(doc.get("content", ""))
        
    await db.conversation_history.insert_one(doc)


# ----------------------------------------------------------------------------------------
# User identity resolution
# Collection: user_profiles

async def resolve_global_user_id(
    platform: str,
    platform_user_id: str,
    display_name: str = "",
) -> str:
    """Look up or auto-create a global_user_id for a platform account.

    If no matching (platform, platform_user_id) exists, a new UserProfileDoc
    is created with a fresh UUID4 and returned.
    """
    db = await get_db()
    doc = await db.user_profiles.find_one(
        {"platform_accounts": {"$elemMatch": {"platform": platform, "platform_user_id": platform_user_id}}}
    )
    if doc is not None:
        return doc["global_user_id"]

    # Auto-create
    from datetime import datetime, timezone
    new_id = str(uuid.uuid4())
    new_profile: UserProfileDoc = {
        "global_user_id": new_id,
        "platform_accounts": [{
            "platform": platform,
            "platform_user_id": platform_user_id,
            "display_name": display_name,
            "linked_at": datetime.now(timezone.utc).isoformat(),
        }],
        "suspected_aliases": [],
        "facts": [],
        "affinity": AFFINITY_DEFAULT,
        "last_relationship_insight": "",
        "embedding": [],
    }
    await db.user_profiles.insert_one(new_profile)
    logger.info("Created new user profile %s for %s/%s", new_id, platform, platform_user_id)
    return new_id


async def link_platform_account(
    global_user_id: str,
    platform: str,
    platform_user_id: str,
    display_name: str = "",
) -> None:
    """Add a platform account to an existing user profile."""
    from datetime import datetime, timezone
    db = await get_db()
    await db.user_profiles.update_one(
        {"global_user_id": global_user_id},
        {"$addToSet": {"platform_accounts": {
            "platform": platform,
            "platform_user_id": platform_user_id,
            "display_name": display_name,
            "linked_at": datetime.now(timezone.utc).isoformat(),
        }}},
    )


async def add_suspected_alias(
    global_user_id: str,
    other_global_user_id: str,
) -> None:
    """Record a suspected cross-platform alias between two users."""
    db = await get_db()
    await db.user_profiles.update_one(
        {"global_user_id": global_user_id},
        {"$addToSet": {"suspected_aliases": other_global_user_id}},
    )
    await db.user_profiles.update_one(
        {"global_user_id": other_global_user_id},
        {"$addToSet": {"suspected_aliases": global_user_id}},
    )


# ----------------------------------------------------------------------------------------
# User profile
# Collection: user_profiles

async def get_user_profile(global_user_id: str) -> UserProfileDoc:
    """Retrieve user profile for a user."""
    db = await get_db()
    doc = await db.user_profiles.find_one({"global_user_id": global_user_id})
    if doc is None:
        return {}
    doc.pop("_id", None)
    doc.pop("embedding", None)
    return doc


async def create_user_profile(user_profile: UserProfileDoc) -> None:
    """Create a user profile for a user."""
    db = await get_db()
    facts = user_profile.get("facts", [])
    if facts:
        user_profile["embedding"] = await get_text_embedding("\n".join(facts))
    else:
        user_profile["embedding"] = []
    user_profile.setdefault("global_user_id", str(uuid.uuid4()))
    user_profile.setdefault("platform_accounts", [])
    user_profile.setdefault("suspected_aliases", [])
    user_profile.setdefault("affinity", AFFINITY_DEFAULT)
    user_profile.setdefault("last_relationship_insight", "")
    await db.user_profiles.insert_one(user_profile)


async def get_user_facts(global_user_id: str) -> list[str]:
    """Retrieve long-term memory facts for a user."""
    db = await get_db()
    doc = await db.user_profiles.find_one({"global_user_id": global_user_id})
    if doc is None:
        return []
    return doc.get("facts", [])


async def get_affinity(global_user_id: str) -> int:
    """Return the affinity score for a user (0–1000, default 500)."""
    db = await get_db()
    doc = await db.user_profiles.find_one({"global_user_id": global_user_id})
    if doc is None:
        return AFFINITY_DEFAULT
    return doc.get("affinity", AFFINITY_DEFAULT)


async def update_affinity(global_user_id: str, delta: int) -> int:
    """Apply a delta to the user's affinity score, clamped to 0–1000.

    Creates the user_profiles doc if it doesn't exist yet.
    Returns the new affinity value.
    """
    current = await get_affinity(global_user_id)
    new_value = max(AFFINITY_MIN, min(AFFINITY_MAX, current + delta))
    db = await get_db()
    await db.user_profiles.update_one(
        {"global_user_id": global_user_id},
        {"$set": {"affinity": new_value}},
        upsert=True,
    )
    return new_value


async def update_last_relationship_insight(global_user_id: str, insight: str) -> None:
    """Update the last relationship insight for a user."""
    db = await get_db()
    await db.user_profiles.update_one(
        {"global_user_id": global_user_id},
        {"$set": {"last_relationship_insight": insight}},
        upsert=True,
    )


async def upsert_user_facts(global_user_id: str, new_facts: list[str]) -> None:
    """Insert one character's view to the user to the top"""
    db = await get_db()
    existing = await get_user_facts(global_user_id)
    merged = list(dict.fromkeys(existing + new_facts))

    if merged:
        combined_facts_text = "\n".join(merged)
        embedding = await get_text_embedding(combined_facts_text)
    else:
        embedding = []

    await db.user_profiles.update_one(
        {"global_user_id": global_user_id},
        {"$set": {"global_user_id": global_user_id, "facts": merged, "embedding": embedding}},
        upsert=True,
    )


async def overwrite_user_facts(global_user_id: str, facts: list[str]) -> None:
    """Overwrite all user facts with a new list, replacing existing ones."""
    db = await get_db()
    
    if facts:
        combined_facts_text = "\n".join(facts)
        embedding = await get_text_embedding(combined_facts_text)
    else:
        embedding = []

    await db.user_profiles.update_one(
        {"global_user_id": global_user_id},
        {"$set": {"global_user_id": global_user_id, "facts": facts, "embedding": embedding}},
        upsert=True,
    )


async def enable_user_facts_vector_index() -> None:
    """Create a vector search index on the user_profiles collection for semantic user search."""
    await enable_vector_index("user_profiles", "user_facts_vector_index")


async def search_users_by_facts(
    query: str,
    limit: int = 5,
) -> list[tuple[float, UserProfileDoc]]:
    """Search for users based on semantic similarity of their accumulated facts.
    
    Args:
        query: Search query text
        limit: Maximum number of results to return
        
    Returns:
        List of (score, user_doc) tuples sorted by similarity (highest first)
    """
    query_embedding = await get_text_embedding(query)
    db = await get_db()
    collection = db.user_profiles

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
        {"$project": {"global_user_id": 1, "facts": 1, "affinity": 1, "score": 1, "_id": 0}},
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


async def get_character_profile() -> CharacterProfileDoc | dict:
    """Retrieve the full global document (profile + runtime state).

    All fields live at the top level of ``_id: "global"``.
    Returns an empty dict if the document does not exist.
    """
    db = await get_db()
    doc = await db.character_state.find_one({"_id": "global"})
    if doc is None:
        return {}
    doc.pop("_id", None)
    return doc


async def save_character_profile(profile: dict) -> None:
    """Persist character personality profile fields to the global document.

    Each key in *profile* is written as a top-level field on the
    ``_id: "global"`` document.  Runtime state fields are untouched.
    """
    db = await get_db()
    await db.character_state.update_one(
        {"_id": "global"},
        {"$set": profile},
        upsert=True,
    )


async def get_character_state() -> CharacterProfileDoc | dict:
    """Retrieve the global character state document.

    This returns the same data as :func:`get_character_profile` — all
    top-level fields of ``_id: "global"`` minus ``_id`` itself.
    """
    return await get_character_profile()


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
    source_global_user_id: str = "",
) -> None:
    """Save a memory entry with embedding to the memory collection.
    
    Args:
        memory_name: Name/identifier for the memory
        content: The memory content/text
        timestamp: ISO-8601 UTC timestamp for when the memory was created or updated
        source_global_user_id: UUID4 of the user who triggered this memory (empty for non-user-specific)
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
                "source_global_user_id": source_global_user_id,
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
    source_global_user_id: str | None = None,
) -> list[tuple[float, MemoryDoc]]:
    """
    Search memory collection using keyword or vector relevance.
    
    Args:
        query: The search query string.
        limit: Maximum number of results.
        method: "keyword" for regex text search, "vector" for semantic search.
        source_global_user_id: Optional UUID4 to filter memories originating from a specific user.
        
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
        if source_global_user_id:
            base_filter["source_global_user_id"] = source_global_user_id

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
    ]

    # Post-filter by source user if specified
    if source_global_user_id:
        pipeline.append({"$match": {"source_global_user_id": source_global_user_id}})

    pipeline.append({"$unset": "embedding"})

    cursor = collection.aggregate(pipeline)
    docs = await cursor.to_list(length=limit)
    return [(doc.pop("score", 0.0), doc) for doc in docs]

