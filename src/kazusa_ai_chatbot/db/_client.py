"""MongoDB client, async embedding helper, and vector-index utility.

Shared by every other ``db.*`` submodule. The client is lazily initialised
on first use and reused for the lifetime of the process.
"""

from __future__ import annotations

import logging

from motor.motor_asyncio import AsyncIOMotorClient
from openai import AsyncOpenAI
from pymongo.errors import ConnectionFailure
from pymongo.operations import SearchIndexModel

from kazusa_ai_chatbot.config import (
    EMBEDDING_BASE_URL,
    EMBEDDING_MODEL,
    LLM_API_KEY,
    MONGODB_DB_NAME,
    MONGODB_URI,
)

logger = logging.getLogger(__name__)


# ── Embedding client (lazy) ────────────────────────────────────────

_embed_client: AsyncOpenAI | None = None


def _get_embed_client() -> AsyncOpenAI:
    """Return a lazily-initialised AsyncOpenAI client for embeddings."""
    global _embed_client
    if _embed_client is None:
        _embed_client = AsyncOpenAI(
            base_url=EMBEDDING_BASE_URL,
            api_key=LLM_API_KEY,
        )
    return _embed_client


async def get_text_embedding(text: str) -> list[float]:
    """Compute an embedding vector for a single text string.

    Args:
        text: Source text. Empty strings are accepted by the embedding API
            and return a valid (zero-information) vector.

    Returns:
        The embedding as a list of floats.
    """
    client = _get_embed_client()
    resp = await client.embeddings.create(input=[text], model=EMBEDDING_MODEL)
    return resp.data[0].embedding


# ── MongoDB client (lazy) ──────────────────────────────────────────

_client: AsyncIOMotorClient | None = None
_db = None


async def get_db():
    """Return the async MongoDB database handle, creating the client on first call.

    Raises:
        ConnectionFailure: If the initial ping to MongoDB fails.
    """
    global _client, _db
    if _db is None:
        _client = AsyncIOMotorClient(MONGODB_URI)
        _db = _client[MONGODB_DB_NAME]
        try:
            await _client.admin.command("ping")
            logger.info("Connected to MongoDB at %s", MONGODB_URI)
        except ConnectionFailure:
            logger.error("Failed to connect to MongoDB at %s", MONGODB_URI)
            raise
    return _db


async def close_db() -> None:
    """Close the MongoDB client connection if open."""
    global _client, _db
    if _client is not None:
        _client.close()
        _client = None
        _db = None


# ── Vector search index helper ─────────────────────────────────────

async def enable_vector_index(
    collection_name: str,
    index_name: str,
    *,
    path: str = "embedding",
) -> None:
    """Create a cosine vector-search index on ``collection_name`` if missing.

    Args:
        collection_name: Target collection.
        index_name: Name to assign to the search index.
        path: Field that holds the embedding array. Defaults to ``"embedding"``;
            override for collections with multiple embedding fields
            (e.g. ``"diary_embedding"``, ``"facts_embedding"``).
    """
    db = await get_db()
    collection = db[collection_name]

    try:
        async for index in collection.list_search_indexes():
            if index.get("name") == index_name:
                logger.info("Vector search index '%s' already exists.", index_name)
                return
    except Exception as e:
        logger.debug("Could not list search indexes (might not exist yet or not supported): %s", e)

    logger.info("Vector search index '%s' not found. Creating...", index_name)

    sample_embedding = await get_text_embedding("test")
    num_dimensions = len(sample_embedding)

    search_index_model = SearchIndexModel(
        definition={
            "fields": [
                {
                    "type": "vector",
                    "path": path,
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
        logger.info(
            "Successfully created vector search index '%s' on %s.%s with %d dimensions.",
            index_name, collection_name, path, num_dimensions,
        )
    except Exception as e:
        logger.error("Failed to create vector search index '%s': %s", index_name, e)
        raise
