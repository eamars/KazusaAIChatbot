"""MongoDB client, async embedding helper, and vector-index utility.

Shared by every other ``db.*`` submodule. The client is lazily initialised
on first use and reused for the lifetime of the process.
"""

from __future__ import annotations

import asyncio
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
_embed_client_loop: asyncio.AbstractEventLoop | None = None
_EMBEDDING_REQUEST_SEMAPHORE = asyncio.Semaphore(10)
_EMBEDDING_BATCH_SIZE = 10


def _get_embed_client() -> AsyncOpenAI:
    """Return a lazily-initialised AsyncOpenAI client for embeddings."""
    global _embed_client, _embed_client_loop
    current_loop = asyncio.get_running_loop()
    if _embed_client is None or _embed_client_loop is not current_loop:
        _embed_client = AsyncOpenAI(
            base_url=EMBEDDING_BASE_URL,
            api_key=LLM_API_KEY,
        )
        _embed_client_loop = current_loop
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
    async with _EMBEDDING_REQUEST_SEMAPHORE:
        resp = await client.embeddings.create(input=[text], model=EMBEDDING_MODEL)
    return resp.data[0].embedding


async def get_text_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """Compute embedding vectors for multiple texts in a single API call.

    Args:
        texts: List of source texts. The OpenAI embeddings endpoint accepts
            up to ~2048 inputs per request (model-dependent).

    Returns:
        List of embedding vectors, one per input text, in the same order.
    """
    if not texts:
        return []
    client = _get_embed_client()
    all_embeddings: list[list[float]] = []
    for start in range(0, len(texts), _EMBEDDING_BATCH_SIZE):
        chunk = texts[start:start + _EMBEDDING_BATCH_SIZE]
        async with _EMBEDDING_REQUEST_SEMAPHORE:
            resp = await client.embeddings.create(input=chunk, model=EMBEDDING_MODEL)
        sorted_data = sorted(resp.data, key=lambda d: d.index)
        all_embeddings.extend(d.embedding for d in sorted_data)
    return all_embeddings


# ── MongoDB client (lazy) ──────────────────────────────────────────

_client: AsyncIOMotorClient | None = None
_db = None
_db_loop: asyncio.AbstractEventLoop | None = None


async def get_db():
    """Return the async MongoDB database handle, creating the client on first call.

    Raises:
        ConnectionFailure: If the initial ping to MongoDB fails.
    """
    global _client, _db, _db_loop
    current_loop = asyncio.get_running_loop()

    if _db is not None and _db_loop is not current_loop:
        if _client is not None:
            _client.close()
        _client = None
        _db = None
        _db_loop = None

    if _db is None:
        _client = AsyncIOMotorClient(MONGODB_URI)
        _db = _client[MONGODB_DB_NAME]
        _db_loop = current_loop
        try:
            await _client.admin.command("ping")
            logger.info("Connected to MongoDB at %s", MONGODB_URI)
        except ConnectionFailure:
            logger.error("Failed to connect to MongoDB at %s", MONGODB_URI)
            raise
    return _db


async def close_db() -> None:
    """Close the MongoDB client connection if open."""
    global _client, _db, _db_loop
    if _client is not None:
        _client.close()
        _client = None
        _db = None
        _db_loop = None


# ── Vector search index helper ─────────────────────────────────────

async def enable_vector_index(
    collection_name: str,
    index_name: str,
    *,
    path: str = "embedding",
    filter_paths: list[str] | None = None,
) -> None:
    """Create a cosine vector-search index on ``collection_name`` if missing.

    Args:
        collection_name: Target collection.
        index_name: Name to assign to the search index.
        path: Field that holds the embedding array. Defaults to ``"embedding"``.
        filter_paths: Optional scalar fields to index for vector pre-filtering.
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

    fields = [
        {
            "type": "vector",
            "path": path,
            "numDimensions": num_dimensions,
            "similarity": "cosine",
        }
    ]
    for filter_path in filter_paths or []:
        fields.append({"type": "filter", "path": filter_path})

    search_index_model = SearchIndexModel(
        definition={
            "fields": fields
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
