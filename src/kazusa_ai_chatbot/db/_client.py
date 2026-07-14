"""MongoDB client, async embedding helper, and vector-index utility.

Shared by every other ``db.*`` submodule. The client is lazily initialised
on first use and reused for the lifetime of the process.

``get_db`` is intentionally private to the DB package. Runtime code,
application services, and maintenance scripts should use semantic helpers
exported by ``kazusa_ai_chatbot.db`` instead of holding raw database handles.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Mapping, Sequence
from typing import Any

from motor.motor_asyncio import AsyncIOMotorClient
from openai import AsyncOpenAI
from pymongo.errors import ConnectionFailure
from pymongo.operations import SearchIndexModel

from kazusa_ai_chatbot.config import (
    EMBEDDING_API_KEY,
    EMBEDDING_BASE_URL,
    EMBEDDING_MODEL,
    MONGODB_DB_NAME,
    MONGODB_URI,
)

logger = logging.getLogger(__name__)


# ── Embedding client (lazy) ────────────────────────────────────────

_embed_client: AsyncOpenAI | None = None
_embed_client_loop: asyncio.AbstractEventLoop | None = None
_EMBEDDING_REQUEST_SEMAPHORE = asyncio.Semaphore(10)
_EMBEDDING_BATCH_SIZE = 10
_NOMIC_EMBED_TEXT_V2_MOE_MODEL_FRAGMENT = "nomic-embed-text-v2-moe"
_NOMIC_QUERY_PREFIX = "search_query: "
_NOMIC_DOCUMENT_PREFIX = "search_document: "


def _get_embed_client() -> AsyncOpenAI:
    """Return a lazily-initialised AsyncOpenAI client for embeddings."""
    global _embed_client, _embed_client_loop
    current_loop = asyncio.get_running_loop()
    if _embed_client is None or _embed_client_loop is not current_loop:
        _embed_client = AsyncOpenAI(
            base_url=EMBEDDING_BASE_URL,
            api_key=EMBEDDING_API_KEY,
        )
        _embed_client_loop = current_loop
    return _embed_client


def _embedding_model_uses_nomic_prefixes() -> bool:
    """Return whether the configured embedding model needs Nomic role prefixes."""

    model_name = EMBEDDING_MODEL.lower()
    uses_prefixes = _NOMIC_EMBED_TEXT_V2_MOE_MODEL_FRAGMENT in model_name
    return uses_prefixes


def _apply_embedding_role_prefix(text: str, prefix: str) -> str:
    """Prepend a Nomic role prefix to raw source text."""

    prefixed_text = f"{prefix}{text}"
    return prefixed_text


def _embedding_texts_for_role(texts: list[str], prefix: str) -> list[str]:
    """Return effective embedding endpoint inputs for one semantic role."""

    if not _embedding_model_uses_nomic_prefixes():
        return_value = list(texts)
        return return_value

    return_value = [
        _apply_embedding_role_prefix(text, prefix)
        for text in texts
    ]
    return return_value


async def _request_text_embeddings(texts: list[str]) -> list[list[float]]:
    """Call the embedding endpoint for already-prepared input text."""

    if not texts:
        return_value: list[list[float]] = []
        return return_value

    client = _get_embed_client()
    all_embeddings: list[list[float]] = []
    for start in range(0, len(texts), _EMBEDDING_BATCH_SIZE):
        chunk = texts[start:start + _EMBEDDING_BATCH_SIZE]
        async with _EMBEDDING_REQUEST_SEMAPHORE:
            resp = await client.embeddings.create(input=chunk, model=EMBEDDING_MODEL)
        sorted_data = sorted(resp.data, key=lambda d: d.index)
        all_embeddings.extend(d.embedding for d in sorted_data)
    return all_embeddings


async def get_query_text_embedding(text: str) -> list[float]:
    """Compute a query-role embedding for vector retrieval text.

    Args:
        text: Search intent text.

    Returns:
        Query embedding vector.
    """
    embeddings = await get_query_text_embeddings_batch([text])
    return_value = embeddings[0]
    return return_value


async def get_query_text_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """Compute query-role embeddings for multiple retrieval texts.

    Args:
        texts: Search intent texts.

    Returns:
        Query embedding vectors in input order.
    """
    effective_texts = _embedding_texts_for_role(texts, _NOMIC_QUERY_PREFIX)
    embeddings = await _request_text_embeddings(effective_texts)
    return embeddings


async def get_document_text_embedding(text: str) -> list[float]:
    """Compute a document-role embedding for stored retrievable text.

    Args:
        text: Stored source text.

    Returns:
        Document embedding vector.
    """
    embeddings = await get_document_text_embeddings_batch([text])
    return_value = embeddings[0]
    return return_value


async def get_document_text_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """Compute document-role embeddings for multiple stored source texts.

    Args:
        texts: Stored source texts.

    Returns:
        Document embedding vectors in input order.
    """
    effective_texts = _embedding_texts_for_role(texts, _NOMIC_DOCUMENT_PREFIX)
    embeddings = await _request_text_embeddings(effective_texts)
    return embeddings


async def get_text_embedding(text: str) -> list[float]:
    """Compute a document-role embedding vector for a single text string.

    Args:
        text: Source text. Empty strings are accepted by the embedding API
            and return a valid vector.

    Returns:
        The document embedding as a list of floats.
    """
    embedding = await get_document_text_embedding(text)
    return embedding


async def get_text_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """Compute document-role embedding vectors for multiple texts.

    Args:
        texts: Stored source texts.

    Returns:
        List of document embedding vectors in the same order.
    """
    embeddings = await get_document_text_embeddings_batch(texts)
    return embeddings


# ── MongoDB client (lazy) ──────────────────────────────────────────

_client: AsyncIOMotorClient | None = None
_db = None
_db_loop: asyncio.AbstractEventLoop | None = None
TEST_DATABASE_NAME = "_test_kazusa_live_llm"


class DatabaseTestGuardError(RuntimeError):
    """Raised when guarded tests would connect to a non-test database."""


def _assert_guarded_database_name() -> None:
    """Reject non-isolated database configuration while the test guard is on."""

    if os.getenv("KAZUSA_TEST_DB_GUARD") != "1":
        return
    if MONGODB_DB_NAME != TEST_DATABASE_NAME:
        raise DatabaseTestGuardError(
            f"guarded DB access requires {TEST_DATABASE_NAME!r}; "
            f"received {MONGODB_DB_NAME!r}"
        )


async def get_db():
    """Return the internal async MongoDB database handle.

    This is for ``kazusa_ai_chatbot.db`` submodules only. Callers outside the
    DB package must go through public semantic helpers exposed by the package
    facade.

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
        _assert_guarded_database_name()
        _client = AsyncIOMotorClient(MONGODB_URI)
        _db = _client[MONGODB_DB_NAME]
        _db_loop = current_loop
        try:
            await _client.admin.command("ping")
            logger.info(f'Connected to MongoDB at {MONGODB_URI}')
        except ConnectionFailure as exc:
            logger.error(f"Failed to connect to MongoDB at {MONGODB_URI}: {exc}")
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

def build_vector_search_index_fields(
    *,
    path: str,
    num_dimensions: int,
    filter_paths: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Build Atlas vector-search field definitions.

    Args:
        path: Field containing embedding vectors.
        num_dimensions: Embedding vector dimension count.
        filter_paths: Optional scalar fields for vector pre-filtering.

    Returns:
        Search-index field definitions.
    """

    fields: list[dict[str, Any]] = [
        {
            "type": "vector",
            "path": path,
            "numDimensions": num_dimensions,
            "similarity": "cosine",
        }
    ]
    for filter_path in filter_paths or ():
        if filter_path:
            fields.append({"type": "filter", "path": filter_path})
    return fields


def build_vector_search_index_model(
    *,
    index_name: str,
    path: str,
    num_dimensions: int,
    filter_paths: Sequence[str] | None = None,
) -> SearchIndexModel:
    """Build a ``SearchIndexModel`` for a vector-search index."""

    fields = build_vector_search_index_fields(
        path=path,
        num_dimensions=num_dimensions,
        filter_paths=filter_paths,
    )
    search_index_model = SearchIndexModel(
        definition={"fields": fields},
        name=index_name,
        type="vectorSearch",
    )
    return search_index_model


def _search_index_definition(index_document: Mapping[str, Any]) -> Mapping[str, Any]:
    """Extract a search-index definition from Atlas or PyMongo-shaped docs."""

    definition = index_document.get("latestDefinition")
    if isinstance(definition, Mapping):
        return definition

    definition = index_document.get("definition")
    if isinstance(definition, Mapping):
        return definition

    return_value: Mapping[str, Any] = {}
    return return_value


def vector_index_filter_paths(index_document: Mapping[str, Any]) -> set[str]:
    """Return filter paths configured on one Atlas vector-search index."""

    definition = _search_index_definition(index_document)
    raw_fields = definition.get("fields")
    if not isinstance(raw_fields, list):
        return_value: set[str] = set()
        return return_value

    filter_paths: set[str] = set()
    for raw_field in raw_fields:
        if not isinstance(raw_field, Mapping):
            continue
        if raw_field.get("type") != "filter":
            continue
        path = raw_field.get("path")
        if isinstance(path, str) and path:
            filter_paths.add(path)
    return filter_paths


def vector_index_missing_filter_paths(
    index_document: Mapping[str, Any],
    required_paths: Sequence[str],
) -> list[str]:
    """Return required filter paths absent from one vector-search index."""

    existing_paths = vector_index_filter_paths(index_document)
    missing_paths = [
        path
        for path in required_paths
        if path not in existing_paths
    ]
    return missing_paths


def vector_index_definition_issues(
    index_document: Mapping[str, Any],
    *,
    path: str,
    num_dimensions: int,
    required_filter_paths: Sequence[str],
    similarity: str = "cosine",
) -> list[str]:
    """Return mismatches between an index document and expected vector shape."""

    issues: list[str] = []
    index_type = index_document.get("type")
    if isinstance(index_type, str) and index_type != "vectorSearch":
        issues.append("index_type")

    definition = _search_index_definition(index_document)
    raw_fields = definition.get("fields")
    vector_field: Mapping[str, Any] | None = None
    if isinstance(raw_fields, list):
        for raw_field in raw_fields:
            if not isinstance(raw_field, Mapping):
                continue
            if raw_field.get("type") == "vector":
                vector_field = raw_field
                break

    if vector_field is None:
        issues.append("missing_vector_field")
    else:
        if vector_field.get("path") != path:
            issues.append("vector_path")
        if vector_field.get("numDimensions") != num_dimensions:
            issues.append("num_dimensions")
        if vector_field.get("similarity") != similarity:
            issues.append("similarity")

    missing_filter_issues = [
        f"missing_filter_path:{filter_path}"
        for filter_path in vector_index_missing_filter_paths(
            index_document,
            required_filter_paths,
        )
    ]
    issues.extend(missing_filter_issues)
    return issues


def vector_index_has_filter_paths(
    index_document: Mapping[str, Any],
    required_paths: Sequence[str],
) -> bool:
    """Return whether a vector-search index has all required filter paths."""

    missing_paths = vector_index_missing_filter_paths(
        index_document,
        required_paths,
    )
    has_filter_paths = not missing_paths
    return has_filter_paths


async def get_search_index_definition(
    collection_name: str,
    index_name: str,
) -> dict[str, Any] | None:
    """Load one Atlas search-index definition document by name.

    Args:
        collection_name: Target collection name.
        index_name: Search index name.

    Returns:
        The raw search-index document when found, otherwise ``None``.
    """

    db = await get_db()
    collection = db[collection_name]
    async for index in collection.list_search_indexes():
        if index.get("name") == index_name:
            return_value = dict(index)
            return return_value
    return_value = None
    return return_value


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
                missing_filter_paths = vector_index_missing_filter_paths(
                    index,
                    filter_paths or [],
                )
                if missing_filter_paths:
                    logger.warning(
                        f"Vector search index {index_name!r} exists but is "
                        "missing filter paths "
                        f"{missing_filter_paths}; run the vector-index "
                        "migration script to recreate it."
                    )
                else:
                    logger.info(
                        f'Vector search index \'{index_name}\' already exists.'
                    )
                return
    except Exception as exc:
        logger.exception(
            f"Could not list search indexes "
            f"(might not exist yet or not supported): {exc}"
        )

    logger.info(f'Vector search index \'{index_name}\' not found. Creating...')

    sample_embedding = await get_document_text_embedding("test")
    num_dimensions = len(sample_embedding)

    search_index_model = build_vector_search_index_model(
        index_name=index_name,
        path=path,
        num_dimensions=num_dimensions,
        filter_paths=filter_paths,
    )

    try:
        await collection.create_search_index(search_index_model)
        logger.info(
            f"Successfully created vector search index {index_name!r} on "
            f"{collection_name}.{path} with {num_dimensions} dimensions."
        )
    except Exception as exc:
        logger.exception(
            f"Failed to create vector search index {index_name!r}: {exc}"
        )
        raise
