"""Tests for active-only shared-memory retrieval."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import ANY, AsyncMock, MagicMock

import pytest

from kazusa_ai_chatbot.db import memory as db_memory_module
from kazusa_ai_chatbot.db import memory_evolution as db_memory_evolution_module
from kazusa_ai_chatbot.memory_evolution import repository as repository_module
from kazusa_ai_chatbot.memory_evolution.models import MemoryStatus
from kazusa_ai_chatbot.rag.memory_retrieval_tools import (
    search_persistent_memory,
    search_persistent_memory_keyword,
)


class _Cursor:
    """Async cursor fake used by retrieval tests."""

    def __init__(self, docs: list[dict]) -> None:
        """Create a fake cursor.

        Args:
            docs: Documents returned by ``to_list``.
        """
        self.docs = docs

    def limit(self, limit: int) -> "_Cursor":
        """Return this cursor for Mongo-like chaining."""
        return self

    def sort(self, *_args) -> "_Cursor":
        """Return this cursor for Mongo-like chaining."""
        return self

    async def to_list(self, length: int) -> list[dict]:
        """Return the configured documents."""
        return_value = [dict(doc) for doc in self.docs[:length]]
        return return_value


@pytest.mark.asyncio
async def test_search_memory_keyword_defaults_to_active_non_expired(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Lower-level keyword search applies active lifecycle filtering by default."""
    collection = MagicMock()
    collection.find.return_value = _Cursor([
        {"memory_name": "memory", "content": "content", "embedding": [0.1]},
    ])
    db = SimpleNamespace(memory=collection)
    monkeypatch.setattr(db_memory_module, "get_db", AsyncMock(return_value=db))

    await db_memory_module.search_memory("memory", method="keyword", limit=5)

    query = collection.find.call_args.args[0]
    assert {"status": MemoryStatus.ACTIVE} in query["$and"]
    assert {"expiry_timestamp": None} in query["$and"][0]["$or"]


@pytest.mark.asyncio
async def test_search_memory_vector_post_filters_active_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Vector search keeps lifecycle filtering outside vector prefilters."""
    cursor = MagicMock()
    cursor.to_list = AsyncMock(return_value=[])
    collection = MagicMock()
    collection.aggregate.return_value = cursor
    db = SimpleNamespace(memory=collection)
    monkeypatch.setattr(db_memory_module, "get_db", AsyncMock(return_value=db))
    monkeypatch.setattr(
        db_memory_module,
        "get_text_embedding",
        AsyncMock(return_value=[0.1, 0.2]),
    )

    await db_memory_module.search_memory("memory", method="vector", limit=5)

    pipeline = collection.aggregate.call_args.args[0]
    vector_search = pipeline[0]["$vectorSearch"]
    assert "filter" not in vector_search
    assert vector_search["limit"] > 5
    assert {"$match": {"$and": [
        {
            "$or": [
                {"expiry_timestamp": None},
                {"expiry_timestamp": {"$exists": False}},
                {"expiry_timestamp": {"$gt": ANY}},
            ]
        },
        {"status": MemoryStatus.ACTIVE},
    ]}} in pipeline
    assert pipeline[-1] == {"$limit": 5}


@pytest.mark.asyncio
async def test_find_active_memory_documents_vector_post_filters_active_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The DB interface keeps filters index-compatible after vector search."""
    cursor = MagicMock()
    cursor.to_list = AsyncMock(return_value=[])
    collection = MagicMock()
    collection.aggregate.return_value = cursor
    db = SimpleNamespace(memory=collection)
    monkeypatch.setattr(
        db_memory_evolution_module,
        "get_db",
        AsyncMock(return_value=db),
    )

    await db_memory_evolution_module.find_active_memory_documents(
        query={
            "semantic_query": "memory",
            "source_global_user_id": "user-1",
        },
        limit=4,
        now_timestamp="2026-05-05T00:00:00+00:00",
        query_embedding=[0.1, 0.2],
    )

    pipeline = collection.aggregate.call_args.args[0]
    vector_search = pipeline[0]["$vectorSearch"]
    assert "filter" not in vector_search
    assert vector_search["limit"] > 4
    assert pipeline[1] == {
        "$addFields": {"score": {"$meta": "vectorSearchScore"}}
    }
    match_stage = pipeline[2]
    assert match_stage["$match"]["source_global_user_id"] == "user-1"
    assert {"status": MemoryStatus.ACTIVE} in match_stage["$match"]["$and"]
    assert pipeline[-1] == {"$limit": 4}


@pytest.mark.asyncio
async def test_find_active_memory_documents_vector_returns_scores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Semantic active-memory search returns vector scores with documents."""
    collection = MagicMock()
    collection.aggregate.return_value = _Cursor([
        {
            "memory_unit_id": "unit-1",
            "content": "near duplicate",
            "score": 0.82,
            "embedding": [0.1],
        },
    ])
    db = SimpleNamespace(memory=collection)
    monkeypatch.setattr(
        db_memory_evolution_module,
        "get_db",
        AsyncMock(return_value=db),
    )

    results = await db_memory_evolution_module.find_active_memory_documents(
        query={"semantic_query": "near duplicate"},
        limit=4,
        now_timestamp="2026-05-05T00:00:00+00:00",
        query_embedding=[0.1, 0.2],
    )

    assert results == [
        (
            0.82,
            {
                "memory_unit_id": "unit-1",
                "content": "near duplicate",
            },
        )
    ]


@pytest.mark.asyncio
async def test_find_active_memory_documents_excludes_memory_unit_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The constrained DB query can omit known memory-unit ids."""
    collection = MagicMock()
    collection.find.return_value = _Cursor([])
    db = SimpleNamespace(memory=collection)
    monkeypatch.setattr(
        db_memory_evolution_module,
        "get_db",
        AsyncMock(return_value=db),
    )

    await db_memory_evolution_module.find_active_memory_documents(
        query={"exclude_memory_unit_ids": ["unit-1", "unit-2"]},
        limit=4,
        now_timestamp="2026-05-05T00:00:00+00:00",
        query_embedding=None,
    )

    filter_doc = collection.find.call_args.args[0]
    assert filter_doc["memory_unit_id"] == {"$nin": ["unit-1", "unit-2"]}


@pytest.mark.asyncio
async def test_find_active_memory_documents_metadata_uses_placeholder_score(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Metadata-only active-memory search uses keyword-style score semantics."""
    collection = MagicMock()
    collection.find.return_value = _Cursor([
        {
            "memory_unit_id": "unit-1",
            "content": "metadata match",
            "embedding": [0.1],
        },
    ])
    db = SimpleNamespace(memory=collection)
    monkeypatch.setattr(
        db_memory_evolution_module,
        "get_db",
        AsyncMock(return_value=db),
    )

    results = await db_memory_evolution_module.find_active_memory_documents(
        query={"memory_type": "fact"},
        limit=4,
        now_timestamp="2026-05-05T00:00:00+00:00",
        query_embedding=None,
    )

    assert results == [
        (
            -1.0,
            {
                "memory_unit_id": "unit-1",
                "content": "metadata match",
            },
        )
    ]


@pytest.mark.asyncio
async def test_update_memory_unit_fields_rejects_non_lifecycle_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Single-row DB updates cannot modify arbitrary memory fields."""
    get_db = AsyncMock()
    monkeypatch.setattr(
        db_memory_evolution_module,
        "get_db",
        get_db,
    )

    with pytest.raises(ValueError, match="unsupported memory lifecycle"):
        await db_memory_evolution_module.update_memory_unit_fields(
            "unit-1",
            {"content": "rewritten"},
        )

    get_db.assert_not_awaited()


@pytest.mark.asyncio
async def test_update_many_memory_unit_fields_rejects_non_lifecycle_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bulk DB updates cannot modify arbitrary memory fields."""
    get_db = AsyncMock()
    monkeypatch.setattr(
        db_memory_evolution_module,
        "get_db",
        get_db,
    )

    with pytest.raises(ValueError, match="unsupported memory lifecycle"):
        await db_memory_evolution_module.update_many_memory_unit_fields(
            ["unit-1"],
            {"authority": "manual"},
        )

    get_db.assert_not_awaited()


@pytest.mark.asyncio
async def test_find_active_memory_units_rejects_raw_mongo_filters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The evolving repository accepts only the constrained query shape."""
    with pytest.raises(ValueError, match="unsupported"):
        await repository_module.find_active_memory_units(
            query={"$or": [{"status": "superseded"}]},
            limit=5,
        )


@pytest.mark.asyncio
async def test_find_active_memory_units_returns_similarity_scores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The public repository search preserves DB-interface score tuples."""
    find_active_memory_documents = AsyncMock(return_value=[
        (
            0.91,
            {
                "memory_unit_id": "unit-1",
                "content": "near duplicate",
            },
        )
    ])
    monkeypatch.setattr(
        repository_module.memory_store,
        "compute_memory_embedding",
        AsyncMock(return_value=[0.1, 0.2]),
    )
    monkeypatch.setattr(
        repository_module.memory_store,
        "find_active_memory_documents",
        find_active_memory_documents,
    )

    results = await repository_module.find_active_memory_units(
        query={"semantic_query": "near duplicate"},
        limit=3,
    )

    assert results == [
        (
            0.91,
            {
                "memory_unit_id": "unit-1",
                "content": "near duplicate",
            },
        )
    ]
    repository_module.memory_store.compute_memory_embedding.assert_awaited_once_with(
        "near duplicate",
    )
    find_active_memory_documents.assert_awaited_once()


def test_persistent_memory_tools_do_not_expose_status_or_expiry_filters() -> None:
    """LLM-facing memory tools cannot request inactive or expired rows."""
    semantic_args = set(search_persistent_memory.args)
    keyword_args = set(search_persistent_memory_keyword.args)

    assert "status" not in semantic_args
    assert "expiry_before" not in semantic_args
    assert "expiry_after" not in semantic_args
    assert "status" not in keyword_args
    assert "expiry_before" not in keyword_args
    assert "expiry_after" not in keyword_args


@pytest.mark.asyncio
async def test_search_persistent_memory_delegates_without_status_filters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tool wrappers rely on the DB helper's active default."""
    search_memory = AsyncMock(return_value=[
        (
            0.9,
            {
                "memory_name": "memory",
                "content": "content",
                "timestamp": "2026-05-05T00:00:00+00:00",
            },
        )
    ])
    monkeypatch.setattr(
        "kazusa_ai_chatbot.rag.memory_retrieval_tools.search_memory_db",
        search_memory,
    )

    result = await search_persistent_memory.ainvoke({
        "search_query": "stable memory",
        "top_k": 2,
    })

    search_memory.assert_awaited_once_with(
        query="stable memory",
        limit=2,
        method="vector",
        source_global_user_id=None,
        memory_type=None,
        source_kind=None,
    )
    assert result[0]["content"] == "content"
