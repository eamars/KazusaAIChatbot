"""Tests for resetting shared memory from repository seed data."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from kazusa_ai_chatbot.memory_evolution import repository as repository_module
from kazusa_ai_chatbot.memory_evolution import reset as reset_module
from kazusa_ai_chatbot.memory_evolution.models import (
    MemoryAuthority,
    MemorySourceKind,
    MemoryStatus,
)


class _Collection:
    """In-memory collection fake for reset tests."""

    def __init__(self, docs: list[dict]) -> None:
        """Create the collection with Mongo-like ``_id`` fields."""
        self.docs = {
            doc.get("_id", doc.get("memory_unit_id", str(index))): dict(doc)
            for index, doc in enumerate(docs)
        }

    def _matches(self, doc: dict, query: dict) -> bool:
        for key, expected in query.items():
            if key == "$and":
                if not all(self._matches(doc, part) for part in expected):
                    return False
                continue
            if key == "$or":
                if not any(self._matches(doc, part) for part in expected):
                    return False
                continue

            actual = doc.get(key)
            if isinstance(expected, dict):
                if "$in" in expected and actual not in expected["$in"]:
                    return False
                if "$nin" in expected and actual in expected["$nin"]:
                    return False
                if "$exists" in expected:
                    exists = key in doc
                    if exists is not bool(expected["$exists"]):
                        return False
            elif actual != expected:
                return False
        return True

    async def count_documents(self, query: dict) -> int:
        """Count matching documents."""
        return_value = sum(
            1 for doc in self.docs.values()
            if self._matches(doc, query)
        )
        return return_value

    async def find_one(self, query: dict) -> dict | None:
        """Return the first matching document."""
        for doc in self.docs.values():
            if self._matches(doc, query):
                return_value = dict(doc)
                return return_value
        return None

    async def find_by_id(self, memory_unit_id: str) -> dict | None:
        """Return one document by memory-unit id."""
        for doc in self.docs.values():
            if doc.get("memory_unit_id") == memory_unit_id:
                return_value = dict(doc)
                return return_value
        return None

    async def insert_one(self, document: dict) -> SimpleNamespace:
        """Insert a guard or seed document."""
        doc_id = document.get("_id", document.get("memory_unit_id"))
        self.docs[doc_id] = dict(document)
        return_value = SimpleNamespace(inserted_id=doc_id)
        return return_value

    async def delete_one(self, query: dict) -> SimpleNamespace:
        """Delete the first matching document."""
        for doc_id, doc in list(self.docs.items()):
            if self._matches(doc, query):
                del self.docs[doc_id]
                return SimpleNamespace(deleted_count=1)
        return_value = SimpleNamespace(deleted_count=0)
        return return_value

    async def delete_many(self, query: dict) -> SimpleNamespace:
        """Delete all matching documents."""
        deleted = 0
        for doc_id, doc in list(self.docs.items()):
            if self._matches(doc, query):
                del self.docs[doc_id]
                deleted += 1
        return_value = SimpleNamespace(deleted_count=deleted)
        return return_value

    async def replace_one(
        self,
        query: dict,
        document: dict,
        *,
        upsert: bool = False,
    ) -> SimpleNamespace:
        """Replace by ``memory_unit_id``."""
        existing_id = None
        for doc_id, doc in self.docs.items():
            if self._matches(doc, query):
                existing_id = doc_id
                break
        doc_id = existing_id or document["memory_unit_id"]
        self.docs[doc_id] = dict(document)
        return_value = SimpleNamespace(matched_count=1 if existing_id else 0)
        return return_value

    async def replace_doc(self, document: dict) -> SimpleNamespace:
        """Replace one document by memory-unit id."""
        return_value = await self.replace_one(
            {"memory_unit_id": document["memory_unit_id"]},
            document,
            upsert=True,
        )
        return return_value

    async def count_legacy_seed_managed(self) -> int:
        """Count seed-managed rows without evolving ids."""
        return_value = await self.count_documents({
            "$and": [
                {
                    "source_global_user_id": "",
                    "source_kind": {
                        "$in": [
                            MemorySourceKind.EXTERNAL_IMPORTED,
                            MemorySourceKind.SEEDED_MANUAL,
                        ]
                    },
                },
                {"memory_unit_id": {"$exists": False}},
            ]
        })
        return return_value

    async def count_unmanaged_seed(self, seed_ids: list[str]) -> int:
        """Count seed-managed evolving rows absent from the seed ids."""
        return_value = await self.count_documents({
            "$and": [
                {
                    "source_global_user_id": "",
                    "source_kind": {
                        "$in": [
                            MemorySourceKind.EXTERNAL_IMPORTED,
                            MemorySourceKind.SEEDED_MANUAL,
                        ]
                    },
                },
                {"memory_unit_id": {"$exists": True}},
                {"memory_unit_id": {"$nin": seed_ids}},
            ]
        })
        return return_value

    async def count_runtime_reflection(self) -> int:
        """Count reflection-inferred rows."""
        return_value = await self.count_documents({
            "source_kind": MemorySourceKind.REFLECTION_INFERRED,
        })
        return return_value

    async def delete_reset_seed_managed(self, seed_ids: list[str]) -> int:
        """Delete reset-managed seed-lane rows absent from current seeds."""
        result = await self.delete_many({
            "$and": [
                {
                    "source_global_user_id": "",
                    "source_kind": {
                        "$in": [
                            MemorySourceKind.EXTERNAL_IMPORTED,
                            MemorySourceKind.SEEDED_MANUAL,
                        ]
                    },
                },
                {
                    "$or": [
                        {"memory_unit_id": {"$exists": False}},
                        {"memory_unit_id": {"$nin": seed_ids}},
                    ]
                },
            ]
        })
        return_value = int(result.deleted_count)
        return return_value


def _seed_entry(name: str = "Seed memory") -> dict:
    return_value = {
        "memory_name": name,
        "content": "Seed content",
        "source_global_user_id": "",
        "memory_type": "fact",
        "source_kind": MemorySourceKind.SEEDED_MANUAL,
        "confidence_note": "seed",
        "status": MemoryStatus.ACTIVE,
        "expiry_timestamp": None,
    }
    return return_value


def _patch_reset(monkeypatch: pytest.MonkeyPatch, collection: _Collection) -> MagicMock:
    runtime = MagicMock()
    runtime.invalidate = AsyncMock(return_value=1)
    monkeypatch.setattr(
        reset_module.memory_store,
        "find_memory_unit_by_id",
        AsyncMock(side_effect=collection.find_by_id),
    )
    monkeypatch.setattr(
        reset_module.memory_store,
        "count_legacy_seed_managed_memory",
        AsyncMock(side_effect=collection.count_legacy_seed_managed),
    )
    monkeypatch.setattr(
        reset_module.memory_store,
        "count_unmanaged_seed_memory",
        AsyncMock(side_effect=collection.count_unmanaged_seed),
    )
    monkeypatch.setattr(
        reset_module.memory_store,
        "count_runtime_reflection_memory",
        AsyncMock(side_effect=collection.count_runtime_reflection),
    )
    monkeypatch.setattr(
        reset_module.memory_store,
        "acquire_memory_write_lock",
        AsyncMock(return_value=True),
    )
    monkeypatch.setattr(
        reset_module.memory_store,
        "release_memory_write_lock",
        AsyncMock(),
    )
    monkeypatch.setattr(
        reset_module.memory_store,
        "delete_reset_seed_managed_memory",
        AsyncMock(side_effect=collection.delete_reset_seed_managed),
    )
    monkeypatch.setattr(
        reset_module.memory_store,
        "replace_memory_unit_document",
        AsyncMock(side_effect=collection.replace_doc),
    )
    monkeypatch.setattr(
        repository_module.memory_store,
        "compute_memory_embedding",
        AsyncMock(return_value=[0.4, 0.5]),
    )
    monkeypatch.setattr(
        repository_module,
        "get_rag_cache2_runtime",
        MagicMock(return_value=runtime),
    )
    return runtime


@pytest.mark.asyncio
async def test_reset_memory_from_entries_dry_run_reports_legacy_and_runtime_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dry-run reports destructive seed-lane work without embeddings."""
    collection = _Collection([
        {
            "_id": "legacy",
            "memory_name": "Legacy",
            "source_global_user_id": "",
            "source_kind": MemorySourceKind.SEEDED_MANUAL,
        },
        {
            "_id": "runtime",
            "memory_unit_id": "runtime-1",
            "source_kind": MemorySourceKind.REFLECTION_INFERRED,
        },
    ])
    _patch_reset(monkeypatch, collection)

    result = await reset_module.reset_memory_from_entries(
        [_seed_entry()],
        dry_run=True,
        prune_unmanaged_global=True,
    )

    assert result["dry_run"] is True
    assert result["seed_rows_loaded"] == 1
    assert result["seed_rows_inserted"] == 1
    assert result["legacy_rows_deleted"] == 1
    assert result["runtime_rows_preserved"] == 1
    assert result["embeddings_computed"] == 0
    assert "legacy" in collection.docs
    repository_module.memory_store.compute_memory_embedding.assert_not_awaited()


@pytest.mark.asyncio
async def test_reset_memory_from_entries_apply_reseeds_and_preserves_runtime_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Apply deletes seed-managed rows, inserts seed rows, and keeps runtime lore."""
    collection = _Collection([
        {
            "_id": "legacy",
            "memory_name": "Legacy",
            "source_global_user_id": "",
            "source_kind": MemorySourceKind.SEEDED_MANUAL,
        },
        {
            "_id": "runtime",
            "memory_unit_id": "runtime-1",
            "source_kind": MemorySourceKind.REFLECTION_INFERRED,
        },
    ])
    runtime = _patch_reset(monkeypatch, collection)

    result = await reset_module.reset_memory_from_entries(
        [_seed_entry()],
        dry_run=False,
        prune_unmanaged_global=True,
    )

    assert result["dry_run"] is False
    assert result["seed_rows_inserted"] == 1
    assert result["legacy_rows_deleted"] == 1
    assert result["embeddings_computed"] == 1
    assert result["cache_invalidated"] is True
    assert "legacy" not in collection.docs
    assert "runtime" in collection.docs
    seed_docs = [
        doc for doc in collection.docs.values()
        if doc.get("authority") == MemoryAuthority.SEED
    ]
    assert len(seed_docs) == 1
    assert seed_docs[0]["embedding"] == [0.4, 0.5]
    event = runtime.invalidate.await_args.args[0]
    assert event.source == "memory"
    reset_module.memory_store.release_memory_write_lock.assert_awaited_once()


@pytest.mark.asyncio
async def test_reset_memory_from_entries_apply_fails_when_write_lock_is_held(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Apply reset fails before mutating when another memory write holds the guard."""
    collection = _Collection([])
    _patch_reset(monkeypatch, collection)
    reset_module.memory_store.acquire_memory_write_lock.return_value = False

    with pytest.raises(RuntimeError, match="memory write or reset"):
        await reset_module.reset_memory_from_entries(
            [_seed_entry()],
            dry_run=False,
            prune_unmanaged_global=True,
        )

    reset_module.memory_store.delete_reset_seed_managed_memory.assert_not_awaited()
    reset_module.memory_store.replace_memory_unit_document.assert_not_awaited()
    repository_module.memory_store.compute_memory_embedding.assert_not_awaited()
