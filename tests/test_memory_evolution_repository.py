"""Tests for evolving shared-memory repository writes."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from kazusa_ai_chatbot.memory_evolution.models import (
    MemoryAuthority,
    MemorySourceKind,
    MemoryStatus,
)
from kazusa_ai_chatbot.memory_evolution import repository as repository_module


class _MemoryCollection:
    """Small collection fake for repository write tests."""

    def __init__(self, docs: list[dict] | None = None) -> None:
        """Create the fake collection.

        Args:
            docs: Existing rows keyed by ``memory_unit_id``.
        """
        self.docs = {
            doc["memory_unit_id"]: dict(doc)
            for doc in docs or []
        }
        self.inserted: list[dict] = []
        self.updated: list[tuple[dict, dict]] = []

    async def find_by_id(self, memory_unit_id: str) -> dict | None:
        """Return one document by ``memory_unit_id``."""
        doc = self.docs.get(memory_unit_id)
        return_value = dict(doc) if doc is not None else None
        return return_value

    async def insert_doc(self, document: dict) -> SimpleNamespace:
        """Insert a new document into the fake collection."""
        self.docs[document["memory_unit_id"]] = dict(document)
        self.inserted.append(dict(document))
        return_value = SimpleNamespace(inserted_id=document["memory_unit_id"])
        return return_value

    async def update_fields(self, memory_unit_id: str, fields: dict) -> SimpleNamespace:
        """Apply field updates to one document."""
        doc = self.docs[memory_unit_id]
        doc.update(fields)
        self.updated.append(({"memory_unit_id": memory_unit_id}, {"$set": fields}))
        return_value = SimpleNamespace(matched_count=1)
        return return_value

    async def update_many_fields(self, memory_unit_ids: list[str], fields: dict) -> SimpleNamespace:
        """Apply field updates to many documents."""
        ids = set(memory_unit_ids)
        modified = 0
        for memory_unit_id, doc in self.docs.items():
            if memory_unit_id in ids:
                doc.update(fields)
                modified += 1
        return_value = SimpleNamespace(modified_count=modified)
        return return_value


def _patch_repository(
    monkeypatch: pytest.MonkeyPatch,
    collection: _MemoryCollection,
) -> MagicMock:
    runtime = MagicMock()
    runtime.invalidate = AsyncMock(return_value=1)
    monkeypatch.setattr(
        repository_module.memory_store,
        "compute_memory_embedding",
        AsyncMock(return_value=[0.1, 0.2]),
    )
    monkeypatch.setattr(
        repository_module.memory_store,
        "find_memory_unit_by_id",
        AsyncMock(side_effect=collection.find_by_id),
    )
    monkeypatch.setattr(
        repository_module.memory_store,
        "acquire_memory_write_lock",
        AsyncMock(return_value=True),
    )
    monkeypatch.setattr(
        repository_module.memory_store,
        "release_memory_write_lock",
        AsyncMock(),
    )
    monkeypatch.setattr(
        repository_module.memory_store,
        "insert_memory_unit_document",
        AsyncMock(side_effect=collection.insert_doc),
    )
    monkeypatch.setattr(
        repository_module.memory_store,
        "update_memory_unit_fields",
        AsyncMock(side_effect=collection.update_fields),
    )
    monkeypatch.setattr(
        repository_module.memory_store,
        "update_many_memory_unit_fields",
        AsyncMock(side_effect=collection.update_many_fields),
    )
    monkeypatch.setattr(
        repository_module,
        "get_rag_cache2_runtime",
        MagicMock(return_value=runtime),
    )
    return runtime


def _document(memory_unit_id: str, *, lineage_id: str = "lineage-1") -> dict:
    return_value = {
        "memory_unit_id": memory_unit_id,
        "lineage_id": lineage_id,
        "version": 1,
        "memory_name": "Test memory",
        "content": "The durable memory content.",
        "source_global_user_id": "",
        "memory_type": "fact",
        "source_kind": MemorySourceKind.SEEDED_MANUAL,
        "authority": MemoryAuthority.MANUAL,
        "status": MemoryStatus.ACTIVE,
        "timestamp": "2026-05-05T00:00:00+00:00",
        "expiry_timestamp": None,
    }
    return return_value


@pytest.mark.asyncio
async def test_insert_memory_unit_computes_embedding_and_invalidates_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Runtime inserts own embeddings and emit a memory Cache2 event."""
    collection = _MemoryCollection()
    runtime = _patch_repository(monkeypatch, collection)

    inserted = await repository_module.insert_memory_unit(
        document=_document("unit-1"),
    )

    assert inserted["embedding"] == [0.1, 0.2]
    assert inserted["updated_at"]
    assert collection.inserted[0]["memory_unit_id"] == "unit-1"
    event = runtime.invalidate.await_args.args[0]
    assert event.source == "memory"
    assert event.reason == "memory_unit_inserted"
    repository_module.memory_store.release_memory_write_lock.assert_awaited_once()


@pytest.mark.asyncio
async def test_insert_memory_unit_preserves_evidence_refs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Evidence references survive repository normalization and persistence."""
    collection = _MemoryCollection()
    _patch_repository(monkeypatch, collection)
    document = _document("unit-evidence")
    evidence_refs = [
        {
            "reflection_run_id": "run-1",
            "captured_at": "2026-05-05T00:00:00+00:00",
            "source": "reflection",
            "message_refs": [
                {
                    "conversation_history_id": "conv-1",
                    "platform": "discord",
                    "platform_channel_id": "chan-1",
                    "channel_type": "group",
                    "timestamp": "2026-05-05T00:00:00+00:00",
                    "role": "user",
                }
            ],
        }
    ]
    document["evidence_refs"] = evidence_refs

    inserted = await repository_module.insert_memory_unit(document=document)

    assert inserted["evidence_refs"] == evidence_refs
    assert collection.inserted[0]["evidence_refs"] == evidence_refs


@pytest.mark.asyncio
async def test_insert_memory_unit_fails_when_reset_or_write_lock_is_held(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Runtime writes fail fast when another memory mutation holds the guard."""
    collection = _MemoryCollection()
    _patch_repository(monkeypatch, collection)
    repository_module.memory_store.acquire_memory_write_lock.return_value = False

    with pytest.raises(RuntimeError, match="memory write or reset"):
        await repository_module.insert_memory_unit(
            document=_document("unit-1"),
        )

    repository_module.memory_store.compute_memory_embedding.assert_not_awaited()
    repository_module.memory_store.insert_memory_unit_document.assert_not_awaited()
    repository_module.memory_store.release_memory_write_lock.assert_not_awaited()


@pytest.mark.asyncio
async def test_insert_memory_unit_rejects_caller_supplied_embedding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Callers cannot bypass repository-owned embedding generation."""
    collection = _MemoryCollection()
    _patch_repository(monkeypatch, collection)
    document = _document("unit-1")
    document["embedding"] = [9.9]

    with pytest.raises(ValueError, match="embedding"):
        await repository_module.insert_memory_unit(document=document)


@pytest.mark.asyncio
async def test_supersede_memory_unit_replaces_only_active_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Supersede inserts the next version and marks the source inactive."""
    active = _document("unit-1")
    active["embedding"] = [0.1]
    collection = _MemoryCollection([active])
    runtime = _patch_repository(monkeypatch, collection)
    replacement = _document("unit-2")
    replacement["lineage_id"] = "lineage-1"

    stored = await repository_module.supersede_memory_unit(
        active_unit_id="unit-1",
        replacement=replacement,
    )

    assert stored["version"] == 2
    assert stored["supersedes_memory_unit_ids"] == ["unit-1"]
    assert collection.docs["unit-1"]["status"] == MemoryStatus.SUPERSEDED
    event = runtime.invalidate.await_args.args[0]
    assert event.source == "memory"
    assert event.reason == "memory_unit_superseded"


@pytest.mark.asyncio
async def test_supersede_memory_unit_rejects_inactive_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Supersede fails instead of tracing forward from an inactive row."""
    inactive = _document("unit-1")
    inactive["status"] = MemoryStatus.SUPERSEDED
    collection = _MemoryCollection([inactive])
    _patch_repository(monkeypatch, collection)

    with pytest.raises(ValueError, match="active"):
        await repository_module.supersede_memory_unit(
            active_unit_id="unit-1",
            replacement=_document("unit-2"),
        )


@pytest.mark.asyncio
async def test_merge_memory_units_uses_new_lineage_for_different_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Merging different lineages requires a caller-supplied new lineage."""
    first = _document("unit-1", lineage_id="lineage-a")
    second = _document("unit-2", lineage_id="lineage-b")
    collection = _MemoryCollection([first, second])
    _patch_repository(monkeypatch, collection)
    replacement = _document("unit-3", lineage_id="lineage-new")

    stored = await repository_module.merge_memory_units(
        source_unit_ids=["unit-1", "unit-2"],
        replacement=replacement,
    )

    assert stored["lineage_id"] == "lineage-new"
    assert stored["version"] == 1
    assert stored["merged_from_memory_unit_ids"] == ["unit-1", "unit-2"]
    assert collection.docs["unit-1"]["status"] == MemoryStatus.SUPERSEDED
    assert collection.docs["unit-2"]["status"] == MemoryStatus.SUPERSEDED
