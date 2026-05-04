"""Tests for memory-unit retry idempotency."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from kazusa_ai_chatbot.memory_evolution.models import (
    MemoryAuthority,
    MemorySourceKind,
    MemoryStatus,
)
from kazusa_ai_chatbot.memory_evolution import repository as repository_module


class _Collection:
    """Collection fake with one optional existing memory unit."""

    def __init__(self, existing: dict | None) -> None:
        """Create the fake collection.

        Args:
            existing: Document returned from ``find_one``.
        """
        self.existing = dict(existing) if existing is not None else None
        self.insert_one = AsyncMock()

    async def find_by_id(self, memory_unit_id: str) -> dict | None:
        """Return the configured existing row when ids match."""
        if self.existing is None:
            return None
        if memory_unit_id != self.existing["memory_unit_id"]:
            return None
        return_value = dict(self.existing)
        return return_value


def _document(content: str = "Stable content") -> dict:
    return_value = {
        "memory_unit_id": "unit-1",
        "lineage_id": "unit-1",
        "version": 1,
        "memory_name": "Memory",
        "content": content,
        "source_global_user_id": "",
        "memory_type": "fact",
        "source_kind": MemorySourceKind.SEEDED_MANUAL,
        "authority": MemoryAuthority.MANUAL,
        "status": MemoryStatus.ACTIVE,
        "timestamp": "2026-05-05T00:00:00+00:00",
        "updated_at": "2026-05-05T00:00:00+00:00",
        "expiry_timestamp": None,
        "supersedes_memory_unit_ids": [],
        "merged_from_memory_unit_ids": [],
        "evidence_refs": [],
        "privacy_review": {},
        "confidence_note": "",
    }
    return return_value


def _patch_repository(monkeypatch: pytest.MonkeyPatch, collection: _Collection) -> AsyncMock:
    embedding = AsyncMock(return_value=[0.1])
    runtime = MagicMock()
    runtime.invalidate = AsyncMock(return_value=0)
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
        collection.insert_one,
    )
    monkeypatch.setattr(
        repository_module.memory_store,
        "compute_memory_embedding",
        embedding,
    )
    monkeypatch.setattr(
        repository_module,
        "get_rag_cache2_runtime",
        MagicMock(return_value=runtime),
    )
    return embedding


@pytest.mark.asyncio
async def test_insert_memory_unit_returns_existing_equivalent_row_without_embedding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Retrying the same stable id does not duplicate rows or embeddings."""
    existing = _document()
    existing["embedding"] = [0.7]
    collection = _Collection(existing)
    embedding = _patch_repository(monkeypatch, collection)
    retry_document = _document()
    retry_document.pop("updated_at")

    stored = await repository_module.insert_memory_unit(
        document=retry_document,
    )

    assert stored["memory_unit_id"] == "unit-1"
    embedding.assert_not_awaited()
    collection.insert_one.assert_not_called()


@pytest.mark.asyncio
async def test_insert_memory_unit_rejects_same_id_with_different_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The same id cannot point at different memory semantics."""
    existing = _document()
    existing["embedding"] = [0.7]
    collection = _Collection(existing)
    _patch_repository(monkeypatch, collection)

    with pytest.raises(ValueError, match="different content"):
        await repository_module.insert_memory_unit(
            document=_document(content="Changed content"),
        )
