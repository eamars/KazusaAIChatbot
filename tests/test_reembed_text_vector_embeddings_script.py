from __future__ import annotations

from collections.abc import Iterable
from typing import Any
from unittest.mock import AsyncMock, call

import pytest

from kazusa_ai_chatbot.db import script_operations
from scripts import reembed_text_vector_embeddings as script_module


class _AsyncCursor:
    """Small async cursor fake for collection scans."""

    def __init__(self, rows: Iterable[dict[str, Any]]) -> None:
        self.rows = list(rows)

    def batch_size(self, size: int) -> "_AsyncCursor":
        """Accept Motor's batch-size call and return the same cursor."""

        return self

    def __aiter__(self) -> "_AsyncCursor":
        self._iterator = iter(self.rows)
        return self

    async def __anext__(self) -> dict[str, Any]:
        try:
            row = next(self._iterator)
        except StopIteration as exc:
            raise StopAsyncIteration from exc
        return row


class _Collection:
    """Collection fake that records updates during re-embedding."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows
        self.update_one = AsyncMock()

    async def count_documents(self, query: dict[str, Any]) -> int:
        """Return the number of stored rows."""

        return len(self.rows)

    def find(self, query: dict[str, Any]) -> _AsyncCursor:
        """Return every row for the maintenance scan."""

        return _AsyncCursor(self.rows)


class _Db:
    """Database fake for collection-name item access."""

    def __init__(self) -> None:
        self.collections = {
            "conversation_history": _Collection([
                {"_id": "c1", "body_text": "conversation row", "attachments": []},
                {"_id": "c2", "body_text": "", "attachments": []},
            ]),
            "memory": _Collection([
                {
                    "_id": "m1",
                    "memory_type": "fact",
                    "source_kind": "manual",
                    "memory_name": "Memory",
                    "content": "memory content",
                }
            ]),
            "user_memory_units": _Collection([
                {
                    "_id": "u1",
                    "fact": "user fact",
                    "subjective_appraisal": "appraisal",
                    "relationship_signal": "signal",
                }
            ]),
        }

    def __getitem__(self, collection_name: str) -> _Collection:
        """Return a fake collection by name."""

        return self.collections[collection_name]


@pytest.mark.asyncio
async def test_reembed_dry_run_counts_target_rows_without_updates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dry-run should inspect rows without calling embeddings or updating DB."""

    db = _Db()
    embed = AsyncMock(return_value=[])
    monkeypatch.setattr(script_operations, "get_db", AsyncMock(return_value=db))
    monkeypatch.setattr(
        script_operations,
        "get_document_text_embeddings_batch",
        embed,
    )

    result = await script_operations.reembed_text_vector_embeddings(
        collection_names=["conversation_history"],
        batch_size=2,
        apply=False,
    )

    collection_result = result["collections"][0]
    assert collection_result["collection"] == "conversation_history"
    assert collection_result["total_count"] == 2
    assert collection_result["processed"] == 1
    assert collection_result["skipped"] == 1
    assert collection_result["updated"] == 0
    embed.assert_not_awaited()
    db["conversation_history"].update_one.assert_not_awaited()


@pytest.mark.asyncio
async def test_reembed_apply_updates_conversation_memory_and_user_units(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Apply should update all supported text-vector collections."""

    db = _Db()
    embedding_calls: list[list[str]] = []

    async def _embed(texts: list[str]) -> list[list[float]]:
        embedding_calls.append(list(texts))
        embeddings = [[float(index + 1)] for index, _ in enumerate(texts)]
        return embeddings

    monkeypatch.setattr(script_operations, "get_db", AsyncMock(return_value=db))
    monkeypatch.setattr(
        script_operations,
        "get_document_text_embeddings_batch",
        _embed,
    )

    result = await script_operations.reembed_text_vector_embeddings(
        collection_names=["conversation_history", "memory", "user_memory_units"],
        batch_size=2,
        apply=True,
    )

    assert result["total_processed"] == 3
    assert result["total_updated"] == 3
    assert result["total_cleared"] == 1
    assert embedding_calls == [
        ["conversation row"],
        ["type:fact\nsource:manual\ntitle:Memory\ncontent:memory content"],
        ["user fact\nappraisal\nsignal"],
    ]
    assert db["conversation_history"].update_one.await_count == 2
    db["memory"].update_one.assert_awaited_once()
    db["user_memory_units"].update_one.assert_awaited_once()


@pytest.mark.asyncio
async def test_reembed_skips_empty_source_rows_and_reports_them(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rows without embedding source text should be reported as skipped."""

    db = _Db()
    monkeypatch.setattr(script_operations, "get_db", AsyncMock(return_value=db))
    monkeypatch.setattr(
        script_operations,
        "get_document_text_embeddings_batch",
        AsyncMock(return_value=[]),
    )

    result = await script_operations.reembed_text_vector_embeddings(
        collection_names=["conversation_history"],
        batch_size=100,
        apply=False,
    )

    skipped_rows = result["collections"][0]["skipped_rows"]
    assert skipped_rows == [{"row_id": "c2", "reason": "empty_source_text"}]


@pytest.mark.asyncio
async def test_reembed_apply_clears_embeddings_for_empty_source_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Apply should remove stale vectors when a row has no source text."""

    db = _Db()
    db["conversation_history"].rows[1]["embedding"] = [9.0]
    monkeypatch.setattr(script_operations, "get_db", AsyncMock(return_value=db))
    monkeypatch.setattr(
        script_operations,
        "get_document_text_embeddings_batch",
        AsyncMock(return_value=[[1.0]]),
    )

    result = await script_operations.reembed_text_vector_embeddings(
        collection_names=["conversation_history"],
        batch_size=100,
        apply=True,
    )

    collection_result = result["collections"][0]
    assert collection_result["updated"] == 1
    assert collection_result["cleared"] == 1
    assert result["total_cleared"] == 1
    db["conversation_history"].update_one.assert_has_awaits([
        call({"_id": "c1"}, {"$set": {"embedding": [1.0]}}),
        call({"_id": "c2"}, {"$unset": {"embedding": ""}}),
    ], any_order=True)


def test_reembed_script_rejects_unknown_collection() -> None:
    """The operator script should reject collections outside the approved scope."""

    with pytest.raises(SystemExit):
        script_module.parse_args([
            "--dry-run",
            "--collections",
            "unknown_collection",
        ])
