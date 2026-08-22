"""Deterministic persistence contracts for Cognition V3 chain runs."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from pymongo.errors import DuplicateKeyError

from kazusa_ai_chatbot import db as db_facade
from kazusa_ai_chatbot.db import cognition_chain_runs as chain_runs


class _ChainRunCollection:
    """In-memory collection with unique chain-run id and exact correlation."""

    def __init__(self) -> None:
        self.rows: dict[str, dict[str, object]] = {}
        self.indexes: list[dict[str, object]] = []

    async def find_one(
        self,
        query: dict[str, object],
        projection: dict[str, int] | None = None,
    ) -> dict[str, object] | None:
        """Find one chain-run row by id."""

        del projection
        chain_run_id = str(query.get("chain_run_id"))
        row = self.rows.get(chain_run_id)
        return dict(row) if row is not None else None

    async def insert_one(self, document: dict[str, object]) -> None:
        """Insert one row or raise the Mongo uniqueness race."""

        chain_run_id = str(document["chain_run_id"])
        if chain_run_id in self.rows:
            raise DuplicateKeyError("chain_run_id already exists")
        self.rows[chain_run_id] = dict(document)

    def find(self, query: dict[str, object], projection: dict[str, int] | None = None):
        """Return a cursor-like query object for exact dual-key reads."""

        del projection
        return _ChainRunCursor(self.rows.values(), query)

    async def create_index(self, keys, **kwargs: object) -> None:
        """Capture one requested index definition."""

        self.indexes.append({"keys": keys, "kwargs": kwargs})


class _ChainRunCursor:
    """Small cursor subset used by the exact dual-key read helper."""

    def __init__(self, rows, query: dict[str, object]) -> None:
        self.rows = [
            row
            for row in rows
            if all(row.get(key) == value for key, value in query.items())
        ]
        self.rows.sort(
            key=lambda row: str(row.get("completed_at", "")),
            reverse=True,
        )

    def sort(self, key, direction):
        del key, direction
        return self

    def limit(self, count):
        del count
        return self

    async def to_list(self, length: int) -> list[dict[str, object]]:
        return [dict(row) for row in self.rows[:length]]


class _ChainRunDatabase:
    """Minimal database wrapper for chain-run tests."""

    def __init__(self, collection: _ChainRunCollection) -> None:
        self.collection = collection

    async def list_collection_names(self) -> list[str]:
        return []

    async def create_collection(self, name: str) -> None:
        del name

    def __getitem__(self, name: str) -> _ChainRunCollection:
        del name
        return self.collection


def _document(
    *,
    chain_run_id: str = "cogchain_1",
    run_id: str = "run-1",
    llm_trace_id: str = "trace-1",
    cognition_invocation_id: str = "invocation-1",
) -> dict[str, object]:
    """Build one complete sanitized chain-run document."""

    return {
        "schema_version": "cognition_chain_run.v2",
        "chain_run_id": chain_run_id,
        "engine": "v3",
        "run_id": run_id,
        "llm_trace_id": llm_trace_id,
        "cognition_invocation_id": cognition_invocation_id,
        "source_kind": "live",
        "chain_model_name": "chain-model",
        "sidecar_model_name": "sidecar-model",
        "subconscious_enabled": False,
        "appraisal_stage_layout": "fixed_a1_a2",
        "started_at": "2026-08-20T00:00:00Z",
        "completed_at": "2026-08-20T00:00:01Z",
        "terminal_disposition": "complete",
        "steps": [],
        "ledger": {},
        "sidecar": {},
        "session_events": [],
        "degradation_markers": [],
        "warning_codes": [],
        "expires_at": "2026-08-22T00:00:00Z",
    }


@pytest.mark.asyncio
async def test_chain_run_upsert_is_idempotent_and_rejects_correlation_conflict(
    monkeypatch,
):
    """Duplicate IDs stay stable while immutable correlation conflicts fail."""

    collection = _ChainRunCollection()
    monkeypatch.setattr(
        chain_runs,
        "get_db",
        AsyncMock(return_value=_ChainRunDatabase(collection)),
    )

    assert await chain_runs.save_cognition_chain_run(_document()) is True
    assert await chain_runs.save_cognition_chain_run(_document()) is True
    assert await chain_runs.save_cognition_chain_run(
        _document(run_id="run-other")
    ) is False
    assert len(collection.rows) == 1


@pytest.mark.asyncio
async def test_chain_run_read_requires_exact_run_and_trace_ids(
    monkeypatch,
):
    """Reads require both exact ids and never use a global latest lookup."""

    collection = _ChainRunCollection()
    monkeypatch.setattr(
        chain_runs,
        "get_db",
        AsyncMock(return_value=_ChainRunDatabase(collection)),
    )
    collection.rows["cogchain_1"] = _document()
    collection.rows["cogchain_2"] = _document(
        chain_run_id="cogchain_2",
        llm_trace_id="trace-2",
    )

    found = await chain_runs.get_cognition_chain_run(
        run_id="run-1",
        llm_trace_id="trace-1",
    )
    assert found is not None
    assert found["chain_run_id"] == "cogchain_1"

    assert await chain_runs.get_cognition_chain_run(
        run_id="run-1",
        llm_trace_id="missing",
    ) is None
    assert await chain_runs.get_cognition_chain_run(
        run_id="",
        llm_trace_id="trace-1",
    ) is None


@pytest.mark.asyncio
async def test_chain_run_read_ignores_legacy_schema_rows(monkeypatch):
    """Reads return only complete v2 rows and ignore matching v1 rows."""

    collection = _ChainRunCollection()
    monkeypatch.setattr(
        chain_runs,
        "get_db",
        AsyncMock(return_value=_ChainRunDatabase(collection)),
    )
    legacy = _document(chain_run_id="cogchain-legacy")
    legacy["schema_version"] = "cognition_chain_run.v1"
    collection.rows["cogchain-legacy"] = legacy

    assert await chain_runs.get_cognition_chain_run(
        run_id="run-1",
        llm_trace_id="trace-1",
    ) is None

    current = _document(chain_run_id="cogchain-current")
    collection.rows["cogchain-current"] = current
    found = await chain_runs.get_cognition_chain_run(
        run_id="run-1",
        llm_trace_id="trace-1",
    )
    assert found == current


def test_db_exports_exact_chain_run_helpers() -> None:
    """The public DB facade exposes exactly the three chain-run helpers."""

    assert db_facade.save_cognition_chain_run is chain_runs.save_cognition_chain_run
    assert db_facade.get_cognition_chain_run is chain_runs.get_cognition_chain_run
    assert (
        db_facade.ensure_cognition_chain_run_indexes
        is chain_runs.ensure_cognition_chain_run_indexes
    )


@pytest.mark.asyncio
async def test_chain_run_indexes_match_retention_and_correlation_contract(
    monkeypatch,
):
    """Indexes cover uniqueness, dual correlation, invocation, engine, and TTL."""

    collection = _ChainRunCollection()
    monkeypatch.setattr(
        chain_runs,
        "get_db",
        AsyncMock(return_value=_ChainRunDatabase(collection)),
    )
    await chain_runs.ensure_cognition_chain_run_indexes()

    index_names = {entry["kwargs"]["name"] for entry in collection.indexes}
    assert {
        "cognition_chain_run_id_unique",
        "cognition_chain_run_correlation_completed",
        "cognition_chain_run_invocation_completed",
        "cognition_chain_run_engine_started",
        "cognition_chain_run_expires_at_ttl",
    } <= index_names
