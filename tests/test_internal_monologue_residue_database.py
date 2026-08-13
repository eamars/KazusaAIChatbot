"""Deterministic persistence contracts for v2 residue rows."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from pymongo.errors import DuplicateKeyError

from kazusa_ai_chatbot import config
from kazusa_ai_chatbot.db import internal_monologue_residue as residue_db
from kazusa_ai_chatbot.db.schemas import (
    InternalMonologueResidueV2Doc,
    validate_internal_monologue_residue_v2_doc,
)


class _ResidueCollection:
    """In-memory collection with operation uniqueness semantics."""

    def __init__(self) -> None:
        self.rows: dict[str, dict[str, object]] = {}
        self.indexes: list[dict[str, object]] = []

    async def find_one(
        self,
        query: dict[str, object],
        projection: dict[str, int] | None = None,
    ) -> dict[str, object] | None:
        """Find the one operation row used by the idempotency path."""

        del projection
        operation_id = query.get('operation_id')
        row = self.rows.get(str(operation_id))
        return dict(row) if row is not None else None

    async def insert_one(self, document: dict[str, object]) -> None:
        """Insert one row or raise the Mongo uniqueness race."""

        operation_id = str(document['operation_id'])
        if operation_id in self.rows:
            raise DuplicateKeyError('operation_id already exists')
        self.rows[operation_id] = dict(document)

    async def create_index(self, keys, **kwargs: object) -> None:
        """Capture one requested index definition."""

        self.indexes.append({'keys': keys, 'kwargs': kwargs})


class _ResidueDatabase:
    """Minimal database wrapper for collection bootstrap tests."""

    def __init__(self, collection: _ResidueCollection) -> None:
        self.collection = collection

    async def list_collection_names(self) -> list[str]:
        """Start with no collection so creation remains covered."""

        return []

    async def create_collection(self, name: str) -> None:
        """Record the requested collection creation."""

        del name

    def __getitem__(self, name: str) -> _ResidueCollection:
        """Return the one residue collection."""

        del name
        return self.collection


def _row(
    *,
    residue_text: str = 'carry one related branch',
    disposition: str = 'append',
) -> dict[str, object]:
    """Build one complete v2 user-thread row."""

    return {
        'residue_id': 'residue-1',
        'character_id': 'character-1',
        'scope_key': 'user_thread:character-1:qq:group-1:user-1',
        'scope_kind': 'user_thread',
        'platform': 'qq',
        'platform_channel_id': 'group-1',
        'channel_type': 'group',
        'global_user_id': 'user-1',
        'residue_text': residue_text,
        'source_kind': 'chat',
        'source_refs': [],
        'created_at': '2026-05-20T00:10:00+00:00',
        'schema_version': 'internal_monologue_residue.v2',
        'operation_id': 'residue-v2:operation-1',
        'disposition': disposition,
        'purge_at': datetime(
            2026,
            5,
            22,
            0,
            10,
            tzinfo=timezone.utc,
        ),
    }


@pytest.mark.asyncio
async def test_residue_operation_is_transition_idempotent_and_conflict_safe(
    monkeypatch,
) -> None:
    """Duplicate operations are stable and semantic payload changes conflict."""

    collection = _ResidueCollection()
    monkeypatch.setattr(
        residue_db,
        '_collection',
        AsyncMock(return_value=collection),
    )
    first = await residue_db.insert_internal_monologue_residue_row(_row())
    duplicate = await residue_db.insert_internal_monologue_residue_row(
        _row()
    )
    conflict_row = _row(residue_text='a different branch')
    conflict = await residue_db.insert_internal_monologue_residue_row(
        conflict_row
    )

    assert first == {
        'status': 'written',
        'residue_id': 'residue-1',
    }
    assert duplicate['status'] == 'duplicate_same_payload'
    assert duplicate['residue_id'] == 'residue-1'
    assert conflict['status'] == 'conflict'
    assert conflict['residue_id'] == 'residue-1'
    assert len(collection.rows) == 1


@pytest.mark.asyncio
async def test_residue_indexes_include_operation_uniqueness_and_purge_ttl(
    monkeypatch,
) -> None:
    """Canonical bootstrap requests both transition and retention indexes."""

    collection = _ResidueCollection()
    database = _ResidueDatabase(collection)
    monkeypatch.setattr(residue_db, 'get_db', AsyncMock(return_value=database))

    await residue_db.ensure_internal_monologue_residue_indexes()

    index_names = {
        entry['kwargs']['name']
        for entry in collection.indexes
    }
    assert 'internal_monologue_residue_operation_unique' in index_names
    assert 'internal_monologue_residue_purge_at_ttl' in index_names
    ttl_index = next(
        entry for entry in collection.indexes
        if entry['kwargs']['name'] == 'internal_monologue_residue_purge_at_ttl'
    )
    assert ttl_index['kwargs']['expireAfterSeconds'] == 0


def test_v2_residue_schema_requires_disposition_operation_and_retention() -> None:
    """The required v2 schema rejects noncanonical write documents."""

    assert {
        'schema_version',
        'operation_id',
        'disposition',
        'purge_at',
    }.issubset(InternalMonologueResidueV2Doc.__required_keys__)
    valid = _row()
    validate_internal_monologue_residue_v2_doc(valid)
    invalid = dict(valid)
    invalid.pop('disposition')
    with pytest.raises(ValueError, match='missing required fields'):
        validate_internal_monologue_residue_v2_doc(invalid)


def test_residue_retention_config_is_bounded_and_documented() -> None:
    """Retention stays bounded and its default is recorded in the ICD."""

    assert 1 <= config.INTERNAL_MONOLOGUE_RESIDUE_RETENTION_HOURS <= 720
    readme = Path(
        'src/kazusa_ai_chatbot/internal_monologue_residue/README.md'
    ).read_text(encoding='utf-8')
    assert '48' in readme
    assert 'purge_at' in readme
