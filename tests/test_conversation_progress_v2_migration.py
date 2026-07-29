"""Export-and-reset migration safety contracts for conversation progress V2."""

from __future__ import annotations

import json
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from bson import ObjectId, json_util

from kazusa_ai_chatbot.db import script_operations
from tests.conversation_progress_v2_helpers import packet

_GENERATED_AT = '2026-07-28T10:00:00+00:00'


def _v1_row(
    *,
    row_id: ObjectId | None = None,
    turn_count: int = 7,
) -> dict[str, object]:
    """Build one source V1 row with semantics that must remain untranslated."""

    return {
        '_id': row_id or ObjectId(),
        'episode_state_id': 'legacy-episode',
        'platform': 'qq',
        'platform_channel_id': 'legacy-channel',
        'global_user_id': 'legacy-user',
        'status': 'active',
        'continuity': 'same_episode',
        'turn_count': turn_count,
        'episode_label': 'legacy label',
        'open_loops': [{'text': 'legacy unresolved detail'}],
        'created_at': '2026-07-28T08:00:00+00:00',
        'updated_at': '2026-07-28T09:00:00+00:00',
        'expires_at': '2026-07-30T09:00:00+00:00',
    }


class _Cursor:
    """Async list cursor for complete migration exports."""

    def __init__(self, rows):
        self.rows = rows

    async def to_list(self, *, length):
        del length
        return deepcopy(self.rows)


class _Collection:
    """In-memory replacement collection with exact-filter matching."""

    def __init__(self, rows):
        self.rows = {
            row['_id']: deepcopy(row)
            for row in rows
        }
        self.find_calls = 0
        self.find_one_calls = 0
        self.replace_calls: list[tuple[dict, dict, bool]] = []

    def find(self, query):
        assert query == {}
        self.find_calls += 1
        return _Cursor(list(self.rows.values()))

    async def find_one(self, query):
        self.find_one_calls += 1
        return deepcopy(self.rows.get(query['_id']))

    async def replace_one(self, query, document, *, upsert):
        self.replace_calls.append((
            deepcopy(query),
            deepcopy(document),
            upsert,
        ))
        row_id = query['_id']
        current = self.rows.get(row_id)
        if current is None or not _matches(current, query):
            return SimpleNamespace(modified_count=0)
        self.rows[row_id] = deepcopy(document)
        return SimpleNamespace(modified_count=1)


def _matches(current: dict, query: dict) -> bool:
    """Match the exact fields used by migration guards."""

    for field_name, expected in query.items():
        if isinstance(expected, dict) and '$exists' in expected:
            if (field_name in current) is not expected['$exists']:
                return False
            continue
        if current.get(field_name) != expected:
            return False
    return True


def _patch_collection(monkeypatch, collection):
    monkeypatch.setattr(
        script_operations,
        'get_db',
        AsyncMock(return_value={
            'conversation_episode_state': collection,
        }),
    )


@pytest.mark.parametrize(
    ('row', 'classification'),
    [
        (_v1_row(), 'v1_eligible'),
        ({**_v1_row(), 'schema_version': 'conversation_progress.v1'}, 'v1_eligible'),
        ({**_v1_row(), 'schema_version': 'conversation_progress.v2'}, 'malformed'),
        ({'_id': ObjectId(), **packet()}, 'already_v2'),
        ({**_v1_row(), 'schema_version': 'unknown.v3'}, 'malformed'),
        ({**_v1_row(), 'expires_at': 'tomorrow'}, 'malformed'),
    ],
)
def test_migration_classification_has_no_semantic_translation(
    row,
    classification,
):
    classified = (
        script_operations.classify_conversation_progress_migration_row(row)
    )

    assert classified['classification'] == classification
    assert 'episode_label' not in classified
    assert 'open_loops' not in classified


@pytest.mark.asyncio
async def test_dry_run_exports_every_row_and_performs_zero_db_writes(
    monkeypatch,
    tmp_path,
):
    rows = [_v1_row(), {
        **_v1_row(),
        'schema_version': 'conversation_progress.v2',
    }]
    collection = _Collection(rows)
    _patch_collection(monkeypatch, collection)
    backup_path = tmp_path / 'backup.json'
    report_path = tmp_path / 'dry-run.json'

    report = (
        await script_operations.dry_run_conversation_progress_v2_migration(
            backup_output=backup_path,
            report_output=report_path,
            generated_at=_GENERATED_AT,
        )
    )

    backup_text = backup_path.read_text(encoding='utf-8')
    backup = json_util.loads(backup_text)
    assert len(backup['rows']) == 2
    assert backup['rows'][0]['_id'] == rows[0]['_id']
    assert '"$oid"' in backup_text
    assert report['backup_row_count'] == 2
    assert len(report['backup_sha256']) == 64
    assert report['writes_performed'] == 0
    assert collection.replace_calls == []
    assert json.loads(report_path.read_text(encoding='utf-8')) == report


@pytest.mark.asyncio
async def test_apply_replaces_only_drift_matched_v1_with_exact_tombstone(
    monkeypatch,
    tmp_path,
):
    source = _v1_row()
    collection = _Collection([source])
    _patch_collection(monkeypatch, collection)
    backup_path = tmp_path / 'backup.json'
    dry_path = tmp_path / 'dry-run.json'
    apply_path = tmp_path / 'apply.json'
    await script_operations.dry_run_conversation_progress_v2_migration(
        backup_output=backup_path,
        report_output=dry_path,
        generated_at=_GENERATED_AT,
    )

    report = await script_operations.apply_conversation_progress_v2_migration(
        dry_run_input=dry_path,
        backup_input=backup_path,
        output=apply_path,
        applied_at='2026-07-28T10:05:00+00:00',
    )

    tombstone = collection.rows[source['_id']]
    assert report['counts']['changed'] == 1
    assert report['unrelated_collection_writes'] == 0
    assert report['conversation_history_deletes'] == 0
    assert tombstone['_id'] == source['_id']
    assert tombstone['schema_version'] == 'conversation_progress.v2'
    assert tombstone['status'] == 'closed'
    assert tombstone['turn_count'] == 0
    assert tombstone['events'] == []
    assert tombstone['recent_turn_refs'] == []
    assert tombstone['compacted_block_refs'] == []
    assert set(tombstone).isdisjoint({'episode_label', 'open_loops'})
    assert tombstone['purge_after'].isoformat() == (
        '2026-07-30T10:05:00+00:00'
    )
    query, _, upsert = collection.replace_calls[0]
    assert query['turn_count'] == source['turn_count']
    assert query['updated_at'] == source['updated_at']
    assert query['schema_version'] == {'$exists': False}
    assert upsert is False


@pytest.mark.asyncio
async def test_apply_skips_row_changed_after_review(
    monkeypatch,
    tmp_path,
):
    source = _v1_row()
    collection = _Collection([source])
    _patch_collection(monkeypatch, collection)
    backup_path = tmp_path / 'backup.json'
    dry_path = tmp_path / 'dry-run.json'
    await script_operations.dry_run_conversation_progress_v2_migration(
        backup_output=backup_path,
        report_output=dry_path,
        generated_at=_GENERATED_AT,
    )
    collection.rows[source['_id']]['turn_count'] = 8

    report = await script_operations.apply_conversation_progress_v2_migration(
        dry_run_input=dry_path,
        backup_input=backup_path,
        output=tmp_path / 'apply.json',
        applied_at='2026-07-28T10:05:00+00:00',
    )

    assert report['counts']['changed'] == 0
    assert report['counts']['drift_skipped'] == 1
    assert collection.rows[source['_id']]['turn_count'] == 8
    assert collection.replace_calls == []


@pytest.mark.asyncio
async def test_apply_rejects_backup_not_bound_to_reviewed_digest(
    monkeypatch,
    tmp_path,
):
    collection = _Collection([_v1_row()])
    _patch_collection(monkeypatch, collection)
    backup_path = tmp_path / 'backup.json'
    dry_path = tmp_path / 'dry-run.json'
    await script_operations.dry_run_conversation_progress_v2_migration(
        backup_output=backup_path,
        report_output=dry_path,
        generated_at=_GENERATED_AT,
    )
    backup_path.write_text('{}', encoding='utf-8')

    with pytest.raises(ValueError, match='digest'):
        await script_operations.apply_conversation_progress_v2_migration(
            dry_run_input=dry_path,
            backup_input=backup_path,
            output=tmp_path / 'apply.json',
            applied_at='2026-07-28T10:05:00+00:00',
        )

    assert collection.replace_calls == []


@pytest.mark.asyncio
async def test_restore_requires_exact_tombstone_and_restores_full_v1_row(
    monkeypatch,
    tmp_path,
):
    source = _v1_row()
    collection = _Collection([source])
    _patch_collection(monkeypatch, collection)
    backup_path = tmp_path / 'backup.json'
    dry_path = tmp_path / 'dry-run.json'
    apply_path = tmp_path / 'apply.json'
    restore_path = tmp_path / 'restore.json'
    await script_operations.dry_run_conversation_progress_v2_migration(
        backup_output=backup_path,
        report_output=dry_path,
        generated_at=_GENERATED_AT,
    )
    await script_operations.apply_conversation_progress_v2_migration(
        dry_run_input=dry_path,
        backup_input=backup_path,
        output=apply_path,
        applied_at='2026-07-28T10:05:00+00:00',
    )
    applied_tombstone = deepcopy(collection.rows[source['_id']])

    report = await script_operations.restore_conversation_progress_v1_backup(
        apply_input=apply_path,
        backup_input=backup_path,
        output=restore_path,
        restored_at='2026-07-28T10:10:00+00:00',
    )

    assert report['counts']['restored'] == 1
    assert collection.rows[source['_id']] == source
    restore_query = collection.replace_calls[-1][0]
    assert restore_query == applied_tombstone


@pytest.mark.asyncio
async def test_restore_never_overwrites_new_active_v2_packet(
    monkeypatch,
    tmp_path,
):
    source = _v1_row()
    collection = _Collection([source])
    _patch_collection(monkeypatch, collection)
    backup_path = tmp_path / 'backup.json'
    dry_path = tmp_path / 'dry-run.json'
    apply_path = tmp_path / 'apply.json'
    await script_operations.dry_run_conversation_progress_v2_migration(
        backup_output=backup_path,
        report_output=dry_path,
        generated_at=_GENERATED_AT,
    )
    await script_operations.apply_conversation_progress_v2_migration(
        dry_run_input=dry_path,
        backup_input=backup_path,
        output=apply_path,
        applied_at='2026-07-28T10:05:00+00:00',
    )
    active_packet = deepcopy(collection.rows[source['_id']])
    active_packet['status'] = 'active'
    active_packet['turn_count'] = 1
    collection.rows[source['_id']] = active_packet
    call_count_before_restore = len(collection.replace_calls)

    report = await script_operations.restore_conversation_progress_v1_backup(
        apply_input=apply_path,
        backup_input=backup_path,
        output=tmp_path / 'restore.json',
        restored_at='2026-07-28T10:10:00+00:00',
    )

    assert report['counts']['active_v2_skipped'] == 1
    assert collection.rows[source['_id']] == active_packet
    assert len(collection.replace_calls) == call_count_before_restore
