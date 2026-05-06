"""Tests for the offline memory-writer perspective migration CLI."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from scripts import sanitize_memory_writer_perspective as sanitizer


pytestmark = pytest.mark.asyncio

CHARACTER_NAME = '杏山千纱 (Kyōyama Kazusa)'


async def test_dry_run_report_rewrites_ready_records(monkeypatch) -> None:
    """Dry-run should produce reviewable before/after rows."""

    scan_record = {
        'collection': sanitizer.SCOPE_USER_MEMORY_UNITS,
        'document_key': 'unit_id',
        'document_id': 'unit-1',
        'before': {
            'fact': '用户要求助理不要再叫自己亲爱的。',
            'subjective_appraisal': '角色意识到这是用户的称呼边界。',
            'relationship_signal': 'assistant以后应避免亲昵称呼。',
        },
    }
    monkeypatch.setattr(
        sanitizer,
        'get_character_profile',
        AsyncMock(return_value={'name': CHARACTER_NAME}),
    )
    monkeypatch.setattr(
        sanitizer,
        'scan_memory_writer_records',
        AsyncMock(return_value=[scan_record]),
    )

    async def _rewrite(record, *, character_name):
        assert record is scan_record
        assert character_name == CHARACTER_NAME
        return {
            'fields': {
                'fact': '用户要求以后不要被称呼为亲爱的。',
                'subjective_appraisal': (
                    f'{CHARACTER_NAME}理解这是用户明确的称呼边界。'
                ),
                'relationship_signal': (
                    f'{CHARACTER_NAME}后续应避免使用亲爱的这类称呼。'
                ),
            },
            'notes': ['rewritten through migration prompt'],
        }

    monkeypatch.setattr(sanitizer, 'run_migration_rewrite_llm', _rewrite)

    report = await sanitizer.build_dry_run_report(
        scopes={sanitizer.SCOPE_USER_MEMORY_UNITS},
        limit=10,
    )

    assert report['dry_run'] is True
    assert report['character_name'] == CHARACTER_NAME
    assert report['records_seen'] == 1
    assert report['records_with_proposed_changes'] == 1
    assert report['records'][0]['status'] == 'ready'
    assert report['records'][0]['before'] == scan_record['before']
    assert CHARACTER_NAME in report['records'][0]['after']['subjective_appraisal']


async def test_dry_run_blocks_malformed_llm_rows(monkeypatch) -> None:
    """Dry-run should block rows when the LLM output is structurally invalid."""

    scan_record = {
        'collection': sanitizer.SCOPE_USER_PROFILES,
        'document_key': 'global_user_id',
        'document_id': 'user-1',
        'before': {'last_relationship_insight': '角色觉得用户值得信任。'},
    }
    monkeypatch.setattr(
        sanitizer,
        'get_character_profile',
        AsyncMock(return_value={'name': CHARACTER_NAME}),
    )
    monkeypatch.setattr(
        sanitizer,
        'scan_memory_writer_records',
        AsyncMock(return_value=[scan_record]),
    )
    monkeypatch.setattr(
        sanitizer,
        'run_migration_rewrite_llm',
        AsyncMock(return_value={'notes': ['missing fields']}),
    )

    report = await sanitizer.build_dry_run_report(
        scopes={sanitizer.SCOPE_USER_PROFILES},
        limit=10,
    )

    assert report['records_seen'] == 1
    assert report['records_with_proposed_changes'] == 0
    assert report['records'][0]['status'] == 'blocked'
    assert report['records'][0]['after'] == scan_record['before']


async def test_dry_run_honors_llm_blocked_status(monkeypatch) -> None:
    """Dry-run should preserve original fields when the LLM marks ambiguity."""

    scan_record = {
        'collection': sanitizer.SCOPE_USER_MEMORY_UNITS,
        'document_key': 'unit_id',
        'document_id': 'unit-ambiguous',
        'before': {
            'fact': 'I should remember this promise.',
            'subjective_appraisal': 'I felt unsure.',
            'relationship_signal': 'Keep this in mind.',
        },
    }
    monkeypatch.setattr(
        sanitizer,
        'get_character_profile',
        AsyncMock(return_value={'name': CHARACTER_NAME}),
    )
    monkeypatch.setattr(
        sanitizer,
        'scan_memory_writer_records',
        AsyncMock(return_value=[scan_record]),
    )
    monkeypatch.setattr(
        sanitizer,
        'run_migration_rewrite_llm',
        AsyncMock(return_value={
            'status': 'blocked',
            'fields': {
                'fact': f'{CHARACTER_NAME} should remember this promise.',
                'subjective_appraisal': f'{CHARACTER_NAME} felt unsure.',
                'relationship_signal': 'Keep this in mind.',
            },
            'notes': ['ambiguous first person'],
        }),
    )

    report = await sanitizer.build_dry_run_report(
        scopes={sanitizer.SCOPE_USER_MEMORY_UNITS},
        limit=10,
    )

    assert report['records_seen'] == 1
    assert report['records_with_proposed_changes'] == 0
    assert report['records'][0]['status'] == 'blocked'
    assert report['records'][0]['after'] == scan_record['before']


async def test_apply_report_uses_existing_user_and_character_helpers(
    monkeypatch,
) -> None:
    """Apply should route reviewed rows through owning helper functions."""

    calls = []

    async def _update_unit(unit_id, after, **kwargs):
        calls.append(('unit', unit_id, after, kwargs))

    async def _update_insight(global_user_id, insight):
        calls.append(('insight', global_user_id, insight))

    async def _get_character_state():
        calls.append(('get_character_state',))
        return {'mood': 'Neutral', 'global_vibe': 'Calm'}

    async def _upsert_state(mood, global_vibe, reflection_summary, timestamp):
        calls.append(('state', mood, global_vibe, reflection_summary, timestamp))

    async def _upsert_self_image(image_doc):
        calls.append(('self_image', image_doc))

    monkeypatch.setattr(sanitizer, 'update_user_memory_unit_semantics', _update_unit)
    monkeypatch.setattr(sanitizer, 'update_last_relationship_insight', _update_insight)
    monkeypatch.setattr(sanitizer, 'get_character_state', _get_character_state)
    monkeypatch.setattr(sanitizer, 'upsert_character_state', _upsert_state)
    monkeypatch.setattr(sanitizer, 'upsert_character_self_image', _upsert_self_image)

    report = {
        'dry_run': True,
        'prompt_version': sanitizer.MIGRATION_PROMPT_VERSION,
        'records': [
            {
                'collection': sanitizer.SCOPE_USER_MEMORY_UNITS,
                'document_id': 'unit-1',
                'status': 'ready',
                'after': {
                    'fact': '用户要求以后不要被称呼为亲爱的。',
                    'subjective_appraisal': f'{CHARACTER_NAME}理解这是称呼边界。',
                    'relationship_signal': f'{CHARACTER_NAME}后续应避免该称呼。',
                },
            },
            {
                'collection': sanitizer.SCOPE_USER_PROFILES,
                'document_id': 'user-1',
                'status': 'ready',
                'after': {
                    'last_relationship_insight': (
                        f'{CHARACTER_NAME}认为用户愿意清楚说明边界。'
                    ),
                },
            },
            {
                'collection': sanitizer.SCOPE_CHARACTER_STATE,
                'document_id': 'global',
                'status': 'ready',
                'after': {
                    'reflection_summary': f'{CHARACTER_NAME}平稳接住了说明。',
                    'self_image': {
                        'recent_window': [
                            {'summary': f'{CHARACTER_NAME}更重视边界表达。'},
                        ],
                    },
                },
            },
            {
                'collection': sanitizer.SCOPE_USER_PROFILES,
                'document_id': 'user-2',
                'status': 'unchanged',
                'after': {'last_relationship_insight': 'unchanged'},
            },
        ],
    }

    apply_result = await sanitizer.apply_report(report)

    assert apply_result['applied_count'] == 3
    assert apply_result['skipped_count'] == 1
    assert apply_result['blocked_count'] == 0
    assert calls[0][0] == 'unit'
    assert calls[0][3]['increment_count'] is False
    assert calls[1] == (
        'insight',
        'user-1',
        f'{CHARACTER_NAME}认为用户愿意清楚说明边界。',
    )
    assert calls[2] == ('get_character_state',)
    assert calls[3][0] == 'state'
    assert calls[4][0] == 'self_image'


async def test_apply_persistent_memory_uses_supersede(monkeypatch) -> None:
    """Shared memory apply should use the evolving-memory supersede API."""

    calls = []

    class _MemoryCollection:
        async def find_one(self, query, projection):
            calls.append(('find_one', query, projection))
            return {
                'memory_unit_id': 'memory-1',
                'lineage_id': 'lineage-1',
                'version': 1,
                'memory_name': '旧标题',
                'content': '旧内容',
                'source_global_user_id': '',
                'memory_type': 'defense_rule',
                'source_kind': 'reflection_inferred',
                'authority': 'reflection_promoted',
                'status': 'active',
                'supersedes_memory_unit_ids': [],
                'merged_from_memory_unit_ids': [],
                'evidence_refs': [],
                'privacy_review': {
                    'private_detail_risk': 'low',
                    'user_details_removed': True,
                    'boundary_assessment': '低风险',
                    'reviewer': 'automated_llm',
                },
                'confidence_note': 'reflection promoted',
                'timestamp': '2026-05-06T00:00:00+00:00',
                'updated_at': '2026-05-06T00:00:00+00:00',
                'expiry_timestamp': None,
            }

    class _Db:
        memory = _MemoryCollection()

    async def _supersede_memory_unit(*, active_unit_id, replacement):
        calls.append(('supersede', active_unit_id, replacement))
        return replacement

    monkeypatch.setattr(sanitizer, 'get_db', AsyncMock(return_value=_Db()))
    monkeypatch.setattr(
        sanitizer,
        'supersede_memory_unit',
        _supersede_memory_unit,
    )

    await sanitizer.apply_record({
        'collection': sanitizer.SCOPE_MEMORY,
        'document_id': 'memory-1',
        'after': {
            'memory_name': '公开群规事实性',
            'content': f'{CHARACTER_NAME}同意公开群规只记录事实性规则。',
        },
    })

    assert calls[0][0] == 'find_one'
    assert calls[1][0] == 'supersede'
    assert calls[1][1] == 'memory-1'
    replacement = calls[1][2]
    assert replacement['memory_unit_id'] != 'memory-1'
    assert replacement['memory_name'] == '公开群规事实性'
    assert CHARACTER_NAME in replacement['content']
