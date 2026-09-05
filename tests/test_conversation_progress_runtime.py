"""Canonical facade runtime load, selection, and record contracts."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from kazusa_ai_chatbot.conversation_progress import cache, runtime
from kazusa_ai_chatbot.conversation_progress.policy import (
    prune_aged_progress_packet,
)
from kazusa_ai_chatbot.conversation_progress.recorder import (
    ConversationProgressContextLimitError,
    ConversationProgressRecorderOutputError,
    RecorderInvocationResult,
)
from kazusa_ai_chatbot.conversation_progress.repository import (
    ProgressWriteResult,
)
from tests.conversation_progress_v2_helpers import (
    SCOPE,
    event,
    event_update,
    packet,
    recorder_delta,
    record_input,
)


def _row(row_id: str, timestamp: str) -> dict[str, object]:
    """Build one canonical user history row."""

    return {
        '_id': row_id,
        'role': 'user',
        'timestamp': timestamp,
        'body_text': f'message {row_id}',
        'display_name': 'Test User',
        'platform_user_id': 'platform-user',
        'global_user_id': 'user_test',
        'addressed_to_global_user_ids': [],
        'broadcast': False,
        'reply_context': {},
        'llm_trace_id': f'trace-{row_id}',
    }


def setup_function() -> None:
    """Clear process-local packet state before each runtime test."""

    cache.clear_cache()


@pytest.mark.asyncio
async def test_load_empty_state_uses_independent_history_lanes(monkeypatch):
    ambient_rows = [
        _row('ambient-a', '2026-07-28T09:00:00+00:00'),
        _row('current-row', '2026-07-28T09:01:00+00:00'),
    ]
    participant_rows = [
        _row('participant-a', '2026-07-28T09:02:00+00:00'),
        _row('current-row', '2026-07-28T09:03:00+00:00'),
    ]
    state_read = AsyncMock(return_value=None)
    ambient_read = AsyncMock(return_value=ambient_rows)
    participant_read = AsyncMock(return_value=participant_rows)
    monkeypatch.setattr(runtime, 'load_active_packet', state_read)
    monkeypatch.setattr(
        runtime,
        'get_ambient_conversation_history',
        ambient_read,
    )
    monkeypatch.setattr(
        runtime,
        'get_participant_conversation_history',
        participant_read,
    )

    result = await runtime.ConversationProgressRuntime().load(
        scope=SCOPE,
        current_timestamp_utc='2026-07-28T09:30:00+00:00',
        platform_bot_id='platform-bot',
        active_turn_conversation_row_ids=['current-row'],
    )

    assert result['source'] == 'empty'
    assert result['conversation_progress']['turn_count'] == 0
    assert [
        turn['conversation_row_ids']
        for turn in result['ambient_logical_turns']
    ] == [['ambient-a']]
    assert [
        turn['conversation_row_ids']
        for turn in result['interaction_logical_turns']
    ] == [['participant-a']]
    assert result['diagnostics']['ambient_rows_scanned'] == 2
    assert result['diagnostics']['interaction_rows_scanned'] == 2
    participant_read.assert_awaited_once_with(
        platform='qq',
        platform_channel_id='channel_test',
        current_global_user_id='user_test',
        platform_bot_id='platform-bot',
        excluded_row_ids=['current-row'],
        limit=128,
    )


@pytest.mark.asyncio
async def test_load_group_scene_preserves_current_user_anchors_before_ambient_cap(
    monkeypatch,
):
    """Group mode reserves the current branch before newer public noise."""

    assistant_row = _row(
        'assistant-anchor',
        '2026-07-28T09:02:00+00:00',
    )
    assistant_row.update({
        'role': 'assistant',
        'body_text': 'I will stay with you while you recover.',
        'display_name': 'Asuna',
        'global_user_id': 'character-1',
        'platform_user_id': 'bot',
        'addressed_to_global_user_ids': ['user_test'],
        'reply_context': {
            'reply_to_global_user_id': 'user_test',
            'reply_to_display_name': 'Test User',
        },
        'broadcast': False,
        'llm_trace_id': 'trace-assistant-anchor',
    })
    ambient_rows = [
        _row('current-user', '2026-07-28T09:01:00+00:00'),
        assistant_row,
    ]
    for index in range(10):
        noise_row = _row(
            f'noise-{index}',
            f'2026-07-28T09:{3 + index:02d}:00+00:00',
        )
        noise_row['global_user_id'] = f'noise-user-{index}'
        noise_row['platform_user_id'] = f'noise-platform-{index}'
        ambient_rows.append(noise_row)
    monkeypatch.setattr(
        runtime,
        'load_active_packet',
        AsyncMock(return_value=None),
    )
    monkeypatch.setattr(
        runtime,
        'get_ambient_conversation_history',
        AsyncMock(return_value=ambient_rows),
    )
    monkeypatch.setattr(
        runtime,
        'get_participant_conversation_history',
        AsyncMock(return_value=[]),
    )
    monkeypatch.setattr(
        runtime.event_logging,
        'record_continuity_boundary_event',
        AsyncMock(),
    )

    result = await runtime.ConversationProgressRuntime().load(
        scope=SCOPE,
        current_timestamp_utc='2026-07-28T09:30:00+00:00',
        platform_bot_id='platform-bot',
        active_turn_conversation_row_ids=[],
        group_scene_mode='group',
        group_scene_current_user_id='user_test',
    )

    selected_ids = [
        row_id
        for turn in result['ambient_logical_turns']
        for row_id in turn['conversation_row_ids']
    ]
    assert 'current-user' in selected_ids
    assert 'assistant-anchor' in selected_ids
    assert result['diagnostics']['protected_anchor_count'] == 2


@pytest.mark.asyncio
async def test_progress_diagnostics_expose_packet_age_and_anchor_counts(
    monkeypatch,
):
    """Load diagnostics expose bounded age labels and exact anchor counts."""

    diagnostic_packet = packet(
        events=[event(source_refs=[{
            'ref_kind': 'conversation_row',
            'ref_id': 'old-source',
            'occurred_at': '2026-07-28T08:59:00+00:00',
        }])],
    )
    diagnostic_packet['updated_at'] = '2026-07-28T09:20:00+00:00'
    monkeypatch.setattr(
        runtime,
        'load_active_packet',
        AsyncMock(return_value=diagnostic_packet),
    )
    monkeypatch.setattr(
        runtime,
        'get_ambient_conversation_history',
        AsyncMock(return_value=[]),
    )
    monkeypatch.setattr(
        runtime,
        'get_participant_conversation_history',
        AsyncMock(return_value=[]),
    )
    event_recorder = AsyncMock()
    monkeypatch.setattr(
        runtime.event_logging,
        'record_continuity_boundary_event',
        event_recorder,
    )

    result = await runtime.ConversationProgressRuntime().load(
        scope=SCOPE,
        current_timestamp_utc='2026-07-28T09:30:00+00:00',
        platform_bot_id='platform-bot',
        active_turn_conversation_row_ids=[],
        group_scene_mode='private',
    )

    assert result['diagnostics']['packet_age'] == 'recent'
    assert result['diagnostics']['source_age'] == 'stale'
    assert result['diagnostics']['protected_anchor_count'] == 0
    event_recorder.assert_awaited_once()
    assert event_recorder.await_args.kwargs['packet_age'] == 'recent'
    assert event_recorder.await_args.kwargs['source_age'] == 'stale'


@pytest.mark.asyncio
async def test_progress_diagnostics_classify_guarded_write_outcomes(
    monkeypatch,
):
    """A guarded-write ambiguity is reconciled and labelled explicitly."""

    invocation = RecorderInvocationResult(
        delta=recorder_delta(event_updates=[event_update()]),
        recorder_call_count=2,
        event_attempt_count=1,
        scene_attempt_count=1,
        event_disposition='accepted',
        scene_disposition='accepted',
        event_human_payload_chars=800,
        scene_human_payload_chars=400,
        provider_usage={},
    )
    monkeypatch.setattr(
        runtime,
        'load_referenced_blocks',
        AsyncMock(return_value=[]),
    )
    monkeypatch.setattr(
        runtime,
        'record_with_llm',
        AsyncMock(return_value=invocation),
    )
    monkeypatch.setattr(
        runtime,
        'persist_progress_write',
        AsyncMock(return_value=ProgressWriteResult(
            written=False,
            block_inserted=False,
            disposition='lost_guarded_write',
        )),
    )
    monkeypatch.setattr(
        runtime,
        '_reconcile_progress_operation',
        AsyncMock(return_value='reconciled_absent'),
    )
    event_recorder = AsyncMock()
    monkeypatch.setattr(
        runtime.event_logging,
        'record_continuity_boundary_event',
        event_recorder,
    )

    result = await runtime.ConversationProgressRuntime().record(
        record_input=record_input(),
    )

    assert result['diagnostics']['write_disposition'] == (
        'lost_guarded_write'
    )
    assert result['reconciliation_status'] == 'reconciled_absent'
    event_recorder.assert_awaited_once()
    assert event_recorder.await_args.kwargs['status'] == 'reconciled'
    assert event_recorder.await_args.kwargs['write_disposition'] == (
        'reconciled_absent'
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('recent_turn_refs', 'expected_disposition'),
    [
        (['row:row_source_1'], 'reconciled_written'),
        (['trace:trace_current'], 'reconciled_written'),
        (['trace:another_turn'], 'reconciled_absent'),
    ],
)
async def test_interrupted_record_does_not_publish_uncommitted_cache_state(
    monkeypatch,
    recent_turn_refs,
    expected_disposition,
):
    """Audit the actual committed references, then propagate cancellation."""

    monkeypatch.setattr(
        runtime,
        'load_referenced_blocks',
        AsyncMock(return_value=[]),
    )
    monkeypatch.setattr(
        runtime,
        'record_with_llm',
        AsyncMock(side_effect=asyncio.CancelledError()),
    )
    monkeypatch.setattr(
        runtime,
        'load_active_packet',
        AsyncMock(return_value=packet(recent_turn_refs=recent_turn_refs)),
    )
    event_recorder = AsyncMock()
    monkeypatch.setattr(
        runtime.event_logging,
        'record_continuity_boundary_event',
        event_recorder,
    )

    with pytest.raises(asyncio.CancelledError):
        await runtime.ConversationProgressRuntime().record(
            record_input=record_input(),
        )

    assert cache.get_cached_packet(
        scope=SCOPE,
        current_timestamp_utc='2026-07-28T09:30:00+00:00',
    ) is None
    event_recorder.assert_awaited_once()
    assert event_recorder.await_args.kwargs['status'] == 'reconciled'
    assert event_recorder.await_args.kwargs['write_disposition'] == (
        expected_disposition
    )


def test_database_wins_equal_turn_count_and_cache_wins_only_when_newer():
    db_packet = packet(turn_count=5)
    cached_equal = packet(turn_count=5)
    cached_equal['episode_narrative'] = 'divergent equal cache'
    selected, source = runtime._select_packet(db_packet, cached_equal)
    assert selected == db_packet
    assert source == 'db'

    cached_newer = packet(turn_count=6)
    selected, source = runtime._select_packet(db_packet, cached_newer)
    assert selected == cached_newer
    assert source == 'cache'


@pytest.mark.asyncio
async def test_load_prunes_aged_events_before_projection_and_cache(
    monkeypatch,
):
    """Prune on the read path without issuing any database write."""

    aged = packet(turn_count=2, events=[
        event(
            event_id='aged_background',
            retention='background',
        ),
        event(
            event_id='fresh_event',
            retention='decision_critical',
        ),
    ])
    aged['events'][0]['updated_at'] = '2026-07-27T09:00:00+00:00'
    aged['events'][1]['updated_at'] = '2026-07-28T09:00:00+00:00'
    monkeypatch.setattr(
        runtime,
        'load_active_packet',
        AsyncMock(return_value=aged),
    )
    monkeypatch.setattr(
        runtime,
        'get_ambient_conversation_history',
        AsyncMock(return_value=[]),
    )
    monkeypatch.setattr(
        runtime,
        'get_participant_conversation_history',
        AsyncMock(return_value=[]),
    )
    persist_write = AsyncMock()
    monkeypatch.setattr(runtime, 'persist_progress_write', persist_write)

    result = await runtime.ConversationProgressRuntime().load(
        scope=SCOPE,
        current_timestamp_utc='2026-07-28T09:30:00+00:00',
        platform_bot_id='platform-bot',
        active_turn_conversation_row_ids=[],
    )

    assert [row['event_id'] for row in result['episode_state']['events']] == [
        'fresh_event'
    ]
    assert [
        row['event_id']
        for row in result['conversation_progress']['events']
    ] == ['fresh_event']
    cached = cache.get_cached_packet(
        scope=SCOPE,
        current_timestamp_utc='2026-07-28T09:30:00+00:00',
    )
    assert [row['event_id'] for row in cached['events']] == ['fresh_event']
    persist_write.assert_not_awaited()


@pytest.mark.asyncio
async def test_record_persists_pruned_prior_packet_form(monkeypatch):
    """The next recorded turn persists the read-path pruned packet form."""

    aged = packet(turn_count=3, events=[
        event(
            event_id='aged_background',
            retention='background',
        ),
        event(
            event_id='surviving_event',
            retention='active_scene',
        ),
    ])
    aged['events'][0]['updated_at'] = '2026-07-27T09:00:00+00:00'
    pruned, dropped_count, _ = prune_aged_progress_packet(
        aged,
        current_timestamp_utc='2026-07-28T09:30:00+00:00',
    )
    assert dropped_count == 1
    invocation = RecorderInvocationResult(
        delta=recorder_delta(event_updates=[
            event_update(event_id='surviving_event')
        ]),
        recorder_call_count=2,
        event_attempt_count=1,
        scene_attempt_count=1,
        event_disposition='accepted',
        scene_disposition='accepted',
        event_human_payload_chars=800,
        scene_human_payload_chars=400,
        provider_usage={},
    )
    monkeypatch.setattr(
        runtime,
        'load_referenced_blocks',
        AsyncMock(return_value=[]),
    )
    monkeypatch.setattr(
        runtime,
        'record_with_llm',
        AsyncMock(return_value=invocation),
    )
    persistence = AsyncMock(return_value=ProgressWriteResult(
        written=True,
        block_inserted=False,
        disposition='written',
    ))
    monkeypatch.setattr(runtime, 'persist_progress_write', persistence)

    result = await runtime.ConversationProgressRuntime().record(
        record_input=record_input(prior_packet=pruned),
    )

    assert result['written'] is True
    prepared = persistence.await_args.args[0]
    assert [row['event_id'] for row in prepared.packet['events']] == [
        'surviving_event'
    ]


@pytest.mark.asyncio
async def test_successful_record_prepares_persists_and_caches_v2_packet(
    monkeypatch,
):
    submitted = record_input()
    invocation = RecorderInvocationResult(
        delta=recorder_delta(event_updates=[event_update()]),
        recorder_call_count=2,
        event_attempt_count=1,
        scene_attempt_count=1,
        event_disposition='accepted',
        scene_disposition='accepted',
        event_human_payload_chars=800,
        scene_human_payload_chars=400,
        provider_usage={'event': {'input_tokens': 100}, 'scene': {}},
    )
    monkeypatch.setattr(
        runtime,
        'load_referenced_blocks',
        AsyncMock(return_value=[]),
    )
    monkeypatch.setattr(
        runtime,
        'record_with_llm',
        AsyncMock(return_value=invocation),
    )
    persistence = AsyncMock(return_value=ProgressWriteResult(
        written=True,
        block_inserted=False,
        disposition='written',
    ))
    monkeypatch.setattr(runtime, 'persist_progress_write', persistence)

    result = await runtime.ConversationProgressRuntime().record(
        record_input=submitted,
    )

    assert result['written'] is True
    assert result['turn_count'] == 1
    assert result['cache_updated'] is True
    assert result['diagnostics']['recorder_call_count'] == 2
    assert result['diagnostics']['event_attempt_count'] == 1
    assert result['diagnostics']['scene_attempt_count'] == 1
    prepared = persistence.await_args.args[0]
    assert prepared.packet['schema_version'] == 'conversation_progress.v2'
    assert len(prepared.packet['events']) == 1
    cached = cache.get_cached_packet(
        scope=SCOPE,
        current_timestamp_utc='2026-07-28T09:30:00+00:00',
    )
    assert cached == prepared.packet


@pytest.mark.asyncio
async def test_lost_guarded_write_does_not_publish_packet_to_cache(
    monkeypatch,
):
    submitted = record_input()
    invocation = RecorderInvocationResult(
        delta=recorder_delta(event_updates=[event_update()]),
        recorder_call_count=2,
        event_attempt_count=1,
        scene_attempt_count=1,
        event_disposition='accepted',
        scene_disposition='accepted',
        event_human_payload_chars=800,
        scene_human_payload_chars=400,
        provider_usage={},
    )
    monkeypatch.setattr(
        runtime,
        'load_referenced_blocks',
        AsyncMock(return_value=[]),
    )
    monkeypatch.setattr(
        runtime,
        'record_with_llm',
        AsyncMock(return_value=invocation),
    )
    monkeypatch.setattr(
        runtime,
        'persist_progress_write',
        AsyncMock(return_value=ProgressWriteResult(
            written=False,
            block_inserted=False,
            disposition='lost_guarded_write',
        )),
    )

    result = await runtime.ConversationProgressRuntime().record(
        record_input=submitted,
    )

    assert result['written'] is False
    assert result['cache_updated'] is False
    assert result['diagnostics']['write_disposition'] == (
        'lost_guarded_write'
    )
    assert cache.get_cached_packet(
        scope=SCOPE,
        current_timestamp_utc='2026-07-28T09:30:00+00:00',
    ) is None


@pytest.mark.asyncio
async def test_invalid_event_output_retains_prior_packet(
    monkeypatch,
):
    prior = packet(turn_count=7)
    monkeypatch.setattr(
        runtime,
        'load_referenced_blocks',
        AsyncMock(return_value=[]),
    )
    monkeypatch.setattr(
        runtime,
        'record_with_llm',
        AsyncMock(side_effect=(
            ConversationProgressRecorderOutputError(
                'invalid',
                recorder_call_count=2,
                event_attempt_count=1,
                scene_attempt_count=1,
                event_disposition='failed_contract_or_provider',
                scene_disposition='accepted',
            )
        )),
    )

    result = await runtime.ConversationProgressRuntime().record(
        record_input=record_input(prior_packet=prior),
    )

    assert result['written'] is False
    assert result['turn_count'] == 7
    assert result['status'] == 'active'
    assert result['diagnostics']['recorder_call_count'] == 2
    assert result['diagnostics']['event_attempt_count'] == 1
    assert result['diagnostics']['scene_attempt_count'] == 1
    assert result['diagnostics']['event_disposition'] == (
        'failed_contract_or_provider'
    )
    assert result['diagnostics']['write_disposition'].startswith(
        'failed:ConversationProgressRecorderOutputError'
    )


@pytest.mark.asyncio
async def test_preflight_context_failure_reports_zero_model_attempts(
    monkeypatch,
):
    prior = packet(turn_count=7)
    monkeypatch.setattr(
        runtime,
        'load_referenced_blocks',
        AsyncMock(return_value=[]),
    )
    monkeypatch.setattr(
        runtime,
        'record_with_llm',
        AsyncMock(side_effect=ConversationProgressContextLimitError(
            'too large',
            owner='event',
            event_disposition='context_limit',
        )),
    )

    result = await runtime.ConversationProgressRuntime().record(
        record_input=record_input(prior_packet=prior),
    )

    assert result['written'] is False
    assert result['turn_count'] == 7
    assert result['diagnostics']['recorder_call_count'] == 0
    assert result['diagnostics']['event_attempt_count'] == 0
    assert result['diagnostics']['event_disposition'] == 'context_limit'
