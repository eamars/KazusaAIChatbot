"""Canonical facade runtime load, selection, and record contracts."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from kazusa_ai_chatbot.conversation_progress import cache, runtime
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
