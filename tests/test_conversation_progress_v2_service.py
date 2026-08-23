"""Service lifecycle and call-count contracts for conversation progress V2."""

from __future__ import annotations

import logging
from copy import deepcopy
from unittest.mock import AsyncMock

import pytest

from kazusa_ai_chatbot import service
from kazusa_ai_chatbot.brain_service import post_turn
from kazusa_ai_chatbot.conversation_progress import (
    ConversationProgressScope,
    select_recordable_turn_outcome,
)
from kazusa_ai_chatbot.conversation_progress import cache, runtime
from kazusa_ai_chatbot.conversation_progress.recorder import (
    RecorderInvocationResult,
)
from kazusa_ai_chatbot.conversation_progress.repository import (
    PreparedProgressWrite,
    ProgressWriteResult,
)
from tests.conversation_progress_v2_helpers import (
    event,
    logical_turn,
    packet,
    recorder_delta,
    record_input,
)


def _trace(terminal_status: str) -> dict[str, object]:
    """Build one minimal settled trace with optional visible text."""

    surface_outputs: list[dict[str, object]] = []
    if terminal_status == 'completed_visible':
        surface_outputs.append({
            'schema_version': 'surface_output.v1',
            'surface_kind': 'text',
            'visibility': 'user_visible',
            'action_attempt_id': None,
            'surface_role': 'ordinary',
            'goal_continuation_ref': None,
            'fragments': ['accepted response'],
            'artifact_refs': [],
            'delivery_intent': 'deliver_now',
            'created_at': '2026-07-28T09:30:01+00:00',
        })
    return {
        'schema_version': 'episode_trace.v2',
        'episode_id': 'episode-service-test',
        'trigger_source': 'user_message',
        'terminal_status': terminal_status,
        'cognition_refs': [],
        'action_specs': [],
        'action_results': [],
        'surface_outputs': surface_outputs,
        'attempt_diagnostics': [],
        'delivery_correlation': {
            'schema_version': 'delivery_correlation.v1',
            'delivery_intent': (
                'deliver_now'
                if terminal_status == 'completed_visible'
                else 'do_not_deliver'
            ),
            'tracking_id': (
                'delivery-service-test'
                if terminal_status == 'completed_visible'
                else ''
            ),
            'receipt_status': (
                'pending'
                if terminal_status == 'completed_visible'
                else 'not_applicable'
            ),
            'receipt_ref': '',
        },
        'created_at': '2026-07-28T09:30:00+00:00',
        'settled_at': '2026-07-28T09:30:01+00:00',
    }


def _post_turn_state(turn_outcome: str) -> dict[str, object]:
    """Build the exact state consumed by the post-turn progress owner."""

    terminal_status = (
        'completed_visible'
        if turn_outcome == 'visible_response'
        else 'completed_private'
    )
    return {
        'conversation_progress_turn_outcome': turn_outcome,
        'episode_trace': _trace(terminal_status),
        'character_profile': {
            'name': 'Test Character',
            'boundary_profile': record_input()['boundary_profile'],
        },
        'platform': 'qq',
        'platform_channel_id': 'channel_test',
        'global_user_id': 'user_test',
        'storage_timestamp_utc': '2026-07-28T09:30:00+00:00',
        'active_turn_conversation_row_ids': ['current-row-a'],
        'active_turn_conversation_source_refs': [{
            'ref_kind': 'conversation_row',
            'ref_id': 'current-row-a',
            'occurred_at': '2026-07-28T09:30:00+00:00',
        }],
        'llm_trace_id': 'current-trace-a',
        'conversation_episode_state': None,
        'decontextualized_input': 'current input',
        'interaction_logical_turns': [logical_turn()],
        'logical_stance': 'CONFIRM',
        'character_intent': 'PROVIDE',
    }


@pytest.mark.parametrize(
    ('final_dialog', 'terminal_status', 'route', 'expected'),
    [
        (
            ['accepted response'],
            'completed_visible',
            None,
            'visible_response',
        ),
        ([], 'completed_private', 'silence', 'cognition_silence'),
        ([], 'completed_private', 'respond', None),
    ],
)
def test_recordable_outcomes_use_settled_typed_facts(
    final_dialog,
    terminal_status,
    route,
    expected,
):
    cognition_output = (
        {'intention': {'route': route}}
        if route is not None
        else None
    )
    assert select_recordable_turn_outcome(
        final_dialog=final_dialog,
        episode_trace=_trace(terminal_status),
        cognition_output=cognition_output,
        relevance_approved=True,
        consolidatable=True,
        listen_only=False,
        pruned=False,
    ) == expected


@pytest.mark.parametrize(
    ('relevance_approved', 'consolidatable', 'listen_only', 'pruned'),
    [
        (False, True, False, False),
        (True, False, False, False),
        (True, True, True, False),
        (True, True, False, True),
    ],
)
def test_visible_and_silent_outcomes_honor_every_common_exclusion(
    relevance_approved,
    consolidatable,
    listen_only,
    pruned,
):
    assert select_recordable_turn_outcome(
        final_dialog=['accepted response'],
        episode_trace=_trace('completed_visible'),
        cognition_output=None,
        relevance_approved=relevance_approved,
        consolidatable=consolidatable,
        listen_only=listen_only,
        pruned=pruned,
    ) is None
    assert select_recordable_turn_outcome(
        final_dialog=[],
        episode_trace=_trace('completed_private'),
        cognition_output={'intention': {'route': 'silence'}},
        relevance_approved=relevance_approved,
        consolidatable=consolidatable,
        listen_only=listen_only,
        pruned=pruned,
    ) is None


@pytest.mark.asyncio
async def test_service_load_passes_v2_scope_bot_and_current_row_ids(
    monkeypatch,
):
    load_result = {
        'episode_state': None,
        'conversation_progress': {
            'schema_version': 'conversation_progress_prompt.v2',
            'episode_state_id': '',
            'status': 'empty',
            'continuity': 'sharp_transition',
            'turn_count': 0,
            'current_thread': '',
            'character_stance': '',
            'user_goal': '',
            'current_blocker': '',
            'emotional_trajectory': '',
            'episode_narrative': '',
            'events': [],
            'overused_moves': [],
            'interaction_logical_turns': [],
            'compacted_block_refs': [],
        },
        'ambient_logical_turns': [],
        'interaction_logical_turns': [],
        'diagnostics': {'schema_version': 'conversation_progress_diagnostics.v2'},
        'source': 'empty',
    }
    progress_loader = AsyncMock(return_value=load_result)
    monkeypatch.setattr(service, 'load_progress_context', progress_loader)
    monkeypatch.setattr(
        service,
        'load_residue_context',
        AsyncMock(return_value={
            'status': 'empty',
            'selected_count': 0,
            'internal_monologue_residue_context': {},
        }),
    )
    monkeypatch.setattr(
        service,
        '_load_reply_past_dialog_context',
        AsyncMock(return_value={}),
    )
    state = {
        'platform': 'qq',
        'platform_channel_id': 'channel_test',
        'global_user_id': 'user_test',
        'platform_bot_id': 'bot-platform-id',
        'active_turn_conversation_row_ids': ['mongo-row-a'],
        'storage_timestamp_utc': '2026-07-28T09:30:00+00:00',
        'character_profile': {'global_user_id': 'character-id'},
        'channel_type': 'group',
    }

    result = await service.load_conversation_episode_state(state)

    progress_loader.assert_awaited_once_with(
        scope=ConversationProgressScope(
            'qq',
            'channel_test',
            'user_test',
        ),
        current_timestamp_utc='2026-07-28T09:30:00+00:00',
        platform_bot_id='bot-platform-id',
        active_turn_conversation_row_ids=['mongo-row-a'],
        group_scene_mode='group',
        group_scene_current_user_id='user_test',
    )
    assert result['conversation_episode_state'] is None
    assert result['ambient_logical_turns'] == []
    assert result['interaction_logical_turns'] == []


@pytest.mark.asyncio
async def test_service_load_passes_group_anchor_mode_and_keeps_user_scope(
    monkeypatch,
):
    """The approved group mode test retains the existing per-user scope."""

    await test_service_load_passes_v2_scope_bot_and_current_row_ids(
        monkeypatch,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('turn_outcome', 'expected_dialog'),
    [
        ('visible_response', ['accepted response']),
        ('cognition_silence', []),
    ],
)
async def test_post_turn_builds_exact_v2_record_input(
    monkeypatch,
    turn_outcome,
    expected_dialog,
):
    recorder_call = AsyncMock(return_value={
        'written': True,
        'turn_count': 1,
        'continuity': 'same_episode',
        'status': 'active',
        'cache_updated': True,
        'diagnostics': {},
    })
    state = _post_turn_state(turn_outcome)

    await post_turn.run_conversation_progress_record_background(
        state,
        record_turn_progress_func=recorder_call,
        logger=logging.getLogger(__name__),
    )

    recorder_call.assert_awaited_once()
    submitted = recorder_call.await_args.kwargs['record_input']
    assert submitted['turn_outcome'] == turn_outcome
    assert submitted['final_dialog'] == expected_dialog
    assert submitted['interaction_logical_turns'] == [logical_turn()]
    assert 'compaction_request' not in submitted
    assert submitted['current_turn_source_refs'] == [
        {
            'ref_kind': 'conversation_row',
            'ref_id': 'current-row-a',
            'occurred_at': '2026-07-28T09:30:00+00:00',
        },
        {
            'ref_kind': 'llm_trace',
            'ref_id': 'current-trace-a',
            'occurred_at': '2026-07-28T09:30:01+00:00',
        },
    ]


@pytest.mark.asyncio
async def test_ordinary_response_path_adds_no_llm_call(monkeypatch):
    """Load uses three reads; post-turn uses one semantic recorder call."""

    cache.clear_cache()
    state_read = AsyncMock(return_value=None)
    ambient_read = AsyncMock(return_value=[])
    participant_read = AsyncMock(return_value=[])
    recorder_call = AsyncMock()
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
    monkeypatch.setattr(runtime, 'record_with_llm', recorder_call)
    progress_runtime = runtime.ConversationProgressRuntime()

    await progress_runtime.load(
        scope=ConversationProgressScope(
            'qq',
            'channel_test',
            'user_test',
        ),
        current_timestamp_utc='2026-07-28T09:30:00+00:00',
        platform_bot_id='bot-platform-id',
        active_turn_conversation_row_ids=['current-row-a'],
    )

    state_read.assert_awaited_once()
    ambient_read.assert_awaited_once()
    participant_read.assert_awaited_once()
    recorder_call.assert_not_awaited()

    prior = packet(
        turn_count=18,
        events=[
            event(
                event_id=f'terminal-{index}',
                state='completed',
                retention='background',
            )
            for index in range(18)
        ],
    )
    record = record_input(prior_packet=prior)
    prepared_packet = deepcopy(prior)
    prepared_packet['turn_count'] = 19
    prepared = PreparedProgressWrite(
        packet=prepared_packet,
        block=None,
        source_block_ids=[],
        protected_block_ids=[],
    )
    recorder_call.return_value = RecorderInvocationResult(
        delta=recorder_delta(),
        recorder_call_count=2,
        event_attempt_count=1,
        scene_attempt_count=1,
        event_disposition='accepted',
        scene_disposition='accepted',
        event_human_payload_chars=60,
        scene_human_payload_chars=40,
        provider_usage={'event': {'input_tokens': 10}, 'scene': {}},
    )
    monkeypatch.setattr(
        runtime,
        'load_referenced_blocks',
        AsyncMock(return_value=[]),
    )
    monkeypatch.setattr(runtime, 'prepare_progress_write', lambda **_kwargs: prepared)
    monkeypatch.setattr(
        runtime,
        'persist_progress_write',
        AsyncMock(return_value=ProgressWriteResult(
            written=True,
            block_inserted=False,
            disposition='written',
        )),
    )

    result = await progress_runtime.record(record_input=record)

    recorder_call.assert_awaited_once()
    submitted = recorder_call.await_args.args[0]
    assert 'compaction_request' not in submitted
    assert result['written'] is True
    assert result['diagnostics']['recorder_call_count'] == 2


@pytest.mark.asyncio
async def test_post_turn_emits_trace_linked_progress_disposition(monkeypatch):
    """Post-turn telemetry carries the settled progress trace reference."""

    recorder_call = AsyncMock(return_value={
        'written': True,
        'turn_count': 4,
        'continuity': 'same_episode',
        'status': 'active',
        'cache_updated': True,
        'diagnostics': {'write_disposition': 'written'},
    })
    event_recorder = AsyncMock()
    monkeypatch.setattr(
        post_turn.event_logging,
        'record_continuity_boundary_event',
        event_recorder,
    )

    await post_turn.run_conversation_progress_record_background(
        _post_turn_state('visible_response'),
        record_turn_progress_func=recorder_call,
        logger=logging.getLogger(__name__),
    )

    event_recorder.assert_awaited_once()
    fields = event_recorder.await_args.kwargs
    assert fields['boundary'] == 'post_turn'
    assert fields['status'] == 'succeeded'
    assert fields['trace_ref'] == 'current-trace-a'
    assert fields['write_disposition'] == 'written'
    assert fields['cache_disposition'] == 'published'


@pytest.mark.asyncio
async def test_post_turn_preserves_trace_link_when_diagnostic_event_write_fails(
    monkeypatch,
):
    """A diagnostic failure cannot interrupt background progress recording."""

    recorder_call = AsyncMock(return_value={
        'written': True,
        'turn_count': 4,
        'continuity': 'same_episode',
        'status': 'active',
        'cache_updated': False,
        'diagnostics': {'write_disposition': 'written'},
    })
    event_recorder = AsyncMock(side_effect=RuntimeError('event sink down'))
    monkeypatch.setattr(
        post_turn.event_logging,
        'record_continuity_boundary_event',
        event_recorder,
    )

    await post_turn.run_conversation_progress_record_background(
        _post_turn_state('visible_response'),
        record_turn_progress_func=recorder_call,
        logger=logging.getLogger(__name__),
    )

    recorder_call.assert_awaited_once()
    event_recorder.assert_awaited_once()
    assert event_recorder.await_args.kwargs['trace_ref'] == 'current-trace-a'
