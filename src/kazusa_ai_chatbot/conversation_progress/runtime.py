"""Canonical facade runtime for progress load and post-turn recording."""

from __future__ import annotations

import asyncio
import hashlib
import logging
from copy import deepcopy
from collections.abc import Mapping, Sequence
from typing import Literal

from kazusa_ai_chatbot import event_logging
from kazusa_ai_chatbot.conversation_progress.cache import (
    get_cached_packet,
    invalidate_cached_packet,
    put_cached_packet,
)
from kazusa_ai_chatbot.conversation_progress.compaction import (
    ConversationCompactionContractError,
)
from kazusa_ai_chatbot.conversation_progress.delta_merge import (
    ConversationProgressContractError,
    _source_turn_ref,
)
from kazusa_ai_chatbot.conversation_progress.history import (
    assemble_logical_turns_with_diagnostics,
    select_group_scene_logical_turns,
    select_recent_logical_turns,
)
from kazusa_ai_chatbot.conversation_progress.models import (
    ConversationProgressLoadDiagnosticsV2,
    ConversationProgressLoadResult,
    ConversationProgressRecordInput,
    ConversationProgressRecordResult,
    ConversationProgressScope,
    ConversationProgressStateV2,
)
from kazusa_ai_chatbot.conversation_progress.projection import (
    build_progress_prompt,
    continuation_projection_chars,
)
from kazusa_ai_chatbot.conversation_progress.recorder import (
    ConversationProgressContextLimitError,
    ConversationProgressRecorderOutputError,
    record_with_llm,
)
from kazusa_ai_chatbot.conversation_progress.repository import (
    load_active_packet,
    load_referenced_blocks,
    persist_progress_write,
    prepare_progress_write,
)
from kazusa_ai_chatbot.conversation_progress.policy import (
    AMBIENT_LOGICAL_TURN_LIMIT,
    AMBIENT_ROW_SCAN_LIMIT,
    INTERACTION_LOGICAL_TURN_LIMIT,
    INTERACTION_ROW_SCAN_LIMIT,
    prune_aged_progress_packet,
)
from kazusa_ai_chatbot.db.conversation import (
    get_ambient_conversation_history,
    get_participant_conversation_history,
)
from kazusa_ai_chatbot.db import DatabaseOperationError
from kazusa_ai_chatbot.time_boundary import parse_storage_utc_datetime


logger = logging.getLogger(__name__)


class ConversationProgressRuntime:
    """Bounded implementation behind the two public facade functions."""

    async def load(
        self,
        *,
        scope: ConversationProgressScope,
        current_timestamp_utc: str,
        platform_bot_id: str,
        active_turn_conversation_row_ids: list[str],
        group_scene_mode: Literal['private', 'group', 'targetless'] = (
            'private'
        ),
        group_scene_current_user_id: str = '',
    ) -> ConversationProgressLoadResult:
        """Load packet and independent history lanes in one concurrent gather."""

        if not platform_bot_id:
            raise ValueError('platform_bot_id is required')
        state_result, ambient_rows, interaction_rows = await asyncio.gather(
            load_active_packet(
                scope=scope,
                current_timestamp_utc=current_timestamp_utc,
            ),
            get_ambient_conversation_history(
                platform=scope.platform,
                platform_channel_id=scope.platform_channel_id,
                excluded_row_ids=active_turn_conversation_row_ids,
                limit=AMBIENT_ROW_SCAN_LIMIT,
            ),
            get_participant_conversation_history(
                platform=scope.platform,
                platform_channel_id=scope.platform_channel_id,
                current_global_user_id=scope.global_user_id,
                platform_bot_id=platform_bot_id,
                excluded_row_ids=active_turn_conversation_row_ids,
                limit=INTERACTION_ROW_SCAN_LIMIT,
            ),
        )
        cached = get_cached_packet(
            scope=scope,
            current_timestamp_utc=current_timestamp_utc,
        )
        packet, source = _select_packet(state_result, cached)
        if packet is not None:
            pruned_packet, _, _ = prune_aged_progress_packet(
                packet,
                current_timestamp_utc=current_timestamp_utc,
            )
            packet = pruned_packet
        if source == 'db' and packet is not None:
            invalidate_cached_packet(scope=scope)
            put_cached_packet(scope=scope, packet=packet)

        ambient_assembly = assemble_logical_turns_with_diagnostics(
            rows=ambient_rows,
            excluded_row_ids=active_turn_conversation_row_ids,
        )
        interaction_assembly = assemble_logical_turns_with_diagnostics(
            rows=interaction_rows,
            excluded_row_ids=active_turn_conversation_row_ids,
        )
        if group_scene_mode == 'group':
            ambient_turns = select_group_scene_logical_turns(
                ambient_assembly.turns,
                current_global_user_id=group_scene_current_user_id,
                limit=AMBIENT_LOGICAL_TURN_LIMIT,
            )
        else:
            ambient_turns = select_recent_logical_turns(
                ambient_assembly.turns,
                limit=AMBIENT_LOGICAL_TURN_LIMIT,
            )
        interaction_turns = select_recent_logical_turns(
            interaction_assembly.turns,
            limit=INTERACTION_LOGICAL_TURN_LIMIT,
        )
        prompt = build_progress_prompt(
            active_packet=packet,
            interaction_logical_turns=interaction_turns,
        )
        scene_chars, evidence_chars = continuation_projection_chars(
            prompt,
            current_timestamp_utc,
        )
        diagnostics = _diagnostics(
            ambient_rows_scanned=len(ambient_rows),
            interaction_rows_scanned=len(interaction_rows),
            ambient_turns_selected=len(ambient_turns),
            interaction_turns_selected=len(interaction_turns),
            malformed_count=(
                ambient_assembly.incomplete_or_malformed_turn_count
                + interaction_assembly.incomplete_or_malformed_turn_count
            ),
            packet=packet,
            scene_chars=scene_chars,
            evidence_chars=evidence_chars,
            write_disposition='load_only',
            protected_anchor_count=_protected_anchor_count(
                ambient_turns,
                current_global_user_id=(
                    group_scene_current_user_id
                    if group_scene_mode == 'group'
                    else ''
                ),
            ),
            packet_age=_packet_age(
                packet=packet,
                current_timestamp_utc=current_timestamp_utc,
            ),
            source_age=_source_age(
                packet=packet,
                current_timestamp_utc=current_timestamp_utc,
            ),
            cache_disposition=(
                'cache_hit' if source == 'cache' else 'not_published'
            ),
            barrier_disposition='unknown',
            reconciliation_status='unknown',
        )
        await _record_progress_boundary(
            record_input=None,
            status='succeeded' if source != 'empty' else 'empty',
            candidate_count=len(ambient_rows) + len(interaction_rows),
            selected_count=len(ambient_turns) + len(interaction_turns),
            packet_turn_count=diagnostics['packet_turn_count'],
            protected_anchor_count=diagnostics['protected_anchor_count'],
            cache_disposition=diagnostics['cache_disposition'],
            packet_age=diagnostics['packet_age'],
            source_age=diagnostics['source_age'],
            rendered_chars=(
                diagnostics['scene_chars'] + diagnostics['evidence_chars']
            ),
            scope_kind=_progress_scope_kind(
                group_scene_mode=group_scene_mode,
                current_global_user_id=group_scene_current_user_id,
            ),
        )
        return {
            'episode_state': deepcopy(packet),
            'conversation_progress': prompt,
            'ambient_logical_turns': ambient_turns,
            'interaction_logical_turns': interaction_turns,
            'diagnostics': diagnostics,
            'source': source,
        }

    async def record(
        self,
        *,
        record_input: ConversationProgressRecordInput,
    ) -> ConversationProgressRecordResult:
        """Record one settled visible or eligible cognition-silence turn."""

        prior_packet = record_input['prior_episode_state']
        recorder_call_count = 0
        event_attempt_count = 0
        scene_attempt_count = 0
        event_disposition = 'not_called'
        scene_disposition = 'not_called'
        reconciliation_status = 'unknown'
        try:
            active_blocks = await load_referenced_blocks(
                active_packet=prior_packet,
            )
            invocation = await record_with_llm(record_input)
            recorder_call_count = invocation.recorder_call_count
            event_attempt_count = invocation.event_attempt_count
            scene_attempt_count = invocation.scene_attempt_count
            event_disposition = invocation.event_disposition
            scene_disposition = invocation.scene_disposition
            prepared = prepare_progress_write(
                record_input=record_input,
                delta=invocation.delta,
                active_blocks=active_blocks,
            )
            persistence = await persist_progress_write(prepared)
        except asyncio.CancelledError:
            reconciliation_status = await _reconcile_after_interruption(
                record_input=record_input,
            )
            diagnostics = _diagnostics(
                packet=prior_packet,
                recorder_call_count=recorder_call_count,
                event_attempt_count=event_attempt_count,
                scene_attempt_count=scene_attempt_count,
                event_disposition=event_disposition,
                scene_disposition=scene_disposition,
                write_disposition='interrupted',
                reconciliation_status=reconciliation_status,
            )
            await _record_progress_boundary(
                record_input=record_input,
                status='interrupted',
                write_disposition='interrupted',
                reconciliation_status=reconciliation_status,
                packet_turn_count=diagnostics['packet_turn_count'],
            )
            raise
        except (DatabaseOperationError, TimeoutError) as exc:
            reconciliation_status = await _reconcile_after_interruption(
                record_input=record_input,
            )
            diagnostics = _diagnostics(
                packet=prior_packet,
                recorder_call_count=recorder_call_count,
                event_attempt_count=event_attempt_count,
                scene_attempt_count=scene_attempt_count,
                event_disposition=event_disposition,
                scene_disposition=scene_disposition,
                write_disposition=f'failed:{type(exc).__name__}',
                reconciliation_status=reconciliation_status,
            )
            await _record_progress_boundary(
                record_input=record_input,
                status='persistence_failed',
                write_disposition='write_failed',
                reconciliation_status=reconciliation_status,
                packet_turn_count=diagnostics['packet_turn_count'],
            )
            return _failed_record_result(
                prior_packet=prior_packet,
                diagnostics=diagnostics,
                reconciliation_status=reconciliation_status,
            )
        except ConversationProgressRecorderOutputError as exc:
            diagnostics = _diagnostics(
                packet=prior_packet,
                recorder_call_count=exc.recorder_call_count,
                event_attempt_count=exc.event_attempt_count,
                scene_attempt_count=exc.scene_attempt_count,
                event_disposition=exc.event_disposition,
                scene_disposition=exc.scene_disposition,
                write_disposition=f'failed:{type(exc).__name__}',
                reconciliation_status=reconciliation_status,
            )
            await _record_progress_boundary(
                record_input=record_input,
                status='contract_failed',
                write_disposition='write_failed',
                packet_turn_count=diagnostics['packet_turn_count'],
            )
            return _failed_record_result(
                prior_packet=prior_packet,
                diagnostics=diagnostics,
            )
        except ConversationProgressContextLimitError as exc:
            diagnostics = _diagnostics(
                packet=prior_packet,
                recorder_call_count=exc.recorder_call_count,
                event_attempt_count=exc.event_attempt_count,
                scene_attempt_count=exc.scene_attempt_count,
                event_disposition=exc.event_disposition,
                scene_disposition=exc.scene_disposition,
                write_disposition=f'failed:{type(exc).__name__}',
                reconciliation_status=reconciliation_status,
            )
            await _record_progress_boundary(
                record_input=record_input,
                status='contract_failed',
                write_disposition='write_failed',
                packet_turn_count=diagnostics['packet_turn_count'],
            )
            return _failed_record_result(
                prior_packet=prior_packet,
                diagnostics=diagnostics,
            )
        except (
            ConversationCompactionContractError,
            ConversationProgressContractError,
        ) as exc:
            diagnostics = _diagnostics(
                packet=prior_packet,
                recorder_call_count=recorder_call_count,
                event_attempt_count=event_attempt_count,
                scene_attempt_count=scene_attempt_count,
                event_disposition=event_disposition,
                scene_disposition=scene_disposition,
                write_disposition=f'failed:{type(exc).__name__}',
                reconciliation_status=reconciliation_status,
            )
            await _record_progress_boundary(
                record_input=record_input,
                status='contract_failed',
                write_disposition='write_failed',
                packet_turn_count=diagnostics['packet_turn_count'],
            )
            return _failed_record_result(
                prior_packet=prior_packet,
                diagnostics=diagnostics,
            )

        scope = record_input['scope']
        cache_updated = False
        if persistence.written and prepared.packet['status'] == 'active':
            cache_updated = put_cached_packet(
                scope=scope,
                packet=prepared.packet,
            )
        elif persistence.written:
            invalidate_cached_packet(scope=scope)
        prompt = build_progress_prompt(
            active_packet=prepared.packet,
            interaction_logical_turns=record_input[
                'interaction_logical_turns'
            ],
        )
        scene_chars, evidence_chars = continuation_projection_chars(
            prompt,
            record_input['storage_timestamp_utc'],
        )
        compaction_level = (
            prepared.block['level']
            if prepared.block is not None
            else 0
        )
        diagnostics = _diagnostics(
            packet=prepared.packet,
            scene_chars=scene_chars,
            evidence_chars=evidence_chars,
            compaction_requested=prepared.block is not None,
            compaction_level=compaction_level,
            recorder_call_count=invocation.recorder_call_count,
            event_attempt_count=invocation.event_attempt_count,
            scene_attempt_count=invocation.scene_attempt_count,
            event_disposition=invocation.event_disposition,
            scene_disposition=invocation.scene_disposition,
            write_disposition=persistence.disposition,
            reconciliation_status=reconciliation_status,
        )
        if persistence.disposition == 'lost_guarded_write':
            reconciliation_status = await _reconcile_progress_operation(
                record_input=record_input,
            )
            diagnostics['reconciliation_status'] = reconciliation_status
        await _record_progress_boundary(
            record_input=record_input,
            status=(
                'guarded_write_lost'
                if persistence.disposition == 'lost_guarded_write'
                else 'succeeded'
            ),
            write_disposition=_continuity_write_disposition(
                persistence.disposition,
            ),
            cache_disposition=(
                'published' if cache_updated else 'not_published'
            ),
            reconciliation_status=reconciliation_status,
            packet_turn_count=diagnostics['packet_turn_count'],
            recorder_disposition=_continuity_recorder_disposition(
                diagnostics['event_disposition'],
            ),
            rendered_chars=scene_chars + evidence_chars,
        )
        return {
            'written': persistence.written,
            'turn_count': prepared.packet['turn_count'],
            'continuity': prepared.packet['continuity'],
            'status': prepared.packet['status'],
            'cache_updated': cache_updated,
            'diagnostics': diagnostics,
            'reconciliation_status': reconciliation_status,
        }


def _select_packet(
    db_packet: ConversationProgressStateV2 | None,
    cached_packet: ConversationProgressStateV2 | None,
) -> tuple[ConversationProgressStateV2 | None, str]:
    """Select the newest valid packet after both sources are inspected."""

    if cached_packet is not None and (
        db_packet is None
        or cached_packet['turn_count'] > db_packet['turn_count']
    ):
        return cached_packet, 'cache'
    if db_packet is not None:
        return db_packet, 'db'
    return None, 'empty'


def _failed_record_result(
    *,
    prior_packet: ConversationProgressStateV2 | None,
    diagnostics: ConversationProgressLoadDiagnosticsV2,
    reconciliation_status: str = 'unknown',
) -> ConversationProgressRecordResult:
    """Return a typed fail-closed result while retaining the prior packet."""

    return {
        'written': False,
        'turn_count': prior_packet['turn_count'] if prior_packet else 0,
        'continuity': (
            prior_packet['continuity']
            if prior_packet
            else 'sharp_transition'
        ),
        'status': prior_packet['status'] if prior_packet else 'closed',
        'cache_updated': False,
        'diagnostics': diagnostics,
        'reconciliation_status': reconciliation_status,
    }


def _diagnostics(
    *,
    ambient_rows_scanned: int = 0,
    interaction_rows_scanned: int = 0,
    ambient_turns_selected: int = 0,
    interaction_turns_selected: int = 0,
    malformed_count: int = 0,
    packet: ConversationProgressStateV2 | None,
    scene_chars: int = 0,
    evidence_chars: int = 0,
    compaction_requested: bool = False,
    compaction_level: int = 0,
    recorder_call_count: int = 0,
    event_attempt_count: int = 0,
    scene_attempt_count: int = 0,
    event_disposition: str = 'not_called',
    scene_disposition: str = 'not_called',
    write_disposition: str,
    protected_anchor_count: int = 0,
    packet_age: str = 'unknown',
    source_age: str = 'unknown',
    cache_disposition: str = 'unknown',
    barrier_disposition: str = 'unknown',
    reconciliation_status: str = 'unknown',
) -> ConversationProgressLoadDiagnosticsV2:
    """Build the exact text-free diagnostics envelope."""

    events = packet['events'] if packet is not None else []
    return {
        'schema_version': 'conversation_progress_diagnostics.v2',
        'ambient_rows_scanned': ambient_rows_scanned,
        'interaction_rows_scanned': interaction_rows_scanned,
        'ambient_turns_selected': ambient_turns_selected,
        'interaction_turns_selected': interaction_turns_selected,
        'incomplete_or_malformed_turn_count': malformed_count,
        'packet_turn_count': packet['turn_count'] if packet else 0,
        'active_event_count': len(events),
        'decision_critical_event_count': sum(
            event['retention'] == 'decision_critical'
            for event in events
        ),
        'block_ref_count': (
            len(packet['compacted_block_refs']) if packet else 0
        ),
        'scene_chars': scene_chars,
        'evidence_chars': evidence_chars,
        'compaction_requested': compaction_requested,
        'compaction_level': compaction_level,
        'recorder_call_count': recorder_call_count,
        'event_attempt_count': event_attempt_count,
        'scene_attempt_count': scene_attempt_count,
        'event_disposition': event_disposition,
        'scene_disposition': scene_disposition,
        'write_disposition': write_disposition,
        'protected_anchor_count': protected_anchor_count,
        'packet_age': packet_age,
        'source_age': source_age,
        'cache_disposition': cache_disposition,
        'barrier_disposition': barrier_disposition,
        'reconciliation_status': reconciliation_status,
    }


_default_runtime = ConversationProgressRuntime()


async def load_progress_context(
    *,
    scope: ConversationProgressScope,
    current_timestamp_utc: str,
    platform_bot_id: str,
    active_turn_conversation_row_ids: list[str],
    group_scene_mode: Literal['private', 'group', 'targetless'] = 'private',
    group_scene_current_user_id: str = '',
) -> ConversationProgressLoadResult:
    """Load V2 progress through the sole public read facade."""

    return await _default_runtime.load(
        scope=scope,
        current_timestamp_utc=current_timestamp_utc,
        platform_bot_id=platform_bot_id,
        active_turn_conversation_row_ids=active_turn_conversation_row_ids,
        group_scene_mode=group_scene_mode,
        group_scene_current_user_id=group_scene_current_user_id,
    )


async def record_turn_progress(
    *,
    record_input: ConversationProgressRecordInput,
) -> ConversationProgressRecordResult:
    """Record V2 progress through the sole public write facade."""

    return await _default_runtime.record(record_input=record_input)


def _trace_ref(record_input: ConversationProgressRecordInput) -> str:
    """Return the opaque current-turn trace reference for reconciliation."""

    for source_ref in reversed(record_input['current_turn_source_refs']):
        if source_ref['ref_kind'] == 'llm_trace':
            return source_ref['ref_id']
    return ''


def _progress_operation_ref(
    record_input: ConversationProgressRecordInput,
) -> str:
    """Derive an opaque operation key for reconciliation telemetry."""

    source_ref = _trace_ref(record_input)
    if not source_ref and record_input['current_turn_source_refs']:
        source_ref = record_input['current_turn_source_refs'][-1]['ref_id']
    identity = '|'.join((
        record_input['scope'].platform,
        record_input['scope'].platform_channel_id,
        record_input['scope'].global_user_id,
        source_ref,
    ))
    digest = hashlib.sha256(identity.encode('utf-8')).hexdigest()
    return f'progress-v2:{digest}'


async def _reconcile_progress_operation(
    *,
    record_input: ConversationProgressRecordInput,
) -> str:
    """Determine whether a packet contains the interrupted turn trace."""

    trace_ref = _trace_ref(record_input)
    if not trace_ref and not record_input['current_turn_source_refs']:
        return 'unknown'
    try:
        packet = await load_active_packet(
            scope=record_input['scope'],
            current_timestamp_utc=record_input['storage_timestamp_utc'],
        )
    except (DatabaseOperationError, TimeoutError):
        return 'unknown'
    if packet is None:
        return 'reconciled_absent'
    expected_turn_refs = {
        _source_turn_ref(source_ref)
        for source_ref in record_input['current_turn_source_refs']
    }
    if expected_turn_refs.intersection(packet['recent_turn_refs']):
        return 'reconciled_written'
    return 'reconciled_absent'


async def _reconcile_after_interruption(
    *,
    record_input: ConversationProgressRecordInput,
) -> str:
    """Run the deterministic trace read despite task cancellation."""

    return await asyncio.shield(
        _reconcile_progress_operation(record_input=record_input)
    )


def _protected_anchor_count(
    turns: Sequence[object],
    *,
    current_global_user_id: str,
) -> int:
    """Count only the participant anchors reserved for the current user."""

    if not current_global_user_id:
        return 0
    current_user_anchor: Mapping[str, object] | None = None
    for turn in reversed(turns):
        if (
            isinstance(turn, Mapping)
            and turn.get('role') == 'user'
            and turn.get('global_user_id') == current_global_user_id
        ):
            current_user_anchor = turn
            break
    if current_user_anchor is None:
        return 0

    count = 1
    try:
        current_user_time = parse_storage_utc_datetime(
            str(current_user_anchor['occurred_at'])
        )
    except (KeyError, TypeError, ValueError):
        return count
    for turn in reversed(turns):
        if not isinstance(turn, Mapping):
            continue
        if (
            turn.get('role') != 'assistant'
            or turn.get('broadcast') is True
            or turn.get('addressed_to_global_user_ids')
            != [current_global_user_id]
        ):
            continue
        try:
            assistant_time = parse_storage_utc_datetime(
                str(turn['occurred_at'])
            )
        except (KeyError, TypeError, ValueError):
            continue
        if assistant_time < current_user_time:
            continue
        reply_context = turn.get('reply_context')
        if isinstance(reply_context, Mapping):
            reply_target = reply_context.get('reply_to_global_user_id')
            if isinstance(reply_target, str) and (
                reply_target and reply_target != current_global_user_id
            ):
                continue
        return count + 1
    return count


def _packet_age(
    *,
    packet: ConversationProgressStateV2 | None,
    current_timestamp_utc: str,
) -> str:
    """Classify the stored packet age without exposing timestamps."""

    if packet is None:
        return 'unknown'
    return _age_descriptor(
        source_timestamp=packet.get('updated_at', ''),
        reference_timestamp=current_timestamp_utc,
    )


def _source_age(
    *,
    packet: ConversationProgressStateV2 | None,
    current_timestamp_utc: str,
) -> str:
    """Classify the newest event source age in the packet."""

    if packet is None:
        return 'unknown'
    source_timestamps = [
        source_ref.get('occurred_at', '')
        for event in packet['events']
        for source_ref in event['source_refs']
        if source_ref.get('occurred_at')
    ]
    if not source_timestamps:
        return 'unknown'
    return _age_descriptor(
        source_timestamp=max(source_timestamps),
        reference_timestamp=current_timestamp_utc,
    )


def _age_descriptor(*, source_timestamp: str, reference_timestamp: str) -> str:
    """Return the bounded age label used by continuity diagnostics."""

    try:
        source_time = parse_storage_utc_datetime(source_timestamp)
        reference_time = parse_storage_utc_datetime(reference_timestamp)
    except (TypeError, ValueError):
        return 'unknown'
    age_seconds = max((reference_time - source_time).total_seconds(), 0.0)
    if age_seconds <= 120:
        return 'fresh'
    if age_seconds <= 1800:
        return 'recent'
    return 'stale'


def _progress_scope_kind(
    *,
    group_scene_mode: Literal['private', 'group', 'targetless'],
    current_global_user_id: str,
) -> str:
    """Map the load lane into the bounded continuity scope vocabulary."""

    if group_scene_mode == 'group' and current_global_user_id:
        return 'group_scene'
    if group_scene_mode == 'targetless':
        return 'targetless'
    return 'private'


def _continuity_write_disposition(disposition: str) -> str:
    """Map repository detail to the closed continuity write labels."""

    if disposition == 'written':
        return 'written'
    if disposition == 'lost_guarded_write':
        return 'lost_guarded_write'
    if disposition.startswith('failed'):
        return 'write_failed'
    return 'unknown'


def _continuity_recorder_disposition(disposition: str) -> str:
    """Map recorder detail into the bounded continuity disposition set."""

    if disposition in {
        'not_called',
        'append',
        'replace_scope',
        'clear_scope',
    }:
        return disposition
    return 'unknown'


async def _record_progress_boundary(
    *,
    record_input: ConversationProgressRecordInput | None,
    status: str,
    candidate_count: int = 0,
    selected_count: int = 0,
    packet_turn_count: int = 0,
    protected_anchor_count: int = 0,
    write_disposition: str = 'not_attempted',
    cache_disposition: str = 'not_attempted',
    reconciliation_status: str = 'unknown',
    packet_age: str = 'unknown',
    source_age: str = 'unknown',
    recorder_disposition: str = 'unknown',
    barrier_disposition: str = 'unknown',
    rendered_chars: int = 0,
    scope_kind: str = '',
) -> None:
    """Emit bounded progress telemetry without affecting the result path."""

    if record_input is None:
        boundary = 'progress_load'
        operation_ref = ''
        trace_ref = ''
    else:
        boundary = 'progress_record'
        scope_kind = scope_kind or 'user_thread'
        operation_ref = _progress_operation_ref(record_input)
        trace_ref = _trace_ref(record_input)
    if reconciliation_status in {'reconciled_written', 'reconciled_absent'}:
        status = 'reconciled'
        write_disposition = reconciliation_status
    try:
        await event_logging.record_continuity_boundary_event(
            component='conversation_progress.runtime',
            boundary=boundary,
            status=status,
            scope_kind=scope_kind or 'targetless',
            candidate_count=candidate_count,
            selected_count=selected_count,
            packet_turn_count=packet_turn_count,
            protected_anchor_count=protected_anchor_count,
            rendered_chars=rendered_chars,
            packet_age=(
                packet_age
                if packet_age in {'unknown', 'fresh', 'recent', 'stale'}
                else 'unknown'
            ),
            source_age=(
                source_age
                if source_age in {'unknown', 'fresh', 'recent', 'stale'}
                else 'unknown'
            ),
            recorder_disposition=(
                recorder_disposition
                if recorder_disposition in {
                    'unknown',
                    'not_called',
                    'append',
                    'replace_scope',
                    'clear_scope',
                }
                else 'unknown'
            ),
            barrier_disposition=(
                barrier_disposition
                if barrier_disposition in {
                    'unknown',
                    'none',
                    'append',
                    'replace_scope',
                    'clear_scope',
                }
                else 'unknown'
            ),
            write_disposition=(
                write_disposition
                if write_disposition in {
                    'unknown',
                    'not_attempted',
                    'written',
                    'duplicate_same_payload',
                    'conflict',
                    'write_failed',
                    'lost_guarded_write',
                    'interrupted',
                    'reconciled_written',
                    'reconciled_absent',
                }
                else 'unknown'
            ),
            cache_disposition=(
                cache_disposition
                if cache_disposition in {
                    'unknown',
                    'not_attempted',
                    'cache_hit',
                    'published',
                    'invalidated',
                    'not_published',
                }
                else 'unknown'
            ),
            trace_ref=trace_ref,
            operation_ref=operation_ref,
        )
    except Exception as exc:
        logger.warning(
            'Continuity progress telemetry failed: %s',
            type(exc).__name__,
        )
