"""Canonical facade runtime for progress load and post-turn recording."""

from __future__ import annotations

import asyncio
from copy import deepcopy

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
)
from kazusa_ai_chatbot.conversation_progress.history import (
    assemble_logical_turns_with_diagnostics,
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


class ConversationProgressRuntime:
    """Bounded implementation behind the two public facade functions."""

    async def load(
        self,
        *,
        scope: ConversationProgressScope,
        current_timestamp_utc: str,
        platform_bot_id: str,
        active_turn_conversation_row_ids: list[str],
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
        except ConversationProgressRecorderOutputError as exc:
            diagnostics = _diagnostics(
                packet=prior_packet,
                recorder_call_count=exc.recorder_call_count,
                event_attempt_count=exc.event_attempt_count,
                scene_attempt_count=exc.scene_attempt_count,
                event_disposition=exc.event_disposition,
                scene_disposition=exc.scene_disposition,
                write_disposition=f'failed:{type(exc).__name__}',
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
        )
        return {
            'written': persistence.written,
            'turn_count': prepared.packet['turn_count'],
            'continuity': prepared.packet['continuity'],
            'status': prepared.packet['status'],
            'cache_updated': cache_updated,
            'diagnostics': diagnostics,
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
    }


_default_runtime = ConversationProgressRuntime()


async def load_progress_context(
    *,
    scope: ConversationProgressScope,
    current_timestamp_utc: str,
    platform_bot_id: str,
    active_turn_conversation_row_ids: list[str],
) -> ConversationProgressLoadResult:
    """Load V2 progress through the sole public read facade."""

    return await _default_runtime.load(
        scope=scope,
        current_timestamp_utc=current_timestamp_utc,
        platform_bot_id=platform_bot_id,
        active_turn_conversation_row_ids=active_turn_conversation_row_ids,
    )


async def record_turn_progress(
    *,
    record_input: ConversationProgressRecordInput,
) -> ConversationProgressRecordResult:
    """Record V2 progress through the sole public write facade."""

    return await _default_runtime.record(record_input=record_input)
