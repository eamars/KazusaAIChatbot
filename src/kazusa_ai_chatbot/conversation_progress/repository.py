"""Validated active-packet and compacted-block persistence orchestration."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from typing import cast
from uuid import uuid4

from kazusa_ai_chatbot.conversation_progress.compaction import (
    ConversationCompactionContractError,
    apply_compaction_to_packet,
    block_embedding_text,
    build_compaction_plan,
    create_block_from_plan,
    validate_block,
)
from kazusa_ai_chatbot.conversation_progress.delta_merge import (
    ConversationProgressContractError,
    apply_delta,
)
from kazusa_ai_chatbot.conversation_progress.models import (
    ConversationEpisodeBlockV1,
    ConversationProgressEventV2,
    ConversationProgressRecordInput,
    ConversationProgressRecorderDeltaV2,
    ConversationProgressScope,
    ConversationProgressStateV2,
)
from kazusa_ai_chatbot.conversation_progress.policy import (
    MAX_ACTIVE_BLOCK_REFS,
    MAX_ACTIVE_EVENTS,
    MAX_ACTIVE_PACKET_CHARS,
    MAX_BLOCK_GRAPH_DEPTH,
    MAX_EPISODE_NARRATIVE_CHARS,
    MAX_EVENT_OUTCOME_CHARS,
    MAX_EVENT_ROLE_CHARS,
    MAX_EVENT_SOURCE_REFS,
    MAX_EVENT_SUMMARY_CHARS,
    MAX_MOVE_CHARS,
    MAX_RECENT_TURN_REFS,
    MAX_REACHABLE_BLOCK_REFS,
    MAX_THREAD_FIELD_CHARS,
    VALID_CONTINUITY,
    VALID_EVENT_RETENTIONS,
    VALID_EVENT_STATES,
    VALID_SOURCE_REF_KINDS,
    VALID_STATUS,
    canonical_json_chars,
)
from kazusa_ai_chatbot.db import get_document_text_embedding
from kazusa_ai_chatbot.db.conversation_progress import (
    load_active_episode_state,
    replace_episode_state_guarded,
)
from kazusa_ai_chatbot.db.conversation_progress_blocks import (
    insert_conversation_progress_block,
    load_conversation_progress_block_graph,
    supersede_conversation_progress_blocks,
    touch_conversation_progress_blocks,
)
from kazusa_ai_chatbot.time_boundary import parse_storage_utc_datetime

EmbedBlock = Callable[[str], Awaitable[list[float]]]

_PACKET_FIELDS = {
    'schema_version',
    'episode_state_id',
    'platform',
    'platform_channel_id',
    'global_user_id',
    'status',
    'continuity',
    'turn_count',
    'episode_narrative',
    'current_thread',
    'character_stance',
    'user_goal',
    'current_blocker',
    'emotional_trajectory',
    'events',
    'overused_moves',
    'recent_turn_refs',
    'compacted_block_refs',
    'created_at',
    'updated_at',
    'expires_at',
    'purge_after',
}
_EVENT_FIELDS = {
    'event_id',
    'semantic_summary',
    'is_obligation',
    'actor',
    'action',
    'object',
    'beneficiary',
    'precondition',
    'state',
    'outcome',
    'retention',
    'source_refs',
    'first_seen_at',
    'updated_at',
}
_SOURCE_REF_FIELDS = {'ref_kind', 'ref_id', 'occurred_at'}


@dataclass(frozen=True)
class PreparedProgressWrite:
    """Validated packet plus the optional immutable block written before it."""

    packet: ConversationProgressStateV2
    block: ConversationEpisodeBlockV1 | None
    source_block_ids: list[str]
    protected_block_ids: list[str]


@dataclass(frozen=True)
class ProgressWriteResult:
    """Atomic-boundary persistence disposition."""

    written: bool
    block_inserted: bool
    disposition: str


async def load_active_packet(
    *,
    scope: ConversationProgressScope,
    current_timestamp_utc: str,
) -> ConversationProgressStateV2 | None:
    """Load and fully validate one active packet from the DB boundary."""

    packet = await load_active_episode_state(
        scope=scope,
        current_timestamp_utc=current_timestamp_utc,
    )
    if packet is None:
        return None
    try:
        return validate_active_packet(packet)
    except ValueError:
        return None


async def load_referenced_blocks(
    *,
    active_packet: ConversationProgressStateV2 | None,
) -> list[ConversationEpisodeBlockV1]:
    """Load and validate the full graph protected by active block roots."""

    if active_packet is None or not active_packet['compacted_block_refs']:
        return []
    try:
        documents = await load_conversation_progress_block_graph(
            root_block_ids=active_packet['compacted_block_refs'],
            scope=ConversationProgressScope(
                platform=active_packet['platform'],
                platform_channel_id=active_packet['platform_channel_id'],
                global_user_id=active_packet['global_user_id'],
            ),
            episode_state_id=active_packet['episode_state_id'],
        )
    except ValueError as exc:
        raise ConversationCompactionContractError(
            'protected conversation progress block graph is invalid'
        ) from exc
    return [validate_block(document) for document in documents]


def prepare_progress_write(
    *,
    record_input: ConversationProgressRecordInput,
    delta: ConversationProgressRecorderDeltaV2,
    active_blocks: Sequence[ConversationEpisodeBlockV1] = (),
) -> PreparedProgressWrite:
    """Apply one semantic delta and deterministic compaction plan."""

    prior_packet = record_input['prior_episode_state']
    episode_state_id = (
        prior_packet['episode_state_id']
        if prior_packet is not None
        else uuid4().hex
    )
    packet = apply_delta(
        prior_packet=prior_packet,
        delta=delta,
        record_input=record_input,
        episode_state_id=episode_state_id,
    )
    protected_block_ids = _protected_block_ids(
        active_packet=packet,
        active_blocks=active_blocks,
    )
    block: ConversationEpisodeBlockV1 | None = None
    source_block_ids: list[str] = []
    compaction_plan = build_compaction_plan(
        active_packet=packet,
        active_blocks=active_blocks,
    )
    if (
        compaction_plan is not None
        and len(protected_block_ids) < MAX_REACHABLE_BLOCK_REFS
    ):
        block = create_block_from_plan(
            compaction_plan=compaction_plan,
            active_packet=packet,
            active_blocks=active_blocks,
        )
        packet = apply_compaction_to_packet(
            active_packet=packet,
            compaction_plan=compaction_plan,
            block_id=block['block_id'],
        )
        source_block_ids = list(compaction_plan['source_block_ids'])
        protected_block_ids = _protected_block_ids(
            active_packet=packet,
            active_blocks=[*active_blocks, block],
        )
    validate_active_packet(packet)
    return PreparedProgressWrite(
        packet=packet,
        block=block,
        source_block_ids=source_block_ids,
        protected_block_ids=protected_block_ids,
    )


async def persist_progress_write(
    prepared: PreparedProgressWrite,
    *,
    embed_block: EmbedBlock = get_document_text_embedding,
) -> ProgressWriteResult:
    """Insert a block, guarded-replace the packet, then finalize lineage."""

    block_inserted = False
    persisted_block = prepared.block
    if persisted_block is not None:
        persisted_block = deepcopy(persisted_block)
        persisted_block['embedding'] = await embed_block(
            block_embedding_text(persisted_block)
        )
        validate_block(persisted_block)
        block_inserted = await insert_conversation_progress_block(
            document=persisted_block,
        )

    written = await replace_episode_state_guarded(
        document=prepared.packet,
    )
    if not written:
        return ProgressWriteResult(
            written=False,
            block_inserted=block_inserted,
            disposition='lost_guarded_write',
        )

    if prepared.source_block_ids and persisted_block is not None:
        await supersede_conversation_progress_blocks(
            source_block_ids=prepared.source_block_ids,
            superseded_by_block_id=persisted_block['block_id'],
        )
    await touch_conversation_progress_blocks(
        block_ids=prepared.protected_block_ids,
        expires_at=prepared.packet['expires_at'],
        purge_after=prepared.packet['purge_after'],
    )
    return ProgressWriteResult(
        written=True,
        block_inserted=block_inserted,
        disposition='written',
    )


def _protected_block_ids(
    *,
    active_packet: ConversationProgressStateV2,
    active_blocks: Sequence[ConversationEpisodeBlockV1],
) -> list[str]:
    """Validate and order the complete graph protected by packet roots."""

    by_id: dict[str, ConversationEpisodeBlockV1] = {}
    for raw_block in active_blocks:
        block = validate_block(raw_block)
        block_id = block['block_id']
        if block_id in by_id:
            raise ConversationCompactionContractError(
                'active block graph contains duplicate block IDs'
            )
        if (
            block['episode_state_id'] != active_packet['episode_state_id']
            or block['platform'] != active_packet['platform']
            or (
                block['platform_channel_id']
                != active_packet['platform_channel_id']
            )
            or block['global_user_id'] != active_packet['global_user_id']
        ):
            raise ConversationCompactionContractError(
                'active block graph crosses packet scope'
            )
        by_id[block_id] = block

    ordered_ids: list[str] = []
    pending_ids = list(active_packet['compacted_block_refs'])
    parent_by_id = {
        block_id: ''
        for block_id in pending_ids
    }
    depth = 0
    while pending_ids:
        if depth > MAX_BLOCK_GRAPH_DEPTH:
            raise ConversationCompactionContractError(
                'active block graph exceeds its depth cap'
            )
        next_ids: list[str] = []
        for block_id in pending_ids:
            if block_id in ordered_ids:
                raise ConversationCompactionContractError(
                    'active block graph is cyclic or has shared children'
                )
            block = by_id.get(block_id)
            if block is None:
                raise ConversationCompactionContractError(
                    'active packet references a missing compaction block'
                )
            expected_parent = parent_by_id[block_id]
            actual_parent = block['superseded_by_block_id']
            if expected_parent:
                if actual_parent not in {'', expected_parent}:
                    raise ConversationCompactionContractError(
                        'active child block lineage is invalid'
                    )
            elif actual_parent:
                raise ConversationCompactionContractError(
                    'active root block is already superseded'
                )
            ordered_ids.append(block_id)
            for source_block_id in block['source_block_ids']:
                if source_block_id in parent_by_id:
                    raise ConversationCompactionContractError(
                        'active block graph is cyclic or has shared children'
                    )
                parent_by_id[source_block_id] = block_id
                next_ids.append(source_block_id)
        if len(ordered_ids) + len(next_ids) > MAX_REACHABLE_BLOCK_REFS:
            raise ConversationCompactionContractError(
                'active block graph exceeds its node cap'
            )
        pending_ids = next_ids
        depth += 1
    if set(ordered_ids) != set(by_id):
        raise ConversationCompactionContractError(
            'active block input contains unreferenced blocks'
        )
    return ordered_ids


def validate_active_packet(
    value: object,
) -> ConversationProgressStateV2:
    """Validate the exact stored V2 packet and its final hard size."""

    if not isinstance(value, Mapping) or set(value) != _PACKET_FIELDS:
        raise ConversationProgressContractError(
            'active packet fields are not exact'
        )
    if value['schema_version'] != 'conversation_progress.v2':
        raise ConversationProgressContractError(
            'active packet schema_version is invalid'
        )
    for field_name in ('episode_state_id', 'platform', 'global_user_id'):
        _bounded_text(
            value[field_name],
            MAX_ACTIVE_PACKET_CHARS,
            field_name,
            required=True,
        )
    _bounded_text(
        value['platform_channel_id'],
        MAX_ACTIVE_PACKET_CHARS,
        'platform_channel_id',
    )
    status = _enum_text(value['status'], VALID_STATUS, 'status')
    _enum_text(value['continuity'], VALID_CONTINUITY, 'continuity')
    turn_count = value['turn_count']
    if (
        not isinstance(turn_count, int)
        or isinstance(turn_count, bool)
        or turn_count < 0
        or (status == 'active' and turn_count < 1)
    ):
        raise ConversationProgressContractError(
            'active packet turn_count is invalid'
        )
    _bounded_text(
        value['episode_narrative'],
        MAX_EPISODE_NARRATIVE_CHARS,
        'episode_narrative',
    )
    for field_name in (
        'current_thread',
        'character_stance',
        'user_goal',
        'current_blocker',
        'emotional_trajectory',
    ):
        _bounded_text(
            value[field_name],
            MAX_THREAD_FIELD_CHARS,
            field_name,
        )
    events = value['events']
    if not isinstance(events, list) or len(events) > MAX_ACTIVE_EVENTS:
        raise ConversationProgressContractError(
            'active packet events exceeds its hard cap'
        )
    validated_events = [_validate_stored_event(event) for event in events]
    event_ids = [event['event_id'] for event in validated_events]
    if len(event_ids) != len(set(event_ids)):
        raise ConversationProgressContractError(
            'active packet event IDs are duplicated'
        )
    _bounded_unique_text_list(
        value['overused_moves'],
        maximum_items=8,
        maximum_chars=MAX_MOVE_CHARS,
        field_name='overused_moves',
    )
    _bounded_unique_text_list(
        value['recent_turn_refs'],
        maximum_items=MAX_RECENT_TURN_REFS,
        maximum_chars=MAX_ACTIVE_PACKET_CHARS,
        field_name='recent_turn_refs',
    )
    _bounded_unique_text_list(
        value['compacted_block_refs'],
        maximum_items=MAX_ACTIVE_BLOCK_REFS,
        maximum_chars=MAX_ACTIVE_PACKET_CHARS,
        field_name='compacted_block_refs',
    )
    for field_name in ('created_at', 'updated_at', 'expires_at'):
        timestamp = _bounded_text(
            value[field_name],
            MAX_ACTIVE_PACKET_CHARS,
            field_name,
            required=True,
        )
        parse_storage_utc_datetime(timestamp)
    if not isinstance(value['purge_after'], datetime):
        raise ConversationProgressContractError(
            'active packet purge_after must be a BSON datetime'
        )
    if canonical_json_chars(value) > MAX_ACTIVE_PACKET_CHARS:
        raise ConversationProgressContractError(
            'active packet exceeds its hard character cap'
        )
    return deepcopy(value)  # type: ignore[return-value]


def _validate_stored_event(
    value: object,
) -> ConversationProgressEventV2:
    """Validate one stored event including code-owned lifecycle metadata."""

    if not isinstance(value, Mapping) or set(value) != _EVENT_FIELDS:
        raise ConversationProgressContractError(
            'stored event fields are not exact'
        )
    event_id = _bounded_text(
        value['event_id'],
        MAX_ACTIVE_PACKET_CHARS,
        'event_id',
        required=True,
    )
    semantic_summary = _bounded_text(
        value['semantic_summary'],
        MAX_EVENT_SUMMARY_CHARS,
        'semantic_summary',
        required=True,
    )
    is_obligation = value['is_obligation']
    if not isinstance(is_obligation, bool):
        raise ConversationProgressContractError(
            'stored event is_obligation must be boolean'
        )
    actor = _bounded_text(
        value['actor'],
        MAX_EVENT_ROLE_CHARS,
        'actor',
    )
    action = _bounded_text(
        value['action'],
        MAX_EVENT_ROLE_CHARS,
        'action',
        required=True,
    )
    event_object = _bounded_text(
        value['object'],
        MAX_EVENT_ROLE_CHARS,
        'object',
        required=True,
    )
    beneficiary = _bounded_text(
        value['beneficiary'],
        MAX_EVENT_ROLE_CHARS,
        'beneficiary',
    )
    precondition = _bounded_text(
        value['precondition'],
        MAX_EVENT_ROLE_CHARS,
        'precondition',
    )
    if not actor:
        raise ConversationProgressContractError(
            'stored event requires actor, action, and object'
        )
    state = _enum_text(value['state'], VALID_EVENT_STATES, 'state')
    outcome = _bounded_text(
        value['outcome'],
        MAX_EVENT_OUTCOME_CHARS,
        'outcome',
    )
    retention = _enum_text(
        value['retention'],
        VALID_EVENT_RETENTIONS,
        'retention',
    )
    source_refs = _validate_stored_source_refs(value['source_refs'])
    first_seen_at = _bounded_text(
        value['first_seen_at'],
        MAX_ACTIVE_PACKET_CHARS,
        'first_seen_at',
        required=True,
    )
    updated_at = _bounded_text(
        value['updated_at'],
        MAX_ACTIVE_PACKET_CHARS,
        'updated_at',
        required=True,
    )
    parse_storage_utc_datetime(first_seen_at)
    parse_storage_utc_datetime(updated_at)
    return {
        'event_id': event_id,
        'semantic_summary': semantic_summary,
        'is_obligation': is_obligation,
        'actor': actor,
        'action': action,
        'object': event_object,
        'beneficiary': beneficiary,
        'precondition': precondition,
        'state': state,
        'outcome': outcome,
        'retention': retention,
        'source_refs': source_refs,
        'first_seen_at': first_seen_at,
        'updated_at': updated_at,
    }


def _validate_stored_source_refs(
    value: object,
) -> list[dict[str, str]]:
    """Validate stored source refs without an external allowlist."""

    if (
        not isinstance(value, list)
        or not value
        or len(value) > MAX_EVENT_SOURCE_REFS
    ):
        raise ConversationProgressContractError(
            'stored event source_refs must be non-empty within its cap'
        )
    refs: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for raw_ref in value:
        if not isinstance(raw_ref, Mapping) or set(raw_ref) != _SOURCE_REF_FIELDS:
            raise ConversationProgressContractError(
                'stored source ref fields are not exact'
            )
        ref_kind = _enum_text(
            raw_ref['ref_kind'],
            VALID_SOURCE_REF_KINDS,
            'ref_kind',
        )
        ref_id = _bounded_text(
            raw_ref['ref_id'],
            MAX_ACTIVE_PACKET_CHARS,
            'ref_id',
            required=True,
        )
        occurred_at = _bounded_text(
            raw_ref['occurred_at'],
            MAX_ACTIVE_PACKET_CHARS,
            'occurred_at',
            required=True,
        )
        parse_storage_utc_datetime(occurred_at)
        identity = (ref_kind, ref_id)
        if identity in seen:
            raise ConversationProgressContractError(
                'stored event source_refs contains a duplicate'
            )
        seen.add(identity)
        refs.append({
            'ref_kind': ref_kind,
            'ref_id': ref_id,
            'occurred_at': occurred_at,
        })
    return refs


def _bounded_unique_text_list(
    value: object,
    *,
    maximum_items: int,
    maximum_chars: int,
    field_name: str,
) -> list[str]:
    """Validate one exact, unique string list."""

    if not isinstance(value, list):
        raise ConversationProgressContractError(
            f'{field_name} must be a list'
        )
    if len(value) > maximum_items:
        raise ConversationProgressContractError(
            f'{field_name} exceeds its item cap'
        )
    result: list[str] = []
    for item in value:
        text = _bounded_text(
            item,
            maximum_chars,
            field_name,
            required=True,
        )
        if text in result:
            raise ConversationProgressContractError(
                f'{field_name} contains a duplicate'
            )
        result.append(text)
    return result


def _enum_text(
    value: object,
    choices: frozenset[str],
    field_name: str,
) -> str:
    """Validate one exact string enum."""

    text = _bounded_text(
        value,
        MAX_ACTIVE_PACKET_CHARS,
        field_name,
        required=True,
    )
    if text not in choices:
        raise ConversationProgressContractError(
            f'{field_name} is invalid'
        )
    return text


def _bounded_text(
    value: object,
    maximum_chars: int,
    field_name: str,
    *,
    required: bool = False,
) -> str:
    """Validate one bounded stripped string."""

    if not isinstance(value, str):
        raise ConversationProgressContractError(
            f'{field_name} must be a string'
        )
    text = value.strip()
    if required and not text:
        raise ConversationProgressContractError(f'{field_name} is required')
    if len(text) > maximum_chars:
        raise ConversationProgressContractError(
            f'{field_name} exceeds its character cap'
        )
    return text
