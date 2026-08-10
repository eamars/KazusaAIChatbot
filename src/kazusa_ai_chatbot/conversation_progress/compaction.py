"""Structural compaction requests and immutable block construction."""

from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import datetime

from kazusa_ai_chatbot.conversation_progress.models import (
    ConversationCompactionPlanV2,
    ConversationEpisodeBlockV1,
    ConversationProgressEventV2,
    ConversationProgressStateV2,
)
from kazusa_ai_chatbot.conversation_progress.policy import (
    COMPACTION_EVENT_SOFT_LIMIT,
    COMPACTION_PACKET_SOFT_CHARS,
    COMPACTION_TURN_REF_SOFT_LIMIT,
    MAX_ACTIVE_BLOCK_REFS,
    MAX_BLOCK_CHARS,
    MAX_BLOCK_EVENTS,
    MAX_BLOCK_NARRATIVE_CHARS,
    MAX_BLOCK_SEMANTIC_KEY_CHARS,
    MAX_BLOCK_SEMANTIC_KEYS,
    MAX_BLOCK_SOURCE_BLOCKS,
    MAX_BLOCK_TURN_REFS,
    MAX_EVENT_OUTCOME_CHARS,
    MAX_EVENT_ROLE_CHARS,
    MAX_EVENT_SOURCE_REFS,
    MAX_EVENT_SUMMARY_CHARS,
    TERMINAL_EVENT_STATES,
    VALID_EVENT_RETENTIONS,
    VALID_EVENT_STATES,
    VALID_SOURCE_REF_KINDS,
    canonical_json_chars,
)
from kazusa_ai_chatbot.time_boundary import parse_storage_utc_datetime

_COMPACTION_PLAN_FIELDS = {
    'archive_event_ids',
    'covered_turn_refs',
    'source_block_ids',
}
_BLOCK_FIELDS = {
    'schema_version',
    'block_id',
    'episode_state_id',
    'platform',
    'platform_channel_id',
    'global_user_id',
    'level',
    'source_turn_count',
    'covered_turn_refs',
    'source_block_ids',
    'narrative',
    'events',
    'semantic_keys',
    'source_started_at',
    'source_ended_at',
    'content_hash',
    'superseded_by_block_id',
    'embedding',
    'created_at',
    'expires_at',
    'purge_after',
}


class ConversationCompactionContractError(ValueError):
    """Compaction request, output, or block violates its hard contract."""


def should_compact(
    *,
    active_event_count: int,
    recent_turn_ref_count: int,
    packet_chars: int,
) -> bool:
    """Return whether any approved structural soft threshold is crossed."""

    return (
        active_event_count >= COMPACTION_EVENT_SOFT_LIMIT
        or recent_turn_ref_count >= COMPACTION_TURN_REF_SOFT_LIMIT
        or packet_chars >= COMPACTION_PACKET_SOFT_CHARS
    )


def build_compaction_plan(
    *,
    active_packet: ConversationProgressStateV2 | None,
    active_blocks: Sequence[ConversationEpisodeBlockV1],
) -> ConversationCompactionPlanV2 | None:
    """Select archival structure from validated lifecycle labels.

    Args:
        active_packet: Post-delta active packet being fitted.
        active_blocks: Exact blocks referenced by that packet.

    Returns:
        A deterministic compaction plan, or ``None`` below thresholds.
    """

    if active_packet is None:
        return None
    packet_chars = canonical_json_chars(active_packet)
    if not should_compact(
        active_event_count=len(active_packet['events']),
        recent_turn_ref_count=len(active_packet['recent_turn_refs']),
        packet_chars=packet_chars,
    ):
        return None

    archive_candidates = sorted(
        (
            event
            for event in active_packet['events']
            if (
                event['state'] in TERMINAL_EVENT_STATES
                and event['retention'] != 'decision_critical'
            )
        ),
        key=lambda event: (
            0 if event['retention'] == 'background' else 1,
            event['updated_at'],
            event['event_id'],
        ),
    )[:MAX_BLOCK_EVENTS]
    if not archive_candidates:
        return None

    source_block_ids: list[str] = []
    if len(active_packet['compacted_block_refs']) >= MAX_ACTIVE_BLOCK_REFS:
        source_block_ids = [
            block['block_id']
            for block in _select_compaction_source_blocks(
                active_packet=active_packet,
                active_blocks=active_blocks,
            )
        ]

    plan: ConversationCompactionPlanV2 = {
        'archive_event_ids': [
            event['event_id'] for event in archive_candidates
        ],
        'covered_turn_refs': list(
            active_packet['recent_turn_refs'][:MAX_BLOCK_TURN_REFS]
        ),
        'source_block_ids': source_block_ids,
    }
    validate_compaction_plan(
        plan,
        active_packet=active_packet,
        active_blocks=active_blocks,
    )
    return plan


def validate_compaction_plan(
    value: object,
    *,
    active_packet: ConversationProgressStateV2,
    active_blocks: Sequence[ConversationEpisodeBlockV1],
) -> ConversationCompactionPlanV2:
    """Validate a code-owned plan against the exact active state."""

    if not isinstance(value, Mapping) or set(value) != _COMPACTION_PLAN_FIELDS:
        raise ConversationCompactionContractError(
            'compaction plan fields are not exact'
        )
    packet_events = {
        event['event_id']: event for event in active_packet['events']
    }
    archive_event_ids = _unique_subset(
        value['archive_event_ids'],
        allowed=set(packet_events),
        maximum_items=MAX_BLOCK_EVENTS,
        field_name='archive_event_ids',
    )
    if not archive_event_ids:
        raise ConversationCompactionContractError(
            'compaction plan requires an archived event'
        )
    for event_id in archive_event_ids:
        event = packet_events[event_id]
        if (
            event['state'] not in TERMINAL_EVENT_STATES
            or event['retention'] == 'decision_critical'
        ):
            raise ConversationCompactionContractError(
                'compaction plan contains a protected event'
            )
    covered_turn_refs = _unique_subset(
        value['covered_turn_refs'],
        allowed=set(active_packet['recent_turn_refs']),
        maximum_items=MAX_BLOCK_TURN_REFS,
        field_name='covered_turn_refs',
    )
    active_block_ids = {
        block['block_id'] for block in active_blocks
    }
    source_block_ids = _unique_subset(
        value['source_block_ids'],
        allowed=active_block_ids,
        maximum_items=MAX_BLOCK_SOURCE_BLOCKS,
        field_name='source_block_ids',
    )
    expected_source_ids: list[str] = []
    if len(active_packet['compacted_block_refs']) >= MAX_ACTIVE_BLOCK_REFS:
        expected_source_ids = [
            block['block_id']
            for block in _select_compaction_source_blocks(
                active_packet=active_packet,
                active_blocks=active_blocks,
            )
        ]
    if source_block_ids != expected_source_ids:
        raise ConversationCompactionContractError(
            'compaction plan source blocks are not the balanced active refs'
        )
    plan: ConversationCompactionPlanV2 = {
        'archive_event_ids': archive_event_ids,
        'covered_turn_refs': covered_turn_refs,
        'source_block_ids': source_block_ids,
    }
    return plan


def create_block_from_plan(
    *,
    compaction_plan: ConversationCompactionPlanV2,
    active_packet: ConversationProgressStateV2,
    active_blocks: Sequence[ConversationEpisodeBlockV1],
) -> ConversationEpisodeBlockV1:
    """Create one immutable block from exact stored semantic snapshots."""

    plan = validate_compaction_plan(
        compaction_plan,
        active_packet=active_packet,
        active_blocks=active_blocks,
    )
    packet_events = {
        event['event_id']: event for event in active_packet['events']
    }
    retained_events = [
        deepcopy(packet_events[event_id])
        for event_id in plan['archive_event_ids']
    ]
    source_blocks = {
        block['block_id']: block for block in active_blocks
    }
    source_block_ids = list(plan['source_block_ids'])
    level = 0
    if source_block_ids:
        level = (
            max(
                source_blocks[block_id]['level']
                for block_id in source_block_ids
            )
            + 1
        )
    narrative = _deterministic_block_narrative(
        events=retained_events,
        source_block_ids=source_block_ids,
        source_blocks=source_blocks,
    )
    semantic_keys = _deterministic_semantic_keys(
        events=retained_events,
        source_block_ids=source_block_ids,
        source_blocks=source_blocks,
    )
    source_times = _source_times(
        events=retained_events,
        source_block_ids=source_block_ids,
        source_blocks=source_blocks,
        fallback=active_packet['updated_at'],
    )
    block_id = _block_id(
        episode_state_id=active_packet['episode_state_id'],
        source_turn_count=active_packet['turn_count'],
        level=level,
        archive_event_ids=plan['archive_event_ids'],
        covered_turn_refs=plan['covered_turn_refs'],
        source_block_ids=source_block_ids,
    )
    immutable_content = {
        'schema_version': 'conversation_progress_block.v1',
        'block_id': block_id,
        'episode_state_id': active_packet['episode_state_id'],
        'platform': active_packet['platform'],
        'platform_channel_id': active_packet['platform_channel_id'],
        'global_user_id': active_packet['global_user_id'],
        'level': level,
        'source_turn_count': active_packet['turn_count'],
        'covered_turn_refs': list(plan['covered_turn_refs']),
        'source_block_ids': source_block_ids,
        'narrative': narrative,
        'events': retained_events,
        'semantic_keys': semantic_keys,
        'source_started_at': source_times[0],
        'source_ended_at': source_times[1],
    }
    content_hash = hashlib.sha256(
        json.dumps(
            immutable_content,
            ensure_ascii=False,
            sort_keys=True,
            separators=(',', ':'),
        ).encode('utf-8')
    ).hexdigest()
    block: ConversationEpisodeBlockV1 = {
        **immutable_content,
        'content_hash': content_hash,
        'superseded_by_block_id': '',
        'embedding': [],
        'created_at': active_packet['updated_at'],
        'expires_at': active_packet['expires_at'],
        'purge_after': active_packet['purge_after'],
    }
    validate_block(block)
    return block


def apply_compaction_to_packet(
    *,
    active_packet: ConversationProgressStateV2,
    compaction_plan: ConversationCompactionPlanV2,
    block_id: str,
) -> ConversationProgressStateV2:
    """Remove archived events and replace active source-block references."""

    archived_ids = set(compaction_plan['archive_event_ids'])
    source_block_ids = set(compaction_plan['source_block_ids'])
    covered_turn_refs = set(compaction_plan['covered_turn_refs'])
    result = deepcopy(active_packet)
    result['events'] = [
        event
        for event in result['events']
        if event['event_id'] not in archived_ids
    ]
    result['recent_turn_refs'] = [
        turn_ref
        for turn_ref in result['recent_turn_refs']
        if turn_ref not in covered_turn_refs
    ]
    block_refs = [
        existing_id
        for existing_id in result['compacted_block_refs']
        if existing_id not in source_block_ids
    ]
    if block_id in block_refs:
        block_refs.remove(block_id)
    block_refs.append(block_id)
    if len(block_refs) > MAX_ACTIVE_BLOCK_REFS:
        raise ConversationCompactionContractError(
            'compaction leaves too many active block references'
        )
    result['compacted_block_refs'] = block_refs
    return result


def block_embedding_text(block: ConversationEpisodeBlockV1) -> str:
    """Render only validated semantic block content for embedding."""

    event_lines = [
        (
            f'{event["semantic_summary"]}; '
            f'state={event["state"]}; retention={event["retention"]}'
        )
        for event in block['events']
    ]
    parts = [
        block['narrative'],
        '; '.join(block['semantic_keys']),
        '\n'.join(event_lines),
    ]
    return '\n'.join(part for part in parts if part)


def validate_block(value: object) -> ConversationEpisodeBlockV1:
    """Validate the exact immutable block document and every hard cap."""

    if not isinstance(value, Mapping) or set(value) != _BLOCK_FIELDS:
        raise ConversationCompactionContractError(
            'conversation progress block fields are not exact'
        )
    if value['schema_version'] != 'conversation_progress_block.v1':
        raise ConversationCompactionContractError(
            'conversation progress block schema is invalid'
        )
    for field_name in (
        'block_id',
        'episode_state_id',
        'platform',
        'global_user_id',
        'narrative',
        'source_started_at',
        'source_ended_at',
        'content_hash',
        'created_at',
        'expires_at',
    ):
        _bounded_text(
            value[field_name],
            MAX_BLOCK_CHARS,
            field_name,
            required=True,
        )
    _bounded_text(
        value['platform_channel_id'],
        MAX_BLOCK_CHARS,
        'platform_channel_id',
    )
    _bounded_text(
        value['superseded_by_block_id'],
        MAX_BLOCK_CHARS,
        'superseded_by_block_id',
    )
    if (
        not isinstance(value['level'], int)
        or isinstance(value['level'], bool)
        or value['level'] < 0
    ):
        raise ConversationCompactionContractError('block level is invalid')
    if (
        not isinstance(value['source_turn_count'], int)
        or isinstance(value['source_turn_count'], bool)
        or value['source_turn_count'] < 1
    ):
        raise ConversationCompactionContractError(
            'block source_turn_count is invalid'
        )
    _unique_text_list(
        value['covered_turn_refs'],
        maximum_items=MAX_BLOCK_TURN_REFS,
        maximum_chars=MAX_BLOCK_CHARS,
        field_name='covered_turn_refs',
    )
    _unique_text_list(
        value['source_block_ids'],
        maximum_items=MAX_BLOCK_SOURCE_BLOCKS,
        maximum_chars=MAX_BLOCK_CHARS,
        field_name='source_block_ids',
    )
    _unique_text_list(
        value['semantic_keys'],
        maximum_items=MAX_BLOCK_SEMANTIC_KEYS,
        maximum_chars=MAX_BLOCK_SEMANTIC_KEY_CHARS,
        field_name='semantic_keys',
    )
    events = value['events']
    if not isinstance(events, list) or len(events) > MAX_BLOCK_EVENTS:
        raise ConversationCompactionContractError(
            'block events exceeds its hard cap'
        )
    validated_events = [_validate_block_event(event) for event in events]
    event_ids = [event['event_id'] for event in validated_events]
    if len(event_ids) != len(set(event_ids)):
        raise ConversationCompactionContractError(
            'block event IDs are duplicated'
        )
    if not isinstance(value['embedding'], list) or any(
        not isinstance(component, float)
        for component in value['embedding']
    ):
        raise ConversationCompactionContractError(
            'block embedding must be a float list'
        )
    if not isinstance(value['purge_after'], datetime):
        raise ConversationCompactionContractError(
            'block purge_after must be a BSON datetime'
        )
    for field_name in (
        'source_started_at',
        'source_ended_at',
        'created_at',
        'expires_at',
    ):
        parse_storage_utc_datetime(str(value[field_name]))
    measured = dict(value)
    measured.pop('embedding')
    if canonical_json_chars(measured) > MAX_BLOCK_CHARS:
        raise ConversationCompactionContractError(
            'conversation progress block exceeds its hard character cap'
        )
    immutable_content = {
        field_name: value[field_name]
        for field_name in (
            'schema_version',
            'block_id',
            'episode_state_id',
            'platform',
            'platform_channel_id',
            'global_user_id',
            'level',
            'source_turn_count',
            'covered_turn_refs',
            'source_block_ids',
            'narrative',
            'events',
            'semantic_keys',
            'source_started_at',
            'source_ended_at',
        )
    }
    expected_hash = hashlib.sha256(
        json.dumps(
            immutable_content,
            ensure_ascii=False,
            sort_keys=True,
            separators=(',', ':'),
        ).encode('utf-8')
    ).hexdigest()
    if value['content_hash'] != expected_hash:
        raise ConversationCompactionContractError(
            'conversation progress block content hash is invalid'
        )
    return deepcopy(value)  # type: ignore[return-value]


def _validate_block_event(value: object) -> ConversationProgressEventV2:
    """Validate one retained event snapshot inside an immutable block."""

    event_fields = {
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
    if not isinstance(value, Mapping) or set(value) != event_fields:
        raise ConversationCompactionContractError(
            'block event fields are not exact'
        )
    for field_name in ('event_id', 'first_seen_at', 'updated_at'):
        _bounded_text(
            value[field_name],
            MAX_BLOCK_CHARS,
            field_name,
            required=True,
        )
    _bounded_text(
        value['semantic_summary'],
        MAX_EVENT_SUMMARY_CHARS,
        'semantic_summary',
        required=True,
    )
    for field_name in (
        'actor',
        'action',
        'object',
        'beneficiary',
        'precondition',
    ):
        _bounded_text(
            value[field_name],
            MAX_EVENT_ROLE_CHARS,
            field_name,
        )
    _bounded_text(
        value['outcome'],
        MAX_EVENT_OUTCOME_CHARS,
        'outcome',
    )
    if not isinstance(value['is_obligation'], bool):
        raise ConversationCompactionContractError(
            'block event is_obligation must be boolean'
        )
    if not value['actor'] or not value['action'] or not value['object']:
        raise ConversationCompactionContractError(
            'block event requires actor, action, and object'
        )
    if value['state'] not in VALID_EVENT_STATES:
        raise ConversationCompactionContractError(
            'block event state is invalid'
        )
    if value['retention'] not in VALID_EVENT_RETENTIONS:
        raise ConversationCompactionContractError(
            'block event retention is invalid'
        )
    source_refs = value['source_refs']
    if (
        not isinstance(source_refs, list)
        or not source_refs
        or len(source_refs) > MAX_EVENT_SOURCE_REFS
    ):
        raise ConversationCompactionContractError(
            'block event source_refs must be non-empty'
        )
    seen_refs: set[tuple[str, str]] = set()
    for source_ref in source_refs:
        if (
            not isinstance(source_ref, Mapping)
            or set(source_ref) != {'ref_kind', 'ref_id', 'occurred_at'}
            or source_ref['ref_kind'] not in VALID_SOURCE_REF_KINDS
        ):
            raise ConversationCompactionContractError(
                'block event source ref is invalid'
            )
        ref_id = _bounded_text(
            source_ref['ref_id'],
            MAX_BLOCK_CHARS,
            'ref_id',
            required=True,
        )
        occurred_at = _bounded_text(
            source_ref['occurred_at'],
            MAX_BLOCK_CHARS,
            'occurred_at',
            required=True,
        )
        parse_storage_utc_datetime(occurred_at)
        identity = (str(source_ref['ref_kind']), ref_id)
        if identity in seen_refs:
            raise ConversationCompactionContractError(
                'block event source refs are duplicated'
            )
        seen_refs.add(identity)
    parse_storage_utc_datetime(str(value['first_seen_at']))
    parse_storage_utc_datetime(str(value['updated_at']))
    return deepcopy(value)  # type: ignore[return-value]


def _deterministic_block_narrative(
    *,
    events: Sequence[ConversationProgressEventV2],
    source_block_ids: Sequence[str],
    source_blocks: Mapping[str, ConversationEpisodeBlockV1],
) -> str:
    """Join only previously authored semantic text for block retrieval."""

    parts: list[str] = []
    for block_id in source_block_ids:
        narrative = source_blocks[block_id]['narrative'].strip()
        if narrative and narrative not in parts:
            parts.append(narrative)
    for event in events:
        event_text = event['semantic_summary'].strip()
        if event['outcome']:
            event_text = f'{event_text}; {event["outcome"]}'
        if event_text and event_text not in parts:
            parts.append(event_text)
    narrative = '\n'.join(parts)[:MAX_BLOCK_NARRATIVE_CHARS].rstrip()
    if not narrative:
        raise ConversationCompactionContractError(
            'compaction has no stored semantic narrative'
        )
    return narrative


def _deterministic_semantic_keys(
    *,
    events: Sequence[ConversationProgressEventV2],
    source_block_ids: Sequence[str],
    source_blocks: Mapping[str, ConversationEpisodeBlockV1],
) -> list[str]:
    """Reuse bounded stored semantic excerpts without new interpretation."""

    candidates: list[str] = []
    for event in events:
        candidates.append(event['semantic_summary'])
    for block_id in source_block_ids:
        candidates.extend(source_blocks[block_id]['semantic_keys'])
    keys: list[str] = []
    for candidate in candidates:
        key = candidate[:MAX_BLOCK_SEMANTIC_KEY_CHARS].strip()
        if not key or key in keys:
            continue
        keys.append(key)
        if len(keys) >= MAX_BLOCK_SEMANTIC_KEYS:
            break
    return keys


def _source_times(
    *,
    events: Sequence[ConversationProgressEventV2],
    source_block_ids: Sequence[str],
    source_blocks: Mapping[str, ConversationEpisodeBlockV1],
    fallback: str,
) -> tuple[str, str]:
    """Compute the exact source span from event and block metadata."""

    timestamps = [
        ref['occurred_at']
        for event in events
        for ref in event['source_refs']
    ]
    for block_id in source_block_ids:
        block = source_blocks[block_id]
        timestamps.extend([
            block['source_started_at'],
            block['source_ended_at'],
        ])
    if not timestamps:
        timestamps = [fallback]
    for timestamp in timestamps:
        parse_storage_utc_datetime(timestamp)
    source_times = (min(timestamps), max(timestamps))
    return source_times


def _block_id(
    *,
    episode_state_id: str,
    source_turn_count: int,
    level: int,
    archive_event_ids: Sequence[str],
    covered_turn_refs: Sequence[str],
    source_block_ids: Sequence[str],
) -> str:
    """Return the approved stable UUID5 block identity."""

    canonical_payload = {
        'episode_state_id': episode_state_id,
        'source_turn_count': source_turn_count,
        'level': level,
        'archive_event_ids': sorted(archive_event_ids),
        'covered_turn_refs': sorted(covered_turn_refs),
        'source_block_ids': sorted(source_block_ids),
    }
    canonical_json = json.dumps(
        canonical_payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(',', ':'),
    )
    return uuid.uuid5(uuid.NAMESPACE_URL, canonical_json).hex


def _unique_subset(
    value: object,
    *,
    allowed: set[str],
    maximum_items: int,
    field_name: str,
) -> list[str]:
    """Validate one ordered unique list against a supplied candidate set."""

    values = _unique_text_list(
        value,
        maximum_items=maximum_items,
        maximum_chars=MAX_BLOCK_CHARS,
        field_name=field_name,
    )
    if not set(values).issubset(allowed):
        raise ConversationCompactionContractError(
            f'{field_name} contains an unsupplied candidate'
        )
    return values


def _select_compaction_source_blocks(
    *,
    active_packet: ConversationProgressStateV2,
    active_blocks: Sequence[ConversationEpisodeBlockV1],
) -> list[ConversationEpisodeBlockV1]:
    """Select low-level roots first and use age only within one level."""

    by_id: dict[str, ConversationEpisodeBlockV1] = {}
    for block in active_blocks:
        block_id = block['block_id']
        if block_id in by_id:
            raise ConversationCompactionContractError(
                'active block input contains duplicate block IDs'
            )
        by_id[block_id] = block
    missing_ids = [
        block_id
        for block_id in active_packet['compacted_block_refs']
        if block_id not in by_id
    ]
    if missing_ids:
        raise ConversationCompactionContractError(
            'active packet references a missing compaction block'
        )
    return sorted(
        (
            by_id[block_id]
            for block_id in active_packet['compacted_block_refs']
        ),
        key=lambda block: (
            block['level'],
            block['source_turn_count'],
            block['source_ended_at'],
            block['block_id'],
        ),
    )[:MAX_BLOCK_SOURCE_BLOCKS]


def _unique_text_list(
    value: object,
    *,
    maximum_items: int,
    maximum_chars: int,
    field_name: str,
) -> list[str]:
    """Validate one bounded, unique string list."""

    if not isinstance(value, list):
        raise ConversationCompactionContractError(
            f'{field_name} must be a list'
        )
    if len(value) > maximum_items:
        raise ConversationCompactionContractError(
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
            raise ConversationCompactionContractError(
                f'{field_name} contains a duplicate'
            )
        result.append(text)
    return result


def _bounded_text(
    value: object,
    maximum_chars: int,
    field_name: str,
    *,
    required: bool = False,
) -> str:
    """Validate one bounded string field."""

    if not isinstance(value, str):
        raise ConversationCompactionContractError(
            f'{field_name} must be a string'
        )
    text = value.strip()
    if required and not text:
        raise ConversationCompactionContractError(
            f'{field_name} is required'
        )
    if len(text) > maximum_chars:
        raise ConversationCompactionContractError(
            f'{field_name} exceeds its character cap'
        )
    return text
