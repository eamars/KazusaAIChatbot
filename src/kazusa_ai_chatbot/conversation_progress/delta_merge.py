"""Validate semantic recorder changes and map private storage identity."""

from __future__ import annotations

import json
import uuid
from collections.abc import Mapping, Sequence
from copy import deepcopy

from kazusa_ai_chatbot.conversation_progress.history import (
    logical_turn_source_refs,
)
from kazusa_ai_chatbot.conversation_progress.models import (
    ConversationProgressEventUpdateV2,
    ConversationProgressEventV2,
    ConversationProgressRecordInput,
    ConversationProgressRecorderDeltaV2,
    ConversationProgressSceneUpdateV2,
    ConversationProgressSourceRefV2,
    ConversationProgressStateV2,
)
from kazusa_ai_chatbot.conversation_progress.policy import (
    MAX_ACTIVE_EVENTS,
    MAX_BLOCK_EVENTS,
    MAX_EPISODE_NARRATIVE_CHARS,
    MAX_EVENT_OUTCOME_CHARS,
    MAX_EVENT_ROLE_CHARS,
    MAX_EVENT_SOURCE_REFS,
    MAX_EVENT_SUMMARY_CHARS,
    MAX_MOVE_CHARS,
    MAX_RECENT_TURN_REFS,
    MAX_THREAD_FIELD_CHARS,
    storage_expiry,
)
from kazusa_ai_chatbot.time_boundary import parse_storage_utc_datetime

_SCENE_OBSERVATION_FIELDS = {
    'schema_version',
    'scene_relation',
    'episode_change',
    'episode_narrative',
    'current_thread',
    'character_stance',
    'user_goal',
    'current_blocker',
    'emotional_trajectory',
    'overused_moves',
}
_EVENT_OBSERVATION_BATCH_FIELDS = {
    'schema_version',
    'existing_events',
    'new_events',
}
_VALID_SCENE_RELATIONS = frozenset({
    'same',
    'related',
    'new',
})
_VALID_EPISODE_CHANGES = frozenset({
    'none',
    'paused',
    'finished',
    'resumed',
})
_UNCHANGED_EVENT_OBSERVATION_FIELDS = {
    'event_handle',
    'observation',
}
_CHANGED_EVENT_OBSERVATION_FIELDS = {
    'event_handle',
    'observation',
    'semantic_summary',
    'outcome',
    'lifecycle_change',
    'relevance',
    'source_turn_handles',
}
_NEW_EVENT_OBSERVATION_FIELDS = {
    'semantic_summary',
    'is_obligation',
    'actor',
    'action',
    'object',
    'beneficiary',
    'precondition',
    'outcome',
    'lifecycle_change',
    'relevance',
    'source_turn_handles',
}
_VALID_LIFECYCLE_CHANGES = {
    'none',
    'began',
    'concluded',
    'declined',
    'replaced',
    'reopened',
}
_VALID_NEW_LIFECYCLE_CHANGES = _VALID_LIFECYCLE_CHANGES - {'reopened'}
_VALID_EVENT_RELEVANCE = {
    'decision',
    'scene',
    'history',
}
_EVENT_SEMANTIC_FIELDS = (
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
)
_SCENE_TEXT_LIMITS = {
    'episode_narrative': MAX_EPISODE_NARRATIVE_CHARS,
    'current_thread': MAX_THREAD_FIELD_CHARS,
    'character_stance': MAX_THREAD_FIELD_CHARS,
    'user_goal': MAX_THREAD_FIELD_CHARS,
    'current_blocker': MAX_THREAD_FIELD_CHARS,
    'emotional_trajectory': MAX_THREAD_FIELD_CHARS,
}
_EVENT_TEXT_LIMITS = {
    'semantic_summary': MAX_EVENT_SUMMARY_CHARS,
    'actor': MAX_EVENT_ROLE_CHARS,
    'action': MAX_EVENT_ROLE_CHARS,
    'object': MAX_EVENT_ROLE_CHARS,
    'beneficiary': MAX_EVENT_ROLE_CHARS,
    'precondition': MAX_EVENT_ROLE_CHARS,
    'outcome': MAX_EVENT_OUTCOME_CHARS,
}


class ConversationProgressContractError(ValueError):
    """Recorder candidate violates the canonical semantic contract."""


def normalize_scene_observation_bounds(
    payload: object,
) -> tuple[object, tuple[dict[str, object], ...]]:
    """Clamp recoverable scene string lengths before exact validation.

    Args:
        payload: Canonically parsed scene-observer output.

    Returns:
        A copied candidate and text-free normalization telemetry.
    """

    if not isinstance(payload, Mapping):
        return payload, ()
    normalized: dict[object, object] = dict(deepcopy(payload))
    reports: list[dict[str, object]] = []
    for field_name, maximum_chars in _SCENE_TEXT_LIMITS.items():
        _clamp_mapping_text(
            normalized,
            field_name=field_name,
            field_path=field_name,
            maximum_chars=maximum_chars,
            reports=reports,
        )

    raw_moves = normalized.get('overused_moves')
    if isinstance(raw_moves, list):
        for index, raw_move in enumerate(raw_moves):
            if not isinstance(raw_move, str):
                continue
            text = raw_move.strip()
            if len(text) <= MAX_MOVE_CHARS:
                continue
            bounded = text[:MAX_MOVE_CHARS].rstrip()
            raw_moves[index] = bounded
            reports.append({
                'field_path': f'overused_moves[{index}]',
                'original_length': len(text),
                'normalized_length': len(bounded),
            })
    return normalized, tuple(reports)


def normalize_event_observation_bounds(
    payload: object,
) -> tuple[object, tuple[dict[str, object], ...]]:
    """Clamp recoverable event text lengths before exact validation."""

    if not isinstance(payload, Mapping):
        return payload, ()
    normalized: dict[object, object] = dict(deepcopy(payload))
    reports: list[dict[str, object]] = []
    for collection_name in ('existing_events', 'new_events'):
        raw_events = normalized.get(collection_name)
        if not isinstance(raw_events, list):
            continue
        for index, raw_event in enumerate(raw_events):
            if not isinstance(raw_event, Mapping):
                continue
            event_observation = dict(raw_event)
            raw_events[index] = event_observation
            for field_name, maximum_chars in _EVENT_TEXT_LIMITS.items():
                _clamp_mapping_text(
                    event_observation,
                    field_name=field_name,
                    field_path=(
                        f'{collection_name}[{index}].{field_name}'
                    ),
                    maximum_chars=maximum_chars,
                    reports=reports,
                )
    return normalized, tuple(reports)


def event_handle_map(
    record_input: ConversationProgressRecordInput,
) -> dict[str, ConversationProgressEventV2]:
    """Assign short model handles while retaining real event IDs privately.

    Args:
        record_input: Settled-turn context containing the prior active packet.

    Returns:
        Ordered ``eN`` handles mapped to exact prior event snapshots.
    """

    prior_packet = record_input['prior_episode_state']
    if prior_packet is None:
        return {}
    handles = {
        f'e{index}': event
        for index, event in enumerate(prior_packet['events'], start=1)
    }
    return handles


def source_handle_map(
    record_input: ConversationProgressRecordInput,
) -> dict[str, list[ConversationProgressSourceRefV2]]:
    """Map semantic turn handles to canonical storage lineage privately.

    Args:
        record_input: Settled-turn context with prior turns and current refs.

    Returns:
        Short turn handles mapped to canonical row or trace references.
    """

    handles: dict[str, list[ConversationProgressSourceRefV2]] = {}
    logical_turns = record_input['interaction_logical_turns']
    canonical_refs = logical_turn_source_refs(logical_turns)
    refs_by_identity = {
        (ref['ref_kind'], ref['ref_id']): ref
        for ref in canonical_refs
    }
    for index, turn in enumerate(
        logical_turns,
        start=1,
    ):
        canonical_ref = None
        if turn['llm_trace_id']:
            canonical_ref = refs_by_identity.get(
                ('llm_trace', turn['llm_trace_id'])
            )
        if canonical_ref is None:
            for row_id in reversed(turn['conversation_row_ids']):
                canonical_ref = refs_by_identity.get(
                    ('conversation_row', row_id)
                )
                if canonical_ref is not None:
                    break
        if canonical_ref is not None:
            handles[f't{index}'] = [deepcopy(canonical_ref)]

    current_refs = record_input['current_turn_source_refs']
    current_input_refs = [
        deepcopy(ref)
        for ref in current_refs
        if ref['ref_kind'] == 'conversation_row'
    ]
    if current_input_refs:
        handles['current_input'] = current_input_refs
    current_response_ref = _last_ref_of_kind(
        current_refs,
        ref_kind='llm_trace',
    )
    if current_response_ref is not None:
        handles['current_response'] = [current_response_ref]
    return handles


def validate_scene_observation(
    payload: object,
    *,
    record_input: ConversationProgressRecordInput,
) -> ConversationProgressSceneUpdateV2:
    """Validate scene-only output and map its storage-facing enums.

    Args:
        payload: Parsed scene-observer output.
        record_input: Settled-turn context containing the prior packet.

    Returns:
        Validated scene facts with deterministically mapped state.

    Raises:
        ConversationProgressContractError: The output violates the contract.
    """

    if not isinstance(payload, Mapping):
        raise ConversationProgressContractError(
            'scene observation must be an object'
        )
    if set(payload) != _SCENE_OBSERVATION_FIELDS:
        raise ConversationProgressContractError(
            'scene observation fields are not exact'
        )
    if (
        payload['schema_version']
        != 'conversation_progress_scene_observation.v2'
    ):
        raise ConversationProgressContractError(
            'scene observation schema_version is invalid'
        )

    scene_relation = _enum_text(
        payload['scene_relation'],
        _VALID_SCENE_RELATIONS,
        'scene_relation',
    )
    episode_change = _enum_text(
        payload['episode_change'],
        _VALID_EPISODE_CHANGES,
        'episode_change',
    )
    continuity = _derive_scene_continuity(scene_relation)
    status = _derive_episode_status(
        episode_change=episode_change,
        prior_packet=record_input['prior_episode_state'],
    )
    scene_update: ConversationProgressSceneUpdateV2 = {
        'continuity': continuity,
        'status': status,
        'episode_narrative': _bounded_text(
            payload['episode_narrative'],
            MAX_EPISODE_NARRATIVE_CHARS,
            'episode_narrative',
        ),
        'current_thread': _bounded_text(
            payload['current_thread'],
            MAX_THREAD_FIELD_CHARS,
            'current_thread',
        ),
        'character_stance': _bounded_text(
            payload['character_stance'],
            MAX_THREAD_FIELD_CHARS,
            'character_stance',
        ),
        'user_goal': _bounded_text(
            payload['user_goal'],
            MAX_THREAD_FIELD_CHARS,
            'user_goal',
        ),
        'current_blocker': _bounded_text(
            payload['current_blocker'],
            MAX_THREAD_FIELD_CHARS,
            'current_blocker',
        ),
        'emotional_trajectory': _bounded_text(
            payload['emotional_trajectory'],
            MAX_THREAD_FIELD_CHARS,
            'emotional_trajectory',
        ),
        'overused_moves': _bounded_string_list(
            payload['overused_moves'],
            field_name='overused_moves',
        ),
    }
    return scene_update


def validate_event_observation_batch(
    payload: object,
    *,
    record_input: ConversationProgressRecordInput,
    supplied_event_handles: set[str],
    supplied_source_handles: set[str],
) -> list[ConversationProgressEventUpdateV2]:
    """Validate exact event coverage and resolve private identities."""

    if not isinstance(payload, Mapping):
        raise ConversationProgressContractError(
            'event observation batch must be an object'
        )
    if set(payload) != _EVENT_OBSERVATION_BATCH_FIELDS:
        raise ConversationProgressContractError(
            'event observation batch fields are not exact'
        )
    if (
        payload['schema_version']
        != 'conversation_progress_event_observation_batch.v2'
    ):
        raise ConversationProgressContractError(
            'event observation batch schema_version is invalid'
        )
    prior_events = event_handle_map(record_input)
    if supplied_event_handles != set(prior_events):
        raise ConversationProgressContractError(
            'event validation requires the complete prior-event ledger'
        )
    all_source_handles = source_handle_map(record_input)
    if not supplied_source_handles.issubset(all_source_handles):
        raise ConversationProgressContractError(
            'supplied source handles are invalid'
        )
    allowed_prior_events = dict(prior_events)
    allowed_source_handles = {
        handle: all_source_handles[handle]
        for handle in supplied_source_handles
    }
    existing_updates = _validate_existing_event_observations(
        payload['existing_events'],
        prior_events=allowed_prior_events,
        source_handles=allowed_source_handles,
    )
    new_updates = _validate_new_event_observations(
        payload['new_events'],
        prior_events=allowed_prior_events,
        source_handles=allowed_source_handles,
    )
    return [*existing_updates, *new_updates]


def compose_recorder_delta(
    *,
    scene_observation: ConversationProgressSceneUpdateV2,
    event_updates: Sequence[ConversationProgressEventUpdateV2],
) -> ConversationProgressRecorderDeltaV2:
    """Compose independently validated semantic owners without rejudgment."""

    validated_delta: ConversationProgressRecorderDeltaV2 = {
        'schema_version': 'conversation_progress_recorder_delta.v2',
        **deepcopy(scene_observation),
        'event_updates': [
            deepcopy(update)
            for update in event_updates
        ],
    }
    return validated_delta


def apply_delta(
    *,
    prior_packet: ConversationProgressStateV2 | None,
    delta: ConversationProgressRecorderDeltaV2,
    record_input: ConversationProgressRecordInput,
    episode_state_id: str,
) -> ConversationProgressStateV2:
    """Merge one validated semantic delta without omission deletion.

    Args:
        prior_packet: Last valid active packet, when one exists.
        delta: Exact-ID delta produced by private handle resolution.
        record_input: Settled-turn scope and timestamp context.
        episode_state_id: Stable episode identity for this write.

    Returns:
        A post-delta packet that may undergo deterministic compaction.
    """

    now_iso = record_input['storage_timestamp_utc']
    parse_storage_utc_datetime(now_iso)
    scope = record_input['scope']
    prior_events = (
        {
            event['event_id']: deepcopy(event)
            for event in prior_packet['events']
        }
        if prior_packet is not None
        else {}
    )
    merged_events = dict(prior_events)
    for update in delta['event_updates']:
        event_id = update['event_id']
        if event_id:
            if event_id not in prior_events:
                raise ConversationProgressContractError(
                    'existing event update ID is unknown'
                )
            prior_event = prior_events[event_id]
            merged_events[event_id] = {
                **deepcopy(update),
                'event_id': event_id,
                'source_refs': _merge_source_refs(
                    prior_event['source_refs'],
                    update['source_refs'],
                ),
                'first_seen_at': prior_event['first_seen_at'],
                'updated_at': now_iso,
            }
            continue

        assigned_id = event_id_for_update(
            episode_state_id=episode_state_id,
            update=update,
        )
        if assigned_id in merged_events:
            raise ConversationProgressContractError(
                'new recorder events contain a deterministic ID collision'
            )
        merged_events[assigned_id] = {
            **deepcopy(update),
            'event_id': assigned_id,
            'first_seen_at': now_iso,
            'updated_at': now_iso,
        }

    events = list(merged_events.values())
    transient_event_limit = MAX_ACTIVE_EVENTS + MAX_BLOCK_EVENTS
    if len(events) > transient_event_limit:
        raise ConversationProgressContractError(
            'post-delta event count exceeds the bounded compaction allowance'
        )

    prior_turn_refs = (
        list(prior_packet['recent_turn_refs'])
        if prior_packet is not None
        else []
    )
    current_turn_refs = [
        turn['turn_id'] for turn in record_input['interaction_logical_turns']
    ]
    current_turn_refs.extend(
        _source_turn_ref(ref)
        for ref in record_input['current_turn_source_refs']
    )
    recent_turn_refs = _bounded_recent_refs(
        [*prior_turn_refs, *current_turn_refs],
    )
    expires_at, purge_after = storage_expiry(now_iso)
    created_at = (
        prior_packet['created_at']
        if prior_packet is not None
        else now_iso
    )
    turn_count = (
        prior_packet['turn_count'] + 1
        if prior_packet is not None
        else 1
    )
    compacted_block_refs = (
        list(prior_packet['compacted_block_refs'])
        if prior_packet is not None
        else []
    )
    packet: ConversationProgressStateV2 = {
        'schema_version': 'conversation_progress.v2',
        'episode_state_id': episode_state_id,
        'platform': scope.platform,
        'platform_channel_id': scope.platform_channel_id,
        'global_user_id': scope.global_user_id,
        'status': delta['status'],
        'continuity': delta['continuity'],
        'turn_count': turn_count,
        'episode_narrative': delta['episode_narrative'],
        'current_thread': delta['current_thread'],
        'character_stance': delta['character_stance'],
        'user_goal': delta['user_goal'],
        'current_blocker': delta['current_blocker'],
        'emotional_trajectory': delta['emotional_trajectory'],
        'events': events,
        'overused_moves': list(delta['overused_moves']),
        'recent_turn_refs': recent_turn_refs,
        'compacted_block_refs': compacted_block_refs,
        'created_at': created_at,
        'updated_at': now_iso,
        'expires_at': expires_at,
        'purge_after': purge_after,
    }
    return packet


def event_id_for_update(
    *,
    episode_state_id: str,
    update: ConversationProgressEventUpdateV2,
) -> str:
    """Assign stable UUID5 identity from mapped lineage and semantics.

    Args:
        episode_state_id: Stable active episode identity.
        update: Fully validated new event with canonical source refs.

    Returns:
        A deterministic hexadecimal UUID5 value.
    """

    if not episode_state_id:
        raise ConversationProgressContractError('episode_state_id is required')
    canonical_payload = {
        'episode_state_id': episode_state_id,
        'source_refs': sorted(
            [
                [ref['ref_kind'], ref['ref_id']]
                for ref in update['source_refs']
            ]
        ),
        **{
            field_name: update[field_name]
            for field_name in _EVENT_SEMANTIC_FIELDS
        },
    }
    canonical_json = json.dumps(
        canonical_payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(',', ':'),
    )
    event_id = uuid.uuid5(uuid.NAMESPACE_URL, canonical_json).hex
    return event_id


def _validate_existing_event_observations(
    value: object,
    *,
    prior_events: Mapping[str, ConversationProgressEventV2],
    source_handles: Mapping[
        str,
        Sequence[ConversationProgressSourceRefV2],
    ],
) -> list[ConversationProgressEventUpdateV2]:
    """Validate one explicit observation for every supplied prior event."""

    if not isinstance(value, list):
        raise ConversationProgressContractError(
            'existing_events must be a list'
        )
    if len(value) != len(prior_events):
        raise ConversationProgressContractError(
            'existing_events lacks exact prior-event handle coverage'
        )
    updates: list[ConversationProgressEventUpdateV2] = []
    seen_handles: set[str] = set()
    observed_handles: list[str] = []
    for raw_observation in value:
        update, event_handle = _validate_existing_event_observation(
            raw_observation,
            prior_events=prior_events,
            source_handles=source_handles,
        )
        observed_handles.append(event_handle)
        if event_handle in seen_handles:
            raise ConversationProgressContractError(
                'existing_events lacks exact prior-event handle coverage'
            )
        seen_handles.add(event_handle)
        if update is not None:
            updates.append(update)
    if seen_handles != set(prior_events):
        raise ConversationProgressContractError(
            'existing_events lacks exact prior-event handle coverage'
        )
    if observed_handles != list(prior_events):
        raise ConversationProgressContractError(
            'existing_events must preserve prior-event input order'
        )
    return updates


def _validate_existing_event_observation(
    value: object,
    *,
    prior_events: Mapping[str, ConversationProgressEventV2],
    source_handles: Mapping[
        str,
        Sequence[ConversationProgressSourceRefV2],
    ],
) -> tuple[ConversationProgressEventUpdateV2 | None, str]:
    """Validate one prior-event observation and resolve private identity."""

    if (
        not isinstance(value, Mapping)
        or 'event_handle' not in value
        or 'observation' not in value
    ):
        raise ConversationProgressContractError(
            'existing event observation fields are not exact'
        )
    event_handle = _text(
        value['event_handle'],
        'event_handle',
        required=True,
    )
    if event_handle not in prior_events:
        raise ConversationProgressContractError(
            'existing event observation references an unknown handle'
        )
    observation = _enum_text(
        value['observation'],
        {'unchanged', 'changed'},
        'observation',
    )
    expected_fields = (
        _UNCHANGED_EVENT_OBSERVATION_FIELDS
        if observation == 'unchanged'
        else _CHANGED_EVENT_OBSERVATION_FIELDS
    )
    if set(value) != expected_fields:
        raise ConversationProgressContractError(
            'existing event observation fields are not exact'
        )
    if observation == 'unchanged':
        return None, event_handle

    semantic_summary = _bounded_text(
        value['semantic_summary'],
        MAX_EVENT_SUMMARY_CHARS,
        'semantic_summary',
        required=True,
    )
    prior_event = prior_events[event_handle]
    outcome = _bounded_text(
        value['outcome'],
        MAX_EVENT_OUTCOME_CHARS,
        'outcome',
    )
    lifecycle_change = _enum_text(
        value['lifecycle_change'],
        _VALID_LIFECYCLE_CHANGES,
        'lifecycle_change',
    )
    relevance = _enum_text(
        value['relevance'],
        _VALID_EVENT_RELEVANCE,
        'relevance',
    )
    state = _derive_event_state(
        lifecycle_change=lifecycle_change,
        event_handle=event_handle,
        prior_events=prior_events,
    )
    retention = _derive_event_retention(relevance)
    source_refs = _source_refs_from_handles(
        value['source_turn_handles'],
        source_handles=source_handles,
    )
    update: ConversationProgressEventUpdateV2 = {
        'event_id': prior_event['event_id'],
        'semantic_summary': semantic_summary,
        'is_obligation': prior_event['is_obligation'],
        'actor': prior_event['actor'],
        'action': prior_event['action'],
        'object': prior_event['object'],
        'beneficiary': prior_event['beneficiary'],
        'precondition': prior_event['precondition'],
        'state': state,
        'outcome': outcome,
        'retention': retention,
        'source_refs': source_refs,
    }
    return update, event_handle


def _validate_new_event_observations(
    value: object,
    *,
    prior_events: Mapping[str, ConversationProgressEventV2],
    source_handles: Mapping[
        str,
        Sequence[ConversationProgressSourceRefV2],
    ],
) -> list[ConversationProgressEventUpdateV2]:
    """Validate separately listed concrete new events."""

    if not isinstance(value, list):
        raise ConversationProgressContractError(
            'new_events must be a list'
        )
    if len(value) > MAX_BLOCK_EVENTS:
        raise ConversationProgressContractError(
            'new_events exceeds the per-turn event cap'
        )
    return [
        _validate_new_event_observation(
            raw_observation,
            prior_events=prior_events,
            source_handles=source_handles,
        )
        for raw_observation in value
    ]


def _validate_new_event_observation(
    value: object,
    *,
    prior_events: Mapping[str, ConversationProgressEventV2],
    source_handles: Mapping[
        str,
        Sequence[ConversationProgressSourceRefV2],
    ],
) -> ConversationProgressEventUpdateV2:
    """Validate one new event before deterministic identity assignment."""

    if not isinstance(value, Mapping) or set(
        value
    ) != _NEW_EVENT_OBSERVATION_FIELDS:
        raise ConversationProgressContractError(
            'new event observation fields are not exact'
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
            'event is_obligation must be boolean'
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
    )
    event_object = _bounded_text(
        value['object'],
        MAX_EVENT_ROLE_CHARS,
        'object',
    )
    if not actor or not action or not event_object:
        raise ConversationProgressContractError(
            'new event requires actor, action, and object'
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
    outcome = _bounded_text(
        value['outcome'],
        MAX_EVENT_OUTCOME_CHARS,
        'outcome',
    )
    lifecycle_change = _enum_text(
        value['lifecycle_change'],
        _VALID_NEW_LIFECYCLE_CHANGES,
        'lifecycle_change',
    )
    relevance = _enum_text(
        value['relevance'],
        _VALID_EVENT_RELEVANCE,
        'relevance',
    )
    state = _derive_event_state(
        lifecycle_change=lifecycle_change,
        event_handle='',
        prior_events=prior_events,
    )
    source_refs = _source_refs_from_handles(
        value['source_turn_handles'],
        source_handles=source_handles,
    )
    return {
        'event_id': '',
        'semantic_summary': semantic_summary,
        'is_obligation': is_obligation,
        'actor': actor,
        'action': action,
        'object': event_object,
        'beneficiary': beneficiary,
        'precondition': precondition,
        'state': state,
        'outcome': outcome,
        'retention': _derive_event_retention(relevance),
        'source_refs': source_refs,
    }


def _derive_event_state(
    *,
    lifecycle_change: str,
    event_handle: str,
    prior_events: Mapping[str, ConversationProgressEventV2],
) -> str:
    """Map one semantic transition onto the persisted lifecycle."""

    prior_state = (
        prior_events[event_handle]['state']
        if event_handle
        else None
    )
    terminal_states = {
        'completed',
        'rejected',
        'superseded',
    }
    if lifecycle_change == 'reopened' and not event_handle:
        raise ConversationProgressContractError(
            'new event cannot claim explicit reopening'
        )
    if (
        lifecycle_change == 'reopened'
        and prior_state not in terminal_states
    ):
        raise ConversationProgressContractError(
            'reopened requires a prior terminal event'
        )

    if lifecycle_change == 'declined':
        state = 'rejected'
    elif lifecycle_change == 'replaced':
        state = 'superseded'
    elif lifecycle_change == 'concluded':
        state = 'completed'
    elif lifecycle_change == 'began':
        state = 'in_progress'
    elif lifecycle_change == 'reopened':
        state = 'open'
    elif prior_state is not None:
        state = prior_state
    else:
        state = 'open'

    if (
        prior_state in terminal_states
        and state in {'open', 'in_progress'}
        and lifecycle_change != 'reopened'
    ):
        raise ConversationProgressContractError(
            'terminal event requires explicit reopening before '
            'a non-terminal transition'
        )
    return state


def _derive_scene_continuity(scene_relation: str) -> str:
    """Map one semantic scene relation to stored continuity."""

    continuity_by_relation = {
        'same': 'same_episode',
        'related': 'related_shift',
        'new': 'sharp_transition',
    }
    return continuity_by_relation[scene_relation]


def _derive_episode_status(
    *,
    episode_change: str,
    prior_packet: ConversationProgressStateV2 | None,
) -> str:
    """Map one semantic episode change to stored status."""

    prior_status = (
        prior_packet['status']
        if prior_packet is not None
        else None
    )
    if (
        episode_change == 'resumed'
        and prior_status not in {'suspended', 'closed'}
    ):
        raise ConversationProgressContractError(
            'resumed requires a prior paused or finished episode'
        )
    if episode_change == 'paused':
        return 'suspended'
    if episode_change == 'finished':
        return 'closed'
    if episode_change == 'resumed':
        return 'active'
    if prior_status is not None:
        return prior_status
    return 'active'


def _derive_event_retention(
    relevance: str,
) -> str:
    """Map one semantic relevance class to storage priority."""

    if relevance == 'decision':
        return 'decision_critical'
    if relevance == 'scene':
        return 'active_scene'
    return 'background'


def _source_refs_from_handles(
    value: object,
    *,
    source_handles: Mapping[
        str,
        Sequence[ConversationProgressSourceRefV2],
    ],
) -> list[ConversationProgressSourceRefV2]:
    """Resolve bounded semantic citations to canonical source refs."""

    if not isinstance(value, list) or not value:
        raise ConversationProgressContractError(
            'source_turn_handles must be a non-empty list'
        )
    if len(value) > len(source_handles):
        raise ConversationProgressContractError(
            'source_turn_handles exceeds the available handle count'
        )
    refs: list[ConversationProgressSourceRefV2] = []
    seen_handles: set[str] = set()
    for raw_handle in value:
        handle = _text(
            raw_handle,
            'source_turn_handle',
            required=True,
        )
        if handle in seen_handles:
            raise ConversationProgressContractError(
                'source_turn_handles contains a duplicate'
            )
        if handle not in source_handles:
            raise ConversationProgressContractError(
                'source_turn_handles contains an unknown handle'
            )
        seen_handles.add(handle)
        refs.extend(deepcopy(list(source_handles[handle])))
    resolved_refs = _deduplicate_source_refs(refs)
    return _bounded_source_refs(resolved_refs)


def _merge_source_refs(
    prior_refs: Sequence[ConversationProgressSourceRefV2],
    current_refs: Sequence[ConversationProgressSourceRefV2],
) -> list[ConversationProgressSourceRefV2]:
    """Preserve original lineage and the newest supporting references."""

    merged = _deduplicate_source_refs([*prior_refs, *current_refs])
    return _bounded_source_refs(merged)


def _bounded_source_refs(
    refs: Sequence[ConversationProgressSourceRefV2],
) -> list[ConversationProgressSourceRefV2]:
    """Preserve the first and newest exact lineage within the storage cap."""

    if len(refs) <= MAX_EVENT_SOURCE_REFS:
        return list(refs)
    return [refs[0], *refs[-(MAX_EVENT_SOURCE_REFS - 1):]]


def _deduplicate_source_refs(
    refs: Sequence[ConversationProgressSourceRefV2],
) -> list[ConversationProgressSourceRefV2]:
    """Deduplicate canonical refs by storage identity in source order."""

    result: list[ConversationProgressSourceRefV2] = []
    seen: set[tuple[str, str]] = set()
    for ref in refs:
        ref_kind = ref['ref_kind']
        ref_id = ref['ref_id']
        occurred_at = ref['occurred_at']
        if not ref_id or not occurred_at:
            raise ConversationProgressContractError(
                'mapped source ref is incomplete'
            )
        parse_storage_utc_datetime(occurred_at)
        identity = (ref_kind, ref_id)
        if identity in seen:
            continue
        seen.add(identity)
        result.append(deepcopy(ref))
    return result


def _last_ref_of_kind(
    refs: Sequence[ConversationProgressSourceRefV2],
    *,
    ref_kind: str,
) -> ConversationProgressSourceRefV2 | None:
    """Return the newest supplied source ref of one exact kind."""

    for ref in reversed(refs):
        if ref['ref_kind'] == ref_kind:
            return deepcopy(ref)
    return None


def _bounded_recent_refs(values: Sequence[str]) -> list[str]:
    """Deduplicate by recency and keep newest protected references."""

    deduplicated: list[str] = []
    for value in values:
        if not value:
            raise ConversationProgressContractError(
                'recent turn ref must be non-empty'
            )
        if value in deduplicated:
            deduplicated.remove(value)
        deduplicated.append(value)
    bounded = deduplicated[-MAX_RECENT_TURN_REFS:]
    return bounded


def _source_turn_ref(ref: ConversationProgressSourceRefV2) -> str:
    """Map one source alias to the protected turn-ref vocabulary."""

    if ref['ref_kind'] == 'conversation_row':
        return f'row:{ref["ref_id"]}'
    return f'trace:{ref["ref_id"]}'


def _bounded_string_list(
    value: object,
    *,
    field_name: str,
) -> list[str]:
    """Validate a compact unique model-authored string list."""

    if not isinstance(value, list):
        raise ConversationProgressContractError(
            f'{field_name} must be a list'
        )
    if len(value) > 8:
        raise ConversationProgressContractError(
            f'{field_name} exceeds its item cap'
        )
    result: list[str] = []
    for item in value:
        text = _bounded_text(
            item,
            MAX_MOVE_CHARS,
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

    text = _text(value, field_name, required=True)
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
    """Validate one stripped text field against its hard cap."""

    text = _text(value, field_name, required=required)
    if len(text) > maximum_chars:
        raise ConversationProgressContractError(
            f'{field_name} exceeds its character cap'
        )
    return text


def _clamp_mapping_text(
    value: dict[object, object],
    *,
    field_name: str,
    field_path: str,
    maximum_chars: int,
    reports: list[dict[str, object]],
) -> None:
    """Clamp one present string while recording lengths without its text."""

    raw_text = value.get(field_name)
    if not isinstance(raw_text, str):
        return
    text = raw_text.strip()
    if len(text) <= maximum_chars:
        return
    bounded = text[:maximum_chars].rstrip()
    value[field_name] = bounded
    reports.append({
        'field_path': field_path,
        'original_length': len(text),
        'normalized_length': len(bounded),
    })


def _text(
    value: object,
    field_name: str,
    *,
    required: bool = False,
) -> str:
    """Validate one scalar string without semantic coercion."""

    if not isinstance(value, str):
        raise ConversationProgressContractError(
            f'{field_name} must be a string'
        )
    text = value.strip()
    if required and not text:
        raise ConversationProgressContractError(f'{field_name} is required')
    return text
