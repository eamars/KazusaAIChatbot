"""Exact V2 packet, observation, handle, and identity contracts."""

from __future__ import annotations

from copy import deepcopy

import pytest

from kazusa_ai_chatbot.conversation_progress.delta_merge import (
    ConversationProgressContractError,
    event_handle_map,
    event_id_for_update,
    source_handle_map,
    validate_event_observation_batch,
    validate_scene_observation,
)
from kazusa_ai_chatbot.conversation_progress.policy import (
    CONVERSATION_PROGRESS_ACTIVE_SCENE_MAX_AGE_MINUTES,
    CONVERSATION_PROGRESS_BACKGROUND_MAX_AGE_MINUTES,
    CONVERSATION_PROGRESS_DECISION_CRITICAL_MAX_AGE_MINUTES,
    CONVERSATION_PROGRESS_NARRATIVE_MAX_AGE_MINUTES,
    GROUP_SCENE_MAX_TURN_AGE_MINUTES,
    MAX_ACTIVE_BLOCK_REFS,
    MAX_ACTIVE_EVENTS,
    MAX_ACTIVE_PACKET_CHARS,
    MAX_BLOCK_GRAPH_DEPTH,
    MAX_EPISODE_NARRATIVE_CHARS,
    MAX_RECENT_TURN_REFS,
    MAX_REACHABLE_BLOCK_REFS,
    MAX_THREAD_FIELD_CHARS,
    prune_aged_progress_packet,
)
from kazusa_ai_chatbot.conversation_progress.repository import (
    validate_active_packet,
)
from tests.test_conversation_progress_group_scene import (
    _anchor_fixture,
    _build_from_fixture,
)
from tests.conversation_progress_v2_helpers import (
    EPISODE_ID,
    changed_event_observation,
    event,
    event_observation_batch,
    new_event_observation,
    packet,
    record_input,
    scene_observation,
)


def test_approved_policy_constants_are_exact() -> None:
    """Keep all deterministic capacity limits at their approved values."""

    assert GROUP_SCENE_MAX_TURN_AGE_MINUTES == 120
    assert CONVERSATION_PROGRESS_BACKGROUND_MAX_AGE_MINUTES == 120
    assert CONVERSATION_PROGRESS_ACTIVE_SCENE_MAX_AGE_MINUTES == 360
    assert CONVERSATION_PROGRESS_DECISION_CRITICAL_MAX_AGE_MINUTES == 2880
    assert CONVERSATION_PROGRESS_NARRATIVE_MAX_AGE_MINUTES == 360
    assert MAX_EPISODE_NARRATIVE_CHARS == 900
    assert MAX_THREAD_FIELD_CHARS == 240
    assert MAX_ACTIVE_EVENTS == 24
    assert MAX_RECENT_TURN_REFS == 16
    assert MAX_ACTIVE_BLOCK_REFS == 8
    assert MAX_BLOCK_GRAPH_DEPTH == 8
    assert MAX_REACHABLE_BLOCK_REFS == 128
    assert MAX_ACTIVE_PACKET_CHARS == 16000


def test_exact_v2_packet_with_bson_expiry_is_accepted() -> None:
    """Accept the factual packet with code-owned expiry metadata."""

    validated = validate_active_packet(packet(events=[event()]))

    assert validated['schema_version'] == 'conversation_progress.v2'


def test_prune_drops_events_at_each_retention_tier_boundary() -> None:
    """Drop an event only when its age exceeds its retention-tier threshold."""

    boundaries = (
        (
            'background',
            '2026-07-28T07:30:00+00:00',
            '2026-07-28T07:29:59+00:00',
        ),
        (
            'active_scene',
            '2026-07-28T03:30:00+00:00',
            '2026-07-28T03:29:59+00:00',
        ),
        (
            'decision_critical',
            '2026-07-26T09:30:00+00:00',
            '2026-07-26T09:29:59+00:00',
        ),
    )
    for retention, at_limit_timestamp, beyond_limit_timestamp in boundaries:
        at_limit = packet(events=[event(
            event_id=f'{retention}_at_limit',
            retention=retention,
        )])
        at_limit['events'][0]['updated_at'] = at_limit_timestamp
        pruned, dropped_count, narrative_cleared = prune_aged_progress_packet(
            at_limit,
            current_timestamp_utc='2026-07-28T09:30:00+00:00',
        )
        assert dropped_count == 0
        assert narrative_cleared is (retention == 'decision_critical')
        assert len(pruned['events']) == 1

        beyond_limit = packet(events=[event(
            event_id=f'{retention}_beyond_limit',
            retention=retention,
        )])
        beyond_limit['events'][0]['updated_at'] = beyond_limit_timestamp
        pruned, dropped_count, _ = prune_aged_progress_packet(
            beyond_limit,
            current_timestamp_utc='2026-07-28T09:30:00+00:00',
        )
        assert dropped_count == 1
        assert pruned['events'] == []


def test_prune_clears_narrative_when_newest_event_is_stale() -> None:
    """Clear the complete narrative set when the newest survivor is stale."""

    stale = packet(events=[event(
        event_id='stale_scene_event',
        retention='decision_critical',
    )])
    stale['events'][0]['updated_at'] = '2026-07-27T00:00:00+00:00'

    pruned, dropped_count, narrative_cleared = prune_aged_progress_packet(
        stale,
        current_timestamp_utc='2026-07-28T09:30:00+00:00',
    )

    assert dropped_count == 0
    assert narrative_cleared is True
    for field_name in (
        'current_thread',
        'character_stance',
        'user_goal',
        'current_blocker',
        'emotional_trajectory',
        'episode_narrative',
    ):
        assert pruned[field_name] == ''
    assert pruned['overused_moves'] == []


def test_prune_preserves_fresh_narrative_and_other_packet_fields() -> None:
    """Keep narrative intact while preserving every non-event field."""

    fresh = packet(
        turn_count=4,
        events=[event(
            event_id='fresh_event',
            retention='active_scene',
        )],
        recent_turn_refs=['row:row_1', 'row:row_2'],
    )
    fresh['current_thread'] = 'keep this thread'
    fresh['overused_moves'] = ['repeating the same offer']

    pruned, dropped_count, narrative_cleared = prune_aged_progress_packet(
        fresh,
        current_timestamp_utc='2026-07-28T09:30:00+00:00',
    )

    assert dropped_count == 0
    assert narrative_cleared is False
    assert pruned['current_thread'] == 'keep this thread'
    assert pruned['overused_moves'] == ['repeating the same offer']
    assert pruned['turn_count'] == 4
    assert pruned['recent_turn_refs'] == ['row:row_1', 'row:row_2']
    assert pruned['created_at'] == '2026-07-28T09:30:00+00:00'
    assert pruned['updated_at'] == '2026-07-28T09:30:00+00:00'
    assert pruned['expires_at'] == '2026-07-30T09:30:00+00:00'


def test_pruned_packet_satisfies_active_packet_validation() -> None:
    """A pruned packet, including an empty event list, stays valid."""

    aged = packet(turn_count=3, events=[
        event(
            event_id='aged_background',
            retention='background',
        ),
        event(
            event_id='aged_scene',
            retention='active_scene',
        ),
    ])
    for event_row in aged['events']:
        event_row['updated_at'] = '2026-07-20T09:30:00+00:00'

    pruned, _, narrative_cleared = prune_aged_progress_packet(
        aged,
        current_timestamp_utc='2026-07-28T09:30:00+00:00',
    )
    validated = validate_active_packet(pruned)

    assert narrative_cleared is True
    assert validated['events'] == []
    assert validated['turn_count'] == 3


@pytest.mark.parametrize(
    'mutation',
    [
        lambda value: value.update(schema_version='conversation_progress.v1'),
        lambda value: value.pop('purge_after'),
        lambda value: value.update(unexpected=True),
        lambda value: value.update(events=[event(source_refs=[])]),
        lambda value: value.update(events=[event(summary='x' * 221)]),
        lambda value: value.update(next_affordances=['future advice']),
        lambda value: value.update(progression_guidance='future advice'),
    ],
)
def test_invalid_or_storage_guidance_packet_is_rejected(mutation) -> None:
    """Reject V1, malformed, and future-planning persistence shapes."""

    candidate = packet(events=[event()])
    mutation(candidate)

    with pytest.raises(ConversationProgressContractError):
        validate_active_packet(candidate)


def test_new_event_rejects_unknown_semantic_source_handle() -> None:
    """Keep source identity mapping inside the deterministic allowlist."""

    submitted = record_input()
    candidate = new_event_observation(
        source_turn_handles=['unknown-turn'],
    )

    with pytest.raises(
        ConversationProgressContractError,
        match='unknown handle',
    ):
        validate_event_observation_batch(
            event_observation_batch(new_events=[candidate]),
            record_input=submitted,
            supplied_event_handles=set(event_handle_map(submitted)),
            supplied_source_handles=set(source_handle_map(submitted)),
        )


def test_changed_prior_event_requires_source_handle() -> None:
    """Require current semantic evidence for every emitted change."""

    prior = packet(events=[event(event_id='prior_event')])
    submitted = record_input(prior_packet=prior)
    change = changed_event_observation(
        event_handle='e1',
        summary='changed semantics',
        source_turn_handles=[],
    )

    with pytest.raises(
        ConversationProgressContractError,
        match='non-empty',
    ):
        validate_event_observation_batch(
            event_observation_batch(existing_events=[change]),
            record_input=submitted,
            supplied_event_handles=set(event_handle_map(submitted)),
            supplied_source_handles=set(source_handle_map(submitted)),
        )


def test_omitted_prior_event_is_rejected_by_exact_coverage_contract() -> None:
    """Make silence distinguishable from an explicit unchanged observation."""

    prior = packet(events=[event(event_id='prior_event')])
    submitted = record_input(prior_packet=prior)

    with pytest.raises(
        ConversationProgressContractError,
        match='coverage',
    ):
        validate_event_observation_batch(
            event_observation_batch(),
            record_input=submitted,
            supplied_event_handles=set(event_handle_map(submitted)),
            supplied_source_handles=set(source_handle_map(submitted)),
        )


@pytest.mark.parametrize(
    'forbidden_field',
    [
        'discard_event_ids',
        'compaction',
        'next_affordances',
        'progression_guidance',
    ],
)
def test_semantic_delta_rejects_operational_or_future_fields(
    forbidden_field: str,
) -> None:
    """Keep storage operations and future goals outside event output."""

    submitted = record_input()
    candidate = event_observation_batch()
    candidate[forbidden_field] = []

    with pytest.raises(
        ConversationProgressContractError,
        match='fields are not exact',
    ):
        validate_event_observation_batch(
            candidate,
            record_input=submitted,
            supplied_event_handles=set(event_handle_map(submitted)),
            supplied_source_handles=set(source_handle_map(submitted)),
        )


@pytest.mark.parametrize(
    'forbidden_field',
    ['status', 'continuity', 'next_affordances', 'progression_guidance'],
)
def test_scene_observation_rejects_operational_or_future_fields(
    forbidden_field: str,
) -> None:
    """Keep persistence and future response planning outside scene output."""

    candidate = scene_observation()
    candidate[forbidden_field] = []

    with pytest.raises(
        ConversationProgressContractError,
        match='fields are not exact',
    ):
        validate_scene_observation(
            candidate,
            record_input=record_input(),
        )


def test_new_event_id_uses_stable_uuid5_over_mapped_semantics() -> None:
    """Generate storage identity only after semantic handle validation."""

    submitted = record_input()
    validated = validate_event_observation_batch(
        event_observation_batch(new_events=[new_event_observation()]),
        record_input=submitted,
        supplied_event_handles=set(event_handle_map(submitted)),
        supplied_source_handles=set(source_handle_map(submitted)),
    )
    update = validated[0]
    first = event_id_for_update(
        episode_state_id=EPISODE_ID,
        update=update,
    )
    second = event_id_for_update(
        episode_state_id=EPISODE_ID,
        update=deepcopy(update),
    )
    changed_updates = validate_event_observation_batch(
        event_observation_batch(new_events=[new_event_observation(
            summary='different event',
        )]),
        record_input=submitted,
        supplied_event_handles=set(event_handle_map(submitted)),
        supplied_source_handles=set(source_handle_map(submitted)),
    )
    changed = event_id_for_update(
        episode_state_id=EPISODE_ID,
        update=changed_updates[0],
    )

    assert first == second
    assert len(first) == 32
    assert changed != first


def test_new_event_requires_actor_action_and_object() -> None:
    """Keep model-owned event identity structurally concrete."""

    submitted = record_input()
    candidate = new_event_observation(actor='')

    with pytest.raises(
        ConversationProgressContractError,
        match='requires actor, action, and object',
    ):
        validate_event_observation_batch(
            event_observation_batch(new_events=[candidate]),
            record_input=submitted,
            supplied_event_handles=set(event_handle_map(submitted)),
            supplied_source_handles=set(source_handle_map(submitted)),
        )


def test_group_scene_anchor_contract_is_prompt_safe() -> None:
    """Protection metadata remains deterministic and out of the prompt."""

    from kazusa_ai_chatbot.conversation_progress.projection import (
        project_group_scene_prompt,
    )

    context = _build_from_fixture(
        _anchor_fixture(),
        current_global_user_id='user-a',
    )
    rendered = project_group_scene_prompt(context)

    assert any(
        turn.get('anchor_kind') == 'current_user'
        for turn in context['turns']
    )
    assert 'anchor_kind' not in rendered
    assert 'global_user_id' not in rendered
    assert 'user-a' not in rendered
