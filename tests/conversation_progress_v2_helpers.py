"""Shared exact V2 fixtures for deterministic conversation-progress tests."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone

from kazusa_ai_chatbot.conversation_progress.models import (
    ConversationProgressRecordInput,
    ConversationProgressStateV2,
)
from kazusa_ai_chatbot.conversation_progress import ConversationProgressScope

NOW = '2026-07-28T09:30:00+00:00'
EXPIRES = '2026-07-30T09:30:00+00:00'
PURGE_AFTER = datetime(2026, 7, 30, 9, 30, tzinfo=timezone.utc)
EPISODE_ID = 'episode_progress_v2_test'
SCOPE = ConversationProgressScope(
    platform='qq',
    platform_channel_id='channel_test',
    global_user_id='user_test',
)
SOURCE_REF = {
    'ref_kind': 'conversation_row',
    'ref_id': 'row_source_1',
    'occurred_at': NOW,
}


def event(
    *,
    event_id: str = 'event_test',
    summary: str = 'one source-backed interaction event',
    state: str = 'open',
    retention: str = 'active_scene',
    source_refs: list[dict] | None = None,
) -> dict:
    """Build one exact stored event."""

    return {
        'event_id': event_id,
        'semantic_summary': summary,
        'is_obligation': False,
        'actor': 'current user',
        'action': 'interact with',
        'object': summary,
        'beneficiary': '',
        'precondition': '',
        'state': state,
        'outcome': '',
        'retention': retention,
        'source_refs': deepcopy(
            source_refs if source_refs is not None else [SOURCE_REF]
        ),
        'first_seen_at': NOW,
        'updated_at': NOW,
    }


def packet(
    *,
    turn_count: int = 1,
    events: list[dict] | None = None,
    recent_turn_refs: list[str] | None = None,
    compacted_block_refs: list[str] | None = None,
    status: str = 'active',
) -> ConversationProgressStateV2:
    """Build one exact active packet."""

    return {
        'schema_version': 'conversation_progress.v2',
        'episode_state_id': EPISODE_ID,
        'platform': SCOPE.platform,
        'platform_channel_id': SCOPE.platform_channel_id,
        'global_user_id': SCOPE.global_user_id,
        'status': status,
        'continuity': 'same_episode',
        'turn_count': turn_count,
        'episode_narrative': 'bounded narrative',
        'current_thread': 'current interaction thread',
        'character_stance': 'engaged',
        'user_goal': '',
        'current_blocker': '',
        'emotional_trajectory': 'stable',
        'events': deepcopy(events or []),
        'overused_moves': [],
        'recent_turn_refs': list(recent_turn_refs or []),
        'compacted_block_refs': list(compacted_block_refs or []),
        'created_at': NOW,
        'updated_at': NOW,
        'expires_at': EXPIRES,
        'purge_after': PURGE_AFTER,
    }


def logical_turn(
    *,
    turn_id: str = 'row:row_source_1',
    row_id: str = 'row_source_1',
    trace_id: str = '',
) -> dict:
    """Build one exact logical turn."""

    return {
        'turn_id': turn_id,
        'role': 'user',
        'occurred_at': NOW,
        'display_name': 'Test User',
        'fragments': ['current input'],
        'conversation_row_ids': [row_id],
        'llm_trace_id': trace_id,
        'platform_user_id': 'platform_user_test',
        'global_user_id': SCOPE.global_user_id,
        'addressed_to_global_user_ids': [],
        'broadcast': False,
        'reply_context': {},
    }


def record_input(
    *,
    prior_packet: ConversationProgressStateV2 | None = None,
) -> ConversationProgressRecordInput:
    """Build one exact settled-turn recorder input."""

    return {
        'scope': SCOPE,
        'storage_timestamp_utc': NOW,
        'character_name': 'Test Character',
        'prior_episode_state': deepcopy(prior_packet),
        'decontextualized_input': 'current input',
        'interaction_logical_turns': [logical_turn()],
        'current_turn_source_refs': [deepcopy(SOURCE_REF), {
            'ref_kind': 'llm_trace',
            'ref_id': 'trace_current',
            'occurred_at': NOW,
        }],
        'turn_outcome': 'visible_response',
        'content_plan': {
            'semantic_content': 'continue without resetting',
            'surface_intent': 'respond',
        },
        'logical_stance': 'CONFIRM',
        'character_intent': 'PROVIDE',
        'final_dialog': ['response'],
        'boundary_profile': {
            'self_integrity': 0.8,
            'control_sensitivity': 0.5,
            'compliance_strategy': 'resist',
            'relational_override': 0.4,
            'control_intimacy_misread': 0.1,
            'boundary_recovery': 'rebound',
            'authority_skepticism': 0.5,
        },
    }


def scene_observation(
    *,
    scene_relation: str = 'same',
    episode_change: str = 'none',
) -> dict:
    """Build one exact model-authored scene observation."""

    return {
        'schema_version': 'conversation_progress_scene_observation.v2',
        'scene_relation': scene_relation,
        'episode_change': episode_change,
        'episode_narrative': 'bounded narrative after the current turn',
        'current_thread': 'current interaction thread',
        'character_stance': 'engaged',
        'user_goal': '',
        'current_blocker': '',
        'emotional_trajectory': 'stable',
        'overused_moves': [],
    }


def unchanged_event_observation(
    *,
    event_handle: str = 'e1',
) -> dict:
    """Build one exact unchanged prior-event observation."""

    return {
        'event_handle': event_handle,
        'observation': 'unchanged',
    }


def changed_event_observation(
    *,
    event_handle: str = 'e1',
    summary: str = 'updated source-backed event',
    lifecycle_change: str = 'none',
    relevance: str = 'scene',
    source_turn_handles: list[str] | None = None,
) -> dict:
    """Build one exact changed prior-event observation."""

    return {
        'event_handle': event_handle,
        'observation': 'changed',
        'semantic_summary': summary,
        'outcome': '',
        'lifecycle_change': lifecycle_change,
        'relevance': relevance,
        'source_turn_handles': list(
            source_turn_handles
            if source_turn_handles is not None
            else ['current_input']
        ),
    }


def new_event_observation(
    *,
    summary: str = 'new source-backed event',
    lifecycle_change: str = 'none',
    relevance: str = 'scene',
    source_turn_handles: list[str] | None = None,
    actor: str = 'current user',
    action: str = 'interact with',
    object_: str | None = None,
) -> dict:
    """Build one exact new-event observation."""

    return {
        'semantic_summary': summary,
        'is_obligation': False,
        'actor': actor,
        'action': action,
        'object': object_ if object_ is not None else summary,
        'beneficiary': '',
        'precondition': '',
        'outcome': '',
        'lifecycle_change': lifecycle_change,
        'relevance': relevance,
        'source_turn_handles': list(
            source_turn_handles
            if source_turn_handles is not None
            else ['current_input']
        ),
    }


def event_observation_batch(
    *,
    existing_events: list[dict] | None = None,
    new_events: list[dict] | None = None,
) -> dict:
    """Build one exact model-authored event observation batch."""

    return {
        'schema_version': (
            'conversation_progress_event_observation_batch.v2'
        ),
        'existing_events': deepcopy(existing_events or []),
        'new_events': deepcopy(new_events or []),
    }


def event_update(
    *,
    event_id: str = '',
    summary: str = 'new source-backed event',
    state: str = 'open',
    retention: str = 'active_scene',
    source_refs: list[dict] | None = None,
) -> dict:
    """Build one exact privately mapped event update."""

    return {
        'event_id': event_id,
        'semantic_summary': summary,
        'is_obligation': False,
        'actor': 'current user',
        'action': 'interact with',
        'object': summary,
        'beneficiary': '',
        'precondition': '',
        'state': state,
        'outcome': '',
        'retention': retention,
        'source_refs': deepcopy(
            source_refs if source_refs is not None else [SOURCE_REF]
        ),
    }


def recorder_delta(
    *,
    event_updates: list[dict] | None = None,
) -> dict:
    """Build one exact privately composed recorder delta."""

    return {
        'schema_version': 'conversation_progress_recorder_delta.v2',
        'continuity': 'same_episode',
        'status': 'active',
        'episode_narrative': 'bounded narrative after the current turn',
        'current_thread': 'current interaction thread',
        'character_stance': 'engaged',
        'user_goal': '',
        'current_blocker': '',
        'emotional_trajectory': 'stable',
        'event_updates': deepcopy(event_updates or []),
        'overused_moves': [],
    }
