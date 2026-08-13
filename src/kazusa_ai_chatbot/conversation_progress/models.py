"""Canonical public contracts for short-term conversation progress."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal, NotRequired, TypedDict


@dataclass(frozen=True)
class ConversationProgressScope:
    """Stable per-user/channel scope for short-term conversation progress."""

    platform: str
    platform_channel_id: str
    global_user_id: str


class ConversationLogicalTurnV1(TypedDict):
    """One complete speaker turn assembled from canonical conversation rows."""

    turn_id: str
    role: Literal['user', 'assistant']
    occurred_at: str
    display_name: str
    fragments: list[str]
    conversation_row_ids: list[str]
    llm_trace_id: str
    platform_user_id: str
    global_user_id: str
    addressed_to_global_user_ids: list[str]
    broadcast: bool
    reply_context: dict[str, object]


class GroupSceneTurnV1(TypedDict):
    """One transient, prompt-safe public turn in a group scene."""

    role: Literal['user', 'assistant']
    speaker_name: str
    text: str
    addressed_names: list[str]
    reply_to_name: str
    scene_position: Literal[
        'before_trigger',
        'trigger',
        'after_trigger',
    ]
    anchor_kind: NotRequired[
        Literal['none', 'current_user', 'explicit_assistant']
    ]


class GroupSceneContextV1(TypedDict):
    """Transient bounded public-scene projection for group Cognition."""

    schema_version: Literal['group_scene_context.v1']
    turns: list[GroupSceneTurnV1]
    visible_participants: list[str]
    omitted_turn_count: int


class GroupSceneProjectionFailure(TypedDict):
    """Typed degraded result for an unfit protected public scene."""

    code: Literal['protected_minimum_unfit', 'trigger_empty']
    protected_anchor_count: int


class ConversationProgressSourceRefV2(TypedDict):
    """Source-lineage reference for one progress event."""

    ref_kind: Literal['conversation_row', 'llm_trace']
    ref_id: str
    occurred_at: str


class ConversationProgressEventV2(TypedDict):
    """One model-authored semantic event with code-owned lifecycle metadata."""

    event_id: str
    semantic_summary: str
    is_obligation: bool
    actor: str
    action: str
    object: str
    beneficiary: str
    precondition: str
    state: Literal[
        'open',
        'in_progress',
        'completed',
        'rejected',
        'superseded',
    ]
    outcome: str
    retention: Literal[
        'decision_critical',
        'active_scene',
        'background',
    ]
    source_refs: list[ConversationProgressSourceRefV2]
    first_seen_at: str
    updated_at: str


class ConversationProgressEventUpdateV2(TypedDict):
    """Privately mapped event snapshot without code-owned timestamps."""

    event_id: str
    semantic_summary: str
    is_obligation: bool
    actor: str
    action: str
    object: str
    beneficiary: str
    precondition: str
    state: Literal[
        'open',
        'in_progress',
        'completed',
        'rejected',
        'superseded',
    ]
    outcome: str
    retention: Literal[
        'decision_critical',
        'active_scene',
        'background',
    ]
    source_refs: list[ConversationProgressSourceRefV2]


class ConversationProgressExistingEventObservationV2(TypedDict):
    """One explicit observation for a supplied prior event handle."""

    event_handle: str
    observation: Literal['unchanged', 'changed']
    semantic_summary: NotRequired[str]
    outcome: NotRequired[str]
    lifecycle_change: NotRequired[Literal[
        'none',
        'began',
        'concluded',
        'declined',
        'replaced',
        'reopened',
    ]]
    relevance: NotRequired[Literal[
        'decision',
        'scene',
        'history',
    ]]
    source_turn_handles: NotRequired[list[str]]


class ConversationProgressNewEventObservationV2(TypedDict):
    """One newly established event with concrete stable identity."""

    semantic_summary: str
    is_obligation: bool
    actor: str
    action: str
    object: str
    beneficiary: str
    precondition: str
    outcome: str
    lifecycle_change: Literal[
        'none',
        'began',
        'concluded',
        'declined',
        'replaced',
    ]
    relevance: Literal['decision', 'scene', 'history']
    source_turn_handles: list[str]


class ConversationProgressEventObservationBatchV2(TypedDict):
    """Exact-coverage event reconciliation from the event specialist."""

    schema_version: Literal[
        'conversation_progress_event_observation_batch.v2'
    ]
    existing_events: list[ConversationProgressExistingEventObservationV2]
    new_events: list[ConversationProgressNewEventObservationV2]


class ConversationProgressSceneObservationV2(TypedDict):
    """Scene-only facts from the independent scene specialist."""

    schema_version: Literal['conversation_progress_scene_observation.v2']
    scene_relation: Literal['same', 'related', 'new']
    episode_change: Literal['none', 'paused', 'finished', 'resumed']
    episode_narrative: str
    current_thread: str
    character_stance: str
    user_goal: str
    current_blocker: str
    emotional_trajectory: str
    overused_moves: list[str]


class ConversationProgressSceneUpdateV2(TypedDict):
    """Validated scene facts after deterministic enum mapping."""

    continuity: Literal[
        'same_episode',
        'related_shift',
        'sharp_transition',
    ]
    status: Literal['active', 'suspended', 'closed']
    episode_narrative: str
    current_thread: str
    character_stance: str
    user_goal: str
    current_blocker: str
    emotional_trajectory: str
    overused_moves: list[str]


class ConversationProgressStateV2(TypedDict):
    """Replacement-written active episode packet."""

    schema_version: Literal['conversation_progress.v2']
    episode_state_id: str
    platform: str
    platform_channel_id: str
    global_user_id: str
    status: Literal['active', 'suspended', 'closed']
    continuity: Literal[
        'same_episode',
        'related_shift',
        'sharp_transition',
    ]
    turn_count: int
    episode_narrative: str
    current_thread: str
    character_stance: str
    user_goal: str
    current_blocker: str
    emotional_trajectory: str
    events: list[ConversationProgressEventV2]
    overused_moves: list[str]
    recent_turn_refs: list[str]
    compacted_block_refs: list[str]
    created_at: str
    updated_at: str
    expires_at: str
    purge_after: datetime


class ConversationCompactionPlanV2(TypedDict):
    """Code-owned structural selection for one immutable block."""

    archive_event_ids: list[str]
    covered_turn_refs: list[str]
    source_block_ids: list[str]


class ConversationProgressRecorderDeltaV2(TypedDict):
    """Private validated delta applied by exact event ID."""

    schema_version: Literal['conversation_progress_recorder_delta.v2']
    continuity: Literal[
        'same_episode',
        'related_shift',
        'sharp_transition',
    ]
    status: Literal['active', 'suspended', 'closed']
    episode_narrative: str
    current_thread: str
    character_stance: str
    user_goal: str
    current_blocker: str
    emotional_trajectory: str
    event_updates: list[ConversationProgressEventUpdateV2]
    overused_moves: list[str]


class ConversationEpisodeBlockV1(TypedDict):
    """Immutable compacted episode block with bounded mutable expiry fields."""

    schema_version: Literal['conversation_progress_block.v1']
    block_id: str
    episode_state_id: str
    platform: str
    platform_channel_id: str
    global_user_id: str
    level: int
    source_turn_count: int
    covered_turn_refs: list[str]
    source_block_ids: list[str]
    narrative: str
    events: list[ConversationProgressEventV2]
    semantic_keys: list[str]
    source_started_at: str
    source_ended_at: str
    content_hash: str
    superseded_by_block_id: str
    embedding: list[float]
    created_at: str
    expires_at: str
    purge_after: datetime


class ConversationProgressPromptV2(TypedDict):
    """Bounded prompt-facing progress projection."""

    schema_version: Literal['conversation_progress_prompt.v2']
    episode_state_id: str
    status: str
    continuity: str
    turn_count: int
    current_thread: str
    character_stance: str
    user_goal: str
    current_blocker: str
    emotional_trajectory: str
    episode_narrative: str
    events: list[ConversationProgressEventV2]
    overused_moves: list[str]
    interaction_logical_turns: list[ConversationLogicalTurnV1]
    compacted_block_refs: list[str]


class ConversationProgressLoadDiagnosticsV2(TypedDict):
    """Text-free diagnostic counters for load and record paths."""

    schema_version: Literal['conversation_progress_diagnostics.v2']
    ambient_rows_scanned: int
    interaction_rows_scanned: int
    ambient_turns_selected: int
    interaction_turns_selected: int
    incomplete_or_malformed_turn_count: int
    packet_turn_count: int
    active_event_count: int
    decision_critical_event_count: int
    block_ref_count: int
    scene_chars: int
    evidence_chars: int
    compaction_requested: bool
    compaction_level: int
    recorder_call_count: int
    event_attempt_count: int
    scene_attempt_count: int
    event_disposition: str
    scene_disposition: str
    write_disposition: str
    protected_anchor_count: int
    packet_age: str
    source_age: str
    cache_disposition: str
    barrier_disposition: str
    reconciliation_status: str


class ConversationProgressLoadResult(TypedDict):
    """Result returned by the canonical load facade."""

    episode_state: ConversationProgressStateV2 | None
    conversation_progress: ConversationProgressPromptV2
    ambient_logical_turns: list[ConversationLogicalTurnV1]
    interaction_logical_turns: list[ConversationLogicalTurnV1]
    diagnostics: ConversationProgressLoadDiagnosticsV2
    source: Literal['db', 'cache', 'empty']


class ConversationProgressRecordInput(TypedDict):
    """Settled turn input for one post-turn recorder call."""

    scope: ConversationProgressScope
    storage_timestamp_utc: str
    character_name: str
    prior_episode_state: ConversationProgressStateV2 | None
    decontextualized_input: str
    interaction_logical_turns: list[ConversationLogicalTurnV1]
    current_turn_source_refs: list[ConversationProgressSourceRefV2]
    turn_outcome: Literal['visible_response', 'cognition_silence']
    content_plan: dict[str, str]
    logical_stance: str
    character_intent: str
    final_dialog: list[str]
    boundary_profile: dict[str, object]


class ConversationProgressRecordResult(TypedDict):
    """Post-turn persistence outcome consumed by telemetry."""

    written: bool
    turn_count: int
    continuity: str
    status: str
    cache_updated: bool
    diagnostics: ConversationProgressLoadDiagnosticsV2
    reconciliation_status: str


class ConversationProgressBlockSearchResultV1(TypedDict):
    """Prompt-safe result from scoped active-block semantic search."""

    source_kind: Literal['conversation_progress_block']
    block_id: str
    narrative: str
    events: list[ConversationProgressEventV2]
    source_started_at: str
    source_ended_at: str
    covered_turn_refs: list[str]
    score: float
