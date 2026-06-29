"""Internal LangGraph state channels for cognition core execution."""

from __future__ import annotations

from typing import Any, NotRequired, TypedDict


class CoreStageState(TypedDict):
    """Loose internal stage state with named channels for parallel graph merges."""

    accepted_user_preferences: list[str]
    action_directives: dict[str, Any]
    action_results: list[dict[str, Any]]
    action_specs: list[dict[str, Any]]
    available_action_affordances: list[dict[str, Any]]
    background_work_output_char_limit: int
    body_language: list[str]
    boundary_core_assessment: dict[str, Any]
    channel_name: str
    channel_topic: str
    channel_type: str
    character_intent: str
    character_name: str
    character_profile: dict[str, Any]
    chat_history_recent: list[dict[str, Any]]
    chat_history_wide: list[dict[str, Any]]
    cognitive_episode: dict[str, Any]
    content_plan: dict[str, str]
    conversation_progress: dict[str, Any]
    decontexualized_input: str
    emotional_appraisal: str
    emotional_intensity: str
    episode_trace: dict[str, Any]
    facial_expression: list[str]
    forbidden_phrases: list[str]
    gaze_direction: list[str]
    global_user_id: str
    group_engagement_action_context: dict[str, Any]
    indirect_speech_context: str
    interaction_style_context: dict[str, Any]
    interaction_subtext: str
    internal_monologue: str
    internal_monologue_residue_context: str
    past_dialog_cognition_context: str
    judgment_note: str
    linguistic_style: str
    local_time_context: dict[str, Any]
    logical_stance: str
    memory_lifecycle_context: dict[str, Any]
    max_action_requests: int
    max_resolver_requests: int
    pending_resolver_resume: dict[str, Any]
    platform: str
    platform_bot_id: str
    platform_channel_id: str
    platform_message_id: str
    platform_user_id: str
    pre_surface_action_results: list[dict[str, Any]]
    promoted_reflection_context: dict[str, Any]
    prompt_message_context: dict[str, Any]
    rag_result: dict[str, Any]
    reason_to_respond: str
    reasoning: str
    referents: list[dict[str, Any]]
    relational_dynamic: str
    reply_context: dict[str, Any]
    resolver_capability_requests: list[dict[str, Any]]
    resolver_context: str
    resolver_cycle_trace: dict[str, Any]
    resolver_goal_progress: dict[str, Any]
    resolver_pending_resolution: dict[str, Any]
    resolver_state: dict[str, Any]
    retry: int
    rhetorical_strategy: str
    selected_text_surface_intent: str
    semantic_action_requests: list[dict[str, Any]]
    should_stop: bool
    social_distance: str
    storage_timestamp_utc: str
    surface_outputs: list[dict[str, Any]]
    target_addressed_user_ids: list[str]
    target_broadcast: bool
    use_reply_feature: bool
    user_input: str
    user_multimedia_input: NotRequired[list[dict[str, Any]]]
    user_name: str
    user_profile: dict[str, Any]
    vibe_check: str
    visual_vibe: list[str]
