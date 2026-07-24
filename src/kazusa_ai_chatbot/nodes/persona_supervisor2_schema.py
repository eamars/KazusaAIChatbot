from typing import Annotated, Literal, NotRequired, TypedDict

from kazusa_ai_chatbot.action_spec.models import ActionSpecV1
from kazusa_ai_chatbot.action_spec.results import (
    ActionResultV1,
    EpisodeTraceV2,
    SurfaceOutputV1,
)
from kazusa_ai_chatbot.cognition_episode import CognitiveEpisodeV1
from kazusa_ai_chatbot.cognition_core_v2.contracts import (
    GoalResolutionV2,
    TextSurfaceOutputV2,
    VisualSurfaceOutputV2,
)
from kazusa_ai_chatbot.cognition_resolver.contracts import (
    ResolverCapabilityRequestV1,
    ResolverCycleStateV1,
    ResolverCycleTraceV1,
    ResolverGoalProgressV1,
    ResolverPendingResolutionV1,
    ResolverPendingResumeV1,
)
from kazusa_ai_chatbot.conversation_progress import ConversationProgressPromptDoc
from kazusa_ai_chatbot.state import (
    DebugModes,
    MultiMediaDoc,
    ReplyContext,
    keep_false,
)
from kazusa_ai_chatbot.db import CharacterProfileDoc, ConversationEpisodeStateDoc, UserProfileDoc
from kazusa_ai_chatbot.message_envelope import PromptMessageContext
from kazusa_ai_chatbot.time_boundary import LocalTimeContextDoc

ReferentRole = Literal["subject", "object", "time"]


class ReferentResolution(TypedDict, total=False):
    """Structured decontextualizer reference-resolution result."""

    phrase: str
    referent_role: ReferentRole
    status: Literal["resolved", "unresolved"]


class ScopeUser(TypedDict):
    """Prompt-facing identity row available in the current turn scope."""

    display_name: str
    platform_user_id: str
    global_user_id: str
    aliases: list[str]


class GlobalPersonaState(TypedDict):
    # Character related
    character_profile: CharacterProfileDoc

    # Inputs
    storage_timestamp_utc: str
    local_time_context: LocalTimeContextDoc
    llm_trace_id: NotRequired[str]
    user_input: str
    prompt_message_context: PromptMessageContext
    cognitive_episode: NotRequired[CognitiveEpisodeV1]
    user_multimedia_input: list[MultiMediaDoc]
    platform: str
    platform_channel_id: str
    channel_type: str
    channel_name: str
    platform_message_id: str
    active_turn_platform_message_ids: NotRequired[list[str]]
    active_turn_conversation_row_ids: NotRequired[list[str]]
    platform_user_id: str
    global_user_id: str
    user_name: str
    user_profile: UserProfileDoc
    platform_bot_id: str
    chat_history_wide: list[dict]
    chat_history_recent: list[dict]
    reply_context: ReplyContext
    indirect_speech_context: str
    channel_topic: str
    scope_users: NotRequired[list[ScopeUser]]
    conversation_episode_state: NotRequired[ConversationEpisodeStateDoc | None]
    conversation_progress: NotRequired[ConversationProgressPromptDoc]
    promoted_reflection_context: NotRequired[dict]
    internal_monologue_residue_context: NotRequired[str]
    past_dialog_cognition_context: NotRequired[str]
    action_availability_runtime: NotRequired[dict[str, object]]

    # Debug
    debug_modes: DebugModes

    # Response continuation
    should_respond: Annotated[bool | None, keep_false]
    dialog_usage_mode: NotRequired[str]

    # Bridge variables populated by persona graph nodes
    # Decontextualizer output
    decontextualized_input: str
    referents: list[ReferentResolution]

    # RAG output
    rag_result: dict

    # Cognition resolver output and recurrence context
    resolver_observations: NotRequired[list[dict]]
    cognition_resolver_requests: NotRequired[list[dict]]
    cognition_resolver_progress: NotRequired[dict]
    cognition_resolver_diagnostics: NotRequired[dict]

    # Cognition output
    internal_monologue: str
    goal_resolution: NotRequired[GoalResolutionV2]
    cognition_core_output: NotRequired[dict]
    cognition_state_update: NotRequired[dict]
    cognition_state_committed: NotRequired[bool]
    text_surface_output_v2: NotRequired[TextSurfaceOutputV2]
    visual_surface_output_v2: NotRequired[VisualSurfaceOutputV2]
    action_specs: NotRequired[list[ActionSpecV1]]
    pre_surface_action_results: NotRequired[list[ActionResultV1]]
    action_results: NotRequired[list[ActionResultV1]]
    surface_outputs: NotRequired[list[SurfaceOutputV1]]
    episode_trace: NotRequired[EpisodeTraceV2]
    memory_lifecycle_context: NotRequired[dict]

    # Semantic cognition projections for downstream consolidation and audit.
    interaction_subtext: NotRequired[str]
    emotional_appraisal: NotRequired[str]
    character_intent: NotRequired[str]
    logical_stance: NotRequired[str]
    judgment_note: NotRequired[str]
    social_distance: NotRequired[str]
    emotional_intensity: NotRequired[str]
    vibe_check: NotRequired[str]
    relational_dynamic: NotRequired[str]

    # Dialog output
    final_dialog: [str]  # -> Will be used for dialog end point (e.g,. Discord)
    target_addressed_user_ids: NotRequired[list[str]]
    target_broadcast: NotRequired[bool]
    # Other outputs from here

    # Consolidation memory rows
    new_facts: NotRequired[list[str]]
    future_promises: NotRequired[list[str]]


class CognitionState(TypedDict):
    character_profile: CharacterProfileDoc

    storage_timestamp_utc: str
    local_time_context: LocalTimeContextDoc
    llm_trace_id: NotRequired[str]
    user_input: str
    prompt_message_context: PromptMessageContext
    cognitive_episode: NotRequired[CognitiveEpisodeV1]
    platform: str
    platform_channel_id: str
    channel_type: str
    channel_name: str
    global_user_id: str
    user_name: str
    user_profile: UserProfileDoc
    platform_bot_id: str
    chat_history_recent: list[dict]
    reply_context: ReplyContext
    indirect_speech_context: str
    channel_topic: str
    conversation_progress: NotRequired[ConversationProgressPromptDoc]
    promoted_reflection_context: NotRequired[dict]
    internal_monologue_residue_context: NotRequired[str]
    past_dialog_cognition_context: NotRequired[str]
    action_availability_runtime: NotRequired[dict[str, object]]
    interaction_style_context: NotRequired[dict]
    group_engagement_action_context: NotRequired[dict]
    action_selection_context: NotRequired[dict]
    coding_run_followup: NotRequired[dict]
    selected_text_surface_intent: NotRequired[str]

    decontextualized_input: str
    referents: list[ReferentResolution]
    rag_result: dict

    resolver_state: NotRequired[ResolverCycleStateV1]
    resolver_context: NotRequired[str]
    resolver_capability_requests: NotRequired[list[ResolverCapabilityRequestV1]]
    resolver_cycle_trace: NotRequired[ResolverCycleTraceV1]
    resolver_goal_progress: NotRequired[ResolverGoalProgressV1]
    pending_resolver_resume: NotRequired[ResolverPendingResumeV1]
    resolver_pending_resolution: NotRequired[ResolverPendingResolutionV1]

    emotional_appraisal: str
    interaction_subtext: str

    internal_monologue: str
    character_intent: str
    logical_stance: str
    judgment_note: str

    boundary_core_assessment: dict

    social_distance: str
    emotional_intensity: str
    vibe_check: str
    relational_dynamic: str

    rhetorical_strategy: str
    linguistic_style: str
    accepted_user_preferences: list[str]
    forbidden_phrases: list[str]

    content_plan: dict[str, str]

    facial_expression: list[str]
    body_language: list[str]
    gaze_direction: list[str]
    visual_vibe: list[str]

    action_directives: dict
    action_specs: NotRequired[list[ActionSpecV1]]
    pre_surface_action_results: NotRequired[list[ActionResultV1]]
    action_results: NotRequired[list[ActionResultV1]]
    surface_outputs: NotRequired[list[SurfaceOutputV1]]
    episode_trace: NotRequired[EpisodeTraceV2]
    memory_lifecycle_context: NotRequired[dict]
    target_addressed_user_ids: NotRequired[list[str]]
    target_broadcast: NotRequired[bool]

    should_stop: bool
    reasoning: str
    retry: int
