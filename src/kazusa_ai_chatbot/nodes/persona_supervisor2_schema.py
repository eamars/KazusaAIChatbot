from typing import Annotated, Literal, NotRequired, TypedDict

from kazusa_ai_chatbot.action_spec.models import ActionSpecV1
from kazusa_ai_chatbot.action_spec.results import (
    ActionResultV1,
    EpisodeTraceV1,
    SurfaceOutputV1,
)
from kazusa_ai_chatbot.cognition_episode import CognitiveEpisode
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


class GlobalPersonaState(TypedDict):
    # Character related
    character_profile: CharacterProfileDoc

    # Inputs
    storage_timestamp_utc: str
    local_time_context: LocalTimeContextDoc
    user_input: str
    prompt_message_context: PromptMessageContext
    cognitive_episode: NotRequired[CognitiveEpisode]
    user_multimedia_input: list[MultiMediaDoc]
    platform: str
    platform_channel_id: str
    channel_type: str
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
    conversation_episode_state: NotRequired[ConversationEpisodeStateDoc | None]
    conversation_progress: NotRequired[ConversationProgressPromptDoc]
    promoted_reflection_context: NotRequired[dict]

    # Debug
    debug_modes: DebugModes

    # Response continuation
    should_respond: Annotated[bool | None, keep_false]
    dialog_usage_mode: NotRequired[str]

    # Bridge variables populated by persona graph nodes
    # Decontextualizer output
    decontexualized_input: str
    referents: list[ReferentResolution]

    # RAG output
    rag_result: dict

    # Cognition output
    internal_monologue: str
    action_directives: dict
    action_specs: NotRequired[list[ActionSpecV1]]
    action_results: NotRequired[list[ActionResultV1]]
    surface_outputs: NotRequired[list[SurfaceOutputV1]]
    episode_trace: NotRequired[EpisodeTraceV1]
    memory_lifecycle_context: NotRequired[dict]

    # Cognition output for consolidation
    interaction_subtext: str
    emotional_appraisal: str
    character_intent: str
    logical_stance: str
    judgment_note: str
    social_distance: str
    emotional_intensity: str
    vibe_check: str
    relational_dynamic: str

    # Dialog output
    final_dialog: [str]  # -> Will be used for dialog end point (e.g,. Discord)
    target_addressed_user_ids: NotRequired[list[str]]
    target_broadcast: NotRequired[bool]
    mention_target_user: NotRequired[bool]
    # Other outputs from here

    # Consolidation output
    # global state updater
    mood: str
    global_vibe: str
    reflection_summary: str

    # Relationship recorder
    subjective_appraisals: [str]
    affinity_delta: int
    last_relationship_insight: str

    # Facts harvester
    new_facts: [str]
    future_promises: [str]


class CognitionState(TypedDict):
    character_profile: CharacterProfileDoc

    storage_timestamp_utc: str
    local_time_context: LocalTimeContextDoc
    user_input: str
    prompt_message_context: PromptMessageContext
    cognitive_episode: NotRequired[CognitiveEpisode]
    platform: str
    platform_channel_id: str
    channel_type: str
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
    interaction_style_context: NotRequired[dict]
    selected_text_surface_intent: NotRequired[str]

    decontexualized_input: str
    referents: list[ReferentResolution]
    rag_result: dict

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

    content_anchors: list[str]

    facial_expression: list[str]
    body_language: list[str]
    gaze_direction: list[str]
    visual_vibe: list[str]

    action_directives: dict
    action_specs: NotRequired[list[ActionSpecV1]]
    action_results: NotRequired[list[ActionResultV1]]
    surface_outputs: NotRequired[list[SurfaceOutputV1]]
    episode_trace: NotRequired[EpisodeTraceV1]
    memory_lifecycle_context: NotRequired[dict]
    target_addressed_user_ids: NotRequired[list[str]]
    target_broadcast: NotRequired[bool]

    should_stop: bool
    reasoning: str
    retry: int
