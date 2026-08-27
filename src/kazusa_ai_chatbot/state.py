from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Any, Literal, NotRequired, TypedDict

from kazusa_ai_chatbot.action_spec.results import EpisodeAttemptDiagnosticV1
from kazusa_ai_chatbot.cognition_episode import CognitiveEpisodeV1
from kazusa_ai_chatbot.conversation_progress import (
    ConversationLogicalTurnV1,
    ConversationProgressLoadDiagnosticsV2,
    ConversationProgressPromptV2,
    ConversationProgressSourceRefV2,
    ConversationProgressStateV2,
)
from kazusa_ai_chatbot.message_envelope import MessageEnvelope, PromptMessageContext
from kazusa_ai_chatbot.time_boundary import LocalTimeContextDoc


class MultiMediaDoc(TypedDict):
    content_type: str  # e.g,. "image/png", "video/mp4"
    base64_data: str
    description: str
    image_observation: NotRequired[dict[str, Any]]


class ReplyAttachmentSummary(TypedDict):
    media_kind: str
    description: str
    summary_status: Literal["available", "unavailable"]


class ReplyContext(TypedDict, total=False):
    reply_to_message_id: str
    reply_to_platform_user_id: str
    reply_to_display_name: str
    reply_excerpt: str
    reply_attachments: list[ReplyAttachmentSummary]


class DebugModes(TypedDict, total=False):
    listen_only: bool      # Record data but skip response graph processing
    think_only: bool       # Full pipeline but suppress dialog in response
    no_remember: bool      # Full pipeline but skip consolidation
    no_visual_directives: bool  # Full pipeline but skip visual LLM directives


def keep_true(current: bool | None, update: bool | None) -> bool:
    """Preserve a true value in monotonic reply-anchor state.

    Args:
        current: Current graph-state value.
        update: Incoming graph-node update.

    Returns:
        True once either side is true; otherwise false.
    """

    if current is True or update is True:
        return_value = True
    else:
        return_value = False
    return return_value


def keep_false(current: bool | None, update: bool | None) -> bool:
    """Preserve a false value in monotonic cognition-continuation state.

    Args:
        current: Current graph-state value.
        update: Incoming graph-node update.

    Returns:
        False once either side is false; otherwise true.
    """

    if current is False or update is False:
        return_value = False
    else:
        return_value = True
    return return_value


MAX_EPISODE_ATTEMPT_DIAGNOSTICS = 16


def append_attempt_diagnostics(
    current: list[EpisodeAttemptDiagnosticV1] | None,
    update: list[EpisodeAttemptDiagnosticV1] | None,
) -> list[EpisodeAttemptDiagnosticV1]:
    """Append diagnostic rows and retain only the newest bounded suffix."""

    current_rows = [] if current is None else current
    update_rows = [] if update is None else update
    if not isinstance(current_rows, list) or not isinstance(update_rows, list):
        raise TypeError("attempt diagnostics updates must be lists")
    combined = [*current_rows, *update_rows]
    normalized: list[EpisodeAttemptDiagnosticV1] = []
    for row in combined:
        if not isinstance(row, Mapping):
            raise TypeError("attempt diagnostic rows must be mappings")
        normalized.append(dict(row))
    return_value = normalized[-MAX_EPISODE_ATTEMPT_DIAGNOSTICS:]
    return return_value


class IMProcessState(TypedDict):
    storage_timestamp_utc: str
    local_time_context: LocalTimeContextDoc
    llm_trace_id: NotRequired[str]

    # Platform identity
    platform: str                # "discord" | "qq" | "wechat" | etc.
    platform_message_id: str     # Original platform message ID when available
    active_turn_platform_message_ids: NotRequired[list[str]]
    active_turn_conversation_row_ids: NotRequired[list[str]]
    active_turn_conversation_source_refs: NotRequired[
        list[ConversationProgressSourceRefV2]
    ]
    platform_user_id: str        # Original platform user ID (e.g. Discord snowflake)
    global_user_id: str          # Internal UUID4 from user_profiles collection

    # Input to Relevance Agent 
    user_name: str  # display name from the platform
    user_input: str  # Body text plus current attachment descriptions.
    message_envelope: MessageEnvelope
    prompt_message_context: PromptMessageContext
    cognitive_episode: NotRequired[CognitiveEpisodeV1]
    user_multimedia_input: list[MultiMediaDoc]
    additional_media_present: NotRequired[bool]
    media_prepared: NotRequired[bool]
    user_profile: dict  # carries the prompt-safe user projection.

    platform_bot_id: str  # Bot's ID on the current platform (provided by the adapter)
    character_name: str
    character_profile: dict
    character_identity_revision_number: NotRequired[int]
    character_identity_context: NotRequired[dict[str, object]]
    character_identity_surface_context: NotRequired[dict[str, object]]
    character_identity_projection_digest: NotRequired[str]
    character_identity_consumer_kinds: NotRequired[list[str]]
    character_identity_episode_id: NotRequired[str]
    character_identity_epistemic_core_included: NotRequired[bool]

    platform_channel_id: str  # Original channel/group/DM ID from the platform
    channel_type: str  # "group" | "private" | "system"
    channel_name: str  # Display name of the channel (used to determine the context)
    chat_history_wide: list[dict]   # Full history slice (CONVERSATION_HISTORY_LIMIT, used by Relevance Agent)
    chat_history_recent: list[dict] # Recent slice (CHAT_HISTORY_RECENT_LIMIT, used by downstream stages)
    reply_context: ReplyContext

    # Relevance turn-settlement state
    response_action: NotRequired[Literal["ignore", "proceed", "wait"]]
    attempt_diagnostics: Annotated[
        list[EpisodeAttemptDiagnosticV1],
        append_attempt_diagnostics,
    ]
    observation_status: NotRequired[
        Literal["more_time_available", "observation_complete"]
    ]
    turn_id: NotRequired[str]
    turn_version: NotRequired[int]
    cognition_claimed: NotRequired[bool]
    assembled_fragments: NotRequired[list[dict[str, Any]]]
    fresh_history: NotRequired[list[dict[str, Any]]]
    media_descriptions: NotRequired[list[dict[str, Any]]]
    scene_context: NotRequired[str]
    relationship_context: NotRequired[str]

    # Output from Relevance Agent
    # Origin contract: service.py seeds this true. LangGraph combines updates
    # through keep_false so any pipeline stage can stop cognition processing,
    # and no later true update can restart it.
    should_respond: Annotated[bool | None, keep_false]
    reason_to_respond: str
    # Origin contract: service.py seeds this false. LangGraph combines updates
    # through keep_true so any pipeline stage can request platform reply
    # anchoring, and no later false update can erase that request.
    use_reply_feature: Annotated[bool, keep_true]
    channel_topic: str
    indirect_speech_context: str  # Only populated for Situation B (user talks about the character to others)
    conversation_episode_state: NotRequired[ConversationProgressStateV2 | None]
    conversation_progress: NotRequired[ConversationProgressPromptV2]
    ambient_logical_turns: NotRequired[list[ConversationLogicalTurnV1]]
    interaction_logical_turns: NotRequired[list[ConversationLogicalTurnV1]]
    conversation_progress_diagnostics: NotRequired[
        ConversationProgressLoadDiagnosticsV2
    ]
    promoted_reflection_context: NotRequired[dict]
    internal_monologue_residue_context: NotRequired[str]
    past_dialog_cognition_context: NotRequired[str]
    action_availability_runtime: NotRequired[dict[str, Any]]
    interaction_style_context: NotRequired[dict[str, Any]]
    settled_relevance_context_consumption: NotRequired[dict[str, Any]]

    # Debug modes (optional, passed from ChatRequest)
    debug_modes: DebugModes

    # Output from Persona Supervisor
    cognition_core_output: NotRequired[dict[str, Any]]
    cognition_state_update: NotRequired[dict[str, Any]]
    cognition_state_committed: NotRequired[bool]
    final_dialog: list[str]
    target_addressed_user_ids: NotRequired[list[str]]
    target_broadcast: NotRequired[bool]
    future_promises: list[dict]
    consolidation_state: dict[str, Any]
