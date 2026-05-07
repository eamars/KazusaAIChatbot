from __future__ import annotations

from typing import Any
from typing import Annotated
from typing import NotRequired
from typing import TypedDict

from kazusa_ai_chatbot.conversation_progress import ConversationProgressPromptDoc
from kazusa_ai_chatbot.db.schemas import ConversationEpisodeStateDoc
from kazusa_ai_chatbot.message_envelope import MessageEnvelope, PromptMessageContext
from kazusa_ai_chatbot.time_context import TimeContextDoc


class MultiMediaDoc(TypedDict):
    content_type: str  # e.g,. "image/png", "video/mp4"
    base64_data: str
    description: str


class ReplyContext(TypedDict, total=False):
    reply_to_message_id: str
    reply_to_platform_user_id: str
    reply_to_display_name: str
    reply_excerpt: str


class DebugModes(TypedDict, total=False):
    listen_only: bool      # Record data but skip response graph processing
    think_only: bool       # Full pipeline but suppress dialog in response
    no_remember: bool      # Full pipeline but skip consolidation


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


class IMProcessState(TypedDict):
    timestamp: str
    time_context: TimeContextDoc

    # Platform identity
    platform: str                # "discord" | "qq" | "wechat" | etc.
    platform_message_id: str     # Original platform message ID when available
    active_turn_platform_message_ids: NotRequired[list[str]]
    platform_user_id: str        # Original platform user ID (e.g. Discord snowflake)
    global_user_id: str          # Internal UUID4 from user_profiles collection

    # Input to Relevance Agent 
    user_name: str  # display name from the platform
    user_input: str  # Body text plus current attachment descriptions.
    message_envelope: MessageEnvelope
    prompt_message_context: PromptMessageContext
    user_multimedia_input: list[MultiMediaDoc]
    user_profile: dict  # used to extract affinity score.

    platform_bot_id: str  # Bot's ID on the current platform (provided by the adapter)
    character_name: str
    character_profile: dict

    platform_channel_id: str  # Original channel/group/DM ID from the platform
    channel_type: str  # "group" | "private" | "system"
    channel_name: str  # Display name of the channel (used to determine the context)
    chat_history_wide: list[dict]   # Full history slice (CONVERSATION_HISTORY_LIMIT, used by Relevance Agent)
    chat_history_recent: list[dict] # Recent slice (CHAT_HISTORY_RECENT_LIMIT, used by downstream stages)
    reply_context: ReplyContext

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
    conversation_episode_state: NotRequired[ConversationEpisodeStateDoc | None]
    conversation_progress: NotRequired[ConversationProgressPromptDoc]
    promoted_reflection_context: NotRequired[dict]

    # Debug modes (optional, passed from ChatRequest)
    debug_modes: DebugModes

    # Output from Persona Supervisor
    final_dialog: list[str]
    target_addressed_user_ids: NotRequired[list[str]]
    target_broadcast: NotRequired[bool]
    future_promises: list[dict]
    consolidation_state: dict[str, Any]
