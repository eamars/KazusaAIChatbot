from __future__ import annotations

from typing import Any
from typing import TypedDict


class MultiMediaDoc(TypedDict):
    content_type: str  # e.g,. "image/png", "video/mp4"
    base64_data: str
    description: str


class ReplyContext(TypedDict, total=False):
    reply_to_message_id: str
    reply_to_platform_user_id: str
    reply_to_display_name: str
    reply_to_current_bot: bool
    reply_excerpt: str


class DebugModes(TypedDict, total=False):
    listen_only: bool      # Record data but skip thinking (no LLM calls beyond relevance)
    think_only: bool       # Full pipeline but suppress dialog in response
    no_remember: bool      # Full pipeline but skip consolidation (stage 4)


class IMProcessState(TypedDict):
    timestamp: str

    # Platform identity
    platform: str                # "discord" | "qq" | "wechat" | etc.
    platform_message_id: str     # Original platform message ID when available
    platform_user_id: str        # Original platform user ID (e.g. Discord snowflake)
    global_user_id: str          # Internal UUID4 from user_profiles collection

    # Input to Relevance Agent 
    user_name: str  # display name from the platform
    user_input: str  # Raw message provided by the user. Can input multimedia content
    user_multimedia_input: list[MultiMediaDoc]
    user_profile: dict  # used to extract affinity score.

    platform_bot_id: str  # Bot's ID on the current platform (provided by the adapter)
    mentioned_bot: bool  # Whether the platform structurally mentioned the bot in this message
    bot_name: str
    character_profile: dict

    platform_channel_id: str  # Original channel/group/DM ID from the platform
    channel_type: str  # "group" | "private" | "system"
    channel_name: str  # Display name of the channel (used to determine the context)
    chat_history_wide: list[dict]   # Full history slice (CONVERSATION_HISTORY_LIMIT, used by Relevance Agent)
    chat_history_recent: list[dict] # Recent slice (CHAT_HISTORY_RECENT_LIMIT, used by downstream stages)
    reply_context: ReplyContext

    # Output from Relevance Agent
    should_respond: bool
    reason_to_respond: str
    use_reply_feature: bool
    channel_topic: str
    indirect_speech_context: str  # Only populated for Situation B (user talks about the character to others)

    # Debug modes (optional, passed from ChatRequest)
    debug_modes: DebugModes

    # Output from Persona Supervisor
    final_dialog: list[str]
    future_promises: list[dict]
    consolidation_state: dict[str, Any]
