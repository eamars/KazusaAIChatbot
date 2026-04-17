from __future__ import annotations

from typing import TypedDict


class MultiMediaDoc(TypedDict):
    content_type: str  # e.g,. "image/png", "video/mp4"
    base64_data: str
    description: str


class IMProcessState(TypedDict):
    timestamp: str

    # Platform identity
    platform: str                # "discord" | "qq" | "wechat" | etc.
    platform_user_id: str        # Original platform user ID (e.g. Discord snowflake)
    global_user_id: str          # Internal UUID4 from user_profiles collection

    # Input to Relevance Agent 
    user_name: str  # display name from the platform
    user_input: str  # Raw message provided by the user. Can input multimedia content
    user_multimedia_input: list[MultiMediaDoc]
    user_profile: dict  # used to extract affinity score.

    platform_bot_id: str  # Bot's ID on the current platform (provided by the adapter)
    bot_name: str
    character_profile: dict

    platform_channel_id: str  # Original channel/group ID from the platform. Empty for private messages
    channel_name: str  # Display name of the channel (used to determine the context)
    chat_history: list[dict]  # Previous messages in the channel (short listed)

    # Output from Relevance Agent
    should_respond: bool
    reason_to_respond: str
    use_reply_feature: bool
    channel_topic: str
    user_topic: str


    # Input to Persona Supervisor
    # character_profile: dict  # Already requested
    # timestamp: str  # Already requested
    # user_input: str  # already requested
    # platform_user_id: str  # Already requested
    # global_user_id: str  # Already requested
    # user_name: str  # Already requested
    # user_profile: dict  # Already requested
    # platform_bot_id: str  # Already requested
    # chat_history: list[dict]  # Already requested
    # user_topic: str  # Already provided
    # channel_topic: str  # Already provided
    
    # Output from Persona Supervisor
    final_dialog: list[str]
    future_promises: list[dict]