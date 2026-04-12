from __future__ import annotations

from typing import TypedDict


class DiscordProcessState(TypedDict):
    timestamp: str

    # Input to Relevance Agent 
    user_name: str  # read from discord
    user_id: str  # read from discord
    user_input: str  # Raw message provided by the user. Can input multimedia content
    user_profile: dict  # used to extract affinity score.

    bot_id: int
    bot_name: str
    character_profile: dict
    character_state: dict

    channel_id: str  # read from discord. Empty if it is private message
    channel_name: str  # read from discord. empty if it is private message  (used to determine the context)
    chat_history: list[dict]  # Previous messages in the channel (short listed)

    # Output from Relevance Agent
    should_respond: bool
    reason_to_respond: str
    use_reply_feature: bool
    channel_topic: str
    user_topic: str


    # Input to Persona Supervisor
    # character_state: str  # Already requested
    # character_profile: dict  # Already requested
    # timestamp: str  # Already requested
    # user_input: str  # already requested
    # user_id: str  # Already requested
    # user_name: str  # Already requested
    # user_profile: dict  # Already requested
    # bot_id: int  # Already requested
    # chat_history: list[dict]  # Already requested
    # user_topic: str  # Already provided
    # channel_topic: str  # Already provided
    
    # Output from Persona Supervisor
    final_dialog: list[str]
    future_promises: list[dict]