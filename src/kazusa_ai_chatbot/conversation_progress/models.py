"""Public type contracts for the conversation-progress module."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypedDict

from kazusa_ai_chatbot.db.schemas import BoundaryProfileDoc, ConversationEpisodeStateDoc


@dataclass(frozen=True)
class ConversationProgressScope:
    """Stable per-user/channel scope for short-term conversation progress.

    Args:
        platform: Runtime platform key, such as ``"qq"`` or ``"discord"``.
        platform_channel_id: Runtime channel/group/private-chat id.
        global_user_id: Internal user UUID for the current speaker.
    """

    platform: str
    platform_channel_id: str
    global_user_id: str


class ConversationProgressEntry(TypedDict):
    """Prompt-facing entry with relative age."""

    text: str
    age_hint: str


class ConversationProgressPromptDoc(TypedDict):
    """Compact prompt-facing progress projection."""

    status: str
    episode_label: str
    continuity: str
    turn_count: int
    conversation_mode: str
    episode_phase: str
    topic_momentum: str
    current_thread: str
    user_goal: str
    current_blocker: str
    user_state_updates: list[ConversationProgressEntry]
    assistant_moves: list[str]
    overused_moves: list[str]
    open_loops: list[ConversationProgressEntry]
    resolved_threads: list[ConversationProgressEntry]
    avoid_reopening: list[ConversationProgressEntry]
    emotional_trajectory: str
    next_affordances: list[str]
    progression_guidance: str


class ConversationProgressLoadResult(TypedDict):
    """Result returned by ``load_progress_context``."""

    episode_state: ConversationEpisodeStateDoc | None
    conversation_progress: ConversationProgressPromptDoc
    source: Literal["db", "cache", "empty"]


class ConversationProgressRecordInput(TypedDict):
    """Input required to record one completed responsive turn."""

    scope: ConversationProgressScope
    timestamp: str
    prior_episode_state: ConversationEpisodeStateDoc | None
    decontexualized_input: str
    chat_history_recent: list[dict]
    content_anchors: list[str]
    logical_stance: str
    character_intent: str
    final_dialog: list[str]
    boundary_profile: BoundaryProfileDoc


class ConversationProgressRecordResult(TypedDict):
    """Record result consumed by background telemetry."""

    written: bool
    turn_count: int
    continuity: str
    status: str
    cache_updated: bool
