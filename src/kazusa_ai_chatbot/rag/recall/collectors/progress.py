"""Current-episode progress recall collector."""

from __future__ import annotations

from typing import Any

from kazusa_ai_chatbot.rag.recall.contracts import (
    _candidate,
    _entry_text,
    _safe_parse_datetime,
)
from kazusa_ai_chatbot.utils import text_or_empty

def _progress_is_active(
    progress: object,
    episode_state: object,
    current_timestamp_utc: str,
) -> bool:
    """Decide whether conversation progress can serve active recall."""

    if not isinstance(progress, dict):
        return False
    status = text_or_empty(progress.get("status"))
    continuity = text_or_empty(progress.get("continuity"))
    if status != "active" or continuity == "sharp_transition":
        return False

    expires_at = progress.get("expires_at")
    if not expires_at and isinstance(episode_state, dict):
        expires_at = episode_state.get("expires_at")
    expiry = _safe_parse_datetime(expires_at)
    current = _safe_parse_datetime(current_timestamp_utc)
    if expiry is None or current is None:
        return True
    return_value = expiry > current
    return return_value

def _progress_evidence_time(progress: dict, episode_state: object) -> str:
    """Choose the best timestamp exposed by progress state."""

    for source in (progress, episode_state):
        if not isinstance(source, dict):
            continue
        for field in ("updated_at", "created_at"):
            timestamp = text_or_empty(source.get(field))
            if timestamp:
                return timestamp
    return_value = ""
    return return_value

def _progress_entries(progress: dict) -> list[str]:
    """Extract compact claims from the active progress document."""

    claims: list[str] = []
    for field in (
        "current_thread",
        "current_blocker",
        "user_goal",
        "progression_guidance",
    ):
        claim = text_or_empty(progress.get(field))
        if claim:
            claims.append(claim)

    for field in (
        "open_loops",
        "resolved_threads",
        "user_state_updates",
        "next_affordances",
        "assistant_moves",
    ):
        values = progress.get(field)
        if not isinstance(values, list):
            continue
        for value in values:
            claim = _entry_text(value)
            if claim:
                claims.append(claim)

    return_value = claims[:8]
    return return_value

class ProgressCollector:
    """Collect active-episode claims from already-loaded progress state."""

    def collect(self, context: dict[str, Any]) -> list[dict[str, str]]:
        """Collect progress candidates when the progress document is active."""

        progress = context.get("conversation_progress")
        episode_state = context.get("conversation_episode_state")
        current_timestamp_utc = text_or_empty(
            context.get("current_timestamp_utc")
        )
        if not _progress_is_active(
            progress,
            episode_state,
            current_timestamp_utc,
        ):
            return_value: list[dict[str, str]] = []
            return return_value

        progress_doc = progress if isinstance(progress, dict) else {}
        evidence_time = _progress_evidence_time(progress_doc, episode_state)
        candidates = [
            _candidate(
                source="conversation_progress",
                claim=claim,
                temporal_scope="current_episode",
                lifecycle_status="active",
                evidence_time=evidence_time,
                authority="primary_for_current_episode",
            )
            for claim in _progress_entries(progress_doc)
        ]
        return_value = candidates[:8]
        return return_value
