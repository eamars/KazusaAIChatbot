"""Recall candidate and result contracts."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from kazusa_ai_chatbot.time_boundary import parse_storage_utc_datetime
from kazusa_ai_chatbot.utils import text_or_empty

_REQUIRED_CONTEXT_FIELDS = (
    "platform",
    "platform_channel_id",
    "global_user_id",
    "current_timestamp_utc",
)

_VOLATILE_CACHE_STATUS = {
    "enabled": False,
    "hit": False,
    "reason": "volatile_recall",
}

_ACTIVE_MODES = {"active_episode_agreement", "episode_position"}

_VALID_MODES = {
    "active_episode_agreement",
    "durable_commitment",
    "episode_position",
    "exact_agreement_history",
}

_SOURCE_ORDERS = {
    "active_episode_agreement": [
        "conversation_progress",
        "user_memory_units",
        "calendar_runs",
        "conversation_history",
    ],
    "episode_position": [
        "conversation_progress",
        "user_memory_units",
        "calendar_runs",
        "conversation_history",
    ],
    "durable_commitment": [
        "user_memory_units",
        "calendar_runs",
        "conversation_progress",
        "conversation_history",
    ],
    "exact_agreement_history": [
        "conversation_history",
        "user_memory_units",
        "conversation_progress",
        "calendar_runs",
    ],
}

_OUTPUT_SELECTED_SUMMARY_LIMIT = 600

_OUTPUT_FRESHNESS_BASIS_LIMIT = 400

_CANDIDATE_CLAIM_LIMIT = 240

_CONFLICT_LIMIT = 5

_CONFLICT_TEXT_LIMIT = 240

def _cache_status() -> dict[str, bool | str]:
    """Build the no-cache status used by volatile Recall results."""
    status = dict(_VOLATILE_CACHE_STATUS)
    return status

def _clip_text(value: object, *, limit: int) -> str:
    """Return compact text suitable for downstream prompt payloads.

    Args:
        value: Candidate text or source value.
        limit: Maximum character count.

    Returns:
        Stripped and clipped text.
    """

    text = text_or_empty(value)
    if len(text) <= limit:
        return text
    clipped = text[: limit - 1].rstrip()
    return_value = f"{clipped}…"
    return return_value

def _safe_parse_datetime(value: object) -> datetime | None:
    """Parse an optional storage UTC instant without treating bad data as fatal.

    Args:
        value: Timestamp-like value from DB or progress state.

    Returns:
        Parsed datetime, or ``None`` when the value is blank or malformed.
    """

    raw_value = text_or_empty(value)
    if not raw_value:
        return None
    try:
        parsed = parse_storage_utc_datetime(raw_value)
    except ValueError:
        return None
    return parsed

def _entry_text(value: object) -> str:
    """Extract a human-readable claim from progress or history entries.

    Args:
        value: Source entry, usually a string or dict with text fields.

    Returns:
        Best available text for the entry.
    """

    if isinstance(value, dict):
        for field in ("text", "fact", "body_text", "current_thread"):
            text = text_or_empty(value.get(field))
            if text:
                return text
        return_value = ""
        return return_value
    return_value = text_or_empty(value)
    return return_value

def _candidate(
    *,
    source: str,
    claim: object,
    temporal_scope: str,
    lifecycle_status: str,
    evidence_time: object = "",
    authority: str = "supporting",
) -> dict[str, str]:
    """Build one compact Recall candidate record.

    Args:
        source: Logical evidence source label.
        claim: Candidate factual text.
        temporal_scope: Time domain for the claim.
        lifecycle_status: Active/pending/historical lifecycle label.
        evidence_time: Optional ISO timestamp from the source.
        authority: Source authority label.

    Returns:
        Candidate record accepted by the Recall result contract.
    """

    return_value = {
        "source": source,
        "claim": _clip_text(claim, limit=_CANDIDATE_CLAIM_LIMIT),
        "temporal_scope": temporal_scope,
        "lifecycle_status": lifecycle_status,
        "evidence_time": text_or_empty(evidence_time),
        "authority": authority,
    }
    return return_value

def _candidate_key(candidate: dict[str, str]) -> tuple[str, str, str]:
    """Return a stable duplicate key for Recall candidates."""

    return_value = (
        candidate["source"],
        candidate["claim"],
        candidate["evidence_time"],
    )
    return return_value

def _dedupe_candidates(
    candidates: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Remove duplicate Recall candidates while preserving first occurrence."""

    deduped: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for candidate in candidates:
        key = _candidate_key(candidate)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
    return_value = deduped
    return return_value


_CALENDAR_RUN_TRIGGER_LABELS = {
    "future_cognition": "future cognition",
    "commitment_due_cognition": "commitment due cognition",
}


def _calendar_run_claim(run: dict[str, Any]) -> str:
    """Render one pending calendar run as compact semantic evidence."""

    trigger_kind = text_or_empty(run.get("trigger_kind"))
    trigger_label = _CALENDAR_RUN_TRIGGER_LABELS.get(
        trigger_kind,
        "future action",
    )
    due_at = text_or_empty(run.get("due_at"))
    payload = run.get("payload")
    objective = ""
    if isinstance(payload, dict):
        objective = text_or_empty(payload.get("continuation_objective"))

    claim = f"Pending calendar {trigger_label} at {due_at}"
    if objective:
        claim = f"{claim}: {objective}"
    return claim


def _result_payload(
    *,
    selected_summary: object = "",
    recall_type: str = "active_episode_agreement",
    primary_source: str = "",
    supporting_sources: list[str] | None = None,
    freshness_basis: object = "",
    conflicts: list[str] | None = None,
    candidates: list[dict[str, str]] | None = None,
    missing_context: list[str] | None = None,
    observation_candidates: list[dict[str, str]] | None = None,
    source_hints: list[dict[str, str]] | None = None,
    review_uncertainty: object = "",
    error: str | None = None,
) -> dict[str, Any]:
    """Build the nested Recall result payload with fixed size caps."""

    result_payload: dict[str, Any] = {
        "selected_summary": _clip_text(
            selected_summary,
            limit=_OUTPUT_SELECTED_SUMMARY_LIMIT,
        ),
        "recall_type": recall_type,
        "primary_source": primary_source,
        "supporting_sources": list(supporting_sources or []),
        "freshness_basis": _clip_text(
            freshness_basis,
            limit=_OUTPUT_FRESHNESS_BASIS_LIMIT,
        ),
        "conflicts": [
            _clip_text(conflict, limit=_CONFLICT_TEXT_LIMIT)
            for conflict in list(conflicts or [])[:_CONFLICT_LIMIT]
        ],
        "candidates": list(candidates or [])[:29],
        "missing_context": list(missing_context or []),
        "observation_candidates": list(observation_candidates or []),
        "source_hints": list(source_hints or []),
        "review_uncertainty": _clip_text(
            review_uncertainty,
            limit=_OUTPUT_FRESHNESS_BASIS_LIMIT,
        ),
    }
    if error is not None:
        result_payload["error"] = error
    return result_payload

def _agent_result(
    *,
    resolved: bool,
    result_payload: dict[str, Any],
) -> dict[str, Any]:
    """Build the outer helper result with volatile cache metadata."""

    return_value = {
        "resolved": resolved,
        "result": result_payload,
        "attempts": 1,
        "cache": _cache_status(),
    }
    return return_value

def _missing_context_result(missing_fields: list[str]) -> dict[str, Any]:
    """Build the explicit missing-context failure result."""

    missing_text = ", ".join(missing_fields)
    result_payload = _result_payload(
        selected_summary="",
        recall_type="unsupported",
        primary_source="",
        supporting_sources=[],
        freshness_basis=f"missing mandatory recall context: {missing_text}",
        conflicts=[],
        candidates=[],
        error="missing_mandatory_context",
    )
    result = _agent_result(resolved=False, result_payload=result_payload)
    return result

def _mode_from_task(task: str) -> str:
    """Extract the fixed Recall slot mode from the supervisor task text."""

    for mode in _VALID_MODES:
        if mode in task:
            return mode
    return_value = "active_episode_agreement"
    return return_value
