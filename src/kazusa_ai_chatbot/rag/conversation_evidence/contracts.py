"""Conversation evidence contracts and result payload helpers."""

from __future__ import annotations

from typing import Any, TypedDict

from kazusa_ai_chatbot.rag.evidence_coverage import (
    EvidenceCoverage,
    assess_evidence_coverage,
)
from kazusa_ai_chatbot.utils import text_or_empty

_CAPABILITY_NAME = "conversation_evidence"

_AGENT_NAME = "conversation_evidence_agent"

_UNCACHED_REASON = "capability_orchestrator_uncached"

class _ConversationProjection(TypedDict):
    """Canonical evidence shape exposed by the conversation capability."""

    summaries: list[str]
    rows: list[dict[str, Any]]
    packets: list[dict[str, Any]]
    resolved_refs: list[dict[str, Any]]

class _ActiveTurnExclusionCounts(TypedDict):
    """Counts of active-turn rows removed by deterministic identity type."""

    conversation_row_id: int
    platform_message_id: int

def _clip_text(value: object, *, limit: int = 1000) -> str:
    """Return compact prompt-facing text.

    Args:
        value: Source value to convert to text.
        limit: Maximum number of characters.

    Returns:
        Stripped and clipped text.
    """

    text = text_or_empty(value)
    if len(text) <= limit:
        return text
    clipped = text[: limit - 1].rstrip()
    return_value = f"{clipped}…"
    return return_value

def _cache_status() -> dict[str, Any]:
    """Build no-cache metadata for top-level capability orchestration."""
    status = {
        "enabled": False,
        "hit": False,
        "cache_name": "",
        "reason": _UNCACHED_REASON,
    }
    return status

def _result_payload(
    *,
    selected_summary: object = "",
    primary_worker: str = "",
    supporting_workers: list[str] | None = None,
    source_policy: object = "",
    resolved_refs: list[dict[str, Any]] | None = None,
    projection_payload: dict[str, Any] | None = None,
    worker_payloads: dict[str, Any] | None = None,
    evidence: list[str] | None = None,
    missing_context: list[str] | None = None,
    conflicts: list[str] | None = None,
    observation_candidates: list[dict[str, Any]] | None = None,
    source_hints: list[dict[str, Any]] | None = None,
    coverage: EvidenceCoverage | None = None,
    confirmed_evidence: list[str] | None = None,
    partial_evidence: list[str] | None = None,
    nearby_evidence: list[str] | None = None,
) -> dict[str, Any]:
    """Build the standard top-level conversation capability payload."""

    coverage_payload = coverage or assess_evidence_coverage(
        task="",
        evidence_items=[],
        worker_resolved=False,
    )
    payload = {
        "selected_summary": _clip_text(selected_summary),
        "capability": _CAPABILITY_NAME,
        "primary_worker": primary_worker,
        "supporting_workers": list(supporting_workers or []),
        "source_policy": _clip_text(source_policy, limit=400),
        "resolved_refs": list(resolved_refs or []),
        "projection_payload": dict(projection_payload or {}),
        "worker_payloads": dict(worker_payloads or {}),
        "evidence": list(evidence or []),
        "missing_context": list(missing_context or []),
        "conflicts": list(conflicts or []),
        "observation_candidates": list(observation_candidates or []),
        "source_hints": list(source_hints or []),
        "coverage": coverage_payload,
        "confirmed_evidence": list(confirmed_evidence or []),
        "partial_evidence": list(partial_evidence or []),
        "nearby_evidence": list(nearby_evidence or []),
    }
    return payload

def _agent_result(*, resolved: bool, payload: dict[str, Any]) -> dict[str, Any]:
    """Build the outer helper result for one capability-agent execution."""
    result = {
        "resolved": resolved,
        "result": payload,
        "attempts": 1,
        "cache": _cache_status(),
    }
    return result
