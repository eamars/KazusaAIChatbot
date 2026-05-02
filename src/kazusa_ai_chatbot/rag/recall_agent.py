"""Deterministic RAG2 helper for active agreement and commitment recall."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from kazusa_ai_chatbot.db import (
    UserMemoryUnitStatus,
    UserMemoryUnitType,
    get_conversation_history,
    query_pending_scheduled_events,
    query_user_memory_units,
)
from kazusa_ai_chatbot.dispatcher.task import parse_iso_datetime
from kazusa_ai_chatbot.rag.helper_agent import BaseRAGHelperAgent
from kazusa_ai_chatbot.utils import text_or_empty

_REQUIRED_CONTEXT_FIELDS = (
    "platform",
    "platform_channel_id",
    "global_user_id",
    "current_timestamp",
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
        "scheduled_events",
        "conversation_history",
    ],
    "episode_position": [
        "conversation_progress",
        "user_memory_units",
        "scheduled_events",
        "conversation_history",
    ],
    "durable_commitment": [
        "user_memory_units",
        "scheduled_events",
        "conversation_progress",
        "conversation_history",
    ],
    "exact_agreement_history": [
        "conversation_history",
        "user_memory_units",
        "conversation_progress",
        "scheduled_events",
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
    """Parse an optional external timestamp without treating bad data as fatal.

    Args:
        value: Timestamp-like value from DB or progress state.

    Returns:
        Parsed datetime, or ``None`` when the value is blank or malformed.
    """

    raw_value = text_or_empty(value)
    if not raw_value:
        return None
    try:
        parsed = parse_iso_datetime(raw_value)
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


def _result_payload(
    *,
    selected_summary: object = "",
    recall_type: str = "active_episode_agreement",
    primary_source: str = "",
    supporting_sources: list[str] | None = None,
    freshness_basis: object = "",
    conflicts: list[str] | None = None,
    candidates: list[dict[str, str]] | None = None,
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


def _progress_is_active(
    progress: object,
    episode_state: object,
    current_timestamp: str,
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
    current = _safe_parse_datetime(current_timestamp)
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


def _event_claim(event: dict[str, Any]) -> str:
    """Render one pending scheduled event as compact factual evidence."""

    tool = text_or_empty(event.get("tool"))
    execute_at = text_or_empty(event.get("execute_at"))
    args = event.get("args")
    text = ""
    if tool == "send_message" and isinstance(args, dict):
        text = text_or_empty(args.get("text"))
        if not text:
            text = text_or_empty(args.get("message"))

    if text:
        claim = f"Pending scheduled event {tool} at {execute_at}: {text}"
        return claim

    claim = f"Pending scheduled event {tool} at {execute_at}"
    return claim


def _has_source_conflict(candidates: list[dict[str, str]]) -> bool:
    """Detect source-level disagreement before using transcript proof."""

    source_claims: dict[str, str] = {}
    for candidate in candidates:
        source = candidate["source"]
        if source == "conversation_history" or source in source_claims:
            continue
        source_claims[source] = candidate["claim"].casefold()

    if len(source_claims) <= 1:
        return False

    distinct_claims = set(source_claims.values())
    return_value = len(distinct_claims) > 1
    return return_value


def _conflict_notes(candidates: list[dict[str, str]]) -> list[str]:
    """Create compact conflict notes from candidate source disagreement."""

    if not _has_source_conflict(candidates):
        return_value: list[str] = []
        return return_value

    notes: list[str] = []
    seen_sources: set[str] = set()
    for candidate in candidates:
        source = candidate["source"]
        if source in seen_sources or source == "conversation_history":
            continue
        seen_sources.add(source)
        notes.append(f"{source}: {candidate['claim']}")
        if len(notes) >= _CONFLICT_LIMIT:
            break
    return notes


def _rank_candidates(mode: str, candidates: list[dict[str, str]]) -> list[dict[str, str]]:
    """Sort candidates by the fixed source authority order for a Recall mode."""

    source_order = _SOURCE_ORDERS[mode]
    order_index = {
        source: index
        for index, source in enumerate(source_order)
    }
    ranked_candidates = sorted(
        candidates,
        key=lambda candidate: (
            order_index.get(candidate["source"], len(source_order)),
            0 if candidate["lifecycle_status"] in {"active", "pending"} else 1,
            candidate["evidence_time"],
        ),
    )
    return ranked_candidates


def _recall_type_for(mode: str, selected: dict[str, str], conflicts: list[str]) -> str:
    """Map slot mode and selected source to the public Recall result type."""

    if mode == "exact_agreement_history":
        return "exact_history"
    if selected["source"] == "user_memory_units":
        return "durable_commitment"
    if conflicts:
        return "mixed"
    return_value = "active_episode_agreement"
    return return_value


def _freshness_basis(selected: dict[str, str], mode: str, progress_unavailable: bool) -> str:
    """Explain why the selected evidence source is authoritative."""

    if progress_unavailable and selected["source"] == "user_memory_units":
        return_value = (
            "Active-episode state was unavailable, so active commitment memory "
            "is the best durable ongoing source."
        )
        return return_value
    evidence_time = selected["evidence_time"]
    if evidence_time:
        return_value = (
            f"Selected {selected['source']} for {mode}; evidence_time={evidence_time}."
        )
        return return_value
    return_value = f"Selected {selected['source']} for {mode}."
    return return_value


class ProgressCollector:
    """Collect active-episode claims from already-loaded progress state."""

    def collect(self, context: dict[str, Any]) -> list[dict[str, str]]:
        """Collect progress candidates when the progress document is active."""

        progress = context.get("conversation_progress")
        episode_state = context.get("conversation_episode_state")
        current_timestamp = text_or_empty(context.get("current_timestamp"))
        if not _progress_is_active(progress, episode_state, current_timestamp):
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


class ActiveCommitmentCollector:
    """Collect active durable commitments for the current user."""

    async def collect(self, global_user_id: str) -> list[dict[str, str]]:
        """Read active commitment memory units and convert them to candidates."""

        units = await query_user_memory_units(
            global_user_id,
            unit_types=[UserMemoryUnitType.ACTIVE_COMMITMENT],
            statuses=[UserMemoryUnitStatus.ACTIVE],
            limit=6,
        )
        candidates: list[dict[str, str]] = []
        for unit in units:
            if unit.get("status") != UserMemoryUnitStatus.ACTIVE:
                continue
            claim = text_or_empty(unit.get("fact"))
            if not claim:
                continue
            evidence_time = (
                text_or_empty(unit.get("updated_at"))
                or text_or_empty(unit.get("last_seen_at"))
            )
            candidates.append(
                _candidate(
                    source="user_memory_units",
                    claim=claim,
                    temporal_scope="durable_ongoing",
                    lifecycle_status="active",
                    evidence_time=evidence_time,
                    authority="primary_for_durable_commitment",
                )
            )
        return candidates[:6]


class ScheduledEventCollector:
    """Collect pending executable future actions for the current user."""

    async def collect(self, context: dict[str, Any]) -> list[dict[str, str]]:
        """Read pending scheduled events and convert them to candidates."""

        events = await query_pending_scheduled_events(
            platform=context["platform"],
            platform_channel_id=context["platform_channel_id"],
            global_user_id=context["global_user_id"],
            current_timestamp=context["current_timestamp"],
            limit=10,
        )
        candidates: list[dict[str, str]] = []
        for event in events:
            candidates.append(
                _candidate(
                    source="scheduled_events",
                    claim=_event_claim(event),
                    temporal_scope="pending_future_action",
                    lifecycle_status="pending",
                    evidence_time=event.get("execute_at", ""),
                    authority="supporting",
                )
            )
        return candidates[:10]


class HistoryEvidenceCollector:
    """Collect bounded transcript proof for exact history or conflicts."""

    async def collect(self, context: dict[str, Any]) -> list[dict[str, str]]:
        """Read recent conversation history and convert rows to candidates."""

        rows = await get_conversation_history(
            platform=context["platform"],
            platform_channel_id=context["platform_channel_id"],
            global_user_id=context["global_user_id"],
            limit=20,
        )
        candidates: list[dict[str, str]] = []
        for row in rows[:5]:
            claim = _entry_text(row)
            if not claim:
                continue
            candidates.append(
                _candidate(
                    source="conversation_history",
                    claim=claim,
                    temporal_scope="historical_proof",
                    lifecycle_status="historical",
                    evidence_time=row.get("timestamp", ""),
                    authority="primary_for_exact_wording",
                )
            )
        return candidates


class RecallAgent(BaseRAGHelperAgent):
    """RAG helper agent for active agreements and ongoing commitments."""

    def __init__(self, *, cache_runtime=None) -> None:
        """Create the no-cache Recall helper.

        Args:
            cache_runtime: Accepted for interface compatibility; unused because
                Recall results depend on volatile progress and pending events.
        """

        super().__init__(
            name="recall_agent",
            cache_name="",
            cache_runtime=cache_runtime,
        )
        self.progress_collector = ProgressCollector()
        self.active_commitment_collector = ActiveCommitmentCollector()
        self.scheduled_event_collector = ScheduledEventCollector()
        self.history_evidence_collector = HistoryEvidenceCollector()

    async def run(
        self,
        task: str,
        context: dict[str, Any],
        max_attempts: int = 1,
    ) -> dict[str, Any]:
        """Recall one active agreement, commitment, or exact agreement history.

        Args:
            task: ``Recall:`` slot text emitted by the RAG2 initializer.
            context: Scoped RAG2 context containing platform, user, and time.
            max_attempts: Accepted for interface compatibility; Recall runs once.

        Returns:
            Standard RAG helper result with ``resolved``, ``result``,
            ``attempts``, and volatile no-cache metadata.
        """

        del max_attempts

        missing_fields = [
            field
            for field in _REQUIRED_CONTEXT_FIELDS
            if not text_or_empty(context.get(field))
        ]
        if missing_fields:
            result = _missing_context_result(missing_fields)
            return result

        mode = _mode_from_task(task)
        progress_candidates = self.progress_collector.collect(context)
        commitment_candidates = await self.active_commitment_collector.collect(
            context["global_user_id"]
        )
        scheduled_candidates = await self.scheduled_event_collector.collect(context)
        progress_unavailable = not progress_candidates

        candidates = [
            *progress_candidates,
            *commitment_candidates,
            *scheduled_candidates,
        ]

        if mode in _ACTIVE_MODES and progress_unavailable:
            if commitment_candidates:
                candidates = [*commitment_candidates, *scheduled_candidates]
            else:
                result_payload = _result_payload(
                    recall_type="active_episode_agreement",
                    freshness_basis=(
                        "active-episode state was unavailable and no active "
                        "commitment memory was found"
                    ),
                    candidates=candidates,
                )
                result = _agent_result(
                    resolved=False,
                    result_payload=result_payload,
                )
                return result

        history_required = mode == "exact_agreement_history"
        if not history_required:
            history_required = _has_source_conflict(candidates)
        if history_required:
            history_candidates = await self.history_evidence_collector.collect(context)
            candidates.extend(history_candidates)

        if not candidates:
            result_payload = _result_payload(
                recall_type="active_episode_agreement",
                freshness_basis="no recall evidence found",
                candidates=[],
            )
            result = _agent_result(resolved=False, result_payload=result_payload)
            return result

        ranked_candidates = _rank_candidates(mode, candidates)
        selected = ranked_candidates[0]
        conflicts = _conflict_notes(candidates)
        supporting_sources = sorted(
            {
                candidate["source"]
                for candidate in ranked_candidates
                if candidate["source"] != selected["source"]
            }
        )
        recall_type = _recall_type_for(mode, selected, conflicts)
        freshness_basis = _freshness_basis(selected, mode, progress_unavailable)

        result_payload = _result_payload(
            selected_summary=selected["claim"],
            recall_type=recall_type,
            primary_source=selected["source"],
            supporting_sources=supporting_sources,
            freshness_basis=freshness_basis,
            conflicts=conflicts,
            candidates=ranked_candidates,
        )
        result = _agent_result(resolved=True, result_payload=result_payload)
        return result
