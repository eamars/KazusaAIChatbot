"""Typed, bounded contracts for protected trace correlation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast


TRACE_CORRELATION_MANIFEST_SCHEMA = "trace_correlation_manifest.v1"
TRACE_CORRELATION_CONTEXT_SCHEMA = "trace_correlation_context.v1"

CorrelationSourceSurface = Literal[
    "web_control_trace_id",
    "protected_llm_trace_id",
    "protected_cognition_invocation_id",
    "protected_global_user_id",
    "protected_action_attempt_id",
    "protected_background_work_job_id",
    "protected_accepted_task_id",
    "protected_calendar_schedule_id",
    "protected_calendar_run_id",
    "unknown",
]
CorrelationStatus = Literal[
    "confirmed",
    "not_found",
    "ambiguous",
    "not_captured",
    "not_available",
    "not_available_from_web",
    "not_applicable",
    "conflict",
]

SOURCE_SURFACES: tuple[CorrelationSourceSurface, ...] = (
    "web_control_trace_id",
    "protected_llm_trace_id",
    "protected_cognition_invocation_id",
    "protected_global_user_id",
    "protected_action_attempt_id",
    "protected_background_work_job_id",
    "protected_accepted_task_id",
    "protected_calendar_schedule_id",
    "protected_calendar_run_id",
    "unknown",
)

MAX_PARENT_CANDIDATES = 2
MAX_COMPANION_CANDIDATES = 32
MAX_CONVERSATION_ROWS = 64
MAX_FAILURE_CAPSULES = 32

_PARENT_TRACE_SURFACES = frozenset({
    "web_control_trace_id",
    "protected_llm_trace_id",
    "protected_global_user_id",
    "protected_cognition_invocation_id",
})
_COMPANION_TRACE_SURFACES = frozenset({
    "protected_action_attempt_id",
    "protected_background_work_job_id",
    "protected_accepted_task_id",
    "protected_calendar_schedule_id",
    "protected_calendar_run_id",
})


@dataclass(frozen=True, slots=True)
class TraceCorrelationResolution:
    """Resolution result that preserves zero and multiple candidates."""

    source_surface: CorrelationSourceSurface
    identifier: str
    status: CorrelationStatus
    trace_ids: tuple[str, ...] = ()
    reason: str = ""

    @property
    def trace_id(self) -> str:
        """Return the single confirmed parent trace, otherwise empty."""

        if self.status != "confirmed" or len(self.trace_ids) != 1:
            return ""
        return self.trace_ids[0]

    def as_dict(self) -> dict[str, Any]:
        """Project the resolution into a manifest-safe mapping."""

        return {
            "source_surface": self.source_surface,
            "identifier": self.identifier,
            "status": self.status,
            "trace_id": self.trace_id,
            "candidate_trace_ids": list(self.trace_ids),
            "reason": self.reason,
        }


def normalize_source_surface(value: str) -> CorrelationSourceSurface:
    """Validate one source-surface label without inferring its meaning."""

    normalized = str(value).strip().lower()
    if normalized not in SOURCE_SURFACES:
        raise ValueError(f"unsupported correlation source surface: {value}")
    return cast(CorrelationSourceSurface, normalized)


def normalize_identifier(value: str) -> str:
    """Normalize one copied identifier without shape-based classification."""

    normalized = str(value).strip()
    if not normalized:
        raise ValueError("correlation identifier must not be empty")
    return normalized


def unique_trace_ids(values: Sequence[object]) -> tuple[str, ...]:
    """Return non-empty trace ids in stable first-seen order."""

    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            continue
        trace_id = value.strip()
        if not trace_id or trace_id in seen:
            continue
        result.append(trace_id)
        seen.add(trace_id)
    return tuple(result)


def _trace_id_values(
    rows: Sequence[Mapping[str, Any]],
    *,
    source_surface: CorrelationSourceSurface,
) -> tuple[str, ...]:
    """Extract only the canonical trace field for one typed source."""

    if source_surface in _PARENT_TRACE_SURFACES:
        fields = ("trace_id",)
    elif source_surface in _COMPANION_TRACE_SURFACES:
        fields = ("source_llm_trace_id",)
    else:
        fields = ()
    values = [row.get(field) for row in rows for field in fields]
    return unique_trace_ids(values)


def resolve_trace_candidates(
    *,
    source_surface: str,
    identifier: str,
    rows: Sequence[Mapping[str, Any]],
    protected_available: bool = True,
) -> TraceCorrelationResolution:
    """Resolve a typed candidate set into a bounded terminal status.

    The function deliberately receives already-filtered exact rows. It never
    classifies an opaque value by length, hexadecimal shape, timestamps, or
    textual similarity.
    """

    normalized_surface = normalize_source_surface(source_surface)
    normalized_identifier = normalize_identifier(identifier)
    if normalized_surface == "unknown":
        return TraceCorrelationResolution(
            source_surface=normalized_surface,
            identifier=normalized_identifier,
            status="not_available_from_web",
            reason="source surface is not allowlisted",
        )
    if not protected_available:
        return TraceCorrelationResolution(
            source_surface=normalized_surface,
            identifier=normalized_identifier,
            status="not_available",
            reason="protected read boundary unavailable",
        )

    trace_ids = _trace_id_values(
        rows,
        source_surface=normalized_surface,
    )
    if normalized_surface in _COMPANION_TRACE_SURFACES:
        conflict_trace_ids = unique_trace_ids(
            [
                value
                for row in rows
                if row.get("correlation_write_status") == "conflict"
                for value in (
                    row.get("source_llm_trace_id"),
                    row.get("correlation_conflict_source_llm_trace_id"),
                )
            ]
        )
        if conflict_trace_ids:
            return TraceCorrelationResolution(
                source_surface=normalized_surface,
                identifier=normalized_identifier,
                status="conflict",
                trace_ids=conflict_trace_ids[:MAX_PARENT_CANDIDATES],
                reason=(
                    "durable source trace conflict was recorded; parent "
                    "selection is blocked"
                ),
            )
    if not trace_ids:
        if normalized_surface in _COMPANION_TRACE_SURFACES and rows:
            return TraceCorrelationResolution(
                source_surface=normalized_surface,
                identifier=normalized_identifier,
                status="not_captured",
                reason=(
                    "exact durable source row exists without a captured "
                    "source trace"
                ),
            )
        return TraceCorrelationResolution(
            source_surface=normalized_surface,
            identifier=normalized_identifier,
            status="not_found",
            reason="exact query returned no source trace",
        )
    if len(trace_ids) > 1:
        return TraceCorrelationResolution(
            source_surface=normalized_surface,
            identifier=normalized_identifier,
            status="ambiguous",
            trace_ids=trace_ids[:MAX_PARENT_CANDIDATES],
            reason="exact source maps to multiple parent traces",
        )
    return TraceCorrelationResolution(
        source_surface=normalized_surface,
        identifier=normalized_identifier,
        status="confirmed",
        trace_ids=trace_ids,
    )


def merge_trace_candidates(
    *,
    source_surface: str,
    identifier: str,
    candidate_sets: Sequence[Sequence[Mapping[str, Any]]],
    protected_available: bool = True,
) -> TraceCorrelationResolution:
    """Resolve candidates collected from multiple exact lookup paths."""

    rows = [row for candidate_set in candidate_sets for row in candidate_set]
    return resolve_trace_candidates(
        source_surface=source_surface,
        identifier=identifier,
        rows=rows,
        protected_available=protected_available,
    )


def bounded_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    """Copy a fixed number of mapping rows for manifest assembly."""

    if limit < 1:
        raise ValueError("correlation row limit must be positive")
    return [dict(row) for row in rows[:limit]]


def safe_identifier_row(
    row: Mapping[str, Any],
    *,
    owner: str,
    collection: str,
) -> dict[str, Any]:
    """Project stable identifier metadata without payload or raw content."""

    allowed_fields = (
        "trace_id",
        "llm_trace_id",
        "row_id",
        "step_id",
        "episode_id",
        "source_llm_trace_id",
        "parent_llm_trace_id",
        "source_background_work_job_id",
        "source_calendar_run_id",
        "source_action_attempt_id",
        "attempt_id",
        "action_attempt_id",
        "job_id",
        "accepted_task_id",
        "calendar_schedule_id",
        "calendar_run_id",
        "schedule_id",
        "run_id",
        "cognition_invocation_id",
        "global_user_id",
        "status",
        "platform",
        "platform_channel_id",
        "platform_message_id",
        "channel_type",
        "delivery_tracking_id",
        "correlation_write_status",
        "correlation_conflict_source_llm_trace_id",
        "started_at",
        "completed_at",
        "created_at",
        "due_at",
    )
    values: dict[str, Any] = {
        field: row[field]
        for field in allowed_fields
        if field in row and isinstance(row[field], (str, int, float, bool))
    }
    values["owner"] = owner
    values["collection"] = collection
    return values


def build_trace_correlation_manifest(
    *,
    generated_at: str,
    resolution: TraceCorrelationResolution,
    explicit_trace_id: str = "",
    cognition_invocation_id: str = "",
    parent_rows: Sequence[Mapping[str, Any]] = (),
    identifiers: Mapping[str, Mapping[str, Any]] | None = None,
    joins: Mapping[str, Mapping[str, Any]] | None = None,
    unresolved: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Build the bounded manifest envelope shared by diagnostic exporters."""

    parent_trace: dict[str, Any] = resolution.as_dict()
    parent_trace["capture_availability"] = (
        "recorded" if resolution.status == "confirmed" else resolution.status
    )
    parent_trace["runs"] = [
        safe_identifier_row(
            row,
            owner="llm_trace_run",
            collection="llm_trace_runs",
        )
        for row in bounded_rows(parent_rows, limit=MAX_PARENT_CANDIDATES)
    ]
    manifest = {
        "schema_version": TRACE_CORRELATION_MANIFEST_SCHEMA,
        "generated_at": generated_at,
        "input": {
            "identifier": resolution.identifier,
            "source_surface": resolution.source_surface,
            "trace_id_override": explicit_trace_id,
            "cognition_invocation_id": cognition_invocation_id,
        },
        "parent_trace": parent_trace,
        "identifiers": dict(identifiers or {}),
        "joins": dict(joins or {}),
        "availability": {
            "source_surface": (
                "rendered"
                if resolution.source_surface == "web_control_trace_id"
                else "protected_only"
            ),
            "parent_trace": resolution.status,
        },
        "unresolved": [dict(row) for row in unresolved[:MAX_COMPANION_CANDIDATES]],
    }
    return manifest


def build_trace_correlation_context(
    *,
    source_llm_trace_id: str = "",
    source_episode_id: str = "",
    source_background_work_job_id: str = "",
    source_calendar_run_id: str = "",
) -> dict[str, str]:
    """Build the bounded source-owner context carried by durable records."""

    return {
        "schema_version": TRACE_CORRELATION_CONTEXT_SCHEMA,
        "source_llm_trace_id": source_llm_trace_id.strip(),
        "source_episode_id": source_episode_id.strip(),
        "source_background_work_job_id": source_background_work_job_id.strip(),
        "source_calendar_run_id": source_calendar_run_id.strip(),
    }
