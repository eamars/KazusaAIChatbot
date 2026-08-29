"""Semantic active-context and calendar inspection services."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from copy import copy
from datetime import UTC, datetime
from typing import Any

from kazusa_ai_chatbot.calendar_scheduler.repository import (
    list_calendar_schedules_for_inspection,
    list_pending_calendar_runs_for_source,
    list_recent_calendar_runs,
)
from kazusa_ai_chatbot.dsh_tool_gateway.contracts import (
    KazusaSemanticCapabilityResultV1,
    OpaqueReferenceCodec,
    SemanticPageV1,
    content_digest,
    new_evidence_receipt,
)

_MAX_RESULTS = 50
_CONTEXT_KINDS = frozenset({"commitments", "progress", "history", "calendar"})


def _limit(value: object, default: int = 10) -> int:
    """Clamp a semantic result limit."""

    if isinstance(value, bool) or not isinstance(value, int):
        return default
    return max(1, min(value, _MAX_RESULTS))


def _mapping(value: object, field: str) -> dict[str, Any]:
    """Convert one service row to a mapping."""

    if isinstance(value, Mapping):
        return dict(value)
    try:
        return dict(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be an object") from exc


class RecallCalendarSemanticService:
    """Expose bounded active-context and calendar semantic views."""

    def __init__(
        self,
        *,
        codec: OpaqueReferenceCodec,
        recall: Callable[..., Awaitable[list[Mapping[str, Any]]]] | None = None,
        schedules: Callable[..., Awaitable[list[dict[str, Any]]]] = list_calendar_schedules_for_inspection,
        recent_runs: Callable[..., Awaitable[list[dict[str, Any]]]] = list_recent_calendar_runs,
        pending_runs: Callable[..., Awaitable[list[dict[str, Any]]]] = list_pending_calendar_runs_for_source,
    ) -> None:
        self._codec = codec
        self._recall = recall
        self._schedules = schedules
        self._recent_runs = recent_runs
        self._pending_runs = pending_runs

    def with_authority(self, authority: Mapping[str, Any] | object) -> "RecallCalendarSemanticService":
        """Return a call-local service bound to the signed authority."""

        bound = copy(self)
        bound._codec = self._codec.with_authority(authority)
        return bound

    async def recall_active_context(
        self,
        *,
        kinds: Sequence[str],
        max_results: int = 10,
        context: Mapping[str, Any] | None = None,
    ) -> KazusaSemanticCapabilityResultV1:
        """Return current active context with semantic provenance."""

        if (
            not isinstance(kinds, Sequence)
            or isinstance(kinds, (str, bytes))
            or not kinds
            or any(kind not in _CONTEXT_KINDS for kind in kinds)
        ):
            return KazusaSemanticCapabilityResultV1.failure(
                "invalid", "CONTEXT_KINDS_INVALID", "At least one supported context kind is required."
            )
        selected_kinds = list(kinds)
        if self._recall is None:
            return KazusaSemanticCapabilityResultV1.success()
        rows = await self._recall(kinds=selected_kinds, context=dict(context or {}), limit=_limit(max_results))
        return _rows_result(self._codec, rows, "active_context")

    async def read_calendar_context(
        self,
        *,
        view: str,
        max_results: int = 10,
        next_page_ref: str | None = None,
        source_scope: Mapping[str, str] | None = None,
        current_timestamp_utc: str | None = None,
    ) -> KazusaSemanticCapabilityResultV1:
        """Read schedules or calendar runs through one semantic view."""

        if view not in {"schedules", "recent_runs", "pending_runs"}:
            return KazusaSemanticCapabilityResultV1.failure(
                "invalid", "CALENDAR_VIEW_INVALID", "The calendar view is unsupported."
            )
        limit = _limit(max_results)
        offset = 0
        if next_page_ref is not None:
            try:
                payload = self._codec.resolve(next_page_ref, "calendar-page")
                offset = int(payload["offset"])
            except (KeyError, TypeError, ValueError):
                return KazusaSemanticCapabilityResultV1.failure(
                    "invalid", "PAGE_REFERENCE_INVALID", "The continuation reference is invalid."
                )
        if view == "schedules":
            rows = await self._schedules(limit=offset + limit + 1)
        elif view == "recent_runs":
            rows = await self._recent_runs(limit=offset + limit + 1)
        else:
            scope = dict(source_scope or {})
            required = ("platform", "platform_channel_id", "global_user_id")
            if any(not isinstance(scope.get(key), str) or not scope[key] for key in required):
                return KazusaSemanticCapabilityResultV1.failure(
                    "invalid", "CALENDAR_SCOPE_REQUIRED", "A calendar source scope is required."
                )
            timestamp = current_timestamp_utc or datetime.now(UTC).isoformat().replace("+00:00", "Z")
            rows = await self._pending_runs(
                platform=scope["platform"],
                platform_channel_id=scope["platform_channel_id"],
                global_user_id=scope["global_user_id"],
                current_timestamp_utc=timestamp,
                limit=offset + limit + 1,
            )
        selected = list(rows)[offset: offset + limit + 1]
        has_more = len(selected) > limit
        selected = selected[:limit]
        result = _rows_result(self._codec, selected, f"calendar-{view}")
        page = SemanticPageV1(
            has_more=has_more,
            next_page_ref=(
                self._codec.issue("calendar-page", {"offset": offset + limit})
                if has_more
                else None
            ),
        )
        return KazusaSemanticCapabilityResultV1(
            schema_version=result.schema_version,
            status=result.status,
            entities=result.entities,
            page=page,
            evidence=result.evidence,
            mutation=result.mutation,
            error=result.error,
        )


def _rows_result(
    codec: OpaqueReferenceCodec,
    rows: Sequence[Mapping[str, Any]],
    source_kind: str,
) -> KazusaSemanticCapabilityResultV1:
    """Project arbitrary active-context rows into semantic entities."""

    entities: list[dict[str, Any]] = []
    evidence = []
    for index, value in enumerate(rows[:_MAX_RESULTS]):
        row = _mapping(value, "context result")
        source_id = row.get("id") or row.get("run_id") or row.get("schedule_id") or str(index)
        reference = codec.issue("context", {"source_id": str(source_id), "kind": source_kind})
        entity: dict[str, Any] = {"context_ref": reference}
        for key in ("title", "summary", "status", "due_at", "next_run_at", "updated_at", "claim", "temporal_scope"):
            item = row.get(key)
            if isinstance(item, (str, int, float, bool)):
                entity[key] = item
        entities.append(entity)
        evidence.append(new_evidence_receipt(
            receipt_id=f"receipt-{source_kind}-{content_digest(reference)}",
            source_kind=source_kind,
            semantic_ref=reference,
            value=entity,
        ))
    return KazusaSemanticCapabilityResultV1.success(entities=entities, evidence=evidence)
