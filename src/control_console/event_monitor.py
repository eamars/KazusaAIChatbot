"""Bounded structured event monitor for Kazusa application events."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from typing import Any

from control_console.contracts import OperationalEventPage, OperationalEventQuery
from control_console.redaction import redact_mapping


EventReader = Callable[[OperationalEventQuery], Awaitable[list[dict[str, Any]]]]
DEFAULT_AGGREGATE_EVENT_TYPES = frozenset({
    "load_residue_context",
    "tick",
})
ATTENTION_LEVELS = frozenset({"error", "warning"})
ATTENTION_STATUSES = frozenset({
    "deferred",
    "degraded",
    "failed",
    "unavailable",
    "warning",
})


class EventMonitor:
    """Expose bounded structured application events with deterministic redaction."""

    def __init__(
        self,
        *,
        read_kazusa_events: EventReader,
    ) -> None:
        """Create an application-event monitor."""

        self._read_kazusa_events = read_kazusa_events

    async def query(self, query: OperationalEventQuery) -> OperationalEventPage:
        """Return bounded application events without aggregate-owned chatter."""

        rows = await self._read_kazusa_events(query)
        rows = [
            row
            for row in rows
            if _include_default_event(row, query=query)
        ]

        redacted_rows = [redact_mapping(row) for row in rows]
        redacted_rows.sort(key=lambda row: str(row.get("created_at", "")), reverse=True)
        bounded_rows = redacted_rows[:query.limit]
        page = OperationalEventPage(
            generated_at=datetime.now(timezone.utc),
            items=bounded_rows,
            facets=_event_facets(bounded_rows),
            query=query.model_dump(mode="json"),
            next_cursor=None,
        )
        return page


def _include_default_event(
    row: dict[str, Any],
    *,
    query: OperationalEventQuery,
) -> bool:
    """Keep aggregate-owned event types only when filtered or actionable."""

    if query.event_type:
        return True
    event_type = str(row.get("event_type", "")).strip()
    if event_type not in DEFAULT_AGGREGATE_EVENT_TYPES:
        return True
    level = str(row.get("level", "")).strip().lower()
    status = str(row.get("status", "")).strip().lower()
    return level in ATTENTION_LEVELS or status in ATTENTION_STATUSES


def _event_facets(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    """Count dynamic structured-event dimensions in the bounded result."""

    field_map = {
        "sources": "source",
        "levels": "level",
        "statuses": "status",
        "components": "component",
        "event_types": "event_type",
    }
    facets: dict[str, dict[str, int]] = {}
    for facet_name, field_name in field_map.items():
        counts: dict[str, int] = {}
        for row in rows:
            value = str(row.get(field_name, "")).strip()
            if not value:
                continue
            counts[value] = counts.get(value, 0) + 1
        facets[facet_name] = counts
    return facets
