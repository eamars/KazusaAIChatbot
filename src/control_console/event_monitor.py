"""Bounded merged event monitor for console, process, and Kazusa events."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from typing import Any

from control_console.contracts import OperationalEventPage, OperationalEventQuery
from control_console.redaction import redact_mapping


EventReader = Callable[[OperationalEventQuery], Awaitable[list[dict[str, Any]]]]


class EventMonitor:
    """Merge bounded operational event sources with deterministic redaction."""

    def __init__(
        self,
        *,
        read_audit_events: EventReader,
        read_process_logs: EventReader,
        read_kazusa_events: EventReader,
    ) -> None:
        """Create a merged event monitor."""

        self._read_audit_events = read_audit_events
        self._read_process_logs = read_process_logs
        self._read_kazusa_events = read_kazusa_events

    async def query(self, query: OperationalEventQuery) -> OperationalEventPage:
        """Return a bounded merged event page."""

        rows: list[dict[str, Any]] = []
        if query.source in {"all", "console"}:
            rows.extend(await self._read_audit_events(query))
        if query.source in {"all", "process"}:
            rows.extend(await self._read_process_logs(query))
        if query.source in {"all", "kazusa"}:
            rows.extend(await self._read_kazusa_events(query))

        redacted_rows = [redact_mapping(row) for row in rows]
        redacted_rows.sort(key=lambda row: str(row.get("created_at", "")), reverse=True)
        bounded_rows = redacted_rows[:query.limit]
        page = OperationalEventPage(
            generated_at=datetime.now(timezone.utc),
            items=bounded_rows,
            next_cursor=None,
        )
        return page
