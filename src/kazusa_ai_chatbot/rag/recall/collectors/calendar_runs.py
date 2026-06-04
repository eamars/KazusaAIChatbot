"""Calendar-run recall collector."""

from __future__ import annotations

from typing import Any

from kazusa_ai_chatbot.calendar_scheduler.repository import (
    list_pending_calendar_runs_for_source,
)
from kazusa_ai_chatbot.rag.recall.contracts import (
    _calendar_run_claim,
    _candidate,
)


_PENDING_CALENDAR_RUN_LIMIT = 10


class CalendarRunCollector:
    """Collect pending calendar actions for the current user scope."""

    async def collect(self, context: dict[str, Any]) -> list[dict[str, str]]:
        """Read pending calendar runs and convert them to Recall candidates."""

        query_args = {
            "platform": context["platform"],
            "platform_channel_id": context["platform_channel_id"],
            "global_user_id": context["global_user_id"],
            "current_timestamp_utc": context["current_timestamp_utc"],
            "limit": _PENDING_CALENDAR_RUN_LIMIT,
        }
        runs = await list_pending_calendar_runs_for_source(**query_args)
        candidates: list[dict[str, str]] = []
        for run in runs:
            candidates.append(
                _candidate(
                    source="calendar_runs",
                    claim=_calendar_run_claim(run),
                    temporal_scope="pending_future_action",
                    lifecycle_status="pending",
                    evidence_time=run.get("due_at", ""),
                    authority="supporting",
                )
            )
        selected_candidates = candidates[:_PENDING_CALENDAR_RUN_LIMIT]
        return selected_candidates
