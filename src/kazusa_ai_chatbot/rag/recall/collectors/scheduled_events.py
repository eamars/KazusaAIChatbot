"""Scheduled-event recall collector."""

from __future__ import annotations

from kazusa_ai_chatbot.db import query_pending_scheduled_events
from kazusa_ai_chatbot.rag.recall.contracts import _candidate, _event_claim

class ScheduledEventCollector:
    """Collect pending executable future actions for the current user."""

    async def collect(self, context: dict[str, Any]) -> list[dict[str, str]]:
        """Read pending scheduled events and convert them to candidates."""

        query_args = {
            "platform": context["platform"],
            "platform_channel_id": context["platform_channel_id"],
            "global_user_id": context["global_user_id"],
            "current_timestamp_utc": context["current_timestamp_utc"],
            "limit": 10,
        }
        events = await query_pending_scheduled_events(**query_args)
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
