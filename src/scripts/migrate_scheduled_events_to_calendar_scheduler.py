"""One-time migration from legacy scheduled events into calendar runs."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from kazusa_ai_chatbot.calendar_scheduler import models, repository
from kazusa_ai_chatbot.db.script_operations import (
    cancel_pending_send_message_for_calendar_migration,
    list_scheduled_events_for_calendar_migration,
    mark_pending_future_cognition_migrated_for_calendar_migration,
)
from kazusa_ai_chatbot.time_boundary import storage_utc_now_iso


PENDING_STATUS = "pending"
LEGACY_FUTURE_COGNITION_TOOL = "trigger_future_cognition"
LEGACY_SEND_MESSAGE_TOOL = "send_message"


def build_migration_plan(
    legacy_events: list[dict[str, Any]],
    *,
    storage_timestamp_utc: str,
) -> dict[str, Any]:
    """Build an idempotent migration plan from legacy scheduled events."""

    pending_events = [
        event for event in legacy_events
        if event["status"] == PENDING_STATUS
    ]
    terminal_ignored = len(legacy_events) - len(pending_events)
    unknown_tool_events = [
        {
            "event_id": event["event_id"],
            "tool": event["tool"],
        }
        for event in pending_events
        if event["tool"] not in (
            LEGACY_FUTURE_COGNITION_TOOL,
            LEGACY_SEND_MESSAGE_TOOL,
        )
    ]
    future_events = [
        event for event in pending_events
        if event["tool"] == LEGACY_FUTURE_COGNITION_TOOL
    ]
    send_events = [
        event for event in pending_events
        if event["tool"] == LEGACY_SEND_MESSAGE_TOOL
    ]
    blocked = bool(unknown_tool_events)
    calendar_schedules: list[dict[str, Any]] = []
    calendar_runs: list[dict[str, Any]] = []
    if not blocked:
        for event in future_events:
            schedule = _future_cognition_schedule(
                event,
                storage_timestamp_utc=storage_timestamp_utc,
            )
            run = models.build_calendar_run_from_schedule(
                schedule,
                due_at=event["execute_at"],
                storage_timestamp_utc=storage_timestamp_utc,
            )
            calendar_schedules.append(schedule)
            calendar_runs.append(run)

    counts = {
        "total_legacy_rows": len(legacy_events),
        "total_pending": len(pending_events),
        "future_cognition_to_migrate": len(future_events),
        "legacy_send_message_to_cancel": len(send_events),
        "terminal_ignored": terminal_ignored,
        "unknown_tool": len(unknown_tool_events),
    }
    sample_ids = {
        "future_cognition_to_migrate": [
            event["event_id"] for event in future_events[:10]
        ],
    }
    plan = {
        "blocked": blocked,
        "counts": counts,
        "sample_ids": sample_ids,
        "cancelled_scheduled_event_ids": [
            event["event_id"] for event in send_events
        ],
        "migrated_scheduled_event_ids": [
            event["event_id"] for event in future_events
        ],
        "unknown_tool_events": unknown_tool_events,
        "calendar_schedules": calendar_schedules,
        "calendar_runs": calendar_runs,
    }
    return plan


async def apply_migration_plan(
    plan: dict[str, Any],
    *,
    repository: Any,
    dry_run: bool,
) -> dict[str, Any]:
    """Apply a migration plan unless running in dry-run mode."""

    if plan["blocked"]:
        summary = {
            "dry_run": dry_run,
            "applied": False,
            "blocked": True,
            "counts": plan["counts"],
        }
        return summary

    if dry_run:
        summary = {
            "dry_run": True,
            "applied": False,
            "blocked": False,
            "counts": plan["counts"],
        }
        return summary

    for schedule in plan["calendar_schedules"]:
        await repository.upsert_calendar_schedule(schedule)
    for run in plan["calendar_runs"]:
        await repository.upsert_calendar_run(run)

    failed_cancel_ids: list[str] = []
    for event_id in plan["cancelled_scheduled_event_ids"]:
        cancelled = await repository.cancel_pending_scheduled_event(event_id)
        if not cancelled:
            failed_cancel_ids.append(event_id)

    failed_migrated_ids: list[str] = []
    for event_id in plan["migrated_scheduled_event_ids"]:
        migrated = await repository.mark_pending_future_cognition_migrated(
            event_id,
        )
        if not migrated:
            failed_migrated_ids.append(event_id)

    failure_count = len(failed_cancel_ids) + len(failed_migrated_ids)

    summary = {
        "dry_run": False,
        "applied": failure_count == 0,
        "blocked": False,
        "legacy_mutation_failure_count": failure_count,
        "legacy_mutation_failures": {
            "cancelled_scheduled_event_ids": failed_cancel_ids,
            "migrated_scheduled_event_ids": failed_migrated_ids,
        },
        "counts": plan["counts"],
    }
    return summary


async def main() -> None:
    """Run the operator migration script."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    dry_run = not args.apply

    legacy_events = await list_scheduled_events_for_calendar_migration()
    now_utc = storage_utc_now_iso()
    plan = build_migration_plan(
        legacy_events,
        storage_timestamp_utc=now_utc,
    )
    summary = await apply_migration_plan(
        plan,
        repository=_MigrationRepository(),
        dry_run=dry_run,
    )
    output = {"plan": plan, "summary": summary}
    rendered = json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered, encoding="utf-8")
    else:
        print(rendered)


def _future_cognition_schedule(
    event: dict[str, Any],
    *,
    storage_timestamp_utc: str,
) -> dict[str, Any]:
    """Convert one pending future-cognition event to a calendar schedule."""

    source_scope = {
        "source_platform": event.get("source_platform"),
        "source_channel_id": event.get("source_channel_id"),
        "source_channel_type": event.get("source_channel_type"),
        "source_user_id": event.get("source_user_id"),
        "source_message_id": event.get("source_message_id"),
        "source_platform_bot_id": event.get("source_platform_bot_id"),
        "source_character_name": event.get("source_character_name"),
        "guild_id": event.get("guild_id"),
        "bot_role": event.get("bot_role"),
    }
    schedule = models.build_one_time_calendar_schedule(
        trigger_kind=models.TRIGGER_FUTURE_COGNITION,
        due_at=event["execute_at"],
        payload=event["args"],
        source_scope=source_scope,
        idempotency_key=event["event_id"],
        storage_timestamp_utc=storage_timestamp_utc,
    )
    schedule["legacy_source"] = {
        "collection": "scheduled_events",
        "event_id": event["event_id"],
    }
    return schedule


class _MigrationRepository:
    """Repository adapter for the one-time legacy scheduled-event migration."""

    async def upsert_calendar_schedule(self, schedule: dict[str, Any]) -> object:
        """Upsert a calendar schedule through the calendar repository."""

        result = await repository.upsert_calendar_schedule(schedule)
        return result

    async def upsert_calendar_run(self, run: dict[str, Any]) -> object:
        """Upsert a calendar run through the calendar repository."""

        result = await repository.upsert_calendar_run(run)
        return result

    async def cancel_pending_scheduled_event(self, event_id: str) -> bool:
        """Cancel one pending legacy visible-send event."""

        cancelled = await cancel_pending_send_message_for_calendar_migration(
            event_id,
        )
        return cancelled

    async def mark_pending_future_cognition_migrated(
        self,
        event_id: str,
    ) -> bool:
        """Mark one pending legacy future-cognition event migrated."""

        migrated = (
            await mark_pending_future_cognition_migrated_for_calendar_migration(
                event_id,
            )
        )
        return migrated


if __name__ == "__main__":
    asyncio.run(main())
