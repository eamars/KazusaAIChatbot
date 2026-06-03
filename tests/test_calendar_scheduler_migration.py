"""Contract tests for the one-time scheduled-events migration."""

from __future__ import annotations

import pytest


NOW_UTC = "2026-06-04T00:00:00+00:00"


def _future_cognition_event() -> dict:
    return {
        "event_id": "future_cognition:action_attempt:future-123",
        "tool": "trigger_future_cognition",
        "execute_at": "2026-06-04T00:15:00+00:00",
        "created_at": "2026-06-04T00:00:00+00:00",
        "status": "pending",
        "args": {
            "episode_type": "self_cognition",
            "continuation_objective": "Re-check whether a natural pause appeared.",
            "source_action_attempt_id": "action_attempt:future-123",
            "source_refs": [],
            "continuation": {
                "mode": "scheduled_followup",
                "episode_type": "self_cognition",
                "max_depth": 1,
                "include_result_as": "scheduled_event",
            },
        },
        "source_platform": "qq",
        "source_channel_id": "group-1",
        "source_channel_type": "group",
        "source_user_id": "user-1",
        "source_message_id": "action_attempt:future-123",
        "source_platform_bot_id": "bot-1",
        "source_character_name": "Character",
        "guild_id": None,
        "bot_role": "system",
    }


def _send_message_event() -> dict:
    return {
        "event_id": "legacy-send-1",
        "tool": "send_message",
        "execute_at": "2026-06-04T00:15:00+00:00",
        "status": "pending",
        "args": {"text": "old delayed visible text"},
    }


def _completed_future_cognition_event() -> dict:
    event = _future_cognition_event()
    event["event_id"] = "completed-future-1"
    event["status"] = "completed"
    return event


def test_migration_plan_converts_future_cognition_only() -> None:
    """Future-cognition rows migrate; delayed visible sends are cancelled."""

    from scripts import migrate_scheduled_events_to_calendar_scheduler

    plan = migrate_scheduled_events_to_calendar_scheduler.build_migration_plan(
        [
            _future_cognition_event(),
            _send_message_event(),
            _completed_future_cognition_event(),
        ],
        storage_timestamp_utc=NOW_UTC,
    )
    repeated_plan = (
        migrate_scheduled_events_to_calendar_scheduler.build_migration_plan(
            [
                _future_cognition_event(),
                _send_message_event(),
                _completed_future_cognition_event(),
            ],
            storage_timestamp_utc=NOW_UTC,
        )
    )

    assert plan["blocked"] is False
    assert plan["counts"] == {
        "total_legacy_rows": 3,
        "total_pending": 2,
        "future_cognition_to_migrate": 1,
        "legacy_send_message_to_cancel": 1,
        "terminal_ignored": 1,
        "unknown_tool": 0,
    }
    assert plan["sample_ids"]["future_cognition_to_migrate"] == [
        "future_cognition:action_attempt:future-123"
    ]
    assert plan["cancelled_scheduled_event_ids"] == ["legacy-send-1"]
    assert plan["migrated_scheduled_event_ids"] == [
        "future_cognition:action_attempt:future-123"
    ]
    assert len(plan["calendar_schedules"]) == 1
    assert len(plan["calendar_runs"]) == 1
    assert plan["calendar_schedules"][0]["schedule_id"] == (
        repeated_plan["calendar_schedules"][0]["schedule_id"]
    )
    assert plan["calendar_runs"][0]["run_id"] == (
        repeated_plan["calendar_runs"][0]["run_id"]
    )
    assert plan["calendar_runs"][0]["trigger_kind"] == "future_cognition"
    assert plan["calendar_runs"][0]["due_at"] == "2026-06-04T00:15:00+00:00"


def test_migration_plan_blocks_unknown_pending_tools() -> None:
    """Apply mode must not proceed when legacy rows have unknown semantics."""

    from scripts import migrate_scheduled_events_to_calendar_scheduler

    unknown = {
        "event_id": "legacy-unknown-1",
        "tool": "unreviewed_tool",
        "execute_at": "2026-06-04T00:15:00+00:00",
        "status": "pending",
        "args": {},
    }

    plan = migrate_scheduled_events_to_calendar_scheduler.build_migration_plan(
        [_future_cognition_event(), unknown],
        storage_timestamp_utc=NOW_UTC,
    )

    assert plan["blocked"] is True
    assert plan["calendar_schedules"] == []
    assert plan["calendar_runs"] == []
    assert plan["unknown_tool_events"] == [
        {"event_id": "legacy-unknown-1", "tool": "unreviewed_tool"}
    ]
    assert plan["counts"]["unknown_tool"] == 1


@pytest.mark.asyncio
async def test_migration_dry_run_does_not_write_repository() -> None:
    """Dry-run mode should return a bounded summary without mutations."""

    from scripts import migrate_scheduled_events_to_calendar_scheduler

    class RepositoryDouble:
        def __init__(self) -> None:
            self.writes: list[tuple[str, dict]] = []
            self.cancellations: list[str] = []

        async def upsert_calendar_schedule(self, schedule: dict) -> object:
            self.writes.append(("schedule", schedule))
            return object()

        async def upsert_calendar_run(self, run: dict) -> object:
            self.writes.append(("run", run))
            return object()

        async def cancel_pending_scheduled_event(self, event_id: str) -> bool:
            self.cancellations.append(event_id)
            return True

        async def mark_pending_future_cognition_migrated(
            self,
            event_id: str,
        ) -> bool:
            self.cancellations.append(event_id)
            return True

    repository = RepositoryDouble()
    summary = await (
        migrate_scheduled_events_to_calendar_scheduler.apply_migration_plan(
            migrate_scheduled_events_to_calendar_scheduler.build_migration_plan(
                [_future_cognition_event(), _send_message_event()],
                storage_timestamp_utc=NOW_UTC,
            ),
            repository=repository,
            dry_run=True,
        )
    )

    assert summary["dry_run"] is True
    assert summary["applied"] is False
    assert repository.writes == []
    assert repository.cancellations == []


@pytest.mark.asyncio
async def test_migration_apply_writes_calendar_then_marks_legacy_rows() -> None:
    """Apply mode writes calendar rows before mutating legacy rows."""

    from scripts import migrate_scheduled_events_to_calendar_scheduler

    class RepositoryDouble:
        def __init__(self) -> None:
            self.operations: list[tuple[str, str]] = []

        async def upsert_calendar_schedule(self, schedule: dict) -> object:
            self.operations.append(("schedule", schedule["schedule_id"]))
            return object()

        async def upsert_calendar_run(self, run: dict) -> object:
            self.operations.append(("run", run["run_id"]))
            return object()

        async def cancel_pending_scheduled_event(self, event_id: str) -> bool:
            self.operations.append(("cancel", event_id))
            return True

        async def mark_pending_future_cognition_migrated(
            self,
            event_id: str,
        ) -> bool:
            self.operations.append(("migrated", event_id))
            return True

    repository = RepositoryDouble()
    plan = migrate_scheduled_events_to_calendar_scheduler.build_migration_plan(
        [_future_cognition_event(), _send_message_event()],
        storage_timestamp_utc=NOW_UTC,
    )

    summary = await (
        migrate_scheduled_events_to_calendar_scheduler.apply_migration_plan(
            plan,
            repository=repository,
            dry_run=False,
        )
    )

    assert summary["dry_run"] is False
    assert summary["applied"] is True
    assert summary["legacy_mutation_failure_count"] == 0
    assert summary["legacy_mutation_failures"] == {
        "cancelled_scheduled_event_ids": [],
        "migrated_scheduled_event_ids": [],
    }
    assert [operation[0] for operation in repository.operations] == [
        "schedule",
        "run",
        "cancel",
        "migrated",
    ]
    assert repository.operations[-1] == (
        "migrated",
        "future_cognition:action_attempt:future-123",
    )


@pytest.mark.asyncio
async def test_migration_apply_blocks_unknown_tools_without_writes() -> None:
    """Blocked plans must not partially write calendar or legacy rows."""

    from scripts import migrate_scheduled_events_to_calendar_scheduler

    class RepositoryDouble:
        def __init__(self) -> None:
            self.operations: list[str] = []

        async def upsert_calendar_schedule(self, schedule: dict) -> object:
            self.operations.append("schedule")
            return object()

        async def upsert_calendar_run(self, run: dict) -> object:
            self.operations.append("run")
            return object()

        async def cancel_pending_scheduled_event(self, event_id: str) -> bool:
            self.operations.append("cancel")
            return True

        async def mark_pending_future_cognition_migrated(
            self,
            event_id: str,
        ) -> bool:
            self.operations.append("migrated")
            return True

    unknown = {
        "event_id": "legacy-unknown-1",
        "tool": "unreviewed_tool",
        "execute_at": "2026-06-04T00:15:00+00:00",
        "status": "pending",
        "args": {},
    }
    repository = RepositoryDouble()
    plan = migrate_scheduled_events_to_calendar_scheduler.build_migration_plan(
        [_future_cognition_event(), unknown],
        storage_timestamp_utc=NOW_UTC,
    )

    summary = await (
        migrate_scheduled_events_to_calendar_scheduler.apply_migration_plan(
            plan,
            repository=repository,
            dry_run=False,
        )
    )

    assert summary["blocked"] is True
    assert summary["applied"] is False
    assert repository.operations == []


@pytest.mark.asyncio
async def test_migration_apply_reports_failed_legacy_cancel() -> None:
    """Failed legacy cancellations should make apply status partial."""

    from scripts import migrate_scheduled_events_to_calendar_scheduler

    class RepositoryDouble:
        def __init__(self) -> None:
            self.operations: list[tuple[str, str]] = []

        async def upsert_calendar_schedule(self, schedule: dict) -> object:
            self.operations.append(("schedule", schedule["schedule_id"]))
            return object()

        async def upsert_calendar_run(self, run: dict) -> object:
            self.operations.append(("run", run["run_id"]))
            return object()

        async def cancel_pending_scheduled_event(self, event_id: str) -> bool:
            self.operations.append(("cancel", event_id))
            return False

        async def mark_pending_future_cognition_migrated(
            self,
            event_id: str,
        ) -> bool:
            self.operations.append(("migrated", event_id))
            return True

    repository = RepositoryDouble()
    plan = migrate_scheduled_events_to_calendar_scheduler.build_migration_plan(
        [_future_cognition_event(), _send_message_event()],
        storage_timestamp_utc=NOW_UTC,
    )

    summary = await (
        migrate_scheduled_events_to_calendar_scheduler.apply_migration_plan(
            plan,
            repository=repository,
            dry_run=False,
        )
    )

    assert summary["applied"] is False
    assert summary["legacy_mutation_failure_count"] == 1
    assert summary["legacy_mutation_failures"] == {
        "cancelled_scheduled_event_ids": ["legacy-send-1"],
        "migrated_scheduled_event_ids": [],
    }
    assert [operation[0] for operation in repository.operations] == [
        "schedule",
        "run",
        "cancel",
        "migrated",
    ]


@pytest.mark.asyncio
async def test_migration_apply_reports_failed_legacy_migrate() -> None:
    """Failed migrated marks should make apply status partial."""

    from scripts import migrate_scheduled_events_to_calendar_scheduler

    class RepositoryDouble:
        def __init__(self) -> None:
            self.operations: list[tuple[str, str]] = []

        async def upsert_calendar_schedule(self, schedule: dict) -> object:
            self.operations.append(("schedule", schedule["schedule_id"]))
            return object()

        async def upsert_calendar_run(self, run: dict) -> object:
            self.operations.append(("run", run["run_id"]))
            return object()

        async def cancel_pending_scheduled_event(self, event_id: str) -> bool:
            self.operations.append(("cancel", event_id))
            return True

        async def mark_pending_future_cognition_migrated(
            self,
            event_id: str,
        ) -> bool:
            self.operations.append(("migrated", event_id))
            return False

    repository = RepositoryDouble()
    plan = migrate_scheduled_events_to_calendar_scheduler.build_migration_plan(
        [_future_cognition_event(), _send_message_event()],
        storage_timestamp_utc=NOW_UTC,
    )

    summary = await (
        migrate_scheduled_events_to_calendar_scheduler.apply_migration_plan(
            plan,
            repository=repository,
            dry_run=False,
        )
    )

    assert summary["applied"] is False
    assert summary["legacy_mutation_failure_count"] == 1
    assert summary["legacy_mutation_failures"] == {
        "cancelled_scheduled_event_ids": [],
        "migrated_scheduled_event_ids": [
            "future_cognition:action_attempt:future-123",
        ],
    }
    assert [operation[0] for operation in repository.operations] == [
        "schedule",
        "run",
        "cancel",
        "migrated",
    ]


@pytest.mark.asyncio
async def test_migration_repository_uses_script_operations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The script adapter should use maintenance DB helpers for legacy rows."""

    from scripts import migrate_scheduled_events_to_calendar_scheduler

    operations: list[tuple[str, str]] = []

    async def cancel_pending_send(event_id: str) -> bool:
        operations.append(("cancel", event_id))
        return True

    async def mark_pending_migrated(event_id: str) -> bool:
        operations.append(("migrated", event_id))
        return True

    monkeypatch.setattr(
        migrate_scheduled_events_to_calendar_scheduler,
        "cancel_pending_send_message_for_calendar_migration",
        cancel_pending_send,
    )
    monkeypatch.setattr(
        migrate_scheduled_events_to_calendar_scheduler,
        "mark_pending_future_cognition_migrated_for_calendar_migration",
        mark_pending_migrated,
    )
    migration_repository = (
        migrate_scheduled_events_to_calendar_scheduler._MigrationRepository()
    )

    cancelled = await migration_repository.cancel_pending_scheduled_event(
        "legacy-send-1",
    )
    migrated = (
        await migration_repository.mark_pending_future_cognition_migrated(
            "future_cognition:action_attempt:future-123",
        )
    )

    assert cancelled is True
    assert migrated is True
    assert operations == [
        ("cancel", "legacy-send-1"),
        ("migrated", "future_cognition:action_attempt:future-123"),
    ]
