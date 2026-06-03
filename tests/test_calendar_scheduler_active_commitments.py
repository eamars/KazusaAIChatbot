"""Contract tests for active-commitment calendar reconciliation."""

from __future__ import annotations

from typing import Any

import pytest


NOW_UTC = "2026-06-04T00:00:00+00:00"
DUE_AT = "2026-06-04T00:15:00+00:00"


def _active_commitment(*, due_at: str | None = DUE_AT) -> dict[str, Any]:
    return {
        "unit_id": "commitment-1",
        "global_user_id": "user-1",
        "unit_type": "active_commitment",
        "status": "active",
        "fact": "The character accepted a follow-up.",
        "subjective_appraisal": "The commitment should be checked later.",
        "relationship_signal": "Follow through at the due time.",
        "due_at": due_at,
        "updated_at": NOW_UTC,
    }


class _RepositoryDouble:
    def __init__(self) -> None:
        self.upserted: list[dict[str, Any]] = []
        self.cancelled: list[dict[str, Any]] = []

    async def upsert_calendar_schedule(self, schedule: dict[str, Any]) -> object:
        self.upserted.append(schedule)
        return object()

    async def cancel_calendar_schedule_by_idempotency_key(
        self,
        idempotency_key: str,
        *,
        storage_timestamp_utc: str,
        reason: str,
    ) -> bool:
        self.cancelled.append({
            "idempotency_key": idempotency_key,
            "storage_timestamp_utc": storage_timestamp_utc,
            "reason": reason,
        })
        return True


@pytest.mark.asyncio
async def test_reconcile_active_commitment_upserts_due_schedule() -> None:
    """Active commitments with absolute due_at should get one due trigger."""

    from kazusa_ai_chatbot.calendar_scheduler import handlers, models

    repository = _RepositoryDouble()

    result = await handlers.reconcile_active_commitment_calendar_schedule(
        _active_commitment(),
        repository=repository,
        storage_timestamp_utc=NOW_UTC,
    )

    assert result["status"] == "scheduled"
    assert len(repository.upserted) == 1
    schedule = repository.upserted[0]
    assert schedule["trigger_kind"] == models.TRIGGER_COMMITMENT_DUE_COGNITION
    assert schedule["next_run_at"] == DUE_AT
    assert schedule["idempotency_key"] == "commitment_due:commitment-1"
    assert schedule["payload"] == {
        "unit_id": "commitment-1",
        "global_user_id": "user-1",
        "due_at": DUE_AT,
    }
    assert "fact" not in schedule["payload"]
    assert "subjective_appraisal" not in schedule["payload"]
    assert repository.cancelled == []


@pytest.mark.asyncio
async def test_reconcile_active_commitment_cancels_closed_or_missing_due() -> None:
    """Closed or undated commitment units should cancel their due schedule."""

    from kazusa_ai_chatbot.calendar_scheduler import handlers

    repository = _RepositoryDouble()
    closed_unit = _active_commitment(due_at=None)
    closed_unit["status"] = "completed"

    result = await handlers.reconcile_active_commitment_calendar_schedule(
        closed_unit,
        repository=repository,
        storage_timestamp_utc=NOW_UTC,
    )

    assert result["status"] == "cancelled"
    assert repository.upserted == []
    assert repository.cancelled == [
        {
            "idempotency_key": "commitment_due:commitment-1",
            "storage_timestamp_utc": NOW_UTC,
            "reason": "active_commitment_not_schedulable",
        }
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("unit_patch", "expected_status"),
    [
        ({"unit_type": "objective_fact"}, "cancelled"),
        ({"status": "cancelled"}, "cancelled"),
        ({"due_at": None}, "cancelled"),
        ({"due_at": ""}, "cancelled"),
        ({"due_at": "not-a-storage-timestamp"}, "cancelled"),
    ],
)
async def test_reconcile_active_commitment_rejects_unschedulable_units(
    unit_patch: dict[str, Any],
    expected_status: str,
) -> None:
    """Only active commitments with valid absolute UTC due_at are scheduled."""

    from kazusa_ai_chatbot.calendar_scheduler import handlers

    repository = _RepositoryDouble()
    unit = _active_commitment()
    unit.update(unit_patch)

    result = await handlers.reconcile_active_commitment_calendar_schedule(
        unit,
        repository=repository,
        storage_timestamp_utc=NOW_UTC,
    )

    assert result["status"] == expected_status
    assert repository.upserted == []
    assert repository.cancelled[0]["idempotency_key"] == (
        "commitment_due:commitment-1"
    )


@pytest.mark.asyncio
async def test_commitment_due_handler_skips_stale_due_payload() -> None:
    """Execution must re-read the memory unit and reject due-time drift."""

    from kazusa_ai_chatbot.calendar_scheduler import handlers

    stale_unit = _active_commitment(due_at="2026-06-04T02:00:00+00:00")
    built_cases: list[dict[str, Any]] = []

    async def read_memory_unit(unit_id: str) -> dict[str, Any]:
        assert unit_id == "commitment-1"
        return stale_unit

    async def build_case(unit: dict[str, Any]) -> dict[str, Any]:
        built_cases.append(unit)
        return {"case_id": "case-1"}

    result = await handlers.handle_commitment_due_cognition_run(
        {
            "run_id": "run-commitment-1",
            "payload": {
                "unit_id": "commitment-1",
                "global_user_id": "user-1",
                "due_at": DUE_AT,
            },
        },
        memory_unit_reader=read_memory_unit,
        active_commitment_case_builder=build_case,
    )

    assert result == {
        "status": "skipped",
        "reason": "stale_active_commitment_due_at",
        "unit_id": "commitment-1",
    }
    assert built_cases == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("stored_unit", "expected_reason"),
    [
        (None, "active_commitment_not_found"),
        (
            {
                **_active_commitment(),
                "status": "completed",
            },
            "active_commitment_not_active",
        ),
        (
            {
                **_active_commitment(),
                "unit_type": "objective_fact",
            },
            "active_commitment_wrong_type",
        ),
        (
            {
                **_active_commitment(),
                "global_user_id": "other-user",
            },
            "active_commitment_wrong_user",
        ),
    ],
)
async def test_commitment_due_handler_skips_structural_mismatches(
    stored_unit: dict[str, Any] | None,
    expected_reason: str,
) -> None:
    """Due handlers validate stored structure before building a source case."""

    from kazusa_ai_chatbot.calendar_scheduler import handlers

    built_cases: list[dict[str, Any]] = []

    async def read_memory_unit(unit_id: str) -> dict[str, Any] | None:
        assert unit_id == "commitment-1"
        return stored_unit

    async def build_case(unit: dict[str, Any]) -> dict[str, Any]:
        built_cases.append(unit)
        return {"case_id": "case-1"}

    result = await handlers.handle_commitment_due_cognition_run(
        {
            "run_id": "run-commitment-1",
            "payload": {
                "unit_id": "commitment-1",
                "global_user_id": "user-1",
                "due_at": DUE_AT,
            },
        },
        memory_unit_reader=read_memory_unit,
        active_commitment_case_builder=build_case,
    )

    assert result == {
        "status": "skipped",
        "reason": expected_reason,
        "unit_id": "commitment-1",
    }
    assert built_cases == []
