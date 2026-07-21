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
        self.refreshed: list[dict[str, Any]] = []
        self.upserted_runs: list[dict[str, Any]] = []
        self.cancelled: list[dict[str, Any]] = []

    async def upsert_calendar_schedule(self, schedule: dict[str, Any]) -> object:
        self.upserted.append(schedule)
        return object()

    async def refresh_calendar_schedule_state(
        self,
        schedule: dict[str, Any],
    ) -> object:
        self.refreshed.append(schedule)
        return object()

    async def upsert_calendar_run(self, run: dict[str, Any]) -> object:
        self.upserted_runs.append(run)
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


def _consolidation_state() -> dict[str, Any]:
    return {
        "storage_timestamp_utc": NOW_UTC,
        "global_user_id": "user-1",
        "rag_result": {"user_memory_unit_candidates": []},
    }


def _candidate(*, due_at: str = DUE_AT) -> dict[str, Any]:
    return {
        "candidate_id": "candidate-1",
        "unit_type": "active_commitment",
        "fact": "The character accepted a follow-up.",
        "subjective_appraisal": "The commitment should be checked later.",
        "relationship_signal": "Follow through at the due time.",
        "due_at": due_at,
        "evidence_refs": [],
    }


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
    assert repository.upserted == []
    assert len(repository.refreshed) == 1
    schedule = repository.refreshed[0]
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
    assert len(repository.upserted_runs) == 1
    run = repository.upserted_runs[0]
    assert run["trigger_kind"] == models.TRIGGER_COMMITMENT_DUE_COGNITION
    assert run["status"] == models.RUN_STATUS_PENDING
    assert run["due_at"] == DUE_AT
    assert run["schedule_id"] == schedule["schedule_id"]
    assert run["payload"] == schedule["payload"]
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
    assert repository.refreshed == []
    assert repository.upserted_runs == []
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
    assert repository.refreshed == []
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
        "cognition_source": {
            "source_kind": "scheduler_event",
            "source_id": "run-commitment-1",
            "occurred_at": DUE_AT,
            "semantic_summary": (
                "scheduled commitment was skipped: "
                "stale_active_commitment_due_at"
            ),
        },
    }
    assert built_cases == []


@pytest.mark.asyncio
async def test_consolidation_create_reconciles_active_commitment_schedule(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """New active commitments must materialize one calendar due schedule."""

    from kazusa_ai_chatbot.consolidation import memory_units as memory_units_module

    captured: dict[str, Any] = {}

    async def retrieve_candidates(global_user_id: str, **kwargs: Any) -> list[dict]:
        captured["retrieval_user_id"] = global_user_id
        captured["surfaced_units"] = kwargs["surfaced_units"]
        return []

    async def judge_merge(candidate: dict, candidate_clusters: list[dict]) -> dict:
        captured["merge_candidate"] = candidate
        captured["merge_clusters"] = candidate_clusters
        return {
            "candidate_id": candidate["candidate_id"],
            "decision": "create",
            "cluster_id": "",
            "reason": "new active commitment",
        }

    async def insert_units(
        global_user_id: str,
        units: list[dict],
        **kwargs: Any,
    ) -> list[dict]:
        captured["insert_user_id"] = global_user_id
        captured["insert_units"] = units
        captured["insert_timestamp"] = kwargs["storage_timestamp_utc"]
        return [{
            **units[0],
            "unit_id": "commitment-created",
            "global_user_id": global_user_id,
            "status": "active",
            "updated_at": kwargs["storage_timestamp_utc"],
        }]

    async def reconcile_schedule(unit: dict, **kwargs: Any) -> dict[str, str]:
        captured["reconciled_unit"] = unit
        captured["reconcile_repository"] = kwargs["repository"]
        captured["reconcile_timestamp"] = kwargs["storage_timestamp_utc"]
        return {"status": "scheduled", "unit_id": unit["unit_id"]}

    monkeypatch.setattr(
        memory_units_module,
        "retrieve_memory_unit_merge_candidates",
        retrieve_candidates,
    )
    monkeypatch.setattr(
        memory_units_module,
        "_judge_memory_unit_merge",
        judge_merge,
    )
    monkeypatch.setattr(
        memory_units_module,
        "insert_user_memory_units",
        insert_units,
    )
    monkeypatch.setattr(
        memory_units_module,
        "reconcile_active_commitment_calendar_schedule",
        reconcile_schedule,
    )

    result = await memory_units_module.process_memory_unit_candidate(
        _consolidation_state(),
        _candidate(),
    )

    assert result["unit_id"] == "commitment-created"
    assert captured["insert_user_id"] == "user-1"
    assert captured["insert_timestamp"] == NOW_UTC
    assert captured["reconcile_timestamp"] == NOW_UTC
    assert captured["reconciled_unit"]["unit_id"] == "commitment-created"
    assert captured["reconciled_unit"]["unit_type"] == "active_commitment"
    assert captured["reconciled_unit"]["status"] == "active"
    assert captured["reconciled_unit"]["global_user_id"] == "user-1"
    assert captured["reconciled_unit"]["due_at"] == DUE_AT


@pytest.mark.asyncio
async def test_consolidation_merge_reconciles_changed_commitment_due_schedule(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Merge/evolve updates that carry due_at must refresh calendar timing."""

    from kazusa_ai_chatbot.consolidation import memory_units as memory_units_module

    captured: dict[str, Any] = {}
    updated_due_at = "2026-06-04T01:15:00+00:00"
    existing_unit = {
        **_active_commitment(),
        "unit_id": "commitment-existing",
        "due_at": DUE_AT,
    }

    async def retrieve_candidates(global_user_id: str, **kwargs: Any) -> list[dict]:
        captured["retrieval_user_id"] = global_user_id
        captured["surfaced_units"] = kwargs["surfaced_units"]
        return [existing_unit]

    async def judge_merge(candidate: dict, candidate_clusters: list[dict]) -> dict:
        captured["merge_candidate"] = candidate
        captured["merge_clusters"] = candidate_clusters
        return {
            "candidate_id": candidate["candidate_id"],
            "decision": "evolve",
            "cluster_id": "commitment-existing",
            "reason": "same commitment with a new due time",
        }

    async def rewrite_unit(
        state: dict,
        candidate: dict,
        merge_result: dict,
    ) -> dict:
        captured["rewrite_state"] = state
        captured["rewrite_merge_result"] = merge_result
        return {
            "fact": candidate["fact"],
            "subjective_appraisal": candidate["subjective_appraisal"],
            "relationship_signal": candidate["relationship_signal"],
        }

    async def update_semantics(
        unit_id: str,
        semantics: dict,
        **kwargs: Any,
    ) -> None:
        captured["updated_unit_id"] = unit_id
        captured["updated_semantics"] = semantics
        captured["lifecycle_fields"] = kwargs["lifecycle_fields"]
        captured["merge_history"] = kwargs["merge_history_entry"]

    async def reconcile_schedule(unit: dict, **kwargs: Any) -> dict[str, str]:
        captured["reconciled_unit"] = unit
        captured["reconcile_timestamp"] = kwargs["storage_timestamp_utc"]
        return {"status": "scheduled", "unit_id": unit["unit_id"]}

    monkeypatch.setattr(
        memory_units_module,
        "retrieve_memory_unit_merge_candidates",
        retrieve_candidates,
    )
    monkeypatch.setattr(
        memory_units_module,
        "_judge_memory_unit_merge",
        judge_merge,
    )
    monkeypatch.setattr(memory_units_module, "_rewrite_memory_unit", rewrite_unit)
    monkeypatch.setattr(
        memory_units_module,
        "update_user_memory_unit_semantics",
        update_semantics,
    )
    monkeypatch.setattr(
        memory_units_module,
        "reconcile_active_commitment_calendar_schedule",
        reconcile_schedule,
    )

    result = await memory_units_module.process_memory_unit_candidate(
        _consolidation_state(),
        _candidate(due_at=updated_due_at),
    )

    assert result["unit_id"] == "commitment-existing"
    assert captured["updated_unit_id"] == "commitment-existing"
    assert captured["lifecycle_fields"] == {"due_at": updated_due_at}
    assert captured["reconcile_timestamp"] == NOW_UTC
    assert captured["reconciled_unit"] == {
        "unit_id": "commitment-existing",
        "global_user_id": "user-1",
        "unit_type": "active_commitment",
        "status": "active",
        "fact": "The character accepted a follow-up.",
        "subjective_appraisal": "The commitment should be checked later.",
        "relationship_signal": "Follow through at the due time.",
        "due_at": updated_due_at,
        "updated_at": NOW_UTC,
    }


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
        "cognition_source": {
            "source_kind": "scheduler_event",
            "source_id": "run-commitment-1",
            "occurred_at": DUE_AT,
            "semantic_summary": (
                f"scheduled commitment was skipped: {expected_reason}"
            ),
        },
    }
    assert built_cases == []
