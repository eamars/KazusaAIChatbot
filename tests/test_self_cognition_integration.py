"""Deterministic integration tests for the self-cognition runtime boundary."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from kazusa_ai_chatbot.dispatcher.task import DispatchResult, Task
from kazusa_ai_chatbot.db import user_memory_units as memory_units_module
from kazusa_ai_chatbot.self_cognition import artifacts, models, sources
from kazusa_ai_chatbot.self_cognition import tracking, worker
from kazusa_ai_chatbot.self_cognition.handoff import dispatch_action_candidate


def _target_scope() -> dict[str, str | None]:
    scope = {
        "platform": "qq",
        "platform_channel_id": "673225019",
        "channel_type": "private",
        "user_id": "673225019",
    }
    return scope


def _commitment_case() -> dict[str, Any]:
    case = {
        "case_name": models.CASE_COMMITMENT_PAST_DUE,
        "case_id": "commitment:promise-001",
        "idle_timestamp": "2026-05-13T00:30:00+00:00",
        "last_evidence_timestamp": "2026-05-13T00:00:00+00:00",
        "trigger_kind": models.TRIGGER_ACTIVE_COMMITMENT_DUE_CHECK,
        "semantic_due_state": models.DUE_STATE_PAST_DUE,
        "actionability": "past_due_commitment_contact_socially_available",
        "target_scope": _target_scope(),
        "source_refs": [
            {
                "source_kind": "user_memory_unit",
                "source_id": "promise-001",
                "due_at": "2026-05-13T00:00:00+00:00",
                "summary": "A promised follow-up is due.",
            }
        ],
        "visible_context": [
            {
                "role": "user",
                "body_text": "Please check back after the appointment.",
                "timestamp": "2026-05-12T23:50:00+00:00",
            }
        ],
    }
    return case


def _action_attempt(case: dict[str, Any], *, status: str) -> dict[str, Any]:
    source_ref = case["source_refs"][0]
    idempotency_key = tracking.build_idempotency_key(
        source_ref["source_kind"],
        source_ref["source_id"],
        source_ref["due_at"],
        case["target_scope"],
        models.ACTION_KIND_SEND_MESSAGE,
    )
    attempt = {
        "attempt_id": "self_cognition_attempt:promise-001",
        "run_id": "self_cognition_run:promise-001",
        "trigger_id": "self_cognition_trigger:promise-001",
        "source_kind": source_ref["source_kind"],
        "source_id": source_ref["source_id"],
        "target_scope": case["target_scope"],
        "action_kind": models.ACTION_KIND_SEND_MESSAGE,
        "due_at": source_ref["due_at"],
        "idempotency_key": idempotency_key,
        "status": status,
    }
    return attempt


def _action_candidate(attempt: dict[str, Any]) -> dict[str, Any]:
    candidate = {
        "attempt_id": attempt["attempt_id"],
        "target_platform": "qq",
        "target_channel": "673225019",
        "target_channel_type": "private",
        "text": "Checking in now.",
        "execute_at": None,
        "dispatch_shape": models.ACTION_KIND_SEND_MESSAGE,
        "production_handoff": False,
    }
    return candidate


def _write_json(path: Path, payload: dict[str, Any]) -> str:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    rendered_path = str(path)
    return rendered_path


def _case_runner_with_candidate(
    case: dict[str, Any],
    output_dir: Path,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    attempt = _action_attempt(
        case,
        status=models.ACTION_ATTEMPT_STATUS_CANDIDATE,
    )
    candidate = _action_candidate(attempt)
    paths = {
        models.ARTIFACT_ACTION_ATTEMPT: _write_json(
            output_dir / models.ARTIFACT_ACTION_ATTEMPT,
            attempt,
        ),
        models.ARTIFACT_ACTION_CANDIDATE: _write_json(
            output_dir / models.ARTIFACT_ACTION_CANDIDATE,
            candidate,
        ),
    }
    return paths


def _case_runner_with_tracking(
    case: dict[str, Any],
    output_dir: Path,
) -> dict[str, str]:
    """Build action artifacts using real tracking duplicate logic."""

    output_dir.mkdir(parents=True, exist_ok=True)
    trigger_record = tracking.build_trigger_record(case)
    existing_attempts = case.get("existing_attempts")
    if not isinstance(existing_attempts, list):
        existing_attempts = []
    action_attempt = tracking.build_action_attempt(
        case,
        trigger_record,
        [
            attempt
            for attempt in existing_attempts
            if isinstance(attempt, dict)
        ],
    )
    action_candidate = tracking.build_action_candidate(
        case,
        action_attempt,
        "Checking in now.",
    )
    paths = {
        models.ARTIFACT_ACTION_ATTEMPT: _write_json(
            output_dir / models.ARTIFACT_ACTION_ATTEMPT,
            action_attempt,
        )
    }
    if action_candidate is not None:
        paths[models.ARTIFACT_ACTION_CANDIDATE] = _write_json(
            output_dir / models.ARTIFACT_ACTION_CANDIDATE,
            action_candidate,
        )
    return paths


class _FakeDispatcher:
    def __init__(self, *, reject: bool = False) -> None:
        self.reject = reject
        self.calls: list[dict[str, Any]] = []

    async def dispatch(self, raw_calls, ctx, *, instruction: str = ""):
        self.calls.append(
            {
                "raw_calls": raw_calls,
                "ctx": ctx,
                "instruction": instruction,
            }
        )
        raw_call = raw_calls[0]
        if self.reject:
            result = DispatchResult(
                scheduled=[],
                rejected=[(raw_call, "no adapters registered")],
            )
            return result
        task = Task(
            tool=raw_call.tool,
            args=dict(raw_call.args),
            execute_at=ctx.now,
        )
        result = DispatchResult(
            scheduled=[(task, "event-001")],
            rejected=[],
        )
        return result


class _AsyncCursor:
    def __init__(self, docs: list[dict[str, Any]]) -> None:
        self._docs = iter(docs)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            row = next(self._docs)
        except StopIteration as exc:
            raise StopAsyncIteration from exc
        return row


class _FakeUserMemoryUnitsCollection:
    def __init__(self) -> None:
        self.pipeline: list[dict[str, Any]] = []

    def aggregate(self, pipeline: list[dict[str, Any]]):
        self.pipeline = pipeline
        cursor = _AsyncCursor([{"unit_id": "promise-001"}])
        return cursor


@pytest.mark.asyncio
async def test_worker_tick_loads_ledger_before_running_case(tmp_path) -> None:
    """Prior attempts from the local ledger should enter the next case run."""

    case = _commitment_case()
    prior_attempt = _action_attempt(
        case,
        status=models.ACTION_ATTEMPT_STATUS_SCHEDULED,
    )
    artifacts.append_action_attempt_ledger(tmp_path, prior_attempt)
    captured_case: dict[str, Any] = {}

    async def collect_cases(*, now: datetime, max_cases: int) -> list[dict[str, Any]]:
        assert max_cases == 3
        return [case]

    async def run_case(next_case: dict[str, Any], output_dir: Path) -> dict[str, str]:
        del output_dir
        captured_case.update(next_case)
        return {}

    result = await worker.run_self_cognition_worker_tick(
        output_root=tmp_path,
        dispatcher=None,
        now=datetime(2026, 5, 13, tzinfo=timezone.utc),
        is_primary_interaction_busy=lambda: False,
        collect_cases_func=collect_cases,
        run_case_func=run_case,
        max_cases=3,
    )

    assert result.processed_count == 1
    assert captured_case["existing_attempts"][0]["idempotency_key"] == (
        prior_attempt["idempotency_key"]
    )


@pytest.mark.asyncio
async def test_worker_tick_dispatches_candidate_through_task_dispatcher(
    tmp_path,
) -> None:
    """A live candidate should use TaskDispatcher instead of direct adapter send."""

    case = _commitment_case()
    dispatcher = _FakeDispatcher()

    async def collect_cases(*, now: datetime, max_cases: int) -> list[dict[str, Any]]:
        del now, max_cases
        return [case]

    result = await worker.run_self_cognition_worker_tick(
        output_root=tmp_path,
        dispatcher=dispatcher,
        now=datetime(2026, 5, 13, tzinfo=timezone.utc),
        is_primary_interaction_busy=lambda: False,
        collect_cases_func=collect_cases,
        run_case_func=_case_runner_with_candidate,
        max_cases=3,
    )

    ledger = artifacts.read_action_attempt_ledger(tmp_path)
    dispatch_result_path = next(
        Path(path)
        for path in result.artifact_paths
        if Path(path).name == models.ARTIFACT_DISPATCH_RESULT
    )
    dispatch_result = json.loads(dispatch_result_path.read_text(encoding="utf-8"))

    assert result.dispatched_count == 1
    assert dispatcher.calls[0]["raw_calls"][0].tool == "send_message"
    assert dispatcher.calls[0]["raw_calls"][0].args["text"] == "Checking in now."
    assert dispatch_result["status"] == "accepted"
    assert dispatch_result["scheduled_event_ids"] == ["event-001"]
    assert ledger[0]["status"] == models.ACTION_ATTEMPT_STATUS_SCHEDULED


@pytest.mark.asyncio
async def test_worker_tick_records_dispatch_rejection_without_adapter_send(
    tmp_path,
) -> None:
    """Dispatcher rejection should be tracked locally without calling adapters."""

    case = _commitment_case()
    dispatcher = _FakeDispatcher(reject=True)

    async def collect_cases(*, now: datetime, max_cases: int) -> list[dict[str, Any]]:
        del now, max_cases
        return [case]

    result = await worker.run_self_cognition_worker_tick(
        output_root=tmp_path,
        dispatcher=dispatcher,
        now=datetime(2026, 5, 13, tzinfo=timezone.utc),
        is_primary_interaction_busy=lambda: False,
        collect_cases_func=collect_cases,
        run_case_func=_case_runner_with_candidate,
        max_cases=3,
    )

    ledger = artifacts.read_action_attempt_ledger(tmp_path)

    assert result.rejected_count == 1
    assert ledger[0]["status"] == models.ACTION_ATTEMPT_STATUS_HELD
    assert ledger[0]["dispatch_status"] == "rejected"


@pytest.mark.asyncio
async def test_worker_tick_suppresses_duplicate_due_occurrence_from_ledger(
    tmp_path,
) -> None:
    """A prior ledger attempt should prevent a repeated dispatcher handoff."""

    case = _commitment_case()
    prior_attempt = _action_attempt(
        case,
        status=models.ACTION_ATTEMPT_STATUS_SCHEDULED,
    )
    artifacts.append_action_attempt_ledger(tmp_path, prior_attempt)
    dispatcher = _FakeDispatcher()

    async def collect_cases(*, now: datetime, max_cases: int) -> list[dict[str, Any]]:
        del now, max_cases
        return [case]

    def run_duplicate_case(
        next_case: dict[str, Any],
        output_dir: Path,
    ) -> dict[str, str]:
        output_dir.mkdir(parents=True, exist_ok=True)
        assert next_case["existing_attempts"][0]["idempotency_key"] == (
            prior_attempt["idempotency_key"]
        )
        duplicate = _action_attempt(
            next_case,
            status=models.ACTION_ATTEMPT_STATUS_DUPLICATE,
        )
        paths = {
            models.ARTIFACT_ACTION_ATTEMPT: _write_json(
                output_dir / models.ARTIFACT_ACTION_ATTEMPT,
                duplicate,
            )
        }
        return paths

    result = await worker.run_self_cognition_worker_tick(
        output_root=tmp_path,
        dispatcher=dispatcher,
        now=datetime(2026, 5, 13, tzinfo=timezone.utc),
        is_primary_interaction_busy=lambda: False,
        collect_cases_func=collect_cases,
        run_case_func=run_duplicate_case,
        max_cases=3,
    )

    assert result.dispatched_count == 0
    assert dispatcher.calls == []


@pytest.mark.asyncio
async def test_worker_tick_uses_ledger_updates_between_cases(tmp_path) -> None:
    """Same-tick duplicate cases should see attempts recorded earlier."""

    case = _commitment_case()
    dispatcher = _FakeDispatcher()

    async def collect_cases(
        *,
        now: datetime,
        max_cases: int,
    ) -> list[dict[str, Any]]:
        del now, max_cases
        return [case, dict(case)]

    result = await worker.run_self_cognition_worker_tick(
        output_root=tmp_path,
        dispatcher=dispatcher,
        now=datetime(2026, 5, 13, tzinfo=timezone.utc),
        is_primary_interaction_busy=lambda: False,
        collect_cases_func=collect_cases,
        run_case_func=_case_runner_with_tracking,
        max_cases=3,
    )
    ledger = artifacts.read_action_attempt_ledger(tmp_path)

    assert result.processed_count == 2
    assert result.dispatched_count == 1
    assert len(dispatcher.calls) == 1
    assert ledger[0]["status"] == models.ACTION_ATTEMPT_STATUS_SCHEDULED
    assert ledger[1]["status"] == models.ACTION_ATTEMPT_STATUS_DUPLICATE


@pytest.mark.asyncio
async def test_worker_tick_defers_when_primary_interaction_is_busy(tmp_path) -> None:
    """The idle worker should not compete with active chat work."""

    async def collect_cases(*, now: datetime, max_cases: int) -> list[dict[str, Any]]:
        raise AssertionError("busy tick should not collect cases")

    result = await worker.run_self_cognition_worker_tick(
        output_root=tmp_path,
        dispatcher=None,
        now=datetime(2026, 5, 13, tzinfo=timezone.utc),
        is_primary_interaction_busy=lambda: True,
        collect_cases_func=collect_cases,
        max_cases=3,
    )

    assert result.deferred is True
    assert result.defer_reason == "primary interaction busy"
    assert result.processed_count == 0


@pytest.mark.asyncio
async def test_dispatch_action_candidate_builds_existing_dispatch_context() -> None:
    """The handoff layer should preserve candidate target and source context."""

    case = _commitment_case()
    attempt = _action_attempt(
        case,
        status=models.ACTION_ATTEMPT_STATUS_CANDIDATE,
    )
    candidate = _action_candidate(attempt)
    dispatcher = _FakeDispatcher()

    result = await dispatch_action_candidate(
        case,
        attempt,
        candidate,
        dispatcher,
        now=datetime(2026, 5, 13, tzinfo=timezone.utc),
    )

    raw_call = dispatcher.calls[0]["raw_calls"][0]
    ctx = dispatcher.calls[0]["ctx"]

    assert raw_call.tool == "send_message"
    assert raw_call.args["target_channel"] == "673225019"
    assert ctx.source_platform == "qq"
    assert ctx.source_channel_type == "private"
    assert result["production_handoff"] is True


@pytest.mark.asyncio
async def test_active_commitment_source_builds_due_case_from_memory_unit() -> None:
    """Active commitment collection should build visible/actionable case input."""

    unit = {
        "unit_id": "promise-001",
        "global_user_id": "673225019",
        "unit_type": "active_commitment",
        "status": "active",
        "fact": "A promised follow-up is due.",
        "subjective_appraisal": "The user may expect a check-in.",
        "relationship_signal": "Following through matters.",
        "due_at": "2026-05-13T00:00:00+00:00",
        "last_seen_at": "2026-05-12T23:55:00+00:00",
        "updated_at": "2026-05-12T23:55:00+00:00",
    }
    rows = [
        {
            "platform": "qq",
            "platform_channel_id": "673225019",
            "channel_type": "private",
            "role": "user",
            "global_user_id": "673225019",
            "display_name": "User",
            "body_text": "Please check back after the appointment.",
            "timestamp": "2026-05-12T23:50:00+00:00",
        }
    ]

    async def list_commitments(*, current_timestamp: str, limit: int):
        assert current_timestamp == "2026-05-13T00:30:00+00:00"
        assert limit == 3
        return [unit]

    async def get_history(**kwargs):
        assert kwargs["global_user_id"] == "673225019"
        return rows

    async def get_profile(global_user_id: str):
        assert global_user_id == "673225019"
        return {"affinity": 600, "display_name": "User"}

    cases = await sources.collect_active_commitment_cases(
        now=datetime(2026, 5, 13, 0, 30, tzinfo=timezone.utc),
        character_profile={"name": "Character", "mood": "focused"},
        max_cases=3,
        list_active_commitments_func=list_commitments,
        get_conversation_history_func=get_history,
        get_user_profile_func=get_profile,
    )

    assert len(cases) == 1
    assert cases[0]["case_name"] == models.CASE_COMMITMENT_PAST_DUE
    assert cases[0]["target_scope"] == _target_scope()
    assert cases[0]["source_refs"][0]["source_id"] == "promise-001"
    assert cases[0]["visible_context"][0]["body_text"].startswith("Please")


@pytest.mark.asyncio
async def test_active_commitment_query_prioritizes_due_work(
    monkeypatch,
) -> None:
    """Active commitment reads should prioritize due items inside the tick cap."""

    collection = _FakeUserMemoryUnitsCollection()

    class FakeDatabase:
        user_memory_units = collection

    async def fake_get_db():
        database = FakeDatabase()
        return database

    monkeypatch.setattr(memory_units_module, "get_db", fake_get_db)

    rows = await memory_units_module.query_active_commitment_memory_units(
        current_timestamp="2026-05-13T00:30:00+00:00",
        limit=3,
    )
    pipeline = collection.pipeline

    assert rows == [{"unit_id": "promise-001"}]
    assert pipeline[0]["$match"]["due_at"] == {"$type": "string", "$ne": ""}
    assert pipeline[1]["$addFields"]["_self_cognition_due_at"] == {
        "$dateFromString": {
            "dateString": {
                "$replaceOne": {
                    "input": "$due_at",
                    "find": " ",
                    "replacement": "T",
                }
            },
            "onError": None,
            "onNull": None,
        }
    }
    assert pipeline[2] == {"$match": {"_self_cognition_due_at": {"$ne": None}}}
    assert pipeline[3]["$addFields"]["_self_cognition_due_bucket"] == {
        "$cond": [
            {
                "$lte": [
                    "$_self_cognition_due_at",
                    datetime(2026, 5, 13, 0, 30, tzinfo=timezone.utc),
                ]
            },
            memory_units_module.ACTIVE_COMMITMENT_DUE_BUCKET_READY,
            memory_units_module.ACTIVE_COMMITMENT_DUE_BUCKET_FUTURE,
        ]
    }
    assert pipeline[4]["$sort"] == {
        "_self_cognition_due_bucket": 1,
        "_self_cognition_due_at": 1,
        "last_seen_at": -1,
        "updated_at": -1,
    }
    assert pipeline[5] == {"$limit": 3}
    assert pipeline[6]["$project"]["_self_cognition_due_at"] == 0
    assert pipeline[6]["$project"]["_self_cognition_due_bucket"] == 0
