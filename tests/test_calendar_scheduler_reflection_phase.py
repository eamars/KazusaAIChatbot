"""Contract tests for calendar-backed reflection phase slots."""

from __future__ import annotations

import inspect
from datetime import datetime, timedelta, timezone

import pytest

from kazusa_ai_chatbot.config import REFLECTION_WORKER_INTERVAL_SECONDS
from kazusa_ai_chatbot.reflection_cycle import phase_scheduler
from kazusa_ai_chatbot.reflection_cycle.models import (
    ReflectionScopeInput,
    ReflectionWorkerResult,
)


PERIOD_START = datetime(1970, 1, 1, tzinfo=timezone.utc)
STORAGE_NOW = "1970-01-01T00:00:00+00:00"
OFFSET_WITHIN_PERIOD_SECONDS = 123


def test_phase_intent_maps_mechanically_to_calendar_run() -> None:
    """Calendar storage should preserve the existing phase intent shape."""

    from kazusa_ai_chatbot.calendar_scheduler import reflection_phase

    intent = _phase_intent()
    runs = reflection_phase.build_reflection_phase_calendar_runs(
        [intent],
        storage_timestamp_utc=STORAGE_NOW,
    )

    assert len(runs) == 1
    run = runs[0]
    assert run["run_id"] == intent["run_id"]
    assert run["trigger_kind"] == "reflection_phase_slot"
    assert run["due_at"] == intent["due_at"]
    assert run["idempotency_key"] == intent["idempotency_key"]
    assert run["payload"]["reflection_phase_intent"] == intent
    assert run["source_scope"] == intent["source_scope"]
    assert _parse_utc(intent["due_at"]) == (
        _parse_utc(intent["period_start_utc"])
        + timedelta(seconds=int(intent["offset_seconds"]))
    )

    run_with_scheduler_metadata = {
        **run,
        "lease_owner": "worker-a",
        "attempt_count": 2,
        "migration_source": {"collection": "scheduled_events"},
    }

    restored_intent = reflection_phase.calendar_run_to_reflection_phase_intent(
        run_with_scheduler_metadata
    )
    assert restored_intent == intent
    assert "lease_owner" not in restored_intent
    assert "attempt_count" not in restored_intent
    assert "migration_source" not in restored_intent


def test_phase_calendar_mapping_does_not_split_trigger_kinds() -> None:
    """Hourly reflection and group review remain payload actions only."""

    from kazusa_ai_chatbot.calendar_scheduler import reflection_phase

    intent = _phase_intent()
    run = reflection_phase.build_reflection_phase_calendar_runs(
        [intent],
        storage_timestamp_utc=STORAGE_NOW,
    )[0]

    assert run["trigger_kind"] == "reflection_phase_slot"
    assert run["payload"]["reflection_phase_intent"]["payload"][
        "allowed_actions"
    ] == [
        "reflection_hourly_slot",
        "group_self_cognition_review",
    ]
    assert "reflection_hourly_slot" != run["trigger_kind"]
    assert "group_self_cognition_review" != run["trigger_kind"]


def test_calendar_reflection_module_has_no_side_collection() -> None:
    """The calendar path must not introduce a side reflection collection."""

    from kazusa_ai_chatbot.calendar_scheduler import reflection_phase

    source = inspect.getsource(reflection_phase)
    forbidden_collection = "reflection" + "_phase" + "_runs"

    assert forbidden_collection not in source


@pytest.mark.asyncio
async def test_materialize_phase_period_snapshots_at_period_start(
    monkeypatch,
) -> None:
    """Calendar materialization should snapshot eligible scopes per period."""

    from kazusa_ai_chatbot.calendar_scheduler import reflection_phase

    captured_collect_kwargs: dict[str, object] = {}
    upserted_runs: list[dict] = []

    class _InputSet:
        selected_scopes = [_scope("scope-a")]

    class _Repository:
        async def upsert_calendar_run(self, run: dict) -> object:
            upserted_runs.append(run)
            return object()

    async def _collect_reflection_inputs(**kwargs) -> _InputSet:
        captured_collect_kwargs.update(kwargs)
        input_set = _InputSet()
        return input_set

    monkeypatch.setattr(
        reflection_phase,
        "collect_reflection_inputs",
        _collect_reflection_inputs,
        raising=False,
    )

    summary = await reflection_phase.materialize_reflection_phase_period(
        period_start_utc=PERIOD_START,
        storage_timestamp_utc=STORAGE_NOW,
        repository=_Repository(),
    )

    assert captured_collect_kwargs["now"] == PERIOD_START
    assert captured_collect_kwargs["allow_fallback"] is False
    assert upserted_runs[0]["trigger_kind"] == "reflection_phase_slot"
    assert upserted_runs[0]["period_start_utc"] == (
        "1970-01-01T00:00:00+00:00"
    )
    assert summary == {
        "materialized_count": 1,
        "run_ids": [upserted_runs[0]["run_id"]],
    }


@pytest.mark.asyncio
async def test_materialize_phase_period_floors_unaligned_tick_to_boundary(
    monkeypatch,
) -> None:
    """A worker poll inside a period should not create a new period id."""

    from kazusa_ai_chatbot.calendar_scheduler import reflection_phase

    captured_collect_kwargs: dict[str, object] = {}
    upserted_runs: list[dict] = []
    expected_period_start = PERIOD_START + timedelta(
        seconds=REFLECTION_WORKER_INTERVAL_SECONDS,
    )
    offset_seconds = min(
        OFFSET_WITHIN_PERIOD_SECONDS,
        REFLECTION_WORKER_INTERVAL_SECONDS - 1,
    )
    unaligned_tick = expected_period_start + timedelta(seconds=offset_seconds)

    class _InputSet:
        selected_scopes = [_scope("scope-a")]

    class _Repository:
        async def upsert_calendar_run(self, run: dict) -> object:
            upserted_runs.append(run)
            return object()

    async def _collect_reflection_inputs(**kwargs) -> _InputSet:
        captured_collect_kwargs.update(kwargs)
        input_set = _InputSet()
        return input_set

    monkeypatch.setattr(
        reflection_phase,
        "collect_reflection_inputs",
        _collect_reflection_inputs,
        raising=False,
    )

    await reflection_phase.materialize_reflection_phase_period(
        period_start_utc=unaligned_tick,
        storage_timestamp_utc=STORAGE_NOW,
        repository=_Repository(),
    )

    assert captured_collect_kwargs["now"] == expected_period_start
    assert upserted_runs[0]["period_start_utc"] == (
        expected_period_start.isoformat()
    )
    assert upserted_runs[0]["due_at"] == expected_period_start.isoformat()


@pytest.mark.asyncio
async def test_reflection_phase_calendar_handler_uses_execution_seam() -> None:
    """Claimed calendar runs should execute through the reflection seam."""

    from kazusa_ai_chatbot.calendar_scheduler import reflection_phase

    intent = _phase_intent()
    run = reflection_phase.build_reflection_phase_calendar_runs(
        [intent],
        storage_timestamp_utc=STORAGE_NOW,
    )[0]
    run["lease_owner"] = "calendar-worker"
    captured: dict[str, object] = {}

    async def _execute_phase_intent(**kwargs) -> list[ReflectionWorkerResult]:
        captured.update(kwargs)
        result = ReflectionWorkerResult(
            run_kind="reflection_phase_slot",
            dry_run=False,
            processed_count=1,
            succeeded_count=1,
            run_ids=["hourly-run-1"],
        )
        return [result]

    handler_result = await reflection_phase.handle_reflection_phase_calendar_run(
        run,
        now=PERIOD_START,
        dry_run=False,
        is_primary_interaction_busy=lambda: False,
        adapter_registry_provider=None,
        execute_phase_intent_func=_execute_phase_intent,
    )

    assert captured["intent"] == intent
    assert "lease_owner" not in captured["intent"]
    assert captured["now"] == PERIOD_START
    assert handler_result == {
        "status": "completed",
        "run_kind": "reflection_phase_slot",
        "processed_count": 1,
        "succeeded_count": 1,
        "failed_count": 0,
        "skipped_count": 0,
        "run_ids": ["hourly-run-1"],
    }


@pytest.mark.asyncio
async def test_calendar_provider_daily_readiness_uses_calendar_runs() -> None:
    """Daily readiness should derive expected hourly ids from calendar runs."""

    from kazusa_ai_chatbot.calendar_scheduler import reflection_phase

    intent = _phase_intent()
    run = reflection_phase.build_reflection_phase_calendar_runs(
        [intent],
        storage_timestamp_utc=STORAGE_NOW,
    )[0]
    captured: dict[str, object] = {}
    channel_scope = _scope("scope-a")

    class _Repository:
        async def list_reflection_phase_slot_calendar_runs_for_character_local_date(
            self,
            *,
            character_local_date: str,
        ) -> list[dict]:
            captured["character_local_date"] = character_local_date
            return [run]

    async def _collect_phase_scope_input(**kwargs) -> ReflectionScopeInput:
        captured["intent"] = kwargs["intent"]
        captured["now"] = kwargs["now"]
        return channel_scope

    def _expected_hourly_run_ids_for_scope(**kwargs) -> list[str]:
        captured["channel_scope"] = kwargs["channel_scope"]
        captured["expected_date"] = kwargs["character_local_date"]
        captured["expected_now"] = kwargs["now"]
        return ["hourly-run-1"]

    provider = reflection_phase.CalendarReflectionPhaseRunProvider(
        repository=_Repository(),
        collect_phase_scope_input_func=_collect_phase_scope_input,
        expected_hourly_run_ids_func=_expected_hourly_run_ids_for_scope,
    )

    rows = await provider.expected_hourly_runs_for_character_local_date(
        character_local_date="1970-01-01",
    )

    assert captured["character_local_date"] == "1970-01-01"
    assert captured["intent"] == intent
    assert captured["now"] == PERIOD_START
    assert captured["channel_scope"] == channel_scope
    assert captured["expected_date"] == "1970-01-01"
    assert captured["expected_now"] == PERIOD_START
    assert len(rows) == 1
    assert rows[0].channel_scope == channel_scope
    assert rows[0].expected_run_ids == ["hourly-run-1"]


def _phase_intent() -> phase_scheduler.ReflectionPhaseRunIntent:
    intents = phase_scheduler.build_phase_run_intents(
        period_start_utc=PERIOD_START,
        eligible_scopes=[_scope("scope-a")],
        phase_period_seconds=900,
        max_slots_per_period=3,
        min_slot_spacing_seconds=60,
        prompt_version="readonly_reflection_v1",
    )
    return intents[0]


def _parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value).astimezone(timezone.utc)


def _scope(scope_ref: str) -> ReflectionScopeInput:
    return ReflectionScopeInput(
        scope_ref=scope_ref,
        platform="qq",
        platform_channel_id="group-1",
        channel_type="group",
        assistant_message_count=1,
        user_message_count=1,
        total_message_count=2,
        first_timestamp="1970-01-01T00:00:00+00:00",
        last_timestamp="1970-01-01T00:01:00+00:00",
        messages=[],
    )
