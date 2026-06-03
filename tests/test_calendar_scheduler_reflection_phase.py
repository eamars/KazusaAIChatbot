"""Contract tests for calendar-backed reflection phase slots."""

from __future__ import annotations

import inspect
from datetime import datetime, timedelta, timezone

from kazusa_ai_chatbot.reflection_cycle import phase_scheduler
from kazusa_ai_chatbot.reflection_cycle.models import ReflectionScopeInput


PERIOD_START = datetime(1970, 1, 1, tzinfo=timezone.utc)
STORAGE_NOW = "1970-01-01T00:00:00+00:00"


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
    """The calendar path must not introduce reflection_phase_runs storage."""

    from kazusa_ai_chatbot.calendar_scheduler import reflection_phase

    source = inspect.getsource(reflection_phase)

    assert "reflection_phase_runs" not in source


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
