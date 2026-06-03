"""Contract tests for calendar scheduler document builders."""

from __future__ import annotations

import json


CREATED_AT = "2026-06-04T00:00:00+00:00"
DUE_AT = "2026-06-04T00:15:00+00:00"


def _source_scope() -> dict[str, str]:
    return {
        "platform": "qq",
        "platform_channel_id": "group-1",
        "channel_type": "group",
        "global_user_id": "user-1",
    }


def _future_payload() -> dict:
    return {
        "episode_type": "self_cognition",
        "continuation_objective": "Re-check whether a natural pause appeared.",
        "source_action_attempt_id": "action_attempt:future-123",
        "source_refs": [
            {
                "ref_kind": "cognitive_episode",
                "ref_id": "episode-123",
                "owner": "cognition",
                "relationship": "basis",
                "evidence_refs": [],
            }
        ],
        "continuation": {
            "mode": "scheduled_followup",
            "episode_type": "self_cognition",
            "max_depth": 1,
            "include_result_as": "scheduled_event",
        },
    }


def test_one_time_schedule_builder_uses_closed_trigger_contract() -> None:
    """Schedules should store typed trigger metadata, not callbacks."""

    from kazusa_ai_chatbot.calendar_scheduler import models

    schedule = models.build_one_time_calendar_schedule(
        trigger_kind=models.TRIGGER_FUTURE_COGNITION,
        due_at=DUE_AT,
        payload=_future_payload(),
        source_scope=_source_scope(),
        idempotency_key="future_cognition:action_attempt:future-123",
        storage_timestamp_utc=CREATED_AT,
    )
    repeat_schedule = models.build_one_time_calendar_schedule(
        trigger_kind=models.TRIGGER_FUTURE_COGNITION,
        due_at=DUE_AT,
        payload=_future_payload(),
        source_scope=_source_scope(),
        idempotency_key="future_cognition:action_attempt:future-123",
        storage_timestamp_utc=CREATED_AT,
    )
    serialized = json.dumps(schedule, sort_keys=True)

    assert schedule == repeat_schedule
    assert schedule["schema_version"] == "calendar_schedule.v1"
    assert schedule["trigger_kind"] == models.TRIGGER_FUTURE_COGNITION
    assert schedule["owner"] == "calendar_scheduler"
    assert schedule["recurrence"] == {"kind": "once"}
    assert schedule["status"] == models.SCHEDULE_STATUS_ACTIVE
    assert schedule["next_run_at"] == DUE_AT
    assert schedule["created_at"] == CREATED_AT
    assert schedule["updated_at"] == CREATED_AT
    assert schedule["payload"] == _future_payload()
    assert schedule["source_scope"] == _source_scope()

    for forbidden in (
        "callback",
        "callable",
        "python_path",
        "adapter_send",
        "send_message",
        "raw_channel",
        "mongodb",
        "credential",
    ):
        assert forbidden not in serialized


def test_calendar_run_builder_is_due_run_not_visible_message() -> None:
    """Runs should execute fresh cognition handlers, not delayed text."""

    from kazusa_ai_chatbot.calendar_scheduler import models

    schedule = models.build_one_time_calendar_schedule(
        trigger_kind=models.TRIGGER_FUTURE_COGNITION,
        due_at=DUE_AT,
        payload=_future_payload(),
        source_scope=_source_scope(),
        idempotency_key="future_cognition:action_attempt:future-123",
        storage_timestamp_utc=CREATED_AT,
    )

    run = models.build_calendar_run_from_schedule(
        schedule,
        due_at=DUE_AT,
        storage_timestamp_utc=CREATED_AT,
    )
    repeat_run = models.build_calendar_run_from_schedule(
        schedule,
        due_at=DUE_AT,
        storage_timestamp_utc=CREATED_AT,
    )

    assert run == repeat_run
    assert run["schema_version"] == "calendar_run.v1"
    assert run["owner"] == "calendar_scheduler"
    assert run["schedule_id"] == schedule["schedule_id"]
    assert run["trigger_kind"] == models.TRIGGER_FUTURE_COGNITION
    assert run["status"] == models.RUN_STATUS_PENDING
    assert run["due_at"] == DUE_AT
    assert run["payload"] == schedule["payload"]
    assert run["source_scope"] == schedule["source_scope"]
    assert run["attempt_count"] == 0
    assert run["lease_owner"] == ""
    assert run["lease_expires_at"] == ""
    assert run["period_start_utc"] is None
    assert run["slot_index"] is None
    assert run["offset_seconds"] is None
    assert "text" not in run["payload"]


def test_trigger_kind_roster_is_explicit_and_closed() -> None:
    """The calendar is a typed scheduler, not a generic job runner."""

    from kazusa_ai_chatbot.calendar_scheduler import models

    assert models.CALENDAR_TRIGGER_KINDS == {
        models.TRIGGER_FUTURE_COGNITION,
        models.TRIGGER_COMMITMENT_DUE_COGNITION,
        models.TRIGGER_REFLECTION_PHASE_SLOT,
        models.TRIGGER_RECURRING_SELF_CHECK,
    }
    assert "send_message" not in models.CALENDAR_TRIGGER_KINDS
    assert "reflection_hourly_slot" not in models.CALENDAR_TRIGGER_KINDS
    assert "group_self_cognition_review" not in models.CALENDAR_TRIGGER_KINDS
