"""Tests for trusted operator status endpoints and event-log export."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from kazusa_ai_chatbot import service as service_module
from scripts import export_event_log as export_module


class _FakeTask:
    """Small task stand-in exposing the ``done`` method used by service."""

    def __init__(self, *, done: bool) -> None:
        """Store whether the fake task should be considered complete."""

        self._done = done

    def done(self) -> bool:
        """Return the configured completion state."""

        return_value = self._done
        return return_value


def _runtime_status_payload() -> dict[str, object]:
    """Build a sanitized event-log runtime status payload."""

    payload = {
        "status": "ok",
        "generated_at": "2026-05-14T00:00:00+00:00",
        "window_hours": 6,
        "process": {
            "last_event_at": "2026-05-14T00:01:00+00:00",
            "last_status": "ready",
        },
        "workers": {
            "calendar_scheduler": {
                "last_event_at": "2026-05-14T00:01:30+00:00",
                "last_status": "succeeded",
            },
            "reflection_cycle": {
                "last_event_at": "2026-05-14T00:02:00+00:00",
                "last_status": "succeeded",
            },
            "self_cognition": {
                "last_event_at": "2026-05-14T00:03:00+00:00",
                "last_status": "disabled",
            },
            "background_work": {
                "last_event_at": "2026-05-14T00:03:30+00:00",
                "last_status": "succeeded",
            },
        },
        "semantic_descriptors": {
            "worker_error_level": "none",
        },
    }
    return payload


def _reflection_stats_payload() -> dict[str, object]:
    """Build a sanitized reflection stats payload."""

    payload = {
        "status": "ok",
        "generated_at": "2026-05-14T00:04:00+00:00",
        "window_hours": 12,
        "counts": {
            "succeeded": 4,
            "failed": 0,
            "skipped": 1,
        },
        "latest": {
            "event_id": "event-reflection-1",
            "run_id": "run-reflection-1",
            "occurred_at": "2026-05-14T00:05:00+00:00",
            "status": "succeeded",
        },
        "semantic_descriptors": {
            "reflection_health": "healthy",
        },
    }
    return payload


def _self_cognition_stats_payload() -> dict[str, object]:
    """Build a sanitized self-cognition stats payload."""

    payload = {
        "status": "ok",
        "generated_at": "2026-05-14T00:06:00+00:00",
        "window_hours": 12,
        "counts": {
            "runs": 2,
            "dispatch_accepted": 1,
        },
        "latest": {
            "event_id": "event-self-1",
            "run_id": "run-self-1",
            "trigger_id": "trigger-self-1",
            "attempt_id": "attempt-self-1",
            "occurred_at": "2026-05-14T00:07:00+00:00",
            "status": "succeeded",
        },
        "semantic_descriptors": {
            "self_cognition_liveness": "active_with_handoff",
        },
    }
    return payload


@pytest.mark.asyncio
async def test_ops_runtime_status_merges_config_and_worker_liveness(
    monkeypatch,
) -> None:
    """Runtime endpoint should add service config without raw runtime data."""

    build_runtime_status = AsyncMock(return_value=_runtime_status_payload())
    monkeypatch.setattr(
        service_module.event_logging,
        "build_runtime_status",
        build_runtime_status,
    )
    monkeypatch.setattr(service_module, "REFLECTION_CYCLE_ENABLED", True)
    monkeypatch.setattr(service_module, "SELF_COGNITION_ENABLED", False)
    monkeypatch.setattr(service_module, "BACKGROUND_WORK_WORKER_ENABLED", True)
    monkeypatch.setattr(service_module, "CALENDAR_SCHEDULER_ENABLED", True)
    monkeypatch.setattr(
        service_module,
        "CALENDAR_SCHEDULER_POLL_INTERVAL_SECONDS",
        30,
    )
    monkeypatch.setattr(
        service_module,
        "CALENDAR_SCHEDULER_CLAIM_LIMIT",
        4,
    )
    monkeypatch.setattr(
        service_module,
        "CALENDAR_SCHEDULER_LEASE_SECONDS",
        120,
    )
    monkeypatch.setattr(
        service_module,
        "CALENDAR_SCHEDULER_MAX_ATTEMPTS",
        5,
    )
    monkeypatch.setattr(
        service_module,
        "REFLECTION_WORKER_INTERVAL_SECONDS",
        900,
    )
    monkeypatch.setattr(
        service_module,
        "REFLECTION_PHASE_MIN_SLOT_SPACING_SECONDS",
        60,
    )
    monkeypatch.setattr(
        service_module,
        "REFLECTION_PHASE_MAX_SLOTS_PER_PERIOD",
        3,
    )
    monkeypatch.setattr(
        service_module,
        "REFLECTION_PHASE_GROUPS_PER_SLOT",
        1,
    )
    monkeypatch.setattr(
        service_module,
        "SELF_COGNITION_WORKER_INTERVAL_SECONDS",
        3600,
    )
    monkeypatch.setattr(
        service_module,
        "SELF_COGNITION_MAX_CASES_PER_TICK",
        3,
    )
    monkeypatch.setattr(
        service_module,
        "BACKGROUND_WORK_WORKER_INTERVAL_SECONDS",
        45,
    )
    monkeypatch.setattr(
        service_module,
        "BACKGROUND_WORK_WORKER_CLAIM_LIMIT",
        2,
    )
    monkeypatch.setattr(
        service_module,
        "BACKGROUND_WORK_WORKER_LEASE_SECONDS",
        180,
    )
    monkeypatch.setattr(
        service_module,
        "BACKGROUND_WORK_WORKER_MAX_ATTEMPTS",
        4,
    )
    monkeypatch.setattr(
        service_module,
        "BACKGROUND_WORK_INPUT_CHAR_LIMIT",
        8000,
    )
    monkeypatch.setattr(
        service_module,
        "BACKGROUND_WORK_OUTPUT_CHAR_LIMIT",
        3000,
    )
    monkeypatch.setattr(
        service_module,
        "_reflection_worker_handle",
        SimpleNamespace(task=_FakeTask(done=False)),
    )
    monkeypatch.setattr(
        service_module,
        "_calendar_worker_handle",
        SimpleNamespace(task=_FakeTask(done=False)),
    )
    monkeypatch.setattr(
        service_module,
        "_self_cognition_worker_handle",
        None,
    )
    monkeypatch.setattr(
        service_module,
        "_background_work_worker_handle",
        SimpleNamespace(task=_FakeTask(done=False)),
    )

    response = await service_module.ops_runtime_status(window_hours=6)
    payload = response.model_dump()

    assert payload["status"] == "ok"
    assert payload["window_hours"] == 6
    assert payload["config"] == {
        "calendar_scheduler_enabled": True,
        "calendar_scheduler_poll_interval_seconds": 30,
        "calendar_scheduler_claim_limit": 4,
        "calendar_scheduler_lease_seconds": 120,
        "calendar_scheduler_max_attempts": 5,
        "reflection_cycle_enabled": True,
        "self_cognition_enabled": False,
        "background_work_worker_enabled": True,
        "reflection_worker_interval_seconds": 900,
        "reflection_phase_min_slot_spacing_seconds": 60,
        "reflection_phase_max_slots_per_period": 3,
        "reflection_phase_groups_per_slot": 1,
        "self_cognition_worker_interval_seconds": 3600,
        "self_cognition_max_cases_per_tick": 3,
        "background_work_worker_interval_seconds": 45,
        "background_work_worker_claim_limit": 2,
        "background_work_worker_lease_seconds": 180,
        "background_work_worker_max_attempts": 4,
        "background_work_input_char_limit": 8000,
        "background_work_output_char_limit": 3000,
    }
    assert payload["workers"]["calendar_scheduler"]["task_alive"] is True
    assert payload["workers"]["calendar_scheduler"]["enabled"] is True
    assert payload["workers"]["reflection_cycle"]["task_alive"] is True
    assert payload["workers"]["reflection_cycle"]["enabled"] is True
    assert payload["workers"]["self_cognition"]["task_alive"] is False
    assert payload["workers"]["self_cognition"]["enabled"] is False
    assert payload["workers"]["background_work"]["task_alive"] is True
    assert payload["workers"]["background_work"]["enabled"] is True
    assert "private" not in json.dumps(payload, sort_keys=True)
    build_runtime_status.assert_awaited_once_with(window_hours=6)


@pytest.mark.asyncio
async def test_ops_reflection_stats_returns_aggregate_payload(monkeypatch) -> None:
    """Reflection endpoint should return bounded aggregate stats."""

    build_reflection_stats = AsyncMock(return_value=_reflection_stats_payload())
    monkeypatch.setattr(
        service_module.event_logging,
        "build_reflection_stats",
        build_reflection_stats,
    )

    response = await service_module.ops_reflection_stats(window_hours=12)
    payload = response.model_dump()

    assert payload["counts"] == {
        "succeeded": 4,
        "failed": 0,
        "skipped": 1,
    }
    assert payload["latest"]["run_id"] == "run-reflection-1"
    assert payload["semantic_descriptors"]["reflection_health"] == "healthy"
    build_reflection_stats.assert_awaited_once_with(window_hours=12)


@pytest.mark.asyncio
async def test_ops_self_cognition_stats_returns_aggregate_payload(
    monkeypatch,
) -> None:
    """Self-cognition endpoint should return aggregate run and handoff counts."""

    build_self_cognition_stats = AsyncMock(
        return_value=_self_cognition_stats_payload(),
    )
    monkeypatch.setattr(
        service_module.event_logging,
        "build_self_cognition_stats",
        build_self_cognition_stats,
    )
    monkeypatch.setattr(service_module, "SELF_COGNITION_ENABLED", False)
    monkeypatch.setattr(
        service_module,
        "_self_cognition_worker_handle",
        None,
    )

    response = await service_module.ops_self_cognition_stats(window_hours=12)
    payload = response.model_dump()

    assert payload["enabled"] is False
    assert payload["task_alive"] is False
    assert payload["counts"] == {
        "runs": 2,
        "dispatch_accepted": 1,
    }
    assert payload["latest"]["trigger_id"] == "trigger-self-1"
    assert (
        payload["semantic_descriptors"]["self_cognition_liveness"]
        == "active_with_handoff"
    )
    build_self_cognition_stats.assert_awaited_once_with(window_hours=12)


@pytest.mark.asyncio
async def test_export_event_log_writes_sanitized_aggregate_document(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Export command helper should write aggregate data and snapshot refs."""

    monkeypatch.setattr(
        export_module.event_logging,
        "build_runtime_status",
        AsyncMock(return_value=_runtime_status_payload()),
    )
    monkeypatch.setattr(
        export_module.event_logging,
        "build_reflection_stats",
        AsyncMock(return_value=_reflection_stats_payload()),
    )
    monkeypatch.setattr(
        export_module.event_logging,
        "build_self_cognition_stats",
        AsyncMock(return_value=_self_cognition_stats_payload()),
    )
    monkeypatch.setattr(
        export_module.event_logging,
        "write_analysis_snapshot",
        AsyncMock(return_value={
            "accepted": True,
            "event_id": "snapshot-1",
            "status": "recorded",
            "reason": "",
        }),
    )
    output_path = tmp_path / "event_log.json"

    written_path = await export_module.export_event_log(
        window_hours=3,
        output_path=output_path,
    )

    assert written_path == output_path
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["window_hours"] == 3
    assert payload["snapshot_write"]["event_id"] == "snapshot-1"
    assert payload["runtime_status"]["semantic_descriptors"] == {
        "worker_error_level": "none",
    }
    serialized = json.dumps(payload, sort_keys=True)
    assert "private message" not in serialized
    assert "secret" not in serialized
