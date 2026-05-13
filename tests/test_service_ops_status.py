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
            "reflection_cycle": {
                "last_event_at": "2026-05-14T00:02:00+00:00",
                "last_status": "succeeded",
            },
            "self_cognition": {
                "last_event_at": "2026-05-14T00:03:00+00:00",
                "last_status": "disabled",
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
    monkeypatch.setattr(
        service_module,
        "REFLECTION_WORKER_INTERVAL_SECONDS",
        900,
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
        "_reflection_worker_handle",
        SimpleNamespace(task=_FakeTask(done=False)),
    )
    monkeypatch.setattr(
        service_module,
        "_self_cognition_worker_handle",
        None,
    )

    response = await service_module.ops_runtime_status(window_hours=6)
    payload = response.model_dump()

    assert payload["status"] == "ok"
    assert payload["window_hours"] == 6
    assert payload["config"] == {
        "reflection_cycle_enabled": True,
        "self_cognition_enabled": False,
        "reflection_worker_interval_seconds": 900,
        "self_cognition_worker_interval_seconds": 3600,
        "self_cognition_max_cases_per_tick": 3,
    }
    assert payload["workers"]["reflection_cycle"]["task_alive"] is True
    assert payload["workers"]["reflection_cycle"]["enabled"] is True
    assert payload["workers"]["self_cognition"]["task_alive"] is False
    assert payload["workers"]["self_cognition"]["enabled"] is False
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

    response = await service_module.ops_self_cognition_stats(window_hours=12)
    payload = response.model_dump()

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
