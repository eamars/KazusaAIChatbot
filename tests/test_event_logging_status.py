"""Tests for generic event-log status builders and snapshots."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

import kazusa_ai_chatbot.event_logging.snapshots as snapshots_module
import kazusa_ai_chatbot.event_logging.status as status_module
from kazusa_ai_chatbot import event_logging
from kazusa_ai_chatbot.event_logging.models import (
    llm_parse_stability_label,
    reflection_health_label,
    self_cognition_liveness_label,
    worker_error_level_label,
)


def test_semantic_descriptor_labels() -> None:
    assert reflection_health_label(failed_count=0, succeeded_count=3) == "healthy"
    assert reflection_health_label(failed_count=2, succeeded_count=1) == "mixed"
    assert self_cognition_liveness_label(run_count=0, dispatch_count=0) == "inactive"
    assert self_cognition_liveness_label(run_count=2, dispatch_count=1) == "active_with_handoff"
    assert llm_parse_stability_label(failed_count=0, repaired_count=1) == "watch"
    assert worker_error_level_label(error_count=0) == "none"
    assert worker_error_level_label(error_count=12) == "high"


@pytest.mark.asyncio
async def test_build_runtime_status_uses_bounded_latest_events(monkeypatch) -> None:
    count_events = AsyncMock(return_value=2)
    find_events = AsyncMock(side_effect=[
        [{"occurred_at": "2026-05-14T00:00:00+00:00", "status": "startup"}],
        [{"occurred_at": "2026-05-14T00:01:00+00:00", "status": "succeeded"}],
        [{"occurred_at": "2026-05-14T00:02:00+00:00", "status": "disabled"}],
        [{"occurred_at": "2026-05-14T00:03:00+00:00", "status": "succeeded"}],
    ])
    monkeypatch.setattr(status_module.repository, "count_events", count_events)
    monkeypatch.setattr(status_module.repository, "find_events", find_events)

    status = await event_logging.build_runtime_status(window_hours=6)

    assert status["status"] == "ok"
    assert status["workers"]["reflection_cycle"]["last_status"] == "succeeded"
    assert status["workers"]["self_cognition"]["last_status"] == "disabled"
    assert status["workers"]["background_work"]["last_status"] == "succeeded"
    assert status["semantic_descriptors"]["worker_error_level"] == "low"
    assert find_events.await_count == 4
    assert count_events.await_count == 1


@pytest.mark.asyncio
async def test_build_reflection_and_self_cognition_stats(monkeypatch) -> None:
    count_events = AsyncMock(side_effect=[4, 1, 2, 3, 0])
    find_events = AsyncMock(return_value=[{
        "event_id": "event-1",
        "run_id": "run-1",
        "trigger_id": "trigger-1",
        "occurred_at": "2026-05-14T00:03:00+00:00",
        "status": "succeeded",
    }])
    monkeypatch.setattr(status_module.repository, "count_events", count_events)
    monkeypatch.setattr(status_module.repository, "find_events", find_events)

    reflection = await event_logging.build_reflection_stats(window_hours=24)
    self_cognition = await event_logging.build_self_cognition_stats(window_hours=24)

    assert reflection["counts"] == {"succeeded": 4, "failed": 1, "skipped": 2}
    assert reflection["semantic_descriptors"]["reflection_health"] == "mixed"
    assert self_cognition["counts"] == {"runs": 3, "dispatch_accepted": 0}
    assert self_cognition["semantic_descriptors"]["self_cognition_liveness"] == "active_internal_only"


@pytest.mark.asyncio
async def test_write_analysis_snapshot_persists_semantic_descriptors(monkeypatch) -> None:
    captured: dict[str, object] = {}

    async def write_snapshot(document):
        captured.update(document)
        return str(document["snapshot_id"])

    monkeypatch.setattr(
        snapshots_module,
        "build_snapshot_source_counts",
        AsyncMock(return_value={
            "reflection_succeeded": 2,
            "reflection_failed": 0,
            "self_cognition_runs": 0,
            "self_cognition_dispatch_accepted": 0,
            "llm_failed": 0,
            "llm_repaired": 1,
            "worker_errors": 0,
        }),
    )
    monkeypatch.setattr(snapshots_module.repository, "write_snapshot", write_snapshot)

    result = await event_logging.write_analysis_snapshot(window_hours=12)

    assert result["accepted"] is True
    assert captured["snapshot_kind"] == "event_log_snapshot"
    assert captured["semantic_descriptors"]["llm_parse_stability"] == "watch"
    assert "cognition_v2_summary" not in captured
