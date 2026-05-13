"""Tests for the recent ops status CLI helper."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from scripts import fetch_ops_status as fetch_module


def _runtime_status_payload() -> dict[str, object]:
    """Build a sanitized runtime status payload for CLI tests."""

    payload = {
        "status": "ok",
        "generated_at": "2026-05-14T00:00:00+00:00",
        "window_hours": 4,
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
    """Build a sanitized reflection stats payload for CLI tests."""

    payload = {
        "status": "ok",
        "generated_at": "2026-05-14T00:04:00+00:00",
        "window_hours": 4,
        "counts": {
            "succeeded": 3,
            "failed": 1,
            "skipped": 2,
        },
        "latest": {
            "event_id": "event-reflection-1",
            "run_id": "run-reflection-1",
            "occurred_at": "2026-05-14T00:05:00+00:00",
            "status": "failed",
        },
        "semantic_descriptors": {
            "reflection_health": "mixed",
        },
    }
    return payload


def _self_cognition_stats_payload() -> dict[str, object]:
    """Build a sanitized self-cognition stats payload for CLI tests."""

    payload = {
        "status": "ok",
        "generated_at": "2026-05-14T00:06:00+00:00",
        "window_hours": 4,
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
async def test_build_ops_status_document_reads_aggregate_builders(
    monkeypatch,
) -> None:
    """Status helper should read all aggregate builders with one window."""

    build_runtime_status = AsyncMock(return_value=_runtime_status_payload())
    build_reflection_stats = AsyncMock(return_value=_reflection_stats_payload())
    build_self_cognition_stats = AsyncMock(
        return_value=_self_cognition_stats_payload(),
    )
    monkeypatch.setattr(
        fetch_module.event_logging,
        "build_runtime_status",
        build_runtime_status,
    )
    monkeypatch.setattr(
        fetch_module.event_logging,
        "build_reflection_stats",
        build_reflection_stats,
    )
    monkeypatch.setattr(
        fetch_module.event_logging,
        "build_self_cognition_stats",
        build_self_cognition_stats,
    )

    status_document = await fetch_module.build_ops_status_document(
        window_hours=4,
    )

    assert status_document["window_hours"] == 4
    assert status_document["runtime_status"]["status"] == "ok"
    assert status_document["self_cognition_runtime"]["enabled"] in {True, False}
    assert status_document["reflection_stats"]["counts"]["failed"] == 1
    assert status_document["self_cognition_stats"]["counts"]["runs"] == 2
    build_runtime_status.assert_awaited_once_with(window_hours=4)
    build_reflection_stats.assert_awaited_once_with(window_hours=4)
    build_self_cognition_stats.assert_awaited_once_with(window_hours=4)


def test_format_ops_status_document_prints_sanitized_summary() -> None:
    """Terminal formatter should include aggregate status and refs only."""

    status_document = {
        "generated_at": "2026-05-14T00:08:00+00:00",
        "window_hours": 4,
        "runtime_status": _runtime_status_payload(),
        "reflection_stats": _reflection_stats_payload(),
        "self_cognition_stats": _self_cognition_stats_payload(),
    }

    text = fetch_module.format_ops_status_document(status_document)

    assert "process_status: ready" in text
    assert "worker_error_level: none" in text
    assert "succeeded: 3" in text
    assert "health: mixed" in text
    assert "enabled:" in text
    assert "runs: 2" in text
    assert "liveness: active_with_handoff" in text
    assert "event_id=event-self-1" in text
    assert "private message" not in text
    assert "secret" not in text


def test_compact_json_serializes_status_document() -> None:
    """JSON mode should preserve aggregate keys in stable readable output."""

    rendered = fetch_module._compact_json({
        "window_hours": 4,
        "runtime_status": _runtime_status_payload(),
    })
    payload = json.loads(rendered)

    assert payload["window_hours"] == 4
    assert payload["runtime_status"]["process"]["last_status"] == "ready"
