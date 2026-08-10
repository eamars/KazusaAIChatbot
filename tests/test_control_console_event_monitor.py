"""Event monitor merge and redaction tests."""

from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_event_monitor_owns_application_events_without_audit_views() -> None:
    """Event monitor should exclude Audit-owned control-console records."""

    from pydantic import ValidationError
    from control_console.contracts import OperationalEventQuery
    from control_console.event_monitor import EventMonitor

    async def read_kazusa_events(query):
        assert query.limit == 5
        if query.event_type == "tick":
            return [{
                "source": "kazusa",
                "event_type": "tick",
                "component": "background_work.worker",
                "level": "info",
                "status": "succeeded",
                "created_at": "2026-06-17T00:00:01+00:00",
            }]
        rows = [{
            "source": "kazusa",
            "event_type": "resource_health",
            "component": "brain_service",
            "level": "warning",
            "status": "degraded",
            "embedding": [0.1, 0.2],
            "created_at": "2026-06-17T00:00:02+00:00",
        }, {
            "source": "kazusa",
            "event_type": "tick",
            "component": "background_work.worker",
            "level": "info",
            "status": "succeeded",
            "created_at": "2026-06-17T00:00:01+00:00",
        }, {
            "source": "kazusa",
            "event_type": "load_residue_context",
            "component": "internal_monologue_residue",
            "level": "info",
            "status": "empty",
            "created_at": "2026-06-17T00:00:00+00:00",
        }]
        return rows

    monitor = EventMonitor(read_kazusa_events=read_kazusa_events)
    result = await monitor.query(
        OperationalEventQuery.model_validate({"source": "all", "limit": 5})
    )

    assert len(result.items) == 1
    assert result.items[0]["event_type"] == "resource_health"
    assert result.facets["sources"] == {"kazusa": 1}
    assert result.facets["levels"] == {"warning": 1}
    assert result.facets["statuses"] == {"degraded": 1}
    rendered = repr(result)
    assert "secret" not in rendered
    assert "hidden" not in rendered
    assert "0.1" not in rendered
    assert "process" not in rendered

    tick_result = await monitor.query(
        OperationalEventQuery.model_validate({
            "source": "kazusa",
            "event_type": "tick",
            "limit": 5,
        })
    )
    assert len(tick_result.items) == 1
    assert tick_result.items[0]["event_type"] == "tick"

    with pytest.raises(ValidationError):
        OperationalEventQuery.model_validate({"source": "process", "limit": 5})
    with pytest.raises(ValidationError):
        OperationalEventQuery.model_validate({"source": "console", "limit": 5})


@pytest.mark.asyncio
async def test_kazusa_event_reader_pushes_default_noise_filter_to_database() -> None:
    """Routine aggregate events must not consume the bounded database window."""

    from control_console import app as app_module
    from control_console.contracts import OperationalEventQuery

    calls: list[dict] = []

    async def find_events(filter_doc, *, sort, limit):
        calls.append({
            "filter_doc": filter_doc,
            "sort": sort,
            "limit": limit,
        })
        return [{
            "event_id": "evt-meaningful",
            "event_type": "queue_intake",
            "component": "brain_service",
            "severity": "info",
            "status": "accepted",
            "occurred_at": "2026-06-17T00:00:00+00:00",
        }]

    query = OperationalEventQuery.model_validate({
        "source": "all",
        "limit": 5,
    })

    rows = await app_module._read_kazusa_events(query, find_events=find_events)

    assert calls == [{
        "filter_doc": {
            "$or": [
                {
                    "event_type": {
                        "$nin": ["load_residue_context", "tick"],
                    },
                },
                {"severity": {"$in": ["error", "warning"]}},
                {
                    "status": {
                        "$in": [
                            "deferred",
                            "degraded",
                            "failed",
                            "unavailable",
                            "warning",
                        ],
                    },
                },
            ],
        },
        "sort": [("occurred_at", -1)],
        "limit": 5,
    }]
    assert [row["event_type"] for row in rows] == ["queue_intake"]


@pytest.mark.asyncio
async def test_kazusa_event_reader_projects_event_log_rows() -> None:
    """Kazusa event rows should come from event-log helpers without raw payloads."""

    from control_console import app as app_module
    from control_console.contracts import OperationalEventQuery

    calls: list[dict] = []

    async def find_events(filter_doc, *, sort, limit):
        calls.append({
            "filter_doc": filter_doc,
            "sort": sort,
            "limit": limit,
        })
        rows = [{
            "event_id": "evt-1",
            "event_family": "worker",
            "event_type": "tick",
            "component": "background_work.worker",
            "severity": "info",
            "status": "succeeded",
            "correlation_id": "cc-req-1",
            "run_id": "run-1",
            "trigger_id": "trigger-1",
            "attempt_id": "attempt-1",
            "occurred_at": "2026-06-17T00:00:00+00:00",
            "created_at": "2026-06-17T00:00:01+00:00",
            "duration_ms": 42,
            "payload": {
                "processed_count": 4,
                "succeeded_count": 3,
                "failed_count": 0,
                "skipped_count": 1,
                "deferred": True,
                "defer_reason": "worker capacity reached",
                "run_kind": "background_tick",
                "worker_name": "text_artifact",
                "raw_output": "do not expose",
            },
            "human_prompt": "do not expose",
            "embedding": [0.1],
        }]
        return rows

    query = OperationalEventQuery.model_validate({
        "source": "kazusa",
        "service_id": "background_work.worker",
        "event_type": "tick",
        "level": "info",
        "request_id": "cc-req-1",
        "limit": 5,
    })

    rows = await app_module._read_kazusa_events(query, find_events=find_events)

    assert calls == [{
        "filter_doc": {
            "component": "background_work.worker",
            "event_type": "tick",
            "severity": "info",
            "correlation_id": "cc-req-1",
        },
        "sort": [("occurred_at", -1)],
        "limit": 5,
    }]
    assert rows == [{
        "source": "kazusa",
        "event_id": "evt-1",
        "event_family": "worker",
        "event_type": "tick",
        "component": "background_work.worker",
        "level": "info",
        "status": "succeeded",
        "correlation_id": "cc-req-1",
        "run_id": "run-1",
        "trigger_id": "trigger-1",
        "attempt_id": "attempt-1",
        "created_at": "2026-06-17T00:00:00+00:00",
        "duration_ms": 42,
        "processed_count": 4,
        "succeeded_count": 3,
        "failed_count": 0,
        "skipped_count": 1,
        "deferred": True,
        "defer_reason": "worker capacity reached",
        "run_kind": "background_tick",
        "worker_name": "text_artifact",
    }]
    rendered = repr(rows)
    assert "human_prompt" not in rendered
    assert "raw_output" not in rendered
    assert "0.1" not in rendered


@pytest.mark.asyncio
async def test_kazusa_event_reader_handles_tracking_filters_and_failures() -> None:
    """Kazusa event reads should expose safe unavailable rows on helper errors."""

    from control_console import app as app_module
    from control_console.contracts import OperationalEventQuery

    async def find_events(filter_doc, *, sort, limit):
        assert filter_doc == {
            "$and": [
                {
                    "$or": [
                        {"run_id": "tracking-1"},
                        {"trigger_id": "tracking-1"},
                        {"attempt_id": "tracking-1"},
                        {"refs.ref_id": "tracking-1"},
                    ],
                },
                {
                    "$or": [
                        {
                            "event_type": {
                                "$nin": ["load_residue_context", "tick"],
                            },
                        },
                        {"severity": {"$in": ["error", "warning"]}},
                        {
                            "status": {
                                "$in": [
                                    "deferred",
                                    "degraded",
                                    "failed",
                                    "unavailable",
                                    "warning",
                                ],
                            },
                        },
                    ],
                },
            ],
            "occurred_at": {"$gte": "2026-06-17T00:00:00+00:00"},
        }
        assert sort == [("occurred_at", -1)]
        assert limit == 5
        raise ValueError("event log config missing")

    query = OperationalEventQuery.model_validate({
        "source": "kazusa",
        "tracking_id": "tracking-1",
        "since": "2026-06-17T00:00:00+00:00",
        "limit": 5,
    })

    rows = await app_module._read_kazusa_events(query, find_events=find_events)

    assert rows[0]["source"] == "kazusa"
    assert rows[0]["event_type"] == "event_log.unavailable"
    assert rows[0]["status"] == "unavailable"
    assert "event log config missing" in rows[0]["message"]


@pytest.mark.asyncio
async def test_kazusa_event_reader_projects_error_summary() -> None:
    """Kazusa event projection should include bounded error class and preview."""

    from control_console import app as app_module
    from control_console.contracts import OperationalEventQuery

    async def find_events(filter_doc, *, sort, limit):
        _ = filter_doc
        _ = sort
        _ = limit
        rows = [{
            "event_id": "evt-error",
            "event_family": "runtime_error",
            "event_type": "runtime_error",
            "component": "background_work.worker",
            "severity": "error",
            "status": "failed",
            "created_at": "2026-06-17T00:01:00+00:00",
            "error": {
                "error_class": "RuntimeError",
                "error_preview": "worker failed safely",
            },
        }]
        return rows

    query = OperationalEventQuery.model_validate({"source": "kazusa", "limit": 1})

    rows = await app_module._read_kazusa_events(query, find_events=find_events)

    assert rows == [{
        "source": "kazusa",
        "event_id": "evt-error",
        "event_family": "runtime_error",
        "event_type": "runtime_error",
        "component": "background_work.worker",
        "level": "error",
        "status": "failed",
        "created_at": "2026-06-17T00:01:00+00:00",
        "error_class": "RuntimeError",
        "message": "worker failed safely",
    }]
