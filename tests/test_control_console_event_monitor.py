"""Event monitor merge and redaction tests."""

from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_event_monitor_merges_and_redacts_bounded_sources() -> None:
    """Operator event views should merge sources without leaking sensitive data."""

    from control_console.contracts import OperationalEventQuery
    from control_console.event_monitor import EventMonitor

    async def read_audit_events(query):
        _ = query
        rows = [{
            "source": "console",
            "event_type": "lookup_executed",
            "message": "token=secret prompt hidden",
            "created_at": "2026-06-17T00:00:00+00:00",
        }]
        return rows

    async def read_process_logs(query):
        _ = query
        rows = [{
            "source": "process",
            "event_type": "stderr",
            "line": "api_key=abc raw_message=hello",
            "created_at": "2026-06-17T00:00:01+00:00",
        }]
        return rows

    async def read_kazusa_events(query):
        _ = query
        rows = [{
            "source": "kazusa",
            "event_type": "resource_health",
            "embedding": [0.1, 0.2],
            "created_at": "2026-06-17T00:00:02+00:00",
        }]
        return rows

    monitor = EventMonitor(
        read_audit_events=read_audit_events,
        read_process_logs=read_process_logs,
        read_kazusa_events=read_kazusa_events,
    )
    result = await monitor.query(
        OperationalEventQuery.model_validate({"source": "all", "limit": 2})
    )

    rendered = repr(result)
    assert len(result.items) == 2
    assert "secret" not in rendered
    assert "hidden" not in rendered
    assert "abc" not in rendered
    assert "hello" not in rendered
    assert "0.1" not in rendered


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
            "payload": {"raw_output": "do not expose"},
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
            "$or": [
                {"run_id": "tracking-1"},
                {"trigger_id": "tracking-1"},
                {"attempt_id": "tracking-1"},
                {"refs.ref_id": "tracking-1"},
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
