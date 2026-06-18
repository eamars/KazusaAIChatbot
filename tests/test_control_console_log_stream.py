"""Live process-log stream tests for the control console."""

from __future__ import annotations

import asyncio


def _login(client) -> dict[str, str]:
    """Authenticate a test client and return CSRF metadata."""

    response = client.post("/api/auth/login", json={"token": "secret"})
    assert response.status_code == 200
    payload = response.json()
    return payload


def test_log_stream_requires_authenticated_session(tmp_path) -> None:
    """The live-log stream should not expose process output before login."""

    from fastapi.testclient import TestClient

    from control_console.app import create_app
    from control_console.auth import hash_operator_token
    from control_console.settings import ControlConsoleSettings

    settings = ControlConsoleSettings(
        state_dir=tmp_path,
        operator_token_hash=hash_operator_token("secret"),
    )
    client = TestClient(create_app(settings=settings))

    response = client.get("/api/logs/stream")

    assert response.status_code == 401


def test_log_stream_route_uses_last_event_id_header(
    monkeypatch,
    tmp_path,
) -> None:
    """Normal EventSource reconnects should resume from Last-Event-ID."""

    from fastapi.testclient import TestClient

    import control_console.app as app_module
    from control_console.auth import hash_operator_token
    from control_console.settings import ControlConsoleSettings

    captured = {}

    async def fake_stream_process_logs(**kwargs):
        captured["cursor"] = kwargs["cursor"]
        yield "event: log.ready\ndata: {\"status\":\"live\"}\n\n"

    monkeypatch.setattr(
        app_module,
        "_stream_process_logs",
        fake_stream_process_logs,
    )
    settings = ControlConsoleSettings(
        state_dir=tmp_path,
        operator_token_hash=hash_operator_token("secret"),
    )
    client = TestClient(app_module.create_app(settings=settings))
    _login(client)

    response = client.get(
        "/api/logs/stream",
        headers={"last-event-id": "brain-log-12"},
    )

    assert response.status_code == 200
    assert captured["cursor"] == "brain-log-12"


def test_log_stream_route_query_cursor_overrides_last_event_id_header(
    monkeypatch,
    tmp_path,
) -> None:
    """Explicit cursor query parameters should win over reconnect headers."""

    from fastapi.testclient import TestClient

    import control_console.app as app_module
    from control_console.auth import hash_operator_token
    from control_console.settings import ControlConsoleSettings

    captured = {}

    async def fake_stream_process_logs(**kwargs):
        captured["cursor"] = kwargs["cursor"]
        yield "event: log.ready\ndata: {\"status\":\"live\"}\n\n"

    monkeypatch.setattr(
        app_module,
        "_stream_process_logs",
        fake_stream_process_logs,
    )
    settings = ControlConsoleSettings(
        state_dir=tmp_path,
        operator_token_hash=hash_operator_token("secret"),
    )
    client = TestClient(app_module.create_app(settings=settings))
    _login(client)

    response = client.get(
        "/api/logs/stream?cursor=query-cursor",
        headers={"last-event-id": "header-cursor"},
    )

    assert response.status_code == 200
    assert captured["cursor"] == "query-cursor"


async def test_log_stream_replays_tail_then_live_line(tmp_path) -> None:
    """Opening the stream should emit retained lines before live updates."""

    from control_console.app import _stream_process_logs
    from control_console.log_store import ProcessLogStore
    from control_console.stream import LogStreamHub
    from control_console.settings import ControlConsoleSettings

    class DisconnectingRequest:
        async def is_disconnected(self) -> bool:
            return True

    settings = ControlConsoleSettings(
        state_dir=tmp_path,
    )
    store = ProcessLogStore(settings.log_dir)
    store.append_line(service_id="brain", stream="stdout", line="retained line")
    iterator = _stream_process_logs(
        request=DisconnectingRequest(),
        hub=LogStreamHub(),
        log_store=store,
        services={"brain": object()},
        service_id="brain",
        streams={"stdout"},
        tail=1,
        cursor=None,
        keepalive_seconds=0.01,
    )
    first = await iterator.__anext__()
    second = await iterator.__anext__()
    body = first + second

    assert "event: log.snapshot" in body
    assert "event: log.ready" in body
    assert "service_id" in body
    assert "brain" in body
    assert "retained line" in body


async def test_log_stream_does_not_duplicate_store_backed_snapshot(tmp_path) -> None:
    """A fresh stream should not replay the same stored line twice."""

    from control_console.app import _stream_process_logs
    from control_console.log_store import ProcessLogStore
    from control_console.stream import LogStreamHub
    from control_console.settings import ControlConsoleSettings

    class DisconnectingRequest:
        async def is_disconnected(self) -> bool:
            return True

    settings = ControlConsoleSettings(
        state_dir=tmp_path,
    )
    hub = LogStreamHub()
    store = ProcessLogStore(settings.log_dir, log_publisher=hub.publish_log_line)
    store.append_line(service_id="brain", stream="stdout", line="startup once")
    iterator = _stream_process_logs(
        request=DisconnectingRequest(),
        hub=hub,
        log_store=store,
        services={"brain": object()},
        service_id="brain",
        streams={"stdout"},
        tail=10,
        cursor=None,
        keepalive_seconds=0.01,
    )
    first = await iterator.__anext__()
    second = await iterator.__anext__()
    body = first + second

    assert body.count("startup once") == 1
    assert "event: log.snapshot" in body
    assert "event: log.line" not in body


async def test_log_stream_filters_by_service_and_stream(tmp_path) -> None:
    """Stream snapshots should include only the requested service and streams."""

    from control_console.app import _stream_process_logs
    from control_console.log_store import ProcessLogStore
    from control_console.stream import LogStreamHub
    from control_console.settings import ControlConsoleSettings

    class DisconnectingRequest:
        async def is_disconnected(self) -> bool:
            return True

    settings = ControlConsoleSettings(
        state_dir=tmp_path,
    )
    store = ProcessLogStore(settings.log_dir)
    store.append_line(service_id="brain", stream="stdout", line="brain stdout")
    store.append_line(service_id="brain", stream="stderr", line="brain stderr")
    store.append_line(service_id="adapter.debug", stream="stdout", line="adapter")
    iterator = _stream_process_logs(
        request=DisconnectingRequest(),
        hub=LogStreamHub(),
        log_store=store,
        services={"brain": object(), "adapter.debug": object()},
        service_id="brain",
        streams={"stdout"},
        tail=10,
        cursor=None,
        keepalive_seconds=0.01,
    )
    first = await iterator.__anext__()
    body = first

    assert "brain stdout" in body
    assert "brain stderr" not in body
    assert "adapter" not in body


async def test_log_stream_reports_gap_when_cursor_is_outside_buffer() -> None:
    """Old cursors should produce an explicit log.gap event."""

    from control_console.stream import LogStreamHub

    hub = LogStreamHub(max_events=1, subscriber_queue_size=2)
    first = hub.publish(
        service_id="brain",
        stream="stdout",
        line="first",
    )
    hub.publish(service_id="brain", stream="stdout", line="second")

    replay = hub.replay_after(
        cursor=first.cursor,
        service_id="brain",
        streams={"stdout"},
        tail=10,
    )

    assert len(replay) == 1
    assert replay[0].event_type == "log.gap"
    assert replay[0].data["reason"] == "replay_unavailable"


async def test_log_stream_drops_slow_subscriber_without_blocking_publish() -> None:
    """A full subscriber queue should report a gap without blocking publishers."""

    from control_console.stream import LogStreamHub

    hub = LogStreamHub(max_events=10, subscriber_queue_size=1)
    subscriber = hub.subscribe(service_id="brain", streams={"stdout"})

    hub.publish(service_id="brain", stream="stdout", line="line 1")
    hub.publish(service_id="brain", stream="stdout", line="line 2")
    event = await asyncio.wait_for(subscriber.queue.get(), timeout=0.1)

    assert event.event_type == "log.gap"
    assert event.data["dropped"] >= 1
    hub.unsubscribe(subscriber)
    assert hub.subscriber_count == 0


def test_log_stream_redacts_and_truncates_lines() -> None:
    """Log payloads should be safe before they reach subscribers."""

    from datetime import datetime, timezone

    from control_console.contracts import ProcessLogLine
    from control_console.stream import LOG_STREAM_MAX_LINE_CHARS, LogStreamHub
    from control_console.stream import log_snapshot_event

    hub = LogStreamHub(max_events=10, subscriber_queue_size=2)
    event = hub.publish(
        service_id="brain",
        stream="stderr",
        line=f"api_key=secret {'x' * (LOG_STREAM_MAX_LINE_CHARS + 20)}",
    )

    assert event.data["truncated"] is True
    assert "secret" not in event.data["line"]
    assert len(event.data["line"]) <= LOG_STREAM_MAX_LINE_CHARS

    snapshot = log_snapshot_event(
        ProcessLogLine(
            service_id="brain",
            stream="stderr",
            line=f"api_key=secret {'x' * (LOG_STREAM_MAX_LINE_CHARS + 20)}",
            created_at=datetime.now(timezone.utc),
            cursor="snapshot-1",
        ),
    )
    assert snapshot.data["truncated"] is True
    assert "secret" not in snapshot.data["line"]
    assert len(snapshot.data["line"]) <= LOG_STREAM_MAX_LINE_CHARS


async def test_log_stream_disconnect_cleans_up_subscriber() -> None:
    """Stream generators should remove subscribers when clients disconnect."""

    from control_console.app import _stream_process_logs
    from control_console.stream import LogStreamHub

    class DisconnectingRequest:
        async def is_disconnected(self) -> bool:
            return True

    hub = LogStreamHub(max_events=10, subscriber_queue_size=2)
    iterator = _stream_process_logs(
        request=DisconnectingRequest(),
        hub=hub,
        service_id="brain",
        streams={"stdout"},
        tail=0,
        cursor=None,
        keepalive_seconds=0.01,
    )

    first = await iterator.__anext__()
    assert "event: log.ready" in first
    await iterator.aclose()
    assert hub.subscriber_count == 0


async def test_log_status_reports_unmanaged_conflict_unavailable(tmp_path) -> None:
    """Unmanaged service conflicts should be visible to the logs UI."""

    from control_console.app import _stream_process_logs
    from control_console.contracts import ServiceRuntimeState
    from control_console.stream import LogStreamHub
    from control_console.supervisor import ENDPOINT_CONFLICT_MESSAGE

    class DisconnectingRequest:
        async def is_disconnected(self) -> bool:
            return True

    class ConflictSupervisor:
        def all_service_states(self):
            return [self.service_state("brain")]

        def service_state(self, service_id: str):
            assert service_id == "brain"
            return ServiceRuntimeState(
                id="brain",
                display_name="Brain service",
                kind="backend",
                actual_state="conflict",
                last_error_preview=ENDPOINT_CONFLICT_MESSAGE,
            )

    iterator = _stream_process_logs(
        request=DisconnectingRequest(),
        hub=LogStreamHub(),
        supervisor=ConflictSupervisor(),
        services={"brain": object()},
        service_id="brain",
        streams={"stdout", "stderr", "supervisor"},
        tail=0,
        cursor=None,
        keepalive_seconds=0.01,
    )
    body = await iterator.__anext__()

    assert "event: log.status" in body
    assert "logs unavailable from this console run" in body
