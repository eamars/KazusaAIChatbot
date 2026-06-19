"""SSE event-shape tests for the control console."""

from __future__ import annotations

import asyncio
from time import perf_counter


async def test_stream_wait_returns_immediately_after_shutdown_signal() -> None:
    """SSE waits should not keep Ctrl+C blocked after shutdown is requested."""

    from control_console.app import _wait_for_stream_tick

    shutdown_event = asyncio.Event()
    shutdown_event.set()

    started_at = perf_counter()
    should_stop = await _wait_for_stream_tick(
        shutdown_event=shutdown_event,
        interval_seconds=30.0,
    )
    elapsed_seconds = perf_counter() - started_at

    assert should_stop is True
    assert elapsed_seconds < 0.1


async def test_stream_wait_times_out_when_shutdown_is_not_signaled() -> None:
    """SSE waits should continue normally when the console is still running."""

    from control_console.app import _wait_for_stream_tick

    should_stop = await _wait_for_stream_tick(
        shutdown_event=asyncio.Event(),
        interval_seconds=0.01,
    )

    assert should_stop is False


def test_summary_stream_emits_bounded_service_event_cursor_and_gap_payload() -> None:
    """SSE helpers should emit compact events and explicit replay gaps."""

    from control_console.stream import SSEEventBuffer, encode_sse_event

    buffer = SSEEventBuffer(max_events=3)
    first_id = buffer.append("control.service", {"service_id": "brain"})
    buffer.append("control.audit", {"event_type": "service_started"})
    third_id = buffer.append("control.heartbeat", {"generated_at": "now"})

    replay = buffer.replay_after(first_id)
    assert [event.event_type for event in replay] == [
        "control.audit",
        "control.heartbeat",
    ]

    gap = buffer.replay_after("missing-old-id")
    assert len(gap) == 1
    assert gap[0].event_type == "control.gap"
    assert gap[0].data["latest_event_id"] == third_id

    encoded = encode_sse_event(gap[0])
    assert "event: control.gap" in encoded
    assert "data:" in encoded
    assert "raw_message" not in encoded


def test_stream_gap_forces_bootstrap_refetch() -> None:
    """Missing replay windows should produce a compact gap event."""

    from control_console.stream import SSEEventBuffer

    buffer = SSEEventBuffer(max_events=1)
    latest_id = buffer.append("control.service", {"service_id": "brain"})
    gap = buffer.replay_after("old-event-id")

    assert len(gap) == 1
    assert gap[0].event_type == "control.gap"
    assert gap[0].data["reason"] == "replay_unavailable"
    assert gap[0].data["latest_event_id"] == latest_id


def test_numeric_cursor_before_replay_window_returns_gap() -> None:
    """Numeric cursors outside the retained window should not replay partial data."""

    from control_console.stream import SSEEventBuffer

    buffer = SSEEventBuffer(max_events=2)
    buffer.append("control.service", {"service_id": "brain"})
    buffer.append("control.audit", {"event_type": "service_started"})
    latest_id = buffer.append("control.heartbeat", {"generated_at": "now"})

    gap = buffer.replay_after("1")

    assert len(gap) == 1
    assert gap[0].event_type == "control.gap"
    assert gap[0].data["latest_event_id"] == latest_id


async def test_stream_poll_appends_graph_invalidation_for_new_latest_run() -> None:
    """SSE should notify the UI when the brain latest cognition graph changes."""

    from control_console.app import _append_cognition_graph_invalidation_if_changed
    from control_console.kazusa_client import project_cognition_graph_snapshot
    from control_console.stream import SSEEventBuffer

    class FakeKazusaClient:
        async def get_latest_cognition_graph(self):
            return project_cognition_graph_snapshot(
                source="overview_latest",
                payload={
                    "cognition_graph": {
                        "run_id": "self_cognition_run:future-123",
                        "status": "completed",
                        "nodes": [
                            {
                                "id": "self.reasoning",
                                "label": "Reasoning",
                                "stage": "L2",
                                "lane": "cognition",
                                "column": 2,
                                "branch": "reasoning",
                                "status": "completed",
                                "detail": {
                                    "internal_monologue": "bounded reason",
                                },
                            },
                        ],
                        "edges": [],
                    },
                },
            )

    buffer = SSEEventBuffer(max_events=5)

    latest_run_id = await _append_cognition_graph_invalidation_if_changed(
        kazusa_client=FakeKazusaClient(),
        stream_buffer=buffer,
        previous_run_id="debug_turn_1",
    )

    replay = buffer.replay_after(None)
    assert latest_run_id == "self_cognition_run:future-123"
    assert replay[-1].event_type == "control.cognition_graph_invalidated"
    assert replay[-1].data["run_id"] == "self_cognition_run:future-123"


async def test_stream_poll_ignores_missing_run_id_and_client_errors() -> None:
    """SSE graph polling should keep the previous cursor on non-actionable reads."""

    import httpx

    from control_console.app import _append_cognition_graph_invalidation_if_changed
    from control_console.kazusa_client import not_reported_cognition_graph
    from control_console.stream import SSEEventBuffer

    class MissingRunClient:
        async def get_latest_cognition_graph(self):
            return not_reported_cognition_graph(source="overview_latest")

    class ErrorClient:
        async def get_latest_cognition_graph(self):
            raise httpx.ConnectError("brain down")

    buffer = SSEEventBuffer(max_events=5)
    missing_result = await _append_cognition_graph_invalidation_if_changed(
        kazusa_client=MissingRunClient(),
        stream_buffer=buffer,
        previous_run_id="old-run",
    )
    error_result = await _append_cognition_graph_invalidation_if_changed(
        kazusa_client=ErrorClient(),
        stream_buffer=buffer,
        previous_run_id="old-run",
    )

    assert missing_result == "old-run"
    assert error_result == "old-run"
    assert buffer.replay_after(None) == []


async def test_stream_iterator_emits_graph_invalidation_and_heartbeat() -> None:
    """The SSE iterator should yield graph invalidations added in its loop."""

    import pytest

    from control_console.app import _stream_console_events
    from control_console.contracts import ServiceRuntimeState
    from control_console.kazusa_client import project_cognition_graph_snapshot
    from control_console.stream import SSEEventBuffer

    class FakeRequest:
        headers: dict[str, str] = {}

        async def is_disconnected(self) -> bool:
            return False

    class RunningSupervisor:
        def all_service_states(self):
            return [
                ServiceRuntimeState(
                    id="brain",
                    display_name="Brain service",
                    kind="backend",
                    actual_state="running",
                ),
            ]

        def service_state(self, service_id: str):
            assert service_id == "brain"
            return self.all_service_states()[0]

    class FakeKazusaClient:
        async def get_health(self):
            return {"status": "healthy"}

        async def get_runtime_status(self):
            return {"status": "running"}

        async def get_latest_cognition_graph(self):
            return project_cognition_graph_snapshot(
                source="overview_latest",
                payload={
                    "cognition_graph": {
                        "run_id": "stream-run",
                        "status": "completed",
                        "nodes": [],
                        "edges": [],
                    },
                },
            )

    stream_buffer = SSEEventBuffer(max_events=10)
    stream_buffer.append("control.heartbeat", {"generated_at": "startup"})
    shutdown_event = asyncio.Event()
    iterator = _stream_console_events(
        request=FakeRequest(),
        stream_buffer=stream_buffer,
        shutdown_event=shutdown_event,
        supervisor=RunningSupervisor(),
        services={"brain": object()},
        kazusa_client=FakeKazusaClient(),
        latest_cognition_graph_state={"run_id": "old-run"},
        interval_seconds=0.01,
    )

    first = await iterator.__anext__()
    second = await iterator.__anext__()
    third = await iterator.__anext__()
    shutdown_event.set()
    with pytest.raises(StopAsyncIteration):
        await iterator.__anext__()

    body = "\n".join([first, second, third])
    assert "event: control.heartbeat" in body
    assert "event: control.cognition_graph_invalidated" in body
