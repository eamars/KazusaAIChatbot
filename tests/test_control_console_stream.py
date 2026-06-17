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
