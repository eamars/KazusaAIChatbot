"""Compact server-sent event helpers for the control console."""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import uuid
from typing import Any

from control_console.contracts import ProcessLogLine
from control_console.redaction import redact_mapping
from control_console.redaction import redact_text


LOG_STREAM_MAX_LINE_CHARS = 600
LOG_STREAM_ALLOWED_STREAMS = frozenset({"stdout", "stderr", "supervisor"})


@dataclass(frozen=True, slots=True)
class SSEEvent:
    """One compact SSE event."""

    event_id: str
    event_type: str
    data: dict[str, Any]

    @property
    def cursor(self) -> str:
        """Return the event cursor used by EventSource replay."""

        return self.event_id


@dataclass(slots=True)
class LogStreamSubscription:
    """One live-log subscriber with a bounded delivery queue."""

    subscription_id: str
    service_id: str
    streams: set[str]
    queue: asyncio.Queue[SSEEvent]


class SSEEventBuffer:
    """Bounded replay buffer for compact console events."""

    def __init__(self, *, max_events: int = 100) -> None:
        """Create a bounded SSE replay buffer."""

        self._max_events = max_events
        self._events: deque[SSEEvent] = deque(maxlen=max_events)
        self._sequence = 0

    def append(self, event_type: str, data: dict[str, Any]) -> str:
        """Append an event and return its monotonic event id."""

        self._sequence += 1
        event_id = str(self._sequence)
        event = SSEEvent(
            event_id=event_id,
            event_type=event_type,
            data=redact_mapping(data),
        )
        self._events.append(event)
        return event_id

    def replay_after(self, last_event_id: str | None) -> list[SSEEvent]:
        """Replay buffered events after an id or return a gap event."""

        if last_event_id is None:
            events = list(self._events)
            return events

        event_ids = [event.event_id for event in self._events]
        if last_event_id in event_ids:
            index = event_ids.index(last_event_id)
            events = list(self._events)[index + 1 :]
            return events
        if _is_before_current_window(
            last_event_id=last_event_id,
            event_ids=event_ids,
        ):
            latest_event_id = event_ids[-1] if event_ids else "0"
            gap = SSEEvent(
                event_id=latest_event_id,
                event_type="control.gap",
                data={
                    "reason": "replay_unavailable",
                    "latest_event_id": latest_event_id,
                },
            )
            return [gap]

        latest_event_id = event_ids[-1] if event_ids else "0"
        gap = SSEEvent(
            event_id=latest_event_id,
            event_type="control.gap",
            data={
                "reason": "replay_unavailable",
                "latest_event_id": latest_event_id,
            },
        )
        return [gap]


def encode_sse_event(event: SSEEvent) -> str:
    """Encode one SSE event."""

    payload = json.dumps(event.data, separators=(",", ":"), sort_keys=True)
    encoded = f"id: {event.event_id}\nevent: {event.event_type}\ndata: {payload}\n\n"
    return encoded


def _is_before_current_window(*, last_event_id: str, event_ids: list[str]) -> bool:
    """Return whether a numeric cursor precedes the retained event window."""

    if not event_ids:
        return False
    try:
        requested_id = int(last_event_id)
        oldest_id = int(event_ids[0])
    except ValueError:
        return False
    is_before = requested_id < oldest_id
    return is_before


class LogStreamHub:
    """Bounded live-log publisher with replay and slow-subscriber gaps."""

    def __init__(
        self,
        *,
        max_events: int = 500,
        subscriber_queue_size: int = 200,
    ) -> None:
        """Create the process-local live-log stream hub."""

        self._max_events = max_events
        self._subscriber_queue_size = subscriber_queue_size
        self._events: deque[SSEEvent] = deque(maxlen=max_events)
        self._subscribers: dict[str, LogStreamSubscription] = {}
        self._sequence = 0

    @property
    def subscriber_count(self) -> int:
        """Return the number of active log subscribers."""

        return len(self._subscribers)

    def publish(
        self,
        *,
        service_id: str,
        stream: str,
        line: str,
    ) -> SSEEvent:
        """Publish one process-log line to replay storage and subscribers."""

        self._sequence += 1
        event = self._build_line_event(
            cursor=f"log-{self._sequence}",
            service_id=service_id,
            stream=stream,
            line=line,
            sequence=self._sequence,
        )
        self._events.append(event)
        self._publish_to_subscribers(event)
        return event

    def publish_log_line(self, log_line: ProcessLogLine) -> SSEEvent:
        """Publish a stored process-log line to live subscribers."""

        self._sequence += 1
        event = self._build_line_event(
            cursor=log_line.cursor,
            service_id=log_line.service_id,
            stream=log_line.stream,
            line=log_line.line,
            sequence=self._sequence,
            created_at=log_line.created_at.isoformat(),
        )
        self._events.append(event)
        self._publish_to_subscribers(event)
        return event

    def replay_after(
        self,
        *,
        cursor: str | None,
        service_id: str,
        streams: set[str],
        tail: int,
    ) -> list[SSEEvent]:
        """Return retained events after a cursor or an explicit gap event."""

        matched_events = [
            event
            for event in self._events
            if _log_event_matches(
                event=event,
                service_id=service_id,
                streams=streams,
            )
        ]
        if cursor is None:
            events = matched_events[-tail:] if tail > 0 else []
            return events

        event_ids = [event.event_id for event in matched_events]
        if cursor in event_ids:
            index = event_ids.index(cursor)
            events = matched_events[index + 1 :]
            return events

        if not matched_events:
            events: list[SSEEvent] = []
            return events

        latest_cursor = matched_events[-1].event_id
        gap = _log_gap_event(
            latest_cursor=latest_cursor,
            reason="replay_unavailable",
            dropped=0,
        )
        return [gap]

    def subscribe(
        self,
        *,
        service_id: str,
        streams: set[str],
    ) -> LogStreamSubscription:
        """Register one live-log subscriber."""

        subscription = LogStreamSubscription(
            subscription_id=f"log-sub-{uuid.uuid4().hex}",
            service_id=service_id,
            streams=set(streams),
            queue=asyncio.Queue(maxsize=self._subscriber_queue_size),
        )
        self._subscribers[subscription.subscription_id] = subscription
        return subscription

    def unsubscribe(self, subscription: LogStreamSubscription) -> None:
        """Remove one live-log subscriber."""

        self._subscribers.pop(subscription.subscription_id, None)

    def _build_line_event(
        self,
        *,
        cursor: str,
        service_id: str,
        stream: str,
        line: str,
        sequence: int,
        created_at: str | None = None,
    ) -> SSEEvent:
        """Build one redacted and bounded live-log event."""

        redacted_line = redact_text(line)
        truncated = len(redacted_line) > LOG_STREAM_MAX_LINE_CHARS
        if truncated:
            redacted_line = redacted_line[:LOG_STREAM_MAX_LINE_CHARS]
        event = SSEEvent(
            event_id=cursor,
            event_type="log.line",
            data={
                "cursor": cursor,
                "service_id": service_id,
                "stream": stream,
                "sequence": sequence,
                "created_at": created_at
                or datetime.now(timezone.utc).isoformat(),
                "line": redacted_line,
                "truncated": truncated,
            },
        )
        return event

    def _publish_to_subscribers(self, event: SSEEvent) -> None:
        """Deliver one event without allowing slow clients to block."""

        for subscription in list(self._subscribers.values()):
            if not _log_event_matches(
                event=event,
                service_id=subscription.service_id,
                streams=subscription.streams,
            ):
                continue
            _offer_event(subscription.queue, event)


def log_snapshot_event(log_line: ProcessLogLine) -> SSEEvent:
    """Project a stored process-log line into the live-log SSE shape."""

    redacted_line = redact_text(log_line.line)
    truncated = len(redacted_line) > LOG_STREAM_MAX_LINE_CHARS
    if truncated:
        redacted_line = redacted_line[:LOG_STREAM_MAX_LINE_CHARS]
    event = SSEEvent(
        event_id=log_line.cursor,
        event_type="log.snapshot",
        data={
            "cursor": log_line.cursor,
            "service_id": log_line.service_id,
            "stream": log_line.stream,
            "sequence": 0,
            "created_at": log_line.created_at.isoformat(),
            "line": redacted_line,
            "truncated": truncated,
        },
    )
    return event


def log_ready_event() -> SSEEvent:
    """Return the stream-ready marker emitted after initial replay."""

    event = SSEEvent(
        event_id=f"log-ready-{uuid.uuid4().hex}",
        event_type="log.ready",
        data={"status": "live"},
    )
    return event


def log_status_event(*, service_id: str, status: str, message: str) -> SSEEvent:
    """Return a live-log availability status event."""

    event = SSEEvent(
        event_id=f"log-status-{uuid.uuid4().hex}",
        event_type="log.status",
        data={
            "service_id": service_id,
            "status": status,
            "message": message,
        },
    )
    return event


def log_keepalive_event() -> SSEEvent:
    """Return an idle keepalive event for live-log streams."""

    event = SSEEvent(
        event_id=f"log-keepalive-{uuid.uuid4().hex}",
        event_type="log.keepalive",
        data={"generated_at": datetime.now(timezone.utc).isoformat()},
    )
    return event


def parse_log_streams(raw_streams: str) -> set[str]:
    """Parse a comma-separated process-log stream allowlist."""

    streams = {
        item.strip()
        for item in raw_streams.split(",")
        if item.strip()
    }
    if not streams:
        parsed_streams = set(LOG_STREAM_ALLOWED_STREAMS)
        return parsed_streams
    unknown_streams = streams.difference(LOG_STREAM_ALLOWED_STREAMS)
    if unknown_streams:
        unknown_list = ", ".join(sorted(unknown_streams))
        raise ValueError(f"unknown log streams: {unknown_list}")
    return streams


def _offer_event(queue: asyncio.Queue[SSEEvent], event: SSEEvent) -> None:
    """Place an event into a subscriber queue or replace backlog with a gap."""

    if not queue.full():
        queue.put_nowait(event)
        return

    dropped = 0
    while not queue.empty():
        queue.get_nowait()
        dropped += 1
    gap = _log_gap_event(
        latest_cursor=event.event_id,
        reason="subscriber_backpressure",
        dropped=dropped,
    )
    queue.put_nowait(gap)


def _log_event_matches(
    *,
    event: SSEEvent,
    service_id: str,
    streams: set[str],
) -> bool:
    """Return whether a log event is visible to one filter."""

    event_service_id = event.data.get("service_id")
    event_stream = event.data.get("stream")
    if service_id != "all" and event_service_id != service_id:
        return False
    if event_stream not in streams:
        return False
    return True


def _log_gap_event(
    *,
    latest_cursor: str,
    reason: str,
    dropped: int,
) -> SSEEvent:
    """Return an explicit live-log gap event."""

    event = SSEEvent(
        event_id=latest_cursor,
        event_type="log.gap",
        data={
            "reason": reason,
            "latest_cursor": latest_cursor,
            "dropped": dropped,
        },
    )
    return event
