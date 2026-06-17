"""Compact server-sent event helpers for the control console."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
from typing import Any

from control_console.redaction import redact_mapping


@dataclass(frozen=True, slots=True)
class SSEEvent:
    """One compact SSE event."""

    event_id: str
    event_type: str
    data: dict[str, Any]


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
