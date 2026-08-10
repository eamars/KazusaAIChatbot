"""Bounded process-local ordering for character operational state updates."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal


MAX_PENDING_CHARACTER_PREDECESSORS = 256


@dataclass(frozen=True)
class PredecessorBarrierResultV1:
    """Describe whether prior operational work completed before a new turn."""

    status: Literal["healthy", "degraded"]
    watermark: int
    awaited_count: int
    timed_out_count: int
    wait_ms: int


@dataclass
class _PredecessorEntry:
    """Keep one process-local receipt completion signal until waiters release."""

    protected_episode_id: str
    process_sequence: int
    registered_at: str
    loop_id: int | None
    receipt: Mapping[str, Any] | None = None
    event: asyncio.Event | None = None


_NEXT_SEQUENCE = 0
_ENTRIES_BY_EPISODE: dict[str, _PredecessorEntry] = {}


def register_predecessor(
    source_episode_id: str,
    *,
    registered_at: str,
) -> dict[str, Any]:
    """Register one operational predecessor before response exposure."""

    if not isinstance(source_episode_id, str) or not source_episode_id.strip():
        raise ValueError("source_episode_id must be non-empty")
    if not isinstance(registered_at, str) or not registered_at.strip():
        raise ValueError("registered_at must be non-empty")
    existing = _ENTRIES_BY_EPISODE.get(source_episode_id)
    if existing is not None:
        token = _token_from_entry(existing)
        return token
    _evict_terminal_entries()
    global _NEXT_SEQUENCE
    _NEXT_SEQUENCE += 1
    entry = _PredecessorEntry(
        protected_episode_id=source_episode_id,
        process_sequence=_NEXT_SEQUENCE,
        registered_at=registered_at,
        loop_id=_current_loop_id(),
    )
    if len(_ENTRIES_BY_EPISODE) >= MAX_PENDING_CHARACTER_PREDECESSORS:
        entry.receipt = {
            "status": "failed",
            "error_code": "capacity_exceeded",
        }
    _ENTRIES_BY_EPISODE[source_episode_id] = entry
    token = _token_from_entry(entry)
    return token


def capture_predecessor_watermark() -> int:
    """Return the newest assigned sequence for an incoming eligible turn."""

    return _NEXT_SEQUENCE


def complete_predecessor(
    token: Mapping[str, Any],
    receipt: Mapping[str, Any],
) -> None:
    """Terminalize one registered predecessor and release active waiters."""

    entry = _entry_for_token(token)
    entry.receipt = dict(receipt)
    if entry.event is not None:
        entry.event.set()


def _registered_predecessor_receipt(
    token: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Return an already-terminal registration receipt for service handling."""

    entry = _entry_for_token(token)
    if entry.receipt is None:
        return None
    return dict(entry.receipt)


async def await_predecessors(
    *,
    before_sequence: int,
    timeout_seconds: float = 45.0,
) -> PredecessorBarrierResultV1:
    """Wait concurrently for predecessors older than one captured watermark."""

    if before_sequence < 0:
        raise ValueError("before_sequence must be non-negative")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    started_at = time.perf_counter()
    entries = [
        entry
        for entry in _ENTRIES_BY_EPISODE.values()
        if (
            entry.process_sequence < before_sequence
            and entry.loop_id == _current_loop_id()
        )
    ]
    pending_entries = [entry for entry in entries if entry.receipt is None]
    if pending_entries:
        events = []
        for entry in pending_entries:
            if entry.event is None:
                entry.event = asyncio.Event()
            events.append(entry.event.wait())
        try:
            await asyncio.wait_for(
                asyncio.gather(*events),
                timeout=timeout_seconds,
            )
        except asyncio.TimeoutError:
            for entry in pending_entries:
                if entry.receipt is None:
                    entry.receipt = {
                        "status": "timed_out",
                        "error_code": "deadline_exceeded",
                    }
                    if entry.event is not None:
                        entry.event.set()
    timed_out_count = sum(
        1
        for entry in entries
        if _receipt_status(entry.receipt) == "timed_out"
    )
    degraded = any(
        _receipt_status(entry.receipt) in {"failed", "timed_out"}
        for entry in entries
    )
    status = "degraded" if degraded else "healthy"
    wait_ms = max(0, int((time.perf_counter() - started_at) * 1000))
    result = PredecessorBarrierResultV1(
        status=status,
        watermark=before_sequence,
        awaited_count=len(entries),
        timed_out_count=timed_out_count,
        wait_ms=wait_ms,
    )
    return result


def _entry_for_token(token: Mapping[str, Any]) -> _PredecessorEntry:
    """Resolve one typed public token to its registered private entry."""

    protected_episode_id = token.get("protected_episode_id")
    process_sequence = token.get("process_sequence")
    if (
        not isinstance(protected_episode_id, str)
        or not isinstance(process_sequence, int)
    ):
        raise ValueError("predecessor token is invalid")
    entry = _ENTRIES_BY_EPISODE.get(protected_episode_id)
    if entry is None or entry.process_sequence != process_sequence:
        raise ValueError("predecessor token is unknown")
    return entry


def _current_loop_id() -> int | None:
    """Return the active event-loop identity when registration is asynchronous."""

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return None
    return id(loop)


def _token_from_entry(entry: _PredecessorEntry) -> dict[str, Any]:
    """Project one private registry entry into its public registration token."""

    token = {
        "protected_episode_id": entry.protected_episode_id,
        "process_sequence": entry.process_sequence,
        "registered_at": entry.registered_at,
    }
    return token


def _receipt_status(receipt: Mapping[str, Any] | None) -> str:
    """Read a validated terminal status without exposing receipt content."""

    if receipt is None:
        return "pending"
    status = receipt.get("status")
    if not isinstance(status, str):
        return "failed"
    return status


def _evict_terminal_entries() -> None:
    """Release oldest terminal entries before enforcing the pending cap."""

    terminal_entries = sorted(
        (
            entry
            for entry in _ENTRIES_BY_EPISODE.values()
            if entry.receipt is not None
        ),
        key=lambda entry: entry.process_sequence,
    )
    while (
        len(_ENTRIES_BY_EPISODE) >= MAX_PENDING_CHARACTER_PREDECESSORS
        and terminal_entries
    ):
        entry = terminal_entries.pop(0)
        _ENTRIES_BY_EPISODE.pop(entry.protected_episode_id, None)


__all__ = [
    "PredecessorBarrierResultV1",
    "await_predecessors",
    "capture_predecessor_watermark",
    "complete_predecessor",
    "register_predecessor",
]
