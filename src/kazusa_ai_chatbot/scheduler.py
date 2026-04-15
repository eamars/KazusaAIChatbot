"""Lightweight scheduler backed by the ``scheduled_events`` MongoDB collection.

On startup the scheduler loads all ``pending`` events and registers them with
asyncio.  When an event fires it is dispatched to the appropriate handler.

Currently only ``followup_message`` is implemented — other event types are
stubbed for future extension.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Awaitable

from kazusa_ai_chatbot.db import ScheduledEventDoc, get_db

logger = logging.getLogger(__name__)


# ── Event handler registry ──────────────────────────────────────────

_handlers: dict[str, Callable[[ScheduledEventDoc], Awaitable[None]]] = {}


def register_handler(event_type: str, handler: Callable[[ScheduledEventDoc], Awaitable[None]]) -> None:
    """Register an async handler for a given event_type."""
    _handlers[event_type] = handler


# ── Scheduler core ──────────────────────────────────────────────────

_pending_tasks: dict[str, asyncio.Task] = {}
_callback: Callable[[ScheduledEventDoc], Awaitable[None]] | None = None


def set_delivery_callback(cb: Callable[[ScheduledEventDoc], Awaitable[None]]) -> None:
    """Set the callback used by ``followup_message`` events to deliver a message.

    The callback receives the full event doc and is responsible for sending
    the message through the appropriate adapter.
    """
    global _callback
    _callback = cb


async def _fire_event(event: ScheduledEventDoc) -> None:
    """Execute when the scheduled time arrives."""
    event_id = event["event_id"]
    event_type = event["event_type"]

    db = await get_db()

    # Mark as running
    await db.scheduled_events.update_one(
        {"event_id": event_id},
        {"$set": {"status": "running"}},
    )

    try:
        handler = _handlers.get(event_type)
        if handler:
            await handler(event)
        else:
            logger.warning("No handler for event_type '%s' — skipping event %s", event_type, event_id)

        await db.scheduled_events.update_one(
            {"event_id": event_id},
            {"$set": {"status": "completed"}},
        )
    except Exception:
        logger.exception("Event %s failed", event_id)
        await db.scheduled_events.update_one(
            {"event_id": event_id},
            {"$set": {"status": "failed"}},
        )
    finally:
        _pending_tasks.pop(event_id, None)


async def _schedule_task(event: ScheduledEventDoc) -> None:
    """Wait until the event's scheduled time, then fire it."""
    scheduled_at = datetime.fromisoformat(event["scheduled_at"])
    now = datetime.now(timezone.utc)
    delay = max(0.0, (scheduled_at - now).total_seconds())

    if delay > 0:
        await asyncio.sleep(delay)

    await _fire_event(event)


async def schedule_event(event: ScheduledEventDoc) -> str:
    """Persist a new event to MongoDB and register it with asyncio.

    Returns the event_id.
    """
    event.setdefault("event_id", str(uuid.uuid4()))
    event.setdefault("created_at", datetime.now(timezone.utc).isoformat())
    event.setdefault("status", "pending")

    db = await get_db()
    await db.scheduled_events.insert_one(event)

    task = asyncio.create_task(_schedule_task(event))
    _pending_tasks[event["event_id"]] = task

    logger.info("Scheduled event %s (%s) at %s", event["event_id"], event["event_type"], event["scheduled_at"])
    return event["event_id"]


async def load_pending_events() -> int:
    """Load all ``pending`` events from the database and register them.

    Called once at startup.  Returns the number of events loaded.
    """
    db = await get_db()
    cursor = db.scheduled_events.find({"status": "pending"}).sort("scheduled_at", 1)
    count = 0
    async for doc in cursor:
        task = asyncio.create_task(_schedule_task(doc))
        _pending_tasks[doc["event_id"]] = task
        count += 1

    logger.info("Loaded %d pending scheduled events", count)
    return count


async def cancel_event(event_id: str) -> bool:
    """Cancel a pending event.  Returns True if it was found and cancelled."""
    task = _pending_tasks.pop(event_id, None)
    if task:
        task.cancel()

    db = await get_db()
    result = await db.scheduled_events.update_one(
        {"event_id": event_id, "status": "pending"},
        {"$set": {"status": "cancelled"}},
    )
    return result.modified_count > 0


async def shutdown() -> None:
    """Cancel all pending tasks.  Called during service shutdown."""
    for task in _pending_tasks.values():
        task.cancel()
    _pending_tasks.clear()
    logger.info("Scheduler shut down")


# ── Built-in handlers ───────────────────────────────────────────────

async def _handle_followup_message(event: ScheduledEventDoc) -> None:
    """Send a follow-up message via the delivery callback."""
    if _callback is None:
        logger.error("No delivery callback set — cannot send followup for event %s", event["event_id"])
        return
    await _callback(event)


register_handler("followup_message", _handle_followup_message)
