"""Lightweight scheduler backed by the ``scheduled_events`` MongoDB collection."""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime, timezone

from kazusa_ai_chatbot.db import (
    ScheduledEventDoc,
    cancel_pending_scheduled_event,
    insert_scheduled_event,
    list_pending_scheduler_events,
    mark_scheduled_event_completed,
    mark_scheduled_event_failed,
    mark_scheduled_event_running,
)
from kazusa_ai_chatbot.dispatcher.adapter_iface import AdapterRegistry
from kazusa_ai_chatbot.dispatcher.pending_index import PendingTaskIndex
from kazusa_ai_chatbot.dispatcher.task import DispatchContext, Task, parse_iso_datetime
from kazusa_ai_chatbot.dispatcher.tool_spec import ToolRegistry

logger = logging.getLogger(__name__)

_pending_tasks: dict[str, asyncio.Task] = {}
_tool_registry: ToolRegistry | None = None
_adapter_registry: AdapterRegistry | None = None
_pending_index: PendingTaskIndex | None = None


def configure_runtime(
    *,
    tool_registry: ToolRegistry,
    adapter_registry: AdapterRegistry,
    pending_index: PendingTaskIndex,
) -> None:
    """Install the runtime dependencies used when scheduled tasks fire.

    Args:
        tool_registry: Registered tool specifications and handlers.
        adapter_registry: Platform adapter registry used for tool delivery.
        pending_index: In-memory index of pending tasks for cleanup and dedup.
    """

    global _tool_registry, _adapter_registry, _pending_index
    _tool_registry = tool_registry
    _adapter_registry = adapter_registry
    _pending_index = pending_index


async def _handle_task(event: ScheduledEventDoc) -> None:
    """Rehydrate and execute one scheduled task document.

    Args:
        event: Persisted scheduled event document.
    """

    if _tool_registry is None or _adapter_registry is None:
        raise RuntimeError("Scheduler runtime is not configured")

    task = Task.from_scheduler_doc(event)
    ctx = DispatchContext.from_scheduler_doc(event)
    handler = _tool_registry.get(task.tool).handler
    await handler(task.args, ctx, _adapter_registry)


async def _fire_event(event: ScheduledEventDoc) -> None:
    """Execute a scheduled event when its time arrives.

    Args:
        event: Persisted scheduled event document to fire.
    """

    event_id = event["event_id"]
    await mark_scheduled_event_running(event_id)

    try:
        await _handle_task(event)
    except Exception as exc:
        logger.exception(f"Event {event_id} failed: {exc}")
        await mark_scheduled_event_failed(event_id)
    else:
        await mark_scheduled_event_completed(event_id)
    finally:
        _pending_tasks.pop(event_id, None)
        if _pending_index is not None:
            _pending_index.remove(event_id)


async def _schedule_task(event: ScheduledEventDoc) -> None:
    """Wait until the event's ``execute_at`` time, then fire it.

    Args:
        event: Persisted scheduled event document to await.
    """

    execute_at = parse_iso_datetime(event["execute_at"])
    now = datetime.now(timezone.utc)
    delay = max(0.0, (execute_at - now).total_seconds())
    if delay > 0:
        await asyncio.sleep(delay)
    await _fire_event(event)


async def schedule_event(event: ScheduledEventDoc) -> str:
    """Persist a new scheduled event and register it with asyncio.

    Args:
        event: Scheduled event document to persist.

    Returns:
        Persisted event identifier.
    """

    event.setdefault("event_id", str(uuid.uuid4()))
    event.setdefault("created_at", datetime.now(timezone.utc).isoformat())
    event.setdefault("status", "pending")

    await insert_scheduled_event(event)

    task = asyncio.create_task(_schedule_task(dict(event)))
    _pending_tasks[event["event_id"]] = task

    logger.info(f'Scheduled event {event["event_id"]} ({event["tool"]}) at {event["execute_at"]}')
    return event["event_id"]


async def load_pending_events() -> int:
    """Load all pending events from MongoDB and register them with asyncio.

    Returns:
        Number of pending events loaded.
    """

    count = 0
    for doc in await list_pending_scheduler_events():
        if "tool" not in doc or "execute_at" not in doc:
            logger.warning(f'Skipping legacy scheduled event without tool/execute_at: {doc.get("event_id")}')
            continue
        task = asyncio.create_task(_schedule_task(doc))
        _pending_tasks[doc["event_id"]] = task
        count += 1

    if count:
        logger.info(f'Loaded {count} pending scheduled events')
    return count


async def cancel_event(event_id: str) -> bool:
    """Cancel one pending event and stamp the cancellation time in MongoDB.

    Args:
        event_id: Persisted scheduler event identifier.

    Returns:
        ``True`` when the event existed and was marked cancelled.
    """

    task = _pending_tasks.pop(event_id, None)
    if task is not None:
        task.cancel()

    if _pending_index is not None:
        _pending_index.remove(event_id)

    return_value = await cancel_pending_scheduled_event(
        event_id,
        cancelled_at=datetime.now(timezone.utc).isoformat(),
    )
    return return_value


async def shutdown() -> None:
    """Cancel all in-memory scheduled tasks during service shutdown."""

    for task in _pending_tasks.values():
        task.cancel()
    _pending_tasks.clear()
    logger.info("Scheduler shut down")
