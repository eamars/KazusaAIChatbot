"""Unit tests for the scheduler's task-based execution path."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import kazusa_ai_chatbot.scheduler as scheduler
from kazusa_ai_chatbot.dispatcher import (
    AdapterRegistry,
    PendingTaskIndex,
    SendResult,
    Task,
    ToolRegistry,
    ToolSpec,
    build_send_message_tool,
)


class _StubAdapter:
    platform = "discord"

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def send_message(
        self,
        channel_id: str,
        text: str,
        *,
        reply_to_msg_id: str | None = None,
    ) -> SendResult:
        self.calls.append(
            {
                "channel_id": channel_id,
                "text": text,
                "reply_to_msg_id": reply_to_msg_id,
            }
        )
        return SendResult(
            platform=self.platform,
            channel_id=channel_id,
            message_id="m-1",
            sent_at=datetime.now(timezone.utc),
        )


def _make_mock_db() -> MagicMock:
    scheduled_events = MagicMock()
    scheduled_events.insert_one = AsyncMock()
    scheduled_events.update_one = AsyncMock(return_value=MagicMock(modified_count=1))
    scheduled_events.find = MagicMock(return_value=[])

    db = MagicMock()
    db.scheduled_events = scheduled_events
    return db


@pytest.fixture(autouse=True)
def _reset_scheduler_runtime():
    scheduler._pending_tasks.clear()
    scheduler._tool_registry = None
    scheduler._adapter_registry = None
    scheduler._pending_index = None
    yield
    for task in list(scheduler._pending_tasks.values()):
        task.cancel()
    scheduler._pending_tasks.clear()


def _configure_runtime() -> tuple[_StubAdapter, PendingTaskIndex]:
    tool_registry = ToolRegistry()
    tool_registry.register(build_send_message_tool())
    adapter_registry = AdapterRegistry()
    adapter = _StubAdapter()
    adapter_registry.register(adapter)
    pending_index = PendingTaskIndex()
    scheduler.configure_runtime(
        tool_registry=tool_registry,
        adapter_registry=adapter_registry,
        pending_index=pending_index,
    )
    return adapter, pending_index


@pytest.mark.asyncio
async def test_schedule_event_persists_tool_doc_defaults():
    db = _make_mock_db()
    _configure_runtime()

    with patch(
        "kazusa_ai_chatbot.db.scheduled_events.get_db",
        AsyncMock(return_value=db),
    ):
        event = {
            "tool": "send_message",
            "args": {
                "target_platform": "discord",
                "target_channel": "chan-1",
                "text": "hello",
            },
            "execute_at": "2099-01-01T00:00:00+00:00",
            "source_platform": "discord",
            "source_channel_id": "chan-1",
            "source_user_id": "u1",
            "source_message_id": "msg-1",
            "guild_id": None,
            "bot_role": "user",
        }
        event_id = await scheduler.schedule_event(event)

    persisted = db.scheduled_events.insert_one.await_args.args[0]
    assert persisted["event_id"] == event_id
    assert persisted["status"] == "pending"
    assert persisted["tool"] == "send_message"
    assert "created_at" in persisted


@pytest.mark.asyncio
async def test_cancel_event_sets_cancelled_status_and_timestamp():
    db = _make_mock_db()
    _configure_runtime()
    scheduler._pending_tasks["evt-1"] = asyncio.create_task(asyncio.sleep(999))

    with patch(
        "kazusa_ai_chatbot.db.scheduled_events.get_db",
        AsyncMock(return_value=db),
    ):
        ok = await scheduler.cancel_event("evt-1")

    assert ok is True
    update_call = db.scheduled_events.update_one.call_args
    assert update_call.args[0] == {"event_id": "evt-1", "status": "pending"}
    assert update_call.args[1]["$set"]["status"] == "cancelled"
    assert "cancelled_at" in update_call.args[1]["$set"]


@pytest.mark.asyncio
async def test_fire_event_executes_registered_tool_handler_and_marks_completed():
    db = _make_mock_db()
    adapter, pending_index = _configure_runtime()
    event = {
        "event_id": "evt-send",
        "tool": "send_message",
        "args": {
            "target_platform": "discord",
            "target_channel": "chan-1",
            "text": "hey there",
            "reply_to_msg_id": "msg-1",
        },
        "execute_at": "2026-04-25T00:00:00+00:00",
        "status": "pending",
        "source_platform": "discord",
        "source_channel_id": "chan-1",
        "source_user_id": "u1",
        "source_message_id": "msg-0",
        "guild_id": None,
        "bot_role": "user",
    }
    pending_index.add("evt-send", Task.from_scheduler_doc(event))

    with patch(
        "kazusa_ai_chatbot.db.scheduled_events.get_db",
        AsyncMock(return_value=db),
    ):
        await scheduler._fire_event(event)

    assert adapter.calls == [
        {
            "channel_id": "chan-1",
            "text": "hey there",
            "reply_to_msg_id": "msg-1",
        }
    ]
    updates = [call.args[1]["$set"]["status"] for call in db.scheduled_events.update_one.await_args_list]
    assert updates == ["running", "completed"]
    assert not pending_index.contains(Task.from_scheduler_doc(event))


@pytest.mark.asyncio
async def test_fire_event_marks_failed_when_handler_raises():
    db = _make_mock_db()
    tool_registry = ToolRegistry()

    async def _boom(args, ctx, adapters) -> None:
        del args, ctx, adapters
        raise RuntimeError("boom")

    tool_registry.register(
        ToolSpec(
            name="explode",
            description="always fail",
            args_schema={"type": "object", "properties": {}},
            handler=_boom,
        )
    )
    adapter_registry = AdapterRegistry()
    pending_index = PendingTaskIndex()
    scheduler.configure_runtime(
        tool_registry=tool_registry,
        adapter_registry=adapter_registry,
        pending_index=pending_index,
    )
    event = {
        "event_id": "evt-fail",
        "tool": "explode",
        "args": {},
        "execute_at": "2026-04-25T00:00:00+00:00",
        "status": "pending",
        "source_platform": "discord",
        "source_channel_id": "chan-1",
        "source_user_id": "u1",
        "source_message_id": "msg-0",
        "guild_id": None,
        "bot_role": "user",
    }

    with patch(
        "kazusa_ai_chatbot.db.scheduled_events.get_db",
        AsyncMock(return_value=db),
    ):
        await scheduler._fire_event(event)

    updates = [call.args[1]["$set"]["status"] for call in db.scheduled_events.update_one.await_args_list]
    assert updates == ["running", "failed"]
