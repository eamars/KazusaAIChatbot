"""Unit tests for the scheduler's task-based execution path."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import kazusa_ai_chatbot.scheduler as scheduler
from kazusa_ai_chatbot.db import scheduled_events as scheduled_events_module
from kazusa_ai_chatbot.dispatcher import handlers as handlers_module
from kazusa_ai_chatbot.dispatcher import (
    AdapterRegistry,
    DispatchContext,
    PendingTaskIndex,
    RawToolCall,
    SendResult,
    Task,
    ToolCallEvaluator,
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
        channel_type: str,
        reply_to_msg_id: str | None = None,
        delivery_mentions: list[dict] | None = None,
    ) -> SendResult:
        self.calls.append(
            {
                "channel_id": channel_id,
                "text": text,
                "reply_to_msg_id": reply_to_msg_id,
                "channel_type": channel_type,
                "delivery_mentions": delivery_mentions,
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


@pytest.fixture(autouse=True)
def _stub_dispatcher_event_logging(monkeypatch) -> None:
    """Keep scheduler tests off the event-log database."""

    monkeypatch.setattr(
        handlers_module.event_logging,
        "record_dispatcher_event",
        AsyncMock(),
    )
    monkeypatch.setattr(
        handlers_module.event_logging,
        "record_runtime_error_event",
        AsyncMock(),
    )
    monkeypatch.setattr(
        handlers_module,
        "record_assistant_outbound_message",
        AsyncMock(return_value="conversation-row-1"),
    )
    monkeypatch.setattr(
        handlers_module,
        "apply_assistant_delivery_receipt",
        AsyncMock(return_value=True),
    )


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


def _dispatch_context() -> DispatchContext:
    return DispatchContext(
        source_platform="discord",
        source_channel_id="chan-1",
        source_channel_type="group",
        source_user_id="user-1",
        source_message_id="msg-1",
        guild_id=None,
        bot_permission_role="user",
        now=datetime(2026, 5, 14, 0, 0, tzinfo=timezone.utc),
    )


def _target_user_mention() -> dict:
    return {
        "entity_kind": "user",
        "placement": "prefix",
        "platform_user_id": "platform-user-1",
        "global_user_id": "user-1",
        "display_name": "Target User",
        "requested_by": "dialog.mention_target_user",
    }


def test_evaluator_preserves_delivery_mentions_metadata():
    tool_registry = ToolRegistry()
    tool_registry.register(build_send_message_tool())
    adapter_registry = AdapterRegistry()
    adapter_registry.register(_StubAdapter())
    evaluator = ToolCallEvaluator(tool_registry, adapter_registry)
    mention = _target_user_mention()

    result = evaluator.evaluate(
        RawToolCall(
            tool="send_message",
            args={
                "target_platform": "discord",
                "target_channel": "chan-1",
                "target_channel_type": "group",
                "text": "plain scheduler text",
                "delivery_mentions": [mention],
            },
        ),
        _dispatch_context(),
    )

    assert result.ok is True
    assert result.task is not None
    assert result.task.args["text"] == "plain scheduler text"
    assert result.task.args["delivery_mentions"] == [mention]


@pytest.mark.asyncio
async def test_schedule_event_persists_tool_doc_metadata():
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
                "target_channel_type": "group",
                "text": "hello",
            },
            "execute_at": "2099-01-01T00:00:00+00:00",
            "source_platform": "discord",
            "source_channel_id": "chan-1",
            "source_channel_type": "group",
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
    assert persisted["source_channel_type"] == "group"
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
async def test_claim_pending_event_running_requires_pending_status():
    """Future workers should atomically claim only pending scheduler rows."""

    db = _make_mock_db()

    with patch(
        "kazusa_ai_chatbot.db.scheduled_events.get_db",
        AsyncMock(return_value=db),
    ):
        ok = await scheduled_events_module.claim_pending_scheduled_event_running(
            "evt-1",
        )

    assert ok is True
    update_call = db.scheduled_events.update_one.call_args
    assert update_call.args[0] == {"event_id": "evt-1", "status": "pending"}
    assert update_call.args[1] == {"$set": {"status": "running"}}


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
            "target_channel_type": "group",
            "text": "hey there",
            "reply_to_msg_id": "msg-1",
        },
        "execute_at": "2026-04-25T00:00:00+00:00",
        "status": "pending",
        "source_platform": "discord",
        "source_channel_id": "chan-1",
        "source_channel_type": "group",
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
            "channel_type": "group",
            "delivery_mentions": None,
        }
    ]
    updates = [call.args[1]["$set"]["status"] for call in db.scheduled_events.update_one.await_args_list]
    assert updates == ["running", "completed"]
    assert not pending_index.contains(Task.from_scheduler_doc(event))


@pytest.mark.asyncio
async def test_fire_event_passes_delivery_mentions_to_adapter():
    db = _make_mock_db()
    adapter, pending_index = _configure_runtime()
    mention = _target_user_mention()
    event = {
        "event_id": "evt-send-mention",
        "tool": "send_message",
        "args": {
            "target_platform": "discord",
            "target_channel": "chan-1",
            "target_channel_type": "group",
            "text": "hey there",
            "delivery_mentions": [mention],
        },
        "execute_at": "2026-04-25T00:00:00+00:00",
        "status": "pending",
        "source_platform": "discord",
        "source_channel_id": "chan-1",
        "source_channel_type": "group",
        "source_user_id": "u1",
        "source_message_id": "msg-0",
        "guild_id": None,
        "bot_role": "user",
    }
    pending_index.add("evt-send-mention", Task.from_scheduler_doc(event))

    with patch(
        "kazusa_ai_chatbot.db.scheduled_events.get_db",
        AsyncMock(return_value=db),
    ):
        await scheduler._fire_event(event)

    assert adapter.calls == [
        {
            "channel_id": "chan-1",
            "text": "hey there",
            "reply_to_msg_id": None,
            "channel_type": "group",
            "delivery_mentions": [mention],
        }
    ]
    assert not pending_index.contains(Task.from_scheduler_doc(event))


@pytest.mark.asyncio
async def test_fire_event_preserves_empty_delivery_mentions_metadata():
    db = _make_mock_db()
    adapter, pending_index = _configure_runtime()
    event = {
        "event_id": "evt-send-empty-mentions",
        "tool": "send_message",
        "args": {
            "target_platform": "discord",
            "target_channel": "chan-1",
            "target_channel_type": "group",
            "text": "plain text",
            "delivery_mentions": [],
        },
        "execute_at": "2026-04-25T00:00:00+00:00",
        "status": "pending",
        "source_platform": "discord",
        "source_channel_id": "chan-1",
        "source_channel_type": "group",
        "source_user_id": "u1",
        "source_message_id": "msg-0",
        "guild_id": None,
        "bot_role": "user",
    }
    pending_index.add("evt-send-empty-mentions", Task.from_scheduler_doc(event))

    with patch(
        "kazusa_ai_chatbot.db.scheduled_events.get_db",
        AsyncMock(return_value=db),
    ):
        await scheduler._fire_event(event)

    assert adapter.calls == [
        {
            "channel_id": "chan-1",
            "text": "plain text",
            "reply_to_msg_id": None,
            "channel_type": "group",
            "delivery_mentions": [],
        }
    ]
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
        "source_channel_type": "group",
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
