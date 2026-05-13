"""Tests for dispatcher event logging instrumentation."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

from kazusa_ai_chatbot.db import DatabaseOperationError
from kazusa_ai_chatbot.dispatcher import dispatcher as dispatcher_module
from kazusa_ai_chatbot.dispatcher import handlers as handlers_module
from kazusa_ai_chatbot.dispatcher.adapter_iface import AdapterRegistry, SendResult
from kazusa_ai_chatbot.dispatcher.dispatcher import TaskDispatcher
from kazusa_ai_chatbot.dispatcher.evaluator import ToolCallEvaluator
from kazusa_ai_chatbot.dispatcher.pending_index import PendingTaskIndex
from kazusa_ai_chatbot.dispatcher.task import DispatchContext, RawToolCall
from kazusa_ai_chatbot.dispatcher.tool_spec import ToolRegistry


def _context() -> DispatchContext:
    """Build a stable dispatcher source context."""

    ctx = DispatchContext(
        source_platform="discord",
        source_channel_id="chan-private",
        source_user_id="global-user-1",
        source_message_id="source-msg-1",
        guild_id=None,
        bot_permission_role="user",
        now=datetime(2026, 5, 14, tzinfo=timezone.utc),
        source_channel_type="group",
    )
    return ctx


def _raw_send_call(text: str = "private candidate text") -> RawToolCall:
    """Build one send-message raw call."""

    raw = RawToolCall(
        tool="send_message",
        args={
            "target_platform": "discord",
            "target_channel": "chan-1",
            "target_channel_type": "group",
            "text": text,
        },
    )
    return raw


def _dispatcher_with_adapter(adapter: object) -> TaskDispatcher:
    """Build a dispatcher with one registered adapter."""

    tool_registry = ToolRegistry()
    tool_registry.register(handlers_module.build_send_message_tool())
    adapter_registry = AdapterRegistry()
    adapter_registry.register(adapter)
    evaluator = ToolCallEvaluator(tool_registry, adapter_registry)
    dispatcher = TaskDispatcher(evaluator, PendingTaskIndex())
    return dispatcher


class _Adapter:
    """Adapter fake for scheduler/handler tests."""

    platform = "discord"

    async def send_message(
        self,
        channel_id: str,
        text: str,
        *,
        channel_type: str,
        reply_to_msg_id: str | None = None,
    ) -> SendResult:
        """Return one delivery result while retaining normal adapter shape."""

        del text, channel_type, reply_to_msg_id
        result = SendResult(
            platform=self.platform,
            channel_id=channel_id,
            message_id="adapter-message-1",
            sent_at=datetime(2026, 5, 14, tzinfo=timezone.utc),
        )
        return result


class _FailingAdapter(_Adapter):
    """Adapter fake that raises during delivery."""

    async def send_message(
        self,
        channel_id: str,
        text: str,
        *,
        channel_type: str,
        reply_to_msg_id: str | None = None,
    ) -> SendResult:
        """Raise a deterministic send failure."""

        del channel_id, text, channel_type, reply_to_msg_id
        raise RuntimeError("adapter unavailable")


@pytest.mark.asyncio
async def test_dispatcher_records_scheduled_event_without_candidate_text(
    monkeypatch,
) -> None:
    """Successful scheduling should record scheduler refs, not raw args."""

    record_dispatcher_event = AsyncMock()
    record_database_operation_event = AsyncMock()
    monkeypatch.setattr(
        dispatcher_module.event_logging,
        "record_dispatcher_event",
        record_dispatcher_event,
    )
    monkeypatch.setattr(
        dispatcher_module.event_logging,
        "record_database_operation_event",
        record_database_operation_event,
    )
    monkeypatch.setattr(
        dispatcher_module.scheduler,
        "schedule_event",
        AsyncMock(return_value="event-1"),
    )
    dispatcher = _dispatcher_with_adapter(_Adapter())

    result = await dispatcher.dispatch([
        _raw_send_call("do not persist this candidate"),
    ], _context())

    assert result.rejected == []
    assert result.scheduled[0][1] == "event-1"
    record_database_operation_event.assert_awaited_once()
    db_kwargs = record_database_operation_event.await_args.kwargs
    assert db_kwargs["collection"] == "scheduled_events"
    assert db_kwargs["operation_kind"] == "insert_scheduled_event"
    assert db_kwargs["document_ref"] == "event-1"
    record_dispatcher_event.assert_awaited_once()
    dispatch_kwargs = record_dispatcher_event.await_args.kwargs
    assert dispatch_kwargs["status"] == "scheduled"
    assert dispatch_kwargs["scheduled_event_ids"] == ["event-1"]
    serialized = json.dumps(
        {"db": db_kwargs, "dispatch": dispatch_kwargs},
        ensure_ascii=False,
    )
    assert "do not persist this candidate" not in serialized
    assert "chan-private" not in serialized


@pytest.mark.asyncio
async def test_dispatcher_records_validation_rejection_without_args(
    monkeypatch,
) -> None:
    """Validation failures should emit coarse rejection codes only."""

    record_dispatcher_event = AsyncMock()
    monkeypatch.setattr(
        dispatcher_module.event_logging,
        "record_dispatcher_event",
        record_dispatcher_event,
    )
    dispatcher = _dispatcher_with_adapter(_Adapter())

    result = await dispatcher.dispatch([
        RawToolCall(
            tool="send_message",
            args={
                "target_platform": "discord",
                "target_channel": "chan-1",
                "text": "secret rejected candidate",
            },
        )
    ], _context())

    assert result.scheduled == []
    assert len(result.rejected) == 1
    record_dispatcher_event.assert_awaited_once()
    kwargs = record_dispatcher_event.await_args.kwargs
    assert kwargs["status"] == "rejected"
    assert kwargs["rejection_codes"] == ["missing_target_channel_type"]
    serialized = json.dumps(kwargs, ensure_ascii=False)
    assert "secret rejected candidate" not in serialized
    assert "chan-1" not in serialized


@pytest.mark.asyncio
async def test_dispatcher_records_scheduler_write_failure(monkeypatch) -> None:
    """Scheduler insert failures should record DB and dispatcher outcomes."""

    record_dispatcher_event = AsyncMock()
    record_database_operation_event = AsyncMock()
    monkeypatch.setattr(
        dispatcher_module.event_logging,
        "record_dispatcher_event",
        record_dispatcher_event,
    )
    monkeypatch.setattr(
        dispatcher_module.event_logging,
        "record_database_operation_event",
        record_database_operation_event,
    )
    monkeypatch.setattr(
        dispatcher_module.scheduler,
        "schedule_event",
        AsyncMock(side_effect=DatabaseOperationError("write failed")),
    )
    dispatcher = _dispatcher_with_adapter(_Adapter())

    result = await dispatcher.dispatch([
        _raw_send_call("secret scheduler candidate"),
    ], _context())

    assert result.scheduled == []
    assert len(result.rejected) == 1
    record_database_operation_event.assert_awaited_once()
    db_kwargs = record_database_operation_event.await_args.kwargs
    assert db_kwargs["status"] == "failed"
    assert db_kwargs["idempotency_result"] == (
        "exception:DatabaseOperationError"
    )
    record_dispatcher_event.assert_awaited_once()
    dispatch_kwargs = record_dispatcher_event.await_args.kwargs
    assert dispatch_kwargs["status"] == "failed"
    assert dispatch_kwargs["rejection_codes"] == ["scheduler_write_failed"]
    serialized = json.dumps(
        {"db": db_kwargs, "dispatch": dispatch_kwargs},
        ensure_ascii=False,
    )
    assert "secret scheduler candidate" not in serialized


@pytest.mark.asyncio
async def test_send_message_handler_records_success_without_adapter_response(
    monkeypatch,
) -> None:
    """Adapter success should record delivery metadata without raw response."""

    record_dispatcher_event = AsyncMock()
    monkeypatch.setattr(
        handlers_module.event_logging,
        "record_dispatcher_event",
        record_dispatcher_event,
    )
    adapter_registry = AdapterRegistry()
    adapter_registry.register(_Adapter())

    await handlers_module.handle_send_message(
        {
            "target_platform": "discord",
            "target_channel": "chan-1",
            "target_channel_type": "group",
            "text": "scheduled private text",
        },
        _context(),
        adapter_registry,
    )

    record_dispatcher_event.assert_awaited_once()
    kwargs = record_dispatcher_event.await_args.kwargs
    assert kwargs["component"] == handlers_module.HANDLER_COMPONENT
    assert kwargs["status"] == "succeeded"
    serialized = json.dumps(kwargs, ensure_ascii=False)
    assert "scheduled private text" not in serialized
    assert "adapter-message-1" not in serialized


@pytest.mark.asyncio
async def test_send_message_handler_records_send_failure_without_text(
    monkeypatch,
) -> None:
    """Adapter exceptions should record runtime metadata without message text."""

    record_dispatcher_event = AsyncMock()
    record_runtime_error_event = AsyncMock()
    monkeypatch.setattr(
        handlers_module.event_logging,
        "record_dispatcher_event",
        record_dispatcher_event,
    )
    monkeypatch.setattr(
        handlers_module.event_logging,
        "record_runtime_error_event",
        record_runtime_error_event,
    )
    adapter_registry = AdapterRegistry()
    adapter_registry.register(_FailingAdapter())

    with pytest.raises(RuntimeError):
        await handlers_module.handle_send_message(
            {
                "target_platform": "discord",
                "target_channel": "chan-1",
                "target_channel_type": "group",
                "text": "scheduled private failure text",
            },
            _context(),
            adapter_registry,
        )

    record_dispatcher_event.assert_awaited_once()
    dispatch_kwargs = record_dispatcher_event.await_args.kwargs
    assert dispatch_kwargs["status"] == "failed"
    assert dispatch_kwargs["adapter_available"] is True
    record_runtime_error_event.assert_awaited_once()
    runtime_kwargs = record_runtime_error_event.await_args.kwargs
    assert runtime_kwargs["error_class"] == "RuntimeError"
    serialized = json.dumps(
        {"dispatch": dispatch_kwargs, "runtime": runtime_kwargs},
        ensure_ascii=False,
    )
    assert "scheduled private failure text" not in serialized
