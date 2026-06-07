"""Tests for service-owned event logging call sites."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock

import pytest

from kazusa_ai_chatbot import chat_input_queue as queue_module
from kazusa_ai_chatbot import service as service_module
from kazusa_ai_chatbot.config import CHARACTER_GLOBAL_USER_ID
from kazusa_ai_chatbot.time_boundary import build_turn_clock


_TURN_CLOCK = build_turn_clock("2026-05-14 12:00:00")


def _request(message_id: str, *, body_text: str = "private body"):
    """Build a minimal chat request for service event tests."""

    request = service_module.ChatRequest(
        platform="qq",
        platform_channel_id="chan-1",
        channel_type="group",
        platform_message_id=message_id,
        platform_user_id="user-1",
        platform_bot_id="bot-1",
        display_name="User One",
        channel_name="Group",
        content_type="text",
        message_envelope={
            "body_text": body_text,
            "raw_wire_text": body_text,
            "mentions": [{
                "platform_user_id": "bot-1",
                "global_user_id": CHARACTER_GLOBAL_USER_ID,
                "entity_kind": "bot",
                "raw_text": "@bot",
            }],
            "attachments": [],
            "addressed_to_global_user_ids": [CHARACTER_GLOBAL_USER_ID],
            "broadcast": False,
        },
        debug_modes=service_module.DebugModesIn(),
    )
    return request


def _item(message_id: str, *, body_text: str = "private body"):
    """Build one queued item for direct worker-call tests."""

    future = asyncio.get_running_loop().create_future()
    item = queue_module.QueuedChatItem(
        sequence=1,
        request=_request(message_id, body_text=body_text),
        storage_timestamp_utc=_TURN_CLOCK["storage_timestamp_utc"],
        local_timestamp=_TURN_CLOCK["local_timestamp"],
        local_time_context=_TURN_CLOCK["local_time_context"],
        future=future,
    )
    return item


@pytest.mark.asyncio
async def test_enqueue_suppresses_routine_accepted_queue_event(
    monkeypatch,
) -> None:
    """Routine enqueue should rely on conversation history for audit."""

    await service_module._stop_chat_input_worker()
    service_module._chat_input_queue.reset_for_test()
    record_queue_intake_event = AsyncMock()
    monkeypatch.setattr(
        service_module.event_logging,
        "record_queue_intake_event",
        record_queue_intake_event,
    )
    monkeypatch.setattr(
        service_module,
        "_ensure_chat_input_worker_started",
        lambda: None,
    )

    chat_task = asyncio.create_task(
        service_module._enqueue_chat_request(
            _request("enqueue", body_text="do not store this"),
        )
    )
    await asyncio.sleep(0)

    record_queue_intake_event.assert_not_awaited()

    queued_item = service_module._chat_input_queue.pop_left_for_test()
    queued_item.future.set_result(service_module.ChatResponse())
    response = await asyncio.wait_for(chat_task, timeout=1.0)

    assert response.messages == []
    await service_module._stop_chat_input_worker()
    service_module._chat_input_queue.reset_for_test()


@pytest.mark.asyncio
async def test_process_queued_item_suppresses_routine_success_events(
    monkeypatch,
) -> None:
    """Successful chat processing should not duplicate history writes."""

    record_database_operation_event = AsyncMock()
    record_pipeline_turn_event = AsyncMock()
    record_runtime_error_event = AsyncMock()
    monkeypatch.setattr(
        service_module.event_logging,
        "record_database_operation_event",
        record_database_operation_event,
    )
    monkeypatch.setattr(
        service_module.event_logging,
        "record_pipeline_turn_event",
        record_pipeline_turn_event,
    )
    monkeypatch.setattr(
        service_module.event_logging,
        "record_runtime_error_event",
        record_runtime_error_event,
    )
    monkeypatch.setattr(
        service_module,
        "_static_character_profile",
        {"name": "Character"},
    )
    monkeypatch.setattr(service_module, "_runtime_character_state", {})
    monkeypatch.setattr(
        service_module,
        "_ensure_character_global_identity",
        AsyncMock(return_value="character-global-id"),
    )
    monkeypatch.setattr(
        service_module,
        "_resolve_queued_user",
        AsyncMock(return_value=("global-user-1", {})),
    )
    monkeypatch.setattr(
        service_module,
        "_resolve_message_envelope_identities",
        AsyncMock(return_value=_request("msg").message_envelope.model_dump()),
    )
    monkeypatch.setattr(
        service_module,
        "get_conversation_history",
        AsyncMock(return_value=[]),
    )
    monkeypatch.setattr(
        service_module,
        "_hydrate_reply_context",
        AsyncMock(return_value={}),
    )
    monkeypatch.setattr(
        service_module,
        "_save_user_message_from_item",
        AsyncMock(return_value="row-user"),
    )
    monkeypatch.setattr(
        service_module,
        "save_conversation",
        AsyncMock(return_value="row-assistant"),
    )
    monkeypatch.setattr(
        service_module,
        "_refresh_runtime_character_state",
        AsyncMock(),
    )
    monkeypatch.setattr(
        service_module,
        "build_promoted_reflection_context",
        AsyncMock(return_value={}),
    )
    monkeypatch.setattr(
        service_module,
        "_save_assistant_message",
        AsyncMock(),
    )

    class _Graph:
        """Return a visible response without invoking any LLM."""

        async def ainvoke(self, _state):
            result = {
                "should_respond": True,
                "use_reply_feature": False,
                "final_dialog": ["visible response"],
                "future_promises": [],
                "consolidation_state": {},
            }
            return result

    monkeypatch.setattr(service_module, "_graph", _Graph())
    item = _item("msg", body_text="highly private user text")

    await service_module._process_queued_chat_item(item)

    assert item.future.result().messages == ["visible response"]
    record_database_operation_event.assert_not_awaited()
    record_pipeline_turn_event.assert_not_awaited()
    record_runtime_error_event.assert_not_awaited()


@pytest.mark.asyncio
async def test_graph_failure_records_runtime_error_and_failed_pipeline(
    monkeypatch,
) -> None:
    """Recoverable graph failures should produce sanitized failure telemetry."""

    record_database_operation_event = AsyncMock()
    record_pipeline_turn_event = AsyncMock()
    record_runtime_error_event = AsyncMock()
    monkeypatch.setattr(
        service_module.event_logging,
        "record_database_operation_event",
        record_database_operation_event,
    )
    monkeypatch.setattr(
        service_module.event_logging,
        "record_pipeline_turn_event",
        record_pipeline_turn_event,
    )
    monkeypatch.setattr(
        service_module.event_logging,
        "record_runtime_error_event",
        record_runtime_error_event,
    )
    monkeypatch.setattr(
        service_module,
        "_static_character_profile",
        {"name": "Character"},
    )
    monkeypatch.setattr(service_module, "_runtime_character_state", {})
    monkeypatch.setattr(
        service_module,
        "_ensure_character_global_identity",
        AsyncMock(return_value="character-global-id"),
    )
    monkeypatch.setattr(
        service_module,
        "_resolve_queued_user",
        AsyncMock(return_value=("global-user-1", {})),
    )
    monkeypatch.setattr(
        service_module,
        "_resolve_message_envelope_identities",
        AsyncMock(return_value=_request("msg").message_envelope.model_dump()),
    )
    monkeypatch.setattr(
        service_module,
        "get_conversation_history",
        AsyncMock(return_value=[]),
    )
    monkeypatch.setattr(
        service_module,
        "_hydrate_reply_context",
        AsyncMock(return_value={}),
    )
    monkeypatch.setattr(
        service_module,
        "_save_user_message_from_item",
        AsyncMock(return_value="row-user"),
    )
    monkeypatch.setattr(
        service_module,
        "save_conversation",
        AsyncMock(return_value="row-assistant"),
    )
    monkeypatch.setattr(
        service_module,
        "_refresh_runtime_character_state",
        AsyncMock(),
    )
    monkeypatch.setattr(
        service_module,
        "build_promoted_reflection_context",
        AsyncMock(return_value={}),
    )

    class _Graph:
        """Raise from the graph boundary."""

        async def ainvoke(self, _state):
            raise RuntimeError("graph unavailable")

    monkeypatch.setattr(service_module, "_graph", _Graph())
    item = _item("msg", body_text="private failure text")

    await service_module._process_queued_chat_item(item)

    assert item.future.result().messages == [
        "Character is busy right now, please try again later."
    ]
    record_runtime_error_event.assert_awaited_once()
    runtime_kwargs = record_runtime_error_event.await_args.kwargs
    assert runtime_kwargs["error_class"] == "RuntimeError"
    assert runtime_kwargs["status"] == "failed"
    record_pipeline_turn_event.assert_awaited_once()
    pipeline_kwargs = record_pipeline_turn_event.await_args.kwargs
    assert pipeline_kwargs["status"] == "failed"
    assert pipeline_kwargs["final_outcome"] == "graph_error"
    serialized = json.dumps(
        {"runtime": runtime_kwargs, "pipeline": pipeline_kwargs},
        ensure_ascii=False,
    )
    assert "private failure text" not in serialized


@pytest.mark.asyncio
async def test_user_persistence_failure_keeps_failure_telemetry(
    monkeypatch,
) -> None:
    """Missing conversation-history commits should still emit failure events."""

    record_database_operation_event = AsyncMock()
    record_pipeline_turn_event = AsyncMock()
    record_runtime_error_event = AsyncMock()
    monkeypatch.setattr(
        service_module.event_logging,
        "record_database_operation_event",
        record_database_operation_event,
    )
    monkeypatch.setattr(
        service_module.event_logging,
        "record_pipeline_turn_event",
        record_pipeline_turn_event,
    )
    monkeypatch.setattr(
        service_module.event_logging,
        "record_runtime_error_event",
        record_runtime_error_event,
    )
    monkeypatch.setattr(
        service_module,
        "_static_character_profile",
        {"name": "Character"},
    )
    monkeypatch.setattr(service_module, "_runtime_character_state", {})
    monkeypatch.setattr(
        service_module,
        "_ensure_character_global_identity",
        AsyncMock(return_value="character-global-id"),
    )
    monkeypatch.setattr(
        service_module,
        "_resolve_queued_user",
        AsyncMock(return_value=("global-user-1", {})),
    )
    monkeypatch.setattr(
        service_module,
        "_resolve_message_envelope_identities",
        AsyncMock(return_value=_request("msg").message_envelope.model_dump()),
    )
    monkeypatch.setattr(
        service_module,
        "get_conversation_history",
        AsyncMock(return_value=[]),
    )
    monkeypatch.setattr(
        service_module,
        "_hydrate_reply_context",
        AsyncMock(return_value={}),
    )
    monkeypatch.setattr(
        service_module,
        "_save_user_message_from_item",
        AsyncMock(return_value=None),
    )

    class _Graph:
        """Expose a mock graph call for fail-closed assertions."""

        def __init__(self):
            self.ainvoke = AsyncMock(return_value={})

    graph = _Graph()
    monkeypatch.setattr(service_module, "_graph", graph)
    item = _item("msg", body_text="private uncommitted text")

    await service_module._process_queued_chat_item(item)

    with pytest.raises(RuntimeError):
        await item.future
    graph.ainvoke.assert_not_awaited()
    record_database_operation_event.assert_awaited_once()
    db_kwargs = record_database_operation_event.await_args.kwargs
    assert db_kwargs["operation_kind"] == "insert_user_message"
    assert db_kwargs["status"] == "failed"
    assert db_kwargs["idempotency_result"] == "not_committed"
    record_pipeline_turn_event.assert_awaited_once()
    pipeline_kwargs = record_pipeline_turn_event.await_args.kwargs
    assert pipeline_kwargs["status"] == "failed"
    assert pipeline_kwargs["final_outcome"] == "user_persist_failed"
    serialized = json.dumps(
        {"db": db_kwargs, "pipeline": pipeline_kwargs},
        ensure_ascii=False,
    )
    assert "private uncommitted text" not in serialized
    record_runtime_error_event.assert_not_awaited()


@pytest.mark.asyncio
async def test_lifespan_records_process_and_resource_events(monkeypatch) -> None:
    """Startup and shutdown should emit bounded lifecycle metadata."""

    record_process_event = AsyncMock()
    record_resource_health_event = AsyncMock()
    monkeypatch.setattr(
        service_module.event_logging,
        "record_process_event",
        record_process_event,
    )
    monkeypatch.setattr(
        service_module.event_logging,
        "record_resource_health_event",
        record_resource_health_event,
    )
    monkeypatch.setattr(service_module, "db_bootstrap", AsyncMock())
    monkeypatch.setattr(
        service_module,
        "_hydrate_rag_initializer_cache",
        AsyncMock(return_value=0),
    )
    monkeypatch.setattr(
        service_module,
        "get_character_profile",
        AsyncMock(return_value={"name": "Character"}),
    )
    monkeypatch.setattr(
        service_module,
        "_refresh_runtime_character_state",
        AsyncMock(),
    )
    monkeypatch.setattr(
        service_module,
        "_build_graph",
        lambda: object(),
    )
    monkeypatch.setattr(service_module.mcp_manager, "start", AsyncMock())
    monkeypatch.setattr(service_module.mcp_manager, "stop", AsyncMock())
    monkeypatch.setattr(
        service_module,
        "CALENDAR_SCHEDULER_ENABLED",
        False,
        raising=False,
    )
    monkeypatch.setattr(
        service_module,
        "Pending" + "TaskIndex",
        _ForbiddenLegacyRuntime,
        raising=False,
    )
    legacy_scheduler = getattr(service_module, "scheduler", None)
    if legacy_scheduler is not None:
        monkeypatch.setattr(
            legacy_scheduler,
            "configure_runtime",
            _forbidden_legacy_sync,
            raising=False,
        )
        monkeypatch.setattr(
            legacy_scheduler,
            "load" + "_pending_events",
            _forbidden_legacy_async,
            raising=False,
        )
        monkeypatch.setattr(
            legacy_scheduler,
            "shutdown",
            _forbidden_legacy_async,
            raising=False,
        )
    monkeypatch.setattr(
        service_module,
        "render_llm_route_table",
        lambda: "routes",
    )
    monkeypatch.setattr(
        service_module,
        "_ensure_chat_input_worker_started",
        lambda: None,
    )
    monkeypatch.setattr(service_module, "_stop_chat_input_worker", AsyncMock())
    monkeypatch.setattr(service_module, "close_db", AsyncMock())
    monkeypatch.setattr(service_module, "REFLECTION_CYCLE_ENABLED", False)
    monkeypatch.setattr(service_module, "SELF_COGNITION_ENABLED", False)
    monkeypatch.setattr(
        service_module,
        "BACKGROUND_WORK_WORKER_ENABLED",
        False,
        raising=False,
    )

    async with service_module.lifespan(service_module.app):
        pass

    process_types = [
        call.kwargs["event_type"]
        for call in record_process_event.await_args_list
    ]
    assert process_types == ["startup", "shutdown"]
    resource_names = [
        call.kwargs["resource_name"]
        for call in record_resource_health_event.await_args_list
    ]
    assert resource_names == [
        "mongo",
        "rag_initializer_cache",
        "media_descriptor_cache",
        "mcp_manager",
    ]


class _ForbiddenLegacyRuntime:
    """Fail if the retired delayed-task runtime is still constructed."""

    def __init__(self) -> None:
        raise AssertionError("retired scheduler runtime should not start")


async def _forbidden_legacy_async(*args, **kwargs) -> None:
    """Fail if an async legacy runtime hook is still called."""

    raise AssertionError("retired scheduler runtime should not be called")


def _forbidden_legacy_sync(*args, **kwargs) -> None:
    """Fail if a sync legacy runtime hook is still called."""

    raise AssertionError("retired scheduler runtime should not be called")
