"""Tests for service-owned event logging call sites."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from kazusa_ai_chatbot import chat_input_queue as queue_module
from kazusa_ai_chatbot import service as service_module
from kazusa_ai_chatbot.brain_service import post_turn as post_turn_module
from kazusa_ai_chatbot.config import CHARACTER_GLOBAL_USER_ID
from kazusa_ai_chatbot.time_boundary import build_turn_clock
from tests.cognition_core_v2_test_helpers import (
    canonical_episode_identity_snapshot,
    canonical_service_character_profile,
)


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


def _patch_character_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch deterministic latest-profile and episode-snapshot boundaries."""

    profile = canonical_service_character_profile(
        marker="service-event",
        global_user_id="character-global-id",
    )
    snapshot = canonical_episode_identity_snapshot(
        marker="service-event",
        global_user_id="character-global-id",
    )
    monkeypatch.setattr(
        service_module,
        "_load_latest_character_profile_snapshot",
        AsyncMock(return_value=profile),
    )
    monkeypatch.setattr(
        service_module,
        "load_latest_identity_for_episode",
        AsyncMock(return_value=snapshot),
    )
    monkeypatch.setattr(
        service_module,
        "set_conversation_source_episode_id",
        AsyncMock(return_value=1),
    )
    monkeypatch.setattr(
        service_module.llm_tracing,
        "ensure_llm_trace_run",
        AsyncMock(),
    )
    monkeypatch.setattr(
        service_module.llm_tracing,
        "finalize_llm_trace_run",
        AsyncMock(),
    )


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
    monkeypatch.setattr(
        service_module,
        "_commit_ingress_receipt",
        AsyncMock(return_value={
            "conversation_row_id": "row-1",
            "received_at": "2026-04-29T00:00:00+00:00",
        }),
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
async def test_progress_disposition_telemetry_is_trace_linked_and_sanitized(
    monkeypatch,
) -> None:
    """Post-turn progress telemetry carries only bounded opaque references."""

    record_event = AsyncMock()
    monkeypatch.setattr(
        post_turn_module.event_logging,
        "record_continuity_boundary_event",
        record_event,
    )

    await post_turn_module._record_post_turn_continuity_event(
        component="brain_service.post_turn",
        boundary="post_turn",
        status="persistence_failed",
        scope_kind="group_scene",
        write_disposition="write_failed",
        trace_ref="trace-opaque-1",
        operation_ref="progress-post-turn:opaque-1",
    )

    record_event.assert_awaited_once()
    fields = record_event.await_args.kwargs
    assert fields["trace_ref"] == "trace-opaque-1"
    assert fields["operation_ref"] == "progress-post-turn:opaque-1"
    serialized = json.dumps(fields, sort_keys=True)
    assert "private body" not in serialized
    assert "prompt_text" not in serialized
    assert "residue_text" not in serialized


def test_post_turn_continuity_ownership_is_documented() -> None:
    """Brain-service ICD keeps post-turn telemetry text-free and best effort."""

    readme = Path("src/kazusa_ai_chatbot/brain_service/README.md")
    text = readme.read_text(encoding="utf-8")
    assert "Post-turn continuity instrumentation is text-free" in text
    assert "record_continuity_boundary_event" in text
    assert "cannot interrupt post-turn persistence" in text


@pytest.mark.asyncio
async def test_process_queued_item_suppresses_routine_success_events(
    monkeypatch,
) -> None:
    """Successful chat processing should not duplicate history writes."""

    _patch_character_identity(monkeypatch)
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
        "_active_character_name_snapshot",
        "Character",
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
    monkeypatch.setattr(
        service_module,
        "upsert_post_turn_lifecycle_record",
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

    _patch_character_identity(monkeypatch)
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
        "_active_character_name_snapshot",
        "Character",
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
    save_conversation = AsyncMock(return_value="row-assistant")
    monkeypatch.setattr(
        service_module,
        "save_conversation",
        save_conversation,
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
    finalize_trace = AsyncMock()
    monkeypatch.setattr(
        service_module.llm_tracing,
        "finalize_llm_trace_run",
        finalize_trace,
    )

    class _Graph:
        """Raise from the graph boundary."""

        async def ainvoke(self, _state):
            raise RuntimeError("graph unavailable")

    monkeypatch.setattr(service_module, "_graph", _Graph())
    item = _item("msg", body_text="private failure text")

    await service_module._process_queued_chat_item(item)

    response = item.future.result()
    assert response.messages == [service_module.OPERATIONAL_FAILURE_NOTICE]
    assert response.content_type == "operational_error"
    assert response.delivery_tracking_id == ""
    assert response.operational_error is not None
    assert response.operational_error.error_code == "internal_invariant"
    assert response.operational_error.status == "failed"
    assert response.operational_error.exhausted is False
    assert response.operational_error.attempt_count == 1
    save_conversation.assert_not_awaited()
    finalize_trace.assert_awaited_once()
    assert finalize_trace.await_args.kwargs["final_dialog_count"] == 0
    assert finalize_trace.await_args.kwargs["delivery_tracking_id"] == ""
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

    _patch_character_identity(monkeypatch)
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
        "_active_character_name_snapshot",
        "Character",
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
    startup_profile = canonical_service_character_profile(
        marker="startup",
        global_user_id=CHARACTER_GLOBAL_USER_ID,
    )
    monkeypatch.setattr(
        service_module,
        "_load_startup_character_profile",
        AsyncMock(return_value=(
            {
                key: value
                for key, value in startup_profile.items()
                if key not in {"global_user_id", "cognition_state", "updated_at"}
            },
            {
                "cognition_state": startup_profile["cognition_state"],
                "updated_at": startup_profile["updated_at"],
            },
        )),
    )
    monkeypatch.setattr(
        service_module,
        "reconcile_identity_growth_post_commit",
        AsyncMock(return_value={
            "completed_count": 0,
            "failed_count": 0,
        }),
    )
    monkeypatch.setattr(
        service_module,
        "_load_latest_character_profile_snapshot",
        AsyncMock(return_value=startup_profile),
    )
    monkeypatch.setattr(
        service_module,
        "_hydrate_media_descriptor_cache",
        AsyncMock(return_value=0),
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
