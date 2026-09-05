"""Tests for service-owned event logging call sites."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock

import pytest

from kazusa_ai_chatbot import chat_input_queue as queue_module
from kazusa_ai_chatbot import service as service_module
from kazusa_ai_chatbot.brain_service import post_turn as post_turn_module
from kazusa_ai_chatbot.config import CHARACTER_GLOBAL_USER_ID
from kazusa_ai_chatbot.time_boundary import build_turn_clock
from tests.cognition_test_helpers import (
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


def _patch_graph_worker_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[AsyncMock, AsyncMock, AsyncMock]:
    """Patch worker I/O while retaining the real graph retry policy."""

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
    return (
        record_database_operation_event,
        record_pipeline_turn_event,
        record_runtime_error_event,
    )




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
