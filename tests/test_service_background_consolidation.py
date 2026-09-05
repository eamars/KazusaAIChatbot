"""Unit tests for background consolidation scheduling in the service layer."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from fastapi import BackgroundTasks, Request

from kazusa_ai_chatbot import service as service_module
from kazusa_ai_chatbot.action_spec.registry import (
    APPLY_MEMORY_LIFECYCLE_UPDATE_CAPABILITY,
)
from kazusa_ai_chatbot.brain_service import post_turn as post_turn_module
from kazusa_ai_chatbot.chat_input_queue import QueuedChatItem
from kazusa_ai_chatbot.cognition_shared.state_models import (
    build_character_production_state,
)
from kazusa_ai_chatbot.consolidation import (
    character_operational_state as operational_state_module,
)
from kazusa_ai_chatbot.consolidation.character_operational_state import (
    CharacterOperationalExecutionContext,
)
from kazusa_ai_chatbot.llm_tracing import failure_capsule
from kazusa_ai_chatbot.time_boundary import build_turn_clock
from tests.cognition_test_helpers import (
    canonical_episode_identity_snapshot,
    canonical_service_character_profile,
)

_CONSOLIDATION_TURN_CLOCK = build_turn_clock("2026-04-25 18:00:58")
_GRAPH_TURN_CLOCK = build_turn_clock("2026-04-25 18:07:24")


class _MappingState(Mapping):
    """Small mapping-like wrapper that is not a literal ``dict``."""

    def __init__(self, payload: dict):
        self._payload = payload

    def __getitem__(self, key):
        return self._payload[key]

    def __iter__(self):
        return iter(self._payload)

    def __len__(self):
        return len(self._payload)

    def get(self, key, default=None):
        return self._payload.get(key, default)


class _FakeGraph:
    """Return a fixed graph result for one service.chat invocation."""

    def __init__(self, result: dict):
        self._result = result

    async def ainvoke(self, _state):
        return self._result


class _FailingGraph:
    """Raise a fixed error for one service.chat invocation."""

    async def ainvoke(self, _state):
        raise RuntimeError("graph failed")


def _chat_request(
    *,
    message_id: str = "msg-1",
    channel_type: str = "private",
    debug_modes: service_module.DebugModesIn | None = None,
) -> service_module.ChatRequest:
    """Build a minimal chat request for service-layer tests.

    Args:
        message_id: Platform message identifier for the request.
        channel_type: Channel surface for the request.
        debug_modes: Optional debug-mode flags for the request.

    Returns:
        ChatRequest with deterministic ASCII payload fields.
    """

    request = service_module.ChatRequest(
        platform="qq",
        platform_channel_id="chan-1",
        channel_type=channel_type,
        platform_message_id=message_id,
        platform_user_id="user-1",
        platform_bot_id="bot-1",
        display_name="Test User",
        channel_name="Private",
        message_envelope={
            "body_text": "please remember this",
            "raw_wire_text": "please remember this",
            "mentions": [],
            "attachments": [],
            "addressed_to_global_user_ids": [
                service_module.CHARACTER_GLOBAL_USER_ID,
            ],
            "broadcast": False,
        },
        debug_modes=debug_modes or service_module.DebugModesIn(),
    )
    return request


def _chat_http_request() -> Request:
    """Build the HTTP request context required by the chat authorization seam."""

    return Request({"type": "http", "headers": []})


def _consolidation_state() -> dict:
    """Return the common consolidation-state fixture used by chat tests.

    Returns:
        Consolidation state with all fields required by background recorders.
    """

    return_value = {
        "storage_timestamp_utc": (
            _CONSOLIDATION_TURN_CLOCK["storage_timestamp_utc"]
        ),
        "local_time_context": _CONSOLIDATION_TURN_CLOCK["local_time_context"],
        "platform": "qq",
        "platform_channel_id": "chan-1",
        "platform_message_id": "msg-1",
        "global_user_id": "global-user-1",
        "user_name": "Test User",
        "user_profile": {"global_user_id": "global-user-1", "relationship_state": 500},
        "character_profile": {"name": "Character"},
        "action_directives": {"linguistic_directives": {"content_plan": {}}},
        "internal_monologue": "test",
        "final_dialog": ["ok"],
        "interaction_subtext": "",
        "emotional_appraisal": "",
        "character_intent": "PROVIDE",
        "logical_stance": "CONFIRM",
        "rag_result": {},
        "decontextualized_input": "please remember this",
        "chat_history_recent": [],
        "interaction_logical_turns": [],
        "active_turn_conversation_row_ids": ["conversation-row-1"],
        "active_turn_conversation_source_refs": [{
            "ref_kind": "conversation_row",
            "ref_id": "conversation-row-1",
            "occurred_at": "2026-04-24T18:00:00+00:00",
        }],
        "llm_trace_id": "trace-background-consolidation",
        "conversation_episode_state": None,
        "conversation_progress_turn_outcome": "visible_response",
    }
    return return_value


def _settled_visible_trace() -> dict:
    """Return a minimal canonical settled trace for consumer tests."""

    return {
        "schema_version": "episode_trace.v2",
        "episode_id": "episode-001",
        "trigger_source": "user_message",
        "terminal_status": "completed_visible",
        "cognition_refs": [],
        "action_specs": [],
        "action_results": [],
        "surface_outputs": [{
            "schema_version": "surface_output.v1",
            "surface_kind": "text",
            "visibility": "user_visible",
            "action_attempt_id": None,
            "fragments": ["ok"],
            "artifact_refs": [],
            "delivery_intent": "deliver_now",
            "surface_role": "ordinary",
            "goal_continuation_ref": None,
            "created_at": _CONSOLIDATION_TURN_CLOCK[
                "storage_timestamp_utc"
            ],
        }],
        "attempt_diagnostics": [],
        "delivery_correlation": {
            "schema_version": "delivery_correlation.v1",
            "delivery_intent": "deliver_now",
            "tracking_id": "delivery-001",
            "receipt_status": "pending",
            "receipt_ref": "",
        },
        "created_at": _CONSOLIDATION_TURN_CLOCK["storage_timestamp_utc"],
        "settled_at": _CONSOLIDATION_TURN_CLOCK["storage_timestamp_utc"],
    }


def _boundary_profile() -> dict:
    """Return a complete character boundary-profile fixture.

    Returns:
        Boundary profile with all configured fields present.
    """

    return_value = {
        "self_integrity": 0.8,
        "control_sensitivity": 0.3,
        "compliance_strategy": "comply",
        "relational_override": 0.25,
        "control_intimacy_misread": 0.2,
        "boundary_recovery": "rebound",
        "authority_skepticism": 0.35,
    }
    return return_value


def _graph_result(consolidation_state: Mapping | dict | None = None) -> dict:
    """Build a fixed successful service graph result.

    Args:
        consolidation_state: Optional consolidation snapshot for background work.

    Returns:
        Graph result shaped like persona_supervisor2 output.
    """

    if consolidation_state is None:
        consolidation_state = _consolidation_state()

    return_value = {
        "should_respond": True,
        "response_action": "proceed",
        "use_reply_feature": False,
        "final_dialog": ["ok"],
        "future_promises": [],
        "consolidation_state": consolidation_state,
    }
    return return_value


def _memory_lifecycle_action_spec(unit_id: str) -> dict[str, object]:
    """Build one executable memory-lifecycle action fixture."""

    return_value = {
        "schema_version": "action_spec.v1",
        "kind": APPLY_MEMORY_LIFECYCLE_UPDATE_CAPABILITY,
        "cognition_mode": "deliberative",
        "source_refs": [],
        "target": {
            "schema_version": "action_target.v1",
            "target_kind": "memory_unit",
            "target_id": unit_id,
            "owner": "user_memory_units",
            "scope": {"unit_type": "active_commitment"},
        },
        "params": {
            "memory_kind": "user_memory_unit",
            "unit_type": "active_commitment",
            "unit_id": unit_id,
            "lifecycle_decision": "fulfilled",
            "due_at": None,
        },
        "urgency": "background",
        "visibility": "private",
        "deadline": None,
        "continuation": {
            "schema_version": "action_continuation.v1",
            "mode": "none",
            "episode_type": None,
            "max_depth": 0,
            "include_result_as": None,
        },
        "reason": "The final visible dialog says the commitment is complete.",
    }
    return return_value


def _memory_lifecycle_action_result(unit_id: str) -> dict[str, object]:
    """Build one executed memory-lifecycle action-result fixture."""

    return_value = {
        "schema_version": "action_result.v1",
        "action_attempt_id": f"action_attempt:{unit_id}",
        "action_kind": APPLY_MEMORY_LIFECYCLE_UPDATE_CAPABILITY,
        "handler_owner": "memory_lifecycle",
        "status": "executed",
        "visibility": "private",
        "result_summary": f"memory lifecycle updated: {unit_id}",
        "result_refs": [],
        "continuation": {
            "schema_version": "action_continuation.v1",
            "mode": "none",
            "episode_type": None,
            "max_depth": 0,
            "include_result_as": None,
        },
        "completed_at": "2026-04-25T18:00:58+12:00",
    }
    return return_value


def _post_turn_lifecycle_state() -> dict:
    """Build a completed visible-turn state for lifecycle background tests."""

    state = _consolidation_state()
    state["cognitive_episode"] = {
        "episode_id": "episode-001",
        "trigger_source": "user_message",
    }
    state["action_specs"] = []
    state["action_results"] = []
    state["surface_outputs"] = [
        {
            "schema_version": "surface_output.v1",
            "surface_kind": "text",
            "visibility": "user_visible",
            "action_attempt_id": None,
            "fragments": state["final_dialog"],
            "artifact_refs": [],
            "delivery_intent": "deliver_now",
            "created_at": state["storage_timestamp_utc"],
        }
    ]
    state["episode_trace"] = {
        "schema_version": "episode_trace.v2",
        "episode_id": "episode-001",
        "trigger_source": "user_message",
        "terminal_status": "completed_visible",
        "cognition_refs": [],
        "action_specs": [],
        "action_results": [],
        "surface_outputs": state["surface_outputs"],
        "attempt_diagnostics": [],
        "delivery_correlation": {
            "schema_version": "delivery_correlation.v1",
            "delivery_intent": "deliver_now",
            "tracking_id": "delivery-001",
            "receipt_status": "pending",
            "receipt_ref": "",
        },
        "created_at": state["storage_timestamp_utc"],
        "settled_at": state["storage_timestamp_utc"],
    }
    return state


def _patch_chat_dependencies(
    monkeypatch,
    graph,
    *,
    patch_post_turn_lifecycle: bool = True,
) -> None:
    """Patch service dependencies that are outside queue-worker behavior.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        graph: Fake graph object installed as the service graph.
        patch_post_turn_lifecycle: Whether to stub the real post-turn DB path.

    Returns:
        None.
    """

    monkeypatch.setattr(
        service_module,
        "_active_character_name_snapshot",
        "Character",
    )
    character_profile = canonical_service_character_profile(
        marker="background-consolidation",
        global_user_id=service_module.CHARACTER_GLOBAL_USER_ID,
    )
    monkeypatch.setattr(
        service_module,
        "_runtime_character_state",
        {
            "cognition_state": character_profile["cognition_state"],
            "updated_at": character_profile["updated_at"],
        },
    )
    identity_snapshot = canonical_episode_identity_snapshot(
        marker="background-consolidation",
        global_user_id=service_module.CHARACTER_GLOBAL_USER_ID,
    )
    monkeypatch.setattr(
        service_module,
        "_load_latest_character_profile_snapshot",
        AsyncMock(return_value=character_profile),
    )
    monkeypatch.setattr(
        service_module,
        "load_latest_identity_for_episode",
        AsyncMock(return_value=identity_snapshot),
    )
    monkeypatch.setattr(
        service_module,
        "get_character_runtime_state",
        AsyncMock(return_value={
            "mood": "fresh mood",
            "vibe_check": "fresh vibe",
            "character_reflection": "fresh reflection",
        }),
    )
    monkeypatch.setattr(
        service_module,
        "_ensure_character_global_identity",
        AsyncMock(return_value=service_module.CHARACTER_GLOBAL_USER_ID),
    )
    monkeypatch.setattr(
        service_module,
        "resolve_global_user_id",
        AsyncMock(return_value="global-user-1"),
    )
    monkeypatch.setattr(
        service_module,
        "get_user_profile",
        AsyncMock(
            return_value={"global_user_id": "global-user-1", "relationship_state": 500},
        ),
    )
    monkeypatch.setattr(
        service_module,
        "get_conversation_history",
        AsyncMock(return_value=[]),
    )
    monkeypatch.setattr(
        service_module,
        "get_user_message_by_row_id",
        AsyncMock(return_value=None),
    )
    monkeypatch.setattr(
        service_module,
        "set_conversation_source_episode_id",
        AsyncMock(return_value=1),
    )
    monkeypatch.setattr(
        service_module,
        "build_promoted_reflection_context",
        AsyncMock(return_value={}),
    )
    monkeypatch.setattr(
        service_module,
        "_hydrate_reply_context",
        AsyncMock(return_value={}),
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

    async def _frontline(_state):
        return {
            "decision": {
                "intake_action": "start",
                "append_target": "none",
                "prelude_targets": [],
                "reason": "deterministic service fixture",
            },
            "attempt_diagnostics": [],
        }

    async def _settled(_state):
        return {
            "decision": {
                "response_action": "proceed",
                "reason_to_respond": "deterministic service fixture",
                "use_reply_feature": False,
                "channel_topic": "",
                "indirect_speech_context": "",
            },
            "attempt_diagnostics": [],
        }

    async def _media(state):
        return {
            "user_multimedia_input": state.get("user_multimedia_input", []),
            "additional_media_present": False,
        }

    monkeypatch.setattr(
        service_module,
        "frontline_relevance_agent",
        _frontline,
    )
    monkeypatch.setattr(service_module, "relevance_agent", _settled)
    monkeypatch.setattr(
        service_module,
        "multimedia_descriptor_agent",
        _media,
    )
    monkeypatch.setattr(
        service_module.event_logging,
        "record_database_operation_event",
        AsyncMock(),
    )
    monkeypatch.setattr(
        service_module.event_logging,
        "record_pipeline_turn_event",
        AsyncMock(),
    )
    monkeypatch.setattr(
        service_module.event_logging,
        "record_runtime_error_event",
        AsyncMock(),
    )
    monkeypatch.setattr(
        service_module.event_logging,
        "record_queue_intake_event",
        AsyncMock(),
    )
    monkeypatch.setattr(
        service_module,
        "save_conversation",
        AsyncMock(return_value="conversation-row-1"),
    )
    monkeypatch.setattr(
        service_module,
        "upsert_post_turn_lifecycle_record",
        AsyncMock(),
    )
    monkeypatch.setattr(
        service_module,
        "_run_internal_monologue_residue_record_background",
        AsyncMock(),
    )
    if patch_post_turn_lifecycle:
        monkeypatch.setattr(
            service_module,
            "_run_post_turn_memory_lifecycle_background",
            AsyncMock(side_effect=lambda state: state),
        )
    monkeypatch.setattr(service_module, "_graph", graph)


async def _reset_queue_state() -> None:
    """Reset global queue state between service endpoint tests.

    Returns:
        None.
    """

    await service_module._stop_chat_input_worker()
    service_module._chat_input_queue.reset_for_test()


def test_self_cognition_worker_receives_adapter_registry_provider() -> None:
    """Service startup should pass the runtime adapter registry to the worker."""

    source_text = Path(service_module.__file__).read_text(encoding="utf-8")
    call_start = source_text.index(
        "_self_cognition_worker_handle = start_self_cognition_worker("
    )
    call_block = source_text[call_start: call_start + 500]

    assert "adapter_registry_provider" in call_block
    assert "_adapter_registry" in call_block
































@pytest.mark.asyncio
async def test_delivery_receipt_endpoint_returns_updated_and_not_found(monkeypatch):
    """Delivery receipt endpoint should expose idempotent update status."""
    apply_receipt = AsyncMock(side_effect=[True, False])
    monkeypatch.setattr(
        service_module,
        "apply_assistant_delivery_receipt",
        apply_receipt,
    )

    updated = await service_module.delivery_receipt(
        service_module.DeliveryReceiptRequest(
            platform="qq",
            platform_channel_id="chan-1",
            delivery_tracking_id="delivery-1",
            logical_message_index=1,
            platform_message_id="platform-123",
            delivered_at="2026-05-07T11:00:00+00:00",
            adapter="napcat",
        )
    )
    missed = await service_module.delivery_receipt(
        service_module.DeliveryReceiptRequest(
            platform="qq",
            platform_channel_id="chan-1",
            delivery_tracking_id="delivery-2",
            logical_message_index=0,
            platform_message_id="platform-456",
        )
    )

    assert updated.status == "updated"
    assert updated.updated is True
    assert missed.status == "not_found"
    assert missed.updated is False
    assert apply_receipt.await_count == 2
    assert apply_receipt.await_args_list[0].kwargs["logical_message_index"] == 1
    assert apply_receipt.await_args_list[1].kwargs["logical_message_index"] == 0


@pytest.mark.asyncio
async def test_hydrate_reply_context_fills_missing_metadata_from_delivered_row(
    monkeypatch,
) -> None:
    """Historical bot replies should use the current active brain name."""
    lookup = AsyncMock(return_value={
        "platform_user_id": "bot-1",
        "display_name": '杏山千纱',
        "body_text": "previous assistant answer",
    })
    monkeypatch.setattr(
        service_module,
        "get_conversation_by_platform_message_id",
        lookup,
    )
    monkeypatch.setattr(
        service_module,
        "_active_character_name_snapshot",
        "Character",
    )
    request = _chat_request()
    request.message_envelope.reply = service_module.ReplyTargetIn(
        platform_message_id="platform-123",
    )

    reply_context = await service_module._hydrate_reply_context(request)

    assert reply_context["reply_to_message_id"] == "platform-123"
    assert reply_context["reply_to_platform_user_id"] == "bot-1"
    assert reply_context["reply_to_display_name"] == "Character"
    assert reply_context["reply_excerpt"] == "previous assistant answer"
    lookup.assert_awaited_once_with(
        platform="qq",
        platform_channel_id="chan-1",
        platform_message_id="platform-123",
    )


@pytest.mark.asyncio
async def test_hydrate_reply_context_overrides_adapter_bot_display_name(
    monkeypatch,
) -> None:
    """Typed adapter bot metadata should use the active brain character name."""

    lookup = AsyncMock(return_value=None)
    monkeypatch.setattr(
        service_module,
        "get_conversation_by_platform_message_id",
        lookup,
    )
    monkeypatch.setattr(
        service_module,
        "_active_character_name_snapshot",
        "Character",
    )
    request = _chat_request()
    request.message_envelope.reply = service_module.ReplyTargetIn(
        platform_message_id="platform-123",
        platform_user_id="bot-1",
        display_name='杏山千纱',
        excerpt="adapter excerpt",
    )

    reply_context = await service_module._hydrate_reply_context(request)

    assert reply_context["reply_to_display_name"] == "Character"
    lookup.assert_awaited_once()


@pytest.mark.asyncio
async def test_hydrate_reply_context_keeps_adapter_supplied_metadata(
    monkeypatch,
) -> None:
    """Adapter-provided reply fields should stay authoritative."""
    lookup = AsyncMock(return_value={
        "platform_user_id": "db-user",
        "display_name": "DB Name",
        "body_text": "db excerpt",
        "attachments": [
            {
                "media_type": "image/png",
                "description": "stored reply image summary",
            },
        ],
    })
    monkeypatch.setattr(
        service_module,
        "get_conversation_by_platform_message_id",
        lookup,
    )
    request = _chat_request()
    request.message_envelope.reply = service_module.ReplyTargetIn(
        platform_message_id="platform-123",
        platform_user_id="adapter-user",
        display_name="Adapter Name",
        excerpt="adapter excerpt",
    )

    reply_context = await service_module._hydrate_reply_context(request)

    assert reply_context["reply_to_platform_user_id"] == "adapter-user"
    assert reply_context["reply_to_display_name"] == "Adapter Name"
    assert reply_context["reply_excerpt"] == "adapter excerpt"
    assert reply_context["reply_attachments"] == [
        {
            "media_kind": "image",
            "description": "stored reply image summary",
            "summary_status": "available",
        },
    ]
    lookup.assert_awaited_once_with(
        platform="qq",
        platform_channel_id="chan-1",
        platform_message_id="platform-123",
    )


@pytest.mark.asyncio
async def test_user_message_storage_rejects_invalid_semantic_fields(
    monkeypatch,
) -> None:
    """Invalid semantic envelope fields should fail before persistence."""

    save_conversation = AsyncMock()
    monkeypatch.setattr(service_module, "save_conversation", save_conversation)
    request = _chat_request()
    request.message_envelope.body_text = "@mentioned-user-1 poisoned"
    loop = asyncio.get_running_loop()
    item = QueuedChatItem(
        sequence=1,
        request=request,
        storage_timestamp_utc="2026-07-03T00:00:00+00:00",
        local_timestamp="2026-07-03 00:00:00",
        local_time_context=_GRAPH_TURN_CLOCK["local_time_context"],
        future=loop.create_future(),
    )

    with pytest.raises(ValueError, match="body_text"):
        await service_module._save_user_message_from_item(
            item,
            global_user_id="global-user-1",
            reply_context={},
        )

    save_conversation.assert_not_awaited()


@pytest.mark.asyncio
async def test_user_message_storage_rejects_platform_qualified_semantic_label(
    monkeypatch,
) -> None:
    """Platform-qualified adapter fallbacks should fail before persistence."""

    save_conversation = AsyncMock()
    monkeypatch.setattr(service_module, "save_conversation", save_conversation)
    request = _chat_request()
    request.message_envelope.body_text = "@qq-user:673225019 poisoned"
    loop = asyncio.get_running_loop()
    item = QueuedChatItem(
        sequence=1,
        request=request,
        storage_timestamp_utc="2026-07-03T00:00:00+00:00",
        local_timestamp="2026-07-03 00:00:00",
        local_time_context=_GRAPH_TURN_CLOCK["local_time_context"],
        future=loop.create_future(),
    )

    with pytest.raises(ValueError, match="platform-qualified"):
        await service_module._save_user_message_from_item(
            item,
            global_user_id="global-user-1",
            reply_context={},
        )

    save_conversation.assert_not_awaited()










@pytest.mark.asyncio
async def test_progress_background_passes_character_boundary_profile(monkeypatch):
    """Progress recorder receives the character boundary profile from the snapshot."""

    state = _consolidation_state()
    state["episode_trace"] = _settled_visible_trace()
    boundary_profile = _boundary_profile()
    state["character_profile"]["boundary_profile"] = boundary_profile
    record_turn_progress = AsyncMock(return_value={
        "written": True,
        "turn_count": 1,
        "continuity": "same_episode",
        "status": "active",
        "cache_updated": True,
    })
    monkeypatch.setattr(
        service_module,
        "record_turn_progress",
        record_turn_progress,
    )

    await service_module._run_conversation_progress_record_background(state)

    record_turn_progress.assert_awaited_once()
    record_input = record_turn_progress.await_args.kwargs["record_input"]
    assert record_input["character_name"] == "Character"
    assert record_input["boundary_profile"] == boundary_profile
    assert record_input["current_turn_source_refs"][0] == {
        "ref_kind": "conversation_row",
        "ref_id": "conversation-row-1",
        "occurred_at": "2026-04-24T18:00:00+00:00",
    }


@pytest.mark.asyncio
async def test_progress_background_requires_character_boundary_profile(monkeypatch):
    """Missing character boundary configuration is a state-shape bug."""

    state = _consolidation_state()
    state["episode_trace"] = _settled_visible_trace()
    record_turn_progress = AsyncMock()
    monkeypatch.setattr(
        service_module,
        "record_turn_progress",
        record_turn_progress,
    )

    with pytest.raises(KeyError, match="boundary_profile"):
        await service_module._run_conversation_progress_record_background(state)

    record_turn_progress.assert_not_awaited()






def test_brain_terminal_requires_v2_output_update_and_commit_marker() -> None:
    """Terminal handling should fail closed before an incomplete V2 commit."""

    from kazusa_ai_chatbot.brain_service.graph import validate_v2_terminal_state

    with pytest.raises(ValueError, match="not committed"):
        validate_v2_terminal_state({})  # type: ignore[arg-type]
    assert validate_v2_terminal_state({
        "cognition_core_output": {},
        "cognition_state_update": {},
        "cognition_state_committed": True,
    }) == {}  # type: ignore[arg-type]






@pytest.mark.asyncio
async def test_operational_carryover_failure_persists_protected_capsule(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Background carry-over failures persist only inside the protected capsule."""

    persisted = asyncio.Event()
    written_rows: list[dict] = []
    completed_receipts: list[dict] = []
    now = "2026-08-02T00:00:00Z"

    async def capture_insert_trace_step(document: dict) -> str:
        written_rows.append(dict(document))
        persisted.set()
        return "step"

    async def raising_carryover(**kwargs):
        del kwargs
        raise RuntimeError("provider boundary failure")

    async def fake_completion(**kwargs):
        return {
            "status": kwargs["status"],
            "error_code": kwargs["error_code"],
        }

    monkeypatch.setattr(failure_capsule, "LLM_TRACE_CAPTURE_MODE", "metadata")
    monkeypatch.setattr(
        failure_capsule.db_llm_tracing,
        "insert_trace_step",
        capture_insert_trace_step,
    )
    monkeypatch.setattr(
        operational_state_module,
        "run_character_carryover_cognition",
        raising_carryover,
    )
    monkeypatch.setattr(
        operational_state_module,
        "_remaining_lease_seconds",
        lambda context: 30.0,
    )
    monkeypatch.setattr(
        operational_state_module,
        "_complete_or_in_memory_failure",
        fake_completion,
    )
    monkeypatch.setattr(
        service_module,
        "complete_predecessor",
        lambda token, receipt: completed_receipts.append(dict(receipt)),
    )

    context = CharacterOperationalExecutionContext(
        claim={
            "claim_status": "claimed",
            "receipt": {"lease_expires_at": "2026-08-02T00:00:30Z"},
        },
        base_state=build_character_production_state(updated_at=now),
        lease_owner="operational-test-lease",
        registered_at=now,
    )
    settled_state = {
        "character_operational_execution_context": context,
        "character_operational_predecessor_token": {
            "protected_episode_id": "episode:protected-capsule",
            "process_sequence": 1,
        },
        "character_operational_episode_id": "episode:protected-capsule",
        "character_operational_effective_at": now,
        "llm_trace_id": "trace-background-consolidation",
    }
    consolidation_result = {
        "character_operational_work": {
            "status": "selected",
            "evidence": [{
                "source_key": "episode_trace",
                "source_kind": "episode",
                "source_id": "episode:protected-capsule",
                "occurred_at": now,
                "semantic_text": "closed operational event",
            }],
        },
    }

    with caplog.at_level(
        logging.ERROR,
        logger="kazusa_ai_chatbot.consolidation.character_operational_state",
    ):
        await service_module._run_operational_work_from_consolidation(
            settled_state,
            consolidation_result,
        )

    await asyncio.wait_for(persisted.wait(), timeout=1.0)

    capsule_rows = [
        row
        for row in written_rows
        if row.get("capture_reason") == "cognition_failure_capsule"
    ]
    assert len(capsule_rows) == 1
    capsule = capsule_rows[0]["capsule"]
    assert capsule["trace_id"] == "trace-background-consolidation"
    assert capsule["entrypoint"] == "character_operational_carryover"
    assert capsule["outcome"] == "partial_failure"
    failure_kinds = [
        event["failure_kind"]
        for event in capsule["failure_events"]
    ]
    assert "carryover_execution_error" in failure_kinds
    cause_chain = capsule["failure_events"][0]["cause_chain"]
    assert cause_chain[0]["type"] == "RuntimeError"
    assert cause_chain[0]["message"] == "provider boundary failure"
    assert failure_capsule._CURRENT_SESSION.get() is None
    assert "provider boundary failure" not in caplog.text
    assert "transaction_failed" in caplog.text
    assert "RuntimeError" in caplog.text
    assert completed_receipts[0]["status"] == "failed"
    assert completed_receipts[0]["error_code"] == "transaction_failed"


@pytest.mark.asyncio
async def test_accepted_task_operational_failure_persists_protected_capsule(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Accepted-task carry-over failures stay inside the protected capsule."""

    persisted = asyncio.Event()
    written_rows: list[dict] = []
    completed_receipts: list[dict] = []
    now = "2026-08-02T00:00:00Z"
    trace_id = "trace-accepted-task-operational"

    async def capture_insert_trace_step(document: dict) -> str:
        written_rows.append(dict(document))
        persisted.set()
        return "step"

    async def raising_carryover(**kwargs):
        del kwargs
        raise RuntimeError("provider boundary failure")

    async def fake_completion(**kwargs):
        return {
            "status": kwargs["status"],
            "error_code": kwargs["error_code"],
        }

    monkeypatch.setattr(failure_capsule, "LLM_TRACE_CAPTURE_MODE", "metadata")
    monkeypatch.setattr(
        failure_capsule.db_llm_tracing,
        "insert_trace_step",
        capture_insert_trace_step,
    )
    monkeypatch.setattr(
        operational_state_module,
        "run_character_carryover_cognition",
        raising_carryover,
    )
    monkeypatch.setattr(
        operational_state_module,
        "_remaining_lease_seconds",
        lambda context: 30.0,
    )
    monkeypatch.setattr(
        operational_state_module,
        "_complete_or_in_memory_failure",
        fake_completion,
    )
    monkeypatch.setattr(
        service_module,
        "complete_predecessor",
        lambda token, receipt: completed_receipts.append(dict(receipt)),
    )

    context = CharacterOperationalExecutionContext(
        claim={
            "claim_status": "claimed",
            "receipt": {"lease_expires_at": "2026-08-02T00:00:30Z"},
        },
        base_state=build_character_production_state(updated_at=now),
        lease_owner="operational-test-lease",
        registered_at=now,
    )
    token = {
        "protected_episode_id": "episode:accepted-task-failure",
        "process_sequence": 1,
    }
    episode = {
        "schema_version": "cognitive_episode.v1",
        "episode_id": "episode:accepted-task-failure",
        "trigger_source": "tool_result",
        "created_at": now,
    }
    settled_trace = {"settled_at": now}
    operational_state = {
        "llm_trace_id": trace_id,
        "internal_monologue": "internal accepted-task thought",
    }

    with caplog.at_level(
        logging.ERROR,
        logger="kazusa_ai_chatbot.consolidation.character_operational_state",
    ):
        receipt = await service_module._run_accepted_task_operational_work(
            context=context,
            token=token,
            episode=episode,
            settled_trace=settled_trace,
            final_dialog=["accepted task result"],
            operational_state=operational_state,
            effective_at=now,
        )

    await asyncio.wait_for(persisted.wait(), timeout=1.0)

    capsule_rows = [
        row
        for row in written_rows
        if row.get("capture_reason") == "cognition_failure_capsule"
    ]
    assert len(capsule_rows) == 1
    capsule = capsule_rows[0]["capsule"]
    assert capsule["trace_id"] == trace_id
    assert capsule["entrypoint"] == "character_operational_carryover"
    assert capsule["outcome"] == "partial_failure"
    failure_kinds = [
        event["failure_kind"]
        for event in capsule["failure_events"]
    ]
    assert "carryover_execution_error" in failure_kinds
    cause_chain = capsule["failure_events"][0]["cause_chain"]
    assert cause_chain[0]["type"] == "RuntimeError"
    assert cause_chain[0]["message"] == "provider boundary failure"
    assert failure_capsule._CURRENT_SESSION.get() is None
    assert "provider boundary failure" not in caplog.text
    assert "transaction_failed" in caplog.text
    assert "RuntimeError" in caplog.text
    assert receipt["status"] == "failed"
    assert receipt["error_code"] == "transaction_failed"
    assert completed_receipts[0]["status"] == "failed"
    assert completed_receipts[0]["error_code"] == "transaction_failed"
