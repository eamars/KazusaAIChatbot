"""Unit tests for background consolidation scheduling in the service layer."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from fastapi import BackgroundTasks

from kazusa_ai_chatbot import service as service_module
from kazusa_ai_chatbot.action_spec.registry import (
    APPLY_MEMORY_LIFECYCLE_UPDATE_CAPABILITY,
)
from kazusa_ai_chatbot.brain_service import post_turn as post_turn_module
from kazusa_ai_chatbot.chat_input_queue import QueuedChatItem
from kazusa_ai_chatbot.time_boundary import build_turn_clock


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
        "decontexualized_input": "please remember this",
        "chat_history_recent": [],
    }
    return return_value


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
        "schema_version": "episode_trace.v1",
        "episode_id": "episode-001",
        "trigger_source": "user_message",
        "cognition_refs": [],
        "action_specs": [],
        "action_results": [],
        "surface_outputs": state["surface_outputs"],
        "created_at": state["storage_timestamp_utc"],
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
        "_static_character_profile",
        {"name": "Character"},
    )
    monkeypatch.setattr(
        service_module,
        "_runtime_character_state",
        {
            "mood": "old mood",
            "vibe_check": "old vibe",
            "character_reflection": "old reflection",
        },
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
        AsyncMock(return_value="character-global-id"),
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
        "_hydrate_reply_context",
        AsyncMock(return_value={}),
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
    monkeypatch.setattr(service_module, "save_conversation", AsyncMock())
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
async def test_chat_queues_background_consolidation_for_mapping_state(monkeypatch):
    """Mapping-like consolidation state should still queue the background task."""

    await _reset_queue_state()
    save_assistant_message = AsyncMock()
    progress_recorder = AsyncMock()
    consolidation_done = asyncio.Event()

    async def _consolidation_runner(_state):
        consolidation_done.set()

    monkeypatch.setattr(
        service_module,
        "_save_assistant_message",
        save_assistant_message,
    )
    monkeypatch.setattr(
        service_module,
        "_run_conversation_progress_record_background",
        progress_recorder,
    )
    monkeypatch.setattr(
        service_module,
        "_run_consolidation_background",
        _consolidation_runner,
    )
    consolidation_state = _MappingState(_consolidation_state())
    _patch_chat_dependencies(monkeypatch, _FakeGraph(_graph_result(consolidation_state)))

    background_tasks = BackgroundTasks()
    response = await service_module.chat(
        _chat_request(),
        background_tasks,
    )

    assert response.messages == ["ok"]
    assert len(background_tasks.tasks) == 0
    await asyncio.wait_for(consolidation_done.wait(), timeout=1.0)

    save_assistant_message.assert_awaited_once()
    progress_recorder.assert_awaited_once()
    await _reset_queue_state()


@pytest.mark.asyncio
async def test_chat_response_uses_true_reply_feature_from_graph(monkeypatch):
    """The chat response should quote when any graph stage requests it."""

    await _reset_queue_state()
    save_assistant_message = AsyncMock()
    progress_recorder = AsyncMock()
    consolidation_runner = AsyncMock()

    monkeypatch.setattr(
        service_module,
        "_save_assistant_message",
        save_assistant_message,
    )
    monkeypatch.setattr(
        service_module,
        "_run_conversation_progress_record_background",
        progress_recorder,
    )
    monkeypatch.setattr(
        service_module,
        "_run_consolidation_background",
        consolidation_runner,
    )
    graph_result = _graph_result()
    graph_result["use_reply_feature"] = True
    _patch_chat_dependencies(monkeypatch, _FakeGraph(graph_result))

    response = await service_module.chat(
        _chat_request(),
        BackgroundTasks(),
    )

    assert response.messages == ["ok"]
    assert response.use_reply_feature is True
    await _reset_queue_state()


@pytest.mark.asyncio
async def test_chat_response_adds_inline_delivery_mentions_without_channel_gate(
    monkeypatch,
):
    """Chat responses should carry minimal candidates for authored tags."""

    await _reset_queue_state()
    monkeypatch.setattr(
        service_module,
        "_save_assistant_message",
        AsyncMock(),
    )
    monkeypatch.setattr(
        service_module,
        "_run_conversation_progress_record_background",
        AsyncMock(),
    )
    monkeypatch.setattr(
        service_module,
        "_run_consolidation_background",
        AsyncMock(),
    )
    graph_result = _graph_result()
    graph_result["final_dialog"] = ["@Test User ok"]
    _patch_chat_dependencies(monkeypatch, _FakeGraph(graph_result))

    response = await service_module.chat(
        _chat_request(),
        BackgroundTasks(),
    )

    assert response.messages == ["@Test User ok"]
    assert response.delivery_mentions == [
        {
            "entity_kind": "user",
            "platform_user_id": "user-1",
            "display_name": "Test User",
        }
    ]
    await _reset_queue_state()


@pytest.mark.asyncio
async def test_chat_response_reply_feature_keeps_inline_delivery_mentions(
    monkeypatch,
):
    """Reply anchoring should not suppress authored inline tags."""

    await _reset_queue_state()
    monkeypatch.setattr(
        service_module,
        "_save_assistant_message",
        AsyncMock(),
    )
    monkeypatch.setattr(
        service_module,
        "_run_conversation_progress_record_background",
        AsyncMock(),
    )
    monkeypatch.setattr(
        service_module,
        "_run_consolidation_background",
        AsyncMock(),
    )
    graph_result = _graph_result()
    graph_result["use_reply_feature"] = True
    graph_result["final_dialog"] = ["@Test User ok"]
    _patch_chat_dependencies(monkeypatch, _FakeGraph(graph_result))

    response = await service_module.chat(
        _chat_request(channel_type="group"),
        BackgroundTasks(),
    )

    assert response.messages == ["@Test User ok"]
    assert response.use_reply_feature is True
    assert response.delivery_mentions == [
        {
            "entity_kind": "user",
            "display_name": "Test User",
            "platform_user_id": "user-1",
        }
    ]
    await _reset_queue_state()


@pytest.mark.asyncio
async def test_chat_response_adds_multiple_inline_delivery_mentions(
    monkeypatch,
):
    """Chat responses should expose every exact matched scoped user."""

    await _reset_queue_state()
    monkeypatch.setattr(
        service_module,
        "_save_assistant_message",
        AsyncMock(),
    )
    monkeypatch.setattr(
        service_module,
        "_run_conversation_progress_record_background",
        AsyncMock(),
    )
    monkeypatch.setattr(
        service_module,
        "_run_consolidation_background",
        AsyncMock(),
    )
    graph_result = _graph_result()
    graph_result["final_dialog"] = ["@Test User @Moca ok"]
    graph_result["scope_users"] = [
        {
            "display_name": "Moca",
            "platform_user_id": "user-2",
            "global_user_id": "global-user-2",
        }
    ]
    _patch_chat_dependencies(monkeypatch, _FakeGraph(graph_result))

    response = await service_module.chat(
        _chat_request(),
        BackgroundTasks(),
    )

    assert response.messages == ["@Test User @Moca ok"]
    assert response.delivery_mentions == [
        {
            "entity_kind": "user",
            "display_name": "Test User",
            "platform_user_id": "user-1",
        },
        {
            "entity_kind": "user",
            "display_name": "Moca",
            "platform_user_id": "user-2",
        },
    ]
    await _reset_queue_state()


@pytest.mark.asyncio
async def test_chat_response_keeps_inline_mention_when_scope_repeats_current_user(
    monkeypatch,
):
    """Current-user tags should survive duplicate same-identity scope rows."""

    await _reset_queue_state()
    monkeypatch.setattr(
        service_module,
        "_save_assistant_message",
        AsyncMock(),
    )
    monkeypatch.setattr(
        service_module,
        "_run_conversation_progress_record_background",
        AsyncMock(),
    )
    monkeypatch.setattr(
        service_module,
        "_run_consolidation_background",
        AsyncMock(),
    )
    graph_result = _graph_result()
    graph_result["final_dialog"] = ["@Test User ok"]
    graph_result["scope_users"] = [
        {
            "display_name": "Test User",
            "platform_user_id": "user-1",
            "global_user_id": "global-user-1",
        }
    ]
    _patch_chat_dependencies(monkeypatch, _FakeGraph(graph_result))

    response = await service_module.chat(
        _chat_request(),
        BackgroundTasks(),
    )

    assert response.messages == ["@Test User ok"]
    assert response.delivery_mentions == [
        {
            "entity_kind": "user",
            "display_name": "Test User",
            "platform_user_id": "user-1",
        }
    ]
    await _reset_queue_state()


@pytest.mark.asyncio
async def test_chat_response_preserves_message_sequence_for_inline_mentions(
    monkeypatch,
):
    """Inline mention discovery should not collapse response messages."""

    await _reset_queue_state()
    monkeypatch.setattr(
        service_module,
        "_save_assistant_message",
        AsyncMock(),
    )
    monkeypatch.setattr(
        service_module,
        "_run_conversation_progress_record_background",
        AsyncMock(),
    )
    monkeypatch.setattr(
        service_module,
        "_run_consolidation_background",
        AsyncMock(),
    )
    graph_result = _graph_result()
    graph_result["final_dialog"] = [
        "@Test User first",
        "second @Moca",
    ]
    graph_result["scope_users"] = [
        {
            "display_name": "Moca",
            "platform_user_id": "user-2",
            "global_user_id": "global-user-2",
        }
    ]
    _patch_chat_dependencies(monkeypatch, _FakeGraph(graph_result))

    response = await service_module.chat(
        _chat_request(),
        BackgroundTasks(),
    )

    assert response.messages == ["@Test User first", "second @Moca"]
    assert response.delivery_mentions == [
        {
            "entity_kind": "user",
            "display_name": "Test User",
            "platform_user_id": "user-1",
        },
        {
            "entity_kind": "user",
            "display_name": "Moca",
            "platform_user_id": "user-2",
        },
    ]
    await _reset_queue_state()


@pytest.mark.asyncio
async def test_chat_response_tracks_deliverable_assistant_row(monkeypatch):
    """Non-empty chat responses should carry the assistant row tracking ID."""

    await _reset_queue_state()
    service_module._clear_latest_cognition_graph()
    save_assistant_message = AsyncMock()
    monkeypatch.setattr(
        service_module,
        "_save_assistant_message",
        save_assistant_message,
    )
    monkeypatch.setattr(
        service_module,
        "_run_conversation_progress_record_background",
        AsyncMock(),
    )
    monkeypatch.setattr(
        service_module,
        "_run_consolidation_background",
        AsyncMock(),
    )
    _patch_chat_dependencies(monkeypatch, _FakeGraph(_graph_result()))

    response = await service_module.chat(
        _chat_request(),
        BackgroundTasks(),
    )

    assert response.messages == ["ok"]
    assert response.delivery_tracking_id
    assert response.cognition_graph is not None
    graph = response.cognition_graph
    assert graph["run_id"] == response.delivery_tracking_id
    assert graph["status"] == "completed"
    node_ids = {node["id"] for node in graph["nodes"]}
    assert {"l2.reasoning", "l2.memory", "l2.actions"} <= node_ids
    edge_kinds = {edge["kind"] for edge in graph["edges"]}
    assert {"fork", "join"} <= edge_kinds
    graph_text = repr(graph)
    assert "test" in graph_text
    assert "PROVIDE" in graph_text
    assert "CONFIRM" in graph_text
    detail_text = repr([node["detail"] for node in graph["nodes"]])
    assert "prompt" not in detail_text
    assert "embedding" not in detail_text
    save_assistant_message.assert_awaited_once()
    saved_result = save_assistant_message.await_args.args[0]
    assert saved_result["delivery_tracking_id"] == response.delivery_tracking_id

    latest = await service_module.ops_latest_cognition_graph()
    assert latest.cognition_graph is not None
    assert latest.cognition_graph["run_id"] == response.delivery_tracking_id
    assert latest.cognition_graph["nodes"] == graph["nodes"]
    await _reset_queue_state()


@pytest.mark.asyncio
async def test_chat_response_waits_for_assistant_persistence(monkeypatch):
    """The chat endpoint must not release visible output before persistence."""

    await _reset_queue_state()
    save_started = asyncio.Event()
    save_can_finish = asyncio.Event()

    async def _save_assistant_message(_result):
        save_started.set()
        await save_can_finish.wait()

    monkeypatch.setattr(
        service_module,
        "_save_assistant_message",
        _save_assistant_message,
    )
    monkeypatch.setattr(
        service_module,
        "_run_conversation_progress_record_background",
        AsyncMock(),
    )
    monkeypatch.setattr(
        service_module,
        "_run_consolidation_background",
        AsyncMock(),
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
    _patch_chat_dependencies(monkeypatch, _FakeGraph(_graph_result()))

    chat_task = asyncio.create_task(service_module.chat(
        _chat_request(),
        BackgroundTasks(),
    ))
    await asyncio.wait_for(save_started.wait(), timeout=1.0)
    await asyncio.sleep(0)

    assert not chat_task.done()
    save_can_finish.set()
    response = await asyncio.wait_for(chat_task, timeout=1.0)

    assert response.messages == ["ok"]
    await _reset_queue_state()


@pytest.mark.asyncio
async def test_chat_response_omits_tracking_id_when_no_message(monkeypatch):
    """Empty chat responses should not expose delivery tracking IDs."""

    await _reset_queue_state()
    graph_result = _graph_result()
    graph_result["final_dialog"] = []
    graph_result["consolidation_state"] = None
    _patch_chat_dependencies(monkeypatch, _FakeGraph(graph_result))

    response = await service_module.chat(
        _chat_request(),
        BackgroundTasks(),
    )

    assert response.messages == []
    assert response.delivery_tracking_id == ""
    await _reset_queue_state()


@pytest.mark.asyncio
async def test_chat_cognition_silence_skips_user_visible_work(monkeypatch):
    """A graph-level no-response result should not queue output side effects."""

    await _reset_queue_state()
    save_assistant_message = AsyncMock()
    progress_recorder = AsyncMock()
    consolidation_runner = AsyncMock()

    monkeypatch.setattr(
        service_module,
        "_save_assistant_message",
        save_assistant_message,
    )
    monkeypatch.setattr(
        service_module,
        "_run_conversation_progress_record_background",
        progress_recorder,
    )
    monkeypatch.setattr(
        service_module,
        "_run_consolidation_background",
        consolidation_runner,
    )
    graph_result = _graph_result()
    graph_result["should_respond"] = False
    graph_result["final_dialog"] = []
    graph_result["consolidation_state"] = {}
    _patch_chat_dependencies(monkeypatch, _FakeGraph(graph_result))

    response = await service_module.chat(
        _chat_request(),
        BackgroundTasks(),
    )

    assert response.messages == []
    assert response.delivery_tracking_id == ""
    save_assistant_message.assert_not_awaited()
    progress_recorder.assert_not_awaited()
    consolidation_runner.assert_not_awaited()
    await _reset_queue_state()


@pytest.mark.asyncio
async def test_chat_consolidates_private_action_result_without_dialog(monkeypatch):
    """Private action traces should queue consolidation without visible output."""

    await _reset_queue_state()
    save_assistant_message = AsyncMock()
    progress_recorder = AsyncMock()
    consolidation_done = asyncio.Event()
    captured_state = {}

    async def _consolidation_runner(state):
        captured_state.update(state)
        consolidation_done.set()

    monkeypatch.setattr(
        service_module,
        "_save_assistant_message",
        save_assistant_message,
    )
    monkeypatch.setattr(
        service_module,
        "_run_conversation_progress_record_background",
        progress_recorder,
    )
    monkeypatch.setattr(
        service_module,
        "_run_consolidation_background",
        _consolidation_runner,
    )
    consolidation_state = _consolidation_state()
    consolidation_state["final_dialog"] = []
    consolidation_state["action_results"] = [
        {
            "schema_version": "action_result.v1",
            "action_attempt_id": "action_attempt:abc",
            "action_kind": "apply_memory_lifecycle_update",
            "handler_owner": "memory_lifecycle",
            "status": "executed",
            "visibility": "private",
            "result_summary": "memory lifecycle updated",
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
    ]
    graph_result = _graph_result(consolidation_state)
    graph_result["should_respond"] = False
    graph_result["final_dialog"] = []
    _patch_chat_dependencies(monkeypatch, _FakeGraph(graph_result))

    response = await service_module.chat(
        _chat_request(),
        BackgroundTasks(),
    )

    assert response.messages == []
    await asyncio.wait_for(consolidation_done.wait(), timeout=1.0)
    assert captured_state["action_results"][0]["action_kind"] == (
        "apply_memory_lifecycle_update"
    )
    save_assistant_message.assert_not_awaited()
    progress_recorder.assert_not_awaited()
    await _reset_queue_state()


@pytest.mark.asyncio
async def test_post_turn_lifecycle_iterates_after_productive_passes() -> None:
    """Post-turn lifecycle review should re-query after executed updates."""

    state = _post_turn_lifecycle_state()
    read_batches = [
        [{"unit_id": "unit-001", "fact": "User promised dessert."}],
        [{"unit_id": "unit-002", "fact": "User promised tea."}],
        [],
    ]
    review_unit_ids: list[str] = []

    async def _reader(*, global_user_id: str, limit: int) -> dict[str, object]:
        assert global_user_id == "global-user-1"
        assert limit == post_turn_module.POST_SURFACE_ACTIVE_COMMITMENT_REVIEW_LIMIT
        documents = read_batches.pop(0)
        return {
            "documents": documents,
            "limit": limit,
            "limit_exceeded": False,
        }

    async def _review(
        _state: dict,
        active_commitment_units: list[dict[str, object]],
    ) -> dict[str, object]:
        unit_id = str(active_commitment_units[0]["unit_id"])
        review_unit_ids.append(unit_id)
        return {
            "action_specs": [_memory_lifecycle_action_spec(unit_id)],
            "memory_lifecycle_context": {
                "schema_version": "memory_lifecycle_context.v1",
                "source": "memory_lifecycle_specialist",
                "decision": "lifecycle_change",
                "lifecycle_decisions": [
                    {"target_alias": "commitment_1", "decision": "fulfilled"}
                ],
                "content_plan_roles": [],
                "visible_alias_count": 1,
                "omitted_alias_count": 0,
                "warnings": [],
            },
        }

    async def _execute(
        action_specs: list[dict],
        *,
        storage_timestamp_utc: str,
        executed_action_attempt_ids: set[str] | None = None,
        record_attempt_func=None,
    ) -> list[dict]:
        del storage_timestamp_utc, executed_action_attempt_ids, record_attempt_func
        return [
            _memory_lifecycle_action_result(
                str(action_spec["params"]["unit_id"]),
            )
            for action_spec in action_specs
        ]

    updated = await post_turn_module.run_post_turn_memory_lifecycle_background(
        state,
        active_commitment_reader=_reader,
        review_func=_review,
        execute_action_specs_func=_execute,
        logger=service_module.logger,
        no_remember=False,
        visible_response_sent=True,
        think_only_suppressed=False,
    )

    assert updated is not state
    assert review_unit_ids == ["unit-001", "unit-002"]
    assert [
        action_result["action_kind"]
        for action_result in updated["action_results"]
    ] == [
        APPLY_MEMORY_LIFECYCLE_UPDATE_CAPABILITY,
        APPLY_MEMORY_LIFECYCLE_UPDATE_CAPABILITY,
    ]
    assert updated["episode_trace"]["action_results"] == updated["action_results"]
    assert [
        action_spec["params"]["unit_id"]
        for action_spec in updated["action_specs"]
    ] == ["unit-001", "unit-002"]


@pytest.mark.asyncio
async def test_post_turn_lifecycle_skips_structural_blockers() -> None:
    """Post-turn lifecycle review should leave blocked states unchanged."""

    state = _post_turn_lifecycle_state()
    state_with_existing_result = dict(state)
    state_with_existing_result["action_results"] = [
        _memory_lifecycle_action_result("unit-existing")
    ]

    async def _reader(*, global_user_id: str, limit: int) -> dict[str, object]:
        raise AssertionError(
            f"reader should not run: user={global_user_id} limit={limit}"
        )

    async def _review(
        _state: dict,
        active_commitment_units: list[dict[str, object]],
    ) -> dict[str, object]:
        raise AssertionError(f"review should not run: {active_commitment_units}")

    async def _execute(
        action_specs: list[dict],
        *,
        storage_timestamp_utc: str,
        executed_action_attempt_ids: set[str] | None = None,
        record_attempt_func=None,
    ) -> list[dict]:
        raise AssertionError(f"execute should not run: {action_specs}")

    cases = [
        {
            "state": state,
            "no_remember": True,
            "visible_response_sent": True,
            "think_only_suppressed": False,
        },
        {
            "state": state,
            "no_remember": False,
            "visible_response_sent": False,
            "think_only_suppressed": False,
        },
        {
            "state": state,
            "no_remember": False,
            "visible_response_sent": True,
            "think_only_suppressed": True,
        },
        {
            "state": state_with_existing_result,
            "no_remember": False,
            "visible_response_sent": True,
            "think_only_suppressed": False,
        },
    ]

    for case in cases:
        result = await post_turn_module.run_post_turn_memory_lifecycle_background(
            case["state"],
            active_commitment_reader=_reader,
            review_func=_review,
            execute_action_specs_func=_execute,
            logger=service_module.logger,
            no_remember=case["no_remember"],
            visible_response_sent=case["visible_response_sent"],
            think_only_suppressed=case["think_only_suppressed"],
        )

        assert result is case["state"]


@pytest.mark.asyncio
async def test_chat_runs_post_turn_lifecycle_before_progress_and_consolidation(
    monkeypatch,
) -> None:
    """Progress and consolidation should consume lifecycle-updated state."""

    await _reset_queue_state()
    captured_states: dict[str, dict] = {}
    consolidation_done = asyncio.Event()

    async def _post_turn_lifecycle(state: dict) -> dict:
        updated = dict(state)
        updated["post_lifecycle_marker"] = "updated"
        return updated

    async def _progress_recorder(state: dict) -> None:
        captured_states["progress"] = dict(state)

    async def _consolidation_runner(state: dict) -> None:
        captured_states["consolidation"] = dict(state)
        consolidation_done.set()

    monkeypatch.setattr(service_module, "_save_assistant_message", AsyncMock())
    monkeypatch.setattr(
        service_module,
        "_run_post_turn_memory_lifecycle_background",
        _post_turn_lifecycle,
    )
    monkeypatch.setattr(
        service_module,
        "_run_conversation_progress_record_background",
        _progress_recorder,
    )
    monkeypatch.setattr(
        service_module,
        "_run_consolidation_background",
        _consolidation_runner,
    )
    monkeypatch.setattr(
        service_module,
        "_run_internal_monologue_residue_record_background",
        AsyncMock(),
    )
    _patch_chat_dependencies(
        monkeypatch,
        _FakeGraph(_graph_result()),
        patch_post_turn_lifecycle=False,
    )

    response = await service_module.chat(
        _chat_request(),
        BackgroundTasks(),
    )

    assert response.messages == ["ok"]
    await asyncio.wait_for(consolidation_done.wait(), timeout=1.0)
    assert captured_states["progress"]["post_lifecycle_marker"] == "updated"
    assert captured_states["consolidation"]["post_lifecycle_marker"] == "updated"
    await _reset_queue_state()


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
    """Brain-side fallback should hydrate sparse native reply metadata."""
    lookup = AsyncMock(return_value={
        "platform_user_id": "bot-1",
        "display_name": "Kazusa",
        "body_text": "previous assistant answer",
    })
    monkeypatch.setattr(
        service_module,
        "get_conversation_by_platform_message_id",
        lookup,
    )
    request = _chat_request()
    request.message_envelope.reply = service_module.ReplyTargetIn(
        platform_message_id="platform-123",
    )

    reply_context = await service_module._hydrate_reply_context(request)

    assert reply_context["reply_to_message_id"] == "platform-123"
    assert reply_context["reply_to_platform_user_id"] == "bot-1"
    assert reply_context["reply_to_display_name"] == "Kazusa"
    assert reply_context["reply_excerpt"] == "previous assistant answer"
    lookup.assert_awaited_once_with(
        platform="qq",
        platform_channel_id="chan-1",
        platform_message_id="platform-123",
    )


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
async def test_next_chat_waits_until_background_consolidation_finishes(monkeypatch):
    """The next chat request must not enter the graph while consolidation runs."""

    await _reset_queue_state()
    consolidation_started = asyncio.Event()
    consolidation_can_finish = asyncio.Event()
    graph_calls = 0
    consolidation_calls = 0

    class _CountingGraph:
        """Count graph invocations while returning a successful result."""

        async def ainvoke(self, _state):
            nonlocal graph_calls
            graph_calls += 1
            return _graph_result()

    async def _blocked_consolidation(_state):
        nonlocal consolidation_calls
        consolidation_calls += 1
        if consolidation_calls == 1:
            consolidation_started.set()
            await consolidation_can_finish.wait()

    monkeypatch.setattr(service_module, "_save_assistant_message", AsyncMock())
    monkeypatch.setattr(
        service_module,
        "_run_conversation_progress_record_background",
        AsyncMock(),
    )
    monkeypatch.setattr(
        service_module,
        "_run_consolidation_background",
        _blocked_consolidation,
    )
    _patch_chat_dependencies(monkeypatch, _CountingGraph())

    first_background_tasks = BackgroundTasks()
    first_response = await service_module.chat(
        _chat_request(message_id="msg-1"),
        first_background_tasks,
    )
    assert first_response.messages == ["ok"]
    assert graph_calls == 1

    first_background_runner = asyncio.create_task(first_background_tasks())
    await consolidation_started.wait()

    second_background_tasks = BackgroundTasks()
    second_chat_runner = asyncio.create_task(service_module.chat(
        _chat_request(message_id="msg-2"),
        second_background_tasks,
    ))
    await asyncio.sleep(0.05)

    assert not second_chat_runner.done()
    assert graph_calls == 1

    consolidation_can_finish.set()
    await first_background_runner

    second_response = await asyncio.wait_for(second_chat_runner, timeout=1.0)
    assert second_response.messages == ["ok"]
    assert graph_calls == 2

    await second_background_tasks()
    await _reset_queue_state()


@pytest.mark.asyncio
async def test_no_remember_skips_consolidation_but_releases_after_other_writes(monkeypatch):
    """no_remember should skip consolidation and still release after save/progress."""

    await _reset_queue_state()
    save_assistant_message = AsyncMock()
    progress_done = asyncio.Event()
    consolidation_runner = AsyncMock()

    async def _progress_recorder(_state):
        progress_done.set()

    monkeypatch.setattr(
        service_module,
        "_save_assistant_message",
        save_assistant_message,
    )
    monkeypatch.setattr(
        service_module,
        "_run_conversation_progress_record_background",
        _progress_recorder,
    )
    monkeypatch.setattr(
        service_module,
        "_run_consolidation_background",
        consolidation_runner,
    )
    _patch_chat_dependencies(monkeypatch, _FakeGraph(_graph_result()))

    background_tasks = BackgroundTasks()
    response = await service_module.chat(
        _chat_request(
            debug_modes=service_module.DebugModesIn(no_remember=True),
        ),
        background_tasks,
    )

    assert response.messages == ["ok"]
    assert len(background_tasks.tasks) == 0

    await asyncio.wait_for(progress_done.wait(), timeout=1.0)
    save_assistant_message.assert_awaited_once()
    consolidation_runner.assert_not_awaited()
    await _reset_queue_state()


@pytest.mark.asyncio
async def test_graph_failure_does_not_stop_queue_worker(monkeypatch):
    """A graph failure should let the queue worker continue processing."""

    await _reset_queue_state()
    _patch_chat_dependencies(monkeypatch, _FailingGraph())

    background_tasks = BackgroundTasks()
    response = await service_module.chat(_chat_request(), background_tasks)

    assert response.messages == ["Character is busy right now, please try again later."]
    assert len(background_tasks.tasks) == 0
    await _reset_queue_state()


@pytest.mark.asyncio
async def test_background_consolidation_refreshes_cached_character_state(monkeypatch):
    """Consolidation cannot author the cognition-owned character state cache."""

    monkeypatch.setattr(
        service_module,
        "_runtime_character_state",
        {
            "mood": "old mood",
            "vibe_check": "old vibe",
            "character_reflection": "old reflection",
        },
    )
    monkeypatch.setattr(
        service_module,
        "call_consolidation_subgraph",
        AsyncMock(return_value={
            "mood": "Curious",
            "vibe_check": "Focused",
            "character_reflection": "The previous turn left her attentive.",
            "consolidation_metadata": {
                "write_success": {
                    "character_state": True,
                },
            },
        }),
    )

    await service_module._run_consolidation_background({
        "storage_timestamp_utc": "2026-04-25T06:00:58+00:00",
        "local_time_context": _CONSOLIDATION_TURN_CLOCK["local_time_context"],
    })

    assert service_module._runtime_character_state == {
        "mood": "old mood",
        "vibe_check": "old vibe",
        "character_reflection": "old reflection",
    }


@pytest.mark.asyncio
async def test_progress_background_passes_character_boundary_profile(monkeypatch):
    """Progress recorder receives the character boundary profile from the snapshot."""

    state = _consolidation_state()
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


@pytest.mark.asyncio
async def test_progress_background_requires_character_boundary_profile(monkeypatch):
    """Missing character boundary configuration is a state-shape bug."""

    state = _consolidation_state()
    record_turn_progress = AsyncMock()
    monkeypatch.setattr(
        service_module,
        "record_turn_progress",
        record_turn_progress,
    )

    with pytest.raises(KeyError, match="boundary_profile"):
        await service_module._run_conversation_progress_record_background(state)

    record_turn_progress.assert_not_awaited()


@pytest.mark.asyncio
async def test_build_graph_preserves_consolidation_state_from_supervisor(monkeypatch):
    """The top-level service graph should retain supervisor consolidation_state."""

    async def _relevance_agent(_state):
        return {
            "should_respond": True,
            "reason_to_respond": "test",
            "use_reply_feature": False,
            "channel_topic": "test",
            "indirect_speech_context": "",
        }

    async def _persona_supervisor(_state):
        return {
            "cognition_core_output": {},
            "cognition_state_update": {},
            "cognition_state_committed": True,
            "final_dialog": ["好呀。"],
            "future_promises": [],
            "consolidation_state": {
                "storage_timestamp_utc": _GRAPH_TURN_CLOCK[
                    "storage_timestamp_utc"
                ],
                "local_time_context": _GRAPH_TURN_CLOCK["local_time_context"],
                "final_dialog": ["好呀。"],
                "decontexualized_input": "一分钟后发消息",
            },
        }

    monkeypatch.setattr(service_module, "relevance_agent", _relevance_agent)
    monkeypatch.setattr(service_module, "persona_supervisor2", _persona_supervisor)
    monkeypatch.setattr(
        service_module,
        "load_progress_context",
        AsyncMock(return_value={
            "episode_state": None,
            "conversation_progress": {
                "status": "new_episode",
                "episode_label": "",
                "continuity": "sharp_transition",
                "turn_count": 0,
                "user_state_updates": [],
                "assistant_moves": [],
                "overused_moves": [],
                "open_loops": [],
                "progression_guidance": "",
            },
            "source": "empty",
        }),
    )

    graph = service_module._build_graph()
    result = await graph.ainvoke({
        "storage_timestamp_utc": _GRAPH_TURN_CLOCK["storage_timestamp_utc"],
        "local_time_context": _GRAPH_TURN_CLOCK["local_time_context"],
        "platform": "qq",
        "platform_message_id": "268099968",
        "platform_user_id": "673225019",
        "global_user_id": "global-user-1",
        "user_name": "蚝爹油",
        "user_input": "一分钟后发消息",
        "user_multimedia_input": [],
        "user_profile": {"global_user_id": "global-user-1", "relationship_state": 500},
        "platform_bot_id": "bot-id",
        "character_name": "Character",
        "character_profile": {"name": "杏山千纱"},
        "platform_channel_id": "673225019",
        "channel_type": "private",
        "channel_name": "Private",
        "chat_history_wide": [],
        "chat_history_recent": [],
        "reply_context": {},
        "debug_modes": {},
    })

    assert result["final_dialog"] == ["好呀。"]
    assert result["consolidation_state"]["decontexualized_input"] == "一分钟后发消息"


@pytest.mark.asyncio
async def test_build_graph_preserves_persona_no_response(monkeypatch):
    """Persona can de-assert should_respond after cognition chooses silence."""

    async def _relevance_agent(_state):
        return {
            "should_respond": True,
            "reason_to_respond": "test",
            "use_reply_feature": False,
            "channel_topic": "test",
            "indirect_speech_context": "",
        }

    async def _persona_supervisor(_state):
        return {
            "cognition_core_output": {},
            "cognition_state_update": {},
            "cognition_state_committed": True,
            "should_respond": False,
            "final_dialog": [],
            "target_addressed_user_ids": [],
            "target_broadcast": False,
            "future_promises": [],
            "consolidation_state": {
                "should_respond": False,
                "final_dialog": [],
                "decontexualized_input": "silent turn",
            },
        }

    monkeypatch.setattr(service_module, "relevance_agent", _relevance_agent)
    monkeypatch.setattr(service_module, "persona_supervisor2", _persona_supervisor)
    monkeypatch.setattr(
        service_module,
        "load_progress_context",
        AsyncMock(return_value={
            "episode_state": None,
            "conversation_progress": {
                "status": "new_episode",
                "episode_label": "",
                "continuity": "sharp_transition",
                "turn_count": 0,
                "user_state_updates": [],
                "assistant_moves": [],
                "overused_moves": [],
                "open_loops": [],
                "progression_guidance": "",
            },
            "source": "empty",
        }),
    )

    graph = service_module._build_graph()
    result = await graph.ainvoke({
        "storage_timestamp_utc": _GRAPH_TURN_CLOCK["storage_timestamp_utc"],
        "local_time_context": _GRAPH_TURN_CLOCK["local_time_context"],
        "platform": "qq",
        "platform_message_id": "268099968",
        "platform_user_id": "673225019",
        "global_user_id": "global-user-1",
        "user_name": "Test User",
        "user_input": "ignored message",
        "user_multimedia_input": [],
        "user_profile": {"global_user_id": "global-user-1", "relationship_state": 500},
        "platform_bot_id": "bot-id",
        "character_name": "Character",
        "character_profile": {"name": "Character"},
        "platform_channel_id": "673225019",
        "channel_type": "private",
        "channel_name": "Private",
        "chat_history_wide": [],
        "chat_history_recent": [],
        "reply_context": {},
        "debug_modes": {},
    })

    assert result["should_respond"] is False
    assert result["final_dialog"] == []
    assert result["consolidation_state"]["should_respond"] is False


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
async def test_build_graph_skips_episode_state_loader_when_relevance_declines(monkeypatch):
    """Non-responsive turns exit before conversation progress is loaded."""

    async def _relevance_agent(_state):
        return {
            "should_respond": False,
            "reason_to_respond": "not addressed",
            "use_reply_feature": False,
            "channel_topic": "",
            "indirect_speech_context": "",
        }

    load_progress_context = AsyncMock()
    monkeypatch.setattr(service_module, "relevance_agent", _relevance_agent)
    monkeypatch.setattr(service_module, "load_progress_context", load_progress_context)

    graph = service_module._build_graph()
    result = await graph.ainvoke({
        "storage_timestamp_utc": _GRAPH_TURN_CLOCK["storage_timestamp_utc"],
        "local_time_context": _GRAPH_TURN_CLOCK["local_time_context"],
        "platform": "qq",
        "platform_message_id": "268099968",
        "platform_user_id": "673225019",
        "global_user_id": "global-user-1",
        "user_name": "Test User",
        "user_input": "third party chat",
        "user_multimedia_input": [],
        "user_profile": {"global_user_id": "global-user-1", "relationship_state": 500},
        "platform_bot_id": "bot-id",
        "character_name": "Character",
        "character_profile": {"name": "Character"},
        "platform_channel_id": "673225019",
        "channel_type": "group",
        "channel_name": "Group",
        "chat_history_wide": [],
        "chat_history_recent": [],
        "reply_context": {},
        "debug_modes": {},
    })

    assert result["should_respond"] is False
    load_progress_context.assert_not_awaited()


@pytest.mark.asyncio
async def test_chat_listen_only_drops_before_graph(monkeypatch):
    """Listen-only requests should persist and complete without graph work."""

    await _reset_queue_state()
    monkeypatch.setattr(
        service_module,
        "_static_character_profile",
        {"name": "Character"},
    )
    monkeypatch.setattr(
        service_module,
        "_runtime_character_state",
        {
            "mood": "old mood",
            "vibe_check": "old vibe",
            "character_reflection": "old reflection",
        },
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
        AsyncMock(return_value="character-global-id"),
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
        "_hydrate_reply_context",
        AsyncMock(return_value={}),
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
        "record_queue_intake_event",
        AsyncMock(),
    )
    save_conversation = AsyncMock()
    monkeypatch.setattr(service_module, "save_conversation", save_conversation)

    class _Graph:
        """Expose a mock graph entrypoint for listen-only assertions."""

        def __init__(self):
            self.ainvoke = AsyncMock(return_value={})

    graph = _Graph()
    monkeypatch.setattr(service_module, "_graph", graph)

    background_tasks = BackgroundTasks()
    response = await service_module.chat(
        service_module.ChatRequest(
            platform="qq",
            platform_channel_id="227608960",
            channel_type="group",
            platform_message_id="394466266",
            platform_user_id="876192223",
            platform_bot_id="3768713357",
            display_name="Test User",
            channel_name="Group 227608960",
            message_envelope={
                "body_text": "listen only fixture",
                "raw_wire_text": "listen only fixture",
                "mentions": [],
                "attachments": [],
                "addressed_to_global_user_ids": [],
                "broadcast": True,
            },
            debug_modes=service_module.DebugModesIn(listen_only=True),
        ),
        background_tasks,
    )

    assert response.messages == []
    assert response.use_reply_feature is False
    graph.ainvoke.assert_not_awaited()
    save_conversation.assert_awaited_once()
    await _reset_queue_state()
