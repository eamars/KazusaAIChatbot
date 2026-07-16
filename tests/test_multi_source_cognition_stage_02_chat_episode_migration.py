"""Focused tests for text chat cognitive episode migration."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import BackgroundTasks

from kazusa_ai_chatbot import service as service_module
from kazusa_ai_chatbot.chat_input_queue import QueuedChatItem
from kazusa_ai_chatbot.cognition_episode import build_text_chat_cognitive_episode
from kazusa_ai_chatbot.nodes import persona_supervisor2 as supervisor_module
from kazusa_ai_chatbot.cognition_chain_core.stages import l1 as l1_module
from kazusa_ai_chatbot.cognition_chain_core.stages import l2 as l2_module
from kazusa_ai_chatbot.cognition_chain_core.stages import l2c2 as l2c2_module
from kazusa_ai_chatbot.cognition_chain_core.stages import l2d as l2d_module
from kazusa_ai_chatbot.nodes import persona_supervisor2_cognition as cognition_module
from kazusa_ai_chatbot.time_boundary import build_turn_clock


BACKGROUND_HANDOFF_WAIT_POLLS = 100
BACKGROUND_HANDOFF_WAIT_SECONDS = 0.01


class _FakeGraph:
    """Service graph fake that captures state passed to graph invocation."""

    def __init__(self, result: dict[str, Any]) -> None:
        """Store the result returned by ``ainvoke``.

        Args:
            result: Graph result returned by the fake.
        """

        self.result = result
        self.states: list[dict[str, Any]] = []

    async def ainvoke(self, state: dict[str, Any]) -> dict[str, Any]:
        """Record graph input state and return the configured result.

        Args:
            state: Service graph state passed to the fake.

        Returns:
            Configured graph result.
        """

        self.states.append(dict(state))
        return_value = self.result
        return return_value


def _character_profile() -> dict[str, Any]:
    """Build a character profile with fields read by service code.

    Returns:
        Character profile fixture for service and graph tests.
    """

    return_value = {
        "name": "Character",
        "global_user_id": "character-1",
        "mood": "neutral",
        "global_vibe": "calm",
        "reflection_summary": "quiet baseline",
        "description": "A test character.",
        "personality_brief": {
            "mbti": "INTJ",
            "logic": "precise",
            "tempo": "short",
            "defense": "calm",
            "quirks": "dry",
            "taboos": "none",
        },
        "boundary_profile": {
            "self_integrity": 0.8,
            "control_sensitivity": 0.3,
            "control_intimacy_misread": 0.2,
            "relational_override": 0.25,
            "compliance_strategy": "comply",
            "boundary_recovery": "rebound",
        },
    }
    return return_value


def _message_envelope(
    *,
    body_text: str = "hello there",
    addressed_to_global_user_ids: list[str] | None = None,
    broadcast: bool = False,
) -> dict[str, Any]:
    """Build a service message-envelope payload.

    Args:
        body_text: Text body supplied by the adapter envelope.
        addressed_to_global_user_ids: Prompt-facing addressed users to seed.
        broadcast: Whether the inbound message is broadcast to the channel.

    Returns:
        Message envelope dictionary accepted by ``ChatRequest``.
    """

    addressed_ids = list(addressed_to_global_user_ids or ["character-1"])
    return_value = {
        "body_text": body_text,
        "raw_wire_text": body_text,
        "mentions": [],
        "attachments": [],
        "addressed_to_global_user_ids": addressed_ids,
        "broadcast": broadcast,
    }
    return return_value


def _chat_request(
    *,
    platform_message_id: str = "msg-private-1",
    debug_modes: service_module.DebugModesIn | None = None,
) -> service_module.ChatRequest:
    """Build a chat request for service episode-construction tests.

    Args:
        platform_message_id: Platform message ID to put on the request.
        debug_modes: Optional debug-mode flags to put on the request.

    Returns:
        Chat request fixture for the service endpoint.
    """

    request = service_module.ChatRequest(
        platform="qq",
        platform_channel_id="private-chan-1",
        channel_type="private",
        platform_message_id=platform_message_id,
        platform_user_id="platform-user-1",
        platform_bot_id="bot-1",
        display_name="Test User",
        channel_name="Test Channel",
        message_envelope=_message_envelope(),
        debug_modes=debug_modes or service_module.DebugModesIn(),
    )
    return request


def _graph_result(final_dialog: list[str] | None = None) -> dict[str, Any]:
    """Build a service graph result used by the fake graph.

    Args:
        final_dialog: Dialog fragments returned by the fake graph.

    Returns:
        Graph result fixture with response and consolidation fields.
    """

    dialog = list(final_dialog if final_dialog is not None else ["ok"])
    consolidation_state = {
        "final_dialog": dialog,
        "debug_modes": {},
        "decontexualized_input": "hello there",
    }
    return_value = {
        "should_respond": True,
        "use_reply_feature": False,
        "final_dialog": dialog,
        "target_addressed_user_ids": ["global-user-1"],
        "target_broadcast": False,
        "future_promises": [],
        "consolidation_state": consolidation_state,
    }
    return return_value


def _patch_service_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    graph: _FakeGraph,
) -> dict[str, AsyncMock]:
    """Patch service dependencies outside episode construction.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        graph: Fake graph object installed as the service graph.

    Returns:
        Patched async mocks keyed by contract name.
    """

    save_conversation = AsyncMock(return_value="conversation-row-1")
    save_assistant_message = AsyncMock()
    progress_recorder = AsyncMock()
    consolidation_runner = AsyncMock()
    residue_recorder = AsyncMock()
    frontline_relevance = AsyncMock(return_value={
        "intake_action": "start",
        "append_target": "none",
        "prelude_targets": [],
        "reason": "fixture admission",
    })
    settled_relevance = AsyncMock(return_value={
        "response_action": "proceed",
        "reason_to_respond": "fixture response",
        "use_reply_feature": False,
        "channel_topic": "",
        "indirect_speech_context": "",
    })

    for event_function_name in (
        "record_database_operation_event",
        "record_pipeline_turn_event",
        "record_queue_intake_event",
        "record_runtime_error_event",
    ):
        monkeypatch.setattr(
            service_module.event_logging,
            event_function_name,
            AsyncMock(),
        )
    monkeypatch.setattr(
        service_module,
        "COGNITION_VISUAL_DIRECTIVES_ENABLED",
        True,
    )
    monkeypatch.setattr(
        service_module,
        "_static_character_profile",
        _character_profile(),
    )
    monkeypatch.setattr(
        service_module,
        "_runtime_character_state",
        {
            "mood": "neutral",
            "global_vibe": "calm",
            "reflection_summary": "quiet baseline",
        },
    )
    monkeypatch.setattr(
        service_module,
        "get_character_runtime_state",
        AsyncMock(return_value={
            "mood": "neutral",
            "global_vibe": "calm",
            "reflection_summary": "quiet baseline",
        }),
    )
    monkeypatch.setattr(
        service_module,
        "_ensure_character_global_identity",
        AsyncMock(return_value="character-1"),
    )
    monkeypatch.setattr(
        service_module,
        "resolve_global_user_id",
        AsyncMock(return_value="global-user-1"),
    )
    monkeypatch.setattr(
        service_module,
        "get_user_profile",
        AsyncMock(return_value={
            "affinity": 500,
            "last_relationship_insight": "steady baseline",
        }),
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
        "build_promoted_reflection_context",
        AsyncMock(return_value={}),
    )
    monkeypatch.setattr(service_module, "save_conversation", save_conversation)
    monkeypatch.setattr(
        service_module,
        "frontline_relevance_agent",
        frontline_relevance,
    )
    monkeypatch.setattr(
        service_module,
        "relevance_agent",
        settled_relevance,
    )
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
    monkeypatch.setattr(
        service_module,
        "_run_internal_monologue_residue_record_background",
        residue_recorder,
    )
    monkeypatch.setattr(service_module, "_graph", graph)
    return_value = {
        "save_conversation": save_conversation,
        "save_assistant_message": save_assistant_message,
        "progress_recorder": progress_recorder,
        "consolidation_runner": consolidation_runner,
        "residue_recorder": residue_recorder,
        "frontline_relevance": frontline_relevance,
        "settled_relevance": settled_relevance,
    }
    return return_value


async def _reset_queue_state() -> None:
    """Reset process-local queue state around service endpoint tests."""

    await service_module._stop_chat_input_worker()
    service_module._chat_input_queue.reset_for_test()


async def _wait_for_mock_await(
    mock: AsyncMock,
    *,
    await_count: int = 1,
) -> None:
    """Wait until post-response queue work reaches a patched async seam."""

    for _ in range(BACKGROUND_HANDOFF_WAIT_POLLS):
        if mock.await_count >= await_count:
            return
        await asyncio.sleep(BACKGROUND_HANDOFF_WAIT_SECONDS)

    assert mock.await_count >= await_count


def _episode() -> dict[str, Any]:
    """Build a valid text chat cognitive episode for graph pass-through tests.

    Returns:
        Cognitive episode fixture built through the public text chat builder.
    """

    turn_clock = build_turn_clock("2026-05-01 09:00:00")
    episode = build_text_chat_cognitive_episode(
        episode_id="user_message:qq:private-chan-1:msg-private-1",
        percept_id="user_message:qq:private-chan-1:msg-private-1:dialog_text:0",
        storage_timestamp_utc=turn_clock["storage_timestamp_utc"],
        local_time_context=turn_clock["local_time_context"],
        user_input="hello there",
        platform="qq",
        platform_channel_id="private-chan-1",
        channel_type="private",
        platform_message_id="msg-private-1",
        platform_user_id="platform-user-1",
        global_user_id="global-user-1",
        user_name="Test User",
        active_turn_platform_message_ids=["msg-private-1"],
        active_turn_conversation_row_ids=["conversation-row-1"],
        debug_modes={
            "listen_only": False,
            "think_only": False,
            "no_remember": False,
        },
        target_addressed_user_ids=["character-1"],
        target_broadcast=False,
    )
    return episode


def _base_persona_state() -> dict[str, Any]:
    """Build top-level persona state with a cognitive episode.

    Returns:
        Persona graph input fixture with the episode attached.
    """

    turn_clock = build_turn_clock("2026-05-01 09:00:00")
    return_value = {
        "storage_timestamp_utc": turn_clock["storage_timestamp_utc"],
        "local_time_context": turn_clock["local_time_context"],
        "platform": "qq",
        "platform_message_id": "msg-private-1",
        "active_turn_platform_message_ids": ["msg-private-1"],
        "active_turn_conversation_row_ids": ["conversation-row-1"],
        "platform_user_id": "platform-user-1",
        "global_user_id": "global-user-1",
        "user_name": "Test User",
        "user_input": "hello there",
        "message_envelope": _message_envelope(),
        "prompt_message_context": _message_envelope(),
        "user_multimedia_input": [],
        "user_profile": {
            "affinity": 500,
            "last_relationship_insight": "steady baseline",
        },
        "platform_bot_id": "bot-1",
        "character_name": "Character",
        "character_profile": _character_profile(),
        "platform_channel_id": "private-chan-1",
        "channel_type": "private",
        "channel_name": "Test Channel",
        "chat_history_wide": [],
        "chat_history_recent": [],
        "reply_context": {},
        "should_respond": True,
        "reason_to_respond": "fixture",
        "use_reply_feature": False,
        "channel_topic": "baseline",
        "indirect_speech_context": "",
        "debug_modes": {
            "listen_only": False,
            "think_only": False,
            "no_remember": False,
        },
        "final_dialog": [],
        "target_addressed_user_ids": ["global-user-1"],
        "target_broadcast": False,
        "future_promises": [],
        "consolidation_state": {},
        "promoted_reflection_context": {},
        "cognitive_episode": _episode(),
    }
    return return_value


def _base_cognition_state() -> dict[str, Any]:
    """Build a cognition-subgraph input state with a cognitive episode.

    Returns:
        Cognition subgraph input fixture with prior-stage fields populated.
    """

    state = _base_persona_state()
    state.update({
        "decontexualized_input": "hello there",
        "referents": [],
        "rag_result": {
            "answer": "",
            "user_image": {},
            "user_memory_unit_candidates": [],
            "character_image": {},
            "third_party_profiles": [],
            "memory_evidence": [],
            "recall_evidence": [],
            "conversation_evidence": [],
            "external_evidence": [],
            "supervisor_trace": {
                "loop_count": 0,
                "unknown_slots": [],
                "dispatched": [],
            },
        },
    })
    return state


def test_text_chat_episode_ids_follow_contract() -> None:
    """Episode id helper should follow the approved fallback contract."""

    build_ids = service_module._build_text_chat_episode_ids

    assert build_ids(
        platform="qq",
        platform_channel_id="chan-1",
        platform_message_id="msg-1",
        conversation_row_id="row-1",
        queue_sequence=7,
    ) == (
        "user_message:qq:chan-1:msg-1",
        "user_message:qq:chan-1:msg-1:dialog_text:0",
    )
    assert build_ids(
        platform="qq",
        platform_channel_id="",
        platform_message_id="",
        conversation_row_id="row-1",
        queue_sequence=7,
    ) == (
        "user_message:qq:direct:row-1",
        "user_message:qq:direct:row-1:dialog_text:0",
    )
    assert build_ids(
        platform="qq",
        platform_channel_id="",
        platform_message_id="",
        conversation_row_id=None,
        queue_sequence=7,
    ) == (
        "user_message:qq:direct:queue-7",
        "user_message:qq:direct:queue-7:dialog_text:0",
    )


@pytest.mark.asyncio
async def test_service_builds_text_chat_cognitive_episode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Service state should carry an inert text chat cognitive episode."""

    await _reset_queue_state()
    graph = _FakeGraph(_graph_result())
    mocks = _patch_service_dependencies(monkeypatch, graph)

    response = await service_module.chat(_chat_request(), BackgroundTasks())

    assert response.messages == ["ok"]
    state = graph.states[0]
    episode = state["cognitive_episode"]
    assert state["user_input"] == "hello there"
    assert state["platform_message_id"] == "msg-private-1"
    assert state["active_turn_platform_message_ids"] == ["msg-private-1"]
    assert state["active_turn_conversation_row_ids"] == ["conversation-row-1"]
    assert episode["episode_id"] == (
        "user_message:qq:private-chan-1:msg-private-1"
    )
    assert episode["trigger_source"] == "user_message"
    assert episode["input_sources"] == ["dialog_text"]
    assert episode["output_mode"] == "visible_reply"
    assert episode["percepts"][0]["content"] == "hello there"
    assert episode["target_scope"]["target_addressed_user_ids"] == (
        state["prompt_message_context"]["addressed_to_global_user_ids"]
    )
    assert episode["target_scope"]["target_broadcast"] == (
        state["prompt_message_context"]["broadcast"]
    )
    assert episode["origin_metadata"]["debug_modes"] == {
        "listen_only": False,
        "think_only": False,
        "no_remember": False,
    }
    await _wait_for_mock_await(mocks["consolidation_runner"])
    await _reset_queue_state()


@pytest.mark.asyncio
async def test_service_adds_internal_visual_flag_when_config_disables_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Config-disabled visual directives should appear only in internal state."""

    await _reset_queue_state()
    graph = _FakeGraph(_graph_result())
    mocks = _patch_service_dependencies(monkeypatch, graph)
    monkeypatch.setattr(
        service_module,
        "COGNITION_VISUAL_DIRECTIVES_ENABLED",
        False,
    )

    response = await service_module.chat(_chat_request(), BackgroundTasks())

    assert response.messages == ["ok"]
    state = graph.states[0]
    assert state["debug_modes"] == {
        "listen_only": False,
        "think_only": False,
        "no_remember": False,
        "no_visual_directives": True,
    }
    assert state["cognitive_episode"]["origin_metadata"]["debug_modes"] == (
        state["debug_modes"]
    )
    assert "no_visual_directives" not in service_module.DebugModesIn.model_fields
    await _wait_for_mock_await(mocks["consolidation_runner"])
    await _reset_queue_state()


@pytest.mark.asyncio
async def test_service_maps_debug_modes_to_episode_output_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Episode output mode should reflect explicit debug-mode flags only."""

    await _reset_queue_state()
    think_graph = _FakeGraph(_graph_result())
    think_mocks = _patch_service_dependencies(monkeypatch, think_graph)

    await service_module.chat(
        _chat_request(
            debug_modes=service_module.DebugModesIn(think_only=True),
        ),
        BackgroundTasks(),
    )

    assert think_graph.states[0]["cognitive_episode"]["output_mode"] == (
        "think_only"
    )
    await _wait_for_mock_await(think_mocks["consolidation_runner"])
    await _reset_queue_state()

    listen_graph = _FakeGraph(_graph_result(final_dialog=[]))
    _patch_service_dependencies(monkeypatch, listen_graph)
    future: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
    turn_clock = build_turn_clock("2026-05-01 09:00:00")
    item = QueuedChatItem(
        sequence=7,
        request=_chat_request(
            debug_modes=service_module.DebugModesIn(listen_only=True),
        ),
        storage_timestamp_utc=turn_clock["storage_timestamp_utc"],
        local_timestamp=turn_clock["local_timestamp"],
        local_time_context=turn_clock["local_time_context"],
        future=future,
    )

    await service_module._process_queued_chat_item(item)

    assert listen_graph.states[0]["cognitive_episode"]["output_mode"] == "silent"
    await _reset_queue_state()


@pytest.mark.asyncio
async def test_persona_graph_passes_cognitive_episode_through() -> None:
    """Persona state should pass the episode through each top-level stage."""

    state = _base_persona_state()
    episode = state["cognitive_episode"]

    with (
        patch(
            "kazusa_ai_chatbot.nodes.persona_supervisor2.call_msg_decontexualizer",
            new_callable=AsyncMock,
            return_value={
                "decontexualized_input": "hello there",
                "referents": [],
            },
        ) as decontextualizer,
        patch(
            "kazusa_ai_chatbot.nodes.persona_supervisor2.call_cognition_resolver_loop",
            new_callable=AsyncMock,
            return_value={
                "internal_monologue": "thinking",
                "rag_result": {},
                "action_directives": {},
                "interaction_subtext": "",
                "emotional_appraisal": "",
                "character_intent": "",
                "logical_stance": "",
                "judgment_note": "",
                "social_distance": "",
                "emotional_intensity": "",
                "vibe_check": "",
                "relational_dynamic": "",
                "action_specs": [],
            },
        ) as resolver,
        patch(
            "kazusa_ai_chatbot.nodes.persona_supervisor2.dialog_agent",
            new_callable=AsyncMock,
            return_value={
                "final_dialog": ["ok"],
                "target_addressed_user_ids": ["global-user-1"],
                "target_broadcast": False,
            },
        ),
    ):
        result = await supervisor_module.persona_supervisor2(state)

    assert decontextualizer.await_args.args[0]["cognitive_episode"] == episode
    assert resolver.await_args.args[0]["cognitive_episode"] == episode
    assert result["consolidation_state"]["cognitive_episode"] == episode


@pytest.mark.asyncio
async def test_cognition_subgraph_passes_cognitive_episode_to_nodes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cognition subgraph should carry the episode without prompt use."""

    captured: dict[str, dict[str, Any]] = {}

    async def _capture_subconscious(state: dict[str, Any]) -> dict[str, str]:
        """Capture L1 input state and return a deterministic update."""

        captured["l1"] = dict(state)
        return_value = {
            "emotional_appraisal": "steady",
            "interaction_subtext": "routine",
        }
        return return_value

    async def _capture_consciousness(state: dict[str, Any]) -> dict[str, str]:
        """Capture L2 consciousness input state and return a fixed update."""

        captured["l2a"] = dict(state)
        return_value = {
            "internal_monologue": "answer",
            "character_intent": "PROVIDE",
            "logical_stance": "CONFIRM",
        }
        return return_value

    async def _capture_boundary(state: dict[str, Any]) -> dict[str, Any]:
        """Capture boundary-core input state and return a fixed update."""

        captured["l2b"] = dict(state)
        return_value = {
            "boundary_core_assessment": {
                "boundary_issue": "none",
            },
        }
        return return_value

    async def _capture_judgment(state: dict[str, Any]) -> dict[str, str]:
        """Capture judgment-core input state and return a fixed update."""

        captured["l2c"] = dict(state)
        return_value = {
            "logical_stance": "CONFIRM",
            "character_intent": "PROVIDE",
            "judgment_note": "ok",
        }
        return return_value

    async def _capture_action_selection(
        state: dict[str, Any],
    ) -> dict[str, list]:
        """Capture L2d input state and return no selected actions."""

        captured["l2d"] = dict(state)
        return_value = {
            "action_specs": [],
        }
        return return_value

    monkeypatch.setattr(
        l1_module,
        "call_cognition_subconscious",
        _capture_subconscious,
    )
    monkeypatch.setattr(
        l2_module,
        "call_cognition_consciousness",
        _capture_consciousness,
    )
    monkeypatch.setattr(
        l2_module,
        "call_boundary_core_agent",
        _capture_boundary,
    )
    monkeypatch.setattr(
        l2_module,
        "call_judgment_core_agent",
        _capture_judgment,
    )
    monkeypatch.setattr(
        l2d_module,
        "select_semantic_actions",
        _capture_action_selection,
    )

    async def _fake_social_context(state: dict[str, Any]) -> dict[str, Any]:
        return {
            "social_distance": "neutral",
            "emotional_intensity": "calm",
            "vibe_check": "daily",
            "relational_dynamic": "stable",
        }

    monkeypatch.setattr(
        l2c2_module,
        "call_social_context_appraisal",
        _fake_social_context,
    )

    state = _base_cognition_state()
    result = await cognition_module.call_cognition_subgraph(state)

    chain_episode = captured["l1"]["cognitive_episode"]
    assert chain_episode["episode_id"] == state["cognitive_episode"]["episode_id"]
    assert captured["l2a"]["cognitive_episode"] == chain_episode
    assert captured["l2b"]["cognitive_episode"] == chain_episode
    assert captured["l2c"]["cognitive_episode"] == chain_episode
    assert captured["l2d"]["cognitive_episode"] == chain_episode
    assert result["internal_monologue"] == "answer"
    assert result["logical_stance"] == "CONFIRM"
    assert result["action_specs"] == []
