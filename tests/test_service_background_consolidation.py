"""Unit tests for background consolidation scheduling in the service layer."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from unittest.mock import AsyncMock

import pytest
from fastapi import BackgroundTasks

from kazusa_ai_chatbot import service as service_module


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
    debug_modes: service_module.DebugModesIn | None = None,
) -> service_module.ChatRequest:
    """Build a minimal chat request for service-layer tests.

    Args:
        message_id: Platform message identifier for the request.
        debug_modes: Optional debug-mode flags for the request.

    Returns:
        ChatRequest with deterministic ASCII payload fields.
    """

    request = service_module.ChatRequest(
        platform="qq",
        platform_channel_id="chan-1",
        channel_type="private",
        platform_message_id=message_id,
        platform_user_id="user-1",
        platform_bot_id="bot-1",
        display_name="Test User",
        channel_name="Private",
        content="please remember this",
        debug_modes=debug_modes or service_module.DebugModesIn(),
    )
    return request


def _consolidation_state() -> dict:
    """Return the common consolidation-state fixture used by chat tests.

    Returns:
        Consolidation state with all fields required by background recorders.
    """

    return_value = {
        "timestamp": "2026-04-25T18:00:58+12:00",
        "platform": "qq",
        "platform_channel_id": "chan-1",
        "platform_message_id": "msg-1",
        "global_user_id": "global-user-1",
        "user_name": "Test User",
        "user_profile": {"affinity": 500},
        "character_profile": {"name": "Kazusa"},
        "action_directives": {"linguistic_directives": {"content_anchors": []}},
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


def _patch_chat_dependencies(monkeypatch, graph) -> None:
    """Patch service dependencies that are outside queue-worker behavior.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        graph: Fake graph object installed as the service graph.

    Returns:
        None.
    """

    monkeypatch.setattr(service_module, "_personality", {"name": "Kazusa"})
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
        AsyncMock(return_value={"affinity": 500}),
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
    monkeypatch.setattr(service_module, "save_conversation", AsyncMock())
    monkeypatch.setattr(service_module, "_graph", graph)


async def _reset_queue_state() -> None:
    """Reset global queue state between service endpoint tests.

    Returns:
        None.
    """

    await service_module._stop_chat_input_worker()
    service_module._chat_input_queue.clear()
    service_module._chat_queue_condition = None
    service_module._chat_queue_worker_task = None
    service_module._chat_queue_sequence = 0


@pytest.mark.asyncio
async def test_chat_queues_background_consolidation_for_mapping_state(monkeypatch):
    """Mapping-like consolidation state should still queue the background task."""

    await _reset_queue_state()
    save_bot_message = AsyncMock()
    progress_recorder = AsyncMock()
    consolidation_done = asyncio.Event()

    async def _consolidation_runner(_state):
        consolidation_done.set()

    monkeypatch.setattr(service_module, "_save_bot_message", save_bot_message)
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

    save_bot_message.assert_awaited_once()
    progress_recorder.assert_awaited_once()
    await _reset_queue_state()


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

    monkeypatch.setattr(service_module, "_save_bot_message", AsyncMock())
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
    save_bot_message = AsyncMock()
    progress_done = asyncio.Event()
    consolidation_runner = AsyncMock()

    async def _progress_recorder(_state):
        progress_done.set()

    monkeypatch.setattr(service_module, "_save_bot_message", save_bot_message)
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
    save_bot_message.assert_awaited_once()
    consolidation_runner.assert_not_awaited()
    await _reset_queue_state()


@pytest.mark.asyncio
async def test_graph_failure_does_not_stop_queue_worker(monkeypatch):
    """A graph failure should let the queue worker continue processing."""

    await _reset_queue_state()
    _patch_chat_dependencies(monkeypatch, _FailingGraph())

    background_tasks = BackgroundTasks()
    response = await service_module.chat(_chat_request(), background_tasks)

    assert response.messages == ["Kazusa is busy right now, please try again later."]
    assert len(background_tasks.tasks) == 0
    await _reset_queue_state()


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
            "final_dialog": ["好呀。"],
            "future_promises": [],
            "consolidation_state": {
                "timestamp": "2026-04-25T18:07:24+12:00",
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
        "timestamp": "2026-04-25T18:07:24+12:00",
        "platform": "qq",
        "platform_message_id": "268099968",
        "platform_user_id": "673225019",
        "global_user_id": "global-user-1",
        "user_name": "蚝爹油",
        "user_input": "一分钟后发消息",
        "user_multimedia_input": [],
        "user_profile": {"affinity": 500},
        "platform_bot_id": "bot-id",
        "bot_name": "Kazusa",
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
        "timestamp": "2026-04-25T18:07:24+12:00",
        "platform": "qq",
        "platform_message_id": "268099968",
        "platform_user_id": "673225019",
        "global_user_id": "global-user-1",
        "user_name": "Test User",
        "user_input": "third party chat",
        "user_multimedia_input": [],
        "user_profile": {"affinity": 500},
        "platform_bot_id": "bot-id",
        "bot_name": "Kazusa",
        "character_profile": {"name": "Kazusa"},
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
async def test_chat_listen_only_keeps_boolean_should_respond(monkeypatch):
    """Listen-only requests should skip thinking while preserving state defaults."""

    await _reset_queue_state()
    monkeypatch.setattr(service_module, "_personality", {"name": "Kazusa"})
    monkeypatch.setattr(
        service_module,
        "_ensure_character_global_identity",
        AsyncMock(return_value="character-global-id"),
    )
    monkeypatch.setattr(service_module, "resolve_global_user_id", AsyncMock(return_value="global-user-1"))
    monkeypatch.setattr(service_module, "get_user_profile", AsyncMock(return_value={"affinity": 500}))
    monkeypatch.setattr(service_module, "get_conversation_history", AsyncMock(return_value=[]))
    monkeypatch.setattr(service_module, "_hydrate_reply_context", AsyncMock(return_value={}))
    monkeypatch.setattr(service_module, "save_conversation", AsyncMock())

    captured_state = {}

    class _CapturingGraph:
        """Capture the initial state and emulate the listen-only graph result."""

        async def ainvoke(self, state):
            captured_state.update(state)
            return state

    monkeypatch.setattr(service_module, "_graph", _CapturingGraph())

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
            content="listen only fixture",
            debug_modes=service_module.DebugModesIn(listen_only=True),
        ),
        background_tasks,
    )

    assert captured_state["should_respond"] is False
    assert captured_state["use_reply_feature"] is False
    assert response.messages == []
    assert response.should_reply is False
    await _reset_queue_state()
