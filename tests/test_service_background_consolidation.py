"""Unit tests for background consolidation scheduling in the service layer."""

from __future__ import annotations

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


@pytest.mark.asyncio
async def test_chat_queues_background_consolidation_for_mapping_state(monkeypatch):
    """Mapping-like consolidation state should still queue the background task."""

    monkeypatch.setattr(service_module, "_chat_executor_semaphore", None)
    monkeypatch.setattr(service_module, "_personality", {"name": "Kazusa"})
    monkeypatch.setattr(service_module, "resolve_global_user_id", AsyncMock(return_value="global-user-1"))
    monkeypatch.setattr(service_module, "get_user_profile", AsyncMock(return_value={"affinity": 500}))
    monkeypatch.setattr(service_module, "get_conversation_history", AsyncMock(return_value=[]))
    monkeypatch.setattr(service_module, "_hydrate_reply_context", AsyncMock(return_value={}))
    monkeypatch.setattr(service_module, "save_conversation", AsyncMock())

    consolidation_state = _MappingState({
        "timestamp": "2026-04-25T18:00:58+12:00",
        "platform": "qq",
        "platform_channel_id": "673225019",
        "platform_message_id": "2073572281",
        "global_user_id": "global-user-1",
        "user_name": "蚝爹油",
        "user_profile": {"affinity": 500},
        "character_profile": {"name": "杏山千纱"},
        "action_directives": {"linguistic_directives": {"content_anchors": []}},
        "internal_monologue": "test",
        "final_dialog": ["好呀。"],
        "interaction_subtext": "",
        "emotional_appraisal": "",
        "character_intent": "PROVIDE",
        "logical_stance": "CONFIRM",
        "rag_result": {},
        "decontexualized_input": "一分钟后发消息",
    })
    monkeypatch.setattr(
        service_module,
        "_graph",
        _FakeGraph({
            "should_respond": True,
            "use_reply_feature": False,
            "final_dialog": ["好呀。"],
            "future_promises": [],
            "consolidation_state": consolidation_state,
        }),
    )

    background_tasks = BackgroundTasks()
    response = await service_module.chat(
        service_module.ChatRequest(
            platform="qq",
            platform_channel_id="673225019",
            channel_type="private",
            platform_message_id="2073572281",
            platform_user_id="673225019",
            display_name="蚝爹油",
            channel_name="Private",
            content="咱们再试试，千纱酱一分钟之后能去902317662群里发个消息，内容是今天天气真好呀",
        ),
        background_tasks,
    )

    assert response.messages == ["好呀。"]
    assert len(background_tasks.tasks) == 2


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
async def test_chat_listen_only_keeps_boolean_should_respond(monkeypatch):
    """Listen-only requests should skip thinking while preserving state defaults."""

    monkeypatch.setattr(service_module, "_chat_executor_semaphore", None)
    monkeypatch.setattr(service_module, "_personality", {"name": "Kazusa"})
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
