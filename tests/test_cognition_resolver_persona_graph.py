"""Persona graph integration tests for cognition resolver routing."""

from __future__ import annotations

import importlib

import pytest

from kazusa_ai_chatbot.cognition_episode import build_text_chat_cognitive_episode
from kazusa_ai_chatbot.time_boundary import build_turn_clock


persona_module = importlib.import_module(
    "kazusa_ai_chatbot.nodes.persona_supervisor2",
)
REMOVED_RESOLVER_ENABLE_FLAG = "COGNITION_" + "RESOLVER_ENABLED"
REMOVED_RAG_FIRST_NODE = "stage_" + "1_research"


def _im_state() -> dict:
    turn_clock = build_turn_clock("2026-05-30 09:00:00")
    episode = build_text_chat_cognitive_episode(
        episode_id="resolver-graph-episode",
        percept_id="resolver-graph-percept",
        storage_timestamp_utc=turn_clock["storage_timestamp_utc"],
        local_time_context=turn_clock["local_time_context"],
        user_input="今晚想随便聊两句。",
        platform="debug",
        platform_channel_id="channel-graph",
        channel_type="private",
        platform_message_id="message-graph",
        platform_user_id="platform-user-graph",
        global_user_id="global-user-graph",
        user_name="Graph User",
        active_turn_platform_message_ids=["message-graph"],
        active_turn_conversation_row_ids=["row-graph"],
        debug_modes={},
        target_addressed_user_ids=["character-graph"],
        target_broadcast=False,
    )
    return {
        "character_profile": {
            "name": "Kazusa",
            "global_user_id": "character-graph",
        },
        "storage_timestamp_utc": turn_clock["storage_timestamp_utc"],
        "local_time_context": turn_clock["local_time_context"],
        "user_input": "今晚想随便聊两句。",
        "prompt_message_context": {
            "body_text": "今晚想随便聊两句。",
            "mentions": [],
            "attachments": [],
            "addressed_to_global_user_ids": ["character-graph"],
            "broadcast": False,
        },
        "cognitive_episode": episode,
        "platform": "debug",
        "platform_channel_id": "channel-graph",
        "channel_type": "private",
        "platform_message_id": "message-graph",
        "platform_user_id": "platform-user-graph",
        "global_user_id": "global-user-graph",
        "user_name": "Graph User",
        "user_profile": {},
        "platform_bot_id": "platform-bot-graph",
        "chat_history_wide": [],
        "reply_context": {},
        "indirect_speech_context": "",
        "channel_topic": "graph test",
        "conversation_episode_state": None,
        "conversation_progress": None,
        "promoted_reflection_context": None,
        "internal_monologue_residue_context": "",
        "debug_modes": {},
        "should_respond": True,
    }


def _cognition_output() -> dict:
    return {
        "internal_monologue": "图路由测试里不需要可见回复。",
        "interaction_subtext": "用户只是在测试图路径。",
        "emotional_appraisal": "平稳。",
        "character_intent": "保持私有收束。",
        "logical_stance": "没有必要调用文本表面。",
        "judgment_note": "路由测试。",
        "social_distance": "close",
        "emotional_intensity": "low",
        "vibe_check": "steady",
        "relational_dynamic": "trusted",
        "action_specs": [],
        "resolver_capability_requests": [],
    }


async def _decontextualizer(_state: dict) -> dict:
    return {
        "decontexualized_input": "今晚想随便聊两句。",
        "referents": [],
    }


async def _empty_decontextualizer(_state: dict) -> dict:
    return {
        "decontexualized_input": "",
        "referents": [],
    }


async def _memory_lifecycle(_state: dict) -> dict:
    return {}


def _image_only_state() -> dict:
    """Build an image-only persona state with empty current text."""

    image_summary = '一张浅绿色头发角色头像。'
    turn_clock = build_turn_clock("2026-05-30 09:00:00")
    episode = build_text_chat_cognitive_episode(
        episode_id="resolver-graph-image-episode",
        percept_id="resolver-graph-image-percept",
        storage_timestamp_utc=turn_clock["storage_timestamp_utc"],
        local_time_context=turn_clock["local_time_context"],
        user_input="",
        platform="debug",
        platform_channel_id="channel-graph",
        channel_type="private",
        platform_message_id="message-graph-image",
        platform_user_id="platform-user-graph",
        global_user_id="global-user-graph",
        user_name="Graph User",
        active_turn_platform_message_ids=["message-graph-image"],
        active_turn_conversation_row_ids=["row-graph-image"],
        debug_modes={},
        target_addressed_user_ids=["character-graph"],
        target_broadcast=False,
        media_description_rows=[
            {
                "content_type": "image/png",
                "description": image_summary,
            },
        ],
    )
    state = _im_state()
    state["user_input"] = ""
    state["prompt_message_context"] = {
        "body_text": "",
        "mentions": [],
        "attachments": [{
            "media_type": "image/png",
            "description": image_summary,
        }],
        "addressed_to_global_user_ids": ["character-graph"],
        "broadcast": False,
    }
    state["cognitive_episode"] = episode
    return state


@pytest.mark.asyncio
async def test_persona_graph_default_runs_goal_resolver_without_legacy_rag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The default graph should enter the resolver and skip legacy RAG-first."""

    calls: list[str] = []
    captured: dict = {}

    async def call_v2_resolver_loop(
        state: dict,
        *,
        cognition_func: object,
        capability_func: object,
        max_cycles: int,
        origin_scope: str | None = None,
    ) -> dict:
        calls.append("resolver")
        captured["state"] = dict(state)
        captured["cognition_func"] = cognition_func
        captured["capability_func"] = capability_func
        captured["max_cycles"] = max_cycles
        captured["origin_scope"] = origin_scope
        return {"cognition_output": _cognition_output(), "observations": []}

    monkeypatch.setattr(persona_module, "COGNITION_RESOLVER_MAX_CYCLES", 3)
    monkeypatch.setattr(
        persona_module,
        "call_msg_decontexualizer",
        _decontextualizer,
    )
    monkeypatch.setattr(
        persona_module,
        "call_v2_resolver_loop",
        call_v2_resolver_loop,
    )
    monkeypatch.setattr(
        persona_module,
        "call_memory_lifecycle_update_handler",
        _memory_lifecycle,
    )

    result = await persona_module.persona_supervisor2(_im_state())

    assert calls == ["resolver"]
    assert captured["state"]["decontexualized_input"] == "今晚想随便聊两句。"
    assert "rag_result" not in captured["state"]
    assert "resolver_state" not in captured["state"]
    assert callable(captured["cognition_func"])
    assert callable(captured["capability_func"])
    assert captured["max_cycles"] == 3
    assert captured["origin_scope"] is None
    assert result["should_respond"] is False


def test_persona_graph_exports_no_resolver_enable_or_legacy_rag_node() -> None:
    """Resolver-only graph should not export the old compatibility switch."""

    assert not hasattr(persona_module, REMOVED_RESOLVER_ENABLE_FLAG)
    assert not hasattr(persona_module, REMOVED_RAG_FIRST_NODE)


@pytest.mark.asyncio
async def test_persona_graph_image_only_empty_text_enters_goal_resolver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Image-only input should reach the resolver loop after decontextualizer."""

    calls: list[str] = []
    captured: dict = {}

    async def call_v2_resolver_loop(
        state: dict,
        *,
        cognition_func: object,
        capability_func: object,
        max_cycles: int,
        origin_scope: str | None = None,
    ) -> dict:
        calls.append("resolver")
        captured["state"] = dict(state)
        return {"cognition_output": _cognition_output(), "observations": []}

    monkeypatch.setattr(
        persona_module,
        "call_msg_decontexualizer",
        _empty_decontextualizer,
    )
    monkeypatch.setattr(
        persona_module,
        "call_v2_resolver_loop",
        call_v2_resolver_loop,
    )
    monkeypatch.setattr(
        persona_module,
        "call_memory_lifecycle_update_handler",
        _memory_lifecycle,
    )

    result = await persona_module.persona_supervisor2(_image_only_state())

    expected_goal = '当前输入包含图片观察：一张浅绿色头发角色头像。'
    assert calls == ["resolver"]
    assert captured["state"]["decontexualized_input"] == ""
    assert captured["state"]["decontexualized_input"] == ""
    assert "resolver_state" not in captured["state"]
    assert result["should_respond"] is False
