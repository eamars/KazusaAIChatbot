"""Unit tests for the task dispatcher and evaluator."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from kazusa_ai_chatbot.config import LLM_BASE_URL
from kazusa_ai_chatbot.dispatcher import (
    AdapterRegistry,
    DispatchContext,
    PendingTaskIndex,
    RawToolCall,
    SendResult,
    TaskDispatcher,
    ToolCallEvaluator,
    ToolRegistry,
    build_send_message_tool,
)
from kazusa_ai_chatbot.dispatcher.task import parse_iso_datetime
from kazusa_ai_chatbot.nodes import persona_supervisor2_consolidator_persistence as persistence_module


class _NoopAdapter:
    platform = "discord"

    async def send_message(
        self,
        channel_id: str,
        text: str,
        *,
        reply_to_msg_id: str | None = None,
    ) -> SendResult:
        del channel_id, text, reply_to_msg_id
        return SendResult(
            platform=self.platform,
            channel_id="chan-1",
            message_id="m-1",
            sent_at=datetime.now(timezone.utc),
        )


async def _skip_if_llm_unavailable() -> None:
    """Skip live dispatcher tests when the configured LLM endpoint is unavailable."""

    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(f"{LLM_BASE_URL.rstrip('/')}/models")
    except httpx.HTTPError:
        pytest.skip(f"LLM endpoint is unavailable: {LLM_BASE_URL}")

    if response.status_code >= 500:
        pytest.skip(f"LLM endpoint returned server error {response.status_code}: {LLM_BASE_URL}")


@pytest.fixture()
async def ensure_live_llm() -> None:
    """Ensure the configured live LLM endpoint is reachable before each test."""

    await _skip_if_llm_unavailable()


def _build_dispatcher() -> tuple[TaskDispatcher, PendingTaskIndex]:
    tool_registry = ToolRegistry()
    tool_registry.register(build_send_message_tool())
    adapter_registry = AdapterRegistry()
    adapter_registry.register(_NoopAdapter())
    evaluator = ToolCallEvaluator(tool_registry, adapter_registry)
    pending_index = PendingTaskIndex()
    return TaskDispatcher(evaluator, pending_index), pending_index


def _ctx() -> DispatchContext:
    return DispatchContext(
        source_platform="discord",
        source_channel_id="chan-1",
        source_user_id="user-1",
        source_message_id="msg-1",
        guild_id=None,
        bot_role="user",
        now=datetime(2026, 4, 25, 0, 0, tzinfo=timezone.utc),
    )


def _dispatch_generation_state(
    *,
    future_promises: list[dict],
    final_dialog: list[str],
    decontexualized_input: str,
    content_anchors: list[str],
    platform: str = "discord",
    platform_channel_id: str = "chan-live-1",
    channel_type: str = "group",
) -> dict:
    """Build a minimal consolidator state for live tool-call generation tests."""

    return {
        "character_profile": {
            "name": "杏山千纱",
            "personality_brief": {"mbti": "INFJ"},
        },
        "timestamp": "2026-04-23T06:11:28+12:00",
        "platform": platform,
        "platform_channel_id": platform_channel_id,
        "channel_type": channel_type,
        "platform_message_id": "msg-live-1",
        "global_user_id": "user-live-1",
        "user_name": "提拉米苏",
        "user_profile": {"affinity": 500},
        "decontexualized_input": decontexualized_input,
        "final_dialog": final_dialog,
        "action_directives": {
            "linguistic_directives": {
                "content_anchors": content_anchors,
            },
        },
        "future_promises": future_promises,
    }


def _live_dispatch_ctx(
    *,
    source_platform: str = "discord",
    source_channel_id: str = "chan-live-1",
    source_user_id: str = "user-live-1",
    source_message_id: str = "msg-live-1",
) -> DispatchContext:
    """Return a stable dispatch context for live dispatcher-generation tests."""

    return DispatchContext(
        source_platform=source_platform,
        source_channel_id=source_channel_id,
        source_user_id=source_user_id,
        source_message_id=source_message_id,
        guild_id=None,
        bot_role="user",
        now=datetime(2026, 4, 22, 18, 11, 28, tzinfo=timezone.utc),
    )


def test_evaluator_defaults_platform_channel_and_now():
    tool_registry = ToolRegistry()
    tool_registry.register(build_send_message_tool())
    adapter_registry = AdapterRegistry()
    adapter_registry.register(_NoopAdapter())
    evaluator = ToolCallEvaluator(tool_registry, adapter_registry)

    result = evaluator.evaluate(
        RawToolCall(
            tool="send_message",
            args={
                "target_channel": "same",
                "text": "hello later",
            },
        ),
        _ctx(),
    )

    assert result.ok is True
    assert result.task is not None
    assert result.task.args["target_platform"] == "discord"
    assert result.task.args["target_channel"] == "chan-1"
    assert result.task.execute_at == _ctx().now


def test_evaluator_rejects_unparseable_execute_at():
    tool_registry = ToolRegistry()
    tool_registry.register(build_send_message_tool())
    adapter_registry = AdapterRegistry()
    adapter_registry.register(_NoopAdapter())
    evaluator = ToolCallEvaluator(tool_registry, adapter_registry)

    result = evaluator.evaluate(
        RawToolCall(
            tool="send_message",
            args={
                "target_channel": "same",
                "text": "hello later",
                "execute_at": "tomorrow-ish",
            },
        ),
        _ctx(),
    )

    assert result.ok is False
    assert result.errors == ["unparseable execute_at"]


def test_evaluator_rejects_when_no_adapter_is_registered():
    tool_registry = ToolRegistry()
    tool_registry.register(build_send_message_tool())
    evaluator = ToolCallEvaluator(tool_registry, AdapterRegistry())

    result = evaluator.evaluate(
        RawToolCall(
            tool="send_message",
            args={
                "target_channel": "same",
                "text": "hello later",
            },
        ),
        _ctx(),
    )

    assert result.ok is False
    assert result.errors == ["no adapters registered"]


@pytest.mark.asyncio
async def test_dispatcher_schedules_valid_task_through_scheduler():
    dispatcher, pending_index = _build_dispatcher()
    ctx = _ctx()

    with patch("kazusa_ai_chatbot.scheduler.schedule_event", AsyncMock(return_value="evt-1")) as schedule_event:
        result = await dispatcher.dispatch(
            [
                RawToolCall(
                    tool="send_message",
                    args={
                        "target_channel": "same",
                        "text": "hello later",
                    },
                )
            ],
            ctx,
            instruction="Check in later.",
        )

    schedule_event.assert_awaited_once()
    assert len(result.scheduled) == 1
    task, event_id = result.scheduled[0]
    assert event_id == "evt-1"
    assert task.tags == ["Check in later."]
    assert pending_index.contains(task)
    assert result.rejected == []


@pytest.mark.asyncio
async def test_dispatcher_rejects_duplicate_task_in_batch():
    dispatcher, _pending_index = _build_dispatcher()
    ctx = _ctx()

    with patch("kazusa_ai_chatbot.scheduler.schedule_event", AsyncMock(return_value="evt-1")):
        result = await dispatcher.dispatch(
            [
                RawToolCall(
                    tool="send_message",
                    args={"target_channel": "same", "text": "same thing"},
                ),
                RawToolCall(
                    tool="send_message",
                    args={"target_channel": "same", "text": "same thing"},
                ),
            ],
            ctx,
        )

    assert len(result.scheduled) == 1
    assert len(result.rejected) == 1
    assert result.rejected[0][1] == "duplicate task"


@pytest.mark.live_llm
@pytest.mark.asyncio
async def test_live_dispatcher_generates_send_message_tool_call(ensure_live_llm, monkeypatch):
    """A real accepted delayed follow-up should become a concrete send_message call."""

    del ensure_live_llm
    tool_registry = ToolRegistry()
    tool_registry.register(build_send_message_tool())
    monkeypatch.setattr(persistence_module, "_task_registry", tool_registry)

    due_time = "2026-04-24T08:00:00+12:00"
    state = _dispatch_generation_state(
        future_promises=[
            {
                "target": "提拉米苏",
                "action": "杏山千纱将对提拉米苏执行叫醒动作",
                "due_time": due_time,
                "commitment_type": "future_promise",
                "dedup_key": "wake_up_tiramisu_morning",
            }
        ],
        decontexualized_input="明天早上记得叫我起床。",
        content_anchors=[
            "[DECISION] 明确答应明早叫醒对方。",
            "[ANSWER] 说明会在明早叫他起床。",
        ],
        final_dialog=["好，明早八点我叫你起床。"],
    )

    raw_calls = await persistence_module._generate_raw_tool_calls(state, _live_dispatch_ctx())

    assert raw_calls, "Expected at least one tool call from the live dispatcher LLM."
    assert any(call.tool == "send_message" for call in raw_calls)
    send_call = next(call for call in raw_calls if call.tool == "send_message")
    assert send_call.args.get("target_channel") in {"same", "chan-live-1"}
    assert isinstance(send_call.args.get("text"), str) and send_call.args["text"].strip()
    assert send_call.args.get("execute_at")
    assert parse_iso_datetime(send_call.args["execute_at"]).isoformat() == parse_iso_datetime(due_time).isoformat()


@pytest.mark.live_llm
@pytest.mark.asyncio
async def test_live_dispatcher_rejects_persistent_style_rule_as_tool_call(ensure_live_llm, monkeypatch):
    """An ongoing style/address rule should not turn into a send_message tool call."""

    del ensure_live_llm
    tool_registry = ToolRegistry()
    tool_registry.register(build_send_message_tool())
    monkeypatch.setattr(persistence_module, "_task_registry", tool_registry)

    state = _dispatch_generation_state(
        future_promises=[
            {
                "target": "提拉米苏",
                "action": "杏山千纱将对提拉米苏使用“主人”称呼并以“喵”结尾说话",
                "due_time": None,
                "commitment_type": "address_preference",
                "dedup_key": "address_rule_master_meow",
            }
        ],
        decontexualized_input="以后叫我主人，而且每句话都喵一下。",
        content_anchors=[
            "[DECISION] 勉强接受并沿用这个规则。",
            "[SOCIAL] 带着羞耻和别扭地答应。",
        ],
        final_dialog=["主、主人……这种奇怪的称呼真的很羞耻喵。"],
    )

    raw_calls = await persistence_module._generate_raw_tool_calls(state, _live_dispatch_ctx())

    assert raw_calls == [], f"Expected no tool calls for a persistent style rule, got: {raw_calls}"


@pytest.mark.live_llm
@pytest.mark.asyncio
async def test_live_dispatcher_generates_group_target_from_private_chat(ensure_live_llm, monkeypatch):
    """A private-chat request that names a QQ group should target that group id."""

    del ensure_live_llm
    tool_registry = ToolRegistry()
    tool_registry.register(build_send_message_tool())
    monkeypatch.setattr(persistence_module, "_task_registry", tool_registry)

    due_time = "2026-04-22T18:12:28+00:00"
    state = _dispatch_generation_state(
        future_promises=[
            {
                "target": "54369546群",
                "action": "杏山千纱将在1分钟后于54369546群发送“今天天气真好呀”",
                "due_time": due_time,
                "commitment_type": "future_promise",
                "dedup_key": "send_to_54369546_weather_message",
            }
        ],
        decontexualized_input="千纱，1分钟之后你在54369546群发一条消息，内容是今天天气真好呀",
        content_anchors=[
            "[DECISION] 接受了1分钟后去54369546群发消息的请求。",
            "[ANSWER] 明确说会在54369546群发“今天天气真好呀”。",
        ],
        final_dialog=["好，一分钟后我会去54369546群发“今天天气真好呀”。"],
        platform="qq",
        platform_channel_id="10001",
        channel_type="private",
    )

    raw_calls = await persistence_module._generate_raw_tool_calls(
        state,
        _live_dispatch_ctx(
            source_platform="qq",
            source_channel_id="10001",
        ),
    )

    assert raw_calls, "Expected at least one tool call from the live dispatcher LLM."
    assert any(call.tool == "send_message" for call in raw_calls)
    send_call = next(call for call in raw_calls if call.tool == "send_message")
    assert send_call.args.get("target_channel") == "54369546"
    assert send_call.args.get("target_channel") != "same"
    assert "今天天气真好呀" in str(send_call.args.get("text", ""))
    assert send_call.args.get("execute_at")
    assert parse_iso_datetime(send_call.args["execute_at"]).isoformat() == parse_iso_datetime(due_time).isoformat()


@pytest.mark.live_llm
@pytest.mark.asyncio
async def test_live_dispatcher_assumes_five_minutes_for_implicit_near_future_time(ensure_live_llm, monkeypatch):
    """A vague near-future promise like '一会儿' should normalize to current_utc + 5 minutes."""

    del ensure_live_llm
    tool_registry = ToolRegistry()
    tool_registry.register(build_send_message_tool())
    monkeypatch.setattr(persistence_module, "_task_registry", tool_registry)

    ctx = _live_dispatch_ctx()
    expected_execute_at = datetime(2026, 4, 22, 18, 16, 28, tzinfo=timezone.utc)
    state = _dispatch_generation_state(
        future_promises=[
            {
                "target": "提拉米苏",
                "action": "杏山千纱将于稍后提醒提拉米苏喝水",
                "due_time": None,
                "commitment_type": "future_promise",
                "dedup_key": "remind_tiramisu_drink_water_later",
            }
        ],
        decontexualized_input="千纱，一会儿提醒我喝水。",
        content_anchors=[
            "[DECISION] 接受了稍后提醒对方喝水的请求。",
            "[ANSWER] 明确说一会儿会提醒他喝水。",
        ],
        final_dialog=["好，一会儿我提醒你喝水。"],
    )

    raw_calls = await persistence_module._generate_raw_tool_calls(state, ctx)

    assert raw_calls, "Expected at least one tool call from the live dispatcher LLM."
    assert any(call.tool == "send_message" for call in raw_calls)
    send_call = next(call for call in raw_calls if call.tool == "send_message")
    assert send_call.args.get("target_channel") in {"same", "chan-live-1"}
    assert "喝水" in str(send_call.args.get("text", ""))
    assert send_call.args.get("execute_at")
    assert parse_iso_datetime(send_call.args["execute_at"]).isoformat() == expected_execute_at.isoformat()
