"""Tests for the brain service global input queue."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest
from fastapi import BackgroundTasks

from kazusa_ai_chatbot import service as service_module


def _request(
    message_id: str,
    *,
    mentioned_bot: bool = False,
    reply_to_current_bot: bool | None = None,
) -> service_module.ChatRequest:
    """Build a chat request for queue tests.

    Args:
        message_id: Platform message identifier.
        mentioned_bot: Whether the adapter marked a structural bot mention.
        reply_to_current_bot: Adapter-supplied bot-reply marker.

    Returns:
        ChatRequest with deterministic payload fields.
    """

    reply_context = service_module.ReplyContextIn(
        reply_to_message_id=f"reply-{message_id}",
        reply_to_current_bot=reply_to_current_bot,
    )
    request = service_module.ChatRequest(
        platform="qq",
        platform_channel_id="chan-1",
        channel_type="group",
        platform_message_id=message_id,
        platform_user_id=f"user-{message_id}",
        platform_bot_id="bot-1",
        display_name=f"User {message_id}",
        channel_name="Group",
        content=f"message {message_id}",
        mentioned_bot=mentioned_bot,
        reply_context=reply_context,
    )
    return request


def _item(
    sequence: int,
    *,
    mentioned_bot: bool = False,
    reply_to_current_bot: bool | None = None,
) -> service_module._QueuedChatItem:
    """Build a queued item for deterministic pruning tests.

    Args:
        sequence: Queue sequence number.
        mentioned_bot: Whether the request is tagged.
        reply_to_current_bot: Adapter-supplied bot-reply marker.

    Returns:
        Queued chat item.
    """

    future: asyncio.Future[service_module.ChatResponse] = (
        asyncio.get_running_loop().create_future()
    )
    item = service_module._QueuedChatItem(
        sequence=sequence,
        request=_request(
            str(sequence),
            mentioned_bot=mentioned_bot,
            reply_to_current_bot=reply_to_current_bot,
        ),
        timestamp=f"2026-04-29T00:00:0{sequence}+00:00",
        future=future,
    )
    return item


async def _reset_queue_state() -> None:
    """Reset service queue globals between tests.

    Returns:
        None.
    """

    await service_module._stop_chat_input_worker()
    service_module._chat_input_queue.clear()
    service_module._chat_queue_condition = None
    service_module._chat_queue_worker_task = None
    service_module._chat_queue_sequence = 0


def _patch_common_dependencies(monkeypatch, graph) -> None:
    """Patch external service dependencies for queue-worker tests.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        graph: Fake service graph.

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
    monkeypatch.setattr(service_module, "_graph", graph)


@pytest.mark.asyncio
async def test_prune_over_two_drops_plain_messages() -> None:
    """Queue length greater than two should drop untagged non-replies."""

    plain = _item(1)
    tagged = _item(2, mentioned_bot=True)
    bot_reply = _item(3, reply_to_current_bot=True)

    survivors, dropped = service_module._prune_waiting_queue([
        plain,
        tagged,
        bot_reply,
    ])

    assert [item.sequence for item in survivors] == [2, 3]
    assert [item.sequence for item in dropped] == [1]


@pytest.mark.asyncio
async def test_prune_over_five_keeps_only_bot_replies_after_first_stage() -> None:
    """Queue length over five after first prune should drop mentions."""

    items = [
        _item(1, mentioned_bot=True),
        _item(2, mentioned_bot=True),
        _item(3, reply_to_current_bot=True),
        _item(4, reply_to_current_bot=True),
        _item(5, reply_to_current_bot=True),
        _item(6, reply_to_current_bot=True),
    ]

    survivors, dropped = service_module._prune_waiting_queue(items)

    assert [item.sequence for item in survivors] == [3, 4, 5, 6]
    assert [item.sequence for item in dropped] == [1, 2]


@pytest.mark.asyncio
async def test_prune_still_over_five_keeps_latest_survivor() -> None:
    """Queue length still over five should keep only the latest message."""

    items = [
        _item(1, reply_to_current_bot=True),
        _item(2, reply_to_current_bot=True),
        _item(3, reply_to_current_bot=True),
        _item(4, reply_to_current_bot=True),
        _item(5, reply_to_current_bot=True),
        _item(6, reply_to_current_bot=True),
    ]

    survivors, dropped = service_module._prune_waiting_queue(items)

    assert [item.sequence for item in survivors] == [6]
    assert [item.sequence for item in dropped] == [1, 2, 3, 4, 5]


@pytest.mark.asyncio
async def test_reply_id_without_adapter_bot_reply_marker_is_not_protected() -> None:
    """reply_to_message_id alone should not protect a queued message."""

    plain_reply = _item(1, reply_to_current_bot=None)
    tagged = _item(2, mentioned_bot=True)
    bot_reply = _item(3, reply_to_current_bot=True)

    survivors, dropped = service_module._prune_waiting_queue([
        plain_reply,
        tagged,
        bot_reply,
    ])

    assert [item.sequence for item in survivors] == [2, 3]
    assert [item.sequence for item in dropped] == [1]


@pytest.mark.asyncio
async def test_chat_enqueue_path_does_not_save_directly(monkeypatch) -> None:
    """The endpoint should enqueue only; worker code owns user persistence."""

    await _reset_queue_state()
    save_conversation = AsyncMock()
    monkeypatch.setattr(service_module, "save_conversation", save_conversation)
    monkeypatch.setattr(
        service_module,
        "_ensure_chat_input_worker_started",
        lambda: None,
    )

    chat_task = asyncio.create_task(service_module.chat(
        _request("endpoint"),
        BackgroundTasks(),
    ))
    await asyncio.sleep(0)

    assert len(service_module._chat_input_queue) == 1
    save_conversation.assert_not_awaited()

    queued_item = service_module._chat_input_queue.popleft()
    queued_item.future.set_result(service_module.ChatResponse())
    response = await asyncio.wait_for(chat_task, timeout=1.0)

    assert response.messages == []
    save_conversation.assert_not_awaited()
    await _reset_queue_state()


@pytest.mark.asyncio
async def test_worker_saves_dropped_messages_before_next_graph(monkeypatch) -> None:
    """Dropped rows should persist before the next surviving item reaches graph."""

    await _reset_queue_state()
    call_order = []
    graph_started = asyncio.Event()
    graph_can_finish = asyncio.Event()

    class _Graph:
        """Record graph entry order."""

        async def ainvoke(self, _state):
            call_order.append("graph")
            graph_started.set()
            await graph_can_finish.wait()
            return {
                "should_respond": False,
                "use_reply_feature": False,
                "final_dialog": [],
                "future_promises": [],
                "consolidation_state": None,
            }

    async def _save_conversation(_doc):
        call_order.append("save")

    monkeypatch.setattr(service_module, "save_conversation", _save_conversation)
    _patch_common_dependencies(monkeypatch, _Graph())

    dropped = _item(1)
    tagged = _item(2, mentioned_bot=True)
    bot_reply = _item(3, reply_to_current_bot=True)
    service_module._chat_input_queue.extend([dropped, tagged, bot_reply])

    service_module._ensure_chat_input_worker_started()
    condition = service_module._get_chat_queue_condition()
    async with condition:
        condition.notify()

    dropped_response = await asyncio.wait_for(dropped.future, timeout=1.0)
    await asyncio.wait_for(graph_started.wait(), timeout=1.0)

    assert dropped_response.messages == []
    assert call_order[:2] == ["save", "save"]
    assert call_order[2] == "graph"
    assert not bot_reply.future.done()

    graph_can_finish.set()
    tagged_response = await asyncio.wait_for(tagged.future, timeout=1.0)
    assert tagged_response.messages == []

    await _reset_queue_state()


@pytest.mark.asyncio
async def test_dropped_message_never_invokes_graph(monkeypatch) -> None:
    """A pruned message should be saved and completed without graph execution."""

    await _reset_queue_state()
    graph_message_ids = []
    graph_can_finish = asyncio.Event()

    class _Graph:
        """Record graph message IDs for surviving items."""

        async def ainvoke(self, state):
            graph_message_ids.append(state["platform_message_id"])
            await graph_can_finish.wait()
            return {
                "should_respond": False,
                "use_reply_feature": False,
                "final_dialog": [],
                "future_promises": [],
                "consolidation_state": None,
            }

    save_conversation = AsyncMock()
    monkeypatch.setattr(service_module, "save_conversation", save_conversation)
    _patch_common_dependencies(monkeypatch, _Graph())

    dropped = _item(1)
    tagged = _item(2, mentioned_bot=True)
    bot_reply = _item(3, reply_to_current_bot=True)
    service_module._chat_input_queue.extend([dropped, tagged, bot_reply])

    service_module._ensure_chat_input_worker_started()
    condition = service_module._get_chat_queue_condition()
    async with condition:
        condition.notify()

    response = await asyncio.wait_for(dropped.future, timeout=1.0)

    assert response.messages == []
    save_conversation.assert_awaited()
    await asyncio.sleep(0)
    assert graph_message_ids == ["2"]

    graph_can_finish.set()
    await _reset_queue_state()
