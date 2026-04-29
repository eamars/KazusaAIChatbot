"""Tests for the brain service global input queue."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest
from fastapi import BackgroundTasks

from kazusa_ai_chatbot import chat_input_queue as queue_module
from kazusa_ai_chatbot import service as service_module


def _request(
    message_id: str,
    *,
    channel_type: str = "group",
    platform_channel_id: str = "chan-1",
    platform_user_id: str | None = None,
    content: str | None = None,
    content_type: str = "text",
    attachments: list[dict[str, object]] | None = None,
    mentioned_bot: bool = False,
    reply_to_current_bot: bool | None = None,
) -> service_module.ChatRequest:
    """Build a chat request for queue tests.

    Args:
        message_id: Platform message identifier.
        content: Message body; empty string is preserved for media-only input.
        content_type: Adapter-provided content type.
        attachments: Adapter-provided attachment payloads.
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
        platform_channel_id=platform_channel_id,
        channel_type=channel_type,
        platform_message_id=message_id,
        platform_user_id=platform_user_id or f"user-{message_id}",
        platform_bot_id="bot-1",
        display_name=f"User {message_id}",
        channel_name="Group",
        content=f"message {message_id}" if content is None else content,
        content_type=content_type,
        mentioned_bot=mentioned_bot,
        attachments=[
            service_module.AttachmentIn(**attachment)
            for attachment in attachments or []
        ],
        reply_context=reply_context,
    )
    return request


def _item(
    sequence: int,
    *,
    channel_type: str = "group",
    platform_channel_id: str = "chan-1",
    platform_user_id: str | None = None,
    content: str | None = None,
    content_type: str = "text",
    attachments: list[dict[str, object]] | None = None,
    timestamp: str | None = None,
    mentioned_bot: bool = False,
    reply_to_current_bot: bool | None = None,
) -> queue_module.QueuedChatItem:
    """Build a queued item for deterministic pruning tests.

    Args:
        sequence: Queue sequence number.
        content: Message body; empty string is preserved for media-only input.
        content_type: Adapter-provided content type.
        attachments: Adapter-provided attachment payloads.
        mentioned_bot: Whether the request is tagged.
        reply_to_current_bot: Adapter-supplied bot-reply marker.

    Returns:
        Queued chat item.
    """

    future: asyncio.Future[service_module.ChatResponse] = (
        asyncio.get_running_loop().create_future()
    )
    item = queue_module.QueuedChatItem(
        sequence=sequence,
        request=_request(
            str(sequence),
            channel_type=channel_type,
            platform_channel_id=platform_channel_id,
            platform_user_id=platform_user_id,
            content=content,
            content_type=content_type,
            attachments=attachments,
            mentioned_bot=mentioned_bot,
            reply_to_current_bot=reply_to_current_bot,
        ),
        timestamp=timestamp or f"2026-04-29T00:00:{sequence:02d}+00:00",
        future=future,
    )
    return item


async def _reset_queue_state() -> None:
    """Reset service queue globals between tests.

    Returns:
        None.
    """

    await service_module._stop_chat_input_worker()
    service_module._chat_input_queue.reset_for_test()


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

    queue = queue_module.ChatInputQueue()
    survivors, dropped = queue.prune([
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

    queue = queue_module.ChatInputQueue()
    survivors, dropped = queue.prune(items)

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

    queue = queue_module.ChatInputQueue()
    survivors, dropped = queue.prune(items)

    assert [item.sequence for item in survivors] == [6]
    assert [item.sequence for item in dropped] == [1, 2, 3, 4, 5]


@pytest.mark.asyncio
async def test_reply_id_without_adapter_bot_reply_marker_is_not_protected() -> None:
    """reply_to_message_id alone should not protect a queued message."""

    plain_reply = _item(1, reply_to_current_bot=None)
    tagged = _item(2, mentioned_bot=True)
    bot_reply = _item(3, reply_to_current_bot=True)

    queue = queue_module.ChatInputQueue()
    survivors, dropped = queue.prune([
        plain_reply,
        tagged,
        bot_reply,
    ])

    assert [item.sequence for item in survivors] == [2, 3]
    assert [item.sequence for item in dropped] == [1]


@pytest.mark.asyncio
async def test_private_messages_are_not_pruned_as_group_noise() -> None:
    """Private messages should survive group-noise pruning."""

    plain_group = _item(1)
    private_message = _item(
        2,
        channel_type="private",
        platform_channel_id="dm-1",
        platform_user_id="user-private",
    )
    tagged_group = _item(3, mentioned_bot=True)

    queue = queue_module.ChatInputQueue()
    survivors, dropped = queue.prune([
        plain_group,
        private_message,
        tagged_group,
    ])

    assert [item.sequence for item in survivors] == [2, 3]
    assert [item.sequence for item in dropped] == [1]


@pytest.mark.asyncio
async def test_private_messages_same_scope_coalesce() -> None:
    """Private follow-ups in the same scope should collapse into the first item."""

    first = _item(
        1,
        channel_type="private",
        platform_channel_id="dm-1",
        platform_user_id="user-1",
        content="first",
    )
    second = _item(
        2,
        channel_type="private",
        platform_channel_id="dm-1",
        platform_user_id="user-1",
        content="second",
    )
    third = _item(
        3,
        channel_type="private",
        platform_channel_id="dm-2",
        platform_user_id="user-1",
        content="third",
    )

    queue = queue_module.ChatInputQueue()
    survivors, collapsed = queue.coalesce_private([
        first,
        second,
        third,
    ])

    assert [item.sequence for item in survivors] == [1, 3]
    assert [(item.sequence, survivor.sequence) for item, survivor in collapsed] == [
        (2, 1),
    ]
    assert first.combined_content == "first\nsecond"
    assert [item.sequence for item in first.collapsed_items] == [2]


@pytest.mark.asyncio
async def test_addressed_group_followups_coalesce() -> None:
    """Addressed-start same-author group follow-ups should collapse."""

    first = _item(
        1,
        platform_user_id="user-1",
        content="Kazusa,",
        mentioned_bot=True,
    )
    second = _item(
        2,
        platform_user_id="user-1",
        content="one more detail",
    )
    third = _item(
        3,
        platform_user_id="user-1",
        content="and another",
        reply_to_current_bot=True,
    )

    queue = queue_module.ChatInputQueue()
    survivors, collapsed = queue.coalesce_addressed_group([
        first,
        second,
        third,
    ])

    assert [item.sequence for item in survivors] == [1]
    assert [(item.sequence, survivor.sequence) for item, survivor in collapsed] == [
        (2, 1),
        (3, 1),
    ]
    assert first.combined_content == "Kazusa,\none more detail\nand another"


@pytest.mark.asyncio
async def test_plain_group_runs_do_not_coalesce() -> None:
    """Plain-start group runs should not collapse, even if a later item addresses."""

    first = _item(1, platform_user_id="user-1")
    second = _item(2, platform_user_id="user-1", mentioned_bot=True)

    queue = queue_module.ChatInputQueue()
    survivors, collapsed = queue.coalesce_addressed_group([
        first,
        second,
    ])

    assert [item.sequence for item in survivors] == [1, 2]
    assert collapsed == []


@pytest.mark.asyncio
async def test_group_followups_require_same_author_adjacency_and_time_gap() -> None:
    """Group coalescing should respect author adjacency and the time gap."""

    first = _item(
        1,
        platform_user_id="user-1",
        mentioned_bot=True,
    )
    other_author = _item(2, platform_user_id="user-2")
    same_author_after_other = _item(3, platform_user_id="user-1")
    late_followup = _item(
        4,
        platform_user_id="user-1",
        mentioned_bot=True,
        timestamp="2026-04-29T00:10:00+00:00",
    )
    too_late = _item(
        5,
        platform_user_id="user-1",
        timestamp="2026-04-29T00:13:00+00:00",
    )

    queue = queue_module.ChatInputQueue()
    survivors, collapsed = queue.coalesce_addressed_group([
        first,
        other_author,
        same_author_after_other,
        late_followup,
        too_late,
    ])

    assert [item.sequence for item in survivors] == [1, 2, 3, 4, 5]
    assert collapsed == []


@pytest.mark.asyncio
async def test_chat_enqueue_path_does_not_save_directly(monkeypatch) -> None:
    """The endpoint should enqueue only; worker code owns user persistence."""

    await _reset_queue_state()
    save_conversation = AsyncMock()
    monkeypatch.setattr(service_module, "save_conversation", save_conversation)
    monkeypatch.setattr(
        service_module,
        "_ensure_chat_input_worker_started",
        lambda **_kwargs: None,
    )

    chat_task = asyncio.create_task(service_module.chat(
        _request("endpoint"),
        BackgroundTasks(),
    ))
    await asyncio.sleep(0)

    assert service_module._chat_input_queue.pending_count() == 1
    save_conversation.assert_not_awaited()

    queued_item = service_module._chat_input_queue.pop_left_for_test()
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
    service_module._chat_input_queue.extend_for_test([dropped, tagged, bot_reply])

    service_module._ensure_chat_input_worker_started()
    await service_module._chat_input_queue.notify_for_test()

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
    service_module._chat_input_queue.extend_for_test([dropped, tagged, bot_reply])

    service_module._ensure_chat_input_worker_started()
    await service_module._chat_input_queue.notify_for_test()

    response = await asyncio.wait_for(dropped.future, timeout=1.0)

    assert response.messages == []
    save_conversation.assert_awaited()
    await asyncio.sleep(0)
    assert graph_message_ids == ["2"]

    graph_can_finish.set()
    await _reset_queue_state()


@pytest.mark.asyncio
async def test_worker_saves_collapsed_messages_before_graph(monkeypatch) -> None:
    """Collapsed originals should persist before the surviving graph run."""

    await _reset_queue_state()
    call_order = []
    captured_state = {}

    class _Graph:
        """Capture graph state for the collapsed survivor."""

        async def ainvoke(self, state):
            call_order.append("graph")
            captured_state.update(state)
            return {
                "should_respond": False,
                "use_reply_feature": True,
                "final_dialog": [],
                "future_promises": [],
                "consolidation_state": None,
            }

    async def _save_conversation(doc):
        call_order.append(f"save:{doc['platform_message_id']}")

    monkeypatch.setattr(service_module, "save_conversation", _save_conversation)
    _patch_common_dependencies(monkeypatch, _Graph())

    first = _item(
        1,
        channel_type="private",
        platform_channel_id="dm-1",
        platform_user_id="user-1",
        content="first",
    )
    second = _item(
        2,
        channel_type="private",
        platform_channel_id="dm-1",
        platform_user_id="user-1",
        content="second",
    )
    service_module._chat_input_queue.extend_for_test([first, second])

    service_module._ensure_chat_input_worker_started()
    await service_module._chat_input_queue.notify_for_test()

    collapsed_response = await asyncio.wait_for(second.future, timeout=1.0)
    survivor_response = await asyncio.wait_for(first.future, timeout=1.0)

    assert collapsed_response.messages == []
    assert survivor_response.messages == []
    assert call_order[:3] == ["save:2", "save:1", "graph"]
    assert captured_state["user_input"] == "first\nsecond"
    assert captured_state["use_reply_feature"] is False

    await _reset_queue_state()


@pytest.mark.asyncio
async def test_worker_preserves_collapsed_image_input(monkeypatch) -> None:
    """Collapsed image input should reach graph and persist safe metadata."""

    await _reset_queue_state()
    captured_state = {}
    saved_docs = []

    class _Graph:
        """Capture graph state for the multimedia turn."""

        async def ainvoke(self, state):
            captured_state.update(state)
            return {
                "should_respond": False,
                "use_reply_feature": False,
                "final_dialog": [],
                "future_promises": [],
                "consolidation_state": None,
            }

    async def _save_conversation(doc):
        saved_docs.append(doc)

    monkeypatch.setattr(service_module, "save_conversation", _save_conversation)
    _patch_common_dependencies(monkeypatch, _Graph())

    first = _item(
        1,
        platform_user_id="user-1",
        content="Kazusa, look",
        content_type="mixed",
        mentioned_bot=True,
        attachments=[{
            "media_type": "image/png",
            "base64_data": "first-image-bytes",
            "description": "first image",
            "url": "https://example.test/first.png",
            "size_bytes": 111,
        }],
    )
    second = _item(
        2,
        platform_user_id="user-1",
        content="",
        content_type="image",
        attachments=[{
            "media_type": "image/jpeg",
            "base64_data": "second-image-bytes",
            "description": "second image",
            "url": "https://example.test/second.jpg",
            "size_bytes": 222,
        }],
    )
    service_module._chat_input_queue.extend_for_test([first, second])

    service_module._ensure_chat_input_worker_started()
    await service_module._chat_input_queue.notify_for_test()

    collapsed_response = await asyncio.wait_for(second.future, timeout=1.0)
    survivor_response = await asyncio.wait_for(first.future, timeout=1.0)

    assert collapsed_response.messages == []
    assert survivor_response.messages == []
    assert captured_state["user_multimedia_input"] == [
        {
            "content_type": "image/png",
            "base64_data": "first-image-bytes",
            "description": "first image",
        },
        {
            "content_type": "image/jpeg",
            "base64_data": "second-image-bytes",
            "description": "second image",
        },
    ]

    saved_by_message_id = {
        doc["platform_message_id"]: doc
        for doc in saved_docs
    }
    assert saved_by_message_id["2"]["content"] == ""
    assert saved_by_message_id["2"]["content_type"] == "image"
    assert saved_by_message_id["2"]["attachments"] == [{
        "media_type": "image/jpeg",
        "url": "https://example.test/second.jpg",
        "description": "second image",
        "size_bytes": 222,
    }]
    assert "base64_data" not in saved_by_message_id["1"]["attachments"][0]
    assert "base64_data" not in saved_by_message_id["2"]["attachments"][0]

    await _reset_queue_state()
