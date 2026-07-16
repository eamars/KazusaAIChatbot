"""Tests for the brain service global input queue."""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import BackgroundTasks
from pydantic import ValidationError

from kazusa_ai_chatbot import chat_input_queue as queue_module
from kazusa_ai_chatbot import service as service_module
from kazusa_ai_chatbot.brain_service import intake as brain_intake
from kazusa_ai_chatbot.brain_service.turn_settlement import (
    AssessmentLease,
    PersistedChatFragment,
)
from kazusa_ai_chatbot.config import CHARACTER_GLOBAL_USER_ID
from kazusa_ai_chatbot.time_boundary import build_turn_clock_from_storage_utc


@pytest.fixture(autouse=True)
def _stub_service_event_logging(monkeypatch) -> None:
    """Keep deterministic queue tests off the event-log database."""

    recorder_names = (
        "record_database_operation_event",
        "record_pipeline_turn_event",
        "record_queue_intake_event",
        "record_runtime_error_event",
    )
    for recorder_name in recorder_names:
        monkeypatch.setattr(
            service_module.event_logging,
            recorder_name,
            AsyncMock(),
        )


def _request(
    message_id: str,
    *,
    channel_type: str = "group",
    platform_channel_id: str = "chan-1",
    platform_user_id: str | None = None,
    content: str | None = None,
    content_type: str = "text",
    attachments: list[dict[str, object]] | None = None,
    message_envelope: dict[str, object] | None = None,
    direct_address: bool = False,
    bot_reply: bool = False,
    listen_only: bool = False,
    local_timestamp: str = "",
) -> service_module.ChatRequest:
    """Build a chat request for queue tests.

    Args:
        message_id: Platform message identifier.
        content: Message body; empty string is preserved for media-only input.
        content_type: Adapter-provided content type.
        attachments: Adapter-provided attachment payloads.
        message_envelope: Optional typed envelope supplied by the adapter.
        direct_address: Whether the typed envelope directly mentions the bot.
        bot_reply: Whether the typed envelope addresses the character via reply.
        listen_only: Whether the adapter marked the message as observation-only.
        local_timestamp: Optional configured local timestamp supplied by the
            adapter or debug client.

    Returns:
        ChatRequest with deterministic payload fields.
    """

    body_text = f"message {message_id}" if content is None else content
    if message_envelope is None:
        mentions = []
        if direct_address:
            mentions.append({
                "platform_user_id": "bot-1",
                "global_user_id": CHARACTER_GLOBAL_USER_ID,
                "entity_kind": "bot",
                "raw_text": "@bot",
            })
        addressed_to = []
        if direct_address or bot_reply:
            addressed_to.append(CHARACTER_GLOBAL_USER_ID)
        if bot_reply:
            reply = {
                "platform_message_id": f"reply-{message_id}",
                "platform_user_id": "bot-1",
                "global_user_id": CHARACTER_GLOBAL_USER_ID,
                "display_name": "Kazusa",
                "excerpt": "previous bot message",
                "derivation": "platform_native",
            }
        else:
            reply = None
        message_envelope = {
            "body_text": body_text,
            "raw_wire_text": body_text,
            "mentions": mentions,
            "attachments": attachments or [],
            "addressed_to_global_user_ids": addressed_to,
            "broadcast": False,
        }
        if reply is not None:
            message_envelope["reply"] = reply
    request = service_module.ChatRequest(
        platform="qq",
        platform_channel_id=platform_channel_id,
        channel_type=channel_type,
        platform_message_id=message_id,
        platform_user_id=platform_user_id or f"user-{message_id}",
        platform_bot_id="bot-1",
        display_name=f"User {message_id}",
        channel_name="Group",
        content_type=content_type,
        message_envelope=message_envelope,
        local_timestamp=local_timestamp,
        debug_modes=service_module.DebugModesIn(listen_only=listen_only),
    )
    return request


def _item(
    sequence: int,
    *,
    platform_message_id: str | None = None,
    channel_type: str = "group",
    platform_channel_id: str = "chan-1",
    platform_user_id: str | None = None,
    content: str | None = None,
    content_type: str = "text",
    attachments: list[dict[str, object]] | None = None,
    message_envelope: dict[str, object] | None = None,
    storage_timestamp_utc: str | None = None,
    direct_address: bool = False,
    bot_reply: bool = False,
    listen_only: bool = False,
) -> queue_module.QueuedChatItem:
    """Build a queued item for deterministic pruning tests.

    Args:
        sequence: Queue sequence number.
        platform_message_id: Optional platform message id, for duplicate
            delivery fixtures where the queue sequence must remain distinct.
        content: Message body; empty string is preserved for media-only input.
        content_type: Adapter-provided content type.
        attachments: Adapter-provided attachment payloads.
        message_envelope: Optional typed envelope supplied by the adapter.
        direct_address: Whether the envelope directly mentions the bot.
        bot_reply: Whether the envelope addresses the character via reply.
        listen_only: Whether the adapter marked the message as observation-only.

    Returns:
        Queued chat item.
    """

    future: asyncio.Future[service_module.ChatResponse] = (
        asyncio.get_running_loop().create_future()
    )
    turn_clock = build_turn_clock_from_storage_utc(
        storage_timestamp_utc or f"2026-04-29T00:00:{sequence:02d}+00:00",
    )
    item = queue_module.QueuedChatItem(
        sequence=sequence,
        request=_request(
            platform_message_id or str(sequence),
            channel_type=channel_type,
            platform_channel_id=platform_channel_id,
            platform_user_id=platform_user_id,
            content=content,
            content_type=content_type,
            attachments=attachments,
            message_envelope=message_envelope,
            direct_address=direct_address,
            bot_reply=bot_reply,
            listen_only=listen_only,
        ),
        storage_timestamp_utc=turn_clock["storage_timestamp_utc"],
        local_timestamp=turn_clock["local_timestamp"],
        local_time_context=turn_clock["local_time_context"],
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


async def _resolved_envelope(req: service_module.ChatRequest) -> dict:
    """Return the request envelope using the service's normalized test shape."""

    envelope = req.message_envelope.model_dump(
        exclude_none=True,
        exclude_defaults=True,
    )
    return envelope


def test_frontline_reply_label_distinguishes_unresolved_reply() -> None:
    """A present reply without an author is distinct from no reply."""

    character_id = "character-global-id"

    assert service_module._frontline_reply_label({}, character_id) == "none"
    assert service_module._frontline_reply_label(
        {"reply": {"platform_message_id": "message-1"}},
        character_id,
    ) == "unknown_participant"


def _patch_common_dependencies(monkeypatch, graph) -> None:
    """Patch external service dependencies for queue-worker tests.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        graph: Fake service graph.

    Returns:
        None.
    """

    monkeypatch.setattr(
        service_module,
        "_static_character_profile",
        {
            "name": "Kazusa",
            "personality_brief": "static brief",
        },
    )
    monkeypatch.setattr(
        service_module,
        "_runtime_character_state",
        {
            "mood": "old mood",
            "global_vibe": "old vibe",
            "reflection_summary": "old reflection",
        },
    )
    monkeypatch.setattr(
        service_module,
        "get_character_runtime_state",
        AsyncMock(return_value={
            "mood": "fresh mood",
            "global_vibe": "fresh vibe",
            "reflection_summary": "fresh reflection",
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
        AsyncMock(return_value={"affinity": 500}),
    )
    monkeypatch.setattr(
        service_module,
        "get_conversation_history",
        AsyncMock(return_value=[]),
    )
    monkeypatch.setattr(
        service_module,
        "get_conversation_by_platform_message_id",
        AsyncMock(return_value=None),
    )
    monkeypatch.setattr(
        service_module,
        "build_promoted_reflection_context",
        AsyncMock(return_value={}),
    )
    monkeypatch.setattr(
        service_module,
        "_run_post_turn_memory_lifecycle_background",
        AsyncMock(side_effect=lambda state: state),
    )
    async def _frontline(_state):
        """Admit deterministic service fixtures through the new contract."""

        open_turns = _state.get("open_turns") or []
        if open_turns:
            return {
                "intake_action": "append",
                "append_target": "open_1",
                "prelude_targets": [],
                "reason": "fixture continuation",
            }
        return {
            "intake_action": "start",
            "append_target": "none",
            "prelude_targets": [],
            "reason": "fixture candidate",
        }

    async def _settled(_state):
        """Allow deterministic service fixtures to reach their fake graph."""

        return {
            "response_action": "proceed",
            "reason_to_respond": "fixture response",
            "use_reply_feature": False,
            "channel_topic": "",
            "indirect_speech_context": "",
        }

    async def _media(state):
        """Keep deterministic media fixtures out of the vision endpoint."""

        rows = []
        for row in state.get("user_multimedia_input") or []:
            rows.append({
                "content_type": row.get("content_type", ""),
                "base64_data": row.get("base64_data", ""),
                "description": row.get("description", ""),
            })
        return {
            "user_multimedia_input": rows,
            "additional_media_present": False,
        }

    monkeypatch.setattr(service_module, "frontline_relevance_agent", _frontline)
    monkeypatch.setattr(service_module, "relevance_agent", _settled)
    monkeypatch.setattr(service_module, "multimedia_descriptor_agent", _media)
    monkeypatch.setattr(service_module, "_graph", graph)


@pytest.mark.asyncio
async def test_graph_relevance_missing_decision_fails_closed(
    monkeypatch,
) -> None:
    """Graph entry cannot invoke settled relevance outside its coordinator."""

    relevance_agent = AsyncMock()
    monkeypatch.setattr(service_module, "relevance_agent", relevance_agent)

    result = await service_module._graph_relevance_node({})

    assert result["response_action"] == "ignore"
    assert result["should_respond"] is False
    relevance_agent.assert_not_awaited()


@pytest.mark.asyncio
async def test_settlement_worker_marks_active_model_work(
    monkeypatch,
) -> None:
    """Ready-turn relevance and cognition retain primary-work priority."""

    item = _item(1, direct_address=True)
    fragment = PersistedChatFragment(
        arrival_sequence=1,
        scope=("qq", "chan-1", "group"),
        author_platform_user_id="user-1",
        author_global_user_id="global-user-1",
        platform_message_id="1",
        conversation_row_id="row-1",
        storage_timestamp_utc="2026-04-29T00:00:01+00:00",
        enqueue_monotonic=0.0,
        body_text="Character, answer this question.",
        queue_item=item,
    )
    lease = AssessmentLease(
        turn_id="turn-1",
        version=1,
        observation_status="more_time_available",
        leader_sequence=1,
        response_owner_sequence=1,
        fragments=(fragment,),
    )

    class _ReadyCoordinator:
        def __init__(self) -> None:
            self.calls = 0
            self.block = asyncio.Event()

        async def wait_for_assessment_ready(self):
            self.calls += 1
            if self.calls == 1:
                return lease
            await self.block.wait()
            raise AssertionError("blocked wait unexpectedly resumed")

    entered = asyncio.Event()
    release = asyncio.Event()
    observed_busy: list[bool] = []

    async def _process(_lease, _response_owner) -> None:
        observed_busy.append(service_module._primary_interaction_busy())
        entered.set()
        await release.wait()

    coordinator = _ReadyCoordinator()
    monkeypatch.setattr(
        service_module,
        "_turn_settlement_coordinator",
        coordinator,
    )
    monkeypatch.setattr(service_module, "_process_settlement_lease", _process)
    monkeypatch.setattr(
        service_module,
        "_chat_input_queue",
        queue_module.ChatInputQueue(),
    )
    monkeypatch.setattr(service_module, "_primary_interaction_active_count", 0)

    task = asyncio.create_task(service_module._turn_settlement_worker())
    await entered.wait()
    assert observed_busy == [True]
    release.set()
    for _index in range(10):
        await asyncio.sleep(0)
        if service_module._primary_interaction_active_count == 0:
            break
    assert service_module._primary_interaction_active_count == 0
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


class _ForegroundHandle:
    """Foreground pipeline handle double for queue lifecycle tests."""

    def __init__(self) -> None:
        self.entered = False
        self.closed = False

    async def __aenter__(self):
        self.entered = True
        return self

    async def __aexit__(self, _exc_type, _exc, _tb) -> None:
        self.closed = True

    def cancelled(self) -> bool:
        return_value = False
        return return_value

    def raise_if_cancelled(self, _checkpoint: str) -> None:
        return None


class _CoordinatorDouble:
    """Coordinator double that records foreground queue coordination calls."""

    def __init__(self) -> None:
        self.handle = _ForegroundHandle()
        self.cancelled: list[dict[str, object]] = []
        self.started: list[dict[str, object]] = []

    def request_cancellation(self, **kwargs) -> list[str]:
        self.cancelled.append(kwargs)
        return []

    async def start_run(self, **kwargs):
        self.started.append(kwargs)
        return SimpleNamespace(
            admitted=True,
            handle=self.handle,
            defer_reason=None,
        )


@pytest.mark.asyncio
async def test_intake_save_user_message_from_item_returns_row_id() -> None:
    """Intake should return the inserted conversation row ID from persistence."""
    item = _item(1)
    captured_docs = []

    async def _save_conversation(doc):
        captured_docs.append(doc)
        return_value = "row-1"
        return return_value

    row_id = await brain_intake.save_user_message_from_item(
        item,
        global_user_id="global-user-1",
        reply_context={},
        save_conversation_func=_save_conversation,
        resolve_message_envelope_identities_func=_resolved_envelope,
        logger=logging.getLogger("tests.service_input_queue"),
    )

    assert row_id == "row-1"
    assert captured_docs[0]["timestamp"] == item.storage_timestamp_utc


@pytest.mark.asyncio
async def test_intake_persists_sanitized_channel_name_metadata() -> None:
    """Usable group labels should persist as optional row metadata only."""

    item = _item(1)
    item.request.channel_name = "动画讨论群"
    captured_docs = []

    async def _save_conversation(doc):
        captured_docs.append(doc)
        return "row-1"

    row_id = await brain_intake.save_user_message_from_item(
        item,
        global_user_id="global-user-1",
        reply_context={},
        save_conversation_func=_save_conversation,
        resolve_message_envelope_identities_func=_resolved_envelope,
        logger=logging.getLogger("tests.service_input_queue"),
    )

    assert row_id == "row-1"
    assert captured_docs[0]["channel_name"] == "动画讨论群"


@pytest.mark.asyncio
async def test_intake_drops_synthetic_channel_name_metadata() -> None:
    """Synthetic platform labels must not become durable group names."""

    item = _item(1)
    item.request.channel_name = "Group 227608960"
    captured_docs = []

    async def _save_conversation(doc):
        captured_docs.append(doc)
        return "row-1"

    row_id = await brain_intake.save_user_message_from_item(
        item,
        global_user_id="global-user-1",
        reply_context={},
        save_conversation_func=_save_conversation,
        resolve_message_envelope_identities_func=_resolved_envelope,
        logger=logging.getLogger("tests.service_input_queue"),
    )

    assert row_id == "row-1"
    assert "channel_name" not in captured_docs[0]


@pytest.mark.asyncio
async def test_queue_separates_storage_utc_and_local_timestamp() -> None:
    """Queue enqueue should keep storage UTC separate from configured local time."""

    request = _request(
        "clock",
        local_timestamp="2026-05-17 16:55:28.395",
    )
    queue = queue_module.ChatInputQueue()
    enqueue_task = asyncio.create_task(queue.enqueue(request))
    await asyncio.sleep(0)

    queued_item = queue.pop_left_for_test()

    assert queued_item.storage_timestamp_utc == (
        "2026-05-17T04:55:28.395000+00:00"
    )
    assert queued_item.local_timestamp == "2026-05-17 16:55:28.395000"
    assert queued_item.local_time_context == {
        "current_local_datetime": "2026-05-17 16:55",
        "current_local_weekday": "Sunday",
    }
    assert not hasattr(queued_item, "timestamp")

    queued_item.future.set_result(service_module.ChatResponse())
    response = await asyncio.wait_for(enqueue_task, timeout=1.0)

    assert response.messages == []


def test_chat_request_timestamp_field_is_rejected() -> None:
    """The old `/chat` timestamp field should fail the request contract."""

    payload = _request("legacy").model_dump()
    payload["timestamp"] = "2026-05-17T04:55:28+00:00"

    with pytest.raises(ValidationError):
        service_module.ChatRequest(**payload)


@pytest.mark.asyncio
async def test_intake_save_user_message_from_item_returns_none_on_save_failure(
    caplog,
) -> None:
    """Intake should keep existing save-failure degradation and return None."""
    item = _item(1)
    test_logger = logging.getLogger("tests.service_input_queue")

    async def _save_conversation(doc):
        raise RuntimeError("save failed")

    with caplog.at_level(logging.ERROR, logger="tests.service_input_queue"):
        row_id = await brain_intake.save_user_message_from_item(
            item,
            global_user_id="global-user-1",
            reply_context={},
            save_conversation_func=_save_conversation,
            resolve_message_envelope_identities_func=_resolved_envelope,
            logger=test_logger,
        )

    assert row_id is None
    assert "Failed to save queued user message: save failed" in caplog.text


@pytest.mark.asyncio
async def test_active_turn_conversation_row_ids_skip_empty_and_dedupe() -> None:
    """Active row IDs should preserve arrival order without empty defaults."""
    survivor = _item(1)
    collapsed_empty = _item(2)
    collapsed_duplicate = _item(3)
    survivor.conversation_row_id = "row-1"
    collapsed_empty.conversation_row_id = ""
    collapsed_duplicate.conversation_row_id = "row-1"
    survivor.collapsed_items = [collapsed_empty, collapsed_duplicate]

    row_ids = brain_intake.active_turn_conversation_row_ids(survivor)

    assert row_ids == ["row-1"]


@pytest.mark.asyncio
async def test_active_three_message_burst_is_not_threshold_pruned() -> None:
    """Every active message survives regardless of address metadata."""

    plain = _item(1)
    tagged = _item(2, direct_address=True)
    bot_reply = _item(3, bot_reply=True)

    queue = queue_module.ChatInputQueue()
    survivors, dropped = queue.filter_debug_bypass([
        plain,
        tagged,
        bot_reply,
    ])

    assert [item.sequence for item in survivors] == [1, 2, 3]
    assert dropped == []


@pytest.mark.asyncio
async def test_active_six_message_burst_is_not_threshold_pruned() -> None:
    """A larger active burst still reaches frontline in arrival order."""

    items = [
        _item(1, direct_address=True),
        _item(2, direct_address=True),
        _item(3, bot_reply=True),
        _item(4, bot_reply=True),
        _item(5, bot_reply=True),
        _item(6, bot_reply=True),
    ]

    queue = queue_module.ChatInputQueue()
    survivors, dropped = queue.filter_debug_bypass(items)

    assert [item.sequence for item in survivors] == [1, 2, 3, 4, 5, 6]
    assert dropped == []


@pytest.mark.asyncio
async def test_active_reply_burst_preserves_every_message() -> None:
    """Reply-heavy input has no queue-level semantic discard."""

    items = [
        _item(1, bot_reply=True),
        _item(2, bot_reply=True),
        _item(3, bot_reply=True),
        _item(4, bot_reply=True),
        _item(5, bot_reply=True),
        _item(6, bot_reply=True),
    ]

    queue = queue_module.ChatInputQueue()
    survivors, dropped = queue.filter_debug_bypass(items)

    assert [item.sequence for item in survivors] == [1, 2, 3, 4, 5, 6]
    assert dropped == []


@pytest.mark.asyncio
async def test_active_messages_need_no_queue_level_protection_marker() -> None:
    """Typed target differences do not change active queue retention."""

    plain_reply = _item(1, bot_reply=False)
    tagged = _item(2, direct_address=True)
    bot_reply = _item(3, bot_reply=True)

    queue = queue_module.ChatInputQueue()
    survivors, dropped = queue.filter_debug_bypass([
        plain_reply,
        tagged,
        bot_reply,
    ])

    assert [item.sequence for item in survivors] == [1, 2, 3]
    assert dropped == []


@pytest.mark.asyncio
async def test_private_and_group_messages_share_active_retention() -> None:
    """Private and group input both reach their frontline path."""

    plain_group = _item(1)
    private_message = _item(
        2,
        channel_type="private",
        platform_channel_id="dm-1",
        platform_user_id="user-private",
    )
    tagged_group = _item(3, direct_address=True)

    queue = queue_module.ChatInputQueue()
    survivors, dropped = queue.filter_debug_bypass([
        plain_group,
        private_message,
        tagged_group,
    ])

    assert [item.sequence for item in survivors] == [1, 2, 3]
    assert dropped == []


@pytest.mark.asyncio
async def test_listen_only_messages_are_dropped_under_threshold() -> None:
    """Listen-only messages should drop without running graph pressure policy."""

    plain_group = _item(1)
    listen_only = _item(2, listen_only=True)

    queue = queue_module.ChatInputQueue()
    survivors, dropped = queue.filter_debug_bypass([
        plain_group,
        listen_only,
    ])

    assert [item.sequence for item in survivors] == [1]
    assert [item.sequence for item in dropped] == [2]


@pytest.mark.asyncio
async def test_listen_only_bypass_preserves_neighboring_active_messages() -> None:
    """Listen-only bypass leaves neighboring active messages intact."""

    plain_group = _item(1, channel_type="private")
    listen_only = _item(2, listen_only=True)
    tagged_group = _item(3, channel_type="private", direct_address=True)

    queue = queue_module.ChatInputQueue()
    survivors, dropped = queue.filter_debug_bypass([
        plain_group,
        listen_only,
        tagged_group,
    ])

    assert [item.sequence for item in survivors] == [1, 3]
    assert [item.sequence for item in dropped] == [2]


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
async def test_private_messages_require_adjacency_to_coalesce() -> None:
    """Private follow-ups separated by another scope should not collapse."""

    first = _item(
        1,
        channel_type="private",
        platform_channel_id="dm-1",
        platform_user_id="user-1",
        content="first",
    )
    other_scope = _item(
        2,
        channel_type="private",
        platform_channel_id="dm-2",
        platform_user_id="user-2",
        content="other scope",
    )
    later_same_scope = _item(
        3,
        channel_type="private",
        platform_channel_id="dm-1",
        platform_user_id="user-1",
        content="later same scope",
    )

    queue = queue_module.ChatInputQueue()
    survivors, collapsed = queue.coalesce_private([
        first,
        other_scope,
        later_same_scope,
    ])

    assert [item.sequence for item in survivors] == [1, 2, 3]
    assert collapsed == []
    assert first.collapsed_items == []
    assert first.combined_content is None


@pytest.mark.asyncio
async def test_duplicate_private_platform_message_ids_do_not_coalesce() -> None:
    """Duplicate private deliveries should not be treated as follow-ups."""

    first = _item(
        1,
        platform_message_id="same-message",
        channel_type="private",
        platform_channel_id="dm-1",
        platform_user_id="user-1",
        content="first delivery",
    )
    duplicate = _item(
        2,
        platform_message_id="same-message",
        channel_type="private",
        platform_channel_id="dm-1",
        platform_user_id="user-1",
        content="duplicate delivery",
    )

    queue = queue_module.ChatInputQueue()
    survivors, collapsed = queue.coalesce_private([first, duplicate])

    assert [item.sequence for item in survivors] == [1, 2]
    assert collapsed == []
    assert first.collapsed_items == []
    assert first.combined_content is None


@pytest.mark.asyncio
async def test_group_items_remain_individual_for_frontline() -> None:
    """Group messages reach frontline without adjacency coalescing."""

    first = _item(
        1,
        platform_user_id="user-1",
        content="Character,",
        direct_address=True,
    )
    second = _item(
        2,
        platform_user_id="user-1",
        content="one more detail",
    )

    queue = queue_module.ChatInputQueue()
    survivors, dropped = queue.filter_debug_bypass([first, second])

    assert [item.sequence for item in survivors] == [1, 2]
    assert dropped == []


@pytest.mark.asyncio
async def test_assembled_response_is_delivered_only_to_response_owner() -> None:
    """Appended request futures stay silent when the logical turn responds."""

    first = _item(1, channel_type="private")
    second = _item(2, channel_type="private")
    fragments = tuple(
        PersistedChatFragment(
            arrival_sequence=item.sequence,
            scope=("qq", "chan-1", "private"),
            author_platform_user_id="user-private",
            author_global_user_id="global-user-private",
            platform_message_id=str(item.sequence),
            conversation_row_id=f"row-{item.sequence}",
            storage_timestamp_utc=item.storage_timestamp_utc,
            enqueue_monotonic=item.enqueue_monotonic,
            body_text=f"fragment-{item.sequence}",
            queue_item=item,
        )
        for item in (first, second)
    )
    lease = AssessmentLease(
        turn_id="turn-private",
        version=2,
        observation_status="observation_complete",
        leader_sequence=1,
        response_owner_sequence=1,
        fragments=fragments,
    )

    await service_module._complete_settled_fragments(
        lease,
        service_module.ChatResponse(messages=["one visible reply"]),
    )

    assert first.future.result().messages == ["one visible reply"]
    assert second.future.result().messages == []


@pytest.mark.asyncio
async def test_native_reply_is_suppressed_for_obsolete_response_owner(
    monkeypatch,
) -> None:
    """An assembled answer cannot quote an older opening fragment."""

    await _reset_queue_state()

    class _Graph:
        """Request one visible native-anchored response."""

        async def ainvoke(self, state):
            assert state["use_reply_feature"] is True
            return {
                "should_respond": True,
                "use_reply_feature": state["use_reply_feature"],
                "final_dialog": ["assembled answer"],
                "future_promises": [],
                "consolidation_state": None,
            }

    monkeypatch.setattr(
        service_module,
        "_save_assistant_message",
        AsyncMock(),
    )
    _patch_common_dependencies(monkeypatch, _Graph())
    opening_item = _item(1, direct_address=True)
    opening_item.conversation_row_id = "row-1"
    followup_item = _item(
        2,
        platform_user_id="user-1",
        content="Here is the actual question.",
    )
    followup_item.conversation_row_id = "row-2"
    fragments = [
        PersistedChatFragment(
            arrival_sequence=queued_item.sequence,
            scope=("qq", "chan-1", "group"),
            author_platform_user_id="user-1",
            author_global_user_id="global-user-1",
            platform_message_id=str(queued_item.sequence),
            conversation_row_id=queued_item.conversation_row_id,
            storage_timestamp_utc=queued_item.storage_timestamp_utc,
            enqueue_monotonic=queued_item.enqueue_monotonic,
            body_text=queued_item.request.message_envelope.body_text,
            queue_item=queued_item,
        )
        for queued_item in (opening_item, followup_item)
    ]

    await service_module._process_queued_chat_item(
        opening_item,
        settlement_fragments=fragments,
        settled_decision={
            "response_action": "proceed",
            "reason_to_respond": "specific group question",
            "use_reply_feature": True,
            "channel_topic": "question",
            "indirect_speech_context": "",
        },
        skip_user_persist=True,
        settlement_turn_id="turn-1",
        settlement_version=2,
        settlement_claimed=True,
        prepared_media=[],
        media_prepared=True,
    )

    response = await opening_item.future
    assert response.messages == ["assembled answer"]
    assert response.use_reply_feature is False
    await _reset_queue_state()


@pytest.mark.asyncio
async def test_native_reply_reaches_single_fragment_response(
    monkeypatch,
) -> None:
    """A compatible single-fragment owner preserves the semantic request."""

    await _reset_queue_state()

    class _Graph:
        """Request one visible native-anchored response."""

        async def ainvoke(self, state):
            assert state["use_reply_feature"] is True
            return {
                "should_respond": True,
                "use_reply_feature": state["use_reply_feature"],
                "final_dialog": ["direct answer"],
                "future_promises": [],
                "consolidation_state": None,
            }

    monkeypatch.setattr(
        service_module,
        "_save_assistant_message",
        AsyncMock(),
    )
    _patch_common_dependencies(monkeypatch, _Graph())
    item = _item(1, direct_address=True)
    item.conversation_row_id = "row-1"
    fragment = PersistedChatFragment(
        arrival_sequence=item.sequence,
        scope=("qq", "chan-1", "group"),
        author_platform_user_id="user-1",
        author_global_user_id="global-user-1",
        platform_message_id="1",
        conversation_row_id=item.conversation_row_id,
        storage_timestamp_utc=item.storage_timestamp_utc,
        enqueue_monotonic=item.enqueue_monotonic,
        body_text=item.request.message_envelope.body_text,
        queue_item=item,
    )

    await service_module._process_queued_chat_item(
        item,
        settlement_fragments=[fragment],
        settled_decision={
            "response_action": "proceed",
            "reason_to_respond": "specific group question",
            "use_reply_feature": True,
            "channel_topic": "question",
            "indirect_speech_context": "",
        },
        skip_user_persist=True,
        settlement_turn_id="turn-1",
        settlement_version=1,
        settlement_claimed=True,
        prepared_media=[],
        media_prepared=True,
    )

    response = await item.future
    assert response.messages == ["direct answer"]
    assert response.use_reply_feature is True
    await _reset_queue_state()


@pytest.mark.asyncio
async def test_settled_fresh_history_excludes_active_turn_fragments(
    monkeypatch,
) -> None:
    """Only external rows can act as fresh-history scene evidence."""

    monkeypatch.setattr(
        service_module,
        "_static_character_profile",
        {"name": "Kazusa"},
    )
    item = _item(1)
    fragment = PersistedChatFragment(
        arrival_sequence=1,
        scope=("qq", "chan-1", "group"),
        author_platform_user_id="user-1",
        author_global_user_id="global-user-1",
        platform_message_id="message-active",
        conversation_row_id="row-active",
        storage_timestamp_utc=item.storage_timestamp_utc,
        enqueue_monotonic=item.enqueue_monotonic,
        body_text="active request",
        semantic_target_labels=("character",),
        queue_item=item,
    )
    second_item = _item(2)
    second_fragment = PersistedChatFragment(
        arrival_sequence=2,
        scope=("qq", "chan-1", "group"),
        author_platform_user_id="user-1",
        author_global_user_id="global-user-1",
        platform_message_id="message-active-2",
        conversation_row_id="row-active-2",
        storage_timestamp_utc=second_item.storage_timestamp_utc,
        enqueue_monotonic=second_item.enqueue_monotonic,
        body_text="active follow-up",
        semantic_target_labels=("none",),
        queue_item=second_item,
    )
    lease = AssessmentLease(
        turn_id="turn-history",
        version=1,
        observation_status="observation_complete",
        leader_sequence=1,
        response_owner_sequence=1,
        fragments=(fragment, second_fragment),
    )
    common = {
        "role": "user",
        "platform_user_id": "user-1",
        "global_user_id": "global-user-1",
        "display_name": "User",
        "addressed_to_global_user_ids": [],
        "mentions": [],
        "broadcast": False,
        "attachments": [],
        "timestamp": "2026-07-16T00:00:00+00:00",
    }
    state = service_module._settled_state_from_lease(
        lease,
        history=[
            {
                **common,
                "_id": "row-active",
                "platform_message_id": "message-active",
                "body_text": "active request",
            },
            {
                **common,
                "_id": "row-other",
                "platform_message_id": "message-other",
                "body_text": "another participant answered",
            },
            {
                **common,
                "_id": "row-active-2",
                "platform_message_id": "message-active-2",
                "body_text": "active follow-up",
            },
            {
                **common,
                "_id": "row-after",
                "platform_message_id": "message-after",
                "body_text": "later context",
            },
        ],
    )

    assert [row["platform_message_id"] for row in state["fresh_history"]] == [
        "message-other",
        "message-after",
    ]
    assert state["fresh_history"][0]["turn_temporal_relation"] == (
        "during_active_turn"
    )
    assert state["fresh_history"][1]["turn_temporal_relation"] == (
        "after_active_turn"
    )
    assert state["relationship_context"] == "direct participant"
    assert state["group_attention"] == "medium_noise"
    assert state["conversation_scope"] == "group"
    assert state["active_character_name"] == "Kazusa"
    assert state["current_author_global_user_id"] == "global-user-1"
    assert state["current_author_platform_user_id"] == "user-1"
    assert state["character_global_user_id"] == CHARACTER_GLOBAL_USER_ID
    assert state["platform_bot_id"] == "bot-1"


@pytest.mark.asyncio
async def test_settled_history_uses_timestamps_when_active_row_is_outside_window(
) -> None:
    """Intervening rows stay ordered when a busy group evicts the anchor."""

    item = _item(
        1,
        storage_timestamp_utc="2026-07-16T00:00:05+00:00",
    )
    first_fragment = PersistedChatFragment(
        arrival_sequence=1,
        scope=("qq", "chan-1", "group"),
        author_platform_user_id="user-1",
        author_global_user_id="global-user-1",
        platform_message_id="message-active",
        conversation_row_id="row-active",
        storage_timestamp_utc=item.storage_timestamp_utc,
        enqueue_monotonic=item.enqueue_monotonic,
        body_text="active request",
        semantic_target_labels=("character",),
        queue_item=item,
    )
    second_item = _item(
        2,
        storage_timestamp_utc="2026-07-16T00:00:07+00:00",
    )
    second_fragment = PersistedChatFragment(
        arrival_sequence=2,
        scope=("qq", "chan-1", "group"),
        author_platform_user_id="user-1",
        author_global_user_id="global-user-1",
        platform_message_id="message-active-2",
        conversation_row_id="row-active-2",
        storage_timestamp_utc=second_item.storage_timestamp_utc,
        enqueue_monotonic=second_item.enqueue_monotonic,
        body_text="active follow-up",
        semantic_target_labels=("none",),
        queue_item=second_item,
    )
    lease = AssessmentLease(
        turn_id="turn-history-clipped",
        version=1,
        observation_status="observation_complete",
        leader_sequence=1,
        response_owner_sequence=1,
        fragments=(first_fragment, second_fragment),
    )
    common = {
        "role": "user",
        "platform_user_id": "user-2",
        "global_user_id": "global-user-2",
        "display_name": "Other user",
        "addressed_to_global_user_ids": [],
        "mentions": [],
        "broadcast": False,
        "attachments": [],
    }

    state = service_module._settled_state_from_lease(
        lease,
        history=[
            {
                **common,
                "platform_message_id": "message-before",
                "body_text": "earlier context",
                "timestamp": "2026-07-16T00:00:04+00:00",
            },
            {
                **common,
                "platform_message_id": "message-during",
                "body_text": "intervening answer",
                "timestamp": "2026-07-16T00:00:06+00:00",
            },
            {
                **common,
                "platform_message_id": "message-after",
                "body_text": "later context",
                "timestamp": "2026-07-16T00:00:08+00:00",
            },
        ],
    )

    assert [
        row["turn_temporal_relation"] for row in state["fresh_history"]
    ] == [
        "before_active_turn",
        "during_active_turn",
        "after_active_turn",
    ]


@pytest.mark.asyncio
async def test_bot_continuity_follows_successful_visible_persistence(
    monkeypatch,
) -> None:
    """Only a persisted visible response becomes relevance continuity."""

    await _reset_queue_state()
    call_order: list[str] = []

    class _Graph:
        """Return one visible dialog message through the service path."""

        async def ainvoke(self, _state):
            return {
                "should_respond": True,
                "use_reply_feature": False,
                "final_dialog": ["visible reply"],
                "future_promises": [],
                "consolidation_state": None,
            }

    async def _save_conversation(doc):
        return f"row-{doc['platform_message_id']}"

    async def _save_assistant(_result):
        call_order.append("assistant_persisted")

    async def _record_continuity(**_kwargs):
        call_order.append("continuity_recorded")

    monkeypatch.setattr(service_module, "save_conversation", _save_conversation)
    monkeypatch.setattr(
        service_module,
        "_save_assistant_message",
        _save_assistant,
    )
    monkeypatch.setattr(
        service_module._turn_settlement_coordinator,
        "record_bot_continuity",
        _record_continuity,
    )
    _patch_common_dependencies(monkeypatch, _Graph())

    item = _item(
        1,
        channel_type="private",
        platform_channel_id="dm-1",
        platform_user_id="user-1",
        content="hello",
    )
    service_module._chat_input_queue.extend_for_test([item])

    service_module._ensure_chat_input_worker_started()
    await service_module._chat_input_queue.notify_for_test()
    response = await asyncio.wait_for(item.future, timeout=1.0)

    assert response.messages == ["visible reply"]
    assert call_order == ["assistant_persisted", "continuity_recorded"]

    await _reset_queue_state()


@pytest.mark.asyncio
async def test_failed_assistant_persistence_does_not_record_bot_continuity(
    monkeypatch,
) -> None:
    """A response suppressed by persistence failure is absent from context."""

    await _reset_queue_state()

    class _Graph:
        """Return dialog that fails during assistant persistence."""

        async def ainvoke(self, _state):
            return {
                "should_respond": True,
                "use_reply_feature": False,
                "final_dialog": ["unsaved reply"],
                "future_promises": [],
                "consolidation_state": None,
            }

    async def _save_conversation(doc):
        return f"row-{doc['platform_message_id']}"

    record_continuity = AsyncMock()
    monkeypatch.setattr(service_module, "save_conversation", _save_conversation)
    monkeypatch.setattr(
        service_module,
        "_save_assistant_message",
        AsyncMock(side_effect=RuntimeError("save failed")),
    )
    monkeypatch.setattr(
        service_module._turn_settlement_coordinator,
        "record_bot_continuity",
        record_continuity,
    )
    _patch_common_dependencies(monkeypatch, _Graph())

    item = _item(
        1,
        channel_type="private",
        platform_channel_id="dm-1",
        platform_user_id="user-1",
        content="hello",
    )
    service_module._chat_input_queue.extend_for_test([item])

    service_module._ensure_chat_input_worker_started()
    await service_module._chat_input_queue.notify_for_test()
    response = await asyncio.wait_for(item.future, timeout=1.0)

    assert response.messages == []
    record_continuity.assert_not_awaited()

    await _reset_queue_state()


@pytest.mark.asyncio
async def test_settled_media_budget_is_shared_across_reassessment(
    monkeypatch,
) -> None:
    """A stale lease cannot spend a second four-image descriptor budget."""

    descriptor_calls: list[list[str]] = []

    def _media_fragment(
        item: queue_module.QueuedChatItem,
    ) -> PersistedChatFragment:
        envelope = item.request.message_envelope.model_dump(
            exclude_none=True,
            exclude_defaults=True,
        )
        item.resolved_message_envelope = envelope
        return PersistedChatFragment(
            arrival_sequence=item.sequence,
            scope=("qq", "chan-1", "group"),
            author_platform_user_id="user-1",
            author_global_user_id="global-user-1",
            platform_message_id=item.request.platform_message_id,
            conversation_row_id=f"row-{item.sequence}",
            storage_timestamp_utc=item.storage_timestamp_utc,
            enqueue_monotonic=item.enqueue_monotonic,
            body_text=item.request.message_envelope.body_text,
            semantic_target_labels=("character",),
            reply_target_label="character",
            media_labels=tuple(
                attachment.media_type
                for attachment in item.request.message_envelope.attachments
            ),
            attachments=tuple(dict(row) for row in envelope["attachments"]),
            queue_item=item,
        )

    async def _describe(state):
        rows = state["user_multimedia_input"]
        descriptor_calls.append([row["base64_data"] for row in rows])
        return {
            "user_multimedia_input": [
                {
                    "content_type": row["content_type"],
                    "base64_data": row["base64_data"],
                    "description": f"described {row['base64_data']}",
                }
                for row in rows
            ],
            "additional_media_present": False,
        }

    monkeypatch.setattr(service_module, "multimedia_descriptor_agent", _describe)
    monkeypatch.setattr(
        service_module,
        "_hydrate_reply_context",
        AsyncMock(return_value=None),
    )
    opening_item = _item(
        1,
        attachments=[{
            "media_type": "image/png",
            "base64_data": "opening-0",
            "description": "",
        }],
    )
    followup_item = _item(
        2,
        attachments=[
            {
                "media_type": "image/png",
                "base64_data": f"followup-{index}",
                "description": "",
            }
            for index in range(5)
        ],
    )
    opening = _media_fragment(opening_item)
    followup = _media_fragment(followup_item)
    first_lease = AssessmentLease(
        turn_id="turn-media",
        version=1,
        observation_status="more_time_available",
        leader_sequence=1,
        response_owner_sequence=1,
        fragments=(opening,),
    )
    final_lease = AssessmentLease(
        turn_id="turn-media",
        version=2,
        observation_status="observation_complete",
        leader_sequence=1,
        response_owner_sequence=1,
        fragments=(opening, followup),
    )

    first_rows, first_overflow = await service_module._prepare_settled_media(
        first_lease,
    )
    final_rows, final_overflow = await service_module._prepare_settled_media(
        final_lease,
    )

    assert descriptor_calls == [
        ["opening-0"],
        ["followup-2", "followup-3", "followup-4"],
    ]
    assert len(first_rows) == 1
    assert first_overflow is False
    assert len(final_rows) == 4
    assert final_overflow is True
    assert opening.additional_media_present is True
    assert len(followup.media_descriptions) == 3


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
async def test_enqueue_requests_same_scope_background_cancellation(
    monkeypatch,
) -> None:
    """Foreground enqueue should cancel same-scope background pipelines."""

    from kazusa_ai_chatbot.runtime_coordination import PipelineScope

    await _reset_queue_state()
    coordinator = _CoordinatorDouble()
    monkeypatch.setattr(
        service_module,
        "_pipeline_coordinator",
        coordinator,
        raising=False,
    )
    monkeypatch.setattr(
        service_module,
        "_ensure_chat_input_worker_started",
        lambda **_kwargs: None,
    )

    enqueue_task = asyncio.create_task(
        service_module._enqueue_chat_request(_request("foreground"))
    )
    await asyncio.sleep(0)

    queued_item = service_module._chat_input_queue.pop_left_for_test()

    assert coordinator.cancelled == [
        {
            "scope": PipelineScope(
                platform="qq",
                platform_channel_id="chan-1",
                channel_type="group",
            ),
            "requested_by": "service.chat_queue",
            "reason": "same_scope_foreground_pending",
        }
    ]
    assert coordinator.started[0]["scope"] == PipelineScope(
        platform="qq",
        platform_channel_id="chan-1",
        channel_type="group",
    )
    assert coordinator.started[0]["precedence"] == "foreground"
    assert coordinator.started[0]["run_kind"] == "chat"
    assert getattr(queued_item, "pipeline_run_handle") is coordinator.handle

    queued_item.future.set_result(service_module.ChatResponse())
    response = await asyncio.wait_for(enqueue_task, timeout=1.0)

    assert response.messages == []
    await _reset_queue_state()


@pytest.mark.asyncio
async def test_cancelled_enqueue_wait_keeps_foreground_handle(
    monkeypatch,
) -> None:
    """Caller cancellation after enqueue must not reopen background admission."""

    await _reset_queue_state()
    coordinator = _CoordinatorDouble()
    monkeypatch.setattr(
        service_module,
        "_pipeline_coordinator",
        coordinator,
        raising=False,
    )
    monkeypatch.setattr(
        service_module,
        "_ensure_chat_input_worker_started",
        lambda **_kwargs: None,
    )

    enqueue_task = asyncio.create_task(
        service_module._enqueue_chat_request(_request("foreground"))
    )
    await asyncio.sleep(0)

    assert service_module._chat_input_queue.pending_count() == 1
    enqueue_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await enqueue_task

    queued_item = service_module._chat_input_queue.pop_left_for_test()
    assert getattr(queued_item, "pipeline_run_handle") is coordinator.handle
    assert coordinator.handle.closed is False

    await queued_item.pipeline_run_handle.__aexit__(None, None, None)
    await _reset_queue_state()


@pytest.mark.asyncio
async def test_dropped_queue_item_releases_foreground_handle(monkeypatch) -> None:
    """Pruned foreground items must not leak same-scope coordination handles."""

    await _reset_queue_state()
    item = _item(1)
    handle = _ForegroundHandle()
    item.pipeline_run_handle = handle
    monkeypatch.setattr(
        service_module,
        "_resolve_queued_user",
        AsyncMock(return_value=("global-user-1", {"affinity": 500})),
    )
    monkeypatch.setattr(
        service_module,
        "_hydrate_reply_context",
        AsyncMock(return_value={}),
    )
    monkeypatch.setattr(
        service_module,
        "_save_user_message_from_item",
        AsyncMock(return_value="row-1"),
    )

    committed = await service_module._drop_queued_chat_item(item)

    assert committed is True
    assert item.future.done()
    assert handle.closed is True


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
        return_value = f"row-{len(call_order)}"
        return return_value

    monkeypatch.setattr(service_module, "save_conversation", _save_conversation)
    _patch_common_dependencies(monkeypatch, _Graph())

    dropped = _item(1, listen_only=True)
    tagged = _item(
        2,
        channel_type="private",
        direct_address=True,
    )
    service_module._chat_input_queue.extend_for_test([dropped, tagged])

    service_module._ensure_chat_input_worker_started()
    await service_module._chat_input_queue.notify_for_test()

    dropped_response = await asyncio.wait_for(dropped.future, timeout=1.0)
    await asyncio.wait_for(graph_started.wait(), timeout=1.0)

    assert dropped_response.messages == []
    assert call_order[:2] == ["save", "save"]
    assert call_order[2] == "graph"
    graph_can_finish.set()
    tagged_response = await asyncio.wait_for(tagged.future, timeout=1.0)
    assert tagged_response.messages == []

    await _reset_queue_state()

@pytest.mark.asyncio
async def test_dropped_message_never_invokes_graph(monkeypatch) -> None:
    """A pruned message should be saved and completed without graph execution."""

    await _reset_queue_state()
    graph_message_ids = []
    graph_started = asyncio.Event()
    graph_can_finish = asyncio.Event()

    class _Graph:
        """Record graph message IDs for surviving items."""

        async def ainvoke(self, state):
            graph_message_ids.append(state["platform_message_id"])
            graph_started.set()
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

    dropped = _item(1, listen_only=True)
    tagged = _item(
        2,
        channel_type="private",
        direct_address=True,
    )
    service_module._chat_input_queue.extend_for_test([dropped, tagged])

    service_module._ensure_chat_input_worker_started()
    await service_module._chat_input_queue.notify_for_test()

    response = await asyncio.wait_for(dropped.future, timeout=1.0)

    assert response.messages == []
    save_conversation.assert_awaited()
    await asyncio.wait_for(graph_started.wait(), timeout=1.0)
    assert graph_message_ids == ["2"]

    graph_can_finish.set()
    await _reset_queue_state()


@pytest.mark.asyncio
async def test_worker_suppresses_graph_when_surviving_user_save_fails(
    monkeypatch,
) -> None:
    """Active turns should fail closed when incoming persistence fails."""

    await _reset_queue_state()

    class _Graph:
        """Expose a mock graph call for fail-closed assertions."""

        def __init__(self):
            self.ainvoke = AsyncMock(return_value={})

    graph = _Graph()
    monkeypatch.setattr(
        service_module,
        "save_conversation",
        AsyncMock(return_value=None),
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
    _patch_common_dependencies(monkeypatch, graph)

    item = _item(1, direct_address=True)
    service_module._chat_input_queue.extend_for_test([item])

    service_module._ensure_chat_input_worker_started()
    await service_module._chat_input_queue.notify_for_test()

    with pytest.raises(RuntimeError):
        await asyncio.wait_for(item.future, timeout=1.0)
    graph.ainvoke.assert_not_awaited()
    await _reset_queue_state()


@pytest.mark.asyncio
async def test_drop_queued_item_fails_when_user_save_not_committed(
    monkeypatch,
) -> None:
    """Dropped input should not complete successfully without history."""

    await _reset_queue_state()
    monkeypatch.setattr(
        service_module,
        "save_conversation",
        AsyncMock(return_value=None),
    )
    _patch_common_dependencies(monkeypatch, AsyncMock())

    item = _item(1)
    await service_module._drop_queued_chat_item(item)

    with pytest.raises(RuntimeError):
        await item.future
    await _reset_queue_state()


@pytest.mark.asyncio
async def test_worker_drops_listen_only_without_pruning_active_plain(monkeypatch) -> None:
    """Listen-only rows should persist while active plain messages keep their turn."""

    await _reset_queue_state()
    call_order = []
    graph_started = asyncio.Event()
    graph_can_finish = asyncio.Event()

    class _Graph:
        """Record graph entry for active survivors."""

        async def ainvoke(self, state):
            call_order.append(f"graph:{state['platform_message_id']}")
            graph_started.set()
            await graph_can_finish.wait()
            return {
                "should_respond": False,
                "use_reply_feature": False,
                "final_dialog": [],
                "future_promises": [],
                "consolidation_state": None,
            }

    async def _save_conversation(doc):
        call_order.append(f"save:{doc['platform_message_id']}")
        return_value = f"row-{doc['platform_message_id']}"
        return return_value

    monkeypatch.setattr(service_module, "save_conversation", _save_conversation)
    _patch_common_dependencies(monkeypatch, _Graph())

    plain_group = _item(1, channel_type="private")
    listen_only = _item(2, listen_only=True)
    tagged_group = _item(3, channel_type="private", direct_address=True)
    service_module._chat_input_queue.extend_for_test([
        plain_group,
        listen_only,
        tagged_group,
    ])

    service_module._ensure_chat_input_worker_started()
    await service_module._chat_input_queue.notify_for_test()

    listen_response = await asyncio.wait_for(listen_only.future, timeout=1.0)
    await asyncio.wait_for(graph_started.wait(), timeout=1.0)

    assert listen_response.messages == []
    assert call_order[:2] == ["save:2", "save:1"]
    assert "graph:1" in call_order[2:]
    assert not tagged_group.future.done()

    graph_can_finish.set()
    plain_response = await asyncio.wait_for(plain_group.future, timeout=1.0)
    assert plain_response.messages == []

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
        return_value = f"row-{doc['platform_message_id']}"
        return return_value

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
    assert captured_state["active_turn_platform_message_ids"] == ["1", "2"]
    assert captured_state["active_turn_conversation_row_ids"] == [
        "row-1",
        "row-2",
    ]
    assert first.conversation_row_id == "row-1"
    assert second.conversation_row_id == "row-2"
    assert captured_state["should_respond"] is True
    assert captured_state["use_reply_feature"] is False

    await _reset_queue_state()


@pytest.mark.asyncio
async def test_private_frontline_sees_complete_coalesced_logical_input(
    monkeypatch,
) -> None:
    """Private relevance sees follow-up text before routing the survivor."""

    await _reset_queue_state()
    captured_frontline_state: dict = {}

    class _Graph:
        """Return a silent graph result after private settlement."""

        async def ainvoke(self, _state):
            return {
                "should_respond": False,
                "use_reply_feature": False,
                "final_dialog": [],
                "future_promises": [],
                "consolidation_state": None,
            }

    async def _save_conversation(doc):
        return f"row-{doc['platform_message_id']}"

    async def _frontline(state):
        captured_frontline_state.update(state)
        return {
            "intake_action": "start",
            "append_target": "none",
            "prelude_targets": [],
            "reason": "complete private request",
        }

    monkeypatch.setattr(service_module, "save_conversation", _save_conversation)
    _patch_common_dependencies(monkeypatch, _Graph())
    monkeypatch.setattr(
        service_module,
        "_ensure_character_global_identity",
        AsyncMock(return_value=CHARACTER_GLOBAL_USER_ID),
    )
    monkeypatch.setattr(
        service_module,
        "frontline_relevance_agent",
        _frontline,
    )
    first = _item(
        1,
        channel_type="private",
        platform_channel_id="dm-1",
        platform_user_id="user-1",
        content="I need",
        direct_address=True,
    )
    second = _item(
        2,
        channel_type="private",
        platform_channel_id="dm-1",
        platform_user_id="user-1",
        content="help with this setting",
        direct_address=True,
    )
    service_module._chat_input_queue.extend_for_test([first, second])

    service_module._ensure_chat_input_worker_started()
    await service_module._chat_input_queue.notify_for_test()
    await asyncio.wait_for(second.future, timeout=1.0)
    await asyncio.wait_for(first.future, timeout=1.0)

    current_message = captured_frontline_state["current_message"]
    assert current_message["body_text"] == (
        "I need\nhelp with this setting"
    )
    assert current_message["semantic_target_labels"] == ["character"]
    assert captured_frontline_state["conversation_scope"] == "private"
    assert captured_frontline_state["active_character_name"] == "Kazusa"

    await _reset_queue_state()


@pytest.mark.asyncio
async def test_worker_aborts_survivor_when_collapsed_save_fails(
    monkeypatch,
) -> None:
    """Survivor graph must not run on collapsed text without history rows."""

    await _reset_queue_state()

    class _Graph:
        """Expose a mock graph call for collapsed fail-closed assertions."""

        def __init__(self):
            self.ainvoke = AsyncMock(return_value={})

    graph = _Graph()
    monkeypatch.setattr(
        service_module,
        "save_conversation",
        AsyncMock(return_value=None),
    )
    _patch_common_dependencies(monkeypatch, graph)

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

    with pytest.raises(RuntimeError):
        await asyncio.wait_for(second.future, timeout=1.0)
    with pytest.raises(RuntimeError):
        await asyncio.wait_for(first.future, timeout=1.0)
    graph.ainvoke.assert_not_awaited()

    await _reset_queue_state()


@pytest.mark.asyncio
async def test_worker_derives_graph_input_from_message_envelope(monkeypatch) -> None:
    """Service intake should use envelope body text, not raw wire content."""

    await _reset_queue_state()
    captured_state = {}
    saved_docs = []

    class _Graph:
        """Capture graph state for the envelope turn."""

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
        return_value = f"row-{doc['platform_message_id']}"
        return return_value

    monkeypatch.setattr(service_module, "save_conversation", _save_conversation)
    _patch_common_dependencies(monkeypatch, _Graph())

    item = _item(
        1,
        channel_type="private",
        platform_user_id="user-1",
        content="<@bot-1> clean body",
        message_envelope={
            "body_text": "clean body",
            "raw_wire_text": "<@bot-1> clean body",
            "mentions": [{
                "platform_user_id": "bot-1",
                "entity_kind": "bot",
                "raw_text": "<@bot-1>",
            }],
            "attachments": [],
            "addressed_to_global_user_ids": ["character-global-id"],
            "broadcast": False,
        },
    )
    service_module._chat_input_queue.extend_for_test([item])

    service_module._ensure_chat_input_worker_started()
    await service_module._chat_input_queue.notify_for_test()

    response = await asyncio.wait_for(item.future, timeout=1.0)

    assert response.messages == []
    assert captured_state["user_input"] == "clean body"
    assert captured_state["character_profile"]["name"] == "Kazusa"
    assert (
        captured_state["character_profile"]["personality_brief"]
        == "static brief"
    )
    assert captured_state["character_profile"]["mood"] == "fresh mood"
    assert captured_state["character_profile"]["global_vibe"] == "fresh vibe"
    assert (
        captured_state["character_profile"]["reflection_summary"]
        == "fresh reflection"
    )
    assert captured_state["character_profile"]["global_user_id"] == (
        "character-global-id"
    )
    assert captured_state["message_envelope"]["raw_wire_text"] == "<@bot-1> clean body"
    assert captured_state["response_action"] == "proceed"
    assert captured_state["reason_to_respond"] == "fixture response"
    assert captured_state["cognition_claimed"] is True
    assert captured_state["llm_trace_id"]
    assert all(key != "mentioned" + "_bot" for key in captured_state)
    assert saved_docs[0]["body_text"] == "clean body"
    assert "content" not in saved_docs[0]

    await _reset_queue_state()


@pytest.mark.asyncio
async def test_worker_skips_graph_for_empty_no_content_turn(monkeypatch) -> None:
    """Empty turns without prompt-usable media should persist and stop."""

    await _reset_queue_state()
    saved_docs = []

    class _Graph:
        """Expose a mock graph call for no-content assertions."""

        def __init__(self):
            self.ainvoke = AsyncMock(return_value={
                "should_respond": False,
                "use_reply_feature": False,
                "final_dialog": [],
                "future_promises": [],
                "consolidation_state": None,
            })

    async def _save_conversation(doc):
        saved_docs.append(doc)
        return_value = f"row-{doc['platform_message_id']}"
        return return_value

    graph = _Graph()
    monkeypatch.setattr(service_module, "save_conversation", _save_conversation)
    _patch_common_dependencies(monkeypatch, graph)

    item = _item(
        1,
        channel_type="private",
        platform_user_id="user-1",
        content="",
    )
    service_module._chat_input_queue.extend_for_test([item])

    service_module._ensure_chat_input_worker_started()
    await service_module._chat_input_queue.notify_for_test()

    response = await asyncio.wait_for(item.future, timeout=1.0)

    assert response.messages == []
    assert saved_docs[0]["body_text"] == ""
    graph.ainvoke.assert_not_awaited()
    pipeline_event = service_module.event_logging.record_pipeline_turn_event
    pipeline_event.assert_awaited_once()
    event_kwargs = pipeline_event.await_args.kwargs
    assert event_kwargs["status"] == "completed"
    assert event_kwargs["final_outcome"] == "no_content"
    assert event_kwargs["severity"] == "info"

    await _reset_queue_state()


@pytest.mark.asyncio
async def test_worker_keeps_image_only_turn_on_graph_path(monkeypatch) -> None:
    """Image-only input should not be suppressed by the empty guard."""

    await _reset_queue_state()
    captured_state = {}
    saved_docs = []

    class _Graph:
        """Capture graph state for an image-only turn."""

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
        return_value = f"row-{doc['platform_message_id']}"
        return return_value

    monkeypatch.setattr(service_module, "save_conversation", _save_conversation)
    _patch_common_dependencies(monkeypatch, _Graph())

    item = _item(
        1,
        channel_type="private",
        platform_user_id="user-1",
        content="",
        content_type="image",
        attachments=[{
            "media_type": "image/jpeg",
            "base64_data": "image-bytes",
            "description": "image description",
        }],
    )
    service_module._chat_input_queue.extend_for_test([item])

    service_module._ensure_chat_input_worker_started()
    await service_module._chat_input_queue.notify_for_test()

    response = await asyncio.wait_for(item.future, timeout=1.0)

    assert response.messages == []
    assert captured_state["user_input"] == ""
    assert captured_state["user_multimedia_input"] == [{
        "content_type": "image/jpeg",
        "base64_data": "image-bytes",
        "description": "image description",
    }]
    assert saved_docs[0]["body_text"] == ""

    await _reset_queue_state()


@pytest.mark.asyncio
async def test_worker_keeps_collapsed_non_empty_content_on_graph_path(
    monkeypatch,
) -> None:
    """Collapsed content should remain usable even if the survivor is empty."""

    await _reset_queue_state()
    captured_state = {}

    class _Graph:
        """Capture graph state for collapsed non-empty content."""

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
        return_value = f"row-{doc['platform_message_id']}"
        return return_value

    monkeypatch.setattr(service_module, "save_conversation", _save_conversation)
    _patch_common_dependencies(monkeypatch, _Graph())

    first = _item(
        1,
        channel_type="private",
        platform_channel_id="dm-1",
        platform_user_id="user-1",
        content="",
    )
    second = _item(
        2,
        channel_type="private",
        platform_channel_id="dm-1",
        platform_user_id="user-1",
        content="follow-up",
    )
    service_module._chat_input_queue.extend_for_test([first, second])

    service_module._ensure_chat_input_worker_started()
    await service_module._chat_input_queue.notify_for_test()

    collapsed_response = await asyncio.wait_for(second.future, timeout=1.0)
    survivor_response = await asyncio.wait_for(first.future, timeout=1.0)

    assert collapsed_response.messages == []
    assert survivor_response.messages == []
    assert captured_state["user_input"] == "follow-up"
    assert captured_state["active_turn_platform_message_ids"] == ["1", "2"]

    await _reset_queue_state()


@pytest.mark.asyncio
async def test_worker_resolves_cross_user_envelope_targets(monkeypatch) -> None:
    """Service intake should resolve user mentions and reply targets by profile."""

    await _reset_queue_state()
    captured_state = {}
    saved_docs = []

    class _Graph:
        """Capture graph state after service identity resolution."""

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
        return_value = f"row-{doc['platform_message_id']}"
        return return_value

    async def _resolve_global_user_id(
        *,
        platform: str,
        platform_user_id: str,
        display_name: str = "",
    ) -> str:
        identities = {
            "user-a": "global-user-a",
            "user-b": "global-user-b",
            "user-c": "global-user-c",
        }
        resolved_id = identities[platform_user_id]
        return resolved_id

    monkeypatch.setattr(service_module, "save_conversation", _save_conversation)
    _patch_common_dependencies(monkeypatch, _Graph())
    monkeypatch.setattr(
        service_module,
        "resolve_global_user_id",
        _resolve_global_user_id,
    )

    item = _item(
        1,
        channel_type="private",
        platform_user_id="user-a",
        content="@user-b see this",
        message_envelope={
            "body_text": "see this",
            "raw_wire_text": "@user-b see this",
            "mentions": [{
                "platform_user_id": "user-b",
                "entity_kind": "user",
                "raw_text": "@user-b",
            }],
            "reply": {
                "platform_message_id": "reply-1",
                "platform_user_id": "user-c",
                "display_name": "",
                "excerpt": "third-party message",
                "derivation": "platform_native",
            },
            "attachments": [],
            "addressed_to_global_user_ids": [],
            "broadcast": True,
        },
    )
    service_module._chat_input_queue.extend_for_test([item])

    service_module._ensure_chat_input_worker_started()
    await service_module._chat_input_queue.notify_for_test()

    response = await asyncio.wait_for(item.future, timeout=1.0)

    assert response.messages == []
    state_envelope = captured_state["message_envelope"]
    assert state_envelope["mentions"][0]["global_user_id"] == "global-user-b"
    assert state_envelope["reply"]["global_user_id"] == "global-user-c"
    assert state_envelope["addressed_to_global_user_ids"] == [
        CHARACTER_GLOBAL_USER_ID,
        "global-user-b",
        "global-user-c",
    ]
    assert state_envelope["broadcast"] is False
    assert saved_docs[0]["addressed_to_global_user_ids"] == [
        CHARACTER_GLOBAL_USER_ID,
        "global-user-b",
        "global-user-c",
    ]
    assert saved_docs[0]["broadcast"] is False

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
        return_value = f"row-{doc['platform_message_id']}"
        return return_value

    monkeypatch.setattr(service_module, "save_conversation", _save_conversation)
    _patch_common_dependencies(monkeypatch, _Graph())

    first = _item(
        1,
        channel_type="private",
        platform_user_id="user-1",
        content="Kazusa, look",
        content_type="mixed",
        direct_address=True,
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
        channel_type="private",
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
    assert saved_by_message_id["2"]["body_text"] == ""
    assert saved_by_message_id["2"]["content_type"] == "image"
    assert saved_by_message_id["2"]["attachments"] == [{
        "media_type": "image/jpeg",
        "url": "https://example.test/second.jpg",
        "base64_data": "second-image-bytes",
        "description": "second image",
        "size_bytes": 222,
    }]
    assert saved_by_message_id["1"]["attachments"][0]["base64_data"] == (
        "first-image-bytes"
    )

    await _reset_queue_state()
