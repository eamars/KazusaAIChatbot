"""Tests for the brain service global input queue."""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import BackgroundTasks, Request
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
from tests.cognition_test_helpers import (
    canonical_episode_identity_snapshot,
    canonical_service_character_profile,
)


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


def _chat_http_request() -> Request:
    """Build the HTTP request context required by the chat authorization seam."""

    return Request({"type": "http", "headers": []})


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
        "_active_character_name_snapshot",
        "Test Character",
    )
    character_profile = canonical_service_character_profile(
        marker="input-queue",
        global_user_id=CHARACTER_GLOBAL_USER_ID,
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
        marker="input-queue",
        global_user_id=CHARACTER_GLOBAL_USER_ID,
    )
    character_profile["name"] = "Test Character"
    identity_snapshot["character_profile"]["name"] = "Test Character"
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
        AsyncMock(return_value=CHARACTER_GLOBAL_USER_ID),
    )
    monkeypatch.setattr(
        service_module,
        "resolve_global_user_id",
        AsyncMock(return_value="global-user-1"),
    )
    monkeypatch.setattr(
        service_module,
        "get_user_profile",
        AsyncMock(return_value={"relationship_state": 500}),
    )
    monkeypatch.setattr(
        service_module,
        "get_conversation_history",
        AsyncMock(return_value=[]),
    )
    monkeypatch.setattr(
        service_module,
        "has_inbound_after",
        AsyncMock(return_value=False),
    )
    monkeypatch.setattr(
        service_module,
        "get_user_message_by_row_id",
        AsyncMock(return_value=None),
    )
    monkeypatch.setattr(
        service_module,
        "update_conversation_row_llm_trace_id",
        AsyncMock(return_value=True),
    )
    monkeypatch.setattr(
        service_module,
        "storage_utc_now_iso",
        lambda: "2026-04-29T00:20:00+00:00",
    )
    monkeypatch.setattr(
        service_module,
        "get_conversation_by_platform_message_id",
        AsyncMock(return_value=None),
    )
    monkeypatch.setattr(
        service_module,
        "set_conversation_source_episode_id",
        AsyncMock(side_effect=lambda **kwargs: len(kwargs["row_ids"])),
    )
    monkeypatch.setattr(
        service_module,
        "build_promoted_reflection_context",
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
    monkeypatch.setattr(
        service_module,
        "_run_post_turn_memory_lifecycle_background",
        AsyncMock(side_effect=lambda state: state),
    )
    monkeypatch.setattr(
        service_module,
        "upsert_post_turn_lifecycle_record",
        AsyncMock(),
    )
    async def _frontline(_state):
        """Admit deterministic service fixtures through the new contract."""

        open_turns = _state.get("open_turns") or []
        if open_turns:
            return {
                "decision": {
                    "intake_action": "append",
                    "append_target": "open_1",
                    "prelude_targets": [],
                    "reason": "fixture continuation",
                },
                "attempt_diagnostics": [],
            }
        return {
            "decision": {
                "intake_action": "start",
                "append_target": "none",
                "prelude_targets": [],
                "reason": "fixture candidate",
            },
            "attempt_diagnostics": [],
        }

    async def _settled(_state):
        """Allow deterministic service fixtures to reach their fake graph."""

        return {
            "decision": {
                "response_action": "proceed",
                "reason_to_respond": "fixture response",
                "use_reply_feature": False,
                "channel_topic": "",
                "indirect_speech_context": "",
            },
            "attempt_diagnostics": [],
        }

    async def _settled_evaluator(_lease, state):
        """Adapt the module evaluator fixture to the coordinator call shape."""

        return await service_module.relevance_agent(state)

    async def _frontline_evaluator(state):
        """Delegate through the replaceable module frontline fixture."""

        return await service_module.frontline_relevance_agent(state)

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
    monkeypatch.setattr(
        service_module._turn_settlement_coordinator,
        "_frontline_evaluator",
        _frontline_evaluator,
    )
    monkeypatch.setattr(
        service_module._turn_settlement_coordinator,
        "_settled_evaluator",
        _settled_evaluator,
    )
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
    await asyncio.sleep(0)
    assert coordinator.calls == 1
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
async def test_active_turn_source_refs_preserve_each_row_timestamp() -> None:
    """Collapsed input lineage keeps each persisted row's own UTC time."""

    survivor = _item(
        1,
        storage_timestamp_utc='2026-04-29T00:00:01+00:00',
    )
    collapsed = _item(
        2,
        storage_timestamp_utc='2026-04-29T00:00:09+00:00',
    )
    survivor.conversation_row_id = 'row-1'
    collapsed.conversation_row_id = 'row-2'
    survivor.collapsed_items = [collapsed]

    source_refs = brain_intake.active_turn_conversation_source_refs(
        survivor
    )

    assert source_refs == [
        {
            'ref_kind': 'conversation_row',
            'ref_id': 'row-1',
            'occurred_at': '2026-04-29T00:00:01+00:00',
        },
        {
            'ref_kind': 'conversation_row',
            'ref_id': 'row-2',
            'occurred_at': '2026-04-29T00:00:09+00:00',
        },
    ]


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
async def test_settled_fresh_history_excludes_active_turn_fragments(
    monkeypatch,
) -> None:
    """Only external rows can act as fresh-history scene evidence."""

    character_profile = canonical_service_character_profile(
        marker="settled-relevance",
        global_user_id=CHARACTER_GLOBAL_USER_ID,
    )
    monkeypatch.setattr(
        service_module,
        "_active_character_name_snapshot",
        "Test Character",
    )
    monkeypatch.setattr(
        service_module,
        "_runtime_character_state",
        {
            "cognition_state": character_profile["cognition_state"],
            "updated_at": character_profile["updated_at"],
        },
    )
    item = _item(
        1,
        storage_timestamp_utc="2026-07-16T00:00:05+00:00",
    )
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
    }
    state = service_module._settled_state_from_lease(
        lease,
        history=[
            *[
                {
                    **common,
                    "_id": f"row-older-{index}",
                    "platform_message_id": f"message-older-{index}",
                    "body_text": f"older context {index}",
                    "timestamp": (
                        f"2026-07-15T23:59:{50 + index:02d}+00:00"
                    ),
                }
                for index in range(1, 10)
            ],
            {
                **common,
                "_id": "row-active",
                "platform_message_id": "message-active",
                "body_text": "active request",
                "timestamp": "2026-07-16T00:00:05+00:00",
            },
            {
                **common,
                "_id": "row-other",
                "platform_message_id": "message-other",
                "body_text": "another participant answered",
                "timestamp": "2026-07-16T00:00:06+00:00",
            },
            {
                **common,
                "_id": "row-active-2",
                "platform_message_id": "message-active-2",
                "body_text": "active follow-up",
                "timestamp": "2026-07-16T00:00:07+00:00",
            },
            {
                **common,
                "_id": "row-character-1",
                "role": "assistant",
                "platform_user_id": "bot-1",
                "global_user_id": CHARACTER_GLOBAL_USER_ID,
                "display_name": "Test Character",
                "platform_message_id": "message-character-1",
                "body_text": "first character fragment",
                "addressed_to_global_user_ids": ["global-user-1"],
                "llm_trace_id": "trace-character-1",
                "logical_message_index": 0,
                "timestamp": "2026-07-16T00:00:08+00:00",
            },
            {
                **common,
                "_id": "row-character-2",
                "role": "assistant",
                "platform_user_id": "bot-1",
                "global_user_id": CHARACTER_GLOBAL_USER_ID,
                "display_name": "Test Character",
                "platform_message_id": "message-character-2",
                "body_text": "second character fragment",
                "addressed_to_global_user_ids": ["global-user-1"],
                "llm_trace_id": "trace-character-1",
                "logical_message_index": 1,
                "timestamp": "2026-07-16T00:00:09+00:00",
            },
            {
                **common,
                "_id": "row-character-3",
                "role": "assistant",
                "platform_user_id": "bot-1",
                "global_user_id": CHARACTER_GLOBAL_USER_ID,
                "display_name": "Test Character",
                "platform_message_id": "message-character-3",
                "body_text": "independent character response",
                "addressed_to_global_user_ids": ["global-user-1"],
                "llm_trace_id": "trace-character-2",
                "logical_message_index": 0,
                "timestamp": "2026-07-16T00:00:10+00:00",
            },
        ],
    )

    assert len(state["fresh_history"]) == 10
    assert state["fresh_history"][0]["body_text"] == "older context 3"
    assert [row["body_text"] for row in state["fresh_history"][-3:]] == [
        "another participant answered",
        "first character fragment\nsecond character fragment",
        "independent character response",
    ]
    assert state["fresh_history"][-3]["turn_temporal_relation"] == (
        "during_active_turn"
    )
    assert state["fresh_history"][-2]["turn_temporal_relation"] == (
        "after_active_turn"
    )
    assert state["fresh_history"][-1]["turn_temporal_relation"] == (
        "after_active_turn"
    )
    assert state["relationship_context"] == "direct participant"
    assert state["group_attention"] == "chaotic_noise"
    assert state["conversation_scope"] == "group"
    assert state["active_character_name"] == "Test Character"
    assert state["current_author_global_user_id"] == "global-user-1"
    assert state["current_author_platform_user_id"] == "user-1"
    assert state["character_global_user_id"] == CHARACTER_GLOBAL_USER_ID
    assert state["platform_bot_id"] == "bot-1"
    assert state["character_cognition_state"] == (
        character_profile["cognition_state"]
    )


@pytest.mark.asyncio
async def test_settled_history_uses_timestamps_when_active_row_is_outside_window(
    monkeypatch,
) -> None:
    """Intervening rows stay ordered when a busy group evicts the anchor."""

    character_profile = canonical_service_character_profile(
        marker="settled-clipped-history",
        global_user_id=CHARACTER_GLOBAL_USER_ID,
    )
    monkeypatch.setattr(
        service_module,
        "_active_character_name_snapshot",
        "Test Character",
    )
    monkeypatch.setattr(
        service_module,
        "_runtime_character_state",
        {
            "cognition_state": character_profile["cognition_state"],
            "updated_at": character_profile["updated_at"],
        },
    )
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
                "_id": "row-before",
                "platform_message_id": "message-before",
                "body_text": "earlier context",
                "timestamp": "2026-07-16T00:00:04+00:00",
            },
            {
                **common,
                "_id": "row-during",
                "platform_message_id": "message-during",
                "body_text": "intervening answer",
                "timestamp": "2026-07-16T00:00:06+00:00",
            },
            {
                **common,
                "_id": "row-after",
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
async def test_frontline_discard_keeps_precommitted_receipt_without_duplicate(
    monkeypatch,
) -> None:
    """Frontline discard consumes the committed row without a second insert."""

    await _reset_queue_state()
    save_conversation = AsyncMock()
    monkeypatch.setattr(service_module, "save_conversation", save_conversation)

    async def _discard(_state):
        """Discard every deterministic fixture at the frontline boundary."""

        return {
            "decision": {
                "intake_action": "discard",
                "append_target": "none",
                "prelude_targets": [],
                "reason": "fixture discard",
            },
            "attempt_diagnostics": [],
        }

    _patch_common_dependencies(monkeypatch, AsyncMock())
    monkeypatch.setattr(
        service_module,
        "frontline_relevance_agent",
        _discard,
    )
    item = _item(1, direct_address=True)
    item.received_at = "2026-04-29T00:19:00+00:00"
    item.conversation_row_id = "row-1"

    await service_module._frontline_intake_item(item)

    response = await item.future
    assert response.messages == []
    save_conversation.assert_not_awaited()
    assert item.conversation_row_id == "row-1"
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
        AsyncMock(return_value=("global-user-1", {"relationship_state": 500})),
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




















