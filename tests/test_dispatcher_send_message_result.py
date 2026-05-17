"""Tests for dispatcher send-message delivery metadata."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from kazusa_ai_chatbot.dispatcher import AdapterRegistry, SendResult
from kazusa_ai_chatbot.dispatcher.handlers import handle_send_message
import kazusa_ai_chatbot.dispatcher.handlers as handlers_module
import kazusa_ai_chatbot.dispatcher.tool_spec as tool_spec_module
from kazusa_ai_chatbot.dispatcher.task import DispatchContext


class _FakeAdapter:
    """Messaging adapter double that records one outbound send."""

    platform = "qq"
    platform_bot_id = "bot-1"
    display_name = "Character"

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def send_message(
        self,
        channel_id: str,
        text: str,
        *,
        channel_type: str,
        reply_to_msg_id: str | None = None,
        delivery_mentions: list[dict[str, Any]] | None = None,
    ) -> SendResult:
        """Capture the send request and return deterministic metadata."""

        self.calls.append({
            "channel_id": channel_id,
            "text": text,
            "channel_type": channel_type,
            "reply_to_msg_id": reply_to_msg_id,
            "delivery_mentions": delivery_mentions,
        })
        sent_at = datetime(2026, 5, 17, 5, 57, tzinfo=timezone.utc)
        result = SendResult(
            platform="qq",
            channel_id=channel_id,
            message_id="adapter-message-1",
            sent_at=sent_at,
        )
        return result


def _dispatch_context() -> DispatchContext:
    """Build a source context for autonomous self-cognition delivery."""

    context = DispatchContext(
        source_platform="qq",
        source_channel_id="group-1",
        source_user_id="global-target",
        source_message_id="self_cognition:case-1",
        guild_id="guild-1",
        bot_permission_role="user",
        now=datetime(2026, 5, 17, 5, 57, tzinfo=timezone.utc),
        source_channel_type="group",
        source_platform_bot_id="bot-1",
        source_character_name="Character",
    )
    return context


@pytest.mark.asyncio
async def test_handle_send_message_returns_delivery_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dispatcher should return the ids needed by self-cognition audit."""

    saved_docs: list[dict[str, Any]] = []
    receipts: list[dict[str, Any]] = []

    async def save_conversation(document: dict[str, Any]) -> str:
        saved_docs.append(dict(document))
        return "conversation-row-1"

    async def ensure_character_identity(**kwargs: Any) -> str:
        assert kwargs["platform"] == "qq"
        assert kwargs["platform_user_id"] == "bot-1"
        return "character-global"

    async def apply_receipt(**kwargs: Any) -> None:
        receipts.append(dict(kwargs))

    monkeypatch.setattr(
        handlers_module,
        "save_conversation",
        save_conversation,
    )
    monkeypatch.setattr(
        handlers_module,
        "ensure_character_identity",
        ensure_character_identity,
    )
    monkeypatch.setattr(
        handlers_module,
        "apply_assistant_delivery_receipt",
        apply_receipt,
    )
    monkeypatch.setattr(
        handlers_module.event_logging,
        "record_dispatcher_event",
        AsyncMock(),
    )
    monkeypatch.setattr(
        handlers_module.event_logging,
        "record_runtime_error_event",
        AsyncMock(),
    )

    adapter = _FakeAdapter()
    registry = AdapterRegistry()
    registry.register(adapter)
    args = {
        "target_platform": "qq",
        "target_channel": "dm-1",
        "target_channel_type": "private",
        "text": "Checking in now.",
        "execute_at": "2026-05-17T05:57:00+00:00",
        "reply_to_msg_id": None,
        "delivery_mentions": [],
    }

    result = await handle_send_message(args, _dispatch_context(), registry)

    assert result["conversation_message_id"] == "conversation-row-1"
    assert result["adapter_message_id"] == "adapter-message-1"
    assert result["delivery_tracking_id"]
    assert saved_docs[0]["delivery_tracking_id"] == (
        result["delivery_tracking_id"]
    )
    assert adapter.calls == [
        {
            "channel_id": "dm-1",
            "text": "Checking in now.",
            "channel_type": "private",
            "reply_to_msg_id": None,
            "delivery_mentions": [],
        }
    ]
    assert receipts[0]["delivery_tracking_id"] == result["delivery_tracking_id"]
    assert receipts[0]["platform_message_id"] == "adapter-message-1"


def test_task_handler_type_accepts_ignored_return_value() -> None:
    """Dispatcher handler type should allow delivery metadata returns."""

    source_text = Path(tool_spec_module.__file__).read_text(encoding="utf-8")

    assert "Awaitable[object | None]" in source_text
