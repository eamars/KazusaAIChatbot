"""Delivery mention contract tests."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, get_type_hints
from unittest.mock import AsyncMock

import pytest

from kazusa_ai_chatbot.dispatcher import AdapterRegistry, SendResult
import kazusa_ai_chatbot.dispatcher.handlers as handlers_module
from kazusa_ai_chatbot.self_cognition import models, tracking


def _case_with_scope(target_scope: dict[str, Any]) -> dict[str, Any]:
    case = {
        "case_name": models.CASE_COMMITMENT_PAST_DUE,
        "case_id": "commitment:promise-001",
        "idle_timestamp": "2026-05-10T00:30:00+00:00",
        "last_evidence_timestamp": "2026-05-10T00:00:00+00:00",
        "trigger_kind": models.TRIGGER_ACTIVE_COMMITMENT_DUE_CHECK,
        "semantic_due_state": models.DUE_STATE_PAST_DUE,
        "actionability": "contact_is_socially_available",
        "target_scope": target_scope,
        "source_refs": [
            {
                "source_kind": "future_promise",
                "source_id": "promise-001",
                "due_at": "2026-05-10T00:00:00+00:00",
                "summary": "The user expected a follow-up.",
            }
        ],
        "visible_context": [],
    }
    return case


def _candidate_for_scope(
    target_scope: dict[str, Any],
    *,
    mention_target_user: bool = False,
) -> dict[str, Any] | None:
    case = _case_with_scope(target_scope)
    trigger_record = tracking.build_trigger_record(case)
    action_attempt = tracking.build_action_attempt(
        case,
        trigger_record,
        existing_attempts=[],
    )
    action_candidate = tracking.build_action_candidate(
        case,
        action_attempt,
        "Checking in now.",
        mention_target_user=mention_target_user,
    )
    return action_candidate


class _FakeAdapter:
    """Adapter double that records delivery mentions from self-cognition."""

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
        """Capture delivery mention metadata and return a send result."""

        self.calls.append({
            "channel_id": channel_id,
            "text": text,
            "channel_type": channel_type,
            "reply_to_msg_id": reply_to_msg_id,
            "delivery_mentions": delivery_mentions,
        })
        result = SendResult(
            platform="qq",
            channel_id=channel_id,
            message_id="adapter-message-1",
            sent_at=datetime(2026, 5, 17, 5, 57, tzinfo=timezone.utc),
        )
        return result


def _delivery_target() -> dict[str, Any]:
    """Build a source-channel delivery target for mention tests."""

    target = {
        "schema_version": "self_cognition_delivery_target.v1",
        "platform": "qq",
        "platform_channel_id": "group-1",
        "channel_type": "group",
        "target_global_user_id": "global-target-1",
        "target_platform_user_id": "qq-target",
        "source_kind": "self_cognition_source_channel",
        "source_ref": "promise-001",
        "source_platform_channel_id": "group-1",
        "source_channel_type": "group",
        "source_message_id": "msg-1",
        "source_global_user_id": "global-target-1",
        "source_platform_bot_id": "bot-1",
        "source_character_name": "Character",
        "guild_id": None,
        "bot_permission_role": "user",
        "fallback_reason": "private_channel_unavailable",
    }
    return target


def test_delivery_mention_typed_shape_is_declared() -> None:
    assert get_type_hints(models.DeliveryMention) == {
        "entity_kind": str,
        "placement": str,
        "platform_user_id": str | None,
        "global_user_id": str | None,
        "display_name": str,
        "requested_by": str,
    }


def test_group_delivery_scope_omits_mentions_without_dialog_flag() -> None:
    action_candidate = _candidate_for_scope(
        {
            "platform": "qq",
            "platform_channel_id": "54369546",
            "channel_type": "group",
            "user_id": "global-target-1",
            "display_name": "Target User",
        }
    )

    assert action_candidate is not None
    assert "delivery_mentions" not in action_candidate


def test_group_delivery_mention_preserves_missing_platform_user_id() -> None:
    action_candidate = _candidate_for_scope(
        {
            "platform": "qq",
            "platform_channel_id": "54369546",
            "channel_type": "group",
            "user_id": "global-target-1",
            "display_name": "Target User",
        },
        mention_target_user=True,
    )

    assert action_candidate is not None
    assert action_candidate["delivery_mentions"] == [
        {
            "entity_kind": "user",
            "placement": "prefix",
            "platform_user_id": None,
            "global_user_id": "global-target-1",
            "display_name": "Target User",
            "requested_by": "dialog.mention_target_user",
        }
    ]


def test_private_delivery_scope_keeps_dialog_mention_request_for_adapter_noop(
) -> None:
    action_candidate = _candidate_for_scope(
        {
            "platform": "qq",
            "platform_channel_id": "673225019",
            "channel_type": "private",
            "user_id": "global-target-1",
            "platform_user_id": "qq-target",
            "display_name": "Target User",
        },
        mention_target_user=True,
    )

    assert action_candidate is not None
    assert action_candidate["delivery_mentions"] == [
        {
            "entity_kind": "user",
            "placement": "prefix",
            "platform_user_id": "qq-target",
            "global_user_id": "global-target-1",
            "display_name": "Target User",
            "requested_by": "dialog.mention_target_user",
        }
    ]


def test_group_delivery_scope_without_semantic_target_omits_mentions() -> None:
    action_candidate = _candidate_for_scope(
        {
            "platform": "qq",
            "platform_channel_id": "54369546",
            "channel_type": "group",
            "user_id": None,
            "platform_user_id": "qq-target",
            "display_name": "Target User",
        },
        mention_target_user=True,
    )

    assert action_candidate is not None
    assert "delivery_mentions" not in action_candidate


@pytest.mark.asyncio
async def test_self_cognition_delivery_preserves_mentions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Selected self-cognition delivery should pass mention metadata through."""

    from kazusa_ai_chatbot.self_cognition.delivery import deliver_selected_speak

    async def save_conversation(document: dict[str, Any]) -> str:
        assert document["body_text"] == "Checking in now."
        return "conversation-row-1"

    async def ensure_character_identity(**kwargs: Any) -> str:
        assert kwargs["platform"] == "qq"
        return "character-global"

    async def apply_receipt(**kwargs: Any) -> None:
        assert kwargs["platform_message_id"] == "adapter-message-1"

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

    adapter = _FakeAdapter()
    registry = AdapterRegistry()
    registry.register(adapter)
    mention = {
        "entity_kind": "user",
        "placement": "prefix",
        "platform_user_id": "qq-target",
        "global_user_id": "global-target-1",
        "display_name": "Target User",
        "requested_by": "dialog.mention_target_user",
    }

    result = await deliver_selected_speak(
        text="Checking in now.",
        delivery_target=_delivery_target(),
        character_profile={"name": "Character"},
        adapter_registry=registry,
        now=datetime(2026, 5, 17, 5, 57, tzinfo=timezone.utc),
        delivery_mentions=[mention],
    )

    assert result["status"] == "sent"
    assert adapter.calls[0]["delivery_mentions"] == [mention]
