"""Delivery bridge for selected self-cognition speech."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Any, Literal, TypedDict

from kazusa_ai_chatbot.brain_service.outbound import (
    ConversationHistoryWriteError,
)
from kazusa_ai_chatbot.dispatcher.adapter_iface import (
    AdapterChannelUnavailableError,
    AdapterRegistry,
    UnknownPlatformError,
)
from kazusa_ai_chatbot.dispatcher.handlers import handle_send_message
from kazusa_ai_chatbot.dispatcher.task import DispatchContext
from kazusa_ai_chatbot.self_cognition import models


class SelfCognitionDeliveryResult(TypedDict):
    """Terminal delivery result for one selected self-cognition speak."""

    status: Literal[
        "sent",
        "delivery_failed",
        "held",
        "duplicate_suppressed",
    ]
    conversation_message_id: str | None
    delivery_tracking_id: str | None
    adapter_message_id: str | None
    failure_reason: str | None


async def deliver_selected_speak(
    *,
    text: str,
    delivery_target: models.SelfCognitionDeliveryTarget,
    character_profile: Mapping[str, Any],
    adapter_registry: AdapterRegistry | None,
    now: datetime,
    reply_to_msg_id: str | None = None,
    delivery_mentions: list[dict[str, Any]] | None = None,
) -> SelfCognitionDeliveryResult:
    """Persist and dispatch selected self-cognition speech.

    Args:
        text: Final dialog text rendered by the shared dialog graph.
        delivery_target: Deterministic target metadata bound before cognition.
        character_profile: Runtime character profile used as name fallback.
        adapter_registry: Process-local runtime adapter registry.
        now: Worker tick time used for deterministic dispatch context.
        reply_to_msg_id: Optional platform reply target.
        delivery_mentions: Optional adapter-owned mention metadata.

    Returns:
        Terminal delivery result for worker persistence and event logging.
    """

    clean_text = text.strip()
    if not clean_text:
        result = _failed_result("empty_text")
        return result
    if adapter_registry is None:
        result = _failed_result("adapter_registry_unavailable")
        return result

    ctx = DispatchContext(
        source_platform=delivery_target["platform"],
        source_channel_id=(
            delivery_target["source_platform_channel_id"]
            or delivery_target["platform_channel_id"]
        ),
        source_user_id=(
            delivery_target["target_global_user_id"]
            or delivery_target["source_global_user_id"]
            or ""
        ),
        source_message_id=delivery_target["source_message_id"],
        guild_id=delivery_target["guild_id"],
        bot_permission_role=delivery_target["bot_permission_role"] or "user",
        now=now,
        source_channel_type=(
            delivery_target["source_channel_type"]
            or delivery_target["channel_type"]
        ),
        source_platform_bot_id=delivery_target["source_platform_bot_id"],
        source_character_name=(
            delivery_target["source_character_name"]
            or str(character_profile.get("name") or "active character")
        ),
    )
    args = {
        "target_platform": delivery_target["platform"],
        "target_channel": delivery_target["platform_channel_id"],
        "target_channel_type": delivery_target["channel_type"],
        "text": clean_text,
        "execute_at": now.isoformat(),
        "reply_to_msg_id": reply_to_msg_id,
        "delivery_mentions": delivery_mentions or [],
    }
    try:
        dispatch_result = await handle_send_message(args, ctx, adapter_registry)
    except AdapterChannelUnavailableError:
        result = _failed_result("adapter_channel_unavailable")
        return result
    except UnknownPlatformError:
        result = _failed_result("adapter_unavailable")
        return result
    except ConversationHistoryWriteError:
        result = _failed_result("conversation_history_write_failed")
        return result
    except Exception as exc:
        result = _failed_result(
            f"adapter_send_failed:{exc.__class__.__name__}"
        )
        return result

    if not isinstance(dispatch_result, Mapping):
        result = _failed_result("send_message_missing_delivery_metadata")
        return result

    conversation_message_id = _non_empty_text(
        dispatch_result.get("conversation_message_id")
    )
    delivery_tracking_id = _non_empty_text(
        dispatch_result.get("delivery_tracking_id")
    )
    adapter_message_id = _non_empty_text(
        dispatch_result.get("adapter_message_id")
    )
    if not conversation_message_id or not delivery_tracking_id:
        result = _failed_result("send_message_missing_delivery_metadata")
        return result

    result: SelfCognitionDeliveryResult = {
        "status": "sent",
        "conversation_message_id": conversation_message_id,
        "delivery_tracking_id": delivery_tracking_id,
        "adapter_message_id": adapter_message_id,
        "failure_reason": None,
    }
    return result


def _failed_result(failure_reason: str) -> SelfCognitionDeliveryResult:
    """Build a terminal delivery failure result."""

    result: SelfCognitionDeliveryResult = {
        "status": "delivery_failed",
        "conversation_message_id": None,
        "delivery_tracking_id": None,
        "adapter_message_id": None,
        "failure_reason": failure_reason,
    }
    return result


def _non_empty_text(value: object) -> str | None:
    """Return stripped text only when metadata is present."""

    if not isinstance(value, str):
        return_value = None
        return return_value
    clean_value = value.strip()
    if clean_value:
        return_value = clean_value
    else:
        return_value = None
    return return_value
