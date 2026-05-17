"""Built-in task handlers for the dispatcher."""

from __future__ import annotations

import logging
from typing import TypedDict
from uuid import uuid4

from kazusa_ai_chatbot import event_logging
from kazusa_ai_chatbot.brain_service.outbound import (
    record_assistant_outbound_message,
)
from kazusa_ai_chatbot.config import CHARACTER_GLOBAL_USER_ID
from kazusa_ai_chatbot.db import (
    apply_assistant_delivery_receipt,
    ensure_character_identity,
    save_conversation,
)
from kazusa_ai_chatbot.dispatcher.adapter_iface import AdapterRegistry
from kazusa_ai_chatbot.dispatcher.task import DispatchContext
from kazusa_ai_chatbot.dispatcher.tool_spec import ToolSpec
from kazusa_ai_chatbot.time_boundary import storage_utc_now_iso

HANDLER_COMPONENT = "dispatcher.handlers"
logger = logging.getLogger(__name__)


class SendMessageDispatchResult(TypedDict):
    """Delivery metadata returned by the send-message handler."""

    conversation_message_id: str
    delivery_tracking_id: str
    adapter_message_id: str


def _handler_correlation_id(ctx: DispatchContext) -> str:
    """Build a non-content correlation id for a scheduled handler call."""

    message_ref = ctx.source_message_id or "no-message-id"
    correlation_id = f"dispatch:{ctx.source_platform}:{message_ref}"
    return correlation_id


async def _ensure_dispatcher_character_identity(
    *,
    platform: str,
    platform_bot_id: str,
    character_name: str,
) -> str:
    """Resolve the active character identity for a dispatcher-owned send."""

    if not str(platform_bot_id or "").strip():
        return_value = CHARACTER_GLOBAL_USER_ID
        return return_value

    return_value = await ensure_character_identity(
        platform=platform,
        platform_user_id=platform_bot_id,
        display_name=character_name,
        global_user_id=CHARACTER_GLOBAL_USER_ID,
    )
    return return_value


def _adapter_text_attr(adapter: object, attr_name: str) -> str:
    """Read an optional adapter string attribute."""

    value = getattr(adapter, attr_name, "")
    if not isinstance(value, str):
        return_value = ""
        return return_value
    return_value = value.strip()
    return return_value


def _platform_bot_id(ctx: DispatchContext, adapter: object) -> str:
    """Return the best available platform bot id for outbound history."""

    return_value = (
        ctx.source_platform_bot_id.strip()
        or _adapter_text_attr(adapter, "platform_bot_id")
        or _adapter_text_attr(adapter, "bot_id")
    )
    return return_value


def _character_name(ctx: DispatchContext, adapter: object) -> str:
    """Return the best available display name for outbound history."""

    return_value = (
        ctx.source_character_name.strip()
        or _adapter_text_attr(adapter, "display_name")
        or _adapter_text_attr(adapter, "bot_name")
        or "assistant"
    )
    return return_value


SEND_MESSAGE_SCHEMA = {
    "type": "object",
    "required": ["target_channel", "text"],
    "properties": {
        "target_channel": {
            "type": "string",
        },
        "text": {
            "type": "string",
        },
        "target_platform": {
            "type": "string",
        },
        "target_channel_type": {
            "type": "string",
            "enum": ["group", "private"],
        },
        "execute_at": {
            "type": "string",
        },
        "reply_to_msg_id": {
            "type": ["string", "null"],
        },
        "delivery_mentions": {
            "type": "array",
        },
    },
}


async def handle_send_message(
    args: dict,
    ctx: DispatchContext,
    adapters: AdapterRegistry,
) -> SendMessageDispatchResult:
    """Deliver a scheduled channel message through the target platform adapter.

    Args:
        args: Validated tool arguments for ``send_message``.
        ctx: Source-side dispatch context, available for future extensions.
        adapters: Adapter registry used to look up the platform transport.

    Returns:
        Conversation row id, delivery tracking id, and adapter message id.
    """

    correlation_id = _handler_correlation_id(ctx)
    adapter_available = False
    delivery_attempted = False
    try:
        target_platform = str(args["target_platform"])
        adapter_available = adapters.has(target_platform)
        adapter = adapters.get(target_platform)
        delivery_tracking_id = uuid4().hex
        target_channel = str(args["target_channel"])
        channel_type = str(args["target_channel_type"])
        platform_bot_id = _platform_bot_id(ctx, adapter)
        character_name = _character_name(ctx, adapter)
        addressed_to = [ctx.source_user_id] if ctx.source_user_id.strip() else []
        conversation_message_id = await record_assistant_outbound_message(
            platform=target_platform,
            platform_channel_id=target_channel,
            channel_type=channel_type,
            platform_bot_id=platform_bot_id,
            character_name=character_name,
            body_text=str(args["text"]),
            addressed_to_global_user_ids=addressed_to,
            broadcast=not bool(addressed_to),
            delivery_tracking_id=delivery_tracking_id,
            storage_timestamp_utc=storage_utc_now_iso(),
            ensure_character_global_identity_func=(
                _ensure_dispatcher_character_identity
            ),
            save_conversation_func=save_conversation,
        )
        delivery_attempted = True
        send_kwargs = {
            "channel_id": target_channel,
            "text": str(args["text"]),
            "channel_type": channel_type,
            "reply_to_msg_id": args.get("reply_to_msg_id"),
        }
        delivery_mentions = _delivery_mentions(args)
        if delivery_mentions is not None:
            send_kwargs["delivery_mentions"] = delivery_mentions
        send_result = await adapter.send_message(**send_kwargs)
        try:
            await apply_assistant_delivery_receipt(
                platform=target_platform,
                platform_channel_id=target_channel,
                delivery_tracking_id=delivery_tracking_id,
                platform_message_id=send_result.message_id,
                delivered_at=send_result.sent_at.isoformat(),
                adapter=send_result.platform,
            )
        except Exception as exc:
            logger.warning(
                "Dispatcher delivery receipt update failed after send: "
                f"platform={target_platform} channel={target_channel} "
                f"delivery_tracking_id={delivery_tracking_id} error={exc}"
            )
    except Exception as exc:
        if not adapter_available:
            rejection_code = "adapter_unavailable"
        elif delivery_attempted:
            rejection_code = "adapter_send_failed"
        else:
            rejection_code = "conversation_history_write_failed"
        validation_status = (
            "adapter_available"
            if adapter_available
            else "adapter_unavailable"
        )
        await event_logging.record_dispatcher_event(
            component=HANDLER_COMPONENT,
            action_kind="send_message",
            validation_status=validation_status,
            adapter_available=adapter_available,
            status="failed",
            rejection_codes=[rejection_code],
            correlation_id=correlation_id,
            severity="warning",
        )
        error_class = exc.__class__.__name__
        await event_logging.record_runtime_error_event(
            component=HANDLER_COMPONENT,
            error_class=error_class,
            error_preview=str(exc),
            stack_fingerprint=f"{__name__}:{error_class}",
            top_frame_module=__name__,
            recovered=False,
            status="failed",
            correlation_id=correlation_id,
            severity="error",
        )
        raise

    await event_logging.record_dispatcher_event(
        component=HANDLER_COMPONENT,
        action_kind="send_message",
        validation_status="adapter_available",
        adapter_available=True,
        status="succeeded",
        correlation_id=correlation_id,
    )
    dispatch_result = {
        "conversation_message_id": conversation_message_id,
        "delivery_tracking_id": delivery_tracking_id,
        "adapter_message_id": send_result.message_id,
    }
    return dispatch_result


def _delivery_mentions(args: dict) -> list[dict] | None:
    """Return optional adapter-owned delivery mention metadata."""

    value = args.get("delivery_mentions")
    if isinstance(value, list):
        return_value = value
        return return_value
    return_value = None
    return return_value


def build_send_message_tool() -> ToolSpec:
    """Return the MVP ``send_message`` tool registration."""

    return_value = ToolSpec(
        name="send_message",
        description=(
            "Send a message to a platform channel or the same channel as the"
            " triggering user message. Use execute_at when the message should"
            " be delayed."
        ),
        args_schema=SEND_MESSAGE_SCHEMA,
        handler=handle_send_message,
    )
    return return_value
