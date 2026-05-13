"""Built-in task handlers for the dispatcher."""

from __future__ import annotations

from kazusa_ai_chatbot import event_logging
from kazusa_ai_chatbot.dispatcher.adapter_iface import AdapterRegistry
from kazusa_ai_chatbot.dispatcher.task import DispatchContext
from kazusa_ai_chatbot.dispatcher.tool_spec import ToolSpec

HANDLER_COMPONENT = "dispatcher.handlers"


def _handler_correlation_id(ctx: DispatchContext) -> str:
    """Build a non-content correlation id for a scheduled handler call."""

    message_ref = ctx.source_message_id or "no-message-id"
    correlation_id = f"dispatch:{ctx.source_platform}:{message_ref}"
    return correlation_id


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
    },
}


async def handle_send_message(
    args: dict,
    ctx: DispatchContext,
    adapters: AdapterRegistry,
) -> None:
    """Deliver a scheduled channel message through the target platform adapter.

    Args:
        args: Validated tool arguments for ``send_message``.
        ctx: Source-side dispatch context, available for future extensions.
        adapters: Adapter registry used to look up the platform transport.
    """

    correlation_id = _handler_correlation_id(ctx)
    adapter_available = False
    try:
        target_platform = str(args["target_platform"])
        adapter_available = adapters.has(target_platform)
        adapter = adapters.get(target_platform)
        await adapter.send_message(
            channel_id=args["target_channel"],
            text=args["text"],
            channel_type=args["target_channel_type"],
            reply_to_msg_id=args.get("reply_to_msg_id"),
        )
    except Exception as exc:
        rejection_code = (
            "adapter_send_failed"
            if adapter_available
            else "adapter_unavailable"
        )
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
