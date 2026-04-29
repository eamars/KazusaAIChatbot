"""Built-in task handlers for the dispatcher."""

from __future__ import annotations

from kazusa_ai_chatbot.dispatcher.adapter_iface import AdapterRegistry
from kazusa_ai_chatbot.dispatcher.task import DispatchContext
from kazusa_ai_chatbot.dispatcher.tool_spec import ToolSpec

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

    del ctx
    adapter = adapters.get(args["target_platform"])
    await adapter.send_message(
        channel_id=args["target_channel"],
        text=args["text"],
        reply_to_msg_id=args.get("reply_to_msg_id"),
    )


def build_send_message_tool() -> ToolSpec:
    """Return the MVP ``send_message`` tool registration."""

    return_value = ToolSpec(
        name="send_message",
        description=(
            "Send a message to a platform channel or the same channel as the"
            " triggering user message. Use an absolute UTC execute_at when the"
            " message should be delayed."
        ),
        args_schema=SEND_MESSAGE_SCHEMA,
        handler=handle_send_message,
    )
    return return_value
