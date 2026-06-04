"""Task dispatch primitives for deferred tool execution."""

from kazusa_ai_chatbot.dispatcher.adapter_iface import (
    AdapterChannelUnavailableError,
    AdapterRegistry,
    MessagingAdapter,
    SendResult,
    UnknownPlatformError,
)
from kazusa_ai_chatbot.dispatcher.handlers import build_send_message_tool, handle_send_message
from kazusa_ai_chatbot.dispatcher.remote_adapter import RemoteHttpAdapter
from kazusa_ai_chatbot.dispatcher.task import (
    BotPermissionRole,
    DispatchContext,
    Task,
)
from kazusa_ai_chatbot.dispatcher.tool_spec import ToolRegistry, ToolSpec

__all__ = [
    "AdapterChannelUnavailableError",
    "AdapterRegistry",
    "BotPermissionRole",
    "DispatchContext",
    "MessagingAdapter",
    "RemoteHttpAdapter",
    "SendResult",
    "Task",
    "ToolRegistry",
    "ToolSpec",
    "UnknownPlatformError",
    "build_send_message_tool",
    "handle_send_message",
]
