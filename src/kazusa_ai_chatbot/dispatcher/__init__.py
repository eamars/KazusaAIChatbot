"""Task dispatch primitives for deferred tool execution."""

from kazusa_ai_chatbot.dispatcher.adapter_iface import (
    AdapterRegistry,
    MessagingAdapter,
    SendResult,
    UnknownPlatformError,
)
from kazusa_ai_chatbot.dispatcher.dispatcher import TaskDispatcher
from kazusa_ai_chatbot.dispatcher.evaluator import EvalResult, ToolCallEvaluator
from kazusa_ai_chatbot.dispatcher.handlers import build_send_message_tool, handle_send_message
from kazusa_ai_chatbot.dispatcher.pending_index import PendingTaskIndex
from kazusa_ai_chatbot.dispatcher.remote_adapter import RemoteHttpAdapter
from kazusa_ai_chatbot.dispatcher.task import (
    DispatchContext,
    DispatchResult,
    RawToolCall,
    Task,
)
from kazusa_ai_chatbot.dispatcher.tool_spec import ToolRegistry, ToolSpec

__all__ = [
    "AdapterRegistry",
    "DispatchContext",
    "DispatchResult",
    "EvalResult",
    "MessagingAdapter",
    "PendingTaskIndex",
    "RawToolCall",
    "RemoteHttpAdapter",
    "SendResult",
    "Task",
    "TaskDispatcher",
    "ToolCallEvaluator",
    "ToolRegistry",
    "ToolSpec",
    "UnknownPlatformError",
    "build_send_message_tool",
    "handle_send_message",
]
