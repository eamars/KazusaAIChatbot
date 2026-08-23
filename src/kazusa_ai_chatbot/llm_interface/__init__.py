"""Backend-aware LLM interface package."""

from kazusa_ai_chatbot.llm_interface.contracts import (
    BackendDescriptor,
    LLMCallConfig,
    LLMInvalidToolCall,
    LLMInvoker,
    LLMResponse,
    LLMStreamChunk,
    LLMStreamFinish,
    LLMThinkingConfig,
    LLMToolCall,
    LLMToolDefinition,
    LLMToolHistoryMessage,
    LLMToolStreamInvoker,
)
from kazusa_ai_chatbot.llm_interface.interface import LLInterface

__all__ = [
    "BackendDescriptor",
    "LLInterface",
    "LLMCallConfig",
    "LLMInvalidToolCall",
    "LLMInvoker",
    "LLMResponse",
    "LLMStreamChunk",
    "LLMStreamFinish",
    "LLMThinkingConfig",
    "LLMToolCall",
    "LLMToolDefinition",
    "LLMToolHistoryMessage",
    "LLMToolStreamInvoker",
]
