"""Backend-aware LLM interface package."""

from kazusa_ai_chatbot.llm_interface.contracts import (
    BackendDescriptor,
    LLMCallConfig,
    LLMInvoker,
    LLMResponse,
    LLMThinkingConfig,
)
from kazusa_ai_chatbot.llm_interface.interface import LLInterface

__all__ = [
    "BackendDescriptor",
    "LLInterface",
    "LLMCallConfig",
    "LLMInvoker",
    "LLMResponse",
    "LLMThinkingConfig",
]
