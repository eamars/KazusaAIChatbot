"""Public LLM backend compatibility interface."""

from __future__ import annotations

from typing import Sequence

from langchain_core.messages import BaseMessage

from kazusa_ai_chatbot.llm_interface.contracts import (
    BackendDescriptor,
    LLMCallConfig,
    LLMResponse,
)
from kazusa_ai_chatbot.llm_interface.session import InterfaceSessionCache


class LLInterface:
    """Invoke chat LLMs through backend-aware provider adapters."""

    def __init__(self) -> None:
        self._session_cache = InterfaceSessionCache()

    async def ainvoke(
        self,
        messages: Sequence[BaseMessage],
        *,
        config: LLMCallConfig,
    ) -> LLMResponse:
        """Invoke a configured chat route asynchronously."""

        backend = self.describe_backend(config=config)
        provider = self._session_cache.provider_for(
            config=config,
            descriptor=backend,
        )
        response = await provider.ainvoke(
            messages,
            config=config,
            backend=backend,
        )
        return response

    def invoke(
        self,
        messages: Sequence[BaseMessage],
        *,
        config: LLMCallConfig,
    ) -> LLMResponse:
        """Invoke a configured chat route synchronously."""

        backend = self.describe_backend(config=config)
        provider = self._session_cache.provider_for(
            config=config,
            descriptor=backend,
        )
        response = provider.invoke(
            messages,
            config=config,
            backend=backend,
        )
        return response

    def describe_backend(
        self,
        *,
        config: LLMCallConfig,
    ) -> BackendDescriptor:
        """Return the detected backend descriptor for a call config."""

        descriptor = self._session_cache.describe_backend(config=config)
        return descriptor

    def invalidate_backend(self, *, route_name: str | None = None) -> None:
        """Invalidate cached backend descriptors and provider sessions."""

        self._session_cache.invalidate_backend(route_name=route_name)

    async def aclose(self) -> None:
        """Close cached provider sessions."""

        await self._session_cache.aclose()
