"""OpenAI-compatible provider adapter for chat models."""

from __future__ import annotations

import hashlib
from typing import Callable, Sequence

from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI

from kazusa_ai_chatbot.llm_interface.contracts import (
    BackendDescriptor,
    LLMCallConfig,
    LLMResponse,
)
from kazusa_ai_chatbot.llm_interface.reload import ReloadingChatModel

ChatModelFactory = Callable[..., object]
ChatModelCacheKey = tuple[object, ...]


class OpenAICompatibleProvider:
    """Map public LLM configs into OpenAI-compatible ChatOpenAI calls."""

    def __init__(
        self,
        *,
        chat_model_factory: ChatModelFactory = ChatOpenAI,
    ) -> None:
        self._chat_model_factory = chat_model_factory
        self._chat_models: dict[ChatModelCacheKey, ReloadingChatModel] = {}

    async def ainvoke(
        self,
        messages: Sequence[BaseMessage],
        *,
        config: LLMCallConfig,
        backend: BackendDescriptor,
    ) -> LLMResponse:
        """Invoke an OpenAI-compatible chat model asynchronously."""

        chat_model = self._build_chat_model(config=config, backend=backend)
        raw_response = await chat_model.ainvoke(messages)
        response = LLMResponse.from_raw(raw_response, backend=backend)
        return response

    def invoke(
        self,
        messages: Sequence[BaseMessage],
        *,
        config: LLMCallConfig,
        backend: BackendDescriptor,
    ) -> LLMResponse:
        """Invoke an OpenAI-compatible chat model synchronously."""

        chat_model = self._build_chat_model(config=config, backend=backend)
        raw_response = chat_model.invoke(messages)
        response = LLMResponse.from_raw(raw_response, backend=backend)
        return response

    async def aclose(self) -> None:
        """Close provider resources when present."""

        self._chat_models.clear()

    def _build_chat_model(
        self,
        *,
        config: LLMCallConfig,
        backend: BackendDescriptor,
    ) -> ReloadingChatModel:
        """Build the configured chat model for one provider request."""

        cache_key = _chat_model_cache_key(config=config, backend=backend)
        cached_model = self._chat_models.get(cache_key)
        if cached_model is not None:
            return cached_model

        kwargs: dict[str, object] = {
            "model": config.model,
            "base_url": config.base_url,
            "api_key": config.api_key,
        }
        if config.temperature is not None:
            kwargs["temperature"] = config.temperature
        if config.top_p is not None:
            kwargs["top_p"] = config.top_p
        if config.max_completion_tokens is not None:
            kwargs["max_completion_tokens"] = config.max_completion_tokens
        if config.presence_penalty is not None:
            kwargs["presence_penalty"] = config.presence_penalty
        if backend.thinking_strategy == "gemma4_enabled":
            kwargs["extra_body"] = {
                "chat_template_kwargs": {"enable_thinking": True},
            }

        inner_model = self._chat_model_factory(**kwargs)
        chat_model = ReloadingChatModel(
            inner_model,
            base_url=config.base_url,
            model=config.model,
        )
        self._chat_models[cache_key] = chat_model
        return chat_model


def _api_key_hash(api_key: str) -> str:
    """Return a non-secret identity for cache partitioning."""

    digest = hashlib.sha256(api_key.encode("utf-8")).hexdigest()
    return digest


def _chat_model_cache_key(
    *,
    config: LLMCallConfig,
    backend: BackendDescriptor,
) -> ChatModelCacheKey:
    """Build a provider-local chat model cache key."""

    cache_key = (
        backend.backend_kind,
        backend.normalized_base_url,
        _api_key_hash(config.api_key),
        config.model,
        config.temperature,
        config.top_p,
        config.top_k,
        config.max_completion_tokens,
        config.presence_penalty,
        backend.thinking_strategy,
    )
    return cache_key
