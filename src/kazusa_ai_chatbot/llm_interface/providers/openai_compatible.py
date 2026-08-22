"""OpenAI-compatible provider adapter for chat models."""

from __future__ import annotations

import hashlib
import logging
import re
from typing import Callable, Literal, Sequence

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langchain_openai import ChatOpenAI
from openai import BadRequestError

from kazusa_ai_chatbot.llm_interface.contracts import (
    BackendDescriptor,
    LLMCallConfig,
    LLMResponse,
)
from kazusa_ai_chatbot.llm_interface.reload import ReloadingChatModel

ChatModelFactory = Callable[..., object]
ChatModelCacheKey = tuple[object, ...]
_ProviderOutputTransport = Literal["json_object", "json_schema", "text"]
GEMMA4_THINKING_TRIGGER = "/think"
QWEN3_THINKING_PREFILL = "<think>\n"
_JSON_OBJECT_ALLOWED_MODE_ERROR = re.compile(
    r"response_format\.type['\"]?\s+must\s+be\s+"
    r"['\"]?json_schema['\"]?\s+or\s+['\"]?text['\"]?",
)
logger = logging.getLogger(__name__)


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

        chat_model = self._build_chat_model(
            config=config,
            backend=backend,
            output_transport=config.output_mode,
        )
        provider_messages = _provider_messages(
            messages,
            backend=backend,
        )
        try:
            raw_response = await chat_model.ainvoke(provider_messages)
        except BadRequestError as exc:
            if not _is_unsupported_json_object_error(exc, config=config):
                raise
            logger.warning(
                f"Provider rejected JSON-object output for {config.model}; "
                f"retrying once with JSON Schema: {exc}"
            )
            fallback_model = self._build_chat_model(
                config=config,
                backend=backend,
                output_transport="json_schema",
            )
            raw_response = await fallback_model.ainvoke(provider_messages)
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

        chat_model = self._build_chat_model(
            config=config,
            backend=backend,
            output_transport=config.output_mode,
        )
        provider_messages = _provider_messages(
            messages,
            backend=backend,
        )
        try:
            raw_response = chat_model.invoke(provider_messages)
        except BadRequestError as exc:
            if not _is_unsupported_json_object_error(exc, config=config):
                raise
            logger.warning(
                f"Provider rejected JSON-object output for {config.model}; "
                f"retrying once with JSON Schema: {exc}"
            )
            fallback_model = self._build_chat_model(
                config=config,
                backend=backend,
                output_transport="json_schema",
            )
            raw_response = fallback_model.invoke(provider_messages)
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
        output_transport: _ProviderOutputTransport,
    ) -> ReloadingChatModel:
        """Build the configured chat model for one provider request."""

        cache_key = _chat_model_cache_key(
            config=config,
            backend=backend,
            output_transport=output_transport,
        )
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
        if config.timeout_seconds is not None:
            kwargs["timeout"] = config.timeout_seconds
        if output_transport == "json_object":
            kwargs["model_kwargs"] = {
                "response_format": {"type": "json_object"},
            }
        elif output_transport == "json_schema":
            kwargs["model_kwargs"] = {
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "kazusa_json_object",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "additionalProperties": True,
                        },
                    },
                },
            }
        if backend.thinking_strategy == "gemma4_enabled":
            kwargs["extra_body"] = {
                "chat_template_kwargs": {"enable_thinking": True},
            }
        if backend.thinking_strategy == "qwen3_enabled":
            kwargs["extra_body"] = {
                "chat_template_kwargs": {"enable_thinking": True},
            }
        if backend.thinking_strategy == "qwen3_disabled":
            kwargs["extra_body"] = {
                "chat_template_kwargs": {"enable_thinking": False},
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


def _provider_messages(
    messages: Sequence[BaseMessage],
    *,
    backend: BackendDescriptor,
) -> Sequence[BaseMessage]:
    """Return backend-ready messages for one provider invocation."""

    if backend.thinking_strategy == "gemma4_enabled":
        provider_messages = _gemma4_thinking_messages(messages)
        return provider_messages
    if backend.thinking_strategy == "qwen3_enabled":
        provider_messages = _qwen3_thinking_messages(messages)
        return provider_messages

    return messages


def _gemma4_thinking_messages(
    messages: Sequence[BaseMessage],
) -> list[BaseMessage]:
    """Inject Gemma 4's prompt-level thinking trigger without mutating input."""

    if not messages:
        provider_messages: list[BaseMessage] = [
            SystemMessage(content=GEMMA4_THINKING_TRIGGER),
        ]
        return provider_messages

    first_message = messages[0]
    if isinstance(first_message, SystemMessage):
        content = first_message.content
        if (
            isinstance(content, str)
            and content.lstrip().startswith(GEMMA4_THINKING_TRIGGER)
        ):
            provider_messages = list(messages)
            return provider_messages
        if isinstance(content, str):
            updated_first_message = first_message.model_copy(
                update={
                    "content": f"{GEMMA4_THINKING_TRIGGER}\n{content}",
                }
            )
            provider_messages = [updated_first_message, *messages[1:]]
            return provider_messages

    provider_messages = [
        SystemMessage(content=GEMMA4_THINKING_TRIGGER),
        *messages,
    ]
    return provider_messages


def _qwen3_thinking_messages(
    messages: Sequence[BaseMessage],
) -> list[BaseMessage]:
    """Add Qwen3's assistant prefill without mutating caller messages."""

    if messages:
        last_message = messages[-1]
        if isinstance(last_message, AIMessage):
            content = last_message.content
            if (
                isinstance(content, str)
                and content.lstrip().startswith(QWEN3_THINKING_PREFILL.strip())
            ):
                provider_messages = list(messages)
                return provider_messages

    provider_messages = [
        *messages,
        AIMessage(content=QWEN3_THINKING_PREFILL),
    ]
    return provider_messages


def _chat_model_cache_key(
    *,
    config: LLMCallConfig,
    backend: BackendDescriptor,
    output_transport: _ProviderOutputTransport,
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
        config.timeout_seconds,
        output_transport,
        backend.thinking_strategy,
    )
    return cache_key


def _is_unsupported_json_object_error(
    exc: BadRequestError,
    *,
    config: LLMCallConfig,
) -> bool:
    """Identify an endpoint rejection specific to native JSON-object mode."""

    if config.output_mode != "json_object":
        return False

    error_text = str(exc).lower()
    if (
        _JSON_OBJECT_ALLOWED_MODE_ERROR.search(error_text) is not None
        and "json_object" not in error_text
    ):
        return True

    format_terms = (
        "response_format",
        "json_object",
        "json mode",
        "json-mode",
    )
    if not any(term in error_text for term in format_terms):
        return False

    unsupported_terms = (
        "unsupported",
        "not support",
        "unknown parameter",
        "unrecognized",
        "not implemented",
        "only support",
        "does not accept",
    )
    return any(term in error_text for term in unsupported_terms)
