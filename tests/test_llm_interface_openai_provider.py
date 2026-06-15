from __future__ import annotations

import pytest

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from kazusa_ai_chatbot.llm_interface import (
    BackendDescriptor,
    LLMCallConfig,
    LLMThinkingConfig,
)
from kazusa_ai_chatbot.llm_interface.providers.openai_compatible import (
    OpenAICompatibleProvider,
)


class _FakeChatModel:
    """Capture constructor kwargs and invoke payloads for provider tests."""

    def __init__(self, **kwargs: object) -> None:
        self.constructor_kwargs = kwargs
        self.async_calls: list[list[object]] = []
        self.sync_calls: list[list[object]] = []

    async def ainvoke(self, messages: list[object], *, config=None) -> AIMessage:
        self.async_calls.append(messages)
        response = AIMessage(
            content="async ok",
            response_metadata={"token_usage": {"completion_tokens": 1}},
        )
        return response

    def invoke(self, messages: list[object]) -> AIMessage:
        self.sync_calls.append(messages)
        response = AIMessage(
            content="sync ok",
            response_metadata={"token_usage": {"completion_tokens": 2}},
        )
        return response


def _config(*, thinking_enabled: bool = False) -> LLMCallConfig:
    """Build a provider test config with every supported generation field."""

    config = LLMCallConfig(
        stage_name="tests.provider",
        route_name="COGNITION_LLM",
        base_url="http://localhost:1234/v1/",
        api_key="provider-secret",
        model="gemma-4-27b-it",
        temperature=0.4,
        top_p=0.8,
        top_k=32,
        max_completion_tokens=1234,
        presence_penalty=0.2,
        thinking=LLMThinkingConfig(enabled=thinking_enabled),
    )
    return config


def _backend(*, thinking_strategy: str = "disabled") -> BackendDescriptor:
    """Build a provider backend descriptor for request mapping tests."""

    descriptor = BackendDescriptor(
        route_name="COGNITION_LLM",
        backend_kind="openai_compatible",
        model_family="gemma4",
        model="gemma-4-27b-it",
        normalized_base_url="http://localhost:1234/v1",
        thinking_strategy=thinking_strategy,
        confidence="model_name_inferred",
        generation=1,
    )
    return descriptor


def test_provider_maps_config_to_chat_model_constructor() -> None:
    """OpenAI-compatible construction uses the unified public config fields."""

    created_models: list[_FakeChatModel] = []

    def _factory(**kwargs: object) -> _FakeChatModel:
        model = _FakeChatModel(**kwargs)
        created_models.append(model)
        return model

    provider = OpenAICompatibleProvider(chat_model_factory=_factory)
    messages = [SystemMessage(content="system"), HumanMessage(content="human")]

    response = provider.invoke(
        messages,
        config=_config(),
        backend=_backend(),
    )

    constructed_kwargs = created_models[0].constructor_kwargs
    assert constructed_kwargs["model"] == "gemma-4-27b-it"
    assert constructed_kwargs["base_url"] == "http://localhost:1234/v1/"
    assert constructed_kwargs["api_key"] == "provider-secret"
    assert constructed_kwargs["temperature"] == 0.4
    assert constructed_kwargs["top_p"] == 0.8
    assert constructed_kwargs["max_completion_tokens"] == 1234
    assert constructed_kwargs["presence_penalty"] == 0.2
    assert "max_tokens" not in constructed_kwargs
    assert created_models[0].sync_calls == [messages]
    assert response.content == "sync ok"
    assert response.raw_response.content == "sync ok"
    assert response.usage == {"completion_tokens": 2}


@pytest.mark.asyncio
async def test_provider_async_path_preserves_message_objects() -> None:
    """Async calls pass the same ordered messages through to the backend."""

    created_models: list[_FakeChatModel] = []

    def _factory(**kwargs: object) -> _FakeChatModel:
        model = _FakeChatModel(**kwargs)
        created_models.append(model)
        return model

    provider = OpenAICompatibleProvider(chat_model_factory=_factory)
    messages = [SystemMessage(content="system"), HumanMessage(content="human")]

    response = await provider.ainvoke(
        messages,
        config=_config(),
        backend=_backend(),
    )

    assert created_models[0].async_calls == [messages]
    assert response.content == "async ok"
    assert response.raw_response.content == "async ok"
    assert response.usage == {"completion_tokens": 1}


def test_provider_adds_gemma4_thinking_payload_only_when_enabled() -> None:
    """Gemma 4 thinking maps to provider payload without touching messages."""

    created_models: list[_FakeChatModel] = []

    def _factory(**kwargs: object) -> _FakeChatModel:
        model = _FakeChatModel(**kwargs)
        created_models.append(model)
        return model

    provider = OpenAICompatibleProvider(chat_model_factory=_factory)

    provider.invoke(
        [HumanMessage(content="hello")],
        config=_config(thinking_enabled=True),
        backend=_backend(thinking_strategy="gemma4_enabled"),
    )

    extra_body = created_models[0].constructor_kwargs["extra_body"]
    assert extra_body == {"chat_template_kwargs": {"enable_thinking": True}}


def test_provider_omits_thinking_payload_for_unsupported_or_disabled() -> None:
    """Unsupported thinking never leaks provider-specific fields to a request."""

    created_models: list[_FakeChatModel] = []

    def _factory(**kwargs: object) -> _FakeChatModel:
        model = _FakeChatModel(**kwargs)
        created_models.append(model)
        return model

    provider = OpenAICompatibleProvider(chat_model_factory=_factory)

    provider.invoke(
        [HumanMessage(content="disabled")],
        config=_config(thinking_enabled=False),
        backend=_backend(thinking_strategy="disabled"),
    )
    provider.invoke(
        [HumanMessage(content="unsupported")],
        config=_config(thinking_enabled=True),
        backend=_backend(thinking_strategy="ignored_unsupported_model"),
    )

    assert "extra_body" not in created_models[0].constructor_kwargs
    assert "extra_body" not in created_models[1].constructor_kwargs
