from __future__ import annotations

from dataclasses import fields

from langchain_core.messages import AIMessage, HumanMessage

from kazusa_ai_chatbot.llm_interface import (
    BackendDescriptor,
    LLInterface,
    LLMCallConfig,
    LLMResponse,
    LLMThinkingConfig,
)
from kazusa_ai_chatbot.llm_interface import session as session_module
from kazusa_ai_chatbot.llm_interface.session import (
    build_diagnostic_fingerprint,
)


def _call_config(
    *,
    model: str = "gemma-4-27b-it",
    api_key: str = "secret-api-key",
    route_name: str = "COGNITION_LLM",
    thinking_enabled: bool = False,
) -> LLMCallConfig:
    """Build a representative module-owned LLM call config."""

    config = LLMCallConfig(
        stage_name="tests.contract",
        route_name=route_name,
        base_url="http://localhost:1234/v1/",
        api_key=api_key,
        model=model,
        temperature=0.25,
        top_p=0.9,
        top_k=40,
        max_completion_tokens=777,
        presence_penalty=None,
        thinking=LLMThinkingConfig(enabled=thinking_enabled),
    )
    return config


def test_call_config_uses_explicit_completion_budget_and_redacts_key() -> None:
    """The public config contract exposes max_completion_tokens only."""

    field_names = {field.name for field in fields(LLMCallConfig)}

    assert "max_completion_tokens" in field_names
    assert "max_tokens" not in field_names
    assert _call_config().thinking == LLMThinkingConfig()
    assert "secret-api-key" not in repr(_call_config())


def test_describe_backend_detects_gemma4_thinking_strategy() -> None:
    """Gemma 4 thinking is enabled through backend detection, not config kind."""

    interface = LLInterface()
    descriptor = interface.describe_backend(
        config=_call_config(thinking_enabled=True),
    )

    assert descriptor.backend_kind == "openai_compatible"
    assert descriptor.model_family == "gemma4"
    assert descriptor.normalized_base_url == "http://localhost:1234/v1"
    assert descriptor.thinking_strategy == "gemma4_enabled"
    assert descriptor.confidence == "model_name_inferred"


def test_describe_backend_ignores_unsupported_model_thinking() -> None:
    """Unsupported models continue without provider thinking payloads."""

    interface = LLInterface()
    descriptor = interface.describe_backend(
        config=_call_config(
            model="qwen2.5-32b",
            thinking_enabled=True,
        ),
    )

    assert descriptor.model_family == "qwen"
    assert descriptor.thinking_strategy == "ignored_unsupported_model"


def test_describe_backend_detects_qwen3_thinking_strategies() -> None:
    """Qwen3-family models use explicit enabled and disabled strategies."""

    interface = LLInterface()

    enabled_descriptor = interface.describe_backend(
        config=_call_config(
            model="qwen3.6-34b-80l-fable-5-heretic",
            thinking_enabled=True,
        ),
    )
    disabled_descriptor = interface.describe_backend(
        config=_call_config(
            model="qwen3.6-34b-80l-fable-5-heretic",
            thinking_enabled=False,
        ),
    )

    assert enabled_descriptor.model_family == "qwen"
    assert enabled_descriptor.thinking_strategy == "qwen3_enabled"
    assert disabled_descriptor.model_family == "qwen"
    assert disabled_descriptor.thinking_strategy == "qwen3_disabled"


def test_describe_backend_detects_prefixed_qwen3_model_names() -> None:
    """Repository-prefixed Qwen3 model ids still use Qwen3 thinking support."""

    interface = LLInterface()
    supported_models = (
        "hiebo/Qwen3.6-34B-80L-Fable-5-Heretic",
        "Qwen/Qwen3-32B",
        "Qwen/Qwen3",
    )

    for model in supported_models:
        enabled_descriptor = interface.describe_backend(
            config=_call_config(model=model, thinking_enabled=True),
        )
        disabled_descriptor = interface.describe_backend(
            config=_call_config(model=model, thinking_enabled=False),
        )

        assert enabled_descriptor.model_family == "qwen"
        assert enabled_descriptor.thinking_strategy == "qwen3_enabled"
        assert disabled_descriptor.model_family == "qwen"
        assert disabled_descriptor.thinking_strategy == "qwen3_disabled"


def test_describe_backend_detects_qwopus_qwen_derived_models() -> None:
    """Qwopus 3.x model ids use the same Qwen-compatible thinking support."""

    interface = LLInterface()
    supported_models = (
        "Jackrong/Qwopus3.6-27B-v2-GGUF",
        "Jackrong/Qwopus3.6-27B-v2-MTP-GGUF",
        "Jackrong/Qwopus3.5-9B-v3-GGUF",
        "Jackrong/Qwopus3.5-9B-Coder-GGUF",
        "Jackrong/Qwopus3.6-27B-Coder-MTP-GGUF",
    )

    for model in supported_models:
        enabled_descriptor = interface.describe_backend(
            config=_call_config(model=model, thinking_enabled=True),
        )
        disabled_descriptor = interface.describe_backend(
            config=_call_config(model=model, thinking_enabled=False),
        )

        assert enabled_descriptor.model_family == "qwen"
        assert enabled_descriptor.thinking_strategy == "qwen3_enabled"
        assert disabled_descriptor.model_family == "qwen"
        assert disabled_descriptor.thinking_strategy == "qwen3_disabled"


def test_describe_backend_rejects_qwen3_adjacent_names() -> None:
    """Broad Qwen/Qwopus 3-like names do not get thinking strategy."""

    interface = LLInterface()
    rejected_models = (
        "qwen30-7b",
        "qwen3-coder-30b",
        "qwen3coder",
        "qwen3-vl",
        "qwen3vl",
        "Qwen/Qwen3-Coder-30B",
        "Qwen/Qwen3-VL",
        "qwopus30-7b",
        "qwopus3-coder",
        "Jackrong/Qwopus3-Coder-GGUF",
    )

    for model in rejected_models:
        descriptor = interface.describe_backend(
            config=_call_config(model=model, thinking_enabled=True),
        )

        assert descriptor.model_family == "qwen"
        assert descriptor.thinking_strategy == "ignored_unsupported_model"


def test_backend_descriptor_cache_is_per_interface_and_invalidated() -> None:
    """Each LLInterface owns its descriptor generation and invalidation."""

    config = _call_config()
    first_interface = LLInterface()
    second_interface = LLInterface()

    first_descriptor = first_interface.describe_backend(config=config)
    cached_descriptor = first_interface.describe_backend(config=config)
    second_descriptor = second_interface.describe_backend(config=config)

    assert cached_descriptor.generation == first_descriptor.generation
    assert second_descriptor.generation == first_descriptor.generation

    first_interface.invalidate_backend(route_name=config.route_name)
    invalidated_descriptor = first_interface.describe_backend(config=config)
    still_cached_descriptor = second_interface.describe_backend(config=config)

    assert invalidated_descriptor.generation == first_descriptor.generation + 1
    assert still_cached_descriptor.generation == second_descriptor.generation


def test_route_invalidation_evicts_shared_provider_session(monkeypatch) -> None:
    """Invalidating one route must rebuild shared provider/client sessions."""

    created_providers: list[object] = []

    class _Provider:
        """Minimal provider that records session construction."""

        def __init__(self) -> None:
            created_providers.append(self)

        async def ainvoke(self, messages, *, config, backend):  # pragma: no cover
            del messages, config
            response = LLMResponse(
                content="ok",
                backend=backend,
                raw_response=None,
                usage={},
            )
            return response

        def invoke(self, messages, *, config, backend):
            del messages, config
            response = LLMResponse(
                content="ok",
                backend=backend,
                raw_response=None,
                usage={},
            )
            return response

        async def aclose(self) -> None:
            """No resources are opened by the test provider."""

    monkeypatch.setattr(session_module, "OpenAICompatibleProvider", _Provider)
    interface = LLInterface()
    first_route = _call_config(route_name="FIRST_ROUTE")
    second_route = _call_config(route_name="SECOND_ROUTE")
    messages = [HumanMessage(content="hello")]

    interface.invoke(messages, config=first_route)
    interface.invoke(messages, config=second_route)
    interface.invalidate_backend(route_name=first_route.route_name)
    interface.invoke(messages, config=first_route)

    assert len(created_providers) == 2


def test_diagnostic_fingerprint_excludes_raw_api_key() -> None:
    """Diagnostic identity must be useful without disclosing secrets."""

    config = _call_config(api_key="do-not-log-this-key")
    descriptor = LLInterface().describe_backend(config=config)
    fingerprint = build_diagnostic_fingerprint(
        config=config,
        descriptor=descriptor,
    )
    fingerprint_text = repr(fingerprint)

    assert "do-not-log-this-key" not in fingerprint_text
    assert config.stage_name in fingerprint_text
    assert config.route_name in fingerprint_text
    assert descriptor.model_family in fingerprint_text


def test_llm_response_wraps_content_backend_raw_response_and_usage() -> None:
    """The normalized response keeps model text and raw provider response."""

    backend = BackendDescriptor(
        route_name="COGNITION_LLM",
        backend_kind="openai_compatible",
        model_family="gemma4",
        model="gemma-4-27b-it",
        normalized_base_url="http://localhost:1234/v1",
        thinking_strategy="disabled",
        confidence="model_name_inferred",
        generation=1,
    )
    raw_response = AIMessage(
        content="ok",
        response_metadata={"token_usage": {"completion_tokens": 3}},
    )

    response = LLMResponse.from_raw(raw_response, backend=backend)

    assert response.content == "ok"
    assert response.backend == backend
    assert response.raw_response is raw_response
    assert response.usage == {"completion_tokens": 3}


def test_llm_response_strips_complete_gemma4_thought_channel() -> None:
    """Gemma 4 thinking content is removed from caller-facing text."""

    backend = BackendDescriptor(
        route_name="COGNITION_LLM",
        backend_kind="openai_compatible",
        model_family="gemma4",
        model="gemma-4-27b-it",
        normalized_base_url="http://localhost:1234/v1",
        thinking_strategy="gemma4_enabled",
        confidence="model_name_inferred",
        generation=1,
    )
    raw_response = AIMessage(
        content="<|channel>thought\nprivate reasoning<channel|>{\"ok\": true}",
    )

    response = LLMResponse.from_raw(raw_response, backend=backend)

    assert response.content == '{"ok": true}'
    assert raw_response.content.startswith("<|channel>thought")


def test_llm_response_strips_truncated_gemma4_thought_channel() -> None:
    """Unclosed Gemma 4 thinking content does not leak to callers."""

    backend = BackendDescriptor(
        route_name="COGNITION_LLM",
        backend_kind="openai_compatible",
        model_family="gemma4",
        model="gemma-4-27b-it",
        normalized_base_url="http://localhost:1234/v1",
        thinking_strategy="gemma4_enabled",
        confidence="model_name_inferred",
        generation=1,
    )
    raw_response = AIMessage(
        content="visible prefix\n<|channel>thought\nprivate reasoning",
    )

    response = LLMResponse.from_raw(raw_response, backend=backend)

    assert response.content == "visible prefix"


def test_llm_response_strips_complete_qwen_think_tag() -> None:
    """Visible Qwen think tags are removed from caller-facing text."""

    backend = BackendDescriptor(
        route_name="COGNITION_LLM",
        backend_kind="openai_compatible",
        model_family="qwen",
        model="qwen3.6-34b",
        normalized_base_url="http://localhost:1234/v1",
        thinking_strategy="qwen3_enabled",
        confidence="model_name_inferred",
        generation=1,
    )
    raw_response = AIMessage(
        content="<think>\nprivate reasoning\n</think>\n\n{\"ok\": true}",
    )

    response = LLMResponse.from_raw(raw_response, backend=backend)

    assert response.content == "\n\n{\"ok\": true}"
    assert raw_response.content.startswith("<think>")


def test_llm_response_strips_truncated_qwen_think_tag() -> None:
    """Unclosed Qwen thinking content does not leak to callers."""

    backend = BackendDescriptor(
        route_name="COGNITION_LLM",
        backend_kind="openai_compatible",
        model_family="qwen",
        model="qwen3.6-34b",
        normalized_base_url="http://localhost:1234/v1",
        thinking_strategy="qwen3_enabled",
        confidence="model_name_inferred",
        generation=1,
    )
    raw_response = AIMessage(
        content="visible prefix\n<think>\nprivate reasoning",
    )

    response = LLMResponse.from_raw(raw_response, backend=backend)

    assert response.content == "visible prefix"
