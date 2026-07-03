"""Public contracts for backend-aware chat LLM invocation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Protocol, Sequence

from langchain_core.messages import BaseMessage

GEMMA4_THOUGHT_CHANNEL_START = "<|channel>thought"
GEMMA4_THOUGHT_CHANNEL_END = "<channel|>"
QWEN_THINK_TAG_START = "<think>"
QWEN_THINK_TAG_END = "</think>"


@dataclass(frozen=True)
class LLMThinkingConfig:
    """Boolean provider-side thinking request for a route call."""

    enabled: bool = False


@dataclass(frozen=True)
class LLMCallConfig:
    """Module-owned route, model, and generation config for one LLM stage."""

    stage_name: str
    route_name: str
    base_url: str
    api_key: str = field(repr=False)
    model: str
    temperature: float | None
    top_p: float | None
    top_k: int | None
    max_completion_tokens: int | None
    presence_penalty: float | None
    timeout_seconds: float | None = None
    thinking: LLMThinkingConfig = field(default_factory=LLMThinkingConfig)


@dataclass(frozen=True)
class BackendDescriptor:
    """Detected backend identity and effective provider strategy."""

    route_name: str
    backend_kind: str
    model_family: str
    model: str
    normalized_base_url: str
    thinking_strategy: str
    confidence: str
    generation: int


@dataclass(frozen=True)
class LLMResponse:
    """Normalized response returned by the LLM interface."""

    content: str
    backend: BackendDescriptor
    raw_response: object | None
    usage: Mapping[str, object]

    @classmethod
    def from_raw(
        cls,
        raw_response: object,
        *,
        backend: BackendDescriptor,
    ) -> "LLMResponse":
        """Wrap a provider-native response without hiding the raw object."""

        raw_content = getattr(raw_response, "content", "")
        if isinstance(raw_content, str):
            content = _normalize_response_content(
                raw_content,
                backend=backend,
            )
        else:
            content = str(raw_content)

        usage = _extract_usage(raw_response)
        response = cls(
            content=content,
            backend=backend,
            raw_response=raw_response,
            usage=usage,
        )
        return response


class LLMInvoker(Protocol):
    """Explicit-config LLM invoker used by cognition-chain services."""

    async def ainvoke(
        self,
        messages: Sequence[BaseMessage],
        *,
        config: LLMCallConfig,
    ) -> LLMResponse:
        """Invoke a chat model asynchronously."""

    def invoke(
        self,
        messages: Sequence[BaseMessage],
        *,
        config: LLMCallConfig,
    ) -> LLMResponse:
        """Invoke a chat model synchronously."""


class ProviderAdapter(Protocol):
    """Provider adapter contract used by LLInterface sessions."""

    async def ainvoke(
        self,
        messages: Sequence[BaseMessage],
        *,
        config: LLMCallConfig,
        backend: BackendDescriptor,
    ) -> LLMResponse:
        """Invoke the provider asynchronously."""

    def invoke(
        self,
        messages: Sequence[BaseMessage],
        *,
        config: LLMCallConfig,
        backend: BackendDescriptor,
    ) -> LLMResponse:
        """Invoke the provider synchronously."""

    async def aclose(self) -> None:
        """Close any provider-owned resources."""


def _extract_usage(raw_response: object) -> Mapping[str, object]:
    """Extract token usage from known LangChain response metadata shapes."""

    response_metadata = getattr(raw_response, "response_metadata", None)
    if isinstance(response_metadata, Mapping):
        token_usage = response_metadata.get("token_usage")
        if isinstance(token_usage, Mapping):
            usage = dict(token_usage)
            return usage
        usage_metadata = response_metadata.get("usage")
        if isinstance(usage_metadata, Mapping):
            usage = dict(usage_metadata)
            return usage

    usage_metadata = getattr(raw_response, "usage_metadata", None)
    if isinstance(usage_metadata, Mapping):
        usage = dict(usage_metadata)
        return usage

    return_value: Mapping[str, object] = {}
    return return_value


def _normalize_response_content(
    raw_content: str,
    *,
    backend: BackendDescriptor,
) -> str:
    """Return caller-facing content with provider thought channels removed."""

    if backend.model_family == "qwen":
        content = _strip_qwen_think_tags(raw_content)
        return content
    if backend.model_family != "gemma4":
        return raw_content

    content = _strip_gemma4_thought_channels(raw_content)
    return content


def _strip_gemma4_thought_channels(raw_content: str) -> str:
    """Remove Gemma 4 thought-channel spans from visible response content."""

    content = raw_content
    while True:
        try:
            start_index = content.index(GEMMA4_THOUGHT_CHANNEL_START)
        except ValueError:
            return content

        try:
            end_index = content.index(
                GEMMA4_THOUGHT_CHANNEL_END,
                start_index + len(GEMMA4_THOUGHT_CHANNEL_START),
            )
        except ValueError:
            stripped_content = content[:start_index].rstrip()
            return stripped_content

        after_end_index = end_index + len(GEMMA4_THOUGHT_CHANNEL_END)
        content = content[:start_index] + content[after_end_index:]


def _strip_qwen_think_tags(raw_content: str) -> str:
    """Remove Qwen visible thinking spans from caller-facing content."""

    content = raw_content
    while True:
        try:
            start_index = content.index(QWEN_THINK_TAG_START)
        except ValueError:
            return content

        try:
            end_index = content.index(
                QWEN_THINK_TAG_END,
                start_index + len(QWEN_THINK_TAG_START),
            )
        except ValueError:
            stripped_content = content[:start_index].rstrip()
            return stripped_content

        after_end_index = end_index + len(QWEN_THINK_TAG_END)
        content = content[:start_index] + content[after_end_index:]
