"""Public contracts for backend-aware chat LLM invocation."""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal, Protocol

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
    output_mode: Literal["json_object", "text"] = "json_object"
    context_window_tokens: int | None = None


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
    ) -> LLMResponse:
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


@dataclass(frozen=True)
class LLMToolDefinition:
    """Provider-neutral native tool definition for one streamed model call."""

    name: str
    description: str
    parameters: Mapping[str, object]

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("LLM tool name must be non-empty")
        if not self.description.strip():
            raise ValueError("LLM tool description must be non-empty")
        if self.parameters.get("type") != "object":
            raise ValueError("LLM tool parameters must be an object schema")
        frozen_parameters = MappingProxyType(dict(self.parameters))
        object.__setattr__(self, "parameters", frozen_parameters)


@dataclass(frozen=True)
class LLMToolCall:
    """One complete provider-neutral native tool call."""

    call_id: str
    name: str
    arguments: Mapping[str, object]

    def __post_init__(self) -> None:
        if not self.call_id.strip():
            raise ValueError("LLM tool call id must be non-empty")
        if not self.name.strip():
            raise ValueError("LLM tool call name must be non-empty")
        frozen_arguments = MappingProxyType(dict(self.arguments))
        object.__setattr__(self, "arguments", frozen_arguments)


@dataclass(frozen=True)
class LLMInvalidToolCall:
    """Bounded structural error for one provider-native tool call."""

    call_id: str | None
    name: str | None
    error: str

    def __post_init__(self) -> None:
        if not self.error.strip():
            raise ValueError("invalid LLM tool call error must be non-empty")


@dataclass(frozen=True)
class LLMToolHistoryMessage:
    """Role-discriminated history row for native tool streaming."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str = ""
    reasoning: str | None = None
    tool_calls: tuple[LLMToolCall, ...] = ()
    tool_call_id: str | None = None

    def __post_init__(self) -> None:
        if self.role in {"system", "user"}:
            if not self.content:
                raise ValueError(f"{self.role} tool-history content is required")
            if self.reasoning is not None or self.tool_calls or self.tool_call_id:
                raise ValueError(f"{self.role} tool-history fields are invalid")
            return
        if self.role == "assistant":
            if self.tool_call_id is not None:
                raise ValueError("assistant tool history cannot carry tool_call_id")
            return
        if not self.content or not self.tool_call_id:
            raise ValueError("tool history requires content and tool_call_id")
        if self.reasoning is not None or self.tool_calls:
            raise ValueError("tool history cannot carry assistant fields")


@dataclass(frozen=True)
class LLMStreamFinish:
    """Terminal disposition emitted once for a native tool stream."""

    reason: Literal["stop", "tool_calls", "max_tokens", "aborted", "error"]
    detail: str = ""


@dataclass(frozen=True)
class LLMStreamChunk:
    """One normalized block, usage, or finish event from a tool stream."""

    kind: Literal[
        "block_start",
        "reasoning_delta",
        "text_delta",
        "tool_call_delta",
        "block_end",
        "usage",
        "finish",
    ]
    block_index: int | None = None
    block_type: Literal["reasoning", "text", "tool_call"] | None = None
    reasoning_delta: str = ""
    text_delta: str = ""
    tool_call_id: str | None = None
    tool_name: str | None = None
    tool_arguments_delta: str = ""
    completed_block: Mapping[str, object] = field(default_factory=dict)
    usage: Mapping[str, object] = field(default_factory=dict)
    finish: LLMStreamFinish | None = None

    def __post_init__(self) -> None:
        if self.block_index is not None and self.block_index < 0:
            raise ValueError("LLM stream block index must be non-negative")
        object.__setattr__(
            self,
            "completed_block",
            MappingProxyType(dict(self.completed_block)),
        )
        object.__setattr__(self, "usage", MappingProxyType(dict(self.usage)))
        if self.kind == "finish" and self.finish is None:
            raise ValueError("finish chunks require an LLMStreamFinish")
        if self.kind != "finish" and self.finish is not None:
            raise ValueError("only finish chunks may carry LLMStreamFinish")


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


class LLMToolStreamInvoker(Protocol):
    """Explicit-config native tool stream used by agentic runtimes."""

    def astream_tools(
        self,
        messages: Sequence[LLMToolHistoryMessage],
        *,
        tools: Sequence[LLMToolDefinition],
        config: LLMCallConfig,
    ) -> AsyncIterator[LLMStreamChunk]:
        """Stream one provider-neutral native-tool assistant turn."""


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

    def astream_tools(
        self,
        messages: Sequence[LLMToolHistoryMessage],
        *,
        tools: Sequence[LLMToolDefinition],
        config: LLMCallConfig,
        backend: BackendDescriptor,
    ) -> AsyncIterator[LLMStreamChunk]:
        """Stream a provider-neutral native-tool assistant turn."""

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
