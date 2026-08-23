"""Provider-neutral thinking-enabled native-tool model seam."""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal, Protocol

from agentic_resolver.contracts import AgenticResolverContractError


@dataclass(frozen=True)
class AgenticModelCapabilitiesV1:
    """Immutable evidence that a model supports the required transport."""

    thinking_strategy: str
    reasoning_replay_policy: str
    streaming: bool = True
    thinking_enabled: bool = True
    schema_version: str = field(
        default="agentic_model_capabilities.v1",
        init=False,
    )

    def __post_init__(self) -> None:
        if not self.streaming:
            raise AgenticResolverContractError(
                "agentic model requires streaming",
                code="unsupported_model_capability",
            )
        if not self.thinking_enabled:
            raise AgenticResolverContractError(
                "agentic model requires enabled thinking",
                code="unsupported_model_capability",
            )
        if not self.thinking_strategy.strip():
            raise AgenticResolverContractError(
                "agentic model requires a thinking strategy",
                code="unsupported_model_capability",
            )
        if not self.reasoning_replay_policy.strip():
            raise AgenticResolverContractError(
                "agentic model requires a reasoning replay policy",
                code="unsupported_model_capability",
            )


@dataclass(frozen=True)
class AgenticModelToolDefinition:
    """Provider-neutral native tool schema supplied to each model step."""

    name: str
    description: str
    parameters: Mapping[str, object]

    def __post_init__(self) -> None:
        if not self.name or not self.description:
            raise AgenticResolverContractError(
                "model tool name and description are required"
            )
        if self.parameters.get("type") != "object":
            raise AgenticResolverContractError(
                "model tool parameters must be an object schema"
            )
        object.__setattr__(
            self,
            "parameters",
            MappingProxyType(dict(self.parameters)),
        )

    def to_dict(self) -> dict[str, object]:
        """Return the canonical context-accounting projection."""

        value = {
            "name": self.name,
            "description": self.description,
            "parameters": dict(self.parameters),
        }
        return value


@dataclass(frozen=True)
class AgenticModelToolCall:
    """One complete model-native tool call accepted by the assembler."""

    call_id: str
    name: str
    arguments: Mapping[str, object]

    def __post_init__(self) -> None:
        if not self.call_id or not self.name:
            raise AgenticResolverContractError(
                "complete model tool calls require id and name"
            )
        object.__setattr__(
            self,
            "arguments",
            MappingProxyType(dict(self.arguments)),
        )

    def to_dict(self) -> dict[str, object]:
        """Return the provider-neutral history projection."""

        value = {
            "call_id": self.call_id,
            "name": self.name,
            "arguments": dict(self.arguments),
        }
        return value


@dataclass(frozen=True)
class AgenticInvalidToolCall:
    """One bounded invalid native tool call produced by assembly."""

    call_id: str | None
    name: str | None
    error: str


@dataclass(frozen=True)
class AgenticModelMessage:
    """Role-discriminated model history independent of any provider SDK."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str = ""
    reasoning: str | None = None
    tool_calls: tuple[AgenticModelToolCall, ...] = ()
    tool_call_id: str | None = None

    def __post_init__(self) -> None:
        if self.role in {"system", "user"}:
            if not self.content:
                raise AgenticResolverContractError(
                    f"{self.role} model content is required"
                )
            if self.reasoning is not None or self.tool_calls or self.tool_call_id:
                raise AgenticResolverContractError(
                    f"{self.role} model history has invalid fields"
                )
            return
        if self.role == "assistant":
            if self.tool_call_id is not None:
                raise AgenticResolverContractError(
                    "assistant history cannot carry tool_call_id"
                )
            return
        if not self.content or not self.tool_call_id:
            raise AgenticResolverContractError(
                "tool history requires content and tool_call_id"
            )
        if self.reasoning is not None or self.tool_calls:
            raise AgenticResolverContractError(
                "tool history cannot carry assistant fields"
            )

    def to_dict(self) -> dict[str, object]:
        """Return the canonical context-accounting projection."""

        value: dict[str, object] = {
            "role": self.role,
            "content": self.content,
        }
        if self.role == "assistant":
            value["reasoning"] = self.reasoning
            value["tool_calls"] = [
                tool_call.to_dict() for tool_call in self.tool_calls
            ]
        if self.role == "tool":
            value["tool_call_id"] = self.tool_call_id
        return value


@dataclass(frozen=True)
class ModelStreamFinish:
    """Terminal disposition for one provider-neutral model stream."""

    reason: Literal["stop", "tool_calls", "max_tokens", "aborted", "error"]
    detail: str = ""


@dataclass(frozen=True)
class ModelStreamChunk:
    """One provider-neutral indexed stream event consumed by the assembler."""

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
    finish: ModelStreamFinish | None = None

    def __post_init__(self) -> None:
        if self.block_index is not None and self.block_index < 0:
            raise AgenticResolverContractError(
                "model stream block index must be non-negative"
            )
        object.__setattr__(
            self,
            "completed_block",
            MappingProxyType(dict(self.completed_block)),
        )
        object.__setattr__(self, "usage", MappingProxyType(dict(self.usage)))
        if self.kind == "finish" and self.finish is None:
            raise AgenticResolverContractError(
                "finish stream event requires a terminal disposition"
            )
        if self.kind != "finish" and self.finish is not None:
            raise AgenticResolverContractError(
                "only finish stream events carry a terminal disposition"
            )


class AgenticModelClient(Protocol):
    """Thinking-enabled streaming model dependency for the resolver runtime."""

    @property
    def capabilities(self) -> AgenticModelCapabilitiesV1:
        """Return immutable provider-neutral transport capabilities."""

    def astream(
        self,
        messages: Sequence[AgenticModelMessage],
        *,
        tools: Sequence[AgenticModelToolDefinition],
    ) -> AsyncIterator[ModelStreamChunk]:
        """Stream one complete native-tool assistant turn."""
