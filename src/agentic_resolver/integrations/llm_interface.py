"""LLInterface adapter for the provider-neutral agentic model protocol."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence

from agentic_resolver.contracts import AgenticResolverContractError
from agentic_resolver.model import (
    AgenticModelCapabilitiesV1,
    AgenticModelMessage,
    AgenticModelToolDefinition,
    ModelStreamChunk,
    ModelStreamFinish,
)
from kazusa_ai_chatbot.llm_interface import (
    LLInterface,
    LLMCallConfig,
    LLMStreamChunk,
    LLMToolCall,
    LLMToolDefinition,
    LLMToolHistoryMessage,
)


class LLInterfaceToolModel:
    """Map LLInterface native streams into the standalone model seam."""

    def __init__(
        self,
        *,
        llm_interface: LLInterface,
        llm_config: LLMCallConfig,
    ) -> None:
        backend = llm_interface.describe_backend(config=llm_config)
        if not llm_config.thinking.enabled:
            raise AgenticResolverContractError(
                "resolver LLInterface config requires thinking enabled",
                code="unsupported_model_capability",
            )
        if not backend.thinking_strategy.endswith("_enabled"):
            raise AgenticResolverContractError(
                "resolver LLInterface backend lacks supported enabled thinking",
                code="unsupported_model_capability",
            )
        replay_policy = (
            "tool_call_turns_required"
            if backend.model_family == "deepseek"
            else "adapter_owned"
        )
        self._llm_interface = llm_interface
        self._llm_config = llm_config
        self._capabilities = AgenticModelCapabilitiesV1(
            thinking_strategy=backend.thinking_strategy,
            reasoning_replay_policy=replay_policy,
        )

    @property
    def capabilities(self) -> AgenticModelCapabilitiesV1:
        """Return immutable proof of streaming and enabled thinking."""

        return self._capabilities

    async def astream(
        self,
        messages: Sequence[AgenticModelMessage],
        *,
        tools: Sequence[AgenticModelToolDefinition],
    ) -> AsyncIterator[ModelStreamChunk]:
        """Map one LLInterface stream without exposing provider-native values."""

        llm_messages = tuple(_llm_history_message(message) for message in messages)
        llm_tools = tuple(
            LLMToolDefinition(
                name=tool.name,
                description=tool.description,
                parameters=tool.parameters,
            )
            for tool in tools
        )
        async for chunk in self._llm_interface.astream_tools(
            llm_messages,
            tools=llm_tools,
            config=self._llm_config,
        ):
            mapped_chunk = _model_stream_chunk(chunk)
            yield mapped_chunk


def _llm_history_message(
    message: AgenticModelMessage,
) -> LLMToolHistoryMessage:
    """Map one core history row into the shared LLInterface contract."""

    tool_calls = tuple(
        LLMToolCall(
            call_id=tool_call.call_id,
            name=tool_call.name,
            arguments=tool_call.arguments,
        )
        for tool_call in message.tool_calls
    )
    llm_message = LLMToolHistoryMessage(
        role=message.role,
        content=message.content,
        reasoning=message.reasoning,
        tool_calls=tool_calls,
        tool_call_id=message.tool_call_id,
    )
    return llm_message


def _model_stream_chunk(chunk: LLMStreamChunk) -> ModelStreamChunk:
    """Map one normalized shared chunk into the provider-neutral core type."""

    finish = None
    if chunk.finish is not None:
        finish = ModelStreamFinish(
            reason=chunk.finish.reason,
            detail=chunk.finish.detail,
        )
    mapped = ModelStreamChunk(
        kind=chunk.kind,
        block_index=chunk.block_index,
        block_type=chunk.block_type,
        reasoning_delta=chunk.reasoning_delta,
        text_delta=chunk.text_delta,
        tool_call_id=chunk.tool_call_id,
        tool_name=chunk.tool_name,
        tool_arguments_delta=chunk.tool_arguments_delta,
        completed_block=chunk.completed_block,
        usage=chunk.usage,
        finish=finish,
    )
    return mapped
