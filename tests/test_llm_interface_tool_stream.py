"""Deterministic coverage for additive native-tool streaming."""

from __future__ import annotations

from dataclasses import replace

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage

from kazusa_ai_chatbot.llm_interface import (
    BackendDescriptor,
    LLInterface,
    LLMCallConfig,
    LLMStreamChunk,
    LLMThinkingConfig,
    LLMToolCall,
    LLMToolDefinition,
    LLMToolHistoryMessage,
)
from kazusa_ai_chatbot.llm_interface.providers.openai_compatible import (
    OpenAICompatibleProvider,
    _ReasoningAwareChatOpenAI,
)


def _config() -> LLMCallConfig:
    """Build one thinking-enabled provider config for stream tests."""

    config = LLMCallConfig(
        stage_name="tests.tool_stream",
        route_name="TEST_TOOL_STREAM_LLM",
        base_url="http://localhost:1234/v1",
        api_key="test-key",
        model="qwen3.6-27b",
        temperature=0.1,
        top_p=0.8,
        top_k=None,
        max_completion_tokens=8000,
        presence_penalty=None,
        thinking=LLMThinkingConfig(enabled=True),
        output_mode="text",
    )
    return config


def _backend(
    *,
    model_family: str = "qwen",
    thinking_strategy: str = "qwen3_enabled",
) -> BackendDescriptor:
    """Build one fixed backend descriptor for provider-only tests."""

    descriptor = BackendDescriptor(
        route_name="TEST_TOOL_STREAM_LLM",
        backend_kind="openai_compatible",
        model_family=model_family,
        model="qwen3.6-27b",
        normalized_base_url="http://localhost:1234/v1",
        thinking_strategy=thinking_strategy,
        confidence="model_name_inferred",
        generation=1,
    )
    return descriptor


def _tool(name: str = "lookup") -> LLMToolDefinition:
    """Build one strict native tool schema."""

    tool = LLMToolDefinition(
        name=name,
        description="Look up one bounded value.",
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
            "additionalProperties": False,
        },
    )
    return tool


def _history() -> list[LLMToolHistoryMessage]:
    """Build one JSON-only system and task history."""

    history = [
        LLMToolHistoryMessage(
            role="system",
            content='{"schema_version":"system.v1"}',
        ),
        LLMToolHistoryMessage(
            role="user",
            content='{"schema_version":"task.v1"}',
        ),
    ]
    return history


def _stream_chunks() -> list[AIMessageChunk]:
    """Build an interleaved reasoning/tool stream with usage and finish."""

    chunks = [
        AIMessageChunk(
            content="",
            additional_kwargs={"reasoning_content": "private reasoning"},
        ),
        AIMessageChunk(
            content="",
            tool_call_chunks=[{
                "name": "lookup",
                "args": '{"query":"',
                "id": "call-1",
                "index": 0,
                "type": "tool_call_chunk",
            }],
        ),
        AIMessageChunk(
            content="",
            tool_call_chunks=[{
                "name": None,
                "args": 'value"}',
                "id": None,
                "index": 0,
                "type": "tool_call_chunk",
            }],
        ),
        AIMessageChunk(
            content="",
            usage_metadata={
                "input_tokens": 10,
                "output_tokens": 4,
                "total_tokens": 14,
            },
            response_metadata={"finish_reason": "tool_calls"},
        ),
    ]
    return chunks


class _BoundStreamModel:
    """Tool-bound fake that records provider history and yields chunks."""

    def __init__(self, chunks: list[AIMessageChunk]) -> None:
        self._chunks = chunks
        self.calls: list[list[object]] = []

    async def astream(self, messages: list[object]):
        """Yield one scripted provider stream."""

        self.calls.append(messages)
        for chunk in self._chunks:
            yield chunk


class _ChatModel:
    """Provider factory fake with ordinary and tool-bound call surfaces."""

    def __init__(
        self,
        *,
        chunks: list[AIMessageChunk],
        constructor_kwargs: dict[str, object],
    ) -> None:
        self.constructor_kwargs = constructor_kwargs
        self.bound_tools: list[list[dict[str, object]]] = []
        self.bind_tool_kwargs: list[dict[str, object]] = []
        self.bound_model = _BoundStreamModel(chunks)

    def bind_tools(
        self,
        tools: list[dict[str, object]],
        **kwargs: object,
    ) -> _BoundStreamModel:
        """Record native schemas and return the stream-capable bound model."""

        self.bound_tools.append(tools)
        self.bind_tool_kwargs.append(kwargs)
        return self.bound_model

    async def ainvoke(self, messages: list[object]) -> AIMessage:
        """Retain the ordinary async call shape for preservation testing."""

        del messages
        response = AIMessage(content="ordinary")
        return response


def _provider_with_models(
    created_models: list[_ChatModel],
) -> OpenAICompatibleProvider:
    """Build a provider whose factory exposes constructed model state."""

    def _factory(**kwargs: object) -> _ChatModel:
        model = _ChatModel(
            chunks=_stream_chunks(),
            constructor_kwargs=kwargs,
        )
        created_models.append(model)
        return model

    provider = OpenAICompatibleProvider(chat_model_factory=_factory)
    return provider


def test_chat_model_preserves_raw_reasoning_content_only() -> None:
    """The provider-local converter retains only raw reasoning content."""

    model = _ReasoningAwareChatOpenAI(
        model="qwen3.6-27b",
        base_url="http://localhost:1234/v1",
        api_key="test-key",
    )
    generation = model._convert_chunk_to_generation_chunk(
        {
            "choices": [{
                "delta": {
                    "role": "assistant",
                    "content": "",
                    "reasoning_content": "private reasoning",
                    "reasoning_details": {"ignored": True},
                },
            }],
        },
        AIMessageChunk,
        {},
    )

    assert generation is not None
    assert isinstance(generation.message, AIMessageChunk)
    assert generation.message.additional_kwargs == {
        "reasoning_content": "private reasoning",
    }


def test_tool_stream_contracts_keep_reasoning_distinct_from_json_content() -> None:
    """Opaque reasoning has a separate typed field from semantic content."""

    message = LLMToolHistoryMessage(
        role="assistant",
        content='{"schema_version":"assistant_note.v1"}',
        reasoning="private reasoning",
        tool_calls=(
            LLMToolCall(
                call_id="call-1",
                name="lookup",
                arguments={"query": "value"},
            ),
        ),
    )

    assert message.reasoning == "private reasoning"
    assert "private reasoning" not in message.content


def test_tool_call_deltas_are_indexed_and_arguments_remain_raw_until_assembly() -> None:
    """Incremental arguments retain raw fragments and explicit block identity."""

    chunk = LLMStreamChunk(
        kind="tool_call_delta",
        block_index=4,
        block_type="tool_call",
        tool_call_id="call-1",
        tool_name="lookup",
        tool_arguments_delta='{"query":"',
    )

    assert chunk.block_index == 4
    assert chunk.tool_arguments_delta == '{"query":"'


@pytest.mark.asyncio
async def test_existing_ainvoke_contract_remains_unchanged() -> None:
    """Ordinary async invocation still returns the normalized response contract."""

    created_models: list[_ChatModel] = []
    provider = _provider_with_models(created_models)
    response = await provider.ainvoke(
        [HumanMessage(content="ordinary")],
        config=replace(_config(), output_mode="text"),
        backend=_backend(),
    )

    assert response.content == "ordinary"
    assert response.backend == _backend()


@pytest.mark.asyncio
async def test_astream_tools_preserves_reasoning_tool_arguments_and_usage() -> None:
    """The provider emits separate normalized stream event families."""

    created_models: list[_ChatModel] = []
    provider = _provider_with_models(created_models)

    chunks = [
        chunk
        async for chunk in provider.astream_tools(
            _history(),
            tools=[_tool()],
            config=_config(),
            backend=_backend(),
        )
    ]

    assert any(
        chunk.kind == "reasoning_delta"
        and chunk.reasoning_delta == "private reasoning"
        for chunk in chunks
    )
    argument_fragments = [
        chunk.tool_arguments_delta
        for chunk in chunks
        if chunk.kind == "tool_call_delta"
    ]
    assert "".join(argument_fragments) == '{"query":"value"}'
    assert any(
        chunk.kind == "usage" and chunk.usage["total_tokens"] == 14
        for chunk in chunks
    )
    assert chunks[-1].finish is not None
    assert chunks[-1].finish.reason == "tool_calls"


@pytest.mark.asyncio
async def test_astream_tools_requires_one_tool_call_and_disables_parallel_tool_calls() -> None:
    """Native binding matches the serialized one-call loop contract."""

    created_models: list[_ChatModel] = []
    provider = _provider_with_models(created_models)

    _ = [
        chunk
        async for chunk in provider.astream_tools(
            _history(),
            tools=[_tool()],
            config=_config(),
            backend=_backend(),
        )
    ]

    assert created_models[0].bind_tool_kwargs == [
        {
            "tool_choice": "required",
            "parallel_tool_calls": False,
        },
    ]


@pytest.mark.asyncio
async def test_qwen_tool_stream_requests_deepseek_reasoning_format() -> None:
    """Qwen native tools request streamed reasoning-content extraction."""

    created_models: list[_ChatModel] = []
    provider = _provider_with_models(created_models)

    _ = [
        chunk
        async for chunk in provider.astream_tools(
            _history(),
            tools=[_tool()],
            config=_config(),
            backend=_backend(),
        )
    ]

    assert created_models[0].constructor_kwargs["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": True},
        "reasoning_format": "deepseek",
    }


@pytest.mark.asyncio
async def test_qwen_tool_stream_does_not_append_legacy_thinking_prefill() -> None:
    """Native Qwen tool history relies on endpoint thinking generation."""

    created_models: list[_ChatModel] = []
    provider = _provider_with_models(created_models)

    _ = [
        chunk
        async for chunk in provider.astream_tools(
            _history(),
            tools=[_tool()],
            config=_config(),
            backend=_backend(),
        )
    ]

    sent_messages = created_models[0].bound_model.calls[0]
    assert not any(
        isinstance(message, AIMessage) and message.content == "<think>\n"
        for message in sent_messages
    )


@pytest.mark.asyncio
async def test_non_qwen_tool_stream_keeps_existing_thinking_payload() -> None:
    """Non-Qwen native tools retain their existing provider payload."""

    created_models: list[_ChatModel] = []
    provider = _provider_with_models(created_models)

    _ = [
        chunk
        async for chunk in provider.astream_tools(
            _history(),
            tools=[_tool()],
            config=_config(),
            backend=_backend(
                model_family="gemma4",
                thinking_strategy="gemma4_enabled",
            ),
        )
    ]

    assert created_models[0].constructor_kwargs["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": True},
    }


@pytest.mark.asyncio
async def test_astream_tools_uses_distinct_tool_schema_cache_key() -> None:
    """Changing the native tool roster creates a separate bound model cache."""

    created_models: list[_ChatModel] = []
    provider = _provider_with_models(created_models)
    for tool in (_tool("lookup"), _tool("calculate"), _tool("lookup")):
        _ = [
            chunk
            async for chunk in provider.astream_tools(
                _history(),
                tools=[tool],
                config=_config(),
                backend=_backend(),
            )
        ]

    assert len(created_models) == 2


@pytest.mark.asyncio
async def test_astream_tools_replays_tool_call_reasoning_and_drops_ignored_tool_free_reasoning() -> None:
    """DeepSeek history replays only reasoning attached to tool-call turns."""

    created_models: list[_ChatModel] = []
    provider = _provider_with_models(created_models)
    history = [
        *_history(),
        LLMToolHistoryMessage(
            role="assistant",
            reasoning="tool reasoning",
            tool_calls=(LLMToolCall(
                call_id="call-1",
                name="lookup",
                arguments={"query": "value"},
            ),),
        ),
        LLMToolHistoryMessage(
            role="tool",
            content='{"status":"success"}',
            tool_call_id="call-1",
        ),
        LLMToolHistoryMessage(
            role="assistant",
            content='{"schema_version":"note.v1"}',
            reasoning="tool-free reasoning",
        ),
        LLMToolHistoryMessage(
            role="user",
            content='{"schema_version":"continue.v1"}',
        ),
    ]

    _ = [
        chunk
        async for chunk in provider.astream_tools(
            history,
            tools=[_tool()],
            config=_config(),
            backend=_backend(
                model_family="deepseek",
                thinking_strategy="deepseek_enabled",
            ),
        )
    ]
    sent_messages = created_models[0].bound_model.calls[0]
    assistant_messages = [
        message for message in sent_messages if isinstance(message, AIMessage)
    ]

    assert assistant_messages[0].additional_kwargs["reasoning_content"] == (
        "tool reasoning"
    )
    assert "reasoning_content" not in assistant_messages[1].additional_kwargs


@pytest.mark.asyncio
async def test_astream_tools_preserves_required_empty_reasoning_field_for_tool_call_turn() -> None:
    """A required native replay field remains present when reasoning is empty."""

    created_models: list[_ChatModel] = []
    provider = _provider_with_models(created_models)
    history = [
        *_history(),
        LLMToolHistoryMessage(
            role="assistant",
            tool_calls=(LLMToolCall(
                call_id="call-1",
                name="lookup",
                arguments={"query": "value"},
            ),),
        ),
        LLMToolHistoryMessage(
            role="tool",
            content='{"status":"success"}',
            tool_call_id="call-1",
        ),
    ]

    _ = [
        chunk
        async for chunk in provider.astream_tools(
            history,
            tools=[_tool()],
            config=_config(),
            backend=_backend(
                model_family="deepseek",
                thinking_strategy="deepseek_enabled",
            ),
        )
    ]
    sent_messages = created_models[0].bound_model.calls[0]
    assistant_message = next(
        message for message in sent_messages if isinstance(message, AIMessage)
    )

    assert assistant_message.additional_kwargs["reasoning_content"] == ""


@pytest.mark.asyncio
async def test_astream_tools_does_not_set_json_response_format() -> None:
    """Native tool arguments replace JSON-object response transport."""

    created_models: list[_ChatModel] = []
    provider = _provider_with_models(created_models)

    _ = [
        chunk
        async for chunk in provider.astream_tools(
            _history(),
            tools=[_tool()],
            config=_config(),
            backend=_backend(),
        )
    ]

    assert "model_kwargs" not in created_models[0].constructor_kwargs


def test_native_tool_stream_contracts_are_public_exports() -> None:
    """Callers can import the complete additive tool stream contract."""

    from kazusa_ai_chatbot import llm_interface

    exported_names = set(llm_interface.__all__)

    assert {
        "LLMInvalidToolCall",
        "LLMStreamChunk",
        "LLMStreamFinish",
        "LLMToolCall",
        "LLMToolDefinition",
        "LLMToolHistoryMessage",
        "LLMToolStreamInvoker",
    } <= exported_names


def test_agentic_adapter_maps_reasoning_and_tool_chunks_without_provider_objects() -> None:
    """The optional adapter maps stream contracts without native object leakage."""

    from agentic_resolver.integrations.llm_interface import LLInterfaceToolModel

    assert LLInterfaceToolModel.__module__.startswith("agentic_resolver.integrations")


def test_agentic_adapter_preserves_reasoning_as_typed_assistant_history() -> None:
    """The optional adapter keeps reasoning in its dedicated history field."""

    from agentic_resolver.model import AgenticModelMessage

    message = AgenticModelMessage(
        role="assistant",
        content="",
        reasoning="private",
        tool_calls=(),
    )

    assert message.reasoning == "private"
    assert message.content == ""


def test_agentic_adapter_requires_supported_thinking_config() -> None:
    """Construction rejects a route without enabled supported thinking."""

    from agentic_resolver.contracts import AgenticResolverContractError
    from agentic_resolver.integrations.llm_interface import LLInterfaceToolModel

    with pytest.raises(AgenticResolverContractError):
        LLInterfaceToolModel(
            llm_interface=LLInterface(),
            llm_config=replace(
                _config(),
                model="qwen2.5-32b",
            ),
        )
