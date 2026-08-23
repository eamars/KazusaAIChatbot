"""OpenAI-compatible provider adapter for chat models."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from typing import Literal

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGenerationChunk
from langchain_openai import ChatOpenAI
from openai import BadRequestError

from kazusa_ai_chatbot.llm_interface.contracts import (
    BackendDescriptor,
    LLMCallConfig,
    LLMResponse,
    LLMStreamChunk,
    LLMStreamFinish,
    LLMToolDefinition,
    LLMToolHistoryMessage,
)
from kazusa_ai_chatbot.llm_interface.reload import ReloadingChatModel

ChatModelFactory = Callable[..., object]
ChatModelCacheKey = tuple[object, ...]
_ProviderOutputTransport = Literal[
    "json_object",
    "json_schema",
    "text",
    "tools",
]
GEMMA4_THINKING_TRIGGER = "/think"
QWEN3_THINKING_PREFILL = "<think>\n"
_JSON_OBJECT_ALLOWED_MODE_ERROR = re.compile(
    r"response_format\.type['\"]?\s+must\s+be\s+"
    r"['\"]?json_schema['\"]?\s+or\s+['\"]?text['\"]?",
)
logger = logging.getLogger(__name__)


class _ReasoningAwareChatOpenAI(ChatOpenAI):
    """Preserve a provider reasoning delta omitted by ChatOpenAI conversion."""

    def _convert_chunk_to_generation_chunk(
        self,
        chunk: dict,
        default_chunk_class: type,
        base_generation_info: dict | None,
    ) -> ChatGenerationChunk | None:
        """Copy only non-empty raw reasoning content after base conversion."""

        generation_chunk = super()._convert_chunk_to_generation_chunk(
            chunk,
            default_chunk_class,
            base_generation_info,
        )
        if generation_chunk is None:
            return None

        choices = chunk.get("choices", [])
        if not choices:
            nested_chunk = chunk.get("chunk")
            if isinstance(nested_chunk, Mapping):
                choices = nested_chunk.get("choices", [])
        if not isinstance(choices, list) or not choices:
            return generation_chunk

        first_choice = choices[0]
        if not isinstance(first_choice, Mapping):
            return generation_chunk
        delta = first_choice.get("delta")
        if not isinstance(delta, Mapping):
            return generation_chunk
        reasoning_content = delta.get("reasoning_content")
        if (
            isinstance(reasoning_content, str)
            and reasoning_content
            and isinstance(generation_chunk.message, AIMessageChunk)
        ):
            generation_chunk.message.additional_kwargs["reasoning_content"] = (
                reasoning_content
            )
        return generation_chunk


class OpenAICompatibleProvider:
    """Map public LLM configs into OpenAI-compatible ChatOpenAI calls."""

    def __init__(
        self,
        *,
        chat_model_factory: ChatModelFactory = _ReasoningAwareChatOpenAI,
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

    async def astream_tools(
        self,
        messages: Sequence[LLMToolHistoryMessage],
        *,
        tools: Sequence[LLMToolDefinition],
        config: LLMCallConfig,
        backend: BackendDescriptor,
    ) -> AsyncIterator[LLMStreamChunk]:
        """Stream native tool chunks without applying resolver semantics."""

        chat_model = self._build_tool_chat_model(
            config=config,
            backend=backend,
            tools=tools,
        )
        provider_messages = _provider_tool_history_messages(
            messages,
            backend=backend,
        )
        started_blocks: dict[int, str] = {}
        finish_reason: str | None = None
        async for raw_chunk in chat_model.astream(provider_messages):
            reasoning_delta = _raw_reasoning_delta(raw_chunk)
            if reasoning_delta:
                if 0 not in started_blocks:
                    started_blocks[0] = "reasoning"
                    yield LLMStreamChunk(
                        kind="block_start",
                        block_index=0,
                        block_type="reasoning",
                    )
                yield LLMStreamChunk(
                    kind="reasoning_delta",
                    block_index=0,
                    block_type="reasoning",
                    reasoning_delta=reasoning_delta,
                )

            text_delta = _raw_text_delta(raw_chunk)
            if text_delta:
                if 1 not in started_blocks:
                    started_blocks[1] = "text"
                    yield LLMStreamChunk(
                        kind="block_start",
                        block_index=1,
                        block_type="text",
                    )
                yield LLMStreamChunk(
                    kind="text_delta",
                    block_index=1,
                    block_type="text",
                    text_delta=text_delta,
                )

            raw_tool_chunks = _raw_tool_call_chunks(raw_chunk)
            for fallback_index, raw_tool_chunk in enumerate(raw_tool_chunks):
                provider_index = _tool_chunk_index(
                    raw_tool_chunk,
                    fallback=fallback_index,
                )
                block_index = provider_index + 2
                if block_index not in started_blocks:
                    started_blocks[block_index] = "tool_call"
                    yield LLMStreamChunk(
                        kind="block_start",
                        block_index=block_index,
                        block_type="tool_call",
                    )
                yield LLMStreamChunk(
                    kind="tool_call_delta",
                    block_index=block_index,
                    block_type="tool_call",
                    tool_call_id=_optional_stream_text(
                        raw_tool_chunk.get("id")
                    ),
                    tool_name=_optional_stream_text(
                        raw_tool_chunk.get("name")
                    ),
                    tool_arguments_delta=_tool_arguments_delta(
                        raw_tool_chunk.get("args")
                    ),
                )

            usage = _raw_stream_usage(raw_chunk)
            if usage:
                yield LLMStreamChunk(kind="usage", usage=usage)
            raw_finish_reason = _raw_stream_finish_reason(raw_chunk)
            if raw_finish_reason:
                finish_reason = raw_finish_reason

        for block_index, block_type in sorted(started_blocks.items()):
            yield LLMStreamChunk(
                kind="block_end",
                block_index=block_index,
                block_type=block_type,
                completed_block={"type": block_type},
            )
        if finish_reason is None:
            stream_finish = LLMStreamFinish(
                reason="error",
                detail="provider stream closed without a finish reason",
            )
        else:
            stream_finish = LLMStreamFinish(
                reason=_normalized_finish_reason(finish_reason),
            )
        yield LLMStreamChunk(kind="finish", finish=stream_finish)

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

        kwargs = _chat_model_kwargs(
            config=config,
            backend=backend,
            output_transport=output_transport,
        )

        inner_model = self._chat_model_factory(**kwargs)
        chat_model = ReloadingChatModel(
            inner_model,
            base_url=config.base_url,
            model=config.model,
        )
        self._chat_models[cache_key] = chat_model
        return chat_model

    def _build_tool_chat_model(
        self,
        *,
        config: LLMCallConfig,
        backend: BackendDescriptor,
        tools: Sequence[LLMToolDefinition],
    ) -> ReloadingChatModel:
        """Build one tool-bound model partitioned by canonical schema digest."""

        schema_digest = _tool_schema_digest(tools)
        cache_key = _chat_model_cache_key(
            config=config,
            backend=backend,
            output_transport="tools",
            tool_schema_digest=schema_digest,
        )
        cached_model = self._chat_models.get(cache_key)
        if cached_model is not None:
            return cached_model

        kwargs = _chat_model_kwargs(
            config=config,
            backend=backend,
            output_transport="tools",
        )
        inner_model = self._chat_model_factory(**kwargs)
        native_tools = _native_tool_definitions(tools)
        bound_model = inner_model.bind_tools(
            native_tools,
            tool_choice="required",
            parallel_tool_calls=False,
        )
        chat_model = ReloadingChatModel(
            bound_model,
            base_url=config.base_url,
            model=config.model,
        )
        self._chat_models[cache_key] = chat_model
        return chat_model


def _chat_model_kwargs(
    *,
    config: LLMCallConfig,
    backend: BackendDescriptor,
    output_transport: _ProviderOutputTransport,
) -> dict[str, object]:
    """Build provider-native construction kwargs for one transport."""

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
        qwen_extra_body: dict[str, object] = {
            "chat_template_kwargs": {"enable_thinking": True},
        }
        if output_transport == "tools":
            qwen_extra_body["reasoning_format"] = "deepseek"
        kwargs["extra_body"] = qwen_extra_body
    if backend.thinking_strategy == "qwen3_disabled":
        kwargs["extra_body"] = {
            "chat_template_kwargs": {"enable_thinking": False},
        }
    return kwargs


def _native_tool_definitions(
    tools: Sequence[LLMToolDefinition],
) -> list[dict[str, object]]:
    """Project provider-neutral schemas into OpenAI native tool objects."""

    native_tools = [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": dict(tool.parameters),
            },
        }
        for tool in tools
    ]
    return native_tools


def _tool_schema_digest(tools: Sequence[LLMToolDefinition]) -> str:
    """Return the canonical cache partition for one exact tool roster."""

    canonical_tools = [
        {
            "name": tool.name,
            "description": tool.description,
            "parameters": dict(tool.parameters),
        }
        for tool in sorted(tools, key=lambda item: item.name)
    ]
    serialized = json.dumps(
        canonical_tools,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    return digest


def _provider_tool_history_messages(
    messages: Sequence[LLMToolHistoryMessage],
    *,
    backend: BackendDescriptor,
) -> Sequence[BaseMessage]:
    """Translate typed tool history and apply adapter-private thinking controls."""

    translated: list[BaseMessage] = []
    for message in messages:
        if message.role == "system":
            translated.append(SystemMessage(content=message.content))
            continue
        if message.role == "user":
            translated.append(HumanMessage(content=message.content))
            continue
        if message.role == "tool":
            translated.append(ToolMessage(
                content=message.content,
                tool_call_id=message.tool_call_id or "",
            ))
            continue

        native_calls = [
            {
                "id": tool_call.call_id,
                "name": tool_call.name,
                "args": dict(tool_call.arguments),
                "type": "tool_call",
            }
            for tool_call in message.tool_calls
        ]
        additional_kwargs: dict[str, object] = {}
        if native_calls and _requires_tool_reasoning_passback(backend):
            additional_kwargs["reasoning_content"] = message.reasoning or ""
        translated.append(AIMessage(
            content=message.content,
            tool_calls=native_calls,
            additional_kwargs=additional_kwargs,
        ))
    if backend.thinking_strategy == "qwen3_enabled":
        return translated

    provider_messages = _provider_messages(translated, backend=backend)
    return provider_messages


def _requires_tool_reasoning_passback(backend: BackendDescriptor) -> bool:
    """Return whether this provider model family requires tool-turn reasoning."""

    return_value = backend.model_family == "deepseek"
    return return_value


def _raw_reasoning_delta(raw_chunk: object) -> str:
    """Extract one opaque reasoning delta from known provider chunk shapes."""

    additional_kwargs = getattr(raw_chunk, "additional_kwargs", None)
    if isinstance(additional_kwargs, Mapping):
        for field_name in ("reasoning_content", "reasoning"):
            value = additional_kwargs.get(field_name)
            if isinstance(value, str):
                return value

    content_blocks = getattr(raw_chunk, "content_blocks", None)
    if isinstance(content_blocks, list):
        for block in content_blocks:
            if not isinstance(block, Mapping):
                continue
            if block.get("type") not in {"reasoning", "reasoning_content"}:
                continue
            for field_name in ("reasoning", "text", "content"):
                value = block.get(field_name)
                if isinstance(value, str):
                    return value
    return ""


def _raw_text_delta(raw_chunk: object) -> str:
    """Extract visible assistant text without including reasoning blocks."""

    content = getattr(raw_chunk, "content", "")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""

    fragments: list[str] = []
    for block in content:
        if isinstance(block, str):
            fragments.append(block)
            continue
        if not isinstance(block, Mapping):
            continue
        if block.get("type") not in {"text", "output_text"}:
            continue
        value = block.get("text")
        if isinstance(value, str):
            fragments.append(value)
    text_delta = "".join(fragments)
    return text_delta


def _raw_tool_call_chunks(raw_chunk: object) -> list[Mapping[str, object]]:
    """Return incremental native tool-call rows from known chunk shapes."""

    raw_chunks = getattr(raw_chunk, "tool_call_chunks", None)
    if isinstance(raw_chunks, list):
        normalized_chunks = [
            item for item in raw_chunks if isinstance(item, Mapping)
        ]
        if normalized_chunks:
            return normalized_chunks

    additional_kwargs = getattr(raw_chunk, "additional_kwargs", None)
    if not isinstance(additional_kwargs, Mapping):
        empty_chunks: list[Mapping[str, object]] = []
        return empty_chunks
    raw_calls = additional_kwargs.get("tool_calls")
    if not isinstance(raw_calls, list):
        empty_chunks = []
        return empty_chunks
    normalized_calls: list[Mapping[str, object]] = []
    for raw_call in raw_calls:
        if not isinstance(raw_call, Mapping):
            continue
        function = raw_call.get("function")
        if not isinstance(function, Mapping):
            continue
        normalized_calls.append({
            "id": raw_call.get("id"),
            "index": raw_call.get("index"),
            "name": function.get("name"),
            "args": function.get("arguments"),
        })
    return normalized_calls


def _tool_chunk_index(
    raw_tool_chunk: Mapping[str, object],
    *,
    fallback: int,
) -> int:
    """Return a non-negative provider tool index with a stable local fallback."""

    raw_index = raw_tool_chunk.get("index")
    try:
        provider_index = int(raw_index)
    except (TypeError, ValueError):
        provider_index = fallback
    if provider_index < 0:
        provider_index = fallback
    return provider_index


def _optional_stream_text(value: object) -> str | None:
    """Return a non-empty provider string or no value."""

    if not isinstance(value, str) or not value:
        return None
    return value


def _tool_arguments_delta(value: object) -> str:
    """Return one raw JSON argument fragment without decoding it."""

    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        arguments_delta = json.dumps(
            dict(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return arguments_delta
    return ""


def _raw_stream_usage(raw_chunk: object) -> Mapping[str, object]:
    """Extract provider-neutral usage counters from a stream chunk."""

    usage_metadata = getattr(raw_chunk, "usage_metadata", None)
    if isinstance(usage_metadata, Mapping):
        usage = dict(usage_metadata)
        return usage
    response_metadata = getattr(raw_chunk, "response_metadata", None)
    if isinstance(response_metadata, Mapping):
        for field_name in ("token_usage", "usage"):
            value = response_metadata.get(field_name)
            if isinstance(value, Mapping):
                usage = dict(value)
                return usage
    return_value: Mapping[str, object] = {}
    return return_value


def _raw_stream_finish_reason(raw_chunk: object) -> str:
    """Extract a provider finish reason without interpreting tool semantics."""

    response_metadata = getattr(raw_chunk, "response_metadata", None)
    if not isinstance(response_metadata, Mapping):
        return ""
    for field_name in ("finish_reason", "stop_reason"):
        value = response_metadata.get(field_name)
        if isinstance(value, str) and value:
            return value
    return ""


def _normalized_finish_reason(
    raw_reason: str,
) -> Literal["stop", "tool_calls", "max_tokens", "aborted", "error"]:
    """Map provider finish vocabulary into the closed stream contract."""

    normalized = raw_reason.strip().lower()
    if normalized in {"tool_calls", "tool_call"}:
        return "tool_calls"
    if normalized in {"length", "max_tokens", "max_token"}:
        return "max_tokens"
    if normalized in {"cancelled", "canceled", "aborted"}:
        return "aborted"
    if normalized in {"error", "failed"}:
        return "error"
    return "stop"


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
    tool_schema_digest: str = "",
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
        tool_schema_digest,
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
