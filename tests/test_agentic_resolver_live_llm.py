"""One inspected real-model workflow for the standalone resolver."""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator, Mapping, Sequence
from pathlib import Path
from typing import cast

import pytest

from agentic_resolver.contracts import (
    AgenticResolverContextV1,
    AgenticResolverLimitsV1,
    AgenticResolverRequestV1,
)
from agentic_resolver.integrations.llm_interface import LLInterfaceToolModel
from agentic_resolver.json_protocol import parse_json_object
from agentic_resolver.runtime import AgenticResolverRuntime
from agentic_resolver.session import ResolverSession
from agentic_resolver.skills import discover_skills
from agentic_resolver.tools import ToolDefinition, ToolRegistry
from kazusa_ai_chatbot.llm_interface import (
    LLInterface,
    LLMCallConfig,
    LLMStreamChunk,
    LLMThinkingConfig,
    LLMToolDefinition,
    LLMToolHistoryMessage,
)
from tests.llm_trace import write_llm_trace

pytestmark = [pytest.mark.asyncio, pytest.mark.live_llm]

_LIVE_BASE_URL_ENV = "AGENTIC_RESOLVER_LIVE_BASE_URL"
_LIVE_MODEL_ENV = "AGENTIC_RESOLVER_LIVE_MODEL"
_LIVE_API_KEY_ENV = "AGENTIC_RESOLVER_LIVE_API_KEY"
_PARENT_LITERAL = "PARENT-SEED-17"
_CHILD_LITERAL = "CHILD-SEED-29"


class _RecordingLLInterface:
    """Record thought-text-free normalized evidence around one real interface."""

    def __init__(self, wrapped: LLInterface) -> None:
        self._wrapped = wrapped
        self.calls: list[dict[str, object]] = []

    def describe_backend(self, *, config: LLMCallConfig):
        """Return the production backend descriptor unchanged."""

        backend = self._wrapped.describe_backend(config=config)
        return backend

    def astream_tools(
        self,
        messages: Sequence[LLMToolHistoryMessage],
        *,
        tools: Sequence[LLMToolDefinition],
        config: LLMCallConfig,
    ) -> AsyncIterator[LLMStreamChunk]:
        """Capture inputs and normalized chunk metadata around the real stream."""

        stream = self._recorded_stream(
            messages=messages,
            tools=tools,
            config=config,
        )
        return stream

    async def _recorded_stream(
        self,
        *,
        messages: Sequence[LLMToolHistoryMessage],
        tools: Sequence[LLMToolDefinition],
        config: LLMCallConfig,
    ) -> AsyncIterator[LLMStreamChunk]:
        """Yield a real call while recording only review-safe evidence."""

        record: dict[str, object] = {
            "messages": [_message_projection(message) for message in messages],
            "tool_roster": [tool.name for tool in tools],
            "chunks": [],
        }
        self.calls.append(record)
        chunks = cast(list[dict[str, object]], record["chunks"])
        async for chunk in self._wrapped.astream_tools(
            messages,
            tools=tools,
            config=config,
        ):
            chunks.append(_chunk_projection(chunk))
            yield chunk

    async def aclose(self) -> None:
        """Close the wrapped production interface."""

        await self._wrapped.aclose()


def _message_projection(message: LLMToolHistoryMessage) -> dict[str, object]:
    """Project one model input without copying opaque reasoning text."""

    semantic_content: dict[str, object] | None = None
    if message.content:
        semantic_content = parse_json_object(message.content)
    projection = {
        "role": message.role,
        "semantic_content": semantic_content,
        "reasoning_present": message.reasoning is not None,
        "reasoning_characters": len(message.reasoning or ""),
        "tool_call_id": message.tool_call_id,
        "tool_calls": [
            {
                "call_id": tool_call.call_id,
                "name": tool_call.name,
                "arguments": dict(tool_call.arguments),
            }
            for tool_call in message.tool_calls
        ],
    }
    return projection


def _chunk_projection(chunk: LLMStreamChunk) -> dict[str, object]:
    """Project ordering and sizes without retaining model thought text."""

    finish_reason = None
    if chunk.finish is not None:
        finish_reason = chunk.finish.reason
    projection = {
        "kind": chunk.kind,
        "block_index": chunk.block_index,
        "block_type": chunk.block_type,
        "reasoning_characters": len(chunk.reasoning_delta),
        "text_characters": len(chunk.text_delta),
        "tool_call_id": chunk.tool_call_id,
        "tool_name": chunk.tool_name,
        "tool_argument_characters": len(chunk.tool_arguments_delta),
        "usage": dict(chunk.usage),
        "finish_reason": finish_reason,
    }
    return projection


def _session_projection(session: ResolverSession) -> dict[str, object]:
    """Project one root or child session into review-safe structured evidence."""

    history = []
    for message in session.model_history():
        semantic_content = None
        if message.content:
            semantic_content = parse_json_object(message.content)
        history.append({
            "role": message.role,
            "semantic_content": semantic_content,
            "reasoning_present": message.reasoning is not None,
            "reasoning_characters": len(message.reasoning or ""),
            "tool_call_id": message.tool_call_id,
            "tool_calls": [
                {
                    "call_id": tool_call.call_id,
                    "name": tool_call.name,
                    "arguments": dict(tool_call.arguments),
                }
                for tool_call in message.tool_calls
            ],
        })
    projection = {
        "session_id": session.session_id,
        "parent_session_id": session.parent_session_id,
        "depth": session.depth,
        "loaded_skills": sorted(session.loaded_skills),
        "usage": session.usage.to_dict(),
        "events": [
            {
                "index": event.index,
                "kind": event.kind,
                "metadata": dict(event.metadata),
            }
            for event in session.events
        ],
        "history": history,
        "observations": [
            {
                "observation_id": observation.observation_id,
                "tool_name": observation.tool_name,
                "status": observation.status,
                "summary": observation.summary,
                "evidence_refs": list(observation.evidence_refs),
            }
            for observation in session.observations.values()
        ],
    }
    return projection


async def test_standalone_resolver_streams_thinking_tool_and_subagent_then_submits_json_result(
    tmp_path: Path,
) -> None:
    """A real thinking model should complete the bounded root-child workflow."""

    required_environment = (
        _LIVE_BASE_URL_ENV,
        _LIVE_MODEL_ENV,
        _LIVE_API_KEY_ENV,
    )
    missing_environment = [
        name for name in required_environment if not os.environ.get(name)
    ]
    if missing_environment:
        pytest.skip(
            "live resolver route requires explicit environment: "
            + ", ".join(missing_environment)
        )

    skill_bundle = tmp_path / "resolver-verification"
    skill_bundle.mkdir()
    (skill_bundle / "SKILL.md").write_text(
        "---\n"
        "name: resolver-verification\n"
        "description: Load this skill before root and delegated-child fixture execution.\n"
        "---\n\n"
        "# Resolver Verification\n\n"
        "Read the task message's `message_type` before acting. This skill is "
        "required in both the root and delegated-child sessions. For a root "
        "`task`, call `read_fixture_fact` with `parent_seed`, then delegate "
        "one self-contained child task whose objective and context explicitly "
        "require loading `resolver-verification` first, then calling "
        "`read_fixture_fact` with `child_seed`, and then using `submit_result`. "
        "For a `subagent_task`, load this skill first, make exactly one ordinary "
        "`read_fixture_fact` call with `child_seed`, and then submit that literal "
        "without repeating the root procedure. The root must integrate both "
        "literals and use `submit_result`. Each model step has exactly one "
        "native tool call; the allowed skill and terminal calls are separate "
        "from the one ordinary fact-read call. "
        "Keep assistant text empty while calling tools. A child must never "
        "try to delegate another child.\n",
        encoding="utf-8",
    )
    execution_keys: list[str] = []

    async def _read_fixture_fact(
        arguments: Mapping[str, object],
    ) -> object:
        key = cast(str, arguments["key"])
        execution_keys.append(key)
        literal_by_key = {
            "parent_seed": _PARENT_LITERAL,
            "child_seed": _CHILD_LITERAL,
        }
        literal = literal_by_key[key]
        output = {
            "summary": f"Fixture {key} is {literal}.",
            "provenance_refs": [f"fixture:{key}"],
            "literal": literal,
        }
        return output

    fixture_tool = ToolDefinition(
        name="read_fixture_fact",
        description=(
            "Read one synthetic verification fact. The root reads parent_seed; "
            "the delegated child reads child_seed."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "key": {
                    "type": "string",
                    "enum": ["parent_seed", "child_seed"],
                },
            },
            "required": ["key"],
            "additionalProperties": False,
        },
        execute=_read_fixture_fact,
    )
    config = LLMCallConfig(
        stage_name="agentic_resolver_live_verification",
        route_name="AGENTIC_RESOLVER_LIVE",
        base_url=os.environ[_LIVE_BASE_URL_ENV],
        api_key=os.environ[_LIVE_API_KEY_ENV],
        model=os.environ[_LIVE_MODEL_ENV],
        temperature=None,
        top_p=None,
        top_k=None,
        max_completion_tokens=4_096,
        presence_penalty=None,
        timeout_seconds=180,
        thinking=LLMThinkingConfig(enabled=True),
        output_mode="text",
        context_window_tokens=50_000,
    )
    interface = LLInterface()
    recording_interface = _RecordingLLInterface(interface)
    model = LLInterfaceToolModel(
        llm_interface=cast(LLInterface, recording_interface),
        llm_config=config,
    )
    limits = AgenticResolverLimitsV1(
        context_window_tokens=50_000,
        completion_reserve_tokens=8_000,
        max_model_steps=8,
        max_tool_calls=6,
        max_subagent_runs=1,
        session_timeout_seconds=480,
        tool_timeout_seconds=30,
    )
    runtime = AgenticResolverRuntime(
        model=model,
        tools=ToolRegistry([fixture_tool]),
        skills=discover_skills([tmp_path], limits=limits),
        limits=limits,
        permission_scope={"fixture_read": True},
    )
    request = AgenticResolverRequestV1(
        objective=(
            "Resolve the standalone integration fixture by obtaining the "
            "root-owned parent seed, delegating the child-owned seed lookup, "
            "and submitting one integrated result."
        ),
        context=AgenticResolverContextV1(
            facts=(
                "The parent seed must be obtained by the root session.",
                "The child seed must be obtained inside one delegated child.",
            ),
            constraints=(
                "Load the resolver-verification skill before task actions.",
                "The root calls read_fixture_fact with parent_seed first.",
                "The root then delegates exactly one self-contained child.",
                "The child loads resolver-verification before its own task action.",
                "The child calls read_fixture_fact with child_seed.",
                "The root summary must preserve both returned literals.",
            ),
            desired_output=(
                "A resolved typed result preserving both fixture literals."
            ),
        ),
    )

    result = await runtime.resolve(request)
    await recording_interface.aclose()

    root_session = runtime._sessions[result.session_id]
    child_sessions = [
        session
        for session in runtime._sessions.values()
        if session.parent_session_id == root_session.session_id
    ]
    session_projections = [
        _session_projection(session)
        for session in runtime._sessions.values()
    ]
    trace_path = write_llm_trace(
        "test_agentic_resolver_live_llm",
        "standalone_root_child_stream",
        {
            "input_source": "synthetic bounded integration fixture",
            "route": {
                "base_url": config.base_url,
                "model": config.model,
                "thinking_strategy": model.capabilities.thinking_strategy,
                "reasoning_replay_policy": (
                    model.capabilities.reasoning_replay_policy
                ),
            },
            "request": request.to_dict(),
            "execution_keys": execution_keys,
            "model_stream_calls": recording_interface.calls,
            "sessions": session_projections,
            "public_result": result.to_dict(),
            "privacy": {
                "credential_recorded": False,
                "thought_text_recorded": False,
                "reasoning_metadata_only": True,
            },
        },
    )
    print(f"AGENTIC_RESOLVER_LIVE_TRACE={trace_path}")

    assert result.status == "resolved"
    assert _PARENT_LITERAL in result.summary
    assert _CHILD_LITERAL in result.summary
    assert execution_keys == ["parent_seed", "child_seed"]
    assert len(child_sessions) == 1
    child_session = child_sessions[0]
    assert root_session.loaded_skills == {"resolver-verification"}
    assert child_session.loaded_skills == {"resolver-verification"}
    assert [
        observation.tool_name
        for observation in root_session.observations.values()
    ] == ["read_fixture_fact", "run_subagent"]
    assert [
        observation.tool_name
        for observation in child_session.observations.values()
    ] == ["read_fixture_fact"]
    assert result.usage.subagent_runs == 1
    assert len(recording_interface.calls) == (
        root_session.usage.model_steps + child_session.usage.model_steps
    )
    reasoning_counts = [
        cast(int, chunk["reasoning_characters"])
        for call in recording_interface.calls
        for chunk in cast(list[dict[str, object]], call["chunks"])
        if chunk["kind"] == "reasoning_delta"
    ]
    assert reasoning_counts
    assert sum(reasoning_counts) > 0
    child_task = parse_json_object(child_session.model_history()[2].content)
    assert child_task["message_type"] == "subagent_task"
    assert _PARENT_LITERAL not in child_session.model_history()[2].content
    child_call_rosters = [
        cast(list[str], call["tool_roster"])
        for call in recording_interface.calls
        if any(
            cast(dict[str, object], message["semantic_content"]).get(
                "message_type"
            ) == "subagent_task"
            for message in cast(list[dict[str, object]], call["messages"])
            if isinstance(message["semantic_content"], dict)
        )
    ]
    assert child_call_rosters
    assert all("run_subagent" not in roster for roster in child_call_rosters)
    parent_subagent_results = [
        cast(dict[str, object], message["semantic_content"])
        for call in recording_interface.calls
        for message in cast(list[dict[str, object]], call["messages"])
        if isinstance(message["semantic_content"], dict)
        and message["semantic_content"].get("message_type")
        == "subagent_result"
    ]
    assert len(parent_subagent_results) == 1
    parent_subagent_result_json = json.dumps(
        parent_subagent_results[0],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    child_observation_ids = tuple(child_session.observations)
    assert child_observation_ids
    assert all(
        observation_id not in parent_subagent_result_json
        for observation_id in child_observation_ids
    )
    public_json = json.dumps(result.to_dict(), ensure_ascii=False)
    assert "reasoning" not in public_json
    artifact_text = trace_path.read_text(encoding="utf-8")
    artifact = json.loads(artifact_text)
    artifact_calls = artifact["payload"]["model_stream_calls"]
    assert all(
        "reasoning_delta" not in chunk
        for call in artifact_calls
        for chunk in call["chunks"]
    )
    assert "api_key" not in artifact_text
