"""Deterministic serialized resolver-loop tests."""

from __future__ import annotations

import json
from collections.abc import (
    AsyncIterator,
    Awaitable,
    Callable,
    Mapping,
    Sequence,
)

import pytest

from agentic_resolver.contracts import (
    AgenticResolverLimitsV1,
    AgenticResolverRequestV1,
)
from agentic_resolver.json_protocol import parse_json_object
from agentic_resolver.model import (
    AgenticModelCapabilitiesV1,
    AgenticModelMessage,
    AgenticModelToolDefinition,
    ModelStreamChunk,
    ModelStreamFinish,
)
from agentic_resolver.runtime import AgenticResolverRuntime
from agentic_resolver.skills import discover_skills
from agentic_resolver.tools import ToolDefinition, ToolRegistry


class _ScriptedModel:
    """Yield complete scripted streams while retaining provider inputs."""

    def __init__(self, responses: Sequence[Sequence[ModelStreamChunk]]) -> None:
        self.capabilities = AgenticModelCapabilitiesV1(
            thinking_strategy="test_thinking",
            reasoning_replay_policy="test_replay",
        )
        self._responses = list(responses)
        self.calls: list[
            tuple[
                tuple[AgenticModelMessage, ...],
                tuple[AgenticModelToolDefinition, ...],
            ]
        ] = []
        self.stream_completed: list[bool] = []

    async def astream(
        self,
        messages: Sequence[AgenticModelMessage],
        *,
        tools: Sequence[AgenticModelToolDefinition],
    ) -> AsyncIterator[ModelStreamChunk]:
        """Yield the next response and mark completion after its final chunk."""

        if not self._responses:
            raise AssertionError("scripted model exhausted")
        self.calls.append((tuple(messages), tuple(tools)))
        response = self._responses.pop(0)
        stream_index = len(self.stream_completed)
        self.stream_completed.append(False)
        for chunk in response:
            yield chunk
        self.stream_completed[stream_index] = True


class _EvidenceModel:
    """Choose evidence arguments from the current typed model history."""

    def __init__(self) -> None:
        self.capabilities = AgenticModelCapabilitiesV1(
            thinking_strategy="test_thinking",
            reasoning_replay_policy="test_replay",
        )
        self.calls: list[
            tuple[
                tuple[AgenticModelMessage, ...],
                tuple[AgenticModelToolDefinition, ...],
            ]
        ] = []

    async def astream(
        self,
        messages: Sequence[AgenticModelMessage],
        *,
        tools: Sequence[AgenticModelToolDefinition],
    ) -> AsyncIterator[ModelStreamChunk]:
        """Use parent and child observation messages to cite valid evidence."""

        self.calls.append((tuple(messages), tuple(tools)))
        task = parse_json_object(messages[2].content)
        tool_messages = [
            message
            for message in messages
            if message.role == "tool"
        ]
        if task["message_type"] == "subagent_task":
            observations = [
                parse_json_object(message.content)
                for message in tool_messages
                if parse_json_object(message.content).get("message_type")
                == "tool_observation"
            ]
            if not observations:
                response = _tool_stream(
                    "child_fact",
                    {},
                    call_id="child-fact",
                )
            else:
                observation = observations[-1]
                output = observation["output"]
                arguments = _submit_arguments("Child evidence is grounded.")
                arguments["evidence"] = [{
                    "observation_id": observation["observation_id"],
                    "summary": output["summary"],
                    "provenance_refs": output["provenance_refs"],
                    "limitations": [],
                }]
                response = _tool_stream(
                    "submit_result",
                    arguments,
                    call_id="child-submit",
                )
        else:
            child_results = [
                parse_json_object(message.content)
                for message in tool_messages
                if parse_json_object(message.content).get("message_type")
                == "subagent_result"
            ]
            if not child_results:
                response = _tool_stream(
                    "run_subagent",
                    _subagent_arguments("evidence"),
                    call_id="root-child",
                )
            else:
                child_result = child_results[-1]
                child_evidence = child_result["evidence"][0]
                arguments = _submit_arguments("Parent cited child evidence.")
                arguments["evidence"] = [{
                    "observation_id": child_result["observation_id"],
                    "summary": child_evidence["summary"],
                    "provenance_refs": child_evidence["provenance_refs"],
                    "limitations": child_evidence["limitations"],
                }]
                response = _tool_stream(
                    "submit_result",
                    arguments,
                    call_id="root-submit",
                )
        for chunk in response:
            yield chunk


class _ObservationHandlePlacementModel:
    """Request one bounded replacement for a misplaced observation handle."""

    def __init__(self) -> None:
        self.capabilities = AgenticModelCapabilitiesV1(
            thinking_strategy="test_thinking",
            reasoning_replay_policy="test_replay",
        )
        self.calls: list[
            tuple[
                tuple[AgenticModelMessage, ...],
                tuple[AgenticModelToolDefinition, ...],
            ]
        ] = []

    async def astream(
        self,
        messages: Sequence[AgenticModelMessage],
        *,
        tools: Sequence[AgenticModelToolDefinition],
    ) -> AsyncIterator[ModelStreamChunk]:
        """First misplace a known handle, then send a clean terminal result."""

        self.calls.append((tuple(messages), tuple(tools)))
        tool_messages = [
            message
            for message in messages
            if message.content and message.role == "tool"
        ]
        observations = [
            parse_json_object(message.content)
            for message in tool_messages
            if parse_json_object(message.content).get("message_type")
            == "tool_observation"
        ]
        contract_errors = [
            parse_json_object(message.content)
            for message in messages
            if message.content
            and parse_json_object(message.content).get("message_type")
            == "contract_error"
        ]
        if not observations:
            response = _tool_stream("lookup", {}, call_id="lookup-1")
        else:
            observation = observations[-1]
            observation_id = observation["observation_id"]
            output = observation["output"]
            arguments = _submit_arguments()
            arguments["evidence"] = [{
                "observation_id": observation_id,
                "summary": output["summary"],
                "provenance_refs": output["provenance_refs"],
                "limitations": [],
            }]
            if not contract_errors:
                arguments["summary"] = (
                    f"The accepted observation is {observation_id}."
                )
            else:
                arguments["summary"] = "The accepted observation is grounded."
            response = _tool_stream(
                "submit_result",
                arguments,
                call_id="submit-replacement" if contract_errors else "submit-bad",
            )
        for chunk in response:
            yield chunk


def _tool_stream(
    name: str,
    arguments: Mapping[str, object],
    *,
    call_id: str,
) -> list[ModelStreamChunk]:
    """Return one reasoning-plus-native-tool stream."""

    serialized = json.dumps(arguments, separators=(",", ":"))
    split_at = max(1, len(serialized) // 2)
    chunks = [
        ModelStreamChunk(
            kind="block_start",
            block_index=0,
            block_type="reasoning",
        ),
        ModelStreamChunk(
            kind="reasoning_delta",
            block_index=0,
            block_type="reasoning",
            reasoning_delta="opaque test reasoning",
        ),
        ModelStreamChunk(
            kind="block_end",
            block_index=0,
            block_type="reasoning",
            completed_block={"type": "reasoning"},
        ),
        ModelStreamChunk(
            kind="block_start",
            block_index=2,
            block_type="tool_call",
        ),
        ModelStreamChunk(
            kind="tool_call_delta",
            block_index=2,
            block_type="tool_call",
            tool_call_id=call_id,
            tool_name=name,
            tool_arguments_delta=serialized[:split_at],
        ),
        ModelStreamChunk(
            kind="tool_call_delta",
            block_index=2,
            block_type="tool_call",
            tool_arguments_delta=serialized[split_at:],
        ),
        ModelStreamChunk(
            kind="block_end",
            block_index=2,
            block_type="tool_call",
            completed_block={"type": "tool_call"},
        ),
        ModelStreamChunk(kind="usage", usage={"output_tokens": 4}),
        ModelStreamChunk(
            kind="finish",
            finish=ModelStreamFinish(reason="tool_calls"),
        ),
    ]
    return chunks


def _empty_stream() -> list[ModelStreamChunk]:
    """Return one structurally complete turn with no native tool call."""

    chunks = [ModelStreamChunk(
        kind="finish",
        finish=ModelStreamFinish(reason="stop"),
    )]
    return chunks


def _multiple_tool_stream() -> list[ModelStreamChunk]:
    """Return one complete turn containing two native tool calls."""

    chunks: list[ModelStreamChunk] = []
    for index, name in ((0, "first_tool"), (1, "second_tool")):
        arguments = '{}'
        chunks.extend([
            ModelStreamChunk(
                kind="block_start",
                block_index=index,
                block_type="tool_call",
            ),
            ModelStreamChunk(
                kind="tool_call_delta",
                block_index=index,
                block_type="tool_call",
                tool_call_id=f"multiple-{index}",
                tool_name=name,
                tool_arguments_delta=arguments,
            ),
            ModelStreamChunk(
                kind="block_end",
                block_index=index,
                block_type="tool_call",
                completed_block={"type": "tool_call"},
            ),
        ])
    chunks.append(ModelStreamChunk(
        kind="finish",
        finish=ModelStreamFinish(reason="tool_calls"),
    ))
    return chunks


def _partial_tool_stream() -> list[ModelStreamChunk]:
    """Return a max-token stream whose tool arguments are incomplete."""

    chunks = [
        ModelStreamChunk(
            kind="block_start",
            block_index=0,
            block_type="tool_call",
        ),
        ModelStreamChunk(
            kind="tool_call_delta",
            block_index=0,
            block_type="tool_call",
            tool_call_id="partial-1",
            tool_name="lookup",
            tool_arguments_delta='{"query":"unfinished',
        ),
        ModelStreamChunk(
            kind="block_end",
            block_index=0,
            block_type="tool_call",
            completed_block={"type": "tool_call"},
        ),
        ModelStreamChunk(
            kind="finish",
            finish=ModelStreamFinish(reason="max_tokens"),
        ),
    ]
    return chunks


def _submit_arguments(summary: str = "Resolved from bounded evidence.") -> dict[str, object]:
    """Return one valid terminal call argument object."""

    arguments = {
        "status": "resolved",
        "summary": summary,
        "evidence": [],
        "completed_tasks": ["Resolve the task."],
        "remaining_needs": [],
    }
    return arguments


def _subagent_arguments(label: str) -> dict[str, object]:
    """Return one self-contained child task."""

    arguments = {
        "description": f"Investigate {label}.",
        "objective": f"Resolve independent branch {label}.",
        "context": {
            "facts": [f"Branch {label} fact."],
            "constraints": ["Use the bounded child roster."],
            "desired_output": "A concise branch result.",
        },
    }
    return arguments


def _request() -> AgenticResolverRequestV1:
    """Return one bounded public request."""

    request = AgenticResolverRequestV1(
        objective="Resolve one scripted task."
    )
    return request


def _ordinary_tool(
    name: str,
    execute: Callable[[Mapping[str, object]], Awaitable[object]],
) -> ToolDefinition:
    """Return one no-argument ordinary test tool."""

    definition = ToolDefinition(
        name=name,
        description=f"Execute bounded test capability {name}.",
        input_schema={
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
        execute=execute,
    )
    return definition


@pytest.mark.asyncio
async def test_loop_consumes_stream_before_executing_one_complete_native_tool_call() -> None:
    """Ordinary dispatch starts only after the complete stream is consumed."""

    model = _ScriptedModel([
        _tool_stream("lookup", {}, call_id="lookup-1"),
        _tool_stream(
            "submit_result",
            _submit_arguments(),
            call_id="submit-1",
        ),
    ])
    execution_state: list[bool] = []

    async def _execute(arguments: Mapping[str, object]) -> object:
        del arguments
        execution_state.append(model.stream_completed[0])
        output = {"summary": "lookup completed", "provenance_refs": []}
        return output

    runtime = AgenticResolverRuntime(
        model=model,
        tools=ToolRegistry([_ordinary_tool("lookup", _execute)]),
        skills=discover_skills([]),
    )

    result = await runtime.resolve(_request())

    assert result.status == "resolved"
    assert execution_state == [True]
    assert all(model.stream_completed)


@pytest.mark.asyncio
async def test_submit_result_replaces_misplaced_observation_handle() -> None:
    """A semantic handle leak receives bounded feedback before clean retry."""

    async def _execute(arguments: Mapping[str, object]) -> object:
        del arguments
        return {
            "summary": "lookup completed",
            "provenance_refs": ["fixture:lookup"],
        }

    model = _ObservationHandlePlacementModel()
    runtime = AgenticResolverRuntime(
        model=model,
        tools=ToolRegistry([_ordinary_tool("lookup", _execute)]),
        skills=discover_skills([]),
    )

    result = await runtime.resolve(_request())

    feedback = [
        parse_json_object(message.content)
        for messages, _ in model.calls
        for message in messages
        if message.content
        and parse_json_object(message.content).get("message_type")
        == "contract_error"
    ]
    assert result.status == "resolved"
    assert result.usage.contract_errors == 1
    assert len(feedback) == 1
    assert feedback[0]["code"] == "invalid_submit_result"
    assert "keep handles only in evidence[].observation_id" in (
        feedback[0]["message"]
    )
    assert result.evidence[0].provenance_refs == ("fixture:lookup",)


@pytest.mark.asyncio
async def test_loop_does_not_execute_interrupted_or_partial_tool_call() -> None:
    """A max-token partial call is discarded before registry dispatch."""

    executions = 0

    async def _execute(arguments: Mapping[str, object]) -> object:
        nonlocal executions
        del arguments
        executions += 1
        output = {"summary": "must not execute"}
        return output

    model = _ScriptedModel([_partial_tool_stream()])
    runtime = AgenticResolverRuntime(
        model=model,
        tools=ToolRegistry([_ordinary_tool("lookup", _execute)]),
        skills=discover_skills([]),
    )

    result = await runtime.resolve(_request())

    assert result.status == "budget_exhausted"
    assert executions == 0
    assert result.usage.tool_calls == 0


@pytest.mark.asyncio
async def test_loop_rejects_zero_or_multiple_tool_calls_with_bounded_json_feedback() -> None:
    """Both call-count violations receive capped object-rooted feedback."""

    for invalid_stream in (_empty_stream(), _multiple_tool_stream()):
        model = _ScriptedModel([
            invalid_stream,
            invalid_stream,
            invalid_stream,
        ])
        runtime = AgenticResolverRuntime(
            model=model,
            tools=ToolRegistry(),
            skills=discover_skills([]),
        )

        result = await runtime.resolve(_request())

        assert result.status == "failed"
        assert result.usage.contract_errors == 2
        feedback = [
            parse_json_object(message.content)
            for message in model.calls[-1][0]
            if message.role == "user"
            and parse_json_object(message.content).get("message_type")
            == "contract_error"
        ]
        assert len(feedback) == 2
        assert feedback[-1]["remaining_replacements"] == 0


@pytest.mark.asyncio
async def test_loop_stops_at_step_tool_and_contract_caps() -> None:
    """Model, non-terminal tool, and replacement caps all fail closed."""

    async def _execute(arguments: Mapping[str, object]) -> object:
        del arguments
        output = {"summary": "bounded result"}
        return output

    tool = _ordinary_tool("lookup", _execute)
    step_model = _ScriptedModel([
        _tool_stream("lookup", {}, call_id="step-1"),
    ])
    step_runtime = AgenticResolverRuntime(
        model=step_model,
        tools=ToolRegistry([tool]),
        skills=discover_skills([]),
        limits=AgenticResolverLimitsV1(max_model_steps=1),
    )
    step_result = await step_runtime.resolve(_request())

    tool_model = _ScriptedModel([
        _tool_stream("lookup", {}, call_id="tool-1"),
        _tool_stream("lookup", {}, call_id="tool-2"),
    ])
    tool_runtime = AgenticResolverRuntime(
        model=tool_model,
        tools=ToolRegistry([tool]),
        skills=discover_skills([]),
        limits=AgenticResolverLimitsV1(max_tool_calls=1),
    )
    tool_result = await tool_runtime.resolve(_request())

    contract_model = _ScriptedModel([
        _empty_stream(),
        _empty_stream(),
    ])
    contract_runtime = AgenticResolverRuntime(
        model=contract_model,
        tools=ToolRegistry(),
        skills=discover_skills([]),
        limits=AgenticResolverLimitsV1(max_contract_replacements=1),
    )
    contract_result = await contract_runtime.resolve(_request())

    assert step_result.status == "budget_exhausted"
    assert step_result.usage.model_steps == 1
    assert tool_result.status == "budget_exhausted"
    assert tool_result.usage.tool_calls == 1
    assert contract_result.status == "failed"
    assert contract_result.usage.contract_errors == 1


@pytest.mark.asyncio
async def test_parent_converges_multiple_subagent_results_into_terminal_result() -> None:
    """A root serially observes two isolated children before terminalization."""

    model = _ScriptedModel([
        _tool_stream(
            "run_subagent",
            _subagent_arguments("alpha"),
            call_id="root-child-1",
        ),
        _tool_stream(
            "submit_result",
            _submit_arguments("Alpha branch complete."),
            call_id="child-submit-1",
        ),
        _tool_stream(
            "run_subagent",
            _subagent_arguments("beta"),
            call_id="root-child-2",
        ),
        _tool_stream(
            "submit_result",
            _submit_arguments("Beta branch complete."),
            call_id="child-submit-2",
        ),
        _tool_stream(
            "submit_result",
            _submit_arguments("Both bounded branches converged."),
            call_id="root-submit",
        ),
    ])
    runtime = AgenticResolverRuntime(
        model=model,
        tools=ToolRegistry(),
        skills=discover_skills([]),
    )

    result = await runtime.resolve(_request())

    root_session = runtime._sessions[result.session_id]
    child_sessions = [
        session
        for session in runtime._sessions.values()
        if session.parent_session_id == result.session_id
    ]
    assert result.status == "resolved"
    assert result.summary == "Both bounded branches converged."
    assert len(child_sessions) == 2
    assert [
        observation.tool_name
        for observation in root_session.observations.values()
    ] == ["run_subagent", "run_subagent"]


@pytest.mark.asyncio
async def test_parent_submits_evidence_using_parent_child_observation_projection() -> None:
    """Parent terminal evidence cites the projected child observation ID."""

    async def _child_fact(arguments: Mapping[str, object]) -> object:
        del arguments
        output = {
            "summary": "Child fact is grounded.",
            "provenance_refs": ["fixture:child"],
        }
        return output

    model = _EvidenceModel()
    runtime = AgenticResolverRuntime(
        model=model,
        tools=ToolRegistry([_ordinary_tool("child_fact", _child_fact)]),
        skills=discover_skills([]),
    )
    result = await runtime.resolve(_request())

    root_session = runtime._sessions[result.session_id]
    child_sessions = [
        session
        for session in runtime._sessions.values()
        if session.parent_session_id == root_session.session_id
    ]
    parent_messages = [
        parse_json_object(message.content)
        for messages, _ in model.calls
        for message in messages
        if message.role == "tool"
        and parse_json_object(message.content).get("message_type")
        == "subagent_result"
    ]
    assert result.status == "resolved"
    assert len(child_sessions) == 1
    assert len(parent_messages) == 1

    parent_projection = parent_messages[0]
    parent_observation_id = parent_projection["observation_id"]
    child_observation_id = next(iter(child_sessions[0].observations))
    serialized_projection = json.dumps(
        parent_projection,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )

    assert parent_observation_id == next(iter(root_session.observations))
    assert parent_observation_id.startswith(root_session.session_id)
    assert child_observation_id.startswith(child_sessions[0].session_id)
    assert child_observation_id not in serialized_projection
    assert len(serialized_projection) <= runtime.limits.max_subagent_result_characters
    assert set(parent_projection["evidence"][0]) == {
        "summary",
        "provenance_refs",
        "limitations",
    }
    assert result.evidence[0].observation_id == parent_observation_id
    assert result.evidence[0].provenance_refs == ("fixture:child",)
    assert "opaque test reasoning" not in serialized_projection
