"""Deterministic same-runtime subagent tests."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Mapping, Sequence
from pathlib import Path

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
    """Serve one shared ordered stream script to root and child sessions."""

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

    async def astream(
        self,
        messages: Sequence[AgenticModelMessage],
        *,
        tools: Sequence[AgenticModelToolDefinition],
    ) -> AsyncIterator[ModelStreamChunk]:
        """Yield the next root-or-child response from one shared model seam."""

        if not self._responses:
            raise AssertionError("scripted model exhausted")
        self.calls.append((tuple(messages), tuple(tools)))
        response = self._responses.pop(0)
        for chunk in response:
            yield chunk


def _tool_stream(
    name: str,
    arguments: Mapping[str, object],
    *,
    call_id: str,
) -> list[ModelStreamChunk]:
    """Return one complete reasoning-plus-tool stream."""

    serialized = json.dumps(arguments, separators=(",", ":"))
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
            reasoning_delta="opaque child planning",
        ),
        ModelStreamChunk(
            kind="block_end",
            block_index=0,
            block_type="reasoning",
            completed_block={"type": "reasoning"},
        ),
        ModelStreamChunk(
            kind="block_start",
            block_index=1,
            block_type="tool_call",
        ),
        ModelStreamChunk(
            kind="tool_call_delta",
            block_index=1,
            block_type="tool_call",
            tool_call_id=call_id,
            tool_name=name,
            tool_arguments_delta=serialized,
        ),
        ModelStreamChunk(
            kind="block_end",
            block_index=1,
            block_type="tool_call",
            completed_block={"type": "tool_call"},
        ),
        ModelStreamChunk(
            kind="finish",
            finish=ModelStreamFinish(reason="tool_calls"),
        ),
    ]
    return chunks


def _child_arguments(label: str = "branch") -> dict[str, object]:
    """Return one exact self-contained child task."""

    arguments = {
        "description": f"Resolve {label}.",
        "objective": f"Find the bounded answer for {label}.",
        "context": {
            "facts": [f"Fact supplied specifically to {label}."],
            "constraints": ["Return a concise supported result."],
            "desired_output": "One typed result.",
        },
    }
    return arguments


def _submit_arguments(summary: str) -> dict[str, object]:
    """Return one valid no-evidence resolved result."""

    arguments = {
        "status": "resolved",
        "summary": summary,
        "evidence": [],
        "completed_tasks": ["Complete the assigned objective."],
        "remaining_needs": [],
    }
    return arguments


def _request() -> AgenticResolverRequestV1:
    """Return one root request."""

    request = AgenticResolverRequestV1(
        objective="Resolve the parent objective."
    )
    return request


def _parent_child_submit_script(
    *,
    child_summary: str = "Child completed its branch.",
) -> list[list[ModelStreamChunk]]:
    """Return one root-child-root terminal stream sequence."""

    responses = [
        _tool_stream(
            "run_subagent",
            _child_arguments(),
            call_id="parent-child",
        ),
        _tool_stream(
            "submit_result",
            _submit_arguments(child_summary),
            call_id="child-submit",
        ),
        _tool_stream(
            "submit_result",
            _submit_arguments("Parent integrated the child result."),
            call_id="parent-submit",
        ),
    ]
    return responses


@pytest.mark.asyncio
async def test_run_subagent_uses_same_runtime_with_isolated_session() -> None:
    """Root and child share one runtime model but retain separate sessions."""

    model = _ScriptedModel(_parent_child_submit_script())
    runtime = AgenticResolverRuntime(
        model=model,
        tools=ToolRegistry(),
        skills=discover_skills([]),
    )

    result = await runtime.resolve(_request())

    root = runtime._sessions[result.session_id]
    children = [
        session
        for session in runtime._sessions.values()
        if session.parent_session_id == result.session_id
    ]
    assert result.status == "resolved"
    assert len(children) == 1
    assert children[0] is not root
    assert children[0].depth == 1
    assert children[0].parent_session_id == root.session_id
    assert root.usage.model_steps == 2
    assert children[0].usage.model_steps == 1


@pytest.mark.asyncio
async def test_child_registry_excludes_run_subagent() -> None:
    """Every child model step sees core tools without recursive delegation."""

    model = _ScriptedModel(_parent_child_submit_script())
    runtime = AgenticResolverRuntime(
        model=model,
        tools=ToolRegistry(),
        skills=discover_skills([]),
    )

    await runtime.resolve(_request())

    root_names = {tool.name for tool in model.calls[0][1]}
    child_names = {tool.name for tool in model.calls[1][1]}
    assert "run_subagent" in root_names
    assert child_names == {"skill", "submit_result"}


@pytest.mark.asyncio
async def test_child_inherits_tools_skills_permissions_json_and_thinking_stream(
    tmp_path: Path,
) -> None:
    """Child composition retains every root capability except recursion."""

    bundle = tmp_path / "sample-skill"
    bundle.mkdir()
    (bundle / "SKILL.md").write_text(
        "---\nname: sample-skill\n"
        "description: Apply bounded sample instructions.\n---\n\n"
        "# Instructions\n\nUse the supplied facts.\n",
        encoding="utf-8",
    )
    observed_permissions: list[dict[str, object]] = []

    async def _inspect(arguments: Mapping[str, object]) -> object:
        del arguments
        output = {"summary": "child ordinary tool completed"}
        return output

    def _permission(scope: Mapping[str, object]) -> bool:
        observed_permissions.append(dict(scope))
        permitted = scope.get("tenant") == "fixture"
        return permitted

    ordinary_tool = ToolDefinition(
        name="inspect_scope",
        description="Inspect inherited test permission scope.",
        input_schema={
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
        execute=_inspect,
        permission_check=_permission,
    )
    model = _ScriptedModel([
        _tool_stream(
            "run_subagent",
            _child_arguments("inheritance"),
            call_id="parent-child",
        ),
        _tool_stream("inspect_scope", {}, call_id="child-inspect"),
        _tool_stream(
            "submit_result",
            _submit_arguments("Child used inherited capability."),
            call_id="child-submit",
        ),
        _tool_stream(
            "submit_result",
            _submit_arguments("Parent accepted bounded child output."),
            call_id="parent-submit",
        ),
    ])
    runtime = AgenticResolverRuntime(
        model=model,
        tools=ToolRegistry([ordinary_tool]),
        skills=discover_skills([tmp_path]),
        permission_scope={"tenant": "fixture"},
    )

    result = await runtime.resolve(_request())

    child_calls = [
        call
        for call in model.calls
        if parse_json_object(call[0][2].content)["message_type"]
        == "subagent_task"
    ]
    assert result.status == "resolved"
    assert observed_permissions == [{"tenant": "fixture"}]
    assert len(child_calls) == 2
    for messages, tools in child_calls:
        assert {tool.name for tool in tools} == {
            "inspect_scope",
            "skill",
            "submit_result",
        }
        assert parse_json_object(messages[1].content)["skills"] == [{
            "name": "sample-skill",
            "description": "Apply bounded sample instructions.",
        }]
        for message in messages:
            if message.content:
                assert isinstance(parse_json_object(message.content), dict)
    assert model.capabilities.thinking_enabled is True
    assert model.capabilities.streaming is True


@pytest.mark.asyncio
async def test_parent_receives_only_bounded_typed_child_result() -> None:
    """Parent history receives one small typed projection and no child trace."""

    long_summary = "bounded child detail " * 140
    model = _ScriptedModel(_parent_child_submit_script(
        child_summary=long_summary,
    ))
    limits = AgenticResolverLimitsV1(max_subagent_result_characters=500)
    runtime = AgenticResolverRuntime(
        model=model,
        tools=ToolRegistry(),
        skills=discover_skills([]),
        limits=limits,
    )

    result = await runtime.resolve(_request())

    parent_followup_messages = model.calls[-1][0]
    child_results = [
        message.content
        for message in parent_followup_messages
        if message.role == "tool"
        and parse_json_object(message.content).get("message_type")
        == "subagent_result"
    ]
    assert result.status == "resolved"
    assert len(child_results) == 1
    assert len(child_results[0]) <= limits.max_subagent_result_characters
    parsed = parse_json_object(child_results[0])
    assert parsed["schema_version"] == "agentic_resolver_subagent_result.v1"
    assert parsed["message_type"] == "subagent_result"
    assert parsed["observation_id"].startswith(result.session_id)
    assert all(
        "observation_id" not in evidence
        for evidence in parsed["evidence"]
    )
    assert "opaque child planning" not in child_results[0]
    assert "model_stream_chunk" not in child_results[0]


@pytest.mark.asyncio
async def test_subagent_run_cap_fails_closed() -> None:
    """The fourth root delegation becomes a bounded error without a child."""

    responses: list[list[ModelStreamChunk]] = []
    for index in range(3):
        responses.extend([
            _tool_stream(
                "run_subagent",
                _child_arguments(f"branch-{index}"),
                call_id=f"parent-child-{index}",
            ),
            _tool_stream(
                "submit_result",
                _submit_arguments(f"Child {index} complete."),
                call_id=f"child-submit-{index}",
            ),
        ])
    responses.extend([
        _tool_stream(
            "run_subagent",
            _child_arguments("branch-over-cap"),
            call_id="parent-child-over-cap",
        ),
        _tool_stream(
            "submit_result",
            _submit_arguments("Parent stopped after the bounded child cap."),
            call_id="parent-submit",
        ),
    ])
    model = _ScriptedModel(responses)
    runtime = AgenticResolverRuntime(
        model=model,
        tools=ToolRegistry(),
        skills=discover_skills([]),
    )

    result = await runtime.resolve(_request())

    child_sessions = [
        session
        for session in runtime._sessions.values()
        if session.parent_session_id == result.session_id
    ]
    final_parent_messages = model.calls[-1][0]
    cap_errors = [
        parse_json_object(message.content)
        for message in final_parent_messages
        if message.role == "tool"
        and parse_json_object(message.content).get("tool_name")
        == "run_subagent"
        and parse_json_object(message.content).get("status") == "error"
    ]
    assert result.status == "resolved"
    assert len(child_sessions) == 3
    assert result.usage.subagent_runs == 3
    assert len(cap_errors) == 1
    assert "cap reached" in str(cap_errors[0]["error"])
