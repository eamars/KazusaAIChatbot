"""Deterministic context accounting and compaction tests."""

from __future__ import annotations

import json

import pytest

from agentic_resolver.context_budget import (
    ContextBudget,
    estimate_request_tokens,
    estimate_tokens_from_characters,
)
from agentic_resolver.contracts import (
    AgenticResolverLimitsV1,
    AgenticResolverRequestV1,
)
from agentic_resolver.model import (
    AgenticModelCapabilitiesV1,
    AgenticModelMessage,
    AgenticModelToolCall,
    AgenticModelToolDefinition,
    ModelStreamFinish,
)
from agentic_resolver.runtime import AgenticResolverRuntime
from agentic_resolver.session import ResolverSession
from agentic_resolver.skills import SkillCatalog
from agentic_resolver.streaming import AssembledAssistantTurn
from agentic_resolver.tools import ToolRegistry


def _session(*, task_size: int = 10) -> ResolverSession:
    """Build one session with configurable task payload size."""

    session = ResolverSession(
        session_id="session-1",
        depth=0,
        parent_session_id=None,
        policy_content='{"schema_version":"system.v1"}',
        catalog_content='{"schema_version":"catalog.v1"}',
        task_content=(
            '{"schema_version":"task.v1","text":"'
            + ("x" * task_size)
            + '"}'
        ),
    )
    return session


def _tool() -> AgenticModelToolDefinition:
    """Build one native schema included in every request estimate."""

    tool = AgenticModelToolDefinition(
        name="lookup",
        description="Look up one value.",
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
            "additionalProperties": False,
        },
    )
    return tool


def _turn(call_id: str, reasoning: str = "private") -> AssembledAssistantTurn:
    """Build one complete reasoning/tool turn for compaction tests."""

    turn = AssembledAssistantTurn(
        reasoning=reasoning,
        content="",
        tool_calls=(AgenticModelToolCall(
            call_id=call_id,
            name="lookup",
            arguments={"query": call_id},
        ),),
        invalid_tool_calls=(),
        usage={},
        finish=ModelStreamFinish(reason="tool_calls"),
    )
    return turn


def test_context_meter_counts_system_catalog_tools_history_and_reserved_completion() -> None:
    """Admission includes all message lanes, schemas, and completion reserve."""

    limits = AgenticResolverLimitsV1(
        context_window_tokens=2000,
        completion_reserve_tokens=200,
    )
    session = _session()
    budget = ContextBudget(limits)

    admission = budget.prepare(session, (_tool(),))

    assert admission is not None
    assert admission.estimated_total_tokens == (
        admission.estimated_input_tokens + 200
    )
    assert admission.estimated_input_tokens > 0
    without_tools = estimate_request_tokens(session.model_history(), ())
    assert admission.estimated_input_tokens > without_tools


def test_context_meter_counts_retained_reasoning_and_provider_replay_once() -> None:
    """Opaque reasoning appears once in the canonical assistant history row."""

    reasoning = "private-replay-text"
    messages = (
        AgenticModelMessage(
            role="assistant",
            reasoning=reasoning,
            tool_calls=(AgenticModelToolCall(
                call_id="call-1",
                name="lookup",
                arguments={"query": "value"},
            ),),
        ),
        AgenticModelMessage(
            role="tool",
            content='{"status":"success"}',
            tool_call_id="call-1",
        ),
    )
    serialized = json.dumps(
        {
            "messages": [message.to_dict() for message in messages],
            "tools": [_tool().to_dict()],
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )

    measured = estimate_request_tokens(messages, (_tool(),))

    assert serialized.count(reasoning) == 1
    assert measured == estimate_tokens_from_characters(len(serialized))


def test_context_meter_compacts_old_tool_results_before_hard_stop() -> None:
    """Old complete exchanges compact while the most recent exchange remains."""

    limits = AgenticResolverLimitsV1(
        context_window_tokens=900,
        completion_reserve_tokens=100,
    )
    session = _session(task_size=200)
    for index in range(3):
        call_id = f"call-{index}"
        session.append_exchange(
            _turn(call_id),
            tool_content=(
                '{"schema_version":"observation.v1","summary":"'
                + ("y" * 700)
                + '"}'
            ),
            tool_call_id=call_id,
            compacted_content=(
                '{"schema_version":"compacted.v1","summary":"short"}'
            ),
        )
    budget = ContextBudget(limits)

    admission = budget.prepare(session, (_tool(),))

    assert admission is not None
    assert session.usage.compactions >= 1
    assert admission.estimated_input_tokens <= limits.input_ceiling_tokens
    assert any(
        message.role == "assistant"
        and message.tool_calls
        and message.tool_calls[0].call_id == "call-2"
        for message in admission.messages
    )


class _NeverCalledModel:
    """Fake model used to prove over-cap admission stops before transport."""

    def __init__(self) -> None:
        self.calls = 0
        self.capabilities = AgenticModelCapabilitiesV1(
            thinking_strategy="qwen3_enabled",
            reasoning_replay_policy="adapter_owned",
        )

    async def astream(self, messages, *, tools):
        """Record an unexpected model call."""

        del messages, tools
        self.calls += 1
        if False:
            yield None


@pytest.mark.asyncio
async def test_context_cap_returns_budget_exhausted_without_over_limit_model_call() -> None:
    """A request that cannot fit terminates before opening the fake stream."""

    model = _NeverCalledModel()
    runtime = AgenticResolverRuntime(
        model=model,
        tools=ToolRegistry(),
        skills=SkillCatalog(),
        limits=AgenticResolverLimitsV1(
            context_window_tokens=500,
            completion_reserve_tokens=100,
        ),
    )

    result = await runtime.resolve(AgenticResolverRequestV1(
        objective="x" * 4000,
    ))

    assert result.status == "budget_exhausted"
    assert model.calls == 0
