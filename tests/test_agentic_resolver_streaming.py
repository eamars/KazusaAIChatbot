"""Deterministic indexed stream-assembly tests."""

from __future__ import annotations

import pytest

from agentic_resolver.contracts import AgenticResolverContractError
from agentic_resolver.model import ModelStreamChunk, ModelStreamFinish
from agentic_resolver.streaming import ModelStreamAssembler


def _consume_complete_tool_stream(
    assembler: ModelStreamAssembler,
) -> None:
    """Feed one complete reasoning, text, and tool-call stream."""

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
            reasoning_delta="private",
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
            tool_call_id="call-1",
            tool_name="lookup",
            tool_arguments_delta='{"query":"',
        ),
        ModelStreamChunk(
            kind="tool_call_delta",
            block_index=2,
            block_type="tool_call",
            tool_arguments_delta='value"}',
        ),
        ModelStreamChunk(
            kind="block_end",
            block_index=2,
            block_type="tool_call",
            completed_block={"type": "tool_call"},
        ),
        ModelStreamChunk(
            kind="finish",
            finish=ModelStreamFinish(reason="tool_calls"),
        ),
    ]
    for chunk in chunks:
        assembler.consume(chunk)


def test_stream_assembler_reconstructs_reasoning_text_and_one_tool_call() -> None:
    """Indexed deltas assemble into distinct reasoning and native arguments."""

    assembler = ModelStreamAssembler()
    _consume_complete_tool_stream(assembler)

    turn = assembler.finalize()

    assert turn.reasoning == "private"
    assert turn.content == ""
    assert len(turn.tool_calls) == 1
    assert turn.tool_calls[0].arguments == {"query": "value"}


def test_stream_assembler_rejects_malformed_block_order() -> None:
    """A delta cannot arrive before its indexed block start."""

    assembler = ModelStreamAssembler()

    with pytest.raises(AgenticResolverContractError, match="open reasoning"):
        assembler.consume(ModelStreamChunk(
            kind="reasoning_delta",
            block_index=0,
            block_type="reasoning",
            reasoning_delta="private",
        ))


def test_stream_assembler_never_exposes_partial_tool_call() -> None:
    """Max-token termination discards a syntactically incomplete tool call."""

    assembler = ModelStreamAssembler()
    chunks = [
        ModelStreamChunk(
            kind="block_start",
            block_index=2,
            block_type="tool_call",
        ),
        ModelStreamChunk(
            kind="tool_call_delta",
            block_index=2,
            block_type="tool_call",
            tool_call_id="call-1",
            tool_name="lookup",
            tool_arguments_delta='{"query":"unfinished',
        ),
        ModelStreamChunk(
            kind="block_end",
            block_index=2,
            block_type="tool_call",
            completed_block={"type": "tool_call"},
        ),
        ModelStreamChunk(
            kind="finish",
            finish=ModelStreamFinish(reason="max_tokens"),
        ),
    ]
    for chunk in chunks:
        assembler.consume(chunk)

    turn = assembler.finalize()

    assert turn.tool_calls == ()
    assert turn.invalid_tool_calls == ()
