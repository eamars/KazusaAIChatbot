"""Deterministic append-only session and history-projection tests."""

from __future__ import annotations

from agentic_resolver.model import (
    AgenticModelToolCall,
    ModelStreamChunk,
    ModelStreamFinish,
)
from agentic_resolver.session import ResolverSession
from agentic_resolver.streaming import AssembledAssistantTurn


def _session() -> ResolverSession:
    """Build one minimal JSON-only session."""

    session = ResolverSession(
        session_id="session-1",
        depth=0,
        parent_session_id=None,
        policy_content='{"schema_version":"system.v1"}',
        catalog_content='{"schema_version":"catalog.v1"}',
        task_content='{"schema_version":"task.v1"}',
    )
    return session


def _turn(call_id: str) -> AssembledAssistantTurn:
    """Build one complete tool-calling assistant turn."""

    turn = AssembledAssistantTurn(
        reasoning="private reasoning",
        content="",
        tool_calls=(AgenticModelToolCall(
            call_id=call_id,
            name="lookup",
            arguments={"query": call_id},
        ),),
        invalid_tool_calls=(),
        usage={"input_tokens": 2},
        finish=ModelStreamFinish(reason="tool_calls"),
    )
    return turn


def test_session_log_records_chunks_and_reconstructs_assistant_history() -> None:
    """The same normalized chunk and assembled turn remain reconstructable."""

    session = _session()
    chunk = ModelStreamChunk(
        kind="reasoning_delta",
        block_index=0,
        block_type="reasoning",
        reasoning_delta="private reasoning",
    )
    session.record_stream_chunk(chunk)
    turn = _turn("call-1")
    session.record_assembled_turn(turn)
    session.append_exchange(
        turn,
        tool_content='{"schema_version":"observation.v1"}',
        tool_call_id="call-1",
    )

    history = session.model_history()

    assert session.stream_chunks == (chunk,)
    assert history[-2].role == "assistant"
    assert history[-2].reasoning == "private reasoning"
    assert history[-1].role == "tool"


def test_compaction_preserves_reasoning_tool_call_and_result_atomically() -> None:
    """Compaction removes a complete old exchange and keeps the recent one."""

    session = _session()
    first_turn = _turn("call-1")
    second_turn = _turn("call-2")
    session.append_exchange(
        first_turn,
        tool_content='{"schema_version":"observation.v1","id":1}',
        tool_call_id="call-1",
        compacted_content='{"schema_version":"compacted.v1","id":1}',
    )
    session.append_exchange(
        second_turn,
        tool_content='{"schema_version":"observation.v1","id":2}',
        tool_call_id="call-2",
        compacted_content='{"schema_version":"compacted.v1","id":2}',
    )

    assert session.compact_oldest_exchange(keep_recent=1) is True
    history = session.model_history()

    assert not any(
        message.role == "assistant"
        and message.tool_calls
        and message.tool_calls[0].call_id == "call-1"
        for message in history
    )
    assert not any(
        message.role == "tool" and message.tool_call_id == "call-1"
        for message in history
    )
    assert any(
        message.role == "user" and '"id":1' in message.content
        for message in history
    )
    assert any(
        message.role == "assistant"
        and message.tool_calls
        and message.tool_calls[0].call_id == "call-2"
        and message.reasoning == "private reasoning"
        for message in history
    )
