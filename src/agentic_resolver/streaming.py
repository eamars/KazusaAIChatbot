"""Indexed stream assembly with complete-turn admission."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from agentic_resolver.contracts import AgenticResolverContractError
from agentic_resolver.model import (
    AgenticInvalidToolCall,
    AgenticModelToolCall,
    ModelStreamChunk,
    ModelStreamFinish,
)


@dataclass(frozen=True)
class AssembledAssistantTurn:
    """One complete assistant turn produced after terminal stream validation."""

    reasoning: str | None
    content: str
    tool_calls: tuple[AgenticModelToolCall, ...]
    invalid_tool_calls: tuple[AgenticInvalidToolCall, ...]
    usage: dict[str, int]
    finish: ModelStreamFinish


@dataclass
class _BlockState:
    """Mutable assembly state for one provider block index."""

    block_type: str
    ended: bool = False
    text_fragments: list[str] = field(default_factory=list)
    call_id: str | None = None
    tool_name: str | None = None


class ModelStreamAssembler:
    """Assemble one bounded indexed stream before exposing any tool call."""

    def __init__(self, *, max_output_characters: int = 32_000) -> None:
        if max_output_characters <= 0:
            raise AgenticResolverContractError(
                "max_output_characters must be positive"
            )
        self._max_output_characters = max_output_characters
        self._output_characters = 0
        self._blocks: dict[int, _BlockState] = {}
        self._usage: dict[str, int] = {}
        self._finish: ModelStreamFinish | None = None

    def consume(self, chunk: ModelStreamChunk) -> None:
        """Consume one ordered normalized chunk under the stream contract."""

        if self._finish is not None:
            raise AgenticResolverContractError(
                "stream emitted data after finish",
                code="malformed_stream",
            )
        if chunk.kind == "block_start":
            self._start_block(chunk)
            return
        if chunk.kind == "reasoning_delta":
            self._append_delta(chunk, expected_type="reasoning")
            return
        if chunk.kind == "text_delta":
            self._append_delta(chunk, expected_type="text")
            return
        if chunk.kind == "tool_call_delta":
            self._append_tool_delta(chunk)
            return
        if chunk.kind == "block_end":
            self._end_block(chunk)
            return
        if chunk.kind == "usage":
            self._merge_usage(chunk)
            return
        if chunk.kind == "finish":
            self._finish_stream(chunk)
            return
        raise AgenticResolverContractError(
            f"unsupported stream chunk kind: {chunk.kind}",
            code="malformed_stream",
        )

    def finalize(self) -> AssembledAssistantTurn:
        """Return one complete turn or reject unfinished provider state."""

        if self._finish is None:
            raise AgenticResolverContractError(
                "stream ended without finish",
                code="malformed_stream",
            )
        open_indexes = [
            index for index, block in self._blocks.items() if not block.ended
        ]
        if open_indexes:
            raise AgenticResolverContractError(
                f"stream ended with open blocks: {sorted(open_indexes)}",
                code="malformed_stream",
            )
        reasoning_fragments: list[str] = []
        text_fragments: list[str] = []
        tool_calls: list[AgenticModelToolCall] = []
        invalid_tool_calls: list[AgenticInvalidToolCall] = []
        for _, block in sorted(self._blocks.items()):
            combined = "".join(block.text_fragments)
            if block.block_type == "reasoning":
                reasoning_fragments.append(combined)
                continue
            if block.block_type == "text":
                text_fragments.append(combined)
                continue
            self._assemble_tool_block(
                block,
                arguments_text=combined,
                tool_calls=tool_calls,
                invalid_tool_calls=invalid_tool_calls,
            )
        if self._finish.reason in {"max_tokens", "aborted", "error"}:
            tool_calls = []
            invalid_tool_calls = []
        reasoning_text = "".join(reasoning_fragments)
        turn = AssembledAssistantTurn(
            reasoning=reasoning_text or None,
            content="".join(text_fragments),
            tool_calls=tuple(tool_calls),
            invalid_tool_calls=tuple(invalid_tool_calls),
            usage=dict(self._usage),
            finish=self._finish,
        )
        return turn

    def _start_block(self, chunk: ModelStreamChunk) -> None:
        """Open one new indexed block with a declared content type."""

        if chunk.block_index is None or chunk.block_type is None:
            raise AgenticResolverContractError(
                "block_start requires index and type",
                code="malformed_stream",
            )
        if chunk.block_index in self._blocks:
            raise AgenticResolverContractError(
                f"block index {chunk.block_index} started more than once",
                code="malformed_stream",
            )
        self._blocks[chunk.block_index] = _BlockState(
            block_type=chunk.block_type
        )

    def _append_delta(
        self,
        chunk: ModelStreamChunk,
        *,
        expected_type: str,
    ) -> None:
        """Append reasoning or visible text to its matching open block."""

        block = self._open_block(chunk, expected_type=expected_type)
        delta = (
            chunk.reasoning_delta
            if expected_type == "reasoning"
            else chunk.text_delta
        )
        self._reserve_output_characters(len(delta))
        block.text_fragments.append(delta)

    def _append_tool_delta(self, chunk: ModelStreamChunk) -> None:
        """Append raw native tool arguments while preserving call identity."""

        block = self._open_block(chunk, expected_type="tool_call")
        if chunk.tool_call_id is not None:
            if block.call_id is not None and block.call_id != chunk.tool_call_id:
                raise AgenticResolverContractError(
                    "tool call id changed within one block",
                    code="malformed_stream",
                )
            block.call_id = chunk.tool_call_id
        if chunk.tool_name is not None:
            if block.tool_name is not None and block.tool_name != chunk.tool_name:
                raise AgenticResolverContractError(
                    "tool name changed within one block",
                    code="malformed_stream",
                )
            block.tool_name = chunk.tool_name
        self._reserve_output_characters(len(chunk.tool_arguments_delta))
        block.text_fragments.append(chunk.tool_arguments_delta)

    def _end_block(self, chunk: ModelStreamChunk) -> None:
        """Close one previously started block exactly once."""

        if chunk.block_index is None:
            raise AgenticResolverContractError(
                "block_end requires an index",
                code="malformed_stream",
            )
        block = self._blocks.get(chunk.block_index)
        if block is None or block.ended:
            raise AgenticResolverContractError(
                f"block index {chunk.block_index} cannot end",
                code="malformed_stream",
            )
        if chunk.block_type is not None and chunk.block_type != block.block_type:
            raise AgenticResolverContractError(
                "block_end type differs from block_start",
                code="malformed_stream",
            )
        completed_type = chunk.completed_block.get("type")
        if completed_type is not None and completed_type != block.block_type:
            raise AgenticResolverContractError(
                "completed block type differs from block_start",
                code="malformed_stream",
            )
        block.ended = True

    def _merge_usage(self, chunk: ModelStreamChunk) -> None:
        """Add non-negative integer usage counters without semantic inference."""

        for key, value in chunk.usage.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                continue
            self._usage[key] = self._usage.get(key, 0) + value

    def _finish_stream(self, chunk: ModelStreamChunk) -> None:
        """Record the one terminal stream disposition after all blocks close."""

        if chunk.finish is None:
            raise AgenticResolverContractError(
                "finish chunk lacks disposition",
                code="malformed_stream",
            )
        open_indexes = [
            index for index, block in self._blocks.items() if not block.ended
        ]
        if open_indexes:
            raise AgenticResolverContractError(
                f"finish arrived before block_end: {sorted(open_indexes)}",
                code="malformed_stream",
            )
        self._finish = chunk.finish

    def _open_block(
        self,
        chunk: ModelStreamChunk,
        *,
        expected_type: str,
    ) -> _BlockState:
        """Return one open block after validating index and declared type."""

        if chunk.block_index is None:
            raise AgenticResolverContractError(
                f"{chunk.kind} requires a block index",
                code="malformed_stream",
            )
        block = self._blocks.get(chunk.block_index)
        if block is None or block.ended or block.block_type != expected_type:
            raise AgenticResolverContractError(
                f"{chunk.kind} does not match an open {expected_type} block",
                code="malformed_stream",
            )
        return block

    def _reserve_output_characters(self, additional_characters: int) -> None:
        """Reject model output that exceeds the configured completion budget."""

        self._output_characters += additional_characters
        if self._output_characters > self._max_output_characters:
            raise AgenticResolverContractError(
                "model stream exceeded its completion character budget",
                code="stream_budget_exhausted",
            )

    @staticmethod
    def _assemble_tool_block(
        block: _BlockState,
        *,
        arguments_text: str,
        tool_calls: list[AgenticModelToolCall],
        invalid_tool_calls: list[AgenticInvalidToolCall],
    ) -> None:
        """Decode one completed raw argument object into a typed call."""

        if not block.call_id or not block.tool_name:
            invalid_tool_calls.append(AgenticInvalidToolCall(
                call_id=block.call_id,
                name=block.tool_name,
                error="tool call is missing id or name",
            ))
            return
        try:
            arguments = json.loads(arguments_text)
        except json.JSONDecodeError as exc:
            invalid_tool_calls.append(AgenticInvalidToolCall(
                call_id=block.call_id,
                name=block.tool_name,
                error=f"tool arguments are malformed JSON: {exc.msg}",
            ))
            return
        if not isinstance(arguments, dict):
            invalid_tool_calls.append(AgenticInvalidToolCall(
                call_id=block.call_id,
                name=block.tool_name,
                error="tool arguments must have an object root",
            ))
            return
        tool_calls.append(AgenticModelToolCall(
            call_id=block.call_id,
            name=block.tool_name,
            arguments=arguments,
        ))
