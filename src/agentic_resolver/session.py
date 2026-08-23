"""Append-only in-memory resolver session and model-history projection."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from agentic_resolver.contracts import AgenticResolverUsageV1
from agentic_resolver.model import AgenticModelMessage, ModelStreamChunk
from agentic_resolver.streaming import AssembledAssistantTurn


@dataclass(frozen=True)
class ResolverSessionEvent:
    """One indexed private runtime event with thought-text-free metadata."""

    index: int
    kind: str
    metadata: Mapping[str, object]


@dataclass(frozen=True)
class ResolverObservation:
    """Accepted root or child observation available to terminal evidence."""

    observation_id: str
    tool_name: str
    status: str
    summary: str
    evidence_refs: tuple[str, ...]


@dataclass
class _HistorySegment:
    """Atomic full exchange with an optional compact model-view replacement."""

    messages: tuple[AgenticModelMessage, ...]
    compacted_message: AgenticModelMessage | None = None
    compactable: bool = False
    compacted: bool = False


class ResolverSession:
    """Retain private stream state while deriving a bounded provider history."""

    def __init__(
        self,
        *,
        session_id: str,
        depth: int,
        parent_session_id: str | None,
        policy_content: str,
        catalog_content: str,
        task_content: str,
    ) -> None:
        self.session_id = session_id
        self.depth = depth
        self.parent_session_id = parent_session_id
        self.usage = AgenticResolverUsageV1()
        self._events: list[ResolverSessionEvent] = []
        self._stream_chunks: list[ModelStreamChunk] = []
        self._segments: list[_HistorySegment] = [
            _HistorySegment(messages=(AgenticModelMessage(
                role="system",
                content=policy_content,
            ),)),
            _HistorySegment(messages=(AgenticModelMessage(
                role="user",
                content=catalog_content,
            ),)),
            _HistorySegment(messages=(AgenticModelMessage(
                role="user",
                content=task_content,
            ),)),
        ]
        self._observations: dict[str, ResolverObservation] = {}
        self._observation_counter = 0
        self._loaded_skills: set[str] = set()
        self._record_event(
            "session_started",
            {
                "session_id": session_id,
                "depth": depth,
                "parent_session_id": parent_session_id,
            },
        )
        self._record_event("policy_appended", {"content_characters": len(policy_content)})
        self._record_event(
            "skill_catalog_appended",
            {"content_characters": len(catalog_content)},
        )
        self._record_event("task_appended", {"content_characters": len(task_content)})

    @property
    def events(self) -> tuple[ResolverSessionEvent, ...]:
        """Return immutable event metadata without opaque reasoning text."""

        events = tuple(self._events)
        return events

    @property
    def stream_chunks(self) -> tuple[ModelStreamChunk, ...]:
        """Return the private normalized chunks retained for process replay."""

        chunks = tuple(self._stream_chunks)
        return chunks

    @property
    def observations(self) -> Mapping[str, ResolverObservation]:
        """Return the immutable observation lookup used by terminal validation."""

        observations = MappingProxyType(dict(self._observations))
        return observations

    @property
    def loaded_skills(self) -> frozenset[str]:
        """Return skill names already loaded into this session history."""

        loaded_skills = frozenset(self._loaded_skills)
        return loaded_skills

    def model_history(self) -> tuple[AgenticModelMessage, ...]:
        """Derive provider history from full or atomically compacted segments."""

        messages: list[AgenticModelMessage] = []
        for segment in self._segments:
            if segment.compacted and segment.compacted_message is not None:
                messages.append(segment.compacted_message)
            else:
                messages.extend(segment.messages)
        history = tuple(messages)
        return history

    def record_model_step_started(self) -> None:
        """Advance the model-step counter and append its event."""

        self.usage.model_steps += 1
        self._record_event(
            "model_step_started",
            {"model_step": self.usage.model_steps},
        )

    def record_stream_chunk(self, chunk: ModelStreamChunk) -> None:
        """Retain each normalized chunk and thought-text-free ordering metadata."""

        self._stream_chunks.append(chunk)
        metadata: dict[str, object] = {
            "kind": chunk.kind,
            "block_index": chunk.block_index,
            "block_type": chunk.block_type,
        }
        if chunk.kind == "reasoning_delta":
            metadata["character_count"] = len(chunk.reasoning_delta)
        if chunk.kind == "text_delta":
            metadata["character_count"] = len(chunk.text_delta)
        if chunk.kind == "tool_call_delta":
            metadata["argument_character_count"] = len(
                chunk.tool_arguments_delta
            )
        self._record_event("model_stream_chunk", metadata)

    def record_assembled_turn(self, turn: AssembledAssistantTurn) -> None:
        """Record one complete assistant turn without publishing thought text."""

        self._merge_provider_usage(turn.usage)
        self._record_event(
            "assistant_turn_assembled",
            {
                "reasoning_characters": len(turn.reasoning or ""),
                "content_characters": len(turn.content),
                "tool_call_count": len(turn.tool_calls),
                "invalid_tool_call_count": len(turn.invalid_tool_calls),
                "finish_reason": turn.finish.reason,
            },
        )

    def append_exchange(
        self,
        turn: AssembledAssistantTurn,
        *,
        tool_content: str,
        tool_call_id: str,
        compacted_content: str | None = None,
        observation: ResolverObservation | None = None,
    ) -> None:
        """Append one atomic reasoning/call/result exchange to model history."""

        assistant_message = AgenticModelMessage(
            role="assistant",
            content=turn.content,
            reasoning=turn.reasoning,
            tool_calls=turn.tool_calls,
        )
        tool_message = AgenticModelMessage(
            role="tool",
            content=tool_content,
            tool_call_id=tool_call_id,
        )
        compacted_message = None
        if compacted_content is not None:
            compacted_message = AgenticModelMessage(
                role="user",
                content=compacted_content,
            )
        segment = _HistorySegment(
            messages=(assistant_message, tool_message),
            compacted_message=compacted_message,
            compactable=compacted_message is not None,
        )
        self._segments.append(segment)
        if observation is not None:
            self._observations[observation.observation_id] = observation
        self._record_event(
            "tool_result_appended",
            {
                "tool_call_id": tool_call_id,
                "observation_id": (
                    observation.observation_id if observation is not None else None
                ),
                "content_characters": len(tool_content),
            },
        )

    def append_terminal_turn(self, turn: AssembledAssistantTurn) -> None:
        """Retain the accepted submit_result assistant turn before termination."""

        message = AgenticModelMessage(
            role="assistant",
            content=turn.content,
            reasoning=turn.reasoning,
            tool_calls=turn.tool_calls,
        )
        self._segments.append(_HistorySegment(messages=(message,)))

    def append_protocol_feedback(self, content: str, *, code: str) -> None:
        """Append JSON structural feedback for a bounded replacement step."""

        message = AgenticModelMessage(role="user", content=content)
        self._segments.append(_HistorySegment(messages=(message,)))
        self.record_contract_error(code=code)

    def record_contract_error(self, *, code: str) -> None:
        """Advance the bounded structural-error counter."""

        self.usage.contract_errors += 1
        self._record_event("contract_error_appended", {"code": code})

    def record_rejected_turn(
        self,
        turn: AssembledAssistantTurn,
        *,
        code: str,
    ) -> None:
        """Keep rejected turn metadata while excluding invalid provider history."""

        self._record_event(
            "assistant_turn_rejected",
            {
                "code": code,
                "reasoning_characters": len(turn.reasoning or ""),
                "content_characters": len(turn.content),
                "tool_call_count": len(turn.tool_calls),
                "invalid_tool_call_count": len(turn.invalid_tool_calls),
            },
        )

    def mark_skill_loaded(self, name: str) -> None:
        """Record that one trusted skill body entered model history."""

        self._loaded_skills.add(name)
        self._record_event("skill_loaded", {"name": name})

    def next_observation_id(self) -> str:
        """Return a deterministic unique observation id within this session."""

        self._observation_counter += 1
        observation_id = (
            f"{self.session_id}:observation:{self._observation_counter}"
        )
        return observation_id

    def compact_oldest_exchange(self, *, keep_recent: int = 1) -> bool:
        """Atomically replace the oldest eligible full exchange in model view."""

        eligible_indexes = [
            index
            for index, segment in enumerate(self._segments)
            if segment.compactable and not segment.compacted
        ]
        if len(eligible_indexes) <= keep_recent:
            return False
        selected_index = eligible_indexes[0]
        self._segments[selected_index].compacted = True
        self.usage.compactions += 1
        self._record_event(
            "context_compaction_applied",
            {"segment_index": selected_index},
        )
        return True

    def record_context_estimate(self, estimated_tokens: int) -> None:
        """Update the peak admitted request estimate."""

        self.usage.estimated_context_tokens_peak = max(
            self.usage.estimated_context_tokens_peak,
            estimated_tokens,
        )
        self._record_event(
            "context_measured",
            {"estimated_tokens": estimated_tokens},
        )

    def record_tool_call(self, *, tool_name: str) -> None:
        """Advance the non-terminal tool counter with sanitized metadata."""

        self.usage.tool_calls += 1
        self._record_event(
            "tool_execution_started",
            {
                "tool_name": tool_name,
                "tool_call_count": self.usage.tool_calls,
            },
        )

    def record_subagent_started(self, *, subagent_id: str) -> None:
        """Advance the child counter and record lineage metadata."""

        self.usage.subagent_runs += 1
        self._record_event(
            "child_started",
            {
                "subagent_id": subagent_id,
                "subagent_run_count": self.usage.subagent_runs,
            },
        )

    def record_subagent_completed(
        self,
        *,
        subagent_id: str,
        status: str,
    ) -> None:
        """Record one bounded child terminal disposition."""

        self._record_event(
            "child_completed",
            {"subagent_id": subagent_id, "status": status},
        )

    def record_terminal(self, *, status: str, reason: str) -> None:
        """Record one deterministic or model-selected session disposition."""

        self._record_event(
            "session_terminalized",
            {"status": status, "reason": reason},
        )

    def _merge_provider_usage(self, usage: Mapping[str, int]) -> None:
        """Accumulate integer provider counters across model steps."""

        for key, value in usage.items():
            self.usage.provider_usage[key] = (
                self.usage.provider_usage.get(key, 0) + value
            )

    def _record_event(
        self,
        kind: str,
        metadata: Mapping[str, object],
    ) -> None:
        """Append one immutable indexed event metadata row."""

        event = ResolverSessionEvent(
            index=len(self._events),
            kind=kind,
            metadata=MappingProxyType(dict(metadata)),
        )
        self._events.append(event)
