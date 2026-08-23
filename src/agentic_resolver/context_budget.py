"""Deterministic request accounting, atomic compaction, and hard admission."""

from __future__ import annotations

import json
from dataclasses import dataclass

from agentic_resolver.contracts import AgenticResolverLimitsV1
from agentic_resolver.model import AgenticModelMessage, AgenticModelToolDefinition
from agentic_resolver.session import ResolverSession

CHARS_PER_ESTIMATED_TOKEN = 4


@dataclass(frozen=True)
class ContextAdmission:
    """One provider-ready request proven to fit the effective context window."""

    messages: tuple[AgenticModelMessage, ...]
    estimated_input_tokens: int
    estimated_total_tokens: int


class ContextBudget:
    """Measure complete model requests and compact old exchanges atomically."""

    def __init__(self, limits: AgenticResolverLimitsV1) -> None:
        self._limits = limits

    def prepare(
        self,
        session: ResolverSession,
        tools: tuple[AgenticModelToolDefinition, ...],
    ) -> ContextAdmission | None:
        """Return an admitted request or a hard-stop result after compaction."""

        while True:
            messages = session.model_history()
            estimated_input_tokens = estimate_request_tokens(messages, tools)
            estimated_total_tokens = (
                estimated_input_tokens
                + self._limits.completion_reserve_tokens
            )
            if estimated_input_tokens <= self._limits.input_ceiling_tokens:
                session.record_context_estimate(estimated_input_tokens)
                admission = ContextAdmission(
                    messages=messages,
                    estimated_input_tokens=estimated_input_tokens,
                    estimated_total_tokens=estimated_total_tokens,
                )
                return admission
            compacted = session.compact_oldest_exchange(keep_recent=1)
            if not compacted:
                return None


def estimate_request_tokens(
    messages: tuple[AgenticModelMessage, ...],
    tools: tuple[AgenticModelToolDefinition, ...],
) -> int:
    """Estimate canonical request tokens including opaque replay fields once."""

    payload = {
        "messages": [message.to_dict() for message in messages],
        "tools": [tool.to_dict() for tool in tools],
    }
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    estimated_tokens = estimate_tokens_from_characters(len(serialized))
    return estimated_tokens


def estimate_tokens_from_characters(character_count: int) -> int:
    """Apply the fixed ceiling(character count divided by four) estimate."""

    if character_count <= 0:
        return 0
    estimated_tokens = (
        character_count + CHARS_PER_ESTIMATED_TOKEN - 1
    ) // CHARS_PER_ESTIMATED_TOKEN
    return estimated_tokens
