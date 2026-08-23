"""Foreground same-runtime child construction and bounded result projection."""

from __future__ import annotations

import json
import uuid
from dataclasses import replace
from typing import TYPE_CHECKING

from agentic_resolver.contracts import (
    AgenticResolverContractError,
    AgenticResolverRequestV1,
    AgenticResolverSubagentEvidenceV1,
    AgenticResolverSubagentResultV1,
    AgenticResolverSubagentTaskV1,
)
from agentic_resolver.session import ResolverSession

if TYPE_CHECKING:
    from agentic_resolver.runtime import AgenticResolverRuntime


class SubagentRunner:
    """Run isolated depth-one children through the owning runtime."""

    def __init__(self, runtime: AgenticResolverRuntime) -> None:
        self._runtime = runtime

    async def run(
        self,
        *,
        parent_session: ResolverSession,
        task: AgenticResolverSubagentTaskV1,
        deadline: float,
    ) -> AgenticResolverSubagentResultV1:
        """Run one child under the root cap and remaining wall-clock deadline."""

        if parent_session.depth != 0:
            raise AgenticResolverContractError(
                "child sessions cannot create subagents",
                code="subagent_depth_exceeded",
            )
        limits = self._runtime.limits
        if parent_session.usage.subagent_runs >= limits.max_subagent_runs:
            raise AgenticResolverContractError(
                "root subagent run cap reached",
                code="subagent_cap_reached",
            )
        subagent_id = f"subagent-{uuid.uuid4().hex}"
        parent_session.record_subagent_started(subagent_id=subagent_id)
        request = AgenticResolverRequestV1(
            objective=task.objective,
            context=task.context,
        )
        child_result = await self._runtime._resolve_child(
            request=request,
            task=task,
            subagent_id=subagent_id,
            parent_session_id=parent_session.session_id,
            deadline=deadline,
        )
        observation_id = parent_session.next_observation_id()
        projected_evidence = tuple(
            AgenticResolverSubagentEvidenceV1.from_terminal_evidence(evidence)
            for evidence in child_result.evidence
        )
        projected = AgenticResolverSubagentResultV1(
            subagent_id=subagent_id,
            observation_id=observation_id,
            description=task.description,
            status=child_result.status,
            summary=child_result.summary,
            evidence=projected_evidence,
            remaining_needs=child_result.remaining_needs,
        )
        bounded = _bounded_child_result(
            projected,
            maximum_characters=limits.max_subagent_result_characters,
        )
        parent_session.record_subagent_completed(
            subagent_id=subagent_id,
            status=bounded.status,
        )
        return bounded


def _bounded_child_result(
    result: AgenticResolverSubagentResultV1,
    *,
    maximum_characters: int,
) -> AgenticResolverSubagentResultV1:
    """Shrink optional child details until the typed JSON result fits."""

    candidate = result
    if _serialized_length(candidate) <= maximum_characters:
        return candidate
    while candidate.evidence:
        candidate = replace(candidate, evidence=candidate.evidence[:-1])
        if _serialized_length(candidate) <= maximum_characters:
            return candidate
    while candidate.remaining_needs:
        candidate = replace(
            candidate,
            remaining_needs=candidate.remaining_needs[:-1],
        )
        if _serialized_length(candidate) <= maximum_characters:
            return candidate
    for summary_limit in (2_000, 1_000, 500, 200, 100, 50):
        candidate = replace(candidate, summary=candidate.summary[:summary_limit])
        if _serialized_length(candidate) <= maximum_characters:
            return candidate
    raise AgenticResolverContractError(
        "child result cannot fit the configured projection bound",
        code="subagent_result_too_large",
    )


def _serialized_length(result: AgenticResolverSubagentResultV1) -> int:
    """Return the canonical character count for a parent-visible child result."""

    serialized = json.dumps(
        result.to_dict(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    serialized_length = len(serialized)
    return serialized_length
