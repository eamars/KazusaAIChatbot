"""Top-level RAG capability agent for durable memory evidence."""

from __future__ import annotations

import logging
from typing import Any

from kazusa_ai_chatbot.config import RAG_SEARCH_SELECTED_SUMMARY_LIMIT
from kazusa_ai_chatbot.rag.evidence_coverage import coverage_allows_resolution
from kazusa_ai_chatbot.rag.helper_agent import BaseRAGHelperAgent
from kazusa_ai_chatbot.rag.memory_evidence.contracts import (
    _AGENT_NAME,
    _UNCACHED_REASON,
    _agent_result,
    _coverage_fields,
    _result_payload,
)
from kazusa_ai_chatbot.rag.memory_evidence.projection import (
    _memory_observation_candidates,
    _memory_rows,
    _nearby_memory_rows,
    _refs_from_rows,
    _summaries_from_rows,
)
from kazusa_ai_chatbot.rag.memory_evidence.selector import (
    _memory_coverage_task,
    _select_plan,
)
from kazusa_ai_chatbot.rag.memory_evidence.workers.persistent_search import (
    PersistentMemorySearchAgent,
)
from kazusa_ai_chatbot.rag.memory_evidence.workers.user_memory import UserMemoryEvidenceAgent
from kazusa_ai_chatbot.utils import text_or_empty

logger = logging.getLogger(__name__)

class MemoryEvidenceAgent(BaseRAGHelperAgent):
    """Top-level RAG helper for durable memory and world evidence."""

    def __init__(self, *, cache_runtime=None) -> None:
        """Create the uncached memory-evidence capability agent."""
        super().__init__(
            name=_AGENT_NAME,
            cache_name="",
            cache_runtime=cache_runtime,
        )
        self.search_agent = PersistentMemorySearchAgent(
            cache_runtime=cache_runtime
        )
        self.user_memory_agent = UserMemoryEvidenceAgent(
            cache_runtime=cache_runtime
        )

    async def run(
        self,
        task: str,
        context: dict[str, Any],
        max_attempts: int = 1,
    ) -> dict[str, Any]:
        """Resolve one durable memory evidence slot through one worker path."""

        del max_attempts

        plan = await _select_plan(task, context)
        primary_worker = plan["worker"]
        if primary_worker == "incompatible":
            reason = text_or_empty(plan["reason"]) or "unsupported"
            result = self._unresolved(
                task=task,
                source_policy=f"incompatible intent; use {reason}",
                missing_context=[f"incompatible_intent:{reason}"],
                primary_worker="",
                worker_payloads={},
            )
            return result

        worker = self._worker_for_name(primary_worker)
        worker_result = await worker.run(task, context, max_attempts=1)
        worker_payloads = {primary_worker: worker_result}
        memory_rows = _memory_rows(worker_result)
        nearby_rows = _nearby_memory_rows(worker_result)
        summaries = _summaries_from_rows(memory_rows)
        worker_resolved = bool(worker_result.get("resolved"))
        coverage_task = _memory_coverage_task(task)
        coverage, evidence_buckets = _coverage_fields(
            task=coverage_task,
            evidence_items=summaries,
            worker_resolved=worker_resolved,
        )
        confirmed_evidence = evidence_buckets["confirmed_evidence"]
        selected_summary = "\n".join(
            confirmed_evidence[:RAG_SEARCH_SELECTED_SUMMARY_LIMIT],
        )
        resolved_refs = _refs_from_rows(memory_rows)
        resolved = (
            worker_resolved
            and bool(summaries)
            and coverage_allows_resolution(coverage)
        )
        missing_context = [] if resolved else ["memory_evidence"]
        projection_rows = memory_rows
        evidence = confirmed_evidence
        observation_candidates: list[dict[str, Any]] = []
        source_hints: list[dict[str, Any]] = []
        if not resolved:
            selected_summary = ""
            resolved_refs = []
            projection_rows = []
            evidence = []
            observation_rows = memory_rows or nearby_rows
            observation_candidates = _memory_observation_candidates(
                observation_rows,
            )
            source_hints = [
                {
                    "kind": "memory",
                    "source": candidate["source"],
                }
                for candidate in observation_candidates
                if candidate.get("source")
            ]
        payload = _result_payload(
            selected_summary=selected_summary,
            primary_worker=primary_worker,
            source_policy=text_or_empty(plan["reason"]),
            resolved_refs=resolved_refs,
            projection_payload={"memory_rows": projection_rows},
            worker_payloads=worker_payloads,
            evidence=evidence,
            missing_context=missing_context,
            conflicts=[],
            observation_candidates=observation_candidates,
            source_hints=source_hints,
            coverage=coverage,
            confirmed_evidence=confirmed_evidence,
            partial_evidence=evidence_buckets["partial_evidence"],
            nearby_evidence=evidence_buckets["nearby_evidence"],
        )
        logger.info(
            f"{_AGENT_NAME} output: resolved={resolved} "
            f"primary_worker={primary_worker} "
            f"missing_context={missing_context} "
            f"selected_summary={payload['selected_summary']} "
            f"cache_reason={_UNCACHED_REASON}"
        )
        logger.debug(
            f"{_AGENT_NAME} debug: resolved_refs={resolved_refs} "
            f"projection_payload={payload['projection_payload']} "
            f"worker_payloads={worker_payloads}"
        )
        result = _agent_result(resolved=resolved, payload=payload)
        return result

    def _worker_for_name(self, worker_name: str) -> BaseRAGHelperAgent:
        """Return the configured memory worker for an approved name."""
        if worker_name == "user_memory_evidence_agent":
            return self.user_memory_agent
        return_value = self.search_agent
        return return_value

    def _unresolved(
        self,
        *,
        task: str,
        source_policy: str,
        missing_context: list[str],
        primary_worker: str,
        worker_payloads: dict[str, Any],
    ) -> dict[str, Any]:
        """Build an unresolved result without calling a memory source."""
        coverage_task = _memory_coverage_task(task)
        coverage, evidence_buckets = _coverage_fields(
            task=coverage_task,
            evidence_items=[],
            worker_resolved=False,
        )
        payload = _result_payload(
            selected_summary="",
            primary_worker=primary_worker,
            source_policy=source_policy,
            resolved_refs=[],
            projection_payload={"memory_rows": []},
            worker_payloads=worker_payloads,
            evidence=[],
            missing_context=missing_context,
            conflicts=[],
            observation_candidates=[],
            source_hints=[],
            coverage=coverage,
            confirmed_evidence=evidence_buckets["confirmed_evidence"],
            partial_evidence=evidence_buckets["partial_evidence"],
            nearby_evidence=evidence_buckets["nearby_evidence"],
        )
        logger.info(
            f"{_AGENT_NAME} output: resolved=False "
            f"primary_worker={primary_worker} "
            f"missing_context={missing_context} selected_summary='' "
            f"cache_reason={_UNCACHED_REASON}"
        )
        logger.debug(
            f"{_AGENT_NAME} debug: resolved_refs=[] "
            f"projection_payload={payload['projection_payload']} "
            f"worker_payloads={worker_payloads}"
        )
        result = _agent_result(resolved=False, payload=payload)
        return result
