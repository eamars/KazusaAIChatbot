"""Top-level RAG capability agent for conversation-history evidence."""

from __future__ import annotations

import logging
from typing import Any

from kazusa_ai_chatbot.config import RAG_SEARCH_SELECTED_SUMMARY_LIMIT
from kazusa_ai_chatbot.rag.conversation_evidence.contracts import (
    _AGENT_NAME,
    _UNCACHED_REASON,
    _agent_result,
    _result_payload,
)
from kazusa_ai_chatbot.rag.conversation_evidence.projection import (
    _confirmed_retrieval_coverage,
    _conversation_projection_source,
    _coverage_confirms_retrieval_task,
    _coverage_fields,
    _projection_evidence_items,
    _projection_evidence_packets,
    _projection_from_worker,
    _projection_has_relation,
    _relation_requirement,
)
from kazusa_ai_chatbot.rag.conversation_evidence.selector import (
    _coverage_requires_value_evidence,
    _first_person_ref,
    _select_plan,
    _speaker_scope,
    _worker_context,
)
from kazusa_ai_chatbot.rag.conversation_evidence.workers.aggregate import (
    ConversationAggregateAgent,
)
from kazusa_ai_chatbot.rag.conversation_evidence.workers.filter import ConversationFilterAgent
from kazusa_ai_chatbot.rag.conversation_evidence.workers.search import ConversationSearchAgent
from kazusa_ai_chatbot.rag.evidence_coverage import (
    coverage_allows_resolution,
    evidence_buckets_for_coverage,
)
from kazusa_ai_chatbot.rag.helper_agent import BaseRAGHelperAgent
from kazusa_ai_chatbot.utils import text_or_empty

logger = logging.getLogger(__name__)

class ConversationEvidenceAgent(BaseRAGHelperAgent):
    """Top-level RAG helper for evidence from conversation history."""

    def __init__(self, *, cache_runtime=None) -> None:
        """Create the uncached conversation-evidence capability agent."""

        super().__init__(
            name=_AGENT_NAME,
            cache_name="",
            cache_runtime=cache_runtime,
        )
        self.search_agent = ConversationSearchAgent(cache_runtime=cache_runtime)
        self.filter_agent = ConversationFilterAgent(cache_runtime=cache_runtime)
        self.aggregate_agent = ConversationAggregateAgent(
            cache_runtime=cache_runtime
        )

    async def run(
        self,
        task: str,
        context: dict[str, Any],
        max_attempts: int = 1,
    ) -> dict[str, Any]:
        """Resolve one conversation evidence slot through one worker path."""

        del max_attempts

        plan = await _select_plan(task, context)
        primary_worker = plan["worker"]
        speaker_scope = _speaker_scope(task)
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

        person_ref = _first_person_ref(context)
        if bool(plan["requires_person_ref"]) and person_ref is None:
            result = self._unresolved(
                task=task,
                source_policy="structured person ref required but missing",
                missing_context=["person_ref"],
                primary_worker=primary_worker,
                worker_payloads={},
            )
            return result

        effective_person_ref = person_ref if bool(plan["requires_person_ref"]) else None
        worker = self._worker_for_name(primary_worker)
        context_for_worker = _worker_context(
            context,
            effective_person_ref,
            speaker_scope,
        )
        worker_result = await worker.run(
            task,
            context_for_worker,
            max_attempts=1,
        )
        worker_payloads = {primary_worker: worker_result}
        projection, exclusion_counts = _projection_from_worker(
            primary_worker,
            worker_result,
            context_for_worker,
        )
        excluded_count = (
            exclusion_counts["conversation_row_id"]
            + exclusion_counts["platform_message_id"]
        )
        relation_requirement = _relation_requirement(task)
        evidence_packets = _projection_evidence_packets(
            projection,
            relation_required=bool(relation_requirement),
        )
        summaries = _projection_evidence_items(
            projection,
            packets=evidence_packets,
        )
        projection_rows = projection["rows"]
        worker_resolved = bool(worker_result.get("resolved"))
        requires_value_evidence = _coverage_requires_value_evidence(
            task,
            context,
        )
        coverage, evidence_buckets = _coverage_fields(
            task=task,
            evidence_items=summaries,
            worker_resolved=worker_resolved,
            requires_value_evidence=requires_value_evidence,
        )
        coverage_confirms_retrieval = _coverage_confirms_retrieval_task(
            task,
            coverage,
        )
        if coverage_confirms_retrieval:
            coverage = _confirmed_retrieval_coverage(coverage)
            evidence_buckets = evidence_buckets_for_coverage(
                coverage,
                summaries,
            )
        confirmed_evidence = evidence_buckets["confirmed_evidence"]
        partial_evidence = evidence_buckets["partial_evidence"]
        nearby_evidence = evidence_buckets["nearby_evidence"]
        legacy_evidence = confirmed_evidence
        if not legacy_evidence:
            legacy_evidence = partial_evidence or nearby_evidence
        selected_summary = "\n".join(
            legacy_evidence[:RAG_SEARCH_SELECTED_SUMMARY_LIMIT],
        )
        resolved_refs = projection["resolved_refs"]
        relation_available = _projection_has_relation(
            projection,
            relation_requirement,
        )
        resolved = (
            bool(summaries)
            and coverage_allows_resolution(coverage)
            and (worker_resolved or coverage_confirms_retrieval)
            and relation_available
        )
        if resolved:
            missing_context = []
        elif relation_requirement and not relation_available:
            missing_context = [f"conversation_relation:{relation_requirement}"]
        else:
            missing_context = ["conversation_evidence"]
        observation_candidates = []
        source_hints: list[dict[str, Any]] = []
        if summaries and not resolved:
            observation_candidates = [
                {
                    "content": row["summary"],
                    "source": _conversation_projection_source(row),
                }
                for row in projection_rows[:RAG_SEARCH_SELECTED_SUMMARY_LIMIT]
            ]
            source_hints = [
                {
                    "kind": "conversation",
                    "source": candidate["source"],
                }
                for candidate in observation_candidates
                if candidate.get("source")
            ]
        payload = _result_payload(
            selected_summary=selected_summary,
            primary_worker=primary_worker,
            supporting_workers=[],
            source_policy=text_or_empty(plan["reason"]),
            resolved_refs=resolved_refs,
            projection_payload={
                "summaries": projection["summaries"],
                "rows": projection_rows,
                "packets": evidence_packets,
            },
            worker_payloads=worker_payloads,
            evidence=legacy_evidence,
            missing_context=missing_context,
            conflicts=[],
            observation_candidates=observation_candidates,
            source_hints=source_hints,
            coverage=coverage,
            confirmed_evidence=confirmed_evidence,
            partial_evidence=partial_evidence,
            nearby_evidence=nearby_evidence,
        )
        logger.info(
            f"{_AGENT_NAME} output: resolved={resolved} "
            f"primary_worker={primary_worker} "
            f"missing_context={missing_context} "
            f"selected_summary={payload['selected_summary']} "
            f"cache_reason={_UNCACHED_REASON}"
        )
        if excluded_count:
            logger.info(
                f"{_AGENT_NAME} active-turn rows excluded: "
                f"primary_worker={primary_worker} "
                f"excluded_active_turn_rows={excluded_count} "
                "excluded_by_conversation_row_id="
                f"{exclusion_counts['conversation_row_id']} "
                "excluded_by_platform_message_id="
                f"{exclusion_counts['platform_message_id']}"
            )
        logger.debug(
            f"{_AGENT_NAME} debug: resolved_refs={resolved_refs} "
            f"projection_payload={payload['projection_payload']} "
            f"worker_payloads={worker_payloads}"
        )
        result = _agent_result(resolved=resolved, payload=payload)
        return result

    def _worker_for_name(self, worker_name: str) -> BaseRAGHelperAgent:
        """Return the configured worker instance for an approved name."""
        if worker_name == "conversation_filter_agent":
            return self.filter_agent
        if worker_name == "conversation_aggregate_agent":
            return self.aggregate_agent
        return self.search_agent

    def _unresolved(
        self,
        *,
        task: str,
        source_policy: str,
        missing_context: list[str],
        primary_worker: str,
        worker_payloads: dict[str, Any],
    ) -> dict[str, Any]:
        """Build an unresolved result without calling another source."""
        coverage, evidence_buckets = _coverage_fields(
            task=task,
            evidence_items=[],
            worker_resolved=False,
        )
        payload = _result_payload(
            selected_summary="",
            primary_worker=primary_worker,
            supporting_workers=[],
            source_policy=source_policy,
            resolved_refs=[],
            projection_payload={"summaries": [], "packets": []},
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
