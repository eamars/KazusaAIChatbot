"""Top-level RAG capability agent for recall evidence."""

from __future__ import annotations

import logging
from typing import Any

from kazusa_ai_chatbot.rag.helper_agent import BaseRAGHelperAgent
from kazusa_ai_chatbot.rag.recall.collectors.commitments import ActiveCommitmentCollector
from kazusa_ai_chatbot.rag.recall.collectors.history import HistoryEvidenceCollector
from kazusa_ai_chatbot.rag.recall.collectors.progress import ProgressCollector
from kazusa_ai_chatbot.rag.recall.collectors.scheduled_events import ScheduledEventCollector
from kazusa_ai_chatbot.rag.recall.contracts import (
    _ACTIVE_MODES,
    _CANDIDATE_CLAIM_LIMIT,
    _REQUIRED_CONTEXT_FIELDS,
    _agent_result,
    _clip_text,
    _dedupe_candidates,
    _missing_context_result,
    _mode_from_task,
    _result_payload,
)
from kazusa_ai_chatbot.rag.recall.review import (
    _conflict_notes,
    _freshness_basis,
    _has_source_conflict,
    _rank_candidates,
    _recall_slot_needs_candidate_review,
    _recall_type_for,
    _review_observation_candidates,
    _review_recall_candidates,
    _review_source_hints,
)
from kazusa_ai_chatbot.utils import text_or_empty

logger = logging.getLogger(__name__)

class RecallAgent(BaseRAGHelperAgent):
    """RAG helper agent for active agreements and ongoing commitments."""

    def __init__(self, *, cache_runtime=None) -> None:
        """Create the no-cache Recall helper.

        Args:
            cache_runtime: Accepted for interface compatibility; unused because
                Recall results depend on volatile progress and pending events.
        """

        super().__init__(
            name="recall_agent",
            cache_name="",
            cache_runtime=cache_runtime,
        )
        self.progress_collector = ProgressCollector()
        self.active_commitment_collector = ActiveCommitmentCollector()
        self.scheduled_event_collector = ScheduledEventCollector()
        self.history_evidence_collector = HistoryEvidenceCollector()

    async def run(
        self,
        task: str,
        context: dict[str, Any],
        max_attempts: int = 1,
    ) -> dict[str, Any]:
        """Recall one active agreement, commitment, or exact agreement history.

        Args:
            task: ``Recall:`` slot text emitted by the RAG2 initializer.
            context: Scoped RAG2 context containing platform, user, and time.
            max_attempts: Accepted for interface compatibility; Recall runs once.

        Returns:
            Standard RAG helper result with ``resolved``, ``result``,
            ``attempts``, and volatile no-cache metadata.
        """

        del max_attempts

        missing_fields = [
            field
            for field in _REQUIRED_CONTEXT_FIELDS
            if not text_or_empty(context.get(field))
        ]
        if missing_fields:
            result = _missing_context_result(missing_fields)
            return result

        mode = _mode_from_task(task)
        progress_candidates = self.progress_collector.collect(context)
        commitment_candidates = await self.active_commitment_collector.collect(
            context["global_user_id"]
        )
        scheduled_candidates = await self.scheduled_event_collector.collect(context)
        progress_unavailable = not progress_candidates

        candidates = _dedupe_candidates(
            [
                *progress_candidates,
                *commitment_candidates,
                *scheduled_candidates,
            ]
        )

        if mode in _ACTIVE_MODES and progress_unavailable:
            if commitment_candidates:
                candidates = _dedupe_candidates(
                    [
                        *commitment_candidates,
                        *scheduled_candidates,
                    ]
                )
            else:
                result_payload = _result_payload(
                    recall_type="active_episode_agreement",
                    freshness_basis=(
                        "active-episode state was unavailable and no active "
                        "commitment memory was found"
                    ),
                    candidates=candidates,
                )
                result = _agent_result(
                    resolved=False,
                    result_payload=result_payload,
                )
                return result

        history_required = mode == "exact_agreement_history"
        if not history_required:
            history_required = _has_source_conflict(candidates)
        if history_required:
            history_candidates = await self.history_evidence_collector.collect(context)
            candidates.extend(history_candidates)

        if not candidates:
            result_payload = _result_payload(
                recall_type="active_episode_agreement",
                freshness_basis="no recall evidence found",
                candidates=[],
            )
            result = _agent_result(resolved=False, result_payload=result_payload)
            return result

        ranked_candidates = _rank_candidates(mode, candidates)
        selected = ranked_candidates[0]
        conflicts = _conflict_notes(candidates)
        if _recall_slot_needs_candidate_review(
            task=task,
            mode=mode,
            progress_unavailable=progress_unavailable,
            candidates=ranked_candidates,
        ):
            review = await _review_recall_candidates(
                task=task,
                mode=mode,
                candidates=ranked_candidates,
            )
            confirmed_indexes = review["confirmed_candidate_indexes"]
            if not confirmed_indexes:
                observation_candidates = _review_observation_candidates(
                    ranked_candidates,
                    indexes=review["nearby_candidate_indexes"],
                )
                result_payload = _result_payload(
                    selected_summary="",
                    recall_type=mode,
                    primary_source="",
                    supporting_sources=[],
                    freshness_basis=(
                        "存在备用 recall 候选，但没有任何候选具备足够权威性来回答"
                        "请求的活跃召回槽位。"
                    ),
                    conflicts=conflicts,
                    candidates=ranked_candidates,
                    missing_context=["recall_evidence"],
                    observation_candidates=observation_candidates,
                    source_hints=_review_source_hints(review),
                    review_uncertainty=review["uncertainty"],
                )
                result = _agent_result(
                    resolved=False,
                    result_payload=result_payload,
                )
                return result

            selected = ranked_candidates[confirmed_indexes[0]]
            if review["summary"]:
                selected["claim"] = _clip_text(
                    review["summary"],
                    limit=_CANDIDATE_CLAIM_LIMIT,
                )

        supporting_sources = sorted(
            {
                candidate["source"]
                for candidate in ranked_candidates
                if candidate["source"] != selected["source"]
            }
        )
        recall_type = _recall_type_for(mode, selected, conflicts)
        freshness_basis = _freshness_basis(selected, mode, progress_unavailable)

        result_payload = _result_payload(
            selected_summary=selected["claim"],
            recall_type=recall_type,
            primary_source=selected["source"],
            supporting_sources=supporting_sources,
            freshness_basis=freshness_basis,
            conflicts=conflicts,
            candidates=ranked_candidates,
        )
        result = _agent_result(resolved=True, result_payload=result_payload)
        return result
