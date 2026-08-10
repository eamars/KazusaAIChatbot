"""Deterministic policy after independent identity semantic judgments."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import cast

from kazusa_ai_chatbot.character_identity_growth import models
from kazusa_ai_chatbot.character_identity_growth.identity import (
    apply_identity_patches,
    dedupe_evidence_refs,
    evidence_counts,
)
from kazusa_ai_chatbot.character_identity_growth.validation import (
    validate_effective_identity,
    validate_evidence_ref,
    validate_identity_evidence_card,
    validate_identity_proposal_decision,
    validate_identity_review_decision,
)


def evaluate_identity_growth_policy(
    *,
    current_identity: Mapping[str, object],
    proposal: Mapping[str, object],
    review: Mapping[str, object],
    evidence_refs: Sequence[Mapping[str, object]],
    evidence_cards: Sequence[Mapping[str, object]],
    current_candidates: Sequence[Mapping[str, object]],
    current_revision_number: int,
    inferred_min_episodes: int,
    inferred_min_local_dates: int,
    inferred_promotions_on_local_date: int,
    max_inferred_promotions_per_local_day: int,
    reversal_cutoffs_by_path: Mapping[str, str],
) -> models.IdentityGrowthPolicyResultV1:
    """Apply provenance, privacy, cadence, and lifecycle policy."""

    validated_identity = validate_effective_identity(current_identity)
    _validate_policy_settings(
        current_revision_number=current_revision_number,
        inferred_min_episodes=inferred_min_episodes,
        inferred_min_local_dates=inferred_min_local_dates,
        inferred_promotions_on_local_date=(
            inferred_promotions_on_local_date
        ),
        max_inferred_promotions_per_local_day=(
            max_inferred_promotions_per_local_day
        ),
    )
    refs_by_id, cards_by_id = _validated_evidence(
        evidence_refs=evidence_refs,
        evidence_cards=evidence_cards,
    )
    candidates_by_id = _candidate_map(current_candidates)
    evidence_ref_ids = set(refs_by_id)
    candidate_ids = set(candidates_by_id)
    validated_proposal = validate_identity_proposal_decision(
        proposal,
        evidence_ref_ids=evidence_ref_ids,
        candidate_ids=candidate_ids,
    )
    validated_review = validate_identity_review_decision(
        review,
        proposal=validated_proposal,
        evidence_ref_ids=evidence_ref_ids,
        candidate_ids=candidate_ids,
    )

    if validated_proposal["action"] == "no_change":
        return _build_policy_result(
            status="no_change",
            candidate_status=None,
            candidate_id=None,
            change_kind=None,
            accepted_changes=[],
            semantic_summary=validated_review["character_owned_summary"],
            privacy_safe_evidence_summaries=[],
            evidence_refs=[],
            reversal_of_paths=[],
            fresh_post_revision_root_count=0,
            rebase_required=False,
            rejected_candidate_ids=validated_review[
                "rejected_candidate_ids"
            ],
            proposal_reason_code=validated_proposal["reason_code"],
            review_reason_code=validated_review["reason_code"],
            policy_reason_code="proposal_no_change",
        )

    selected_candidate = _selected_candidate(
        proposal=validated_proposal,
        candidates_by_id=candidates_by_id,
    )
    combined_refs = _combined_candidate_evidence(
        proposal=validated_proposal,
        refs_by_id=refs_by_id,
        selected_candidate=selected_candidate,
    )
    selected_candidate_id = validated_proposal["candidate_id"]
    if (
        validated_proposal["private_detail_risk"] == "high"
        or validated_review["private_detail_risk"] == "high"
    ):
        return _rejected_result(
            proposal=validated_proposal,
            review=validated_review,
            candidate_id=selected_candidate_id,
            evidence_refs=combined_refs,
            policy_reason_code="privacy_blocked",
        )
    if validated_review["verdict"] != "accept":
        if validated_review["coherence"] == "conflicting":
            policy_reason_code = "contradiction_blocked"
        else:
            policy_reason_code = "review_rejected"
        return _rejected_result(
            proposal=validated_proposal,
            review=validated_review,
            candidate_id=selected_candidate_id,
            evidence_refs=combined_refs,
            policy_reason_code=policy_reason_code,
        )

    change_kind = validated_review["accepted_change_kind"]
    if change_kind is None:
        raise ValueError("accepted review is missing change kind")
    if not _semantic_acceptance_approved(
        proposal=validated_proposal,
        review=validated_review,
        change_kind=change_kind,
    ):
        return _rejected_result(
            proposal=validated_proposal,
            review=validated_review,
            candidate_id=selected_candidate_id,
            evidence_refs=combined_refs,
            policy_reason_code="review_rejected",
        )
    try:
        apply_identity_patches(
            validated_identity,
            validated_review["accepted_changes"],
        )
    except ValueError:
        return _rejected_result(
            proposal=validated_proposal,
            review=validated_review,
            candidate_id=selected_candidate_id,
            evidence_refs=combined_refs,
            policy_reason_code="review_rejected",
        )

    reversal_paths, cutoff = _reversal_context(
        accepted_changes=validated_review["accepted_changes"],
        selected_candidate=selected_candidate,
        reversal_cutoffs_by_path=reversal_cutoffs_by_path,
    )
    fresh_refs = _fresh_refs_after_cutoff(
        combined_refs,
        cutoff=cutoff,
    )
    fresh_counts = evidence_counts(fresh_refs)
    rebase_required = _rebase_required(
        selected_candidate,
        current_revision_number=current_revision_number,
    )

    if change_kind == "explicit_self_redefinition":
        if not _has_explicit_character_evidence(
            validated_proposal,
            cards_by_id=cards_by_id,
        ):
            return _rejected_result(
                proposal=validated_proposal,
                review=validated_review,
                candidate_id=selected_candidate_id,
                evidence_refs=combined_refs,
                policy_reason_code="review_rejected",
            )
        return _accepted_result(
            status="revision_ready",
            candidate_status="ready",
            proposal=validated_proposal,
            review=validated_review,
            candidate_id=selected_candidate_id,
            change_kind=change_kind,
            evidence_refs=combined_refs,
            reversal_of_paths=reversal_paths,
            fresh_post_revision_root_count=(
                fresh_counts["distinct_episode_count"]
                if reversal_paths
                else 0
            ),
            rebase_required=rebase_required,
            policy_reason_code="candidate_ready",
        )

    cadence_refs = fresh_refs if reversal_paths else combined_refs
    cadence_counts = evidence_counts(cadence_refs)
    threshold_met = (
        cadence_counts["distinct_episode_count"] >= inferred_min_episodes
        and len(cadence_counts["distinct_local_dates"])
        >= inferred_min_local_dates
    )
    fresh_root_count = (
        fresh_counts["distinct_episode_count"]
        if reversal_paths
        else 0
    )
    if not threshold_met:
        return _accepted_result(
            status="candidate_updated",
            candidate_status="emerging",
            proposal=validated_proposal,
            review=validated_review,
            candidate_id=selected_candidate_id,
            change_kind=change_kind,
            evidence_refs=combined_refs,
            reversal_of_paths=reversal_paths,
            fresh_post_revision_root_count=fresh_root_count,
            rebase_required=rebase_required,
            policy_reason_code="candidate_emerging",
        )
    if (
        max_inferred_promotions_per_local_day == 0
        or inferred_promotions_on_local_date
        >= max_inferred_promotions_per_local_day
    ):
        return _accepted_result(
            status="deferred",
            candidate_status="emerging",
            proposal=validated_proposal,
            review=validated_review,
            candidate_id=selected_candidate_id,
            change_kind=change_kind,
            evidence_refs=combined_refs,
            reversal_of_paths=reversal_paths,
            fresh_post_revision_root_count=fresh_root_count,
            rebase_required=rebase_required,
            policy_reason_code="cadence_wait",
        )
    return _accepted_result(
        status="revision_ready",
        candidate_status="ready",
        proposal=validated_proposal,
        review=validated_review,
        candidate_id=selected_candidate_id,
        change_kind=change_kind,
        evidence_refs=combined_refs,
        reversal_of_paths=reversal_paths,
        fresh_post_revision_root_count=fresh_root_count,
        rebase_required=rebase_required,
        policy_reason_code="candidate_ready",
    )


def _validated_evidence(
    *,
    evidence_refs: Sequence[Mapping[str, object]],
    evidence_cards: Sequence[Mapping[str, object]],
) -> tuple[
    dict[str, models.IdentityEvidenceRefV1],
    dict[str, models.IdentityEvidenceCardV1],
]:
    """Validate and join prompt cards to repository references."""

    validated_refs = [
        validate_evidence_ref(ref)
        for ref in evidence_refs
    ]
    refs_by_id = {
        ref["evidence_ref_id"]: ref
        for ref in validated_refs
    }
    if len(refs_by_id) != len(validated_refs):
        raise ValueError("identity policy evidence refs must be unique")
    cards_by_id: dict[str, models.IdentityEvidenceCardV1] = {}
    for raw_card in evidence_cards:
        if not isinstance(raw_card, Mapping):
            raise ValueError("identity policy evidence card must be an object")
        evidence_ref_id = raw_card.get("evidence_ref_id")
        if not isinstance(evidence_ref_id, str):
            raise ValueError("identity policy evidence card requires ref id")
        evidence_ref = refs_by_id.get(evidence_ref_id)
        if evidence_ref is None:
            raise ValueError("identity policy evidence card has unknown ref")
        card = validate_identity_evidence_card(
            raw_card,
            evidence_ref=evidence_ref,
        )
        if evidence_ref_id in cards_by_id:
            raise ValueError("identity policy evidence cards must be unique")
        cards_by_id[evidence_ref_id] = card
    if set(cards_by_id) != set(refs_by_id):
        raise ValueError(
            "identity policy evidence cards and refs must match exactly"
        )
    return refs_by_id, cards_by_id


def _candidate_map(
    candidates: Sequence[Mapping[str, object]],
) -> dict[str, Mapping[str, object]]:
    """Return current candidates keyed by validated opaque id."""

    if len(candidates) > models.IDENTITY_CANDIDATE_PROMPT_LIMIT:
        raise ValueError("identity policy candidate limit exceeded")
    by_id: dict[str, Mapping[str, object]] = {}
    for candidate in candidates:
        candidate_id = candidate.get("candidate_id")
        if not isinstance(candidate_id, str) or not candidate_id.strip():
            raise ValueError("identity policy candidate requires candidate_id")
        if candidate_id in by_id:
            raise ValueError("identity policy candidates must be unique")
        by_id[candidate_id] = candidate
    return by_id


def _selected_candidate(
    *,
    proposal: models.IdentityProposalDecisionV1,
    candidates_by_id: Mapping[str, Mapping[str, object]],
) -> Mapping[str, object] | None:
    """Return the semantically selected current candidate."""

    candidate_id = proposal["candidate_id"]
    if candidate_id is None:
        return None
    candidate = candidates_by_id.get(candidate_id)
    if candidate is None:
        raise ValueError("identity policy selected candidate is unavailable")
    return candidate


def _combined_candidate_evidence(
    *,
    proposal: models.IdentityProposalDecisionV1,
    refs_by_id: Mapping[str, models.IdentityEvidenceRefV1],
    selected_candidate: Mapping[str, object] | None,
) -> list[models.IdentityEvidenceRefV1]:
    """Combine retained candidate roots with newly cited roots."""

    combined: list[Mapping[str, object]] = []
    if selected_candidate is not None:
        raw_existing_refs = selected_candidate.get("evidence_refs")
        if not isinstance(raw_existing_refs, list):
            raise ValueError(
                "identity policy selected candidate requires evidence_refs"
            )
        combined.extend(
            cast(list[Mapping[str, object]], raw_existing_refs)
        )
    combined.extend(
        refs_by_id[evidence_ref_id]
        for evidence_ref_id in proposal["evidence_ref_ids"]
    )
    return dedupe_evidence_refs(combined)


def _semantic_acceptance_approved(
    *,
    proposal: models.IdentityProposalDecisionV1,
    review: models.IdentityReviewDecisionV1,
    change_kind: models.AcceptedChangeKind,
) -> bool:
    """Return whether both semantic stages independently clear acceptance."""

    common_requirements_met = (
        proposal["identity_relevance"] == "durable"
        and review["identity_relevance"] == "durable"
        and proposal["global_applicability"] == "global"
        and review["global_applicability"] == "global"
        and proposal["confidence"] == "high"
        and review["review_confidence"] == "high"
        and review["coherence"] == "coherent"
    )
    if not common_requirements_met:
        return False
    expected_authorship = (
        "self_declared"
        if change_kind == "explicit_self_redefinition"
        else "inferred"
    )
    authorship_mismatch = (
        proposal["character_authorship"] != expected_authorship
        or review["character_authorship"] != expected_authorship
    )
    return not authorship_mismatch


def _has_explicit_character_evidence(
    proposal: models.IdentityProposalDecisionV1,
    *,
    cards_by_id: Mapping[str, models.IdentityEvidenceCardV1],
) -> bool:
    """Return whether one cited root has cognition and visible expression."""

    has_evidence = any(
        bool(cards_by_id[evidence_ref_id][
            "character_cognition_summary"
        ])
        and bool(cards_by_id[evidence_ref_id][
            "visible_self_expression_summary"
        ])
        for evidence_ref_id in proposal["evidence_ref_ids"]
    )
    return has_evidence


def _reversal_context(
    *,
    accepted_changes: Sequence[models.IdentityPatchV1],
    selected_candidate: Mapping[str, object] | None,
    reversal_cutoffs_by_path: Mapping[str, str],
) -> tuple[list[str], datetime | None]:
    """Validate reversal paths and return the latest required cutoff."""

    accepted_paths = {
        patch["path"]
        for patch in accepted_changes
    }
    candidate_reversal_paths: set[str] = set()
    if selected_candidate is not None:
        raw_paths = selected_candidate.get("reversal_of_paths")
        if not isinstance(raw_paths, list):
            raise ValueError(
                "identity policy selected candidate requires reversal paths"
            )
        for raw_path in raw_paths:
            if not isinstance(raw_path, str):
                raise ValueError("identity reversal paths must be strings")
            candidate_reversal_paths.add(raw_path)
    cutoff_paths = set(reversal_cutoffs_by_path)
    reversal_paths = candidate_reversal_paths.union(cutoff_paths)
    if not reversal_paths.issubset(accepted_paths):
        raise ValueError("identity reversal paths must be accepted paths")
    missing_cutoffs = sorted(
        candidate_reversal_paths.difference(cutoff_paths)
    )
    if missing_cutoffs:
        raise ValueError(
            "identity reversal paths require revision cutoffs: "
            f"{missing_cutoffs}"
        )
    parsed_cutoffs = [
        _parse_timezone_datetime(
            reversal_cutoffs_by_path[path],
            context=f"identity reversal cutoff {path}",
        )
        for path in sorted(reversal_paths)
    ]
    cutoff = max(parsed_cutoffs) if parsed_cutoffs else None
    return sorted(reversal_paths), cutoff


def _fresh_refs_after_cutoff(
    evidence_refs: Sequence[models.IdentityEvidenceRefV1],
    *,
    cutoff: datetime | None,
) -> list[models.IdentityEvidenceRefV1]:
    """Return roots captured strictly after a reversal cutoff."""

    if cutoff is None:
        return list(evidence_refs)
    fresh = [
        ref
        for ref in evidence_refs
        if _parse_timezone_datetime(
            ref["captured_at"],
            context="identity evidence captured_at",
        ) > cutoff
    ]
    return fresh


def _rebase_required(
    selected_candidate: Mapping[str, object] | None,
    *,
    current_revision_number: int,
) -> bool:
    """Return whether an accepted existing candidate has a stale base."""

    if selected_candidate is None:
        return False
    base_revision_number = selected_candidate.get("base_revision_number")
    _require_nonnegative_integer(
        base_revision_number,
        context="identity candidate base_revision_number",
    )
    return base_revision_number != current_revision_number


def _accepted_result(
    *,
    status: models.IdentityPolicyStatus,
    candidate_status: models.CandidateStatus,
    proposal: models.IdentityProposalDecisionV1,
    review: models.IdentityReviewDecisionV1,
    candidate_id: str | None,
    change_kind: models.AcceptedChangeKind,
    evidence_refs: list[models.IdentityEvidenceRefV1],
    reversal_of_paths: list[str],
    fresh_post_revision_root_count: int,
    rebase_required: bool,
    policy_reason_code: str,
) -> models.IdentityGrowthPolicyResultV1:
    """Build one accepted emerging, deferred, or ready result."""

    return _build_policy_result(
        status=status,
        candidate_status=candidate_status,
        candidate_id=candidate_id,
        change_kind=change_kind,
        accepted_changes=review["accepted_changes"],
        semantic_summary=review["character_owned_summary"],
        privacy_safe_evidence_summaries=review[
            "privacy_safe_evidence_summaries"
        ],
        evidence_refs=evidence_refs,
        reversal_of_paths=reversal_of_paths,
        fresh_post_revision_root_count=fresh_post_revision_root_count,
        rebase_required=rebase_required,
        rejected_candidate_ids=review["rejected_candidate_ids"],
        proposal_reason_code=proposal["reason_code"],
        review_reason_code=review["reason_code"],
        policy_reason_code=policy_reason_code,
    )


def _rejected_result(
    *,
    proposal: models.IdentityProposalDecisionV1,
    review: models.IdentityReviewDecisionV1,
    candidate_id: str | None,
    evidence_refs: list[models.IdentityEvidenceRefV1],
    policy_reason_code: str,
) -> models.IdentityGrowthPolicyResultV1:
    """Build one semantic, contradiction, or privacy rejection."""

    return _build_policy_result(
        status="rejected",
        candidate_status="rejected",
        candidate_id=candidate_id,
        change_kind=None,
        accepted_changes=[],
        semantic_summary=review["character_owned_summary"],
        privacy_safe_evidence_summaries=[],
        evidence_refs=evidence_refs,
        reversal_of_paths=[],
        fresh_post_revision_root_count=0,
        rebase_required=False,
        rejected_candidate_ids=review["rejected_candidate_ids"],
        proposal_reason_code=proposal["reason_code"],
        review_reason_code=review["reason_code"],
        policy_reason_code=policy_reason_code,
    )


def _build_policy_result(
    *,
    status: models.IdentityPolicyStatus,
    candidate_status: models.CandidateStatus | None,
    candidate_id: str | None,
    change_kind: models.AcceptedChangeKind | None,
    accepted_changes: list[models.IdentityPatchV1],
    semantic_summary: str,
    privacy_safe_evidence_summaries: list[str],
    evidence_refs: list[models.IdentityEvidenceRefV1],
    reversal_of_paths: list[str],
    fresh_post_revision_root_count: int,
    rebase_required: bool,
    rejected_candidate_ids: list[str],
    proposal_reason_code: str,
    review_reason_code: str,
    policy_reason_code: str,
) -> models.IdentityGrowthPolicyResultV1:
    """Build one closed policy result with deterministic lineage counts."""

    counts = evidence_counts(evidence_refs)
    result: models.IdentityGrowthPolicyResultV1 = {
        "status": status,
        "candidate_status": candidate_status,
        "candidate_id": candidate_id,
        "change_kind": change_kind,
        "accepted_changes": list(accepted_changes),
        "semantic_summary": semantic_summary,
        "privacy_safe_evidence_summaries": list(
            privacy_safe_evidence_summaries
        ),
        "evidence_refs": list(evidence_refs),
        "distinct_episode_count": counts["distinct_episode_count"],
        "distinct_local_dates": counts["distinct_local_dates"],
        "source_scope_kinds": sorted({
            ref["scope_kind"]
            for ref in evidence_refs
        }),
        "claimed_root_episode_ids": sorted({
            ref["root_episode_id"]
            for ref in evidence_refs
        }),
        "reversal_of_paths": list(reversal_of_paths),
        "fresh_post_revision_root_count": fresh_post_revision_root_count,
        "rebase_required": rebase_required,
        "rejected_candidate_ids": list(rejected_candidate_ids),
        "proposal_reason_code": proposal_reason_code,
        "review_reason_code": review_reason_code,
        "policy_reason_code": policy_reason_code,
    }
    return result


def _validate_policy_settings(
    *,
    current_revision_number: object,
    inferred_min_episodes: object,
    inferred_min_local_dates: object,
    inferred_promotions_on_local_date: object,
    max_inferred_promotions_per_local_day: object,
) -> None:
    """Validate the bounded deterministic policy settings."""

    _require_nonnegative_integer(
        current_revision_number,
        context="current_revision_number",
    )
    _require_bounded_integer(
        inferred_min_episodes,
        context="inferred_min_episodes",
        minimum=2,
        maximum=8,
    )
    _require_bounded_integer(
        inferred_min_local_dates,
        context="inferred_min_local_dates",
        minimum=1,
        maximum=7,
    )
    if inferred_min_local_dates > inferred_min_episodes:
        raise ValueError(
            "inferred_min_local_dates cannot exceed inferred_min_episodes"
        )
    _require_nonnegative_integer(
        inferred_promotions_on_local_date,
        context="inferred_promotions_on_local_date",
    )
    _require_bounded_integer(
        max_inferred_promotions_per_local_day,
        context="max_inferred_promotions_per_local_day",
        minimum=0,
        maximum=3,
    )


def _require_nonnegative_integer(value: object, *, context: str) -> None:
    """Require one non-boolean nonnegative integer."""

    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
    ):
        raise ValueError(f"{context} must be a non-negative integer")


def _require_bounded_integer(
    value: object,
    *,
    context: str,
    minimum: int,
    maximum: int,
) -> None:
    """Require one non-boolean integer within closed bounds."""

    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not minimum <= value <= maximum
    ):
        raise ValueError(
            f"{context} must be between {minimum} and {maximum}"
        )


def _parse_timezone_datetime(value: object, *, context: str) -> datetime:
    """Parse one timezone-aware ISO datetime."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} must be an ISO datetime")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{context} must be an ISO datetime") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{context} must include a timezone")
    return parsed
