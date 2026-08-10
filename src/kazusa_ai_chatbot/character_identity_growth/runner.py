"""Single background owner for character identity growth evaluation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import logging

from kazusa_ai_chatbot.character_identity_growth import models
from kazusa_ai_chatbot.character_identity_growth.llm import (
    IdentityStageError,
    IdentityStageResult,
    propose_identity_growth,
    review_identity_growth,
)
from kazusa_ai_chatbot.character_identity_growth.policy import (
    evaluate_identity_growth_policy,
)
from kazusa_ai_chatbot.character_identity_growth.projection import (
    build_identity_proposal_input,
    build_identity_review_input,
)
from kazusa_ai_chatbot.character_identity_growth.validation import (
    validate_effective_identity,
    validate_evidence_ref,
    validate_identity_evidence_card,
)
from kazusa_ai_chatbot.config import (
    CHARACTER_GLOBAL_USER_ID,
    CHARACTER_IDENTITY_GROWTH_ENABLED,
    CHARACTER_IDENTITY_GROWTH_INFERRED_MIN_EPISODES,
    CHARACTER_IDENTITY_GROWTH_INFERRED_MIN_LOCAL_DATES,
    CHARACTER_IDENTITY_GROWTH_MAX_INFERRED_PROMOTIONS_PER_LOCAL_DAY,
    CHARACTER_IDENTITY_GROWTH_PROMPT_CHAR_BUDGET,
)
from kazusa_ai_chatbot.db.character_identity_growth import (
    CANDIDATE_SCHEMA_VERSION,
    RUN_SCHEMA_VERSION,
    count_inferred_identity_promotions_on_local_date,
    get_current_identity,
    get_growth_run,
    insert_growth_candidate,
    insert_growth_run,
    list_current_growth_candidates,
    list_identity_revisions,
    list_post_commit_pending_growth_runs,
    promote_ready_candidate,
    complete_growth_run_post_commit,
    record_growth_run_post_commit_failure,
    reject_growth_candidates,
    update_growth_candidate,
)
from kazusa_ai_chatbot.event_logging import (
    record_character_identity_growth_event,
)
from kazusa_ai_chatbot.rag.cache2_events import CacheInvalidationEvent
from kazusa_ai_chatbot.rag.cache2_runtime import get_rag_cache2_runtime

logger = logging.getLogger(__name__)


async def reconcile_identity_growth_post_commit(
    *,
    character_id: str | None = None,
    run_id: str | None = None,
    limit: int = 100,
) -> dict[str, int]:
    """Replay revision-keyed cache invalidation and promotion telemetry."""

    if run_id is not None:
        run = await get_growth_run(run_id=run_id)
        pending_runs = (
            [run]
            if (
                run is not None
                and run["lifecycle_state"] == "post_commit_pending"
            )
            else []
        )
    else:
        pending_runs = await list_post_commit_pending_growth_runs(
            character_id=character_id,
            limit=limit,
        )

    completed_count = 0
    failed_count = 0
    for run in pending_runs:
        if run is None:
            continue
        normalized_character_id = str(run["character_id"])
        revision_number = run["promoted_revision_number"]
        if not isinstance(revision_number, int):
            raise ValueError(
                "post-commit identity run requires a promoted revision"
            )
        try:
            await get_rag_cache2_runtime().invalidate(
                CacheInvalidationEvent(
                    source="character_identity",
                    global_user_id=normalized_character_id,
                    reason=f"identity_revision:{revision_number}",
                )
            )
            event_result = await record_character_identity_growth_event(
                event_type="promotion",
                stage="post_commit",
                reason_code="revision_promoted",
                status="completed",
                correlation_id=str(run["correlation_id"]),
                run_id=str(run["run_id"]),
                revision_number=revision_number,
            )
            if not event_result["accepted"]:
                raise RuntimeError(
                    "identity promotion event was not persisted"
                )
            await complete_growth_run_post_commit(
                run_id=str(run["run_id"]),
                character_id=normalized_character_id,
                revision_number=revision_number,
            )
        except Exception as exc:
            failed_count += 1
            try:
                await record_growth_run_post_commit_failure(
                    run_id=str(run["run_id"]),
                    character_id=normalized_character_id,
                    revision_number=revision_number,
                )
            except Exception as persistence_exc:
                logger.exception(
                    "Identity post-commit retry evidence failed: "
                    f"{type(persistence_exc).__name__}: {persistence_exc}"
                )
            logger.exception(
                "Identity post-commit reconciliation failed: "
                f"run={run['run_id']} revision={revision_number} "
                f"error={type(exc).__name__}: {exc}"
            )
            continue
        completed_count += 1
    return {
        "completed_count": completed_count,
        "failed_count": failed_count,
    }


async def evaluate_episode_identity_growth(
    *,
    settled_episode: Mapping[str, object],
    current_revision: Mapping[str, object],
) -> models.IdentityGrowthEvaluationResultV1:
    """Evaluate one settled episode outside the foreground response path."""

    return await _evaluate_identity_growth(
        run_kind="episode",
        source=settled_episode,
        current_revision=current_revision,
        dry_run=False,
        enable_revision_writes=True,
        now=None,
    )


async def run_reflection_identity_growth_pass(
    *,
    character_local_date: str,
    source_reflection_run_ids: Sequence[str],
    dry_run: bool,
    enable_revision_writes: bool,
    now: datetime | None = None,
) -> models.IdentityGrowthEvaluationResultV1:
    """Evaluate daily reflection evidence through the same identity owner."""

    from kazusa_ai_chatbot.reflection_cycle import (
        repository as reflection_repository,
    )

    normalized_date = _require_local_date(character_local_date)
    source_ids = _sorted_unique_text(
        source_reflection_run_ids,
        context="source_reflection_run_ids",
    )
    documents: list[Mapping[str, object]] = []
    for run_id in source_ids:
        document = await reflection_repository.reflection_run_by_id(run_id)
        if document is None:
            continue
        if (
            str(document.get("run_kind", "")) != "daily_channel"
            or str(document.get("character_local_date", ""))
            != normalized_date
            or str(document.get("status", "")) != "succeeded"
        ):
            continue
        documents.append(document)

    refs, cards = _build_reflection_identity_evidence(
        documents,
        character_local_date=normalized_date,
    )
    current_revision = await get_current_identity(
        character_id=CHARACTER_GLOBAL_USER_ID,
    )
    correlation_id = _opaque_identifier(
        "identity-reflection-correlation",
        {
            "character_local_date": normalized_date,
            "source_reflection_run_ids": source_ids,
        },
    )
    source = {
        "correlation_id": correlation_id,
        "llm_trace_id": correlation_id,
        "evidence_refs": refs,
        "evidence_cards": cards,
    }
    return await _evaluate_identity_growth(
        run_kind="daily_reflection",
        source=source,
        current_revision=current_revision,
        dry_run=dry_run,
        enable_revision_writes=enable_revision_writes,
        now=now,
    )


def _build_reflection_identity_evidence(
    documents: Sequence[Mapping[str, object]],
    *,
    character_local_date: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Project daily reflection documents onto repository-rooted cards."""

    _require_local_date(character_local_date)
    pairs: list[tuple[dict[str, object], dict[str, object]]] = []
    for document in documents:
        run_id = _optional_text(document.get("run_id"))
        if not run_id:
            continue
        output = document.get("output")
        if not isinstance(output, Mapping):
            continue
        day_summary = _bounded_text(output.get("day_summary"), required=True)
        if not day_summary:
            continue
        cognition_summary = _conversation_quality_summary(output)
        derivative_ids = _reflection_derivative_ids(document, run_id=run_id)
        raw_roots = document.get("source_episode_refs", [])
        if not isinstance(raw_roots, list):
            raise ValueError("source_episode_refs must be a list")
        for raw_root in raw_roots:
            if not isinstance(raw_root, Mapping):
                raise ValueError("source_episode_refs entries must be objects")
            root = _validate_reflection_root(raw_root)
            evidence_ref_id = _opaque_identifier(
                "identity-evidence",
                {
                    "root_episode_id": root["root_episode_id"],
                    "source_reflection_run_id": run_id,
                },
            )
            evidence_ref = {
                "schema_version": models.IDENTITY_EVIDENCE_SCHEMA_VERSION,
                "evidence_ref_id": evidence_ref_id,
                "root_episode_id": root["root_episode_id"],
                "correlation_id": root["correlation_id"],
                "source_kind": "daily_reflection",
                "derived_reflection_run_ids": derivative_ids,
                "character_local_date": root["character_local_date"],
                "scope_kind": root["scope_kind"],
                "captured_at": root["captured_at"],
            }
            card = {
                "schema_version": (
                    models.IDENTITY_EVIDENCE_CARD_SCHEMA_VERSION
                ),
                "evidence_ref_id": evidence_ref_id,
                "source_kind": "daily_reflection",
                "character_local_date": root["character_local_date"],
                "scope_kind": root["scope_kind"],
                "decontextualized_event": day_summary,
                "character_cognition_summary": cognition_summary,
                "visible_self_expression_summary": "",
            }
            pairs.append((evidence_ref, card))

    pairs.sort(
        key=lambda pair: (
            str(pair[0]["captured_at"]),
            str(pair[0]["root_episode_id"]),
            str(pair[0]["evidence_ref_id"]),
        )
    )
    bounded_pairs = pairs[:models.IDENTITY_EVIDENCE_CARD_LIMIT]
    return (
        [deepcopy(pair[0]) for pair in bounded_pairs],
        [deepcopy(pair[1]) for pair in bounded_pairs],
    )


async def _evaluate_identity_growth(
    *,
    run_kind: str,
    source: Mapping[str, object],
    current_revision: Mapping[str, object],
    dry_run: bool,
    enable_revision_writes: bool,
    now: datetime | None,
) -> models.IdentityGrowthEvaluationResultV1:
    """Run proposal, review, policy, persistence, and optional promotion."""

    (
        character_id,
        revision_number,
        current_identity,
    ) = _current_revision_fields(current_revision)
    correlation_id = _correlation_id(source)
    refs, cards = _validated_source_evidence(source)
    run_id = _growth_run_id(
        run_kind=run_kind,
        character_id=character_id,
        revision_number=revision_number,
        correlation_id=correlation_id,
        evidence_refs=refs,
    )
    existing_run = await get_growth_run(run_id=run_id)
    if existing_run is not None:
        return _evaluation_from_run(existing_run)
    if not CHARACTER_IDENTITY_GROWTH_ENABLED:
        return _not_routed_result(
            run_id=run_id,
            base_revision_number=revision_number,
        )

    timestamp = _timestamp(now)
    if not refs:
        run = _growth_run_document(
            run_id=run_id,
            run_kind=run_kind,
            character_id=character_id,
            base_revision_number=revision_number,
            correlation_id=correlation_id,
            root_episode_ids=[],
            source_evidence_count=0,
            attempt_count_by_stage={"proposal": 0, "review": 0},
            lifecycle_state="complete",
            disposition="no_change",
            proposal_reason_code="no_eligible_evidence",
            review_reason_code="no_eligible_evidence",
            policy_reason_code="no_eligible_evidence",
            persistence_reason_code="no_eligible_evidence",
            candidate_id=None,
            validation_error_codes=[],
            started_at=timestamp,
            completed_at=timestamp,
        )
        if not dry_run:
            run = await insert_growth_run(run)
        return _evaluation_from_run(run)

    current_candidates = await list_current_growth_candidates(
        character_id=character_id,
        base_revision_number=revision_number,
    )
    proposal_input = build_identity_proposal_input(
        current_identity=current_identity,
        evidence_refs=refs,
        evidence_cards=cards,
        current_candidates=current_candidates,
    )
    trace_id = _optional_text(source.get("llm_trace_id")) or correlation_id
    proposal_result: IdentityStageResult | None = None
    try:
        proposal_result = await propose_identity_growth(
            proposal_input,
            trace_id=trace_id,
            prompt_char_budget=CHARACTER_IDENTITY_GROWTH_PROMPT_CHAR_BUDGET,
        )
        review_input = build_identity_review_input(
            proposal_input=proposal_input,
            proposal=proposal_result.decision,
        )
        review_result = await review_identity_growth(
            review_input,
            trace_id=trace_id,
            prompt_char_budget=CHARACTER_IDENTITY_GROWTH_PROMPT_CHAR_BUDGET,
        )
    except IdentityStageError as exc:
        return await _persist_stage_failure(
            exc=exc,
            proposal_result=proposal_result,
            run_id=run_id,
            run_kind=run_kind,
            character_id=character_id,
            base_revision_number=revision_number,
            correlation_id=correlation_id,
            evidence_refs=refs,
            timestamp=timestamp,
            dry_run=dry_run,
        )

    revisions = await list_identity_revisions(character_id=character_id)
    reversal_cutoffs = _reversal_cutoffs(
        revisions,
        accepted_changes=review_result.decision["accepted_changes"],
    )
    current_local_date = max(
        str(ref["character_local_date"])
        for ref in refs
    )
    inferred_promotions_today = (
        await count_inferred_identity_promotions_on_local_date(
            character_id=character_id,
            character_local_date=current_local_date,
        )
    )
    policy = evaluate_identity_growth_policy(
        current_identity=current_identity,
        proposal=proposal_result.decision,
        review=review_result.decision,
        evidence_refs=refs,
        evidence_cards=cards,
        current_candidates=current_candidates,
        current_revision_number=revision_number,
        inferred_min_episodes=(
            CHARACTER_IDENTITY_GROWTH_INFERRED_MIN_EPISODES
        ),
        inferred_min_local_dates=(
            CHARACTER_IDENTITY_GROWTH_INFERRED_MIN_LOCAL_DATES
        ),
        inferred_promotions_on_local_date=inferred_promotions_today,
        max_inferred_promotions_per_local_day=(
            CHARACTER_IDENTITY_GROWTH_MAX_INFERRED_PROMOTIONS_PER_LOCAL_DAY
        ),
        reversal_cutoffs_by_path=reversal_cutoffs,
    )

    validation_error_codes = sorted(set(
        proposal_result.validation_error_codes
        + review_result.validation_error_codes
    ))
    attempts = {
        "proposal": proposal_result.attempt_count,
        "review": review_result.attempt_count,
    }
    rejected_candidate_ids = sorted(set(
        str(candidate_id)
        for candidate_id in policy["rejected_candidate_ids"]
    ))
    candidate_id: str | None = None
    candidate: dict[str, object] | None = None

    if policy["status"] in {
        "candidate_updated",
        "revision_ready",
        "deferred",
    }:
        candidate, existing_candidate = _candidate_for_policy(
            policy=policy,
            proposal=proposal_result.decision,
            review=review_result.decision,
            current_candidates=current_candidates,
            character_id=character_id,
            base_revision_number=revision_number,
            timestamp=timestamp,
        )
        candidate_id = str(candidate["candidate_id"])
        if not dry_run:
            if existing_candidate is None:
                candidate = await insert_growth_candidate(candidate)
            else:
                candidate = await update_growth_candidate(
                    candidate,
                    expected_updated_at=str(
                        existing_candidate["updated_at"]
                    ),
                )
    elif policy["status"] == "rejected" and policy["candidate_id"]:
        candidate_id = str(policy["candidate_id"])
        rejected_candidate_ids.append(candidate_id)
        rejected_candidate_ids = sorted(set(rejected_candidate_ids))

    if rejected_candidate_ids and not dry_run:
        await reject_growth_candidates(
            character_id=character_id,
            base_revision_number=revision_number,
            candidate_ids=rejected_candidate_ids,
            reason_code=policy["policy_reason_code"],
            updated_at=timestamp,
        )

    write_revision = (
        policy["status"] == "revision_ready"
        and enable_revision_writes
        and not dry_run
    )
    disposition = _policy_disposition(
        policy_status=policy["status"],
        write_revision=write_revision,
    )
    lifecycle_state = "in_progress" if write_revision else "complete"
    completed_at = None if write_revision else timestamp
    run_root_episode_ids = list(policy["claimed_root_episode_ids"])
    source_evidence_count = policy["distinct_episode_count"]
    if not run_root_episode_ids:
        run_root_episode_ids = sorted({
            str(ref["root_episode_id"])
            for ref in refs
        })
        source_evidence_count = len(run_root_episode_ids)
    run = _growth_run_document(
        run_id=run_id,
        run_kind=run_kind,
        character_id=character_id,
        base_revision_number=revision_number,
        correlation_id=correlation_id,
        root_episode_ids=run_root_episode_ids,
        source_evidence_count=source_evidence_count,
        attempt_count_by_stage=attempts,
        lifecycle_state=lifecycle_state,
        disposition=disposition,
        proposal_reason_code=policy["proposal_reason_code"],
        review_reason_code=policy["review_reason_code"],
        policy_reason_code=policy["policy_reason_code"],
        persistence_reason_code=policy["policy_reason_code"],
        candidate_id=candidate_id,
        validation_error_codes=validation_error_codes,
        started_at=timestamp,
        completed_at=completed_at,
    )
    if not dry_run:
        run = await insert_growth_run(run)

    if not write_revision:
        return _evaluation_from_run(run)
    if candidate is None:
        raise RuntimeError("revision-ready policy requires a candidate")
    revision = await promote_ready_candidate(
        character_id=character_id,
        candidate_id=str(candidate["candidate_id"]),
        run_id=run_id,
    )
    await reconcile_identity_growth_post_commit(run_id=run_id)
    promoted_run = await get_growth_run(run_id=run_id)
    if promoted_run is None:
        raise RuntimeError("promoted identity run is not readable")
    result = _evaluation_from_run(promoted_run)
    return result


async def _persist_stage_failure(
    *,
    exc: IdentityStageError,
    proposal_result: IdentityStageResult | None,
    run_id: str,
    run_kind: str,
    character_id: str,
    base_revision_number: int,
    correlation_id: str,
    evidence_refs: Sequence[Mapping[str, object]],
    timestamp: str,
    dry_run: bool,
) -> models.IdentityGrowthEvaluationResultV1:
    """Persist one sanitized bounded semantic-stage failure."""

    proposal_attempts = (
        proposal_result.attempt_count
        if proposal_result is not None
        else (exc.attempt_count if exc.stage == "proposal" else 0)
    )
    review_attempts = exc.attempt_count if exc.stage == "review" else 0
    failure_reason = (
        "proposal_contract_failed"
        if exc.stage == "proposal"
        else "review_contract_failed"
    )
    proposal_reason = (
        str(proposal_result.decision["reason_code"])
        if proposal_result is not None
        else failure_reason
    )
    validation_errors = set(exc.validation_error_codes)
    if proposal_result is not None:
        validation_errors.update(proposal_result.validation_error_codes)
    roots = sorted({
        str(ref["root_episode_id"])
        for ref in evidence_refs
    })
    run = _growth_run_document(
        run_id=run_id,
        run_kind=run_kind,
        character_id=character_id,
        base_revision_number=base_revision_number,
        correlation_id=correlation_id,
        root_episode_ids=roots,
        source_evidence_count=len(roots),
        attempt_count_by_stage={
            "proposal": proposal_attempts,
            "review": review_attempts,
        },
        lifecycle_state="failed",
        disposition="failed",
        proposal_reason_code=proposal_reason,
        review_reason_code=failure_reason,
        policy_reason_code=failure_reason,
        persistence_reason_code=failure_reason,
        candidate_id=None,
        validation_error_codes=sorted(validation_errors),
        started_at=timestamp,
        completed_at=timestamp,
    )
    if not dry_run:
        run = await insert_growth_run(run)
    return _evaluation_from_run(run)


def _candidate_for_policy(
    *,
    policy: models.IdentityGrowthPolicyResultV1,
    proposal: Mapping[str, object],
    review: Mapping[str, object],
    current_candidates: Sequence[Mapping[str, object]],
    character_id: str,
    base_revision_number: int,
    timestamp: str,
) -> tuple[dict[str, object], Mapping[str, object] | None]:
    """Build a schema-complete new or updated identity candidate."""

    existing_by_id = {
        str(candidate["candidate_id"]): candidate
        for candidate in current_candidates
    }
    selected_id = policy["candidate_id"]
    existing = (
        existing_by_id.get(str(selected_id))
        if selected_id is not None
        else None
    )
    candidate_id = (
        str(existing["candidate_id"])
        if existing is not None
        else _opaque_identifier(
            "identity-candidate",
            {
                "character_id": character_id,
                "base_revision_number": base_revision_number,
                "change_kind": policy["change_kind"],
                "accepted_changes": policy["accepted_changes"],
                "claimed_root_episode_ids": (
                    policy["claimed_root_episode_ids"]
                ),
            },
        )
    )
    candidate_status = policy["candidate_status"]
    change_kind = policy["change_kind"]
    if candidate_status is None or change_kind is None:
        raise RuntimeError("accepted identity policy lacks candidate fields")
    candidate = {
        "schema_version": CANDIDATE_SCHEMA_VERSION,
        "candidate_id": candidate_id,
        "character_id": character_id,
        "base_revision_number": base_revision_number,
        "status": candidate_status,
        "change_kind": change_kind,
        "proposed_changes": deepcopy(policy["accepted_changes"]),
        "semantic_summary": policy["semantic_summary"],
        "evidence_refs": deepcopy(policy["evidence_refs"]),
        "distinct_episode_count": policy["distinct_episode_count"],
        "distinct_local_dates": list(policy["distinct_local_dates"]),
        "source_scope_kinds": list(policy["source_scope_kinds"]),
        "claimed_root_episode_ids": list(
            policy["claimed_root_episode_ids"]
        ),
        "newest_root_captured_at": max(
            str(ref["captured_at"])
            for ref in policy["evidence_refs"]
        ),
        "reversal_of_paths": list(policy["reversal_of_paths"]),
        "fresh_post_revision_root_count": (
            policy["fresh_post_revision_root_count"]
        ),
        "character_authorship": review["character_authorship"],
        "proposal_confidence": proposal["confidence"],
        "review_confidence": review["review_confidence"],
        "privacy_review": review["private_detail_risk"],
        "promoted_revision_number": None,
        "rejection_reason": None,
        "created_at": (
            str(existing["created_at"])
            if existing is not None
            else timestamp
        ),
        "updated_at": timestamp,
    }
    return candidate, existing


def _growth_run_document(
    *,
    run_id: str,
    run_kind: str,
    character_id: str,
    base_revision_number: int,
    correlation_id: str,
    root_episode_ids: Sequence[str],
    source_evidence_count: int,
    attempt_count_by_stage: Mapping[str, int],
    lifecycle_state: str,
    disposition: str,
    proposal_reason_code: str,
    review_reason_code: str,
    policy_reason_code: str,
    persistence_reason_code: str,
    candidate_id: str | None,
    validation_error_codes: Sequence[str],
    started_at: str,
    completed_at: str | None,
) -> dict[str, object]:
    """Build one sanitized growth-run document."""

    return {
        "schema_version": RUN_SCHEMA_VERSION,
        "run_id": run_id,
        "run_kind": run_kind,
        "character_id": character_id,
        "base_revision_number": base_revision_number,
        "correlation_id": correlation_id,
        "root_episode_ids": sorted(set(root_episode_ids)),
        "source_evidence_count": source_evidence_count,
        "attempt_count_by_stage": dict(attempt_count_by_stage),
        "lifecycle_state": lifecycle_state,
        "disposition": disposition,
        "proposal_reason_code": proposal_reason_code,
        "review_reason_code": review_reason_code,
        "policy_reason_code": policy_reason_code,
        "persistence_reason_code": persistence_reason_code,
        "candidate_id": candidate_id,
        "promoted_revision_number": None,
        "validation_error_codes": sorted(set(validation_error_codes)),
        "first_consumption": None,
        "post_commit_attempt_count": 0,
        "started_at": started_at,
        "completed_at": completed_at,
    }


def _evaluation_from_run(
    run: Mapping[str, object],
) -> models.IdentityGrowthEvaluationResultV1:
    """Project one sanitized persisted run onto the public result."""

    disposition = str(run["disposition"])
    status_by_disposition: dict[str, models.IdentityEvaluationStatus] = {
        "no_change": "no_change",
        "candidate_updated": "candidate_updated",
        "revision_promoted": "revision_promoted",
        "rejected": "rejected",
        "failed": "failed",
        "deferred": "deferred",
    }
    attempts = run["attempt_count_by_stage"]
    if not isinstance(attempts, Mapping):
        raise ValueError("growth run attempt counts must be an object")
    errors = run["validation_error_codes"]
    if not isinstance(errors, list):
        raise ValueError("growth run validation errors must be a list")
    return {
        "status": status_by_disposition[disposition],
        "run_id": str(run["run_id"]),
        "candidate_id": (
            str(run["candidate_id"])
            if run.get("candidate_id") is not None
            else None
        ),
        "base_revision_number": int(run["base_revision_number"]),
        "promoted_revision_number": (
            int(run["promoted_revision_number"])
            if run.get("promoted_revision_number") is not None
            else None
        ),
        "proposal_reason_code": str(run["proposal_reason_code"]),
        "review_reason_code": str(run["review_reason_code"]),
        "policy_reason_code": str(run["policy_reason_code"]),
        "persistence_reason_code": str(
            run["persistence_reason_code"]
        ),
        "validation_error_codes": [str(code) for code in errors],
        "attempt_count_by_stage": {
            "proposal": int(attempts["proposal"]),
            "review": int(attempts["review"]),
        },
        "source_evidence_count": int(run["source_evidence_count"]),
    }


def _not_routed_result(
    *,
    run_id: str,
    base_revision_number: int,
) -> models.IdentityGrowthEvaluationResultV1:
    """Return a sanitized disabled-lane result without persistence."""

    return {
        "status": "no_change",
        "run_id": run_id,
        "candidate_id": None,
        "base_revision_number": base_revision_number,
        "promoted_revision_number": None,
        "proposal_reason_code": "not_routed",
        "review_reason_code": "not_routed",
        "policy_reason_code": "not_routed",
        "persistence_reason_code": "not_routed",
        "validation_error_codes": [],
        "attempt_count_by_stage": {"proposal": 0, "review": 0},
        "source_evidence_count": 0,
    }


def _validated_source_evidence(
    source: Mapping[str, object],
) -> tuple[
    list[models.IdentityEvidenceRefV1],
    list[models.IdentityEvidenceCardV1],
]:
    """Validate joined repository references and prompt-safe cards."""

    raw_refs = source.get("evidence_refs", [])
    raw_cards = source.get("evidence_cards", [])
    if not isinstance(raw_refs, list) or not isinstance(raw_cards, list):
        raise ValueError("identity evidence refs and cards must be lists")
    refs = [
        validate_evidence_ref(_require_mapping(ref, context="evidence ref"))
        for ref in raw_refs
    ]
    refs_by_id = {
        ref["evidence_ref_id"]: ref
        for ref in refs
    }
    if len(refs_by_id) != len(refs):
        raise ValueError("identity evidence reference IDs must be unique")
    cards = []
    for raw_card in raw_cards:
        card = _require_mapping(raw_card, context="evidence card")
        evidence_ref_id = _optional_text(card.get("evidence_ref_id"))
        evidence_ref = refs_by_id.get(evidence_ref_id)
        if evidence_ref is None:
            raise ValueError("identity evidence card lacks a reference")
        cards.append(
            validate_identity_evidence_card(
                card,
                evidence_ref=evidence_ref,
            )
        )
    if {card["evidence_ref_id"] for card in cards} != set(refs_by_id):
        raise ValueError("identity evidence refs and cards must match")
    return refs, cards


def _current_revision_fields(
    current_revision: Mapping[str, object],
) -> tuple[str, int, models.CharacterEffectiveIdentityV1]:
    """Validate runner-facing current revision fields."""

    character_id = _optional_text(current_revision.get("character_id"))
    if not character_id:
        raise ValueError("current identity revision requires character_id")
    revision_number = current_revision.get("revision_number")
    if (
        not isinstance(revision_number, int)
        or isinstance(revision_number, bool)
        or revision_number < 0
    ):
        raise ValueError("current identity revision number is invalid")
    raw_identity = _require_mapping(
        current_revision.get("effective_identity"),
        context="current effective identity",
    )
    return (
        character_id,
        revision_number,
        validate_effective_identity(raw_identity),
    )


def _reversal_cutoffs(
    revisions: Sequence[Mapping[str, object]],
    *,
    accepted_changes: object,
) -> dict[str, str]:
    """Return latest non-seed revision time for each accepted path."""

    if not isinstance(accepted_changes, list):
        raise ValueError("accepted identity changes must be a list")
    accepted_paths = {
        str(change["path"])
        for change in accepted_changes
        if isinstance(change, Mapping) and "path" in change
    }
    cutoffs: dict[str, str] = {}
    for revision in revisions:
        number = revision.get("revision_number")
        if not isinstance(number, int) or number <= 0:
            continue
        changed_paths = revision.get("changed_paths", [])
        if not isinstance(changed_paths, list):
            raise ValueError("identity revision changed_paths must be a list")
        created_at = _optional_text(revision.get("created_at"))
        if not created_at:
            raise ValueError("identity revision requires created_at")
        for path in accepted_paths.intersection(
            str(item)
            for item in changed_paths
        ):
            current = cutoffs.get(path)
            if current is None or created_at > current:
                cutoffs[path] = created_at
    return cutoffs


def _policy_disposition(
    *,
    policy_status: str,
    write_revision: bool,
) -> str:
    """Map policy status to the persisted run disposition."""

    if policy_status == "revision_ready":
        return "candidate_updated" if write_revision else "deferred"
    disposition_by_status = {
        "no_change": "no_change",
        "candidate_updated": "candidate_updated",
        "rejected": "rejected",
        "deferred": "deferred",
    }
    return disposition_by_status[policy_status]


def _growth_run_id(
    *,
    run_kind: str,
    character_id: str,
    revision_number: int,
    correlation_id: str,
    evidence_refs: Sequence[Mapping[str, object]],
) -> str:
    """Derive an idempotent opaque run ID from trusted lineage."""

    return _opaque_identifier(
        "identity-growth-run",
        {
            "run_kind": run_kind,
            "character_id": character_id,
            "revision_number": revision_number,
            "correlation_id": correlation_id,
            "evidence_ref_ids": sorted(
                str(ref["evidence_ref_id"])
                for ref in evidence_refs
            ),
            "root_episode_ids": sorted({
                str(ref["root_episode_id"])
                for ref in evidence_refs
            }),
        },
    )


def _correlation_id(source: Mapping[str, object]) -> str:
    """Require the trusted correlation handle for one evaluation."""

    correlation_id = _optional_text(source.get("correlation_id"))
    if not correlation_id:
        raise ValueError("identity growth source requires correlation_id")
    return correlation_id


def _reflection_derivative_ids(
    document: Mapping[str, object],
    *,
    run_id: str,
) -> list[str]:
    """Return sorted derivative run IDs for one daily document."""

    raw_ids = document.get("source_reflection_run_ids", [])
    if not isinstance(raw_ids, list):
        raise ValueError("source_reflection_run_ids must be a list")
    return sorted({
        run_id,
        *(
            _required_text(item, context="source reflection run ID")
            for item in raw_ids
        ),
    })


def _validate_reflection_root(
    root: Mapping[str, object],
) -> dict[str, str]:
    """Validate one repository-produced recursive episode root."""

    expected = {
        "root_episode_id",
        "correlation_id",
        "character_local_date",
        "scope_kind",
        "captured_at",
    }
    if set(root) != expected:
        raise ValueError("reflection episode root has an invalid shape")
    scope_kind = _required_text(
        root["scope_kind"],
        context="reflection root scope_kind",
    )
    if scope_kind not in models.EVIDENCE_SCOPE_KINDS:
        raise ValueError("reflection root scope_kind is unsupported")
    return {
        "root_episode_id": _required_text(
            root["root_episode_id"],
            context="reflection root episode ID",
        ),
        "correlation_id": _required_text(
            root["correlation_id"],
            context="reflection root correlation ID",
        ),
        "character_local_date": _require_local_date(
            root["character_local_date"],
        ),
        "scope_kind": scope_kind,
        "captured_at": _require_timestamp(root["captured_at"]),
    }


def _conversation_quality_summary(output: Mapping[str, object]) -> str:
    """Return a bounded non-semantic join of daily quality patterns."""

    raw_patterns = output.get("conversation_quality_patterns", [])
    if not isinstance(raw_patterns, list):
        raise ValueError("conversation_quality_patterns must be a list")
    patterns = [
        _required_text(pattern, context="conversation quality pattern")
        for pattern in raw_patterns
    ]
    return _bounded_text(" ".join(patterns), required=False)


def _opaque_identifier(prefix: str, payload: object) -> str:
    """Return a stable opaque identifier for one canonical payload."""

    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    return f"{prefix}:{digest}"


def _timestamp(now: datetime | None) -> str:
    """Return one timezone-aware UTC timestamp."""

    value = now or datetime.now(timezone.utc)
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("identity growth timestamp must include timezone")
    return value.astimezone(timezone.utc).isoformat()


def _require_timestamp(value: object) -> str:
    """Require one timezone-aware ISO timestamp without rewriting it."""

    text = _required_text(value, context="reflection root captured_at")
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("reflection root captured_at must be ISO") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("reflection root captured_at requires timezone")
    return text


def _require_local_date(value: object) -> str:
    """Require one canonical ISO calendar date."""

    text = _required_text(value, context="character_local_date")
    try:
        parsed = datetime.strptime(text, "%Y-%m-%d")
    except ValueError as exc:
        raise ValueError("character_local_date must use YYYY-MM-DD") from exc
    if parsed.strftime("%Y-%m-%d") != text:
        raise ValueError("character_local_date must be canonical")
    return text


def _sorted_unique_text(
    values: Sequence[str],
    *,
    context: str,
) -> list[str]:
    """Require and sort one unique identifier sequence."""

    if isinstance(values, (str, bytes)):
        raise ValueError(f"{context} must be a sequence")
    return sorted({
        _required_text(value, context=context)
        for value in values
    })


def _bounded_text(value: object, *, required: bool) -> str:
    """Return prompt-safe bounded text without semantic rewriting."""

    text = _optional_text(value)
    if required and not text:
        return ""
    return text[:models.IDENTITY_EVIDENCE_CARD_TEXT_LIMIT]


def _required_text(value: object, *, context: str) -> str:
    """Require one nonempty text value."""

    text = _optional_text(value)
    if not text:
        raise ValueError(f"{context} must be nonempty text")
    return text


def _optional_text(value: object) -> str:
    """Return stripped text or an empty string."""

    return value.strip() if isinstance(value, str) else ""


def _require_mapping(
    value: object,
    *,
    context: str,
) -> Mapping[str, object]:
    """Require one string-keyed mapping."""

    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise ValueError(f"{context} keys must be strings")
    return value


__all__ = [
    "evaluate_episode_identity_growth",
    "run_reflection_identity_growth_pass",
]
