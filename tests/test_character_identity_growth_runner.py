"""Orchestration tests for the background identity-growth owner."""

from __future__ import annotations

from copy import deepcopy
from unittest.mock import AsyncMock

import pytest

from kazusa_ai_chatbot.character_identity_growth import models
from kazusa_ai_chatbot.character_identity_growth.identity import (
    dedupe_evidence_refs,
    evidence_counts,
)
from kazusa_ai_chatbot.character_identity_growth.llm import (
    IdentityStageResult,
)


def _identity() -> dict[str, object]:
    """Build one complete character-generic identity."""

    return {
        "name": "Test Character",
        "description": "A grounded character with a revisable identity.",
        "gender": "unspecified",
        "age": 30,
        "birthday": "March 3",
        "backstory": "They learned to distinguish choice from compliance.",
        "personality_brief": {
            "mbti": "ISTP",
            "logic": "Evidence-led and practical.",
            "tempo": "Brief, measured, and responsive.",
            "defense": "Withdraws briefly before reassessing.",
            "quirks": "Checks assumptions aloud.",
            "taboos": "Rejects imposed self-definitions.",
        },
        "boundary_profile": {
            "self_integrity": 0.7,
            "control_sensitivity": 0.7,
            "compliance_strategy": "resist",
            "relational_override": 0.3,
            "control_intimacy_misread": 0.3,
            "boundary_recovery": "rebound",
            "authority_skepticism": 0.6,
        },
        "linguistic_texture_profile": {
            "fragmentation": 0.2,
            "hesitation_density": 0.2,
            "counter_questioning": 0.4,
            "softener_density": 0.3,
            "formalism_avoidance": 0.7,
            "abstraction_reframing": 0.5,
            "direct_assertion": 0.7,
            "emotional_leakage": 0.3,
            "rhythmic_bounce": 0.4,
            "self_deprecation": 0.1,
        },
        "self_image": {
            "self_concept": "I can revise my judgment without losing agency.",
            "current_growth_edges": [
                "Let sustained evidence soften automatic withdrawal.",
            ],
        },
        "visual_characterization": (
            "An alert adult with practical layers and an open stance."
        ),
    }


def _revision() -> dict[str, object]:
    """Build a current immutable identity revision."""

    return {
        "character_id": "character-global-1",
        "revision_number": 0,
        "effective_identity": _identity(),
        "changed_paths": [],
        "change_diff": [],
        "created_at": "2026-07-27T00:00:00+00:00",
    }


def _evidence_ref(
    *,
    evidence_ref_id: str = "evidence-1",
    root_episode_id: str = "episode-1",
) -> dict[str, object]:
    """Build one repository-owned settled-episode reference."""

    return {
        "schema_version": models.IDENTITY_EVIDENCE_SCHEMA_VERSION,
        "evidence_ref_id": evidence_ref_id,
        "root_episode_id": root_episode_id,
        "correlation_id": "correlation-1",
        "source_kind": "settled_episode",
        "derived_reflection_run_ids": [],
        "character_local_date": "2026-07-28",
        "scope_kind": "private",
        "captured_at": "2026-07-27T12:00:00+00:00",
    }


def _evidence_card(
    *,
    evidence_ref_id: str = "evidence-1",
) -> dict[str, object]:
    """Build one prompt-safe semantic evidence card."""

    return {
        "schema_version": models.IDENTITY_EVIDENCE_CARD_SCHEMA_VERSION,
        "evidence_ref_id": evidence_ref_id,
        "source_kind": "settled_episode",
        "character_local_date": "2026-07-28",
        "scope_kind": "private",
        "decontextualized_event": (
            "The character reconsidered a recurring defensive response."
        ),
        "character_cognition_summary": (
            "The character defined a more deliberate response as her own."
        ),
        "visible_self_expression_summary": (
            "The character explicitly described the changed self-view."
        ),
    }


def _proposal() -> dict[str, object]:
    """Build one accepted explicit self-redefinition proposal."""

    return {
        "schema_version": models.IDENTITY_PROPOSAL_DECISION_SCHEMA_VERSION,
        "action": "explicit_self_redefinition",
        "candidate_id": None,
        "proposed_changes": [{
            "path": "self_image.self_concept",
            "value_kind": "text",
            "replacement_text": (
                "I can choose openness while preserving my own judgment."
            ),
        }],
        "character_authorship": "self_declared",
        "identity_relevance": "durable",
        "global_applicability": "global",
        "confidence": "high",
        "private_detail_risk": "low",
        "character_owned_abstraction": (
            "A self-authored shift toward deliberate openness."
        ),
        "evidence_ref_ids": ["evidence-1"],
        "contradiction_candidate_ids": [],
        "reason_code": "candidate_ready",
    }


def _review() -> dict[str, object]:
    """Build one independent high-confidence acceptance."""

    return {
        "schema_version": models.IDENTITY_REVIEW_DECISION_SCHEMA_VERSION,
        "verdict": "accept",
        "selected_candidate_id": None,
        "rejected_candidate_ids": [],
        "accepted_change_kind": "explicit_self_redefinition",
        "accepted_changes": deepcopy(_proposal()["proposed_changes"]),
        "character_authorship": "self_declared",
        "identity_relevance": "durable",
        "coherence": "coherent",
        "global_applicability": "global",
        "review_confidence": "high",
        "private_detail_risk": "low",
        "character_owned_summary": (
            "The character made a durable, self-authored identity choice."
        ),
        "privacy_safe_evidence_summaries": [
            "A recurring response was explicitly reconsidered.",
        ],
        "reason_code": "candidate_ready",
    }


def _stage(decision: dict[str, object]) -> IdentityStageResult:
    """Wrap one validated semantic decision as a stage result."""

    return IdentityStageResult(
        decision=decision,
        attempt_count=1,
        prompt_chars=1000,
        output_chars=400,
        validation_error_codes=(),
        trace_id="trace-1",
    )


@pytest.mark.asyncio
async def test_no_evidence_records_a_sanitized_run_without_llm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A routed but unrooted item is auditable and never reaches semantics."""

    from kazusa_ai_chatbot.character_identity_growth import runner

    insert_run = AsyncMock(side_effect=lambda run: dict(run))
    propose = AsyncMock()
    review = AsyncMock()
    monkeypatch.setattr(runner, "get_growth_run", AsyncMock(return_value=None))
    monkeypatch.setattr(runner, "insert_growth_run", insert_run)
    monkeypatch.setattr(runner, "propose_identity_growth", propose)
    monkeypatch.setattr(runner, "review_identity_growth", review)

    result = await runner.evaluate_episode_identity_growth(
        settled_episode={
            "correlation_id": "correlation-no-evidence",
            "llm_trace_id": "",
            "evidence_refs": [],
            "evidence_cards": [],
        },
        current_revision=_revision(),
    )

    assert result["status"] == "no_change"
    assert result["policy_reason_code"] == "no_eligible_evidence"
    persisted_run = insert_run.await_args.args[0]
    assert persisted_run["root_episode_ids"] == []
    assert persisted_run["source_evidence_count"] == 0
    assert persisted_run["lifecycle_state"] == "complete"
    assert "decontextualized_event" not in str(persisted_run)
    propose.assert_not_awaited()
    review.assert_not_awaited()


@pytest.mark.asyncio
async def test_explicit_episode_promotes_through_the_single_identity_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Proposal, review, candidate, run, and revision share one lineage."""

    from kazusa_ai_chatbot.character_identity_growth import runner

    insert_candidate = AsyncMock(side_effect=lambda candidate: dict(candidate))
    insert_run = AsyncMock(side_effect=lambda run: dict(run))
    promote = AsyncMock(return_value={
        **_revision(),
        "revision_number": 1,
    })

    async def get_growth_run(*, run_id):
        del run_id
        if insert_run.await_count == 0:
            return None
        persisted = dict(insert_run.await_args.args[0])
        return {
            **persisted,
            "lifecycle_state": "complete",
            "disposition": "revision_promoted",
            "persistence_reason_code": "revision_promoted",
            "promoted_revision_number": 1,
            "post_commit_attempt_count": 1,
            "completed_at": "2026-07-28T00:00:00Z",
        }

    reconcile = AsyncMock(return_value={
        "completed_count": 1,
        "failed_count": 0,
    })
    monkeypatch.setattr(runner, "get_growth_run", get_growth_run)
    monkeypatch.setattr(
        runner,
        "list_current_growth_candidates",
        AsyncMock(return_value=[]),
    )
    monkeypatch.setattr(
        runner,
        "list_identity_revisions",
        AsyncMock(return_value=[_revision()]),
    )
    monkeypatch.setattr(
        runner,
        "count_inferred_identity_promotions_on_local_date",
        AsyncMock(return_value=0),
    )
    monkeypatch.setattr(
        runner,
        "propose_identity_growth",
        AsyncMock(return_value=_stage(_proposal())),
    )
    monkeypatch.setattr(
        runner,
        "review_identity_growth",
        AsyncMock(return_value=_stage(_review())),
    )
    monkeypatch.setattr(
        runner,
        "insert_growth_candidate",
        insert_candidate,
    )
    monkeypatch.setattr(runner, "insert_growth_run", insert_run)
    monkeypatch.setattr(runner, "promote_ready_candidate", promote)
    monkeypatch.setattr(
        runner,
        "reconcile_identity_growth_post_commit",
        reconcile,
    )

    result = await runner.evaluate_episode_identity_growth(
        settled_episode={
            "correlation_id": "correlation-1",
            "llm_trace_id": "trace-1",
            "evidence_refs": [_evidence_ref()],
            "evidence_cards": [_evidence_card()],
        },
        current_revision=_revision(),
    )

    assert result["status"] == "revision_promoted"
    assert result["base_revision_number"] == 0
    assert result["promoted_revision_number"] == 1
    candidate = insert_candidate.await_args.args[0]
    persisted_run = insert_run.await_args.args[0]
    assert candidate["status"] == "ready"
    assert candidate["claimed_root_episode_ids"] == ["episode-1"]
    assert persisted_run["candidate_id"] == candidate["candidate_id"]
    assert persisted_run["root_episode_ids"] == ["episode-1"]
    assert persisted_run["lifecycle_state"] == "in_progress"
    promote.assert_awaited_once_with(
        character_id="character-global-1",
        candidate_id=candidate["candidate_id"],
        run_id=persisted_run["run_id"],
    )
    reconcile.assert_awaited_once_with(run_id=persisted_run["run_id"])


@pytest.mark.asyncio
async def test_post_commit_reconciliation_invalidates_then_completes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A promoted run completes only after cache and event side effects."""

    from kazusa_ai_chatbot.character_identity_growth import runner

    pending_run = {
        "run_id": "run-post-commit",
        "character_id": "character-global-1",
        "correlation_id": "correlation-post-commit",
        "lifecycle_state": "post_commit_pending",
        "disposition": "revision_promoted",
        "promoted_revision_number": 4,
    }
    cache_runtime = type("CacheRuntime", (), {})()
    cache_runtime.invalidate = AsyncMock(return_value=2)
    complete = AsyncMock(return_value={
        **pending_run,
        "lifecycle_state": "complete",
    })
    monkeypatch.setattr(
        runner,
        "get_growth_run",
        AsyncMock(return_value=pending_run),
    )
    monkeypatch.setattr(
        runner,
        "get_rag_cache2_runtime",
        lambda: cache_runtime,
    )
    record_event = AsyncMock(return_value={"accepted": True})
    monkeypatch.setattr(
        runner,
        "record_character_identity_growth_event",
        record_event,
    )
    monkeypatch.setattr(
        runner,
        "complete_growth_run_post_commit",
        complete,
    )

    result = await runner.reconcile_identity_growth_post_commit(
        run_id="run-post-commit",
    )

    assert result == {"completed_count": 1, "failed_count": 0}
    invalidation = cache_runtime.invalidate.await_args.args[0]
    assert invalidation.source == "character_identity"
    assert invalidation.global_user_id == "character-global-1"
    record_event.assert_awaited_once()
    complete.assert_awaited_once_with(
        run_id="run-post-commit",
        character_id="character-global-1",
        revision_number=4,
    )


@pytest.mark.asyncio
async def test_post_commit_failure_retains_pending_retry_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A side-effect failure should increment retry evidence and stay pending."""

    from kazusa_ai_chatbot.character_identity_growth import runner

    pending_run = {
        "run_id": "run-post-commit-failure",
        "character_id": "character-global-1",
        "correlation_id": "correlation-post-commit-failure",
        "lifecycle_state": "post_commit_pending",
        "disposition": "revision_promoted",
        "promoted_revision_number": 5,
    }
    cache_runtime = type("CacheRuntime", (), {})()
    cache_runtime.invalidate = AsyncMock(
        side_effect=RuntimeError("cache unavailable")
    )
    retain_failure = AsyncMock(return_value=pending_run)
    complete = AsyncMock()
    monkeypatch.setattr(
        runner,
        "get_growth_run",
        AsyncMock(return_value=pending_run),
    )
    monkeypatch.setattr(
        runner,
        "get_rag_cache2_runtime",
        lambda: cache_runtime,
    )
    monkeypatch.setattr(
        runner,
        "record_growth_run_post_commit_failure",
        retain_failure,
    )
    monkeypatch.setattr(
        runner,
        "complete_growth_run_post_commit",
        complete,
    )

    result = await runner.reconcile_identity_growth_post_commit(
        run_id="run-post-commit-failure",
    )

    assert result == {"completed_count": 0, "failed_count": 1}
    retain_failure.assert_awaited_once_with(
        run_id="run-post-commit-failure",
        character_id="character-global-1",
        revision_number=5,
    )
    complete.assert_not_awaited()


def test_daily_reflection_derivatives_retain_one_count_per_root() -> None:
    """Two daily cards for one episode enrich evidence without new cadence."""

    from kazusa_ai_chatbot.character_identity_growth import runner

    first = _daily_doc(
        run_id="daily-reflection-1",
        day_summary="The character reconsidered an automatic response.",
    )
    second = _daily_doc(
        run_id="daily-reflection-2",
        day_summary="The same choice remained visible in later review.",
    )

    refs, cards = runner._build_reflection_identity_evidence(
        [first, second],
        character_local_date="2026-07-28",
    )
    deduped = dedupe_evidence_refs(refs)

    assert len(cards) == 2
    assert evidence_counts(refs)["distinct_episode_count"] == 1
    assert len(deduped) == 1
    assert deduped[0]["derived_reflection_run_ids"] == [
        "daily-reflection-1",
        "daily-reflection-2",
        "hourly-reflection-1",
    ]


def _daily_doc(
    *,
    run_id: str,
    day_summary: str,
) -> dict[str, object]:
    """Build one validated daily-reflection input with repository roots."""

    return {
        "run_id": run_id,
        "run_kind": "daily_channel",
        "status": "succeeded",
        "character_local_date": "2026-07-28",
        "scope": {"channel_type": "private"},
        "source_reflection_run_ids": ["hourly-reflection-1"],
        "source_episode_refs": [{
            "root_episode_id": "episode-1",
            "correlation_id": "correlation-1",
            "character_local_date": "2026-07-28",
            "scope_kind": "private",
            "captured_at": "2026-07-27T12:00:00+00:00",
        }],
        "output": {
            "day_summary": day_summary,
            "conversation_quality_patterns": [
                "The response was framed as a self-owned judgment.",
            ],
        },
    }
