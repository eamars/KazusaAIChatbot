"""Longitudinal pace and anti-oscillation tests for identity growth."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy

from kazusa_ai_chatbot.character_identity_growth import models
from kazusa_ai_chatbot.character_identity_growth.policy import (
    evaluate_identity_growth_policy,
)


def _identity() -> dict[str, object]:
    """Build one complete character-generic identity."""

    return {
        "name": "Test Character",
        "description": "A reflective person with durable boundaries.",
        "gender": "unspecified",
        "age": 30,
        "birthday": "March 3",
        "backstory": "They revise judgments through lived experience.",
        "personality_brief": {
            "mbti": "ISTP",
            "logic": "Evidence-led and practical.",
            "tempo": "Brief, measured, and responsive.",
            "defense": "Withdraws before reassessing.",
            "quirks": "Checks assumptions aloud.",
            "taboos": "Rejects imposed self-definitions.",
        },
        "boundary_profile": {
            "self_integrity": 0.5,
            "control_sensitivity": 0.5,
            "compliance_strategy": "resist",
            "relational_override": 0.5,
            "control_intimacy_misread": 0.5,
            "boundary_recovery": "rebound",
            "authority_skepticism": 0.5,
        },
        "linguistic_texture_profile": {
            "fragmentation": 0.5,
            "hesitation_density": 0.5,
            "counter_questioning": 0.5,
            "softener_density": 0.5,
            "formalism_avoidance": 0.5,
            "abstraction_reframing": 0.5,
            "direct_assertion": 0.5,
            "emotional_leakage": 0.5,
            "rhythmic_bounce": 0.5,
            "self_deprecation": 0.5,
        },
        "self_image": {
            "self_concept": "I protect myself by withdrawing.",
            "current_growth_edges": [
                "Learn whether earned trust can coexist with agency.",
            ],
        },
        "visual_characterization": (
            "An alert adult with practical layers and a guarded stance."
        ),
    }


def _patch() -> dict[str, object]:
    """Build the stable inferred-growth direction used by the timeline."""

    return {
        "path": "self_image.self_concept",
        "value_kind": "text",
        "replacement_text": (
            "I can remain engaged when trust is repeatedly earned."
        ),
    }


def _evidence_ref(
    number: int,
    *,
    local_date: str,
    root_number: int | None = None,
    source_kind: str = "settled_episode",
    captured_at: str | None = None,
) -> dict[str, object]:
    """Build one repository-owned evidence reference."""

    effective_root = number if root_number is None else root_number
    return {
        "schema_version": models.IDENTITY_EVIDENCE_SCHEMA_VERSION,
        "evidence_ref_id": f"evidence-{number}",
        "root_episode_id": f"episode-{effective_root}",
        "correlation_id": f"correlation-{effective_root}",
        "source_kind": source_kind,
        "derived_reflection_run_ids": (
            [f"reflection-{number}"]
            if source_kind == "daily_reflection"
            else []
        ),
        "character_local_date": local_date,
        "scope_kind": "private",
        "captured_at": captured_at or f"{local_date}T10:00:00+00:00",
    }


def _evidence_card(
    ref: Mapping[str, object],
) -> dict[str, object]:
    """Build one prompt-safe card that matches a repository reference."""

    return {
        "schema_version": models.IDENTITY_EVIDENCE_CARD_SCHEMA_VERSION,
        "evidence_ref_id": ref["evidence_ref_id"],
        "source_kind": ref["source_kind"],
        "character_local_date": ref["character_local_date"],
        "scope_kind": ref["scope_kind"],
        "decontextualized_event": (
            "A separate interaction tested whether earned trust could last."
        ),
        "character_cognition_summary": (
            "The character reconsidered automatic withdrawal."
        ),
        "visible_self_expression_summary": (
            "The character chose to remain engaged."
        ),
    }


def _proposal(
    ref: Mapping[str, object],
    *,
    candidate_id: str | None,
) -> dict[str, object]:
    """Build one inferred proposal for a new or existing candidate."""

    return {
        "schema_version": models.IDENTITY_PROPOSAL_DECISION_SCHEMA_VERSION,
        "action": (
            "corroborate_candidate"
            if candidate_id is not None
            else "inferred_growth"
        ),
        "candidate_id": candidate_id,
        "proposed_changes": [_patch()],
        "character_authorship": "inferred",
        "identity_relevance": "durable",
        "global_applicability": "global",
        "confidence": "high",
        "private_detail_risk": "low",
        "character_owned_abstraction": (
            "Repeated choices are changing defensive distance."
        ),
        "evidence_ref_ids": [ref["evidence_ref_id"]],
        "contradiction_candidate_ids": [],
        "reason_code": "candidate_emerging",
    }


def _review(
    proposal: Mapping[str, object],
    *,
    candidate_id: str | None,
) -> dict[str, object]:
    """Build an independent accepting review for the same direction."""

    return {
        "schema_version": models.IDENTITY_REVIEW_DECISION_SCHEMA_VERSION,
        "verdict": "accept",
        "selected_candidate_id": candidate_id,
        "rejected_candidate_ids": [],
        "accepted_change_kind": "inferred_growth",
        "accepted_changes": deepcopy(proposal["proposed_changes"]),
        "character_authorship": "inferred",
        "identity_relevance": "durable",
        "coherence": "coherent",
        "global_applicability": "global",
        "review_confidence": "high",
        "private_detail_risk": "low",
        "character_owned_summary": (
            "Repeated independent choices support a durable shift."
        ),
        "privacy_safe_evidence_summaries": [
            "Repeated choices support a durable character-owned shift.",
        ],
        "reason_code": "candidate_emerging",
    }


def _candidate_from_result(
    result: Mapping[str, object],
    *,
    candidate_id: str = "candidate-longitudinal",
    base_revision_number: int = 0,
    reversal_of_paths: Sequence[str] = (),
) -> dict[str, object]:
    """Project one policy result into the next evaluation's candidate."""

    return {
        "candidate_id": candidate_id,
        "base_revision_number": base_revision_number,
        "status": result["candidate_status"],
        "change_kind": result["change_kind"],
        "proposed_changes": deepcopy(result["accepted_changes"]),
        "semantic_summary": result["semantic_summary"],
        "evidence_refs": deepcopy(result["evidence_refs"]),
        "reversal_of_paths": list(reversal_of_paths),
        "character_authorship": "inferred",
        "proposal_confidence": "high",
        "review_confidence": "high",
    }


def _evaluate(
    ref: Mapping[str, object],
    *,
    candidate: Mapping[str, object] | None = None,
    inferred_min_episodes: int = 3,
    inferred_min_local_dates: int = 2,
    promotions_today: int = 0,
    max_promotions_per_day: int = 1,
    current_revision_number: int = 0,
    reversal_cutoffs: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Evaluate one chronological root against the selected pace."""

    candidate_id = (
        str(candidate["candidate_id"])
        if candidate is not None
        else None
    )
    proposal = _proposal(ref, candidate_id=candidate_id)
    review = _review(proposal, candidate_id=candidate_id)
    return evaluate_identity_growth_policy(
        current_identity=_identity(),
        proposal=proposal,
        review=review,
        evidence_refs=[ref],
        evidence_cards=[_evidence_card(ref)],
        current_candidates=[candidate] if candidate is not None else [],
        current_revision_number=current_revision_number,
        inferred_min_episodes=inferred_min_episodes,
        inferred_min_local_dates=inferred_min_local_dates,
        inferred_promotions_on_local_date=promotions_today,
        max_inferred_promotions_per_local_day=max_promotions_per_day,
        reversal_cutoffs_by_path=reversal_cutoffs or {},
    )


def test_default_pace_holds_then_promotes_across_distinct_date_fields() -> None:
    """One modeled timeline holds at roots one/two and promotes at three."""

    first_ref = _evidence_ref(1, local_date="2026-07-01")
    first = _evaluate(first_ref)
    assert first["status"] == "candidate_updated"
    assert first["distinct_episode_count"] == 1

    first_candidate = _candidate_from_result(first)
    derivative_ref = _evidence_ref(
        4,
        local_date="2026-07-01",
        root_number=1,
        source_kind="daily_reflection",
    )
    derivative = _evaluate(
        derivative_ref,
        candidate=first_candidate,
    )
    assert derivative["status"] == "candidate_updated"
    assert derivative["distinct_episode_count"] == 1

    second_ref = _evidence_ref(2, local_date="2026-07-01")
    second = _evaluate(
        second_ref,
        candidate=_candidate_from_result(derivative),
    )
    assert second["status"] == "candidate_updated"
    assert second["distinct_episode_count"] == 2
    assert second["distinct_local_dates"] == ["2026-07-01"]

    third_ref = _evidence_ref(3, local_date="2026-07-02")
    third_candidate = _candidate_from_result(second)
    capped = _evaluate(
        third_ref,
        candidate=third_candidate,
        promotions_today=1,
    )
    assert capped["status"] == "deferred"
    assert capped["policy_reason_code"] == "cadence_wait"

    ready = _evaluate(
        third_ref,
        candidate=third_candidate,
        promotions_today=0,
    )
    assert ready["status"] == "revision_ready"
    assert ready["distinct_episode_count"] == 3
    assert ready["distinct_local_dates"] == [
        "2026-07-01",
        "2026-07-02",
    ]


def test_slower_config_requires_four_roots_across_three_dates() -> None:
    """Operators can slow growth without changing semantic ownership."""

    result = _evaluate(_evidence_ref(1, local_date="2026-07-01"))
    candidate = _candidate_from_result(result)
    for number, local_date in (
        (2, "2026-07-01"),
        (3, "2026-07-02"),
    ):
        result = _evaluate(
            _evidence_ref(number, local_date=local_date),
            candidate=candidate,
            inferred_min_episodes=4,
            inferred_min_local_dates=3,
        )
        candidate = _candidate_from_result(result)

    assert result["status"] == "candidate_updated"
    assert result["distinct_episode_count"] == 3
    assert result["distinct_local_dates"] == [
        "2026-07-01",
        "2026-07-02",
    ]

    ready = _evaluate(
        _evidence_ref(4, local_date="2026-07-03"),
        candidate=candidate,
        inferred_min_episodes=4,
        inferred_min_local_dates=3,
    )
    assert ready["status"] == "revision_ready"
    assert ready["distinct_episode_count"] == 4
    assert ready["distinct_local_dates"] == [
        "2026-07-01",
        "2026-07-02",
        "2026-07-03",
    ]


def test_reversal_ignores_old_roots_until_fresh_threshold_is_met() -> None:
    """A promoted direction cannot oscillate using pre-revision roots."""

    cutoff = "2026-07-03T00:00:00+00:00"
    old_result = _evaluate(_evidence_ref(1, local_date="2026-07-01"))
    old_candidate = _candidate_from_result(
        old_result,
        base_revision_number=0,
        reversal_of_paths=["self_image.self_concept"],
    )
    second_old = _evaluate(
        _evidence_ref(2, local_date="2026-07-02"),
        candidate=old_candidate,
        current_revision_number=1,
        reversal_cutoffs={"self_image.self_concept": cutoff},
    )
    candidate = _candidate_from_result(
        second_old,
        base_revision_number=1,
        reversal_of_paths=["self_image.self_concept"],
    )

    for number, local_date in (
        (3, "2026-07-04"),
        (4, "2026-07-04"),
    ):
        result = _evaluate(
            _evidence_ref(number, local_date=local_date),
            candidate=candidate,
            current_revision_number=1,
            reversal_cutoffs={"self_image.self_concept": cutoff},
        )
        candidate = _candidate_from_result(
            result,
            base_revision_number=1,
            reversal_of_paths=["self_image.self_concept"],
        )
        assert result["status"] == "candidate_updated"

    ready = _evaluate(
        _evidence_ref(5, local_date="2026-07-05"),
        candidate=candidate,
        current_revision_number=1,
        reversal_cutoffs={"self_image.self_concept": cutoff},
    )
    assert ready["status"] == "revision_ready"
    assert ready["fresh_post_revision_root_count"] == 3
    assert ready["reversal_of_paths"] == ["self_image.self_concept"]
