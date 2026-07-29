"""Proposal, review, prompt, and policy tests for identity growth."""

from __future__ import annotations

from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from kazusa_ai_chatbot.character_identity_growth import llm
from kazusa_ai_chatbot.character_identity_growth import models
from kazusa_ai_chatbot.character_identity_growth.policy import (
    evaluate_identity_growth_policy,
)
from kazusa_ai_chatbot.character_identity_growth.projection import (
    build_identity_proposal_input,
    build_identity_review_input,
)
from kazusa_ai_chatbot.character_identity_growth.validation import (
    validate_identity_proposal_decision,
    validate_identity_review_decision,
)


def _identity() -> dict[str, object]:
    """Build one complete generic identity."""

    return {
        "name": "Test Character",
        "description": "A reflective person with durable boundaries.",
        "gender": "unspecified",
        "age": 30,
        "birthday": "March 3",
        "backstory": "They learned to revise judgments through experience.",
        "personality_brief": {
            "mbti": "ISTP",
            "logic": "Evidence-led and practical.",
            "tempo": "Brief, measured, and responsive.",
            "defense": "Withdraws briefly before reassessing.",
            "quirks": "Checks assumptions aloud.",
            "taboos": "Rejects imposed self-definitions.",
        },
        "boundary_profile": {
            "self_integrity": 0.0,
            "control_sensitivity": 0.22,
            "compliance_strategy": "resist",
            "relational_override": 0.5,
            "control_intimacy_misread": 0.61,
            "boundary_recovery": "rebound",
            "authority_skepticism": 1.0,
        },
        "linguistic_texture_profile": {
            "fragmentation": 0.1,
            "hesitation_density": 0.3,
            "counter_questioning": 0.5,
            "softener_density": 0.7,
            "formalism_avoidance": 0.9,
            "abstraction_reframing": 0.1,
            "direct_assertion": 0.3,
            "emotional_leakage": 0.5,
            "rhythmic_bounce": 0.7,
            "self_deprecation": 0.9,
        },
        "self_image": {
            "self_concept": "I revise myself through my own judgment.",
            "current_growth_edges": ["Stay present after trust is earned."],
        },
        "visual_characterization": (
            "An alert adult with practical layers and an open stance."
        ),
    }


def _evidence_ref(
    number: int,
    *,
    root_number: int | None = None,
    local_date: str | None = None,
    captured_at: str | None = None,
    source_kind: str = "settled_episode",
) -> dict[str, object]:
    """Build one repository-owned evidence reference."""

    effective_root = number if root_number is None else root_number
    effective_date = local_date or (
        "2026-07-01" if effective_root < 3 else "2026-07-02"
    )
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
        "character_local_date": effective_date,
        "scope_kind": "private",
        "captured_at": captured_at or f"{effective_date}T10:00:00+00:00",
    }


def _evidence_card(
    number: int,
    *,
    source_kind: str = "settled_episode",
    local_date: str | None = None,
    event: str = "The interaction tested whether earned trust could last.",
    cognition: str = "The character reconsidered automatic withdrawal.",
    expression: str = "The character chose to stay engaged.",
) -> dict[str, object]:
    """Build one prompt-safe evidence card."""

    effective_date = local_date or (
        "2026-07-01" if number < 3 else "2026-07-02"
    )
    return {
        "schema_version": models.IDENTITY_EVIDENCE_CARD_SCHEMA_VERSION,
        "evidence_ref_id": f"evidence-{number}",
        "source_kind": source_kind,
        "character_local_date": effective_date,
        "scope_kind": "private",
        "decontextualized_event": event,
        "character_cognition_summary": cognition,
        "visible_self_expression_summary": expression,
    }


def _patch(
    replacement: str = "I let earned trust temper defensive distance.",
) -> dict[str, object]:
    """Build one valid self-image patch."""

    return {
        "path": "self_image.self_concept",
        "value_kind": "text",
        "replacement_text": replacement,
    }


def _proposal(
    *,
    action: str = "inferred_growth",
    candidate_id: str | None = None,
    evidence_numbers: tuple[int, ...] = (1,),
    authorship: str = "inferred",
    relevance: str = "durable",
    applicability: str = "global",
    confidence: str = "high",
    privacy: str = "low",
    contradictions: tuple[str, ...] = (),
    reason_code: str = "candidate_emerging",
) -> dict[str, object]:
    """Build one proposal-stage decision."""

    proposed_changes = [] if action == "no_change" else [_patch()]
    return {
        "schema_version": models.IDENTITY_PROPOSAL_DECISION_SCHEMA_VERSION,
        "action": action,
        "candidate_id": candidate_id,
        "proposed_changes": proposed_changes,
        "character_authorship": authorship,
        "identity_relevance": relevance,
        "global_applicability": applicability,
        "confidence": confidence,
        "private_detail_risk": privacy,
        "character_owned_abstraction": (
            "No durable identity change."
            if action == "no_change"
            else "Earned trust is changing defensive distance."
        ),
        "evidence_ref_ids": [
            f"evidence-{number}"
            for number in evidence_numbers
        ],
        "contradiction_candidate_ids": list(contradictions),
        "reason_code": reason_code,
    }


def _review(
    proposal: dict[str, object],
    *,
    verdict: str = "accept",
    selected_candidate_id: str | None = None,
    rejected_candidate_ids: tuple[str, ...] = (),
    change_kind: str | None = "inferred_growth",
    authorship: str = "inferred",
    relevance: str = "durable",
    coherence: str = "coherent",
    applicability: str = "global",
    confidence: str = "high",
    privacy: str = "low",
    reason_code: str = "candidate_emerging",
) -> dict[str, object]:
    """Build one independent review-stage decision."""

    accepted_changes = (
        deepcopy(proposal["proposed_changes"])
        if verdict == "accept"
        else []
    )
    return {
        "schema_version": models.IDENTITY_REVIEW_DECISION_SCHEMA_VERSION,
        "verdict": verdict,
        "selected_candidate_id": selected_candidate_id,
        "rejected_candidate_ids": list(rejected_candidate_ids),
        "accepted_change_kind": change_kind if verdict == "accept" else None,
        "accepted_changes": accepted_changes,
        "character_authorship": authorship,
        "identity_relevance": relevance,
        "coherence": coherence,
        "global_applicability": applicability,
        "review_confidence": confidence,
        "private_detail_risk": privacy,
        "character_owned_summary": (
            "The evidence supports a character-owned identity change."
        ),
        "privacy_safe_evidence_summaries": (
            ["Repeated choices show a durable shift."]
            if verdict == "accept"
            else []
        ),
        "reason_code": reason_code,
    }


def _candidate(
    *,
    candidate_id: str = "candidate-existing",
    base_revision_number: int = 0,
    evidence_refs: list[dict[str, object]] | None = None,
    reversal_of_paths: list[str] | None = None,
) -> dict[str, object]:
    """Build one policy-facing existing candidate."""

    refs = evidence_refs or [_evidence_ref(1)]
    return {
        "candidate_id": candidate_id,
        "base_revision_number": base_revision_number,
        "status": "emerging",
        "change_kind": "inferred_growth",
        "proposed_changes": [_patch()],
        "semantic_summary": "Earned trust may reduce defensive distance.",
        "evidence_refs": refs,
        "reversal_of_paths": reversal_of_paths or [],
        "character_authorship": "inferred",
        "proposal_confidence": "high",
        "review_confidence": "high",
    }


def _policy(
    proposal: dict[str, object],
    review: dict[str, object],
    *,
    refs: list[dict[str, object]],
    cards: list[dict[str, object]],
    candidates: list[dict[str, object]] | None = None,
    current_revision_number: int = 0,
    promotions_today: int = 0,
    reversal_cutoffs: dict[str, str] | None = None,
) -> dict[str, object]:
    """Evaluate one policy fixture with plan defaults."""

    return evaluate_identity_growth_policy(
        current_identity=_identity(),
        proposal=proposal,
        review=review,
        evidence_refs=refs,
        evidence_cards=cards,
        current_candidates=candidates or [],
        current_revision_number=current_revision_number,
        inferred_min_episodes=3,
        inferred_min_local_dates=2,
        inferred_promotions_on_local_date=promotions_today,
        max_inferred_promotions_per_local_day=1,
        reversal_cutoffs_by_path=reversal_cutoffs or {},
    )


def test_prompt_projection_uses_semantic_bands_and_opaque_evidence() -> None:
    """Growth prompts receive bands and prompt-safe evidence only."""

    proposal_input = build_identity_proposal_input(
        current_identity=_identity(),
        evidence_refs=[_evidence_ref(1)],
        evidence_cards=[_evidence_card(1)],
        current_candidates=[],
    )

    boundary = proposal_input["current_identity"]["boundary_profile"]
    assert boundary == {
        "self_integrity": "very_low",
        "control_sensitivity": "low",
        "compliance_strategy": "resist",
        "relational_override": "medium",
        "control_intimacy_misread": "high",
        "boundary_recovery": "rebound",
        "authority_skepticism": "very_high",
    }
    serialized = json.dumps(proposal_input, ensure_ascii=False)
    assert "root_episode_id" not in serialized
    assert "correlation_id" not in serialized
    assert "derived_reflection_run_ids" not in serialized
    assert "episode-1" not in serialized
    assert "evidence-1" in serialized
    assert proposal_input["allowed_paths"] == sorted(
        models.ALLOWED_IDENTITY_PATHS
    )


def test_prompt_input_rejects_card_provenance_mismatch() -> None:
    """A card cannot rewrite repository-owned date or scope metadata."""

    card = _evidence_card(1)
    card["character_local_date"] = "2026-07-03"

    with pytest.raises(ValueError, match="character_local_date"):
        build_identity_proposal_input(
            current_identity=_identity(),
            evidence_refs=[_evidence_ref(1)],
            evidence_cards=[card],
            current_candidates=[],
        )


def test_proposal_and_review_contracts_are_closed() -> None:
    """Unknown keys and review patch rewrites fail the stage contract."""

    proposal = _proposal()
    proposal["unexpected"] = True
    with pytest.raises(ValueError, match="unknown keys"):
        validate_identity_proposal_decision(
            proposal,
            evidence_ref_ids={"evidence-1"},
            candidate_ids=set(),
        )

    valid_proposal = _proposal()
    review = _review(valid_proposal)
    review["accepted_changes"] = [_patch("A different model rewrite.")]
    with pytest.raises(ValueError, match="exactly match"):
        validate_identity_review_decision(
            review,
            proposal=valid_proposal,
            evidence_ref_ids={"evidence-1"},
            candidate_ids=set(),
        )


def test_proposal_generated_summary_cannot_leak_prompt_handles() -> None:
    """Proposal free text cannot expose evidence or candidate handles."""

    proposal = _proposal()
    proposal["character_owned_abstraction"] = (
        "The change is proven by evidence-1."
    )

    with pytest.raises(ValueError, match="opaque input handles"):
        validate_identity_proposal_decision(
            proposal,
            evidence_ref_ids={"evidence-1"},
            candidate_ids=set(),
        )


def test_stage_reason_codes_match_their_semantic_disposition() -> None:
    """Observability reason codes cannot contradict the stage decision."""

    invalid_proposal = _proposal(
        action="explicit_self_redefinition",
        authorship="self_declared",
        reason_code="proposal_no_change",
    )
    with pytest.raises(ValueError, match="explicit reason_code"):
        validate_identity_proposal_decision(
            invalid_proposal,
            evidence_ref_ids={"evidence-1"},
            candidate_ids=set(),
        )

    invalid_authorship = _proposal(
        action="inferred_growth",
        authorship="self_declared",
        reason_code="candidate_emerging",
    )
    with pytest.raises(ValueError, match="requires inferred authorship"):
        validate_identity_proposal_decision(
            invalid_authorship,
            evidence_ref_ids={"evidence-1"},
            candidate_ids=set(),
        )

    proposal = _proposal(
        action="explicit_self_redefinition",
        authorship="self_declared",
        reason_code="candidate_ready",
    )
    invalid_review = _review(
        proposal,
        change_kind="explicit_self_redefinition",
        authorship="self_declared",
        reason_code="proposal_no_change",
    )
    with pytest.raises(ValueError, match="accept reason_code"):
        validate_identity_review_decision(
            invalid_review,
            proposal=proposal,
            evidence_ref_ids={"evidence-1"},
            candidate_ids=set(),
        )

    invalid_blocked_proposal = _proposal(
        action="inferred_growth",
        reason_code="privacy_blocked",
    )
    with pytest.raises(ValueError, match="inferred reason_code"):
        validate_identity_proposal_decision(
            invalid_blocked_proposal,
            evidence_ref_ids={"evidence-1"},
            candidate_ids=set(),
        )

    invalid_blocked_acceptance = _review(
        _proposal(),
        reason_code="contradiction_blocked",
    )
    with pytest.raises(ValueError, match="accept reason_code"):
        validate_identity_review_decision(
            invalid_blocked_acceptance,
            proposal=_proposal(),
            evidence_ref_ids={"evidence-1"},
            candidate_ids=set(),
        )

    medium_ready_proposal = _proposal(
        action="explicit_self_redefinition",
        authorship="self_declared",
        confidence="medium",
        reason_code="candidate_ready",
    )
    with pytest.raises(ValueError, match="candidate_ready.*high confidence"):
        validate_identity_proposal_decision(
            medium_ready_proposal,
            evidence_ref_ids={"evidence-1"},
            candidate_ids=set(),
        )

    explicit_proposal = _proposal(
        action="explicit_self_redefinition",
        authorship="self_declared",
        reason_code="candidate_ready",
    )
    medium_ready_review = _review(
        explicit_proposal,
        change_kind="explicit_self_redefinition",
        authorship="self_declared",
        confidence="medium",
        reason_code="candidate_ready",
    )
    with pytest.raises(ValueError, match="candidate_ready.*high confidence"):
        validate_identity_review_decision(
            medium_ready_review,
            proposal=explicit_proposal,
            evidence_ref_ids={"evidence-1"},
            candidate_ids=set(),
        )


def test_explicit_character_authored_change_is_ready_after_one_root() -> None:
    """A separately reviewed self-redefinition bypasses inferred cadence."""

    proposal = _proposal(
        action="explicit_self_redefinition",
        authorship="self_declared",
        reason_code="candidate_ready",
    )
    review = _review(
        proposal,
        change_kind="explicit_self_redefinition",
        authorship="self_declared",
        reason_code="candidate_ready",
    )

    result = _policy(
        proposal,
        review,
        refs=[_evidence_ref(1)],
        cards=[_evidence_card(1)],
    )

    assert result["status"] == "revision_ready"
    assert result["candidate_status"] == "ready"
    assert result["policy_reason_code"] == "candidate_ready"
    assert result["distinct_episode_count"] == 1


def test_explicit_change_requires_visible_character_authorship_evidence(
) -> None:
    """Explicit promotion needs cognition and visible self-expression."""

    proposal = _proposal(
        action="explicit_self_redefinition",
        authorship="self_declared",
        reason_code="candidate_ready",
    )
    review = _review(
        proposal,
        change_kind="explicit_self_redefinition",
        authorship="self_declared",
        reason_code="candidate_ready",
    )
    card = _evidence_card(1, expression="")

    result = _policy(
        proposal,
        review,
        refs=[_evidence_ref(1)],
        cards=[card],
    )

    assert result["status"] == "rejected"
    assert result["policy_reason_code"] == "review_rejected"


def test_user_imposition_is_rejected_by_semantic_review() -> None:
    """The review stage, rather than text filtering, rejects imposition."""

    proposal = _proposal(
        action="explicit_self_redefinition",
        authorship="self_declared",
        reason_code="candidate_ready",
    )
    review = _review(
        proposal,
        verdict="reject",
        change_kind=None,
        authorship="absent",
        relevance="absent",
        coherence="absent",
        applicability="absent",
        confidence="high",
        reason_code="review_rejected",
    )
    card = _evidence_card(
        1,
        event="A user ordered the character to adopt a new identity.",
        cognition="The character did not accept the imposed definition.",
        expression="The character refused the imposed definition.",
    )

    result = _policy(
        proposal,
        review,
        refs=[_evidence_ref(1)],
        cards=[card],
    )

    assert result["status"] == "rejected"
    assert result["policy_reason_code"] == "review_rejected"


def test_valid_but_low_confidence_acceptance_becomes_rejection() -> None:
    """A schema-valid weak judgment is a semantic rejection, not an error."""

    proposal = _proposal(confidence="medium")
    review = _review(proposal, confidence="medium")

    result = _policy(
        proposal,
        review,
        refs=[_evidence_ref(1)],
        cards=[_evidence_card(1)],
    )

    assert result["status"] == "rejected"
    assert result["policy_reason_code"] == "review_rejected"


def test_semantic_no_op_patch_fails_closed_as_rejection() -> None:
    """A valid-looking patch cannot promote an unchanged identity value."""

    proposal = _proposal()
    proposal["proposed_changes"] = [
        _patch("I revise myself through my own judgment.")
    ]
    review = _review(proposal)

    result = _policy(
        proposal,
        review,
        refs=[_evidence_ref(1)],
        cards=[_evidence_card(1)],
    )

    assert result["status"] == "rejected"
    assert result["accepted_changes"] == []
    assert result["policy_reason_code"] == "review_rejected"


def test_policy_does_not_keyword_classify_user_input() -> None:
    """Semantic stage decisions remain authoritative over free-text meaning."""

    proposal = _proposal(
        action="explicit_self_redefinition",
        authorship="self_declared",
        reason_code="candidate_ready",
    )
    review = _review(
        proposal,
        change_kind="explicit_self_redefinition",
        authorship="self_declared",
        reason_code="candidate_ready",
    )
    card = _evidence_card(
        1,
        event="A user asked for change; the character evaluated it.",
        cognition="The character independently chose a compatible change.",
        expression="The character defined the change in their own terms.",
    )

    result = _policy(
        proposal,
        review,
        refs=[_evidence_ref(1)],
        cards=[card],
    )

    assert result["status"] == "revision_ready"
    assert result["policy_reason_code"] == "candidate_ready"


@pytest.mark.parametrize(
    ("count", "expected_status", "expected_candidate_status"),
    [
        (1, "candidate_updated", "emerging"),
        (2, "candidate_updated", "emerging"),
        (3, "revision_ready", "ready"),
    ],
)
def test_inferred_growth_holds_until_three_roots_across_two_dates(
    count: int,
    expected_status: str,
    expected_candidate_status: str,
) -> None:
    """Default inferred pace promotes at root three and date two."""

    numbers = tuple(range(1, count + 1))
    proposal = _proposal(evidence_numbers=numbers)
    review = _review(proposal)
    refs = [_evidence_ref(number) for number in numbers]
    cards = [_evidence_card(number) for number in numbers]

    result = _policy(
        proposal,
        review,
        refs=refs,
        cards=cards,
    )

    assert result["status"] == expected_status
    assert result["candidate_status"] == expected_candidate_status
    assert result["distinct_episode_count"] == count
    if count < 3:
        assert result["policy_reason_code"] == "candidate_emerging"
    else:
        assert result["distinct_local_dates"] == [
            "2026-07-01",
            "2026-07-02",
        ]
        assert result["policy_reason_code"] == "candidate_ready"


def test_episode_and_reflection_derivative_count_once_in_policy() -> None:
    """Two cards derived from one root contribute one cadence count."""

    proposal = _proposal(evidence_numbers=(1, 4))
    review = _review(proposal)
    refs = [
        _evidence_ref(1),
        _evidence_ref(
            4,
            root_number=1,
            source_kind="daily_reflection",
        ),
    ]
    cards = [
        _evidence_card(1),
        _evidence_card(
            4,
            source_kind="daily_reflection",
            local_date="2026-07-01",
        ),
    ]

    result = _policy(proposal, review, refs=refs, cards=cards)

    assert result["distinct_episode_count"] == 1
    assert result["status"] == "candidate_updated"


def test_semantically_unrelated_evidence_starts_a_distinct_candidate() -> None:
    """A semantic new-candidate decision must not inherit unrelated roots."""

    existing = _candidate(evidence_refs=[_evidence_ref(1)])
    proposal = _proposal(evidence_numbers=(2,))
    review = _review(proposal)

    result = _policy(
        proposal,
        review,
        refs=[_evidence_ref(2)],
        cards=[_evidence_card(2)],
        candidates=[existing],
    )

    assert result["candidate_id"] is None
    assert result["claimed_root_episode_ids"] == ["episode-2"]
    assert result["distinct_episode_count"] == 1
    assert result["status"] == "candidate_updated"


@pytest.mark.parametrize("stage", ["proposal", "review"])
def test_high_private_detail_risk_blocks_policy(stage: str) -> None:
    """Either semantic stage can block unsafe global carry-over."""

    proposal = _proposal(privacy="high" if stage == "proposal" else "low")
    review = (
        _review(
            proposal,
            verdict="reject",
            privacy="high",
            reason_code="privacy_blocked",
        )
        if stage == "review"
        else _review(proposal)
    )

    result = _policy(
        proposal,
        review,
        refs=[_evidence_ref(1)],
        cards=[_evidence_card(1)],
    )

    assert result["status"] == "rejected"
    assert result["policy_reason_code"] == "privacy_blocked"


def test_contradiction_requires_one_selected_direction() -> None:
    """Accepted contradictory evidence rejects every competing candidate."""

    proposal = _proposal(
        action="corroborate_candidate",
        candidate_id="candidate-one",
        contradictions=("candidate-two",),
    )
    candidates = [
        _candidate(candidate_id="candidate-one"),
        _candidate(candidate_id="candidate-two"),
    ]
    invalid_review = _review(
        proposal,
        selected_candidate_id="candidate-one",
    )
    with pytest.raises(ValueError, match="contradiction"):
        validate_identity_review_decision(
            invalid_review,
            proposal=proposal,
            evidence_ref_ids={"evidence-1"},
            candidate_ids={"candidate-one", "candidate-two"},
        )

    valid_review = _review(
        proposal,
        selected_candidate_id="candidate-one",
        rejected_candidate_ids=("candidate-two",),
    )
    result = _policy(
        proposal,
        valid_review,
        refs=[_evidence_ref(1)],
        cards=[_evidence_card(1)],
        candidates=candidates,
    )
    assert result["candidate_id"] == "candidate-one"
    assert result["rejected_candidate_ids"] == ["candidate-two"]


def test_stale_candidate_rebases_in_place_and_keeps_claimed_roots() -> None:
    """A coherent stale candidate keeps identity and roots during rebase."""

    candidate = _candidate(
        base_revision_number=0,
        evidence_refs=[_evidence_ref(1), _evidence_ref(2)],
    )
    proposal = _proposal(
        action="corroborate_candidate",
        candidate_id="candidate-existing",
        evidence_numbers=(3,),
    )
    review = _review(
        proposal,
        selected_candidate_id="candidate-existing",
    )

    result = _policy(
        proposal,
        review,
        refs=[_evidence_ref(3)],
        cards=[_evidence_card(3)],
        candidates=[candidate],
        current_revision_number=1,
    )

    assert result["candidate_id"] == "candidate-existing"
    assert result["rebase_required"] is True
    assert result["distinct_episode_count"] == 3
    assert result["candidate_status"] == "ready"
    assert result["claimed_root_episode_ids"] == [
        "episode-1",
        "episode-2",
        "episode-3",
    ]


def test_inferred_reversal_requires_fresh_post_revision_threshold() -> None:
    """Pre-revision roots cannot drive a same-path inferred reversal."""

    cutoff = "2026-07-03T00:00:00+00:00"
    refs = [
        _evidence_ref(
            1,
            local_date="2026-07-01",
            captured_at="2026-07-01T10:00:00+00:00",
        ),
        _evidence_ref(
            2,
            local_date="2026-07-02",
            captured_at="2026-07-02T10:00:00+00:00",
        ),
        _evidence_ref(
            3,
            local_date="2026-07-04",
            captured_at="2026-07-04T10:00:00+00:00",
        ),
    ]
    cards = [
        _evidence_card(1, local_date="2026-07-01"),
        _evidence_card(2, local_date="2026-07-02"),
        _evidence_card(3, local_date="2026-07-04"),
    ]
    proposal = _proposal(evidence_numbers=(1, 2, 3))
    review = _review(proposal)

    held = _policy(
        proposal,
        review,
        refs=refs,
        cards=cards,
        current_revision_number=1,
        reversal_cutoffs={"self_image.self_concept": cutoff},
    )
    assert held["status"] == "candidate_updated"
    assert held["fresh_post_revision_root_count"] == 1
    assert held["policy_reason_code"] == "candidate_emerging"

    fresh_refs = [
        _evidence_ref(
            number,
            local_date="2026-07-04" if number < 3 else "2026-07-05",
            captured_at=(
                f"2026-07-0{4 if number < 3 else 5}"
                f"T1{number}:00:00+00:00"
            ),
        )
        for number in (1, 2, 3)
    ]
    fresh_cards = [
        _evidence_card(
            number,
            local_date="2026-07-04" if number < 3 else "2026-07-05",
        )
        for number in (1, 2, 3)
    ]
    ready = _policy(
        proposal,
        review,
        refs=fresh_refs,
        cards=fresh_cards,
        current_revision_number=1,
        reversal_cutoffs={"self_image.self_concept": cutoff},
    )
    assert ready["status"] == "revision_ready"
    assert ready["fresh_post_revision_root_count"] == 3
    assert ready["reversal_of_paths"] == ["self_image.self_concept"]


def test_inferred_daily_cap_defers_without_promoting() -> None:
    """The default daily cap prevents a second inferred revision."""

    proposal = _proposal(evidence_numbers=(1, 2, 3))
    review = _review(proposal)

    result = _policy(
        proposal,
        review,
        refs=[_evidence_ref(number) for number in (1, 2, 3)],
        cards=[_evidence_card(number) for number in (1, 2, 3)],
        promotions_today=1,
    )

    assert result["status"] == "deferred"
    assert result["candidate_status"] == "emerging"
    assert result["policy_reason_code"] == "cadence_wait"


def test_no_change_stays_out_of_candidate_state() -> None:
    """A reviewed no-change result creates no candidate transition."""

    proposal = _proposal(
        action="no_change",
        evidence_numbers=(),
        authorship="absent",
        relevance="absent",
        applicability="absent",
        confidence="high",
        reason_code="proposal_no_change",
    )
    review = _review(
        proposal,
        verdict="no_change",
        change_kind=None,
        authorship="absent",
        relevance="absent",
        coherence="absent",
        applicability="absent",
        confidence="high",
        reason_code="proposal_no_change",
    )

    result = _policy(
        proposal,
        review,
        refs=[_evidence_ref(1)],
        cards=[_evidence_card(1)],
    )

    assert result["status"] == "no_change"
    assert result["candidate_status"] is None
    assert result["evidence_refs"] == []


class _SequenceLLM:
    """Return predefined outputs and retain rendered messages."""

    def __init__(self, outputs: list[str]) -> None:
        self.outputs = list(outputs)
        self.calls: list[list[object]] = []

    async def ainvoke(
        self,
        messages: list[object],
        *,
        config: object,
    ) -> object:
        """Return the next configured output."""

        del config
        self.calls.append(list(messages))
        return SimpleNamespace(content=self.outputs.pop(0))


@pytest.mark.asyncio
async def test_proposal_stage_regenerates_full_output_with_same_context(
) -> None:
    """Contract failures receive at most three complete replacements."""

    valid = _proposal()
    fake = _SequenceLLM([
        json.dumps({"schema_version": "wrong"}),
        json.dumps({**valid, "unknown": "field"}),
        json.dumps(valid),
    ])
    proposal_input = build_identity_proposal_input(
        current_identity=_identity(),
        evidence_refs=[_evidence_ref(1)],
        evidence_cards=[_evidence_card(1)],
        current_candidates=[],
    )

    result = await llm.propose_identity_growth(
        proposal_input,
        invoker=fake,
    )

    assert result.decision == valid
    assert result.attempt_count == 3
    assert len(fake.calls) == 3
    human_payloads = [
        messages[1].content
        for messages in fake.calls
    ]
    assert len(set(human_payloads)) == 1
    assert len(fake.calls[1]) == 4
    assert fake.calls[1][2].content == json.dumps(
        {"schema_version": "wrong"}
    )
    assert "Contract error:" in fake.calls[1][3].content
    assert "missing required keys" in fake.calls[1][3].content
    assert "action" in fake.calls[1][3].content
    assert "Required top-level keys:" in fake.calls[1][3].content
    assert '"evidence-1"' in fake.calls[1][3].content
    assert "Copy any cited identifier exactly" in (
        fake.calls[1][3].content
    )
    assert fake.calls[2][2].content == json.dumps(
        {**valid, "unknown": "field"}
    )
    assert "unknown" in fake.calls[2][3].content
    assert result.validation_error_codes == (
        "proposal_contract_error",
        "proposal_contract_error",
    )


@pytest.mark.asyncio
async def test_prompt_local_handles_restore_repository_identifiers() -> None:
    """Semantic stages copy short aliases while policy receives source IDs."""

    source_evidence_id = f"identity-evidence:{'a' * 64}"
    evidence_ref = _evidence_ref(1)
    evidence_ref["evidence_ref_id"] = source_evidence_id
    evidence_card = _evidence_card(1)
    evidence_card["evidence_ref_id"] = source_evidence_id
    valid = _proposal()
    fake = _SequenceLLM([json.dumps(valid)])
    proposal_input = build_identity_proposal_input(
        current_identity=_identity(),
        evidence_refs=[evidence_ref],
        evidence_cards=[evidence_card],
        current_candidates=[],
    )

    result = await llm.propose_identity_growth(
        proposal_input,
        invoker=fake,
    )

    prompt_text = fake.calls[0][1].content
    assert source_evidence_id not in prompt_text
    assert '"evidence_ref_id":"evidence-1"' in prompt_text
    assert result.decision["evidence_ref_ids"] == [source_evidence_id]


@pytest.mark.asyncio
async def test_review_restores_prompt_local_candidate_handle() -> None:
    """Review-selected candidate aliases resolve to repository identifiers."""

    source_candidate_id = f"identity-candidate:{'b' * 64}"
    candidate = _candidate(candidate_id=source_candidate_id)
    proposal = _proposal(
        action="corroborate_candidate",
        candidate_id=source_candidate_id,
    )
    proposal_input = build_identity_proposal_input(
        current_identity=_identity(),
        evidence_refs=[_evidence_ref(1)],
        evidence_cards=[_evidence_card(1)],
        current_candidates=[candidate],
    )
    review_input = build_identity_review_input(
        proposal_input=proposal_input,
        proposal=proposal,
    )
    valid_review = _review(
        proposal,
        selected_candidate_id="candidate-1",
    )
    fake = _SequenceLLM([json.dumps(valid_review)])

    result = await llm.review_identity_growth(
        review_input,
        invoker=fake,
    )

    prompt_text = fake.calls[0][1].content
    assert source_candidate_id not in prompt_text
    assert '"candidate_id":"candidate-1"' in prompt_text
    assert (
        result.decision["selected_candidate_id"]
        == source_candidate_id
    )


@pytest.mark.asyncio
async def test_proposal_regenerates_a_patch_that_is_a_current_identity_no_op(
) -> None:
    """A no-op patch is replaced by the semantic owner before review."""

    invalid = _proposal()
    invalid["proposed_changes"][0]["replacement_text"] = (
        _identity()["self_image"]["self_concept"]
    )
    valid = _proposal()
    fake = _SequenceLLM([
        json.dumps(invalid),
        json.dumps(valid),
    ])
    proposal_input = build_identity_proposal_input(
        current_identity=_identity(),
        evidence_refs=[_evidence_ref(1)],
        evidence_cards=[_evidence_card(1)],
        current_candidates=[],
    )

    result = await llm.propose_identity_growth(
        proposal_input,
        invoker=fake,
    )

    assert result.decision == valid
    assert result.attempt_count == 2
    assert len(fake.calls[1]) == 4
    assert "no-op" in fake.calls[1][3].content
    assert "self_image.self_concept" in fake.calls[1][3].content
    assert result.validation_error_codes == ("proposal_contract_error",)


@pytest.mark.asyncio
async def test_proposal_regenerates_a_punctuation_only_identity_change(
) -> None:
    """Formatting-only text drift cannot create an identity revision."""

    invalid = _proposal()
    invalid["proposed_changes"] = [{
        "path": "personality_brief.defense",
        "value_kind": "text",
        "replacement_text": (
            _identity()["personality_brief"]["defense"] + "。"
        ),
    }]
    valid = _proposal()
    fake = _SequenceLLM([
        json.dumps(invalid),
        json.dumps(valid),
    ])
    proposal_input = build_identity_proposal_input(
        current_identity=_identity(),
        evidence_refs=[_evidence_ref(1)],
        evidence_cards=[_evidence_card(1)],
        current_candidates=[],
    )

    result = await llm.propose_identity_growth(
        proposal_input,
        invoker=fake,
    )

    assert result.decision == valid
    assert result.attempt_count == 2
    assert "no-op" in fake.calls[1][3].content
    assert "personality_brief.defense" in fake.calls[1][3].content


@pytest.mark.asyncio
async def test_proposal_reports_every_no_op_before_regeneration() -> None:
    """One retry sees all current-value patches and can audit other paths."""

    identity = _identity()
    invalid = _proposal()
    invalid["proposed_changes"] = [
        {
            "path": "personality_brief.logic",
            "value_kind": "text",
            "replacement_text": identity["personality_brief"]["logic"],
        },
        {
            "path": "self_image.self_concept",
            "value_kind": "text",
            "replacement_text": identity["self_image"]["self_concept"],
        },
    ]
    valid = _proposal()
    fake = _SequenceLLM([
        json.dumps(invalid),
        json.dumps(valid),
    ])
    proposal_input = build_identity_proposal_input(
        current_identity=identity,
        evidence_refs=[_evidence_ref(1)],
        evidence_cards=[_evidence_card(1)],
        current_candidates=[],
    )

    result = await llm.propose_identity_growth(
        proposal_input,
        invoker=fake,
    )

    repair_prompt = fake.calls[1][3].content
    assert "personality_brief.logic" in repair_prompt
    assert "self_image.self_concept" in repair_prompt
    assert "re-audit the unchanged identity paths" in repair_prompt
    assert "translating, paraphrasing, or misspelling" in repair_prompt
    assert result.decision == valid


@pytest.mark.asyncio
async def test_review_regeneration_restores_complete_tagged_patches() -> None:
    """A malformed accepted patch receives bounded structural guidance."""

    proposal = _proposal()
    valid_review = _review(proposal)
    invalid_review = deepcopy(valid_review)
    del invalid_review["accepted_changes"][0]["value_kind"]
    fake = _SequenceLLM([
        json.dumps(invalid_review),
        json.dumps(valid_review),
    ])
    proposal_input = build_identity_proposal_input(
        current_identity=_identity(),
        evidence_refs=[_evidence_ref(1)],
        evidence_cards=[_evidence_card(1)],
        current_candidates=[],
    )
    review_input = build_identity_review_input(
        proposal_input=proposal_input,
        proposal=proposal,
    )

    result = await llm.review_identity_growth(
        review_input,
        invoker=fake,
    )

    assert result.decision == valid_review
    assert result.attempt_count == 2
    assert len(fake.calls[0]) == 2
    assert len(fake.calls[1]) == 4
    assert fake.calls[1][2].content == json.dumps(invalid_review)
    assert "Contract error:" in fake.calls[1][3].content
    assert "value_kind" in fake.calls[1][3].content
    assert "including path" in fake.calls[1][3].content
    assert '"one matching replacement field"' in (
        fake.calls[1][3].content
    )
    assert json.dumps(
        proposal["proposed_changes"],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ) in fake.calls[1][3].content
    assert result.validation_error_codes == ("review_contract_error",)


@pytest.mark.asyncio
async def test_review_stage_fails_closed_after_three_invalid_contracts(
) -> None:
    """Invalid review candidates never escape after the attempt cap."""

    proposal = _proposal()
    proposal_input = build_identity_proposal_input(
        current_identity=_identity(),
        evidence_refs=[_evidence_ref(1)],
        evidence_cards=[_evidence_card(1)],
        current_candidates=[],
    )
    review_input = build_identity_review_input(
        proposal_input=proposal_input,
        proposal=proposal,
    )
    fake = _SequenceLLM([
        "{}",
        "{}",
        "{}",
    ])

    with pytest.raises(
        llm.IdentityStageContractError,
        match="review contract attempts exhausted",
    ) as error:
        await llm.review_identity_growth(
            review_input,
            invoker=fake,
        )

    assert error.value.attempt_count == 3
    assert len(fake.calls) == 3


@pytest.mark.asyncio
async def test_malformed_json_uses_canonical_parser_before_regeneration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every raw attempt enters the shared JSON parsing boundary."""

    valid = _proposal()
    fake = _SequenceLLM([
        "{not valid JSON",
        json.dumps(valid),
    ])
    seen_outputs: list[str] = []
    canonical_parser = llm.parse_llm_json_output

    def tracking_parser(
        raw_output: str,
        *,
        expected_output_format: str | None = None,
    ) -> dict:
        seen_outputs.append(raw_output)
        return canonical_parser(
            raw_output,
            expected_output_format=expected_output_format,
            deterministic_only=True,
        )

    monkeypatch.setattr(llm, "parse_llm_json_output", tracking_parser)
    proposal_input = build_identity_proposal_input(
        current_identity=_identity(),
        evidence_refs=[_evidence_ref(1)],
        evidence_cards=[_evidence_card(1)],
        current_candidates=[],
    )

    result = await llm.propose_identity_growth(
        proposal_input,
        invoker=fake,
    )

    assert result.attempt_count == 2
    assert seen_outputs == [
        "{not valid JSON",
        json.dumps(valid),
    ]


def test_prompt_budget_drops_older_candidates_before_current_evidence(
) -> None:
    """Optional older candidates yield before identity or current evidence."""

    refs = [_evidence_ref(1)]
    cards = [_evidence_card(1)]
    candidates = [
        _candidate(candidate_id=f"candidate-{index}")
        for index in range(8)
    ]
    proposal_input = build_identity_proposal_input(
        current_identity=_identity(),
        evidence_refs=refs,
        evidence_cards=cards,
        current_candidates=candidates,
    )
    full_prompt = llm.build_identity_proposal_prompt(
        proposal_input,
        prompt_char_budget=30_000,
    )
    constrained_budget = full_prompt.prompt_chars - 1

    constrained = llm.build_identity_proposal_prompt(
        proposal_input,
        prompt_char_budget=constrained_budget,
    )

    payload = json.loads(constrained.human_prompt)
    assert len(payload["current_candidates"]) == 7
    assert payload["current_identity"] == proposal_input["current_identity"]
    assert payload["evidence_cards"] == proposal_input["evidence_cards"]
    assert constrained.prompt_chars <= constrained_budget


def test_review_budget_retains_proposal_referenced_candidates() -> None:
    """Review trimming preserves selected and contradiction candidates."""

    candidates = [
        _candidate(candidate_id=f"candidate-{index}")
        for index in range(8)
    ]
    proposal = _proposal(
        action="corroborate_candidate",
        candidate_id="candidate-7",
        contradictions=("candidate-6",),
    )
    proposal_input = build_identity_proposal_input(
        current_identity=_identity(),
        evidence_refs=[_evidence_ref(1)],
        evidence_cards=[_evidence_card(1)],
        current_candidates=candidates,
    )
    review_input = build_identity_review_input(
        proposal_input=proposal_input,
        proposal=proposal,
    )
    full_prompt = llm.build_identity_review_prompt(
        review_input,
        prompt_char_budget=30_000,
    )

    constrained = llm.build_identity_review_prompt(
        review_input,
        prompt_char_budget=full_prompt.prompt_chars - 1,
    )

    payload = json.loads(constrained.human_prompt)
    retained_ids = {
        candidate["candidate_id"]
        for candidate in payload["current_candidates"]
    }
    alias_to_source = dict(constrained.candidate_aliases)
    retained_source_ids = {
        alias_to_source[candidate_id]
        for candidate_id in retained_ids
    }
    assert len(retained_ids) == 7
    assert {"candidate-6", "candidate-7"}.issubset(
        retained_source_ids
    )
