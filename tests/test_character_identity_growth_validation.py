"""Validation and full-snapshot tests for character identity growth."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy

import pytest

from kazusa_ai_chatbot.character_identity_growth import models
from kazusa_ai_chatbot.character_identity_growth.identity import (
    apply_identity_patches,
)
from kazusa_ai_chatbot.character_identity_growth.validation import (
    IdentityContractViolation,
    validate_effective_identity,
    validate_identity_patch,
    validate_identity_proposal_wire,
    validate_identity_review_wire,
)


def test_validates_a_complete_effective_identity() -> None:
    """The strict revision snapshot should accept every canonical field."""

    identity = _identity()

    validated = validate_effective_identity(identity)

    assert validated == identity
    assert validated is not identity


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda row: row.update({"unknown": "value"}), "unknown"),
        (lambda row: row.pop("backstory"), "backstory"),
        (
            lambda row: row["self_image"].update({"self_concept": ""}),
            "self_concept",
        ),
        (
            lambda row: row.update({"visual_characterization": ""}),
            "visual_characterization",
        ),
        (
            lambda row: row["boundary_profile"].update({
                "self_integrity": 1.1,
            }),
            "self_integrity",
        ),
        (
            lambda row: row["boundary_profile"].update({
                "compliance_strategy": "submit",
            }),
            "compliance_strategy",
        ),
        (
            lambda row: row["self_image"].update({
                "current_growth_edges": ["edge"] * 6,
            }),
            "current_growth_edges",
        ),
    ],
)
def test_rejects_invalid_full_snapshot(
    mutation: Callable[[dict[str, object]], None],
    match: str,
) -> None:
    """Missing, unknown, out-of-range, and empty values should fail closed."""

    identity = _identity()
    mutation(identity)

    with pytest.raises(ValueError, match=match):
        validate_effective_identity(identity)


@pytest.mark.parametrize(
    ("patch", "expected"),
    [
        (
            {
                "path": "name",
                "value_kind": "text",
                "replacement_text": "A chosen name",
            },
            "A chosen name",
        ),
        (
            {
                "path": "age",
                "value_kind": "integer",
                "replacement_integer": 18,
            },
            18,
        ),
        (
            {
                "path": "boundary_profile.self_integrity",
                "value_kind": "semantic_band",
                "replacement_band": "high",
            },
            "high",
        ),
        (
            {
                "path": "boundary_profile.compliance_strategy",
                "value_kind": "closed_enum",
                "replacement_enum": "resist",
            },
            "resist",
        ),
        (
            {
                "path": "self_image.current_growth_edges",
                "value_kind": "text_list",
                "replacement_items": ["Speak sooner under pressure"],
            },
            ["Speak sooner under pressure"],
        ),
    ],
)
def test_validates_each_tagged_patch_kind(
    patch: dict[str, object],
    expected: object,
) -> None:
    """Each path family should accept only its declared value member."""

    validated = validate_identity_patch(patch)

    assert expected in validated.values()


@pytest.mark.parametrize(
    "patch",
    [
        {
            "path": "global_user_id",
            "value_kind": "text",
            "replacement_text": "forbidden",
        },
        {
            "path": "age",
            "value_kind": "text",
            "replacement_text": "eighteen",
        },
        {
            "path": "boundary_profile.self_integrity",
            "value_kind": "semantic_band",
            "replacement_band": "maximum",
        },
        {
            "path": "name",
            "value_kind": "text",
            "replacement_text": "A chosen name",
            "replacement_integer": 18,
        },
    ],
)
def test_rejects_forbidden_mismatched_or_conflicting_patch(
    patch: dict[str, object],
) -> None:
    """Operational paths and union conflicts must not reach persistence."""

    with pytest.raises(ValueError):
        validate_identity_patch(patch)


def test_applies_typed_patches_to_a_new_full_snapshot() -> None:
    """Accepted patches should replace seed values without mutating history."""

    original = _identity()
    before = deepcopy(original)
    patches = [
        {
            "path": "self_image.self_concept",
            "value_kind": "text",
            "replacement_text": "I act with deliberate calm.",
        },
        {
            "path": "boundary_profile.self_integrity",
            "value_kind": "semantic_band",
            "replacement_band": "very_high",
        },
    ]

    updated, diffs = apply_identity_patches(original, patches)

    assert original == before
    assert updated["self_image"]["self_concept"] == (
        "I act with deliberate calm."
    )
    assert updated["boundary_profile"]["self_integrity"] == pytest.approx(0.9)
    assert [row["path"] for row in diffs] == [
        "boundary_profile.self_integrity",
        "self_image.self_concept",
    ]
    assert validate_effective_identity(updated) == updated


def test_rejects_duplicate_or_noop_patches() -> None:
    """One revision cannot carry duplicate paths or pretend a no-op is growth."""

    identity = _identity()
    duplicate = {
        "path": "name",
        "value_kind": "text",
        "replacement_text": "Changed",
    }
    noop = {
        "path": "name",
        "value_kind": "text",
        "replacement_text": identity["name"],
    }

    with pytest.raises(ValueError, match="duplicate"):
        apply_identity_patches(identity, [duplicate, duplicate])
    with pytest.raises(ValueError, match="no-op"):
        apply_identity_patches(identity, [noop])


def test_maps_v2_proposal_replacements_and_derives_reason_code() -> None:
    """The model-facing proposal shape maps to the stable internal V1 shape."""

    payload = {
        "action": "corroborate_candidate",
        "candidate_index": 2,
        "proposed_changes": [{
            "path": "self_image.self_concept",
            "replacement": "I let earned trust temper defensive distance.",
        }],
        "character_authorship": "inferred",
        "identity_relevance": "durable",
        "global_applicability": "global",
        "confidence": "high",
        "private_detail_risk": "low",
        "character_owned_abstraction": "Earned trust changes my distance.",
        "evidence_indices": [1],
        "contradiction_candidate_indices": [],
    }

    validated = validate_identity_proposal_wire(
        payload,
        evidence_ref_ids={"evidence-1", "evidence-2"},
        candidate_ids={"candidate-1", "candidate-2"},
    )

    assert validated["schema_version"] == (
        models.IDENTITY_PROPOSAL_DECISION_SCHEMA_VERSION
    )
    assert validated["candidate_id"] == "candidate-2"
    assert validated["evidence_ref_ids"] == ["evidence-1"]
    assert validated["proposed_changes"] == [{
        "path": "self_image.self_concept",
        "value_kind": "text",
        "replacement_text": "I let earned trust temper defensive distance.",
    }]
    assert validated["reason_code"] == "candidate_ready"


def test_v2_no_change_derives_privacy_reason_without_model_reason_field() -> None:
    """No-change privacy decisions receive their reason at the boundary."""

    payload = {
        "action": "no_change",
        "candidate_index": None,
        "proposed_changes": [],
        "character_authorship": "absent",
        "identity_relevance": "absent",
        "global_applicability": "absent",
        "confidence": "high",
        "private_detail_risk": "high",
        "character_owned_abstraction": "The detail cannot be made global-safe.",
        "evidence_indices": [1],
        "contradiction_candidate_indices": [],
    }

    validated = validate_identity_proposal_wire(
        payload,
        evidence_ref_ids={"evidence-1"},
        candidate_ids=set(),
    )

    assert validated["reason_code"] == "privacy_blocked"
    assert validated["candidate_id"] is None
    assert validated["proposed_changes"] == []


def test_v2_review_copies_validated_proposal_changes() -> None:
    """An accepted review cannot rewrite the proposal's semantic patch."""

    proposal = {
        "action": "inferred_growth",
        "candidate_index": None,
        "proposed_changes": [{
            "path": "self_image.self_concept",
            "replacement": "I let earned trust temper defensive distance.",
        }],
        "character_authorship": "inferred",
        "identity_relevance": "durable",
        "global_applicability": "global",
        "confidence": "high",
        "private_detail_risk": "low",
        "character_owned_abstraction": "Earned trust changes my distance.",
        "evidence_indices": [1],
        "contradiction_candidate_indices": [],
    }
    review = {
        "verdict": "accept",
        "selected_candidate_index": None,
        "rejected_candidate_indices": [],
        "character_authorship": "inferred",
        "identity_relevance": "durable",
        "coherence": "coherent",
        "global_applicability": "global",
        "review_confidence": "high",
        "private_detail_risk": "low",
        "character_owned_summary": "The change is coherent and global-safe.",
        "privacy_safe_evidence_summaries": [
            "Independent choices support a durable shift."
        ],
    }

    validated = validate_identity_review_wire(
        review,
        proposal=proposal,
        evidence_ref_ids={"evidence-1"},
        candidate_ids=set(),
    )

    assert validated["accepted_change_kind"] == "inferred_growth"
    assert validated["accepted_changes"] == [{
        "path": "self_image.self_concept",
        "value_kind": "text",
        "replacement_text": "I let earned trust temper defensive distance.",
    }]
    assert validated["reason_code"] == "candidate_ready"


def test_v2_invalid_indices_report_bounded_typed_violations() -> None:
    """A bad provenance index is recoverable contract evidence, not a crash string."""

    payload = {
        "action": "no_change",
        "candidate_index": None,
        "proposed_changes": [],
        "character_authorship": "absent",
        "identity_relevance": "absent",
        "global_applicability": "absent",
        "confidence": "medium",
        "private_detail_risk": "low",
        "character_owned_abstraction": "No durable change is supported.",
        "evidence_indices": [2],
        "contradiction_candidate_indices": [],
    }

    with pytest.raises(IdentityContractViolation) as error:
        validate_identity_proposal_wire(
            payload,
            evidence_ref_ids={"evidence-1"},
            candidate_ids=set(),
        )

    assert any(
        violation["code"] == "invalid_index"
        and violation["field"] == "evidence_indices[0]"
        for violation in error.value.violations
    )


def test_v2_indices_follow_display_order_and_new_growth_has_no_candidate() -> None:
    """Prompt row order, rather than lexical handle order, owns index meaning."""

    corroboration = {
        "action": "corroborate_candidate",
        "candidate_index": 2,
        "proposed_changes": [{
            "path": "self_image.self_concept",
            "replacement": "I let earned trust temper defensive distance.",
        }],
        "character_authorship": "inferred",
        "identity_relevance": "durable",
        "global_applicability": "global",
        "confidence": "high",
        "private_detail_risk": "low",
        "character_owned_abstraction": "Earned trust changes my distance.",
        "evidence_indices": [1],
        "contradiction_candidate_indices": [],
    }
    validated = validate_identity_proposal_wire(
        corroboration,
        evidence_ref_ids=("evidence-displayed-first", "evidence-second"),
        candidate_ids=("candidate-displayed-first", "candidate-selected"),
    )

    assert validated["candidate_id"] == "candidate-selected"
    assert validated["evidence_ref_ids"] == ["evidence-displayed-first"]

    inferred = deepcopy(corroboration)
    inferred["action"] = "inferred_growth"
    with pytest.raises(IdentityContractViolation, match="candidate_index"):
        validate_identity_proposal_wire(
            inferred,
            evidence_ref_ids=("evidence-displayed-first",),
            candidate_ids=("candidate-selected",),
        )


def test_v2_wire_boundary_rejects_v1_and_hybrid_payloads() -> None:
    """Raw semantic-stage inputs cannot cross the V2 boundary in V1 form."""

    payload = {
        "action": "no_change",
        "candidate_index": None,
        "proposed_changes": [],
        "character_authorship": "absent",
        "identity_relevance": "absent",
        "global_applicability": "absent",
        "confidence": "high",
        "private_detail_risk": "low",
        "character_owned_abstraction": "No durable identity change.",
        "evidence_indices": [],
        "contradiction_candidate_indices": [],
    }
    v1_or_hybrid = {
        **payload,
        "schema_version": models.IDENTITY_PROPOSAL_DECISION_SCHEMA_VERSION,
        "reason_code": "proposal_no_change",
    }

    with pytest.raises(IdentityContractViolation) as error:
        validate_identity_proposal_wire(
            v1_or_hybrid,
            evidence_ref_ids=(),
            candidate_ids=(),
        )

    assert {entry["code"] for entry in error.value.violations} == {
        "unknown_key",
    }

    with pytest.raises(IdentityContractViolation) as empty_error:
        validate_identity_proposal_wire(
            {},
            evidence_ref_ids=(),
            candidate_ids=(),
        )
    assert len(empty_error.value.violations) == len(models.PROPOSAL_WIRE_KEYS)
    assert {
        entry["code"] for entry in empty_error.value.violations
    } == {"missing_required_key"}

    review = {
        "verdict": "no_change",
        "selected_candidate_index": None,
        "rejected_candidate_indices": [],
        "character_authorship": "absent",
        "identity_relevance": "absent",
        "coherence": "absent",
        "global_applicability": "absent",
        "review_confidence": "high",
        "private_detail_risk": "low",
        "character_owned_summary": "No durable identity change.",
        "privacy_safe_evidence_summaries": [],
        "schema_version": models.IDENTITY_REVIEW_DECISION_SCHEMA_VERSION,
        "reason_code": "proposal_no_change",
    }
    with pytest.raises(IdentityContractViolation) as review_error:
        validate_identity_review_wire(
            review,
            proposal=payload,
            evidence_ref_ids=(),
            candidate_ids=(),
        )
    assert {entry["code"] for entry in review_error.value.violations} == {
        "unknown_key",
    }


def _identity() -> dict[str, object]:
    """Build one character-generic complete identity fixture."""

    return {
        "name": "Initial identity",
        "description": "A complete generic identity used for contract tests.",
        "gender": "unspecified",
        "age": 17,
        "birthday": "01-01",
        "backstory": "A stable initial history.",
        "personality_brief": {
            "mbti": "INTJ",
            "logic": "Tests observations before drawing conclusions.",
            "tempo": "Measured and concise.",
            "defense": "Creates distance before answering pressure.",
            "quirks": "Pauses before important claims.",
            "taboos": "Rejects coerced identity claims.",
        },
        "boundary_profile": {
            "self_integrity": 0.7,
            "control_sensitivity": 0.6,
            "compliance_strategy": "evade",
            "relational_override": 0.3,
            "control_intimacy_misread": 0.2,
            "boundary_recovery": "rebound",
            "authority_skepticism": 0.5,
        },
        "linguistic_texture_profile": {
            "fragmentation": 0.2,
            "hesitation_density": 0.3,
            "counter_questioning": 0.4,
            "softener_density": 0.3,
            "formalism_avoidance": 0.6,
            "abstraction_reframing": 0.5,
            "direct_assertion": 0.7,
            "emotional_leakage": 0.2,
            "rhythmic_bounce": 0.4,
            "self_deprecation": 0.1,
        },
        "self_image": {
            "self_concept": "I value deliberate action.",
            "current_growth_edges": ["Respond before withdrawing"],
        },
        "visual_characterization": "A generic visual description.",
    }
