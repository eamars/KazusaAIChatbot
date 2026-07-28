"""Validation and full-snapshot tests for character identity growth."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy

import pytest

from kazusa_ai_chatbot.character_identity_growth.identity import (
    apply_identity_patches,
)
from kazusa_ai_chatbot.character_identity_growth.validation import (
    validate_effective_identity,
    validate_identity_patch,
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
