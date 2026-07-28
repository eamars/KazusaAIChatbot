"""Core contracts for revisioned character identity growth."""

from __future__ import annotations

from kazusa_ai_chatbot.character_identity_growth import models


def test_supported_identity_path_registry_is_complete_and_closed() -> None:
    """All approved semantic leaves should have one canonical patch path."""

    expected_paths = {
        "name",
        "description",
        "gender",
        "age",
        "birthday",
        "backstory",
        "personality_brief.mbti",
        "personality_brief.logic",
        "personality_brief.tempo",
        "personality_brief.defense",
        "personality_brief.quirks",
        "personality_brief.taboos",
        "boundary_profile.self_integrity",
        "boundary_profile.control_sensitivity",
        "boundary_profile.compliance_strategy",
        "boundary_profile.relational_override",
        "boundary_profile.control_intimacy_misread",
        "boundary_profile.boundary_recovery",
        "boundary_profile.authority_skepticism",
        "linguistic_texture_profile.fragmentation",
        "linguistic_texture_profile.hesitation_density",
        "linguistic_texture_profile.counter_questioning",
        "linguistic_texture_profile.softener_density",
        "linguistic_texture_profile.formalism_avoidance",
        "linguistic_texture_profile.abstraction_reframing",
        "linguistic_texture_profile.direct_assertion",
        "linguistic_texture_profile.emotional_leakage",
        "linguistic_texture_profile.rhythmic_bounce",
        "linguistic_texture_profile.self_deprecation",
        "self_image.self_concept",
        "self_image.current_growth_edges",
        "visual_characterization",
    }

    assert models.ALLOWED_IDENTITY_PATHS == frozenset(expected_paths)
    assert len(models.ALLOWED_IDENTITY_PATHS) == 32


def test_path_kinds_partition_the_supported_surface() -> None:
    """Every path should accept exactly one tagged patch value kind."""

    partitions = (
        models.TEXT_IDENTITY_PATHS,
        models.INTEGER_IDENTITY_PATHS,
        models.NUMERIC_IDENTITY_PATHS,
        models.ENUM_IDENTITY_PATHS,
        models.TEXT_LIST_IDENTITY_PATHS,
    )
    combined = set().union(*partitions)

    assert combined == set(models.ALLOWED_IDENTITY_PATHS)
    for index, paths in enumerate(partitions):
        for other_paths in partitions[index + 1:]:
            assert paths.isdisjoint(other_paths)


def test_closed_enums_and_semantic_bands_match_the_plan() -> None:
    """Numeric and enum normalization should be declared centrally."""

    assert models.SEMANTIC_BAND_VALUES == {
        "very_low": 0.1,
        "low": 0.3,
        "medium": 0.5,
        "high": 0.7,
        "very_high": 0.9,
    }
    assert models.ENUM_VALUES_BY_PATH == {
        "boundary_profile.compliance_strategy": frozenset({
            "resist",
            "evade",
            "comply",
        }),
        "boundary_profile.boundary_recovery": frozenset({
            "rebound",
            "delayed_rebound",
            "decay",
            "detach",
        }),
    }


def test_candidate_transition_table_is_closed() -> None:
    """Terminal candidates must never return to an active state."""

    assert models.CANDIDATE_TRANSITIONS == {
        "emerging": frozenset({
            "emerging",
            "ready",
            "rejected",
            "superseded",
        }),
        "ready": frozenset({
            "promoted",
            "rejected",
            "superseded",
        }),
        "promoted": frozenset(),
        "rejected": frozenset(),
        "superseded": frozenset(),
    }


def test_reason_and_health_vocabularies_are_closed() -> None:
    """Operator diagnosis should not depend on free-form status strings."""

    assert models.IDENTITY_GROWTH_REASON_CODES == frozenset({
        "not_routed",
        "no_eligible_evidence",
        "proposal_no_change",
        "proposal_contract_failed",
        "candidate_emerging",
        "candidate_ready",
        "review_rejected",
        "review_contract_failed",
        "privacy_blocked",
        "cadence_wait",
        "duplicate_root",
        "stale_base",
        "contradiction_blocked",
        "promotion_write_failed",
        "revision_promoted",
        "awaiting_first_consumption",
        "revision_consumed",
        "revision_consumption_mismatch",
    })
    assert models.IDENTITY_GROWTH_HEALTH_STATES == frozenset({
        "healthy_idle",
        "waiting_for_evidence",
        "semantic_rejection",
        "promotion_ready",
        "awaiting_consumption",
        "healthy_active",
        "pipeline_error",
        "consumption_error",
    })
