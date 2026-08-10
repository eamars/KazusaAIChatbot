"""Latest-only consumer projections for character identity growth."""

from __future__ import annotations

from copy import deepcopy

from kazusa_ai_chatbot.character_identity_growth.projection import (
    identity_projection_digest,
    project_identity_for_cognition,
    project_identity_for_surface,
    projected_identity_consumer_kinds,
)


def _identity(*, marker: str) -> dict[str, object]:
    """Build one complete generic identity carrying a revision marker."""

    return {
        "name": f"name-{marker}",
        "description": f"description-{marker}",
        "gender": f"gender-{marker}",
        "age": 20 if marker == "old" else 21,
        "birthday": f"birthday-{marker}",
        "backstory": f"backstory-{marker}",
        "personality_brief": {
            "mbti": f"mbti-{marker}",
            "logic": f"logic-{marker}",
            "tempo": f"tempo-{marker}",
            "defense": f"defense-{marker}",
            "quirks": f"quirks-{marker}",
            "taboos": f"taboos-{marker}",
        },
        "boundary_profile": {
            "self_integrity": 0.3 if marker == "old" else 0.7,
            "control_sensitivity": 0.3 if marker == "old" else 0.7,
            "compliance_strategy": "evade" if marker == "old" else "resist",
            "relational_override": 0.3 if marker == "old" else 0.7,
            "control_intimacy_misread": 0.3 if marker == "old" else 0.7,
            "boundary_recovery": (
                "delayed_rebound" if marker == "old" else "rebound"
            ),
            "authority_skepticism": 0.3 if marker == "old" else 0.7,
        },
        "linguistic_texture_profile": {
            "fragmentation": 0.3 if marker == "old" else 0.7,
            "hesitation_density": 0.3 if marker == "old" else 0.7,
            "counter_questioning": 0.3 if marker == "old" else 0.7,
            "softener_density": 0.3 if marker == "old" else 0.7,
            "formalism_avoidance": 0.3 if marker == "old" else 0.7,
            "abstraction_reframing": 0.3 if marker == "old" else 0.7,
            "direct_assertion": 0.3 if marker == "old" else 0.7,
            "emotional_leakage": 0.3 if marker == "old" else 0.7,
            "rhythmic_bounce": 0.3 if marker == "old" else 0.7,
            "self_deprecation": 0.3 if marker == "old" else 0.7,
        },
        "self_image": {
            "self_concept": f"self-concept-{marker}",
            "current_growth_edges": [f"growth-edge-{marker}"],
        },
        "visual_characterization": f"visual-{marker}",
    }


def _revision(*, marker: str, revision_number: int) -> dict[str, object]:
    """Wrap one identity in metadata that consumers must exclude."""

    return {
        "schema_version": "character_identity_revision.v1",
        "revision_id": f"revision-{revision_number}",
        "revision_number": revision_number,
        "effective_identity": _identity(marker=marker),
        "change_diff": [{
            "path": "name",
            "before": "private-old-value",
            "after": f"name-{marker}",
        }],
        "evidence_refs": [{"root_episode_id": "private-root"}],
    }


def test_cognition_projection_is_closed_and_latest_only() -> None:
    """Each V2 appraisal family should receive only its declared categories."""

    old_revision = _revision(marker="old", revision_number=0)
    latest_revision = _revision(marker="new", revision_number=1)

    old_projection = project_identity_for_cognition(old_revision)
    latest_projection = project_identity_for_cognition(latest_revision)

    assert set(latest_projection) == {
        "moral_identity",
        "existential_drive",
        "relationship_social",
        "event_agency",
        "goal_threat_outcome",
        "goal_cognition",
        "epistemic_comparison_memory",
    }
    assert set(latest_projection["moral_identity"]) == {
        "core",
        "personality",
        "boundaries",
        "self_image",
    }
    assert set(latest_projection["existential_drive"]) == {
        "core",
        "personality",
        "self_image",
    }
    for family in (
        "relationship_social",
        "event_agency",
        "goal_threat_outcome",
    ):
        assert set(latest_projection[family]) == {
            "personality",
            "boundaries",
        }
    assert set(latest_projection["goal_cognition"]) == {
        "core",
        "personality",
        "boundaries",
        "self_image",
    }
    assert latest_projection["epistemic_comparison_memory"] == {}
    assert (
        latest_projection["moral_identity"]["self_image"]["self_concept"]
        == "self-concept-new"
    )
    assert "self-concept-old" in str(old_projection)
    assert "self-concept-old" not in str(latest_projection)
    assert "revision_number" not in str(latest_projection)
    assert "change_diff" not in str(latest_projection)
    assert "private-root" not in str(latest_projection)


def test_character_self_epistemic_projection_is_explicitly_bounded() -> None:
    """Core identity reaches epistemic appraisal only for character-self work."""

    revision = _revision(marker="new", revision_number=1)

    ordinary = project_identity_for_cognition(revision)
    character_self = project_identity_for_cognition(
        revision,
        include_epistemic_core=True,
    )

    assert ordinary["epistemic_comparison_memory"] == {}
    assert set(character_self["epistemic_comparison_memory"]) == {"core"}
    assert (
        character_self["epistemic_comparison_memory"]["core"]["name"]
        == "name-new"
    )


def test_surface_projection_separates_text_visual_and_naming() -> None:
    """Text, visual, and naming consumers should get exact disjoint contexts."""

    revision = _revision(marker="new", revision_number=3)

    projection = project_identity_for_surface(revision)

    assert set(projection) == {"text", "visual", "naming"}
    assert set(projection["text"]) == {
        "name",
        "personality",
        "linguistic_texture_profile",
    }
    assert set(projection["text"]["personality"]) == {
        "tempo",
        "defense",
        "quirks",
    }
    assert projection["visual"] == {
        "name": "name-new",
        "description": "description-new",
        "gender": "gender-new",
        "age": 21,
        "visual_characterization": "visual-new",
    }
    assert projection["naming"] == {"name": "name-new"}
    assert "backstory-new" not in str(projection["text"])
    assert "self-concept-new" not in str(projection["visual"])
    assert "revision-3" not in str(projection)


def test_projection_returns_detached_values() -> None:
    """Consumer mutation must not alter the immutable revision snapshot."""

    revision = _revision(marker="new", revision_number=2)
    before = deepcopy(revision)

    cognition = project_identity_for_cognition(revision)
    surface = project_identity_for_surface(revision)
    cognition["moral_identity"]["core"]["name"] = "mutated"
    surface["text"]["personality"]["tempo"] = "mutated"

    assert revision == before


def test_projection_digest_and_consumers_track_exact_latest_context() -> None:
    """The durable receipt digest should change only with projected revision."""

    old_revision = _revision(marker="old", revision_number=0)
    latest_revision = _revision(marker="new", revision_number=1)
    old_cognition = project_identity_for_cognition(old_revision)
    old_surface = project_identity_for_surface(old_revision)
    latest_cognition = project_identity_for_cognition(latest_revision)
    latest_surface = project_identity_for_surface(latest_revision)

    old_digest = identity_projection_digest(
        revision_number=0,
        cognition_context=old_cognition,
        surface_context=old_surface,
    )
    latest_digest = identity_projection_digest(
        revision_number=1,
        cognition_context=latest_cognition,
        surface_context=latest_surface,
    )

    assert len(old_digest) == 64
    assert old_digest != latest_digest
    assert projected_identity_consumer_kinds(latest_cognition) == [
        "event_agency",
        "existential_drive",
        "goal_cognition",
        "goal_threat_outcome",
        "moral_identity",
        "naming",
        "relationship_social",
        "text",
        "visual",
    ]
