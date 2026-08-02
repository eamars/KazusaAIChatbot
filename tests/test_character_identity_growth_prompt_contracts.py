"""Prompt contracts for character-owned identity growth semantics."""

from __future__ import annotations

import json

from kazusa_ai_chatbot.character_identity_growth import models
from kazusa_ai_chatbot.character_identity_growth.llm import (
    IDENTITY_PROPOSAL_SYSTEM_PROMPT,
    IDENTITY_REVIEW_SYSTEM_PROMPT,
    build_identity_review_prompt,
)


def test_close_relationships_can_cause_private_safe_identity_growth() -> None:
    """Relationship experience is eligible when the change is character-owned."""

    combined = "\n".join((
        IDENTITY_PROPOSAL_SYSTEM_PROMPT,
        IDENTITY_REVIEW_SYSTEM_PROMPT,
    ))
    normalized = " ".join(combined.split())

    assert "Love, intimacy, or another close relationship may be evidence" in (
        normalized
    )
    assert "relationship target and relationship facts remain scoped" in (
        normalized
    )
    assert "proposed persisted abstraction" in normalized


def test_relationship_facts_remain_distinct_from_global_identity() -> None:
    """Global revisions retain the character change, not private relationship data."""

    combined = "\n".join((
        IDENTITY_PROPOSAL_SYSTEM_PROMPT,
        IDENTITY_REVIEW_SYSTEM_PROMPT,
    ))

    assert "participant identity" in combined
    assert "private_detail_risk=high" in combined
    assert "source topic involves intimacy" in combined


def test_open_self_reflection_can_remain_character_authored() -> None:
    """An open question is distinct from a supplied identity conclusion."""

    normalized_proposal = " ".join(
        IDENTITY_PROPOSAL_SYSTEM_PROMPT.split()
    )
    normalized_review = " ".join(
        IDENTITY_REVIEW_SYSTEM_PROMPT.split()
    )

    for normalized in (normalized_proposal, normalized_review):
        assert "open question" in normalized
        assert "retain, change, or reject" in normalized
        assert "supplies the desired identity conclusion" in normalized


def test_review_requires_a_coherent_post_patch_identity_snapshot() -> None:
    """Growth cannot hide behind one weak field while old authority conflicts."""

    normalized_proposal = " ".join(
        IDENTITY_PROPOSAL_SYSTEM_PROMPT.split()
    )
    normalized_review = " ".join(
        IDENTITY_REVIEW_SYSTEM_PROMPT.split()
    )

    assert "internally coherent full identity" in normalized_proposal
    assert "every directly conflicting allowed path" in normalized_proposal
    assert "mentally apply every proposed patch" in normalized_review
    assert "unchanged identity field remains directly incompatible" in (
        normalized_review
    )
    assert "path-by-path contradiction audit" in normalized_review
    assert "pressure-response rule" in normalized_review
    assert "personality_brief.logic" in normalized_proposal
    assert "personality_brief.defense" in normalized_proposal
    assert "matching field proves full coverage" in normalized_review
    assert "explicitly disavows an unchanged current identity field" in (
        normalized_proposal
    )
    assert "patch that exact allowed path" in normalized_proposal
    assert "secondary field already points in a similar direction" in (
        normalized_review
    )


def test_proposal_replaces_partially_disavowed_bundled_fields() -> None:
    """Retaining part of a field cannot preserve its rejected behavior."""

    normalized = " ".join(IDENTITY_PROPOSAL_SYSTEM_PROMPT.split())

    assert "Partial retention does not make a bundled current field" in (
        normalized
    )
    assert "replace the whole field" in normalized
    assert "Each patch has exactly two keys: path plus replacement" in normalized
    assert "Do not emit value_kind" in normalized


def test_v2_system_prompts_exclude_model_owned_internal_metadata() -> None:
    """The model sees semantic decisions while deterministic code owns metadata."""

    proposal = " ".join(IDENTITY_PROPOSAL_SYSTEM_PROMPT.split())
    review = " ".join(IDENTITY_REVIEW_SYSTEM_PROMPT.split())

    assert '"candidate_index"' in proposal
    assert '"evidence_indices"' in proposal
    assert '"schema_version"' not in proposal
    assert '"reason_code"' not in proposal
    assert '"accepted_changes"' not in review
    assert '"accepted_change_kind"' not in review
    assert '"schema_version"' not in review
    assert '"reason_code"' not in review


def test_prompt_renderer_uses_numeric_indices_and_uniform_replacements() -> None:
    """Rendered model context contains local indices and no repository handles."""

    evidence_id = f"identity-evidence:{'a' * 64}"
    candidate_id = f"identity-candidate:{'b' * 64}"
    review_input = {
        "schema_version": models.IDENTITY_REVIEW_INPUT_SCHEMA_VERSION,
        "current_identity": {
            "self_image": {
                "self_concept": "I revise myself through judgment.",
            },
        },
        "evidence_cards": [{
            "schema_version": models.IDENTITY_EVIDENCE_CARD_SCHEMA_VERSION,
            "evidence_ref_id": evidence_id,
            "source_kind": "settled_episode",
            "character_local_date": "2026-07-01",
            "scope_kind": "private",
            "decontextualized_event": "The character stayed present.",
            "character_cognition_summary": "The character reconsidered withdrawal.",
            "visible_self_expression_summary": "The character chose engagement.",
        }],
        "current_candidates": [{
            "candidate_id": candidate_id,
            "change_kind": "inferred_growth",
            "semantic_summary": "Earned trust may reduce distance.",
            "proposed_changes": [{
                "path": "self_image.self_concept",
                "value_kind": "text",
                "replacement_text": "I let earned trust temper distance.",
            }],
        }],
        "proposal_decision": {
            "schema_version": models.IDENTITY_PROPOSAL_DECISION_SCHEMA_VERSION,
            "action": "corroborate_candidate",
            "candidate_id": candidate_id,
            "proposed_changes": [{
                "path": "self_image.self_concept",
                "value_kind": "text",
                "replacement_text": "I let earned trust temper distance.",
            }],
            "character_authorship": "inferred",
            "identity_relevance": "durable",
            "global_applicability": "global",
            "confidence": "high",
            "private_detail_risk": "low",
            "character_owned_abstraction": "Earned trust changes distance.",
            "evidence_ref_ids": [evidence_id],
            "contradiction_candidate_ids": [],
            "reason_code": "candidate_ready",
        },
    }

    built = build_identity_review_prompt(review_input)
    human = json.loads(built.human_prompt)

    assert evidence_id not in built.system_prompt
    assert evidence_id not in built.human_prompt
    assert candidate_id not in built.system_prompt
    assert candidate_id not in built.human_prompt
    assert human["evidence_cards"][0]["evidence_index"] == 1
    assert human["current_candidates"][0]["candidate_index"] == 1
    assert "evidence_ref_id" not in human["evidence_cards"][0]
    assert "candidate_id" not in human["current_candidates"][0]
    assert "schema_version" not in human["evidence_cards"][0]
    assert "schema_version" not in human["current_candidates"][0]
    assert human["current_candidates"][0]["proposed_changes"] == [{
        "path": "self_image.self_concept",
        "replacement": "I let earned trust temper distance.",
    }]
