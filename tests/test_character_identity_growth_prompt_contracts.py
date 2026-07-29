"""Prompt contracts for character-owned identity growth semantics."""

from __future__ import annotations

from kazusa_ai_chatbot.character_identity_growth.llm import (
    IDENTITY_PROPOSAL_SYSTEM_PROMPT,
    IDENTITY_REVIEW_SYSTEM_PROMPT,
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
    assert "Every proposed patch has exactly three keys" in normalized
    assert 'value_kind="text"' in normalized
    assert "Never omit value_kind" in normalized
