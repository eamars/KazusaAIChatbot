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
