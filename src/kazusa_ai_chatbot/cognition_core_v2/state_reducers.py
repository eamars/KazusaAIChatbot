"""Deterministic translation of validated semantic propositions into proposals."""

from __future__ import annotations

from collections.abc import Iterable

from kazusa_ai_chatbot.cognition_core_v2.contracts import (
    SemanticProposition,
    TransitionProposal,
)


ROOT_ACTIVATION_DELTA = 1.0
ROOT_RESOLUTION_DELTA = -1.0


def proposals_from_semantic_propositions(
    propositions: Iterable[SemanticProposition],
    expected_state_version: int,
) -> list[TransitionProposal]:
    """Convert grounded causal appraisals into reducer-owned root proposals.

    Args:
        propositions: Structurally validated appraisals from one model call.
        expected_state_version: Local snapshot version used by that call.

    Returns:
        Causal proposals for the state store; grounded false propositions fade
        their former root instead of leaving it permanently active.
    """

    proposals = [
        TransitionProposal(
            entity_kind="goals",
            entity_ref=proposition.root_id,
            expected_state_version=expected_state_version,
            transition_kind="semantic_proposition",
            causal_source_refs=[proposition.causal_source_ref],
            numeric_delta={
                "activation": (
                    ROOT_ACTIVATION_DELTA
                    if proposition.present
                    else ROOT_RESOLUTION_DELTA
                ),
            },
            semantic_basis=proposition.semantic_basis,
        )
        for proposition in propositions
    ]
    return proposals
