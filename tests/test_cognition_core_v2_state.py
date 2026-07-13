"""Deterministic local-state tests for the validation-only cognition core V2."""

import pytest

from kazusa_ai_chatbot.cognition_core_v2.contracts import (
    LocalStateKey,
    SemanticProposition,
    TransitionProposal,
)
from kazusa_ai_chatbot.cognition_core_v2.state_reducers import (
    proposals_from_semantic_propositions,
)
from kazusa_ai_chatbot.cognition_core_v2.state_store import (
    LocalStateStore,
    StateVersionConflictError,
)


@pytest.mark.asyncio
async def test_local_state_store_rejects_a_stale_transition_version() -> None:
    """Keep one state key serialized and fail closed on stale proposals."""

    state_store = LocalStateStore()
    state_key = LocalStateKey(
        character_global_id="state-character",
        current_user_global_id="state-user",
        trigger_source="user_message",
        target_scope_fingerprint="state-scope",
    )
    first_proposal = TransitionProposal(
        entity_kind="goals",
        entity_ref="goal-1",
        expected_state_version=0,
        transition_kind="causal_input",
        causal_source_refs=["fixture:turn-1"],
        numeric_delta={"satisfaction": 0.5},
        semantic_basis="deterministic fixture input",
    )

    updated_state = await state_store.apply_proposals(
        state_key,
        [first_proposal],
    )
    stale_proposal = TransitionProposal(
        entity_kind="goals",
        entity_ref="goal-1",
        expected_state_version=0,
        transition_kind="causal_input",
        causal_source_refs=["fixture:turn-2"],
        numeric_delta={"satisfaction": 0.1},
        semantic_basis="stale deterministic fixture input",
    )

    assert updated_state.state_version == 1

    with pytest.raises(StateVersionConflictError):
        await state_store.apply_proposals(state_key, [stale_proposal])


@pytest.mark.asyncio
async def test_local_state_store_keeps_fixture_scopes_isolated() -> None:
    """Keep independent cases from sharing process-local motivational state."""

    state_store = LocalStateStore()
    first_key = LocalStateKey(
        character_global_id="isolation-character",
        current_user_global_id="isolation-user-a",
        trigger_source="user_message",
        target_scope_fingerprint="scope-a",
    )
    second_key = LocalStateKey(
        character_global_id="isolation-character",
        current_user_global_id="isolation-user-b",
        trigger_source="user_message",
        target_scope_fingerprint="scope-b",
    )
    proposal = TransitionProposal(
        entity_kind="threats",
        entity_ref="threat-a",
        expected_state_version=0,
        transition_kind="causal_input",
        causal_source_refs=["fixture:isolation"],
        numeric_delta={"expected_harm": 0.8},
        semantic_basis="isolated fixture input",
    )

    await state_store.apply_proposals(first_key, [proposal])
    second_snapshot = await state_store.snapshot(second_key)

    assert second_snapshot.state_version == 0
    assert second_snapshot.threats == {}


def test_absent_semantic_cause_proposes_a_causal_fade() -> None:
    """Let a grounded resolution remove its former root through the reducer."""

    propositions = [
        SemanticProposition(
            root_id="credible_threat",
            present=False,
            causal_source_ref="fixture:resolved-threat",
            semantic_basis="the supplied event establishes that the threat ended",
        ),
    ]

    proposals = proposals_from_semantic_propositions(propositions, 3)

    assert proposals[0].numeric_delta == {"activation": -1.0}
    assert proposals[0].expected_state_version == 3
