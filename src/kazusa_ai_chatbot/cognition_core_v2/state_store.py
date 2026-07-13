"""Versioned process-local motivational state for validation scenarios."""

from __future__ import annotations

import asyncio
from copy import deepcopy

from kazusa_ai_chatbot.cognition_core_v2.contracts import (
    EmotionActivation,
    LocalMotivationalState,
    LocalStateKey,
    TransitionProposal,
)


_ENTITY_FIELDS = {
    "drives": "drives",
    "goals": "goals",
    "bonds": "bonds",
    "threats": "threats",
    "standards": "standards",
    "incidents": "incidents",
    "epistemic_state": "epistemic_state",
    "meaning_state": "meaning_state",
}
_CAUSAL_TRANSITIONS = frozenset(("causal_input", "semantic_proposition"))


class StateVersionConflictError(ValueError):
    """Raise when a proposal was based on a stale local-state snapshot."""


class LocalStateStore:
    """Own serial local mutation for independent validation state keys."""

    def __init__(self) -> None:
        self._states: dict[LocalStateKey, LocalMotivationalState] = {}
        self._locks: dict[LocalStateKey, asyncio.Lock] = {}

    async def snapshot(self, state_key: LocalStateKey) -> LocalMotivationalState:
        """Return an immutable-by-copy view of one local state cell.

        Args:
            state_key: Stable validation scope identifying the requested cell.

        Returns:
            A deep copy of the current state, initialized at version zero.
        """

        state = self._states.setdefault(state_key, LocalMotivationalState())
        snapshot_state = deepcopy(state)
        return snapshot_state

    async def apply_proposals(
        self,
        state_key: LocalStateKey,
        proposals: list[TransitionProposal],
    ) -> LocalMotivationalState:
        """Atomically validate and apply a same-snapshot proposal batch.

        Args:
            state_key: Stable validation scope receiving the proposed changes.
            proposals: Mutations generated from one expected state version.

        Returns:
            A copied state after all validated mutations commit together.

        Raises:
            StateVersionConflictError: A proposal does not match the state
                version present at commit time.
            ValueError: A proposal violates the structural or causal contract.
        """

        lock = self._locks.setdefault(state_key, asyncio.Lock())
        async with lock:
            state = self._states.setdefault(state_key, LocalMotivationalState())
            _validate_proposals(proposals, state.state_version)
            updated_state = deepcopy(state)
            for proposal in sorted(
                proposals,
                key=lambda item: (item.entity_kind, item.entity_ref),
            ):
                _apply_proposal(updated_state, proposal)
            if proposals:
                updated_state.state_version += 1
            self._states[state_key] = updated_state
            committed_state = deepcopy(updated_state)
            return committed_state

    async def reset(self, state_key: LocalStateKey | None = None) -> None:
        """Clear one validation scope or all process-local validation state."""

        if state_key is None:
            self._states.clear()
            self._locks.clear()
            return
        self._states.pop(state_key, None)
        self._locks.pop(state_key, None)

    async def set_derived_activations(
        self,
        state_key: LocalStateKey,
        activations: dict[str, EmotionActivation],
    ) -> LocalMotivationalState:
        """Store derived projections without treating emotions as root authority."""

        lock = self._locks.setdefault(state_key, asyncio.Lock())
        async with lock:
            state = self._states.setdefault(state_key, LocalMotivationalState())
            updated_state = deepcopy(state)
            updated_state.emotion_activations = dict(activations)
            self._states[state_key] = updated_state
            projected_state = deepcopy(updated_state)
            return projected_state


def _validate_proposals(
    proposals: list[TransitionProposal],
    current_version: int,
) -> None:
    """Reject stale, malformed, and causally unsupported mutations."""

    for proposal in proposals:
        if proposal.entity_kind not in _ENTITY_FIELDS:
            raise ValueError(f"unsupported state entity kind: {proposal.entity_kind}")
        if not proposal.entity_ref:
            raise ValueError("state entity reference is required")
        if proposal.expected_state_version != current_version:
            raise StateVersionConflictError(
                f"expected version {proposal.expected_state_version}, "
                f"current version is {current_version}"
            )
        if (
            proposal.transition_kind in _CAUSAL_TRANSITIONS
            and not proposal.causal_source_refs
        ):
            raise ValueError("causal transitions require causal source references")
        if any(not source_ref for source_ref in proposal.causal_source_refs):
            raise ValueError("causal source references must be non-empty")
        if not proposal.numeric_delta:
            raise ValueError("transition proposals require at least one numeric delta")
        for field_name, delta in proposal.numeric_delta.items():
            if not field_name or not isinstance(delta, (int, float)):
                raise ValueError("numeric deltas require named numeric values")


def _apply_proposal(
    state: LocalMotivationalState,
    proposal: TransitionProposal,
) -> None:
    """Apply one prevalidated proposal to its package-owned entity mapping."""

    entity_mapping = getattr(state, _ENTITY_FIELDS[proposal.entity_kind])
    if proposal.entity_kind == "meaning_state":
        for field_name, delta in proposal.numeric_delta.items():
            entity_mapping[field_name] = _bounded_delta(
                entity_mapping.get(field_name, 0.0),
                float(delta),
            )
        return
    entity = entity_mapping.setdefault(proposal.entity_ref, {})
    for field_name, delta in proposal.numeric_delta.items():
        entity[field_name] = _bounded_delta(
            entity.get(field_name, 0.0),
            float(delta),
        )


def _bounded_delta(current_value: float, delta: float) -> float:
    """Keep local motivational dimensions within their normalized range."""

    next_value = max(0.0, min(1.0, current_value + delta))
    return next_value
