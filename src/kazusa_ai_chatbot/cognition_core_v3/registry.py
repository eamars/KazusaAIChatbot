"""Immutable appraisal-chain, stage-order, and goal-topology registry for V3.

The registry is a finite import-time-validated declaration of every first-wave
appraisal chain, the separated terminal-outcome chain, one isolated single-stage
chain per registered goal kind, and each chain's visibility class. Execution
modules read it to select chains, order stage attempts, place waves, and record
typed omission diagnostics; it owns no runtime state.
"""

from __future__ import annotations

from dataclasses import dataclass

from kazusa_ai_chatbot.cognition_core_v2.state_models import GOAL_KINDS


@dataclass(frozen=True)
class ChainSpec:
    """One cache-affine semantic chain with its ordered stages."""

    name: str
    stages: tuple[str, ...]


APPRAISAL_FIRST_WAVE_CHAINS: tuple[ChainSpec, ...] = (
    ChainSpec(name="causal_normative", stages=("event_agency", "moral_identity")),
    ChainSpec(name="relationship", stages=("relationship_social",)),
    ChainSpec(
        name="epistemic_meaning",
        stages=("epistemic_comparison_memory", "existential_drive"),
    ),
)

TERMINAL_OUTCOME_CHAIN = ChainSpec(
    name="terminal_outcome",
    stages=("goal_threat_outcome",),
)

GOAL_CHAINS: tuple[ChainSpec, ...] = tuple(
    ChainSpec(name=goal_kind, stages=(goal_kind,)) for goal_kind in GOAL_KINDS
)

ALL_CHAINS: tuple[ChainSpec, ...] = (
    *APPRAISAL_FIRST_WAVE_CHAINS,
    TERMINAL_OUTCOME_CHAIN,
    *GOAL_CHAINS,
)

APPRAISAL_FAMILIES: frozenset[str] = frozenset(
    stage for chain in APPRAISAL_FIRST_WAVE_CHAINS + (TERMINAL_OUTCOME_CHAIN,) for stage in chain.stages
)

WAVE_A_CHAIN_NAMES: tuple[str, ...] = (
    tuple(chain.name for chain in APPRAISAL_FIRST_WAVE_CHAINS) + GOAL_KINDS
)

ALL_CHAIN_NAMES: tuple[str, ...] = WAVE_A_CHAIN_NAMES + (TERMINAL_OUTCOME_CHAIN.name,)

PRELIMINARY_GOAL_WAVE = 1
REACTIVATION_GOAL_WAVE = 2

VISIBILITY_ACCEPTED_PROJECTION = "accepted_projection"
VISIBILITY_SAME_OWNER_ONLY = "same_owner_only"
VISIBILITY_FRESH_CANONICAL_PROJECTION = "fresh_canonical_projection"
VISIBILITY_ISOLATED_BRANCH = "isolated_branch"

_CHAIN_VISIBILITY: dict[str, str] = {
    "causal_normative": VISIBILITY_ACCEPTED_PROJECTION,
    "relationship": VISIBILITY_SAME_OWNER_ONLY,
    "epistemic_meaning": VISIBILITY_ACCEPTED_PROJECTION,
    "terminal_outcome": VISIBILITY_FRESH_CANONICAL_PROJECTION,
}

CHAIN_VISIBILITY: dict[str, str] = {
    **_CHAIN_VISIBILITY,
    **{chain.name: VISIBILITY_ISOLATED_BRANCH for chain in GOAL_CHAINS},
}


def _validate_registry() -> None:
    """Fail startup when the registry violates its own topology invariants."""
    seen_stages: set[str] = set()
    for chain in ALL_CHAINS:
        if not chain.name or len(chain.stages) != len(set(chain.stages)):
            raise ValueError(f"Chain {chain.name!r} declares an invalid stage sequence")
        for stage in chain.stages:
            if stage in seen_stages:
                raise ValueError(
                    f"Stage {stage!r} appears in more than one registered chain"
                )
            seen_stages.add(stage)

    expected_first_wave = ("causal_normative", "relationship", "epistemic_meaning")
    actual_first_wave = tuple(chain.name for chain in APPRAISAL_FIRST_WAVE_CHAINS)
    if actual_first_wave != expected_first_wave:
        raise ValueError("First-wave appraisal chains deviate from the declared order")

    first_wave_names = set(actual_first_wave)
    if TERMINAL_OUTCOME_CHAIN.name in first_wave_names:
        raise ValueError("Terminal outcome must run after provisional reduction, not in the first wave")

    expected_goal_chains = tuple(GOAL_KINDS)
    actual_goal_chains = tuple(chain.name for chain in GOAL_CHAINS)
    if actual_goal_chains != expected_goal_chains:
        raise ValueError("Goal chains deviate from the registered goal-kind topology")
    if any(len(chain.stages) != 1 or chain.stages[0] != chain.name for chain in GOAL_CHAINS):
        raise ValueError("Each goal chain is an isolated single-stage branch")

    expected_families = frozenset(
        {
            "event_agency",
            "moral_identity",
            "relationship_social",
            "epistemic_comparison_memory",
            "existential_drive",
            "goal_threat_outcome",
        }
    )
    if APPRAISAL_FAMILIES != expected_families:
        raise ValueError("Appraisal family set deviates from the six registered families")

    if frozenset(CHAIN_VISIBILITY) != frozenset(ALL_CHAIN_NAMES):
        raise ValueError("Visibility classes must cover exactly the registered chains")


_validate_registry()
