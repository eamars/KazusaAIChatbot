"""Deterministic tests for the V3 fixed appraisal and goal topology."""

from __future__ import annotations

import dataclasses

import pytest

from kazusa_ai_chatbot.cognition_core_v2.state_models import GOAL_KINDS
from kazusa_ai_chatbot.cognition_core_v3 import registry


def test_registry_exposes_exact_appraisal_and_goal_topology():
    first_wave = {chain.name: chain.stages for chain in registry.APPRAISAL_FIRST_WAVE_CHAINS}
    assert first_wave == {
        "causal_normative": ("event_agency", "moral_identity"),
        "relationship": ("relationship_social",),
        "epistemic_meaning": ("epistemic_comparison_memory", "existential_drive"),
    }

    assert registry.TERMINAL_OUTCOME_CHAIN == registry.ChainSpec(
        name="terminal_outcome",
        stages=("goal_threat_outcome",),
    )

    goal_chains = {chain.name: chain.stages for chain in registry.GOAL_CHAINS}
    assert tuple(goal_chains) == GOAL_KINDS
    assert all(stages == (name,) for name, stages in goal_chains.items())

    seen_stages: list[str] = []
    for chain in registry.ALL_CHAINS:
        for stage in chain.stages:
            assert stage not in seen_stages, f"Stage {stage!r} is registered twice"
            seen_stages.append(stage)

    first_wave_names = {chain.name for chain in registry.APPRAISAL_FIRST_WAVE_CHAINS}
    assert "terminal_outcome" not in first_wave_names
    assert registry.WAVE_A_CHAIN_NAMES == (
        "causal_normative",
        "relationship",
        "epistemic_meaning",
        *GOAL_KINDS,
    )
    assert registry.ALL_CHAIN_NAMES[-1] == "terminal_outcome"

    visibility = dict(registry.CHAIN_VISIBILITY)
    assert visibility["causal_normative"] == registry.VISIBILITY_ACCEPTED_PROJECTION
    assert visibility["relationship"] == registry.VISIBILITY_SAME_OWNER_ONLY
    assert visibility["epistemic_meaning"] == registry.VISIBILITY_ACCEPTED_PROJECTION
    assert visibility["terminal_outcome"] == registry.VISIBILITY_FRESH_CANONICAL_PROJECTION
    assert all(visibility[name] == registry.VISIBILITY_ISOLATED_BRANCH for name in GOAL_KINDS)

    spec = registry.ChainSpec(name="probe", stages=("stage_a",))
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.name = "mutated"  # type: ignore[misc]
    with pytest.raises(TypeError):
        registry.APPRAISAL_FIRST_WAVE_CHAINS[0] = spec

    assert registry.PRELIMINARY_GOAL_WAVE == 1
    assert registry.REACTIVATION_GOAL_WAVE == 2
