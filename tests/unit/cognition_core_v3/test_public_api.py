"""Deterministic tests for the V3 public engine entrypoint exports."""

from __future__ import annotations

import inspect

import kazusa_ai_chatbot.cognition_core_v3 as v3
from kazusa_ai_chatbot.cognition_core_v3 import (
    CANONICAL_A1_FAMILIES,
    CANONICAL_A2_FAMILIES,
    CANONICAL_APPRAISAL_FAMILIES,
    CANONICAL_COGNITION_INPUT_SCHEMA,
    CANONICAL_COGNITION_OUTPUT_SCHEMA,
    CANONICAL_FAMILY_AXES,
    CanonicalAppraisal,
    CanonicalCognitionOutput,
    CanonicalGoal,
    CanonicalResponsePlan,
    CanonicalTurnWorkspace,
    bind_protected_chain_records,
    contracts,
    facade,
    reset_protected_chain_records,
    snapshot_protected_chain_records,
)

EXPECTED_EXPORTS = [
    "CANONICAL_A1_FAMILIES",
    "CANONICAL_A2_FAMILIES",
    "CANONICAL_APPRAISAL_FAMILIES",
    "CANONICAL_COGNITION_INPUT_SCHEMA",
    "CANONICAL_COGNITION_OUTPUT_SCHEMA",
    "CANONICAL_FAMILY_AXES",
    "CanonicalAppraisal",
    "CanonicalCognitionOutput",
    "CanonicalGoal",
    "CanonicalResponsePlan",
    "CanonicalTurnWorkspace",
    "bind_protected_chain_records",
    "reset_protected_chain_records",
    "run_cognition",
    "snapshot_protected_chain_records",
]


def test_v3_exports_exact_engine_entrypoint() -> None:
    """Keep the package export surface pinned to one exact engine contract."""

    assert sorted(v3.__all__) == sorted(EXPECTED_EXPORTS)
    assert v3.run_cognition is facade.run_cognition
    parameters = list(inspect.signature(v3.run_cognition).parameters)
    assert parameters == ["input_payload", "services"]


def test_public_exports_resolve_to_their_owning_module_objects() -> None:
    """Every public export re-exports the owning object, not a shadow copy."""

    assert CANONICAL_A1_FAMILIES is contracts.CANONICAL_A1_FAMILIES
    assert CANONICAL_A2_FAMILIES is contracts.CANONICAL_A2_FAMILIES
    assert CANONICAL_APPRAISAL_FAMILIES is contracts.CANONICAL_APPRAISAL_FAMILIES
    assert CANONICAL_COGNITION_INPUT_SCHEMA == contracts.CANONICAL_COGNITION_INPUT_SCHEMA
    assert CANONICAL_COGNITION_OUTPUT_SCHEMA == contracts.CANONICAL_COGNITION_OUTPUT_SCHEMA
    assert CANONICAL_FAMILY_AXES is contracts.CANONICAL_FAMILY_AXES
    assert CanonicalAppraisal is contracts.CanonicalAppraisal
    assert CanonicalCognitionOutput is contracts.CanonicalCognitionOutput
    assert CanonicalGoal is contracts.CanonicalGoal
    assert CanonicalResponsePlan is contracts.CanonicalResponsePlan
    assert CanonicalTurnWorkspace is contracts.CanonicalTurnWorkspace
    assert bind_protected_chain_records is facade.bind_protected_chain_records
    assert reset_protected_chain_records is (
        facade.reset_protected_chain_records
    )
    assert snapshot_protected_chain_records is (
        facade.snapshot_protected_chain_records
    )


def test_protected_chain_record_scope_api_is_bind_snapshot_reset() -> None:
    """Keep the per-trace replay capture API pinned to its three functions."""

    token = v3.bind_protected_chain_records()
    try:
        assert v3.snapshot_protected_chain_records() == ()
    finally:
        v3.reset_protected_chain_records(token)
    assert inspect.isfunction(v3.bind_protected_chain_records)
    assert inspect.isfunction(v3.snapshot_protected_chain_records)
    assert inspect.isfunction(v3.reset_protected_chain_records)
