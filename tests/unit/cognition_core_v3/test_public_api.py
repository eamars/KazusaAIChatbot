"""Deterministic tests for the V3 public engine entrypoint exports."""

from __future__ import annotations

import inspect

import kazusa_ai_chatbot.cognition_core_v3 as v3
from kazusa_ai_chatbot.cognition_core_v3 import (
    APPRASAL_CONTRACT_EXHAUSTED_ERROR_CODE,
    BOUNDARY_REJECTED_ERROR_CODE,
    EXHAUSTION_FAILURE_CLASS,
    PROVIDER_FAILURE_CLASS,
    StageFailure,
    StageResult,
    STRUCTURAL_FAILURE_CLASS,
    bind_protected_chain_records,
    reset_protected_chain_records,
    run_cognition,
    snapshot_protected_chain_records,
)
from kazusa_ai_chatbot.cognition_core_v3 import contracts, facade

EXPECTED_EXPORTS = [
    "APPRASAL_CONTRACT_EXHAUSTED_ERROR_CODE",
    "BOUNDARY_REJECTED_ERROR_CODE",
    "EXHAUSTION_FAILURE_CLASS",
    "PROVIDER_FAILURE_CLASS",
    "StageFailure",
    "StageResult",
    "STRUCTURAL_FAILURE_CLASS",
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

    assert APPRASAL_CONTRACT_EXHAUSTED_ERROR_CODE is (
        contracts.APPRASAL_CONTRACT_EXHAUSTED_ERROR_CODE
    )
    assert BOUNDARY_REJECTED_ERROR_CODE is contracts.BOUNDARY_REJECTED_ERROR_CODE
    assert EXHAUSTION_FAILURE_CLASS is contracts.EXHAUSTION_FAILURE_CLASS
    assert PROVIDER_FAILURE_CLASS is contracts.PROVIDER_FAILURE_CLASS
    assert StageFailure is contracts.StageFailure
    assert StageResult is contracts.StageResult
    assert STRUCTURAL_FAILURE_CLASS is contracts.STRUCTURAL_FAILURE_CLASS
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
