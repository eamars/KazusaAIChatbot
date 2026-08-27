"""Deterministic ownership test for src/kazusa_ai_chatbot/cognition_resolver/loop.py."""

from __future__ import annotations

from importlib import import_module

import pytest

from kazusa_ai_chatbot.cognition_resolver.contracts import (
    RESOLVER_OBSERVATION_VERSION,
    ResolverValidationError,
)
from kazusa_ai_chatbot.cognition_resolver.loop import call_cognition_resolver_loop
from kazusa_ai_chatbot.cognition_shared.contracts import CognitionExecutionError
from tests.test_cognition_resolver_loop import (
    _cognition_result,
    _resolver_request,
    _resolver_state,
    _task_observation_fields,
)

MODULE_PATH = "kazusa_ai_chatbot.cognition_resolver.loop"
EXPECTED_SYMBOLS = ["call_cognition_resolver_loop"]


def test_loop_exposes_owned_contract() -> None:
    """Keep the module's named owner contract discoverable."""

    module = import_module(MODULE_PATH)
    missing_symbols = [
        symbol
        for symbol in EXPECTED_SYMBOLS
        if not hasattr(module, symbol)
    ]

    assert not missing_symbols, (
        f"{MODULE_PATH} is missing owner symbols: {missing_symbols}"
    )


def test_resolver_surface_provenance_targets_current_user() -> None:
    """Visible resolver fallbacks target the resolved current user exactly."""

    provenance = import_module(MODULE_PATH)._resolver_speak_cognition_provenance({
        "global_user_id": "global-user-123",
    })

    assert provenance == {
        "target_roles": [{
            "role": "target",
            "entity_kind": "user",
            "entity_id": "global-user-123",
        }],
        "evidence_handles": [],
    }


def test_resolver_surface_provenance_requires_current_user() -> None:
    """Missing and blank current-user identity fails closed before L3."""

    for state in ({}, {"global_user_id": ""}, {"global_user_id": "   "}):
        with pytest.raises(ResolverValidationError, match="global_user_id"):
            import_module(MODULE_PATH)._resolver_speak_cognition_provenance(state)


@pytest.mark.asyncio
async def test_cycle_failure_preserves_collected_observations_in_typed_failure(
) -> None:
    request = _resolver_request()
    cognition_calls = 0

    async def call_cognition(state: dict) -> dict:
        nonlocal cognition_calls
        cognition_calls += 1
        if cognition_calls == 1:
            return _cognition_result(
                internal_monologue="The evidence request needs one bounded lookup.",
                resolver_requests=[request],
            )
        raise CognitionExecutionError(
            "cognition stage contract exhausted",
            error_code="cognition_a1_contract_exhausted",
            stage="cognition_core_v3.A1",
            attempt_count=3,
            safe_checkpoint="pre_state_commit",
            retryable=True,
        )

    async def execute_capability(
        capability_request: dict,
        _state: dict,
    ) -> dict:
        return {
            "schema_version": RESOLVER_OBSERVATION_VERSION,
            "observation_id": "resolver_obs_preserved",
            "capability_kind": capability_request["capability_kind"],
            **_task_observation_fields(capability_request),
            "request_objective": capability_request["objective"],
            "request_reason": capability_request["reason"],
            "status": "succeeded",
            "prompt_safe_summary": "The bounded observation was collected.",
            "evidence_refs": [{
                "schema_version": "evidence_ref.v1",
                "evidence_kind": "system_event",
                "evidence_id": "resolver-evidence-1",
                "owner": "cognition_resolver",
                "excerpt": "The bounded observation was collected.",
                "observed_at": "2026-05-29T21:00:00+00:00",
            }],
            "task_resolution_evidence_state": {
                "schema_version": "resolver_evidence_state.v1",
                "state": "complete",
                "remaining_needs": [],
            },
            "created_at_utc": "2026-05-29T21:00:00+00:00",
        }

    with pytest.raises(CognitionExecutionError) as error:
        await call_cognition_resolver_loop(
            _resolver_state(),
            call_cognition_subgraph_func=call_cognition,
            execute_capability_func=execute_capability,
            max_cycles=3,
            capability_timeout_seconds=1.0,
        )

    assert cognition_calls == 2
    assert error.value.error_code == "cognition_a1_contract_exhausted"
    diagnostics = error.value.diagnostics
    assert len(diagnostics["resolver_state"]["observations"]) == 1
    assert diagnostics["resolver_state"]["observations"][0]["observation_id"] == (
        "resolver_obs_preserved"
    )
