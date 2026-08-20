"""Deterministic parent-checkpoint guardrail contracts."""

from __future__ import annotations

import asyncio
from typing import cast, get_type_hints

import pytest

from kazusa_ai_chatbot.cognition_core_v2.contracts import (
    CognitionCoreInputV2,
    CognitionCoreOutputV2,
    CognitionCoreServicesV2,
    CognitionExecutionError,
)
from kazusa_ai_chatbot.cognition_core_v2.model_attempt_policy import (
    bind_v2_attempt_ledger,
    create_v2_attempt_ledger,
    reset_v2_attempt_ledger,
    snapshot_v2_attempt_ledger,
    snapshot_v2_guarded_attempt_ledger,
)
from kazusa_ai_chatbot.cognition_core_v3.contracts import (
    CognitionChainServicesV3,
)
from kazusa_ai_chatbot.cognition_resolver import guardrail


def _input_payload(cycle_index: int = 0) -> CognitionCoreInputV2:
    """Build the smallest JSON-shaped payload accepted by the guardrail."""

    return cast(CognitionCoreInputV2, {
        "schema_version": "cognition_core_input.v2",
        "resolver_cycle_index": cycle_index,
        "episode": {"episode_id": "episode-1"},
    })


def _services() -> CognitionCoreServicesV2:
    """Return a type-only service placeholder for patched child calls."""

    return cast(CognitionCoreServicesV2, object())


def _goal_exhaustion(
    *,
    error_code: str = "goal_bid_structure_exhausted",
    branch_id: str = "ordinary_response",
    safe_checkpoint: str = "pre_state_commit",
) -> CognitionExecutionError:
    """Build one bounded child error for eligibility tests."""

    return CognitionExecutionError(
        "goal child exhausted",
        error_code=error_code,
        branch_id=branch_id,
        stage="goal_cognition",
        attempt_count=3,
        safe_checkpoint=safe_checkpoint,
        retryable=False,
    )


def _output() -> CognitionCoreOutputV2:
    """Return an opaque child result for orchestration tests."""

    return cast(CognitionCoreOutputV2, {"result": "ok"})


def test_guardrail_exposes_owned_contract() -> None:
    """The guardrail module exposes the coordinator and terminal error."""

    assert hasattr(guardrail, "CognitionRetryCoordinator")
    assert hasattr(guardrail, "ParentRecoveryExhaustedError")
    assert guardrail.PARENT_RECOVERY_MAX_REPLAYS == 1


def test_coordinator_claim_is_atomic_between_service_and_parent() -> None:
    """The first owner consumes the only invocation replay token."""

    service_first = guardrail.create_cognition_retry_coordinator("service")
    assert service_first.claim_replay("service_graph") is True
    assert service_first.claim_replay("parent_checkpoint") is False

    parent_first = guardrail.create_cognition_retry_coordinator("parent")
    assert parent_first.claim_replay("parent_checkpoint") is True
    assert parent_first.claim_replay("service_graph") is False

    blocked_parent = guardrail.create_cognition_retry_coordinator("blocked")
    assert blocked_parent.claim_replay("service_graph") is True
    assert blocked_parent.claim_parent_checkpoint(
        _goal_exhaustion(),
        checkpoint_sha256="0" * 64,
        cycle_index=0,
    ) is False
    assert blocked_parent.parent_recovery_disposition == (
        "blocked_by_service_retry"
    )
    assert blocked_parent.snapshot()["parent_recovery_disposition"] == (
        "blocked_by_service_retry"
    )


@pytest.mark.asyncio
async def test_parent_epoch_persists_after_recovery() -> None:
    """Later resolver cycles remain in epoch one after parent recovery."""

    ledger = create_v2_attempt_ledger("epoch-persistence")
    ledger_token = bind_v2_attempt_ledger(ledger, graph_attempt=1)
    coordinator = guardrail.create_cognition_retry_coordinator(
        "epoch-persistence",
    )
    coordinator_token = guardrail.bind_cognition_retry_coordinator(coordinator)
    calls: list[int] = []

    async def run_child(
        payload: CognitionCoreInputV2,
        _services: CognitionCoreServicesV2,
    ) -> CognitionCoreOutputV2:
        calls.append(coordinator.epoch)
        if len(calls) == 1:
            raise _goal_exhaustion()
        return _output()

    try:
        await guardrail.run_guarded_cognition(
            _input_payload(),
            _services(),
            run_child=run_child,
        )
        await guardrail.run_guarded_cognition(
            _input_payload(cycle_index=1),
            _services(),
            run_child=run_child,
        )
        assert coordinator.epoch == 1
        assert calls == [0, 1, 1]
    finally:
        guardrail.reset_cognition_retry_coordinator(coordinator_token)
        reset_v2_attempt_ledger(ledger_token)


@pytest.mark.asyncio
async def test_guardrail_passes_engine_neutral_services_and_preserves_v3_attempt_epochs() -> None:
    """The parent checkpoint preserves one V3 service object across epochs."""

    annotations = get_type_hints(guardrail.run_guarded_cognition)
    assert annotations["services"] is guardrail.ServicesT

    ledger = create_v2_attempt_ledger("v3-engine-neutral")
    ledger_token = bind_v2_attempt_ledger(ledger, graph_attempt=1)
    coordinator = guardrail.create_cognition_retry_coordinator(
        "v3-engine-neutral",
    )
    coordinator_token = guardrail.bind_cognition_retry_coordinator(coordinator)
    v3_services = cast(CognitionChainServicesV3, object())
    observed_services: list[CognitionChainServicesV3] = []
    observed_epochs: list[int] = []

    async def run_child(
        _payload: CognitionCoreInputV2,
        services: CognitionChainServicesV3,
    ) -> CognitionCoreOutputV2:
        observed_services.append(services)
        observed_epochs.append(coordinator.epoch)
        if len(observed_services) == 1:
            raise _goal_exhaustion()
        return _output()

    try:
        output = await guardrail.run_guarded_cognition(
            _input_payload(),
            v3_services,
            run_child=run_child,
        )
    finally:
        guardrail.reset_cognition_retry_coordinator(coordinator_token)
        reset_v2_attempt_ledger(ledger_token)

    assert output == _output()
    assert observed_services == [v3_services, v3_services]
    assert observed_epochs == [0, 1]


def test_parent_recovery_allows_only_goal_exhaustion_codes() -> None:
    """Only the two escaped goal exhaustion codes enter parent recovery."""

    assert guardrail.is_parent_recovery_eligible(_goal_exhaustion())
    assert guardrail.is_parent_recovery_eligible(
        _goal_exhaustion(error_code="goal_bid_provider_exhausted")
    )
    assert not guardrail.is_parent_recovery_eligible(
        _goal_exhaustion(error_code="semantic_appraisal_provider_exhausted")
    )
    assert not guardrail.is_parent_recovery_eligible(
        _goal_exhaustion(safe_checkpoint="post_cognition_commit")
    )


def test_parent_recovery_rejects_postcommit_and_unknown_failures() -> None:
    """Post-commit, unknown, and already-guarded errors fail closed."""

    assert not guardrail.is_parent_recovery_eligible(
        CognitionExecutionError(
            "unknown",
            error_code="goal_bid_structure_exhausted",
            safe_checkpoint="post_cognition_commit",
        )
    )
    assert not guardrail.is_parent_recovery_eligible(
        CognitionExecutionError(
            "unknown",
            error_code="internal_invariant",
            safe_checkpoint="pre_state_commit",
        )
    )
    exhausted = guardrail.ParentRecoveryExhaustedError(
        first_error=_goal_exhaustion(),
        second_error=_goal_exhaustion(),
        checkpoint_sha256="0" * 64,
        recovery_epoch=1,
    )
    assert not guardrail.is_parent_recovery_eligible(exhausted)


@pytest.mark.asyncio
async def test_parent_recovery_isolated_between_concurrent_contexts() -> None:
    """Concurrent service contexts keep independent replay ownership."""

    async def run_one(invocation_id: str) -> tuple[str, int]:
        coordinator = guardrail.create_cognition_retry_coordinator(
            invocation_id,
        )
        token = guardrail.bind_cognition_retry_coordinator(coordinator)
        calls = 0

        async def run_child(
            _payload: CognitionCoreInputV2,
            _services: CognitionCoreServicesV2,
        ) -> CognitionCoreOutputV2:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise _goal_exhaustion()
            return _output()

        try:
            await asyncio.sleep(0)
            await guardrail.run_guarded_cognition(
                _input_payload(),
                _services(),
                run_child=run_child,
            )
            return coordinator.cognition_invocation_id, calls
        finally:
            guardrail.reset_cognition_retry_coordinator(token)

    first, second = await asyncio.gather(
        run_one("concurrent-parent-1"),
        run_one("concurrent-parent-2"),
    )

    assert first == ("concurrent-parent-1", 2)
    assert second == ("concurrent-parent-2", 2)


@pytest.mark.asyncio
async def test_parent_recovery_cancellation_restores_context() -> None:
    """Cancellation leaves the caller's coordinator context unchanged."""

    coordinator = guardrail.create_cognition_retry_coordinator("cancel")
    token = guardrail.bind_cognition_retry_coordinator(coordinator)

    async def cancelled_child(
        _payload: CognitionCoreInputV2,
        _services: CognitionCoreServicesV2,
    ) -> CognitionCoreOutputV2:
        raise asyncio.CancelledError()

    try:
        with pytest.raises(asyncio.CancelledError):
            await guardrail.run_guarded_cognition(
                _input_payload(),
                _services(),
                run_child=cancelled_child,
            )
    finally:
        guardrail.reset_cognition_retry_coordinator(token)

    assert guardrail.current_cognition_retry_coordinator() is None


@pytest.mark.asyncio
async def test_parent_recovery_failure_preserves_typed_error() -> None:
    """A failed second child exposes bounded first and second metadata."""

    coordinator = guardrail.create_cognition_retry_coordinator("failure")
    token = guardrail.bind_cognition_retry_coordinator(coordinator)

    async def failing_child(
        _payload: CognitionCoreInputV2,
        _services: CognitionCoreServicesV2,
    ) -> CognitionCoreOutputV2:
        raise _goal_exhaustion(error_code="goal_bid_provider_exhausted")

    try:
        with pytest.raises(
            guardrail.ParentRecoveryExhaustedError,
        ) as raised:
            await guardrail.run_guarded_cognition(
                _input_payload(),
                _services(),
                run_child=failing_child,
            )
    finally:
        guardrail.reset_cognition_retry_coordinator(token)

    error = raised.value
    assert error.error_code == "goal_bid_provider_exhausted"
    assert error.attempt_count == 3
    assert error.safe_checkpoint == "pre_state_commit"
    assert error.retryable is False
    assert error.parent_recovery_attempted is True
    assert error.parent_recovery_disposition == "exhausted"
    assert error.first_error_code == "goal_bid_provider_exhausted"
    assert error.parent_checkpoint_digest
    assert error.recovery_epoch == 1
    assert isinstance(error.__cause__, CognitionExecutionError)


def test_guarded_ledger_has_two_epochs_and_unguarded_snapshot_stays_v1() -> None:
    """Epoch metadata is additive while the inner snapshot keeps V1 shape."""

    ledger = create_v2_attempt_ledger("guarded-ledger")
    token = bind_v2_attempt_ledger(ledger, graph_attempt=1)
    try:
        from kazusa_ai_chatbot.cognition_core_v2.model_attempt_policy import (
            enable_guarded_v2_attempt_ledger,
            set_v2_attempt_epoch,
            set_v2_parent_recovery_metadata,
        )

        enable_guarded_v2_attempt_ledger()
        set_v2_attempt_epoch(0)
        from kazusa_ai_chatbot.cognition_core_v2 import model_attempt_policy

        attempt_zero = model_attempt_policy.reserve_v2_model_attempt(
            stage="goal_bid_structure",
            branch_id="ordinary_response",
            local_attempt=1,
        )
        set_v2_attempt_epoch(1)
        attempt_one = model_attempt_policy.reserve_v2_model_attempt(
            stage="goal_bid_structure",
            branch_id="ordinary_response",
            local_attempt=1,
        )
        set_v2_parent_recovery_metadata(
            disposition="recovered",
            claimed_by="parent_checkpoint",
            epoch=1,
            checkpoint_sha256="1" * 64,
        )
        inner = snapshot_v2_attempt_ledger()
        outer = snapshot_v2_guarded_attempt_ledger()
    finally:
        reset_v2_attempt_ledger(token)

    assert attempt_zero["cumulative_producer_attempt"] == 1
    assert attempt_one["cumulative_producer_attempt"] == 1
    assert inner is not None
    assert inner["schema_version"] == "cognition_attempt_ledger.v1"
    assert len(inner["attempts"]) == 1
    assert outer is not None
    assert outer["schema_version"] == "cognition_attempt_ledger.v2"
    assert [row["epoch"] for row in outer["epochs"]] == [0, 1]
