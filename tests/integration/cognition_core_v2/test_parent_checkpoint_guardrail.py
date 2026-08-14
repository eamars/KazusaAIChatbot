"""Patched connector tests for parent-checkpoint Cognition V2 recovery."""

from __future__ import annotations

from copy import deepcopy
from typing import cast
from unittest.mock import AsyncMock

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
    reserve_v2_model_attempt,
    snapshot_v2_guarded_attempt_ledger,
)
from kazusa_ai_chatbot.cognition_resolver import (
    capabilities as resolver_capabilities,
    guardrail,
)
from kazusa_ai_chatbot import service as service_module
from kazusa_ai_chatbot.nodes import persona_supervisor2_cognition as connector
from tests.test_cognition_chain_connector_mapping import (
    NOW,
    _core_output,
    _global_state,
)
from tests.test_task_resolution_orchestrator import _goal_continuation_ref
from kazusa_ai_chatbot.cognition_core_v2.state_models import (
    build_acquaintance_user_state,
    build_character_production_state,
)
from tests.test_service_input_queue import (
    _item,
    _patch_common_dependencies,
    _reset_queue_state,
)


def _goal_error(
    *,
    error_code: str = "goal_bid_structure_exhausted",
) -> CognitionExecutionError:
    """Build the exact escaped goal exhaustion accepted by the guardrail."""

    return CognitionExecutionError(
        "goal bid exhausted",
        error_code=error_code,
        branch_id="ordinary_response",
        stage="goal_cognition",
        attempt_count=3,
        safe_checkpoint="pre_state_commit",
        retryable=False,
    )


def _bound_coordinator() -> tuple[object, object, object]:
    """Bind the ledger and coordinator used by one connector invocation."""

    ledger = create_v2_attempt_ledger("parent-connector")
    ledger_token = bind_v2_attempt_ledger(ledger, graph_attempt=1)
    coordinator = guardrail.create_cognition_retry_coordinator(
        ledger.cognition_invocation_id,
    )
    coordinator_token = guardrail.bind_cognition_retry_coordinator(
        coordinator,
    )
    return ledger, ledger_token, coordinator_token


def _reset_bound_coordinator(
    ledger_token: object,
    coordinator_token: object,
) -> None:
    """Restore the context tokens created by the connector fixture."""

    guardrail.reset_cognition_retry_coordinator(coordinator_token)  # type: ignore[arg-type]
    reset_v2_attempt_ledger(ledger_token)  # type: ignore[arg-type]


def _service_cognition_input() -> CognitionCoreInputV2:
    """Build the minimal canonical payload used by service-loop probes."""

    return cast(CognitionCoreInputV2, {
        "schema_version": "cognition_core_input.v2",
        "resolver_cycle_index": 0,
        "episode": {"episode_id": "service-arbitration"},
    })


async def _run_service_arbitration_case(
    monkeypatch: pytest.MonkeyPatch,
    *,
    service_claims_first: bool,
) -> tuple[object, object]:
    """Run one real service retry loop around a guarded cognition child."""

    await _reset_queue_state()

    class _Graph:
        """Exercise the service loop and the child guardrail together."""

        def __init__(self) -> None:
            self.graph_calls = 0
            self.child_calls = 0
            self.coordinator = None
            self.coordinators: list[object] = []
            self.guarded_snapshot: dict[str, object] | None = None

        async def ainvoke(self, _state: object) -> dict[str, object]:
            self.graph_calls += 1
            self.coordinator = (
                guardrail.current_cognition_retry_coordinator()
            )
            assert self.coordinator is not None
            self.coordinators.append(self.coordinator)
            if service_claims_first and self.graph_calls == 1:
                raise CognitionExecutionError(
                    "service retryable pre-commit failure",
                    error_code="workspace_contract_failed",
                    branch_id="",
                    stage="workspace_collapse",
                    attempt_count=1,
                    safe_checkpoint="pre_state_commit",
                    retryable=True,
                )

            async def run_child(
                _payload: CognitionCoreInputV2,
                _services: CognitionCoreServicesV2,
            ) -> CognitionCoreOutputV2:
                self.child_calls += 1
                reserve_v2_model_attempt(
                    stage="goal_bid_structure",
                    branch_id="ordinary_response",
                    local_attempt=1,
                )
                if self.child_calls == 1:
                    raise _goal_error()
                return _core_output()  # type: ignore[return-value]

            await guardrail.run_guarded_cognition(
                _service_cognition_input(),
                cast(CognitionCoreServicesV2, object()),
                run_child=run_child,
            )
            self.guarded_snapshot = snapshot_v2_guarded_attempt_ledger()
            raise service_module.CognitionExecutionError(
                "later service retry boundary failure",
                error_code="workspace_contract_failed",
                branch_id="",
                stage="workspace_collapse",
                attempt_count=1,
                safe_checkpoint="pre_state_commit",
                retryable=True,
            )

    graph = _Graph()
    _patch_common_dependencies(monkeypatch, graph)
    monkeypatch.setattr(service_module, "_save_assistant_message", AsyncMock())
    monkeypatch.setattr(
        service_module,
        "_settle_runtime_episode_trace",
        AsyncMock(return_value={}),
    )
    monkeypatch.setattr(
        service_module,
        "_persist_post_turn_lifecycle_record",
        AsyncMock(),
    )
    monkeypatch.setattr(
        service_module.guardrail_capsule,
        "_schedule_persistence",
        lambda _document: None,
    )
    for recorder_name in (
        "record_database_operation_event",
        "record_pipeline_turn_event",
        "record_runtime_error_event",
    ):
        monkeypatch.setattr(
            service_module.event_logging,
            recorder_name,
            AsyncMock(),
        )

    item = _item(1, direct_address=True)
    await service_module._process_queued_chat_item(
        item,
        settled_decision={
            "response_action": "proceed",
            "reason_to_respond": "direct character request",
            "use_reply_feature": False,
            "channel_topic": "",
            "indirect_speech_context": "",
        },
        settlement_turn_id="turn-arbitration",
        settlement_version=1,
        settlement_claimed=True,
        prepared_media=[],
        media_prepared=True,
    )
    response = await item.future
    await _reset_queue_state()
    return graph, response


@pytest.mark.asyncio
async def test_parent_retry_reuses_canonical_input_digest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both children receive the same canonical checkpoint digest and copy."""

    state = _global_state()
    state["resolver_state"] = {"cycle_index": 0, "observations": []}
    prewarm = AsyncMock(return_value={
        "answer": "",
        "memory_evidence": [],
        "user_memory_unit_candidates": [],
    })
    monkeypatch.setattr(connector, "run_first_cycle_shared_memory_prewarm", prewarm)
    monkeypatch.setattr(
        connector,
        "get_user_cognition_state",
        AsyncMock(return_value=build_acquaintance_user_state(
            global_user_id="user-1",
            updated_at=NOW,
        )),
    )
    monkeypatch.setattr(
        connector,
        "get_character_cognition_state",
        AsyncMock(return_value=build_character_production_state(updated_at=NOW)),
    )
    child_inputs: list[dict[str, object]] = []
    child_calls = 0

    async def run_child(
        payload: CognitionCoreInputV2,
        _services: CognitionCoreServicesV2,
    ) -> CognitionCoreOutputV2:
        nonlocal child_calls
        child_calls += 1
        child_inputs.append(deepcopy(payload))
        if child_calls == 1:
            payload["evidence"].append({"mutated": True})
            raise _goal_error()
        return _core_output()  # type: ignore[return-value]

    monkeypatch.setattr(connector, "run_cognition", run_child)
    ledger, ledger_token, coordinator_token = _bound_coordinator()
    try:
        update = await connector.call_cognition_subgraph(
            state,
            commit=False,
            retry_coordinator=guardrail.current_cognition_retry_coordinator(),
        )
        coordinator = guardrail.current_cognition_retry_coordinator()
        assert coordinator is not None
        assert coordinator.checkpoint_sha256
        assert child_inputs[0] == child_inputs[1]
        assert child_calls == 2
        assert update["cognition_core_output"] == _core_output()
    finally:
        _reset_bound_coordinator(ledger_token, coordinator_token)
    assert ledger.cognition_invocation_id == "parent-connector"


@pytest.mark.asyncio
async def test_successful_canonical_handoff_repairs_resolver_context_before_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A valid resolver handoff completes before parent replay can claim."""

    state = _global_state()
    state["local_time_context"] = {}
    run_cognition = AsyncMock(return_value=_core_output())
    monkeypatch.setattr(
        connector,
        "get_user_cognition_state",
        AsyncMock(return_value=build_acquaintance_user_state(
            global_user_id="user-1",
            updated_at=NOW,
        )),
    )
    monkeypatch.setattr(
        connector,
        "get_character_cognition_state",
        AsyncMock(return_value=build_character_production_state(
            updated_at=NOW,
        )),
    )
    monkeypatch.setattr(connector, "run_cognition", run_cognition)

    _ledger, ledger_token, coordinator_token = _bound_coordinator()
    coordinator = guardrail.current_cognition_retry_coordinator()
    assert coordinator is not None
    try:
        update = await connector.call_cognition_subgraph(
            state,
            commit=False,
            retry_coordinator=coordinator,
        )
        merged_state = dict(state)
        merged_state.update(update)
        task_context = (
            resolver_capabilities._task_resolution_execution_context_from_state(
                merged_state,
                goal_continuation_ref=_goal_continuation_ref(),
            )
        )
        assert task_context["scene_context"] == (
            update["cognition_scene_context"]
        )
        run_cognition.assert_awaited_once()
        assert coordinator.replay_claimed is False
        assert coordinator.claimed_by is None
        assert coordinator.parent_recovery_attempted is False
    finally:
        _reset_bound_coordinator(ledger_token, coordinator_token)


@pytest.mark.asyncio
async def test_parent_retry_runs_preparation_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Identity, mutable-state, character-state, and prewarm work run once."""

    state = _global_state()
    state["resolver_state"] = {"cycle_index": 0, "observations": []}
    prewarm = AsyncMock(return_value={
        "answer": "",
        "memory_evidence": [],
        "user_memory_unit_candidates": [],
    })
    load_user = AsyncMock(return_value=build_acquaintance_user_state(
        global_user_id="user-1",
        updated_at=NOW,
    ))
    load_character = AsyncMock(return_value=build_character_production_state(
        updated_at=NOW,
    ))
    monkeypatch.setattr(connector, "run_first_cycle_shared_memory_prewarm", prewarm)
    monkeypatch.setattr(connector, "get_user_cognition_state", load_user)
    monkeypatch.setattr(connector, "get_character_cognition_state", load_character)
    child_calls = 0

    async def run_child(
        _payload: CognitionCoreInputV2,
        _services: CognitionCoreServicesV2,
    ) -> CognitionCoreOutputV2:
        nonlocal child_calls
        child_calls += 1
        if child_calls == 1:
            raise _goal_error()
        return _core_output()  # type: ignore[return-value]

    monkeypatch.setattr(connector, "run_cognition", run_child)
    ledger, ledger_token, coordinator_token = _bound_coordinator()
    try:
        await connector.call_cognition_subgraph(
            state,
            commit=False,
            retry_coordinator=guardrail.current_cognition_retry_coordinator(),
        )
    finally:
        _reset_bound_coordinator(ledger_token, coordinator_token)

    assert child_calls == 2
    prewarm.assert_awaited_once_with(state)
    load_user.assert_awaited_once_with("user-1")
    load_character.assert_awaited_once_with()
    assert ledger.epoch == 1


@pytest.mark.asyncio
async def test_parent_retry_uses_independent_input_copies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A child mutation cannot alter the parent checkpoint copy."""

    state = _global_state()
    child_inputs: list[dict[str, object]] = []
    calls = 0

    async def run_child(
        payload: CognitionCoreInputV2,
        _services: CognitionCoreServicesV2,
    ) -> CognitionCoreOutputV2:
        nonlocal calls
        calls += 1
        child_inputs.append(deepcopy(payload))
        if calls == 1:
            payload["resolver_context"] = "child mutation"
            raise _goal_error()
        return _core_output()  # type: ignore[return-value]

    monkeypatch.setattr(connector, "run_cognition", run_child)
    ledger, ledger_token, coordinator_token = _bound_coordinator()
    try:
        await connector.call_cognition_subgraph(
            state,
            commit=False,
            retry_coordinator=guardrail.current_cognition_retry_coordinator(),
        )
    finally:
        _reset_bound_coordinator(ledger_token, coordinator_token)

    assert calls == 2
    assert child_inputs[0] == child_inputs[1]


@pytest.mark.asyncio
async def test_parent_retry_does_not_repeat_capability_or_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Parent recovery repeats only the cognition child."""

    state = _global_state()
    child_calls = 0
    commit = AsyncMock()

    async def run_child(
        _payload: CognitionCoreInputV2,
        _services: CognitionCoreServicesV2,
    ) -> CognitionCoreOutputV2:
        nonlocal child_calls
        child_calls += 1
        if child_calls == 1:
            raise _goal_error()
        return _core_output()  # type: ignore[return-value]

    monkeypatch.setattr(connector, "run_cognition", run_child)
    monkeypatch.setattr(connector, "_commit_cognition_state", commit)
    ledger_token = bind_v2_attempt_ledger(
        create_v2_attempt_ledger("side-effects"),
        graph_attempt=1,
    )
    coordinator = guardrail.create_cognition_retry_coordinator("side-effects")
    coordinator_token = guardrail.bind_cognition_retry_coordinator(coordinator)
    try:
        await connector.call_cognition_subgraph(
            state,
            commit=False,
            retry_coordinator=coordinator,
        )
    finally:
        _reset_bound_coordinator(ledger_token, coordinator_token)

    assert child_calls == 2
    commit.assert_not_awaited()


@pytest.mark.asyncio
async def test_parent_recovery_failure_preserves_typed_error_and_no_side_effect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed parent child remains typed and cannot commit state."""

    state = _global_state()
    commit = AsyncMock()
    monkeypatch.setattr(connector, "_commit_cognition_state", commit)

    async def run_child(
        _payload: CognitionCoreInputV2,
        _services: CognitionCoreServicesV2,
    ) -> CognitionCoreOutputV2:
        raise _goal_error(error_code="goal_bid_provider_exhausted")

    monkeypatch.setattr(connector, "run_cognition", run_child)
    ledger, ledger_token, coordinator_token = _bound_coordinator()
    try:
        with pytest.raises(guardrail.ParentRecoveryExhaustedError) as raised:
            await connector.call_cognition_subgraph(
                state,
                commit=False,
                retry_coordinator=(
                    guardrail.current_cognition_retry_coordinator()
                ),
            )
    finally:
        _reset_bound_coordinator(ledger_token, coordinator_token)

    error = raised.value
    assert error.error_code == "goal_bid_provider_exhausted"
    assert error.retryable is False
    assert error.safe_checkpoint == "pre_state_commit"
    commit.assert_not_awaited()
    assert ledger.epoch == 1


@pytest.mark.asyncio
async def test_parent_epoch_remains_active_on_later_resolver_cycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A later resolver cycle stays in epoch one without a third replay."""

    state = _global_state()
    state["resolver_state"] = {"cycle_index": 0, "observations": []}
    monkeypatch.setattr(
        connector,
        "run_first_cycle_shared_memory_prewarm",
        AsyncMock(return_value={
            "answer": "",
            "memory_evidence": [],
            "user_memory_unit_candidates": [],
        }),
    )
    calls: list[int] = []
    coordinator = guardrail.create_cognition_retry_coordinator("later-cycle")
    ledger = create_v2_attempt_ledger("later-cycle")
    ledger_token = bind_v2_attempt_ledger(ledger, graph_attempt=1)
    coordinator_token = guardrail.bind_cognition_retry_coordinator(coordinator)

    async def run_child(
        _payload: CognitionCoreInputV2,
        _services: CognitionCoreServicesV2,
    ) -> CognitionCoreOutputV2:
        calls.append(coordinator.epoch)
        if len(calls) == 1:
            raise _goal_error()
        return _core_output()  # type: ignore[return-value]

    monkeypatch.setattr(connector, "run_cognition", run_child)
    try:
        await connector.call_cognition_subgraph(
            state,
            commit=False,
            retry_coordinator=coordinator,
        )
        later_state = deepcopy(state)
        later_state["resolver_state"] = {
            "cycle_index": 1,
            "observations": [],
        }
        await connector.call_cognition_subgraph(
            later_state,
            commit=False,
            retry_coordinator=coordinator,
        )
    finally:
        guardrail.reset_cognition_retry_coordinator(coordinator_token)
        reset_v2_attempt_ledger(ledger_token)

    assert calls == [0, 1, 1]
    assert coordinator.epoch == 1


@pytest.mark.asyncio
async def test_parent_first_blocks_later_service_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A parent recovery in the real service loop blocks outer retry."""

    graph, response = await _run_service_arbitration_case(
        monkeypatch,
        service_claims_first=False,
    )

    assert graph.graph_calls == 1
    assert graph.child_calls == 2
    assert graph.coordinator is not None
    assert graph.coordinator.claimed_by == "parent_checkpoint"
    assert graph.coordinator.parent_recovery_disposition == "recovered"
    assert response.operational_error is not None
    assert response.operational_error.error_code == "workspace_contract_failed"


@pytest.mark.asyncio
async def test_service_first_blocks_parent_recovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A real service retry claims the token before the child path."""

    graph, response = await _run_service_arbitration_case(
        monkeypatch,
        service_claims_first=True,
    )

    assert graph.graph_calls == 2
    assert graph.child_calls == 1
    assert graph.coordinator is not None
    assert len(graph.coordinators) == 2
    assert graph.coordinators[0] is graph.coordinators[1]
    assert graph.coordinator.claimed_by == "service_graph"
    assert graph.coordinator.parent_recovery_disposition == (
        "blocked_by_service_retry"
    )
    assert graph.coordinator.parent_recovery_attempted is False
    assert response.operational_error is not None
    assert response.operational_error.error_code == (
        "goal_bid_structure_exhausted"
    )


@pytest.mark.asyncio
async def test_parent_recovery_success_then_service_failure_has_one_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A recovered parent has one child replay and no outer graph replay."""

    graph, _response = await _run_service_arbitration_case(
        monkeypatch,
        service_claims_first=False,
    )

    assert graph.graph_calls == 1
    assert graph.child_calls == 2
    assert graph.coordinator is not None
    assert graph.coordinator.claim_replay("service_graph") is False
    assert graph.coordinator.claim_replay("parent_checkpoint") is False
    assert graph.guarded_snapshot is not None
    assert [
        row["epoch"]
        for row in graph.guarded_snapshot["epochs"]
    ] == [0, 1]
