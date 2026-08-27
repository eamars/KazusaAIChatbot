"""Target-state checkpoint and bounded-ledger tests."""

from __future__ import annotations

import importlib

import pytest

from tests.test_task_resolution_orchestrator import (
    _goal_continuation_ref,
    _scene_context,
)


def _checkpoint() -> dict[str, object]:
    return {
        "schema_version": "task_resolution_checkpoint.v1",
        "session_id": "session-1",
        "semantic_objective": "Resolve the user's bounded task.",
        "scene_context": _scene_context(),
        "goal_continuation_ref": _goal_continuation_ref(),
        "source_scope": {
            "trigger_source": "user_message",
            "platform": "debug",
            "channel_id": "channel-1",
            "channel_type": "private",
            "source_message_id": "message-1",
            "requester_global_user_id": "user-1",
            "requester_platform_user_id": "debug-user-1",
        },
        "nodes": [
            {
                "schema_version": "task_resolution_node.v1",
                "node_id": "node-1",
                "objective": "Resolve the task.",
                "status": "pending",
                "depends_on": [],
            }
        ],
        "active_node_id": "node-1",
        "evidence": [],
        "remaining_needs": ["Select a compatible specialist."],
        "attempted_specialists": [],
        "dispatch_count": 0,
        "orchestrator_call_count": 0,
        "route_correction_count": 0,
        "specialist_invocation_counts": [],
        "pending_dispatch": None,
        "terminal_status": "",
        "trace_summary": [],
    }


def test_checkpoint_accepts_initial_bounded_state() -> None:
    """A new session starts with one active node and zero consumed budget."""

    contracts = importlib.import_module(
        "kazusa_ai_chatbot.task_resolution.contracts"
    )

    validated = contracts.validate_task_resolution_checkpoint(_checkpoint())

    assert validated["active_node_id"] == "node-1"
    assert validated["dispatch_count"] == 0
    assert validated["orchestrator_call_count"] == 0
    assert validated["pending_dispatch"] is None


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("dispatch_count", 5),
        ("orchestrator_call_count", 5),
        ("route_correction_count", 3),
    ),
)
def test_checkpoint_rejects_exhausted_counter_overflow(
    field: str,
    value: int,
) -> None:
    """Persisted counters can reach, but never exceed, fixed session caps."""

    contracts = importlib.import_module(
        "kazusa_ai_chatbot.task_resolution.contracts"
    )
    checkpoint = _checkpoint()
    checkpoint[field] = value

    with pytest.raises(contracts.TaskResolutionContractError, match=field):
        contracts.validate_task_resolution_checkpoint(checkpoint)


def test_checkpoint_rejects_repeated_attempt_pair() -> None:
    """One node/specialist pair has one durable attempt-ledger row."""

    contracts = importlib.import_module(
        "kazusa_ai_chatbot.task_resolution.contracts"
    )
    checkpoint = _checkpoint()
    checkpoint["attempted_specialists"] = [
        {
            "task_node_id": "node-1",
            "specialist": "local_context",
        },
        {
            "task_node_id": "node-1",
            "specialist": "local_context",
        },
    ]

    with pytest.raises(
        contracts.TaskResolutionContractError,
        match="attempted_specialists",
    ):
        contracts.validate_task_resolution_checkpoint(checkpoint)


def test_checkpoint_rejects_worker_or_adapter_internals() -> None:
    """Checkpoint state contains semantic progress rather than runtime objects."""

    contracts = importlib.import_module(
        "kazusa_ai_chatbot.task_resolution.contracts"
    )
    checkpoint = _checkpoint()
    checkpoint["worker_metadata"] = {"lease_owner": "worker-1"}

    with pytest.raises(
        contracts.TaskResolutionContractError,
        match="worker_metadata",
    ):
        contracts.validate_task_resolution_checkpoint(checkpoint)


def test_checkpoint_rejects_inconsistent_dispatch_ledgers() -> None:
    """Dispatch, invocation, trace, and correction ledgers move atomically."""

    contracts = importlib.import_module(
        "kazusa_ai_chatbot.task_resolution.contracts"
    )
    checkpoint = _checkpoint()
    checkpoint["dispatch_count"] = 1

    with pytest.raises(
        contracts.TaskResolutionContractError,
        match="dispatch_count",
    ):
        contracts.validate_task_resolution_checkpoint(checkpoint)


def test_checkpoint_rejects_orphaned_node_references() -> None:
    """Every persisted evidence and ledger row belongs to a checkpoint node."""

    contracts = importlib.import_module(
        "kazusa_ai_chatbot.task_resolution.contracts"
    )
    checkpoint = _checkpoint()
    checkpoint["evidence"] = [{
        "schema_version": "task_resolution_evidence.v1",
        "evidence_id": "orphan-evidence",
        "task_node_id": "unknown-node",
        "specialist": "public_research",
        "summary": "This row does not belong to a known task node.",
        "provenance_refs": ["https://example.test/source"],
        "limitations": [],
    }]

    with pytest.raises(
        contracts.TaskResolutionContractError,
        match="task_node_id",
    ):
        contracts.validate_task_resolution_checkpoint(checkpoint)


def _specialist_result(
    *,
    specialist: str = "coding",
    status: str = "incompatible",
    evidence: list[dict[str, object]] | None = None,
    retryable: bool = False,
) -> dict[str, object]:
    return {
        "schema_version": "task_specialist_result.v1",
        "specialist": specialist,
        "status": status,
        "evidence": list(evidence or []),
        "completed_subgoals": [],
        "remaining_needs": ["Use a compatible evidence specialist."],
        "reason": "The selected specialist does not own this objective.",
        "retryable": retryable,
    }


def test_incompatible_result_consumes_dispatch_and_route_correction() -> None:
    """A wrong specialist remains recoverable but permanently accounted for."""

    state = importlib.import_module("kazusa_ai_chatbot.task_resolution.state")

    updated = state.record_specialist_result(
        _checkpoint(),
        _specialist_result(),
    )

    assert updated["dispatch_count"] == 1
    assert updated["route_correction_count"] == 1
    assert updated["attempted_specialists"] == [{
        "task_node_id": "node-1",
        "specialist": "coding",
    }]
    assert updated["terminal_status"] == ""


def test_incompatible_pair_cannot_repeat() -> None:
    """The orchestrator cannot bounce back to an already incompatible pair."""

    contracts = importlib.import_module(
        "kazusa_ai_chatbot.task_resolution.contracts"
    )
    state = importlib.import_module("kazusa_ai_chatbot.task_resolution.state")
    once = state.record_specialist_result(
        _checkpoint(),
        _specialist_result(),
    )

    with pytest.raises(
        contracts.TaskResolutionContractError,
        match="attempted_specialists",
    ):
        state.record_specialist_result(once, _specialist_result())


def test_same_specialist_third_invocation_is_rejected() -> None:
    """The persisted per-node specialist cap is two invocations."""

    contracts = importlib.import_module(
        "kazusa_ai_chatbot.task_resolution.contracts"
    )
    state = importlib.import_module("kazusa_ai_chatbot.task_resolution.state")
    checkpoint = _checkpoint()
    checkpoint["dispatch_count"] = 2
    checkpoint["specialist_invocation_counts"] = [{
        "task_node_id": "node-1",
        "specialist": "public_research",
        "count": 2,
    }]
    checkpoint["trace_summary"] = [
        {
            "dispatch_index": index,
            "task_node_id": "node-1",
            "specialist": "public_research",
            "result_status": "temporarily_unavailable",
            "reason": "The provider was temporarily unavailable.",
        }
        for index in (1, 2)
    ]

    with pytest.raises(
        contracts.TaskResolutionContractError,
        match="specialist_invocation_counts",
    ):
        state.record_specialist_result(
            checkpoint,
            _specialist_result(
                specialist="public_research",
                status="temporarily_unavailable",
                retryable=True,
            ),
        )


def test_checkpoint_rejects_non_coding_pending_dispatch_mode() -> None:
    """Only coding dispatches may carry a coding objective mode."""

    contracts = importlib.import_module(
        "kazusa_ai_chatbot.task_resolution.contracts"
    )
    checkpoint = _checkpoint()
    checkpoint["pending_dispatch"] = {
        "schema_version": "task_pending_dispatch.v1",
        "task_node_id": "node-1",
        "specialist": "public_research",
        "subgoal": "Read the public source.",
        "coding_objective_mode": "read_only",
        "phase": "selected",
    }

    with pytest.raises(
        contracts.TaskResolutionContractError,
        match="coding_objective_mode",
    ):
        contracts.validate_task_resolution_checkpoint(checkpoint)


def test_evidence_bearing_partial_creates_dependency_node() -> None:
    """Validated remaining needs continue through a bounded child node."""

    state = importlib.import_module("kazusa_ai_chatbot.task_resolution.state")
    evidence = {
        "schema_version": "task_resolution_evidence.v1",
        "evidence_id": "evidence-1",
        "task_node_id": "node-1",
        "specialist": "public_research",
        "summary": "One public source supports the bounded answer.",
        "provenance_refs": ["https://example.test/source"],
        "limitations": ["A second source remains unavailable."],
    }

    checkpoint = _checkpoint()
    checkpoint["pending_dispatch"] = {
        "schema_version": "task_pending_dispatch.v1",
        "task_node_id": "node-1",
        "specialist": "public_research",
        "subgoal": "Read the first public source.",
        "coding_objective_mode": "none",
        "phase": "started",
    }
    updated = state.record_specialist_result(
        checkpoint,
        _specialist_result(
            specialist="public_research",
            status="partial",
            evidence=[evidence],
        ),
    )

    assert updated["terminal_status"] == ""
    assert updated["evidence"] == [evidence]
    assert updated["pending_dispatch"] is None
    assert updated["active_node_id"] == "node-2"
    assert updated["nodes"][0]["status"] == "resolved"
    assert updated["nodes"][1] == {
        "schema_version": "task_resolution_node.v1",
        "node_id": "node-2",
        "objective": "Use a compatible evidence specialist.",
        "status": "pending",
        "depends_on": ["node-1"],
    }
