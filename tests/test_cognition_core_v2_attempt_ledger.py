"""Invocation-wide Cognition Core V2 model-attempt ledger contracts."""

from __future__ import annotations

import asyncio

import pytest

from kazusa_ai_chatbot.cognition_core_v2 import model_attempt_policy


@pytest.mark.parametrize("graph_attempt", [True, 0, -1, "1"])
def test_attempt_ledger_rejects_invalid_graph_attempts(
    graph_attempt: object,
) -> None:
    """Graph-attempt coordinates require a positive non-boolean integer."""

    ledger = model_attempt_policy.create_v2_attempt_ledger("invocation-invalid")

    with pytest.raises(ValueError, match="positive integer"):
        model_attempt_policy.bind_v2_attempt_ledger(
            ledger,
            graph_attempt=graph_attempt,  # type: ignore[arg-type]
        )


def test_goal_attempt_budget_is_monotonic_across_graph_attempts() -> None:
    """A clean graph retry receives only the producer budget left unused."""

    ledger = model_attempt_policy.create_v2_attempt_ledger("invocation-1")

    first_token = model_attempt_policy.bind_v2_attempt_ledger(
        ledger,
        graph_attempt=1,
    )
    try:
        first = model_attempt_policy.reserve_v2_model_attempt(
            stage="goal_bid_structure",
            branch_id="ordinary_response",
            local_attempt=1,
        )
        second = model_attempt_policy.reserve_v2_model_attempt(
            stage="goal_bid_structure",
            branch_id="ordinary_response",
            local_attempt=2,
        )
    finally:
        model_attempt_policy.reset_v2_attempt_ledger(first_token)

    second_token = model_attempt_policy.bind_v2_attempt_ledger(
        ledger,
        graph_attempt=2,
    )
    try:
        third = model_attempt_policy.reserve_v2_model_attempt(
            stage="goal_bid_structure",
            branch_id="ordinary_response",
            local_attempt=1,
        )
        with pytest.raises(model_attempt_policy.V2AttemptBudgetExhausted):
            model_attempt_policy.reserve_v2_model_attempt(
                stage="goal_bid_structure",
                branch_id="ordinary_response",
                local_attempt=2,
            )
    finally:
        model_attempt_policy.reset_v2_attempt_ledger(second_token)

    assert first["graph_attempt"] == 1
    assert first["cumulative_producer_attempt"] == 1
    assert second["cumulative_producer_attempt"] == 2
    assert third == {
        "cognition_invocation_id": "invocation-1",
        "graph_attempt": 2,
        "branch_id": "ordinary_response",
        "producing_stage": "goal_bid_structure",
        "local_attempt": 1,
        "cumulative_producer_attempt": 3,
        "configured_limit": 3,
    }


def test_goal_attempt_budgets_are_independent_by_branch() -> None:
    """Parallel branches share an invocation without sharing call counters."""

    ledger = model_attempt_policy.create_v2_attempt_ledger("invocation-2")
    token = model_attempt_policy.bind_v2_attempt_ledger(
        ledger,
        graph_attempt=1,
    )
    try:
        ordinary = model_attempt_policy.reserve_v2_model_attempt(
            stage="goal_bid_structure",
            branch_id="ordinary_response",
            local_attempt=1,
        )
        autonomy = model_attempt_policy.reserve_v2_model_attempt(
            stage="goal_bid_structure",
            branch_id="autonomy_boundary",
            local_attempt=1,
        )
    finally:
        model_attempt_policy.reset_v2_attempt_ledger(token)

    assert ordinary["cumulative_producer_attempt"] == 1
    assert autonomy["cumulative_producer_attempt"] == 1


def test_attempt_ledger_snapshot_records_terminal_branch_dispositions() -> None:
    """Protected snapshots expose bounded coordinates and terminal outcomes."""

    ledger = model_attempt_policy.create_v2_attempt_ledger("invocation-3")
    token = model_attempt_policy.bind_v2_attempt_ledger(
        ledger,
        graph_attempt=1,
    )
    try:
        attempt = model_attempt_policy.reserve_v2_model_attempt(
            stage="goal_bid_structure",
            branch_id="ordinary_response",
            local_attempt=1,
        )
        model_attempt_policy.record_v2_attempt_disposition(
            attempt,
            disposition="regenerate",
        )
        model_attempt_policy.record_v2_branch_disposition(
            branch_id="ordinary_response",
            disposition="exhausted",
            error_code="goal_bid_structure_exhausted",
        )
        snapshot = model_attempt_policy.snapshot_v2_attempt_ledger()
    finally:
        model_attempt_policy.reset_v2_attempt_ledger(token)

    assert snapshot == {
        "schema_version": "cognition_attempt_ledger.v1",
        "cognition_invocation_id": "invocation-3",
        "attempts": [{
            **attempt,
            "attempt_disposition": "regenerate",
        }],
        "branch_dispositions": [{
            "branch_id": "ordinary_response",
            "disposition": "exhausted",
            "error_code": "goal_bid_structure_exhausted",
        }],
    }


@pytest.mark.asyncio
async def test_attempt_ledger_context_isolated_between_concurrent_calls() -> None:
    """Concurrent cognition invocations cannot consume each other's budget."""

    async def run_one(invocation_id: str) -> dict[str, object]:
        ledger = model_attempt_policy.create_v2_attempt_ledger(invocation_id)
        token = model_attempt_policy.bind_v2_attempt_ledger(
            ledger,
            graph_attempt=1,
        )
        try:
            await asyncio.sleep(0)
            return model_attempt_policy.reserve_v2_model_attempt(
                stage="goal_bid_structure",
                branch_id="ordinary_response",
                local_attempt=1,
            )
        finally:
            model_attempt_policy.reset_v2_attempt_ledger(token)

    first, second = await asyncio.gather(
        run_one("concurrent-1"),
        run_one("concurrent-2"),
    )

    assert first["cognition_invocation_id"] == "concurrent-1"
    assert second["cognition_invocation_id"] == "concurrent-2"
    assert first["cumulative_producer_attempt"] == 1
    assert second["cumulative_producer_attempt"] == 1
