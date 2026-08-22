"""Deterministic ownership test for src/kazusa_ai_chatbot/cognition_core_v2/model_attempt_policy.py."""

from __future__ import annotations

from importlib import import_module

import pytest

from kazusa_ai_chatbot.cognition_core_v2.model_attempt_policy import (
    V2AttemptBudgetExhausted,
    bind_v2_attempt_ledger,
    create_v2_attempt_ledger,
    reserve_v2_model_attempt,
    reserve_v2_model_attempt_batch,
    reset_v2_attempt_ledger,
)

MODULE_PATH = "kazusa_ai_chatbot.cognition_core_v2.model_attempt_policy"
EXPECTED_SYMBOLS = ["create_v2_attempt_ledger"]


def test_model_attempt_policy_exposes_owned_contract() -> None:
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


def test_batch_reservation_is_all_or_none_for_real_branch_roster() -> None:
    """A failed sibling preflight leaves every branch count unchanged."""

    ledger = create_v2_attempt_ledger("batch-test")
    token = bind_v2_attempt_ledger(ledger, graph_attempt=1)
    try:
        reserve_v2_model_attempt(
            stage="goal_bid_structure",
            branch_id="branch-a",
            local_attempt=1,
        )
        reserve_v2_model_attempt(
            stage="goal_bid_structure",
            branch_id="branch-b",
            local_attempt=1,
        )
        reserve_v2_model_attempt(
            stage="goal_bid_structure",
            branch_id="branch-b",
            local_attempt=2,
        )
        reserve_v2_model_attempt(
            stage="goal_bid_structure",
            branch_id="branch-b",
            local_attempt=3,
        )
        before = dict(ledger.producer_attempts)
        with pytest.raises(V2AttemptBudgetExhausted) as error:
            reserve_v2_model_attempt_batch(
                stage="goal_bid_structure",
                branch_ids=("branch-a", "branch-b"),
                local_attempt=2,
            )
        assert error.value.branch_id == "branch-b"
        assert ledger.producer_attempts == before
        assert len(ledger.attempts) == 4
    finally:
        reset_v2_attempt_ledger(token)


def test_batch_reservation_tracks_unequal_cumulative_branch_counts() -> None:
    """One physical group attempt advances each branch independently."""

    ledger = create_v2_attempt_ledger("batch-count-test")
    token = bind_v2_attempt_ledger(ledger, graph_attempt=1)
    try:
        reserve_v2_model_attempt(
            stage="goal_bid_structure",
            branch_id="branch-a",
            local_attempt=1,
        )
        coordinates = reserve_v2_model_attempt_batch(
            stage="goal_bid_structure",
            branch_ids=("branch-a", "branch-b"),
            local_attempt=2,
        )
        assert [
            coordinate["branch_id"] for coordinate in coordinates
        ] == ["branch-a", "branch-b"]
        assert [
            coordinate["cumulative_producer_attempt"]
            for coordinate in coordinates
        ] == [2, 1]
        assert ledger.producer_attempts == {
            ("goal_bid_structure", "branch-a"): 2,
            ("goal_bid_structure", "branch-b"): 1,
        }
    finally:
        reset_v2_attempt_ledger(token)
