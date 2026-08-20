"""Deterministic tests for the V3 context estimator and budget ledger."""

from __future__ import annotations

import pytest

from kazusa_ai_chatbot.cognition_core_v3 import budget


def test_budget_ledger_calibration_extension_reanchor_and_overflow() -> None:
    """The ledger enforces normal, one-time extended, re-anchor, and overflow."""

    estimate = budget.estimate_message_tokens(
        ["你好，世界", "ASCII payload"],
        calibration_multiplier=1.0,
    )
    assert estimate >= 1
    assert budget.cjk_codepoint_count("你好，世界") >= 5
    assert budget.cjk_codepoint_count("ASCII payload") == 0

    normal_plan = budget.ContextBudgetPlan(
        serving_window_tokens=50_000,
    )
    normal_ledger = budget.ContextBudgetLedger(normal_plan)
    normal_admission = normal_ledger.admit(
        estimated_prompt_tokens=40_000,
        reserved_completion_tokens=4_096,
    )
    assert normal_admission.extension_available is False
    assert normal_admission.extension_used is False
    assert normal_admission.active_total_ceiling_tokens == 50_000
    assert normal_admission.estimated_total_context_tokens == 44_096

    with pytest.raises(budget.CognitionContextLimitError, match="serving window"):
        normal_ledger.admit(
            estimated_prompt_tokens=46_000,
            reserved_completion_tokens=4_096,
        )

    extended_plan = budget.ContextBudgetPlan(
        serving_window_tokens=70_000,
    )
    extended_ledger = budget.ContextBudgetLedger(extended_plan)
    first = extended_ledger.admit(
        estimated_prompt_tokens=51_000,
        reserved_completion_tokens=4_096,
    )
    assert first.extension_used is True
    assert first.extension_available is True
    assert first.active_total_ceiling_tokens == 65_000

    second = extended_ledger.admit(
        estimated_prompt_tokens=52_000,
        reserved_completion_tokens=4_096,
    )
    assert second.extension_used is True
    assert second.active_total_ceiling_tokens == 65_000

    extended_ledger.consume_reanchor()
    with pytest.raises(budget.CognitionContextLimitError, match="already consumed"):
        extended_ledger.consume_reanchor()

    with pytest.raises(budget.CognitionContextLimitError, match="active total"):
        extended_ledger.admit(
            estimated_prompt_tokens=65_000,
            reserved_completion_tokens=4_096,
        )

    with pytest.raises(budget.CognitionContextLimitError, match="serving window"):
        extended_ledger.admit(
            estimated_prompt_tokens=70_000,
            reserved_completion_tokens=4_096,
        )
