"""Canonical cognition V2 interaction-contract tests."""

from __future__ import annotations

import pytest


def _plan() -> dict[str, object]:
    return {
        "goal_resolution": "answerable_now",
        "response_goal": "回答当前问题",
        "action_requests": [],
        "resolver_requests": [],
        "epistemic_boundary": "仅基于当前观察回答。",
    }


def test_response_plan_requires_exact_kind_compatible_dsh_decision_only_when_context_exists() -> None:
    from kazusa_ai_chatbot.cognition_core_v3.facade import (
        CanonicalContractError,
        _validate_plan,
    )

    plan = _plan()
    validated = _validate_plan(
        plan,
        self_cognition=False,
        capabilities={"actions": [], "resolvers": []},
        response_plan_contract_variant="fresh_ordinary",
    )
    assert validated.response_goal == "回答当前问题"
    with pytest.raises(CanonicalContractError) as error:
        _validate_plan(
            {**plan, "dsh_interaction_decision": {"decision": "answer"}},
            self_cognition=False,
            capabilities={"actions": [], "resolvers": []},
            response_plan_contract_variant="fresh_ordinary",
        )
    assert str(error.value) == (
        "response plan: unexpected fields ['dsh_interaction_decision']"
    )






