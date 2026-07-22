"""Deterministic tests for real-LLM role-ownership guardrails."""

from __future__ import annotations

import pytest

from tests.cognition_core_v2_live_llm_role_guards import (
    evaluate_response_operation_role_bindings,
    validate_expected_role_bindings,
)


def _calls(
    *,
    response_owner_role: str = "当前角色",
    embedded_actor_role: str = "当前角色",
    embedded_target_role: str = "无",
) -> list[dict[str, object]]:
    """Build one captured decontextualizer call for guard tests."""

    return [{
        "stage_name": "message_decontextualizer",
        "parsed_output": {
            "response_operation": {
                "response_owner_role": response_owner_role,
                "selection_owner_role": "无",
                "selection_required": False,
                "embedded_actor_role": embedded_actor_role,
                "embedded_target_role": embedded_target_role,
            },
        },
    }]


def test_role_guard_accepts_expected_character_subject() -> None:
    """A correctly attributed character event passes the guard."""

    passed, details = evaluate_response_operation_role_bindings(
        _calls(),
        [
            {"field": "response_owner_role", "expected": "当前角色"},
            {"field": "embedded_actor_role", "expected": "当前角色"},
        ],
        context="character event",
    )

    assert passed is True
    assert details["mismatches"] == []


def test_role_guard_rejects_user_first_person_as_character_event() -> None:
    """A user-owned first-person event cannot pass a character fixture."""

    passed, details = evaluate_response_operation_role_bindings(
        _calls(embedded_actor_role="当前用户"),
        [{"field": "embedded_actor_role", "expected": "当前角色"}],
        context="character event",
    )

    assert passed is False
    assert details["mismatches"] == [{
        "field": "embedded_actor_role",
        "expected": "当前角色",
        "actual": "当前用户",
    }]


def test_role_guard_rejects_missing_decontextualizer_call() -> None:
    """Missing semantic provenance is a failed guard, not a pass."""

    passed, details = evaluate_response_operation_role_bindings(
        [],
        [{"field": "response_owner_role", "expected": "当前角色"}],
        context="character event",
    )

    assert passed is False
    assert details["decontextualizer_call_count"] == 0
    assert "error" in details


def test_role_guard_rejects_unasserted_english_role_value() -> None:
    """Every live role field must use the Chinese role vocabulary."""

    passed, details = evaluate_response_operation_role_bindings(
        _calls(embedded_target_role="self"),
        [{"field": "response_owner_role", "expected": "当前角色"}],
        context="character event",
    )

    assert passed is False
    assert {
        "field": "embedded_target_role",
        "expected": "中文角色值",
        "actual": "self",
    } in details["mismatches"]


def test_role_fixture_validation_rejects_non_chinese_role_value() -> None:
    """Fixture contracts keep model-facing role values Chinese-only."""

    with pytest.raises(ValueError, match="invalid role"):
        validate_expected_role_bindings(
            [{"field": "embedded_actor_role", "expected": "self"}],
            context="fixture turn",
        )
