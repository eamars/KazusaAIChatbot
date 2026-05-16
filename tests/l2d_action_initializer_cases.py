"""Fixture contract for live L2d action-initializer routing tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from kazusa_ai_chatbot.action_spec.evaluator import ActionSpecEvaluator

L2D_ROUTING_CASE_SET_SCHEMA_VERSION = "l2d_routing_case_set.v1"
L2D_ROUTING_CASE_SCHEMA_VERSION = "l2d_routing_case.v1"
L2D_ROUTING_SOURCE_KINDS = frozenset(("qq_history", "self_cognition"))


def load_l2d_routing_case_set(path: Path) -> dict[str, Any]:
    """Load and validate a frozen L2d routing case file.

    Args:
        path: JSON file containing one case set under the dedicated fixture
            contract.

    Returns:
        Validated fixture document. The function keeps source conversation text
        in the local artifact file and only verifies the fields needed to drive
        a one-case live LLM routing check.
    """

    text = path.read_text(encoding="utf-8")
    document = json.loads(text)
    if not isinstance(document, dict):
        raise ValueError("case set must be a JSON object")
    schema_version = document.get("schema_version")
    if schema_version != L2D_ROUTING_CASE_SET_SCHEMA_VERSION:
        raise ValueError("case set schema_version is not supported")
    cases = document.get("cases")
    if not isinstance(cases, list):
        raise ValueError("cases must be a list")
    for case in cases:
        _validate_case(case)
    return_value = document
    return return_value


def select_l2d_routing_case(
    case_set: dict[str, Any],
    case_id: str,
) -> dict[str, Any]:
    """Return one case by id so live LLM runs stay one-at-a-time.

    Args:
        case_set: Validated case-set document from
            load_l2d_routing_case_set.
        case_id: Stable case identifier selected by the operator.

    Returns:
        The selected case document.
    """

    cases = case_set["cases"]
    for case in cases:
        if case["case_id"] == case_id:
            return_value = case
            return return_value
    raise ValueError(f"case_id not found: {case_id}")


def compare_action_specs_to_expectations(
    case: dict[str, Any],
    action_specs: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compare L2d route output against historical behavior expectations.

    Args:
        case: Frozen fixture with an ``expectations`` object.
        action_specs: Action specs emitted by L2d for the frozen upstream state.

    Returns:
        A JSON-serializable report containing deterministic validation results
        and any routing mismatches. The report intentionally compares action
        route shape, not dialogue wording.
    """

    evaluator = ActionSpecEvaluator()
    evaluator_results = []
    errors: list[str] = []
    observed_kinds: list[str] = []
    observed_visibility_by_kind: dict[str, list[str]] = {}
    observed_params_by_kind: dict[str, list[dict[str, Any]]] = {}

    for index, action_spec in enumerate(action_specs):
        kind = action_spec.get("kind")
        visibility = action_spec.get("visibility")
        params = action_spec.get("params")
        if isinstance(kind, str):
            observed_kinds.append(kind)
        if isinstance(kind, str) and isinstance(params, dict):
            param_values = observed_params_by_kind.setdefault(kind, [])
            param_values.append(params)
        if isinstance(kind, str) and isinstance(visibility, str):
            visibility_values = observed_visibility_by_kind.setdefault(kind, [])
            visibility_values.append(visibility)
        eval_result = evaluator.evaluate(action_spec)
        evaluator_results.append(
            {
                "index": index,
                "ok": eval_result["ok"],
                "kind": kind,
                "handler_owner": eval_result["handler_owner"],
                "errors": eval_result["errors"],
            }
        )
        if not eval_result["ok"]:
            joined_errors = "; ".join(eval_result["errors"])
            errors.append(f"invalid action spec at index {index}: {joined_errors}")

    expectations = case["expectations"]
    _compare_required_action_kinds(expectations, observed_kinds, errors)
    _compare_forbidden_action_kinds(expectations, observed_kinds, errors)
    _compare_required_visibility(
        expectations,
        observed_visibility_by_kind,
        errors,
    )
    _compare_forbidden_user_visible(
        expectations,
        observed_visibility_by_kind,
        errors,
    )
    _compare_required_params(expectations, observed_params_by_kind, errors)

    report = {
        "ok": not errors,
        "errors": errors,
        "observed_kinds": observed_kinds,
        "observed_visibility_by_kind": observed_visibility_by_kind,
        "observed_params_by_kind": observed_params_by_kind,
        "evaluator_results": evaluator_results,
    }
    return report


def _validate_case(value: object) -> None:
    """Validate the minimum fixture shape needed by the live routing test."""

    if not isinstance(value, dict):
        raise ValueError("case must be a JSON object")
    schema_version = value.get("schema_version")
    if schema_version != L2D_ROUTING_CASE_SCHEMA_VERSION:
        raise ValueError("case schema_version is not supported")
    case_id = value.get("case_id")
    if not isinstance(case_id, str) or not case_id:
        raise ValueError("case_id must be a non-empty string")
    source_kind = value.get("source_kind")
    if source_kind not in L2D_ROUTING_SOURCE_KINDS:
        raise ValueError("source_kind is not supported")
    frozen_l2d_state = value.get("frozen_l2d_state")
    if not isinstance(frozen_l2d_state, dict):
        raise ValueError("frozen_l2d_state must be a JSON object")
    historical_comparison = value.get("historical_comparison")
    if not isinstance(historical_comparison, dict):
        raise ValueError("historical_comparison must be a JSON object")
    expectations = value.get("expectations")
    if not isinstance(expectations, dict):
        raise ValueError("expectations must be a JSON object")


def _compare_required_action_kinds(
    expectations: dict[str, Any],
    observed_kinds: list[str],
    errors: list[str],
) -> None:
    """Append errors for required action kinds that were not emitted."""

    required = expectations.get("required_action_kinds")
    if not isinstance(required, list):
        return
    for required_kind in required:
        if not isinstance(required_kind, str):
            continue
        if required_kind not in observed_kinds:
            errors.append(f"missing required action kind: {required_kind}")


def _compare_forbidden_action_kinds(
    expectations: dict[str, Any],
    observed_kinds: list[str],
    errors: list[str],
) -> None:
    """Append errors for action kinds explicitly forbidden for the case."""

    forbidden = expectations.get("forbidden_action_kinds")
    if not isinstance(forbidden, list):
        return
    for forbidden_kind in forbidden:
        if not isinstance(forbidden_kind, str):
            continue
        if forbidden_kind in observed_kinds:
            errors.append(f"forbidden action kind emitted: {forbidden_kind}")


def _compare_required_visibility(
    expectations: dict[str, Any],
    observed_visibility_by_kind: dict[str, list[str]],
    errors: list[str],
) -> None:
    """Append errors when a required action visibility is absent."""

    required = expectations.get("required_visibility_by_kind")
    if not isinstance(required, dict):
        return
    for action_kind, required_visibility in required.items():
        if not isinstance(action_kind, str):
            continue
        if not isinstance(required_visibility, str):
            continue
        observed_visibility = observed_visibility_by_kind.get(action_kind, [])
        if required_visibility not in observed_visibility:
            errors.append(
                f"missing required visibility for {action_kind}: "
                f"{required_visibility}"
            )


def _compare_forbidden_user_visible(
    expectations: dict[str, Any],
    observed_visibility_by_kind: dict[str, list[str]],
    errors: list[str],
) -> None:
    """Append errors for routes that must not become user-visible."""

    forbidden = expectations.get("forbidden_user_visible_kinds")
    if not isinstance(forbidden, list):
        return
    for action_kind in forbidden:
        if not isinstance(action_kind, str):
            continue
        observed_visibility = observed_visibility_by_kind.get(action_kind, [])
        if "user_visible" in observed_visibility:
            errors.append(f"forbidden user-visible action emitted: {action_kind}")


def _compare_required_params(
    expectations: dict[str, Any],
    observed_params_by_kind: dict[str, list[dict[str, Any]]],
    errors: list[str],
) -> None:
    """Append errors when no action of a kind carries required params."""

    required = expectations.get("required_params_by_kind")
    if not isinstance(required, dict):
        return
    for action_kind, required_params in required.items():
        if not isinstance(action_kind, str):
            continue
        if not isinstance(required_params, dict):
            continue
        observed_param_sets = observed_params_by_kind.get(action_kind, [])
        for field_name, expected_value in required_params.items():
            if _param_value_observed(
                observed_param_sets,
                field_name,
                expected_value,
            ):
                continue
            errors.append(
                f"missing required param for {action_kind}: "
                f"{field_name}={expected_value}"
            )


def _param_value_observed(
    observed_param_sets: list[dict[str, Any]],
    field_name: object,
    expected_value: object,
) -> bool:
    """Return whether any action params contain the required field value."""

    if not isinstance(field_name, str):
        return_value = False
        return return_value
    for observed_params in observed_param_sets:
        if observed_params.get(field_name) == expected_value:
            return_value = True
            return return_value
    return_value = False
    return return_value
