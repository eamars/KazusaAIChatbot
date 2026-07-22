"""Role-ownership guards for real-LLM cognition tests."""

from __future__ import annotations

from collections.abc import Mapping, Sequence


_ROLE_OPERATION_FIELDS = frozenset({
    "response_owner_role",
    "selection_owner_role",
    "embedded_actor_role",
    "embedded_target_role",
})
_ROLE_VALUES = frozenset({
    "当前角色",
    "当前用户",
    "其他参与者",
    "无",
})


def validate_expected_role_bindings(
    value: object,
    *,
    context: str,
) -> list[dict[str, str]]:
    """Validate one fixture's Chinese response-operation expectations."""

    if not isinstance(value, list) or not value:
        raise ValueError(f"{context} must declare role bindings")
    normalized: list[dict[str, str]] = []
    seen_fields: set[str] = set()
    for index, row in enumerate(value):
        if not isinstance(row, Mapping):
            raise ValueError(f"{context}[{index}] is invalid")
        field = row.get("field")
        expected = row.get("expected")
        if field not in _ROLE_OPERATION_FIELDS:
            raise ValueError(f"{context}[{index}] has an invalid field")
        if expected not in _ROLE_VALUES:
            raise ValueError(f"{context}[{index}] has an invalid role")
        if field in seen_fields:
            raise ValueError(f"{context}[{index}] repeats a field")
        seen_fields.add(str(field))
        normalized.append({
            "field": str(field),
            "expected": str(expected),
        })
    return normalized


def evaluate_response_operation_role_bindings(
    turn_calls: Sequence[Mapping[str, object]],
    expected_bindings: object,
    *,
    context: str,
) -> tuple[bool, dict[str, object]]:
    """Compare live decontextualizer role fields with fixture expectations."""

    expected = validate_expected_role_bindings(
        expected_bindings,
        context=context,
    )
    decontextualizer_calls = [
        call
        for call in turn_calls
        if call.get("stage_name") == "message_decontextualizer"
    ]
    details: dict[str, object] = {
        "expected": expected,
        "decontextualizer_call_count": len(decontextualizer_calls),
        "observed": {},
        "mismatches": [],
    }
    if len(decontextualizer_calls) != 1:
        details["error"] = (
            "expected exactly one message_decontextualizer call"
        )
        return False, details

    parsed_output = decontextualizer_calls[0].get("parsed_output")
    if not isinstance(parsed_output, Mapping):
        details["error"] = "decontextualizer parsed output is missing"
        return False, details
    response_operation = parsed_output.get("response_operation")
    if not isinstance(response_operation, Mapping):
        details["error"] = "response_operation is missing"
        return False, details

    observed: dict[str, object] = {}
    mismatches: list[dict[str, object]] = []
    expected_by_field = {
        binding["field"]: binding["expected"]
        for binding in expected
    }
    for field in sorted(_ROLE_OPERATION_FIELDS):
        actual_role = response_operation.get(field)
        observed[field] = actual_role
        if actual_role not in _ROLE_VALUES:
            mismatches.append({
                "field": field,
                "expected": "中文角色值",
                "actual": actual_role,
            })
        expected_role = expected_by_field.get(field)
        if expected_role is not None and actual_role != expected_role:
            mismatches.append({
                "field": field,
                "expected": expected_role,
                "actual": actual_role,
            })
    details["observed"] = observed
    details["mismatches"] = mismatches
    return not mismatches, details
