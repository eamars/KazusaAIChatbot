"""Deterministic evaluator for modality-neutral action specs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from kazusa_ai_chatbot.action_spec.attempt_ledger import (
    build_action_idempotency_key,
)
from kazusa_ai_chatbot.action_spec.handlers.memory_lifecycle import (
    validate_memory_lifecycle_action,
)
from kazusa_ai_chatbot.action_spec.handlers.future_cognition import (
    validate_future_cognition_action,
)
from kazusa_ai_chatbot.action_spec.handlers.background_artifact import (
    validate_background_artifact_action,
)
from kazusa_ai_chatbot.action_spec.handlers.accepted_task import (
    validate_accepted_task_status_check_action,
)
from kazusa_ai_chatbot.action_spec.handlers.background_work import (
    validate_background_work_action,
    validate_future_speak_action,
)
from kazusa_ai_chatbot.action_spec.models import (
    ActionEvalResult,
    ActionValidationError,
    CapabilitySpecV1,
    validate_action_spec,
    validate_capability_spec,
)
from kazusa_ai_chatbot.action_spec.registry import (
    ACCEPTED_TASK_STATUS_CHECK_CAPABILITY,
    APPLY_MEMORY_LIFECYCLE_UPDATE_CAPABILITY,
    BACKGROUND_ARTIFACT_REQUEST_CAPABILITY,
    BACKGROUND_WORK_REQUEST_CAPABILITY,
    FUTURE_SPEAK_CAPABILITY,
    MEMORY_LIFECYCLE_UPDATE_CAPABILITY,
    SPEAK_CAPABILITY,
    TRIGGER_FUTURE_COGNITION_CAPABILITY,
    build_initial_action_capabilities,
)

_TYPE_MAP = {
    "array": list,
    "boolean": bool,
    "integer": int,
    "number": (int, float),
    "object": dict,
    "string": str,
}


class ActionSpecEvaluator:
    """Shared deterministic gate for cognition-authored action specs."""

    def __init__(
        self,
        capabilities: Mapping[str, CapabilitySpecV1] | None = None,
    ) -> None:
        if capabilities is None:
            capabilities = build_initial_action_capabilities()
        self._capabilities = dict(capabilities)

    def evaluate(self, action_spec: dict[str, Any]) -> ActionEvalResult:
        """Validate one action spec and resolve its execution owner."""

        try:
            validated = validate_action_spec(action_spec)
        except ActionValidationError as exc:
            return_value = _rejected([str(exc)])
            return return_value

        capability = self._capabilities.get(validated["kind"])
        if capability is None:
            error = f"kind: unsupported capability {validated['kind']}"
            return_value = _rejected([error], validated)
            return return_value

        try:
            validate_capability_spec(capability)
            _validate_params(validated["params"], capability["input_schema"])
            _validate_kind_specific_contract(validated)
        except ActionValidationError as exc:
            return_value = _rejected([str(exc)], validated, capability)
            return return_value

        idempotency_key = build_action_idempotency_key(validated)
        result: ActionEvalResult = {
            "ok": True,
            "action_spec": validated,
            "capability": capability,
            "idempotency_key": idempotency_key,
            "handler_owner": capability["owner_module"],
            "errors": [],
        }
        return result


def _rejected(
    errors: list[str],
    action_spec: dict[str, Any] | None = None,
    capability: CapabilitySpecV1 | None = None,
) -> ActionEvalResult:
    """Build a rejected action-evaluation result."""

    handler_owner = None
    if capability is not None:
        handler_owner = capability["owner_module"]
    result: ActionEvalResult = {
        "ok": False,
        "action_spec": action_spec,
        "capability": capability,
        "idempotency_key": None,
        "handler_owner": handler_owner,
        "errors": errors,
    }
    return result


def _validate_kind_specific_contract(action_spec: dict[str, Any]) -> None:
    """Run owner-specific validation without executing handlers."""

    kind = action_spec["kind"]
    if kind == MEMORY_LIFECYCLE_UPDATE_CAPABILITY:
        _validate_memory_lifecycle_route_contract(action_spec)
    elif kind == APPLY_MEMORY_LIFECYCLE_UPDATE_CAPABILITY:
        validate_memory_lifecycle_action(action_spec)
    elif kind == SPEAK_CAPABILITY:
        _validate_speak_contract(action_spec)
    elif kind == TRIGGER_FUTURE_COGNITION_CAPABILITY:
        validate_future_cognition_action(action_spec)
    elif kind == BACKGROUND_ARTIFACT_REQUEST_CAPABILITY:
        validate_background_artifact_action(action_spec)
    elif kind == BACKGROUND_WORK_REQUEST_CAPABILITY:
        validate_background_work_action(action_spec)
    elif kind == FUTURE_SPEAK_CAPABILITY:
        validate_future_speak_action(action_spec)
    elif kind == ACCEPTED_TASK_STATUS_CHECK_CAPABILITY:
        validate_accepted_task_status_check_action(action_spec)


def _validate_speak_contract(action_spec: dict[str, Any]) -> None:
    """Validate the text-surface action target without dispatching it."""

    target = action_spec["target"]
    owner = target["owner"]
    if owner != "l3_text":
        raise ActionValidationError("owner: expected l3_text")
    target_kind = target["target_kind"]
    if target_kind not in ("current_channel", "self"):
        raise ActionValidationError("target_kind: expected text surface target")


def _validate_memory_lifecycle_route_contract(action_spec: dict[str, Any]) -> None:
    """Validate the specialist route intent without DB execution fields."""

    for source_ref in action_spec["source_refs"]:
        if source_ref["ref_kind"] == "memory_unit":
            raise ActionValidationError(
                "source_refs: route must not bind memory_unit"
            )
        if source_ref["owner"] == "user_memory_units":
            raise ActionValidationError(
                "source_refs: route must not bind user_memory_units"
            )
    target = action_spec["target"]
    if target["target_kind"] != "cognitive_episode":
        raise ActionValidationError("target_kind: expected cognitive_episode")
    if target["target_id"] is not None:
        raise ActionValidationError("target_id: expected null")
    if target["owner"] != "memory_lifecycle_specialist":
        raise ActionValidationError("owner: expected memory_lifecycle_specialist")
    scope = target["scope"]
    if scope.get("unit_type") != "active_commitment":
        raise ActionValidationError("scope.unit_type: expected active_commitment")
    params = action_spec["params"]
    if params.get("review_kind") != "active_commitment_lifecycle":
        raise ActionValidationError(
            "review_kind: expected active_commitment_lifecycle"
        )


def _validate_params(params: dict[str, Any], schema: dict[str, object]) -> None:
    """Validate params against the registry's JSON-schema subset."""

    if not isinstance(params, dict):
        raise ActionValidationError("params: expected object")
    required = schema.get("required")
    if isinstance(required, list):
        for field_name in required:
            if field_name not in params:
                raise ActionValidationError(f"{field_name}: missing required field")
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        return
    for field_name, value in params.items():
        field_schema = properties.get(field_name)
        if isinstance(field_schema, dict):
            _validate_param_value(field_name, value, field_schema)


def _validate_param_value(
    field_name: str,
    value: object,
    schema: dict[str, object],
) -> None:
    """Validate one param value against the supported schema subset."""

    allowed_types = _normalize_schema_types(schema.get("type"))
    if "null" in allowed_types and value is None:
        return
    python_types = tuple(
        _TYPE_MAP[type_name]
        for type_name in allowed_types
        if type_name != "null" and type_name in _TYPE_MAP
    )
    if python_types and not isinstance(value, python_types):
        expected = ", ".join(allowed_types)
        raise ActionValidationError(f"{field_name}: expected {expected}")
    enum_values = schema.get("enum")
    if isinstance(enum_values, list) and value not in enum_values:
        raise ActionValidationError(f"{field_name}: expected one of {enum_values}")


def _normalize_schema_types(schema_type: object) -> tuple[str, ...]:
    """Return a tuple of schema type names."""

    if isinstance(schema_type, list):
        return_value = tuple(
            type_name
            for type_name in schema_type
            if isinstance(type_name, str)
        )
        return return_value
    if isinstance(schema_type, str):
        return_value = (schema_type,)
        return return_value
    return_value = ()
    return return_value
