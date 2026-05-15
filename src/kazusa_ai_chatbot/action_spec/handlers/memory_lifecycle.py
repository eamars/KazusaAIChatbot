"""Narrow handler helpers for user-memory lifecycle actions."""

from __future__ import annotations

from typing import Any

from kazusa_ai_chatbot.action_spec.models import (
    LIFECYCLE_STATUS_BY_DECISION,
    ActionValidationError,
    validate_action_spec,
)


def map_lifecycle_decision_to_status(decision: str) -> str:
    """Map an LLM-owned lifecycle decision to a collection-native status."""

    status = LIFECYCLE_STATUS_BY_DECISION.get(decision)
    if status is None:
        raise ActionValidationError("lifecycle_decision: unsupported decision")
    return status


def validate_memory_lifecycle_action(
    action_spec: dict[str, Any],
) -> dict[str, Any]:
    """Validate the initial memory-lifecycle action slice."""

    _reject_evolving_memory_targets(action_spec)
    validated = validate_action_spec(action_spec)
    if validated["kind"] != "memory_lifecycle_update":
        raise ActionValidationError("kind: expected memory_lifecycle_update")
    params = validated["params"]
    _validate_memory_params(params)
    target = validated["target"]
    _validate_target(target, params)
    return validated


def build_user_memory_lifecycle_update(
    action_spec: dict[str, Any],
    *,
    timestamp: str,
    action_attempt_id: str,
) -> dict[str, Any]:
    """Build the narrow repository call payload for one lifecycle action."""

    if not timestamp.strip():
        raise ActionValidationError("timestamp: expected non-empty string")
    if not action_attempt_id.strip():
        raise ActionValidationError("action_attempt_id: expected non-empty string")
    validated = validate_memory_lifecycle_action(action_spec)
    params = validated["params"]
    decision = str(params["lifecycle_decision"])
    status = map_lifecycle_decision_to_status(decision)
    update = {
        "unit_id": str(params["unit_id"]),
        "status": status,
        "timestamp": timestamp,
        "reason": str(validated["reason"]),
        "action_attempt_id": action_attempt_id,
        "due_at": params.get("due_at"),
    }
    return update


def _validate_memory_params(params: dict[str, Any]) -> None:
    """Validate params for the user-memory active-commitment lifecycle action."""

    memory_kind = params.get("memory_kind")
    if memory_kind != "user_memory_unit":
        raise ActionValidationError("memory_kind: expected user_memory_unit")
    unit_type = params.get("unit_type")
    if unit_type != "active_commitment":
        raise ActionValidationError("unit_type: expected active_commitment")
    unit_id = params.get("unit_id")
    if not isinstance(unit_id, str) or not unit_id.strip():
        raise ActionValidationError("unit_id: expected non-empty string")
    decision = params.get("lifecycle_decision")
    if decision not in LIFECYCLE_STATUS_BY_DECISION:
        raise ActionValidationError("lifecycle_decision: unsupported decision")
    due_at = params.get("due_at")
    if due_at is not None and not isinstance(due_at, str):
        raise ActionValidationError("due_at: expected string or null")


def _validate_target(target: dict[str, Any], params: dict[str, Any]) -> None:
    """Validate target ownership for active-commitment lifecycle updates."""

    if target["target_kind"] != "memory_unit":
        raise ActionValidationError("target_kind: expected memory_unit")
    if target["owner"] != "user_memory_units":
        raise ActionValidationError("owner: expected user_memory_units")
    if target["target_id"] != params["unit_id"]:
        raise ActionValidationError("target_id: expected params.unit_id")
    scope = target["scope"]
    if scope.get("unit_type") != "active_commitment":
        raise ActionValidationError("scope.unit_type: expected active_commitment")


def _reject_evolving_memory_targets(action_spec: dict[str, Any]) -> None:
    """Reject memory-evolution targets before generic action validation."""

    target = action_spec.get("target")
    if isinstance(target, dict):
        scope = target.get("scope")
        if isinstance(scope, dict) and scope.get("memory_doc_type") == "EvolvingMemoryDoc":
            raise ActionValidationError("EvolvingMemoryDoc targets are unsupported")
    params = action_spec.get("params")
    if isinstance(params, dict) and params.get("memory_kind") == "EvolvingMemoryDoc":
        raise ActionValidationError("EvolvingMemoryDoc targets are unsupported")
