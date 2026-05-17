"""Narrow handler helpers for user-memory lifecycle actions."""

from __future__ import annotations

from typing import Any

from kazusa_ai_chatbot.action_spec.models import (
    LIFECYCLE_STATUS_BY_DECISION,
    ActionValidationError,
    validate_action_spec,
)
from kazusa_ai_chatbot.db.user_memory_units import (
    update_user_memory_unit_lifecycle,
)
from kazusa_ai_chatbot.time_boundary import normalize_storage_utc_iso


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
    _validate_source_refs(validated["source_refs"], params)
    return validated


def build_user_memory_lifecycle_update(
    action_spec: dict[str, Any],
    *,
    storage_timestamp_utc: str,
    action_attempt_id: str,
) -> dict[str, Any]:
    """Build the narrow repository call payload for one lifecycle action."""

    normalized_storage_timestamp_utc = _normalize_storage_timestamp(
        storage_timestamp_utc,
    )
    if not action_attempt_id.strip():
        raise ActionValidationError("action_attempt_id: expected non-empty string")
    validated = validate_memory_lifecycle_action(action_spec)
    params = validated["params"]
    decision = str(params["lifecycle_decision"])
    status = map_lifecycle_decision_to_status(decision)
    update = {
        "unit_id": str(params["unit_id"]),
        "status": status,
        "timestamp": normalized_storage_timestamp_utc,
        "reason": str(validated["reason"]),
        "action_attempt_id": action_attempt_id,
        "due_at": params.get("due_at"),
    }
    return update


async def execute_user_memory_lifecycle_action(
    action_spec: dict[str, Any],
    *,
    storage_timestamp_utc: str,
    action_attempt_id: str,
) -> dict[str, Any]:
    """Execute a validated memory lifecycle action through its DB owner."""

    update = build_user_memory_lifecycle_update(
        action_spec,
        storage_timestamp_utc=storage_timestamp_utc,
        action_attempt_id=action_attempt_id,
    )
    result = await update_user_memory_unit_lifecycle(
        update["unit_id"],
        status=update["status"],
        storage_timestamp_utc=update["timestamp"],
        reason=update["reason"],
        action_attempt_id=update["action_attempt_id"],
        due_at=update["due_at"],
    )
    if result["modified_count"]:
        status = "executed"
    elif result["matched_count"]:
        status = "unchanged"
    else:
        status = "not_found"
    return_value = {
        "status": status,
        "unit_id": result["unit_id"],
        "lifecycle_status": result["status"],
        "matched_count": result["matched_count"],
        "modified_count": result["modified_count"],
        "merge_history_entry": result["merge_history_entry"],
    }
    return return_value


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


def _validate_source_refs(
    source_refs: list[dict[str, Any]],
    params: dict[str, Any],
) -> None:
    """Require a matching memory-unit source reference for audit lineage."""

    unit_id = params["unit_id"]
    for source_ref in source_refs:
        if source_ref.get("ref_kind") != "memory_unit":
            continue
        if source_ref.get("owner") != "user_memory_units":
            continue
        if source_ref.get("ref_id") == unit_id:
            return
    raise ActionValidationError("source_refs: expected target memory_unit ref")


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


def _normalize_storage_timestamp(storage_timestamp_utc: str) -> str:
    """Normalize an action audit timestamp before persistence."""

    try:
        normalized_storage_timestamp_utc = normalize_storage_utc_iso(
            storage_timestamp_utc,
        )
    except ValueError as exc:
        raise ActionValidationError(
            f"storage_timestamp_utc: invalid storage UTC timestamp: {exc}"
        ) from exc
    return normalized_storage_timestamp_utc
