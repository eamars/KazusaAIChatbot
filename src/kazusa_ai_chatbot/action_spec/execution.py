"""Shared execution helpers for selected action specs."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from kazusa_ai_chatbot.action_spec.attempt_ledger import (
    build_action_attempt_record,
)
from kazusa_ai_chatbot.action_spec.evaluator import ActionSpecEvaluator
from kazusa_ai_chatbot.action_spec.handlers.future_cognition import (
    execute_future_cognition_action,
)
from kazusa_ai_chatbot.action_spec.handlers.memory_lifecycle import (
    execute_user_memory_lifecycle_action,
)
from kazusa_ai_chatbot.action_spec.models import ActionValidationError
from kazusa_ai_chatbot.action_spec.registry import (
    APPLY_MEMORY_LIFECYCLE_UPDATE_CAPABILITY,
    SPEAK_CAPABILITY,
    TRIGGER_FUTURE_COGNITION_CAPABILITY,
)
from kazusa_ai_chatbot.action_spec.results import (
    ActionResultV1,
    action_attempt_id_from_eval_result,
    build_action_result,
)
from kazusa_ai_chatbot.db import DatabaseOperationError
from kazusa_ai_chatbot.time_boundary import normalize_storage_utc_iso

ActionAttemptRecorder = Callable[[dict[str, Any]], Any]


async def execute_action_specs_for_trace(
    action_specs: list[dict[str, Any]],
    *,
    storage_timestamp_utc: str,
    executed_action_attempt_ids: set[str] | None = None,
    record_attempt_func: ActionAttemptRecorder | None = None,
) -> list[ActionResultV1]:
    """Validate and execute selected actions into auditable trace rows.

    Args:
        action_specs: Materialized action specs selected for the episode.
        storage_timestamp_utc: Episode storage UTC timestamp used for execution
            and completion audit.
        executed_action_attempt_ids: Action attempts already realized by a
            surface handler before this function is called.
        record_attempt_func: Optional existing-ledger writer. When omitted,
            the function remains a deterministic trace builder for tests and
            preview paths.

    Returns:
        Prompt-safe action results for episode trace and consolidation.
    """

    normalized_storage_timestamp_utc = _normalize_storage_timestamp(
        storage_timestamp_utc,
    )
    executed_attempts = executed_action_attempt_ids or set()
    evaluator = ActionSpecEvaluator()
    action_results: list[ActionResultV1] = []
    for action_spec in action_specs:
        eval_result = evaluator.evaluate(action_spec)
        result_summary = ""
        completed_at = None
        execution_result: dict[str, Any] = {}
        action_attempt_id = action_attempt_id_from_eval_result(eval_result)
        validated_spec = eval_result["action_spec"] or action_spec
        if not eval_result["ok"]:
            status = "rejected"
            result_summary = "; ".join(eval_result["errors"])
            execution_result = {
                "status": status,
                "errors": list(eval_result["errors"]),
            }
        elif validated_spec["kind"] == APPLY_MEMORY_LIFECYCLE_UPDATE_CAPABILITY:
            try:
                memory_result = await execute_user_memory_lifecycle_action(
                    validated_spec,
                    storage_timestamp_utc=normalized_storage_timestamp_utc,
                    action_attempt_id=action_attempt_id,
                )
            except ActionValidationError as exc:
                status = "rejected"
                result_summary = (
                    f"{APPLY_MEMORY_LIFECYCLE_UPDATE_CAPABILITY} rejected: {exc}"
                )
                execution_result = {
                    "status": status,
                    "error": str(exc),
                }
            except DatabaseOperationError as exc:
                status = "failed"
                result_summary = (
                    f"{APPLY_MEMORY_LIFECYCLE_UPDATE_CAPABILITY} failed: {exc}"
                )
                execution_result = {
                    "status": status,
                    "error": str(exc),
                }
            except ValueError as exc:
                status = "rejected"
                result_summary = (
                    f"{APPLY_MEMORY_LIFECYCLE_UPDATE_CAPABILITY} rejected: {exc}"
                )
                execution_result = {
                    "status": status,
                    "error": str(exc),
                }
            else:
                if memory_result["status"] in ("executed", "unchanged"):
                    status = "executed"
                    completed_at = normalized_storage_timestamp_utc
                else:
                    status = "failed"
                result_summary = (
                    f"{APPLY_MEMORY_LIFECYCLE_UPDATE_CAPABILITY} "
                    f"{memory_result['status']}: "
                    f"{memory_result['lifecycle_status']}"
                )
                execution_result = {
                    "status": status,
                    "memory_result": memory_result,
                }
        elif validated_spec["kind"] == TRIGGER_FUTURE_COGNITION_CAPABILITY:
            try:
                future_result = await execute_future_cognition_action(
                    validated_spec,
                    storage_timestamp_utc=normalized_storage_timestamp_utc,
                    action_attempt_id=action_attempt_id,
                )
            except ActionValidationError as exc:
                status = "rejected"
                result_summary = f"trigger_future_cognition rejected: {exc}"
                execution_result = {
                    "status": status,
                    "error": str(exc),
                }
            except DatabaseOperationError as exc:
                status = "failed"
                result_summary = f"trigger_future_cognition failed: {exc}"
                execution_result = {
                    "status": status,
                    "error": str(exc),
                }
            else:
                status = "scheduled"
                completed_at = normalized_storage_timestamp_utc
                scheduled_ids = future_result["scheduled_event_ids"]
                result_summary = (
                    "trigger_future_cognition scheduled: "
                    f"{', '.join(scheduled_ids)}"
                )
                execution_result = {
                    "status": status,
                    "scheduled_event_ids": list(scheduled_ids),
                    "future_result": future_result,
                }
        elif action_attempt_id in executed_attempts:
            status = "executed"
            completed_at = normalized_storage_timestamp_utc
            execution_result = {"status": status}
        elif validated_spec["kind"] == SPEAK_CAPABILITY:
            status = "rejected"
            result_summary = "duplicate speak action ignored"
            execution_result = {"status": status}
        else:
            status = "validated"
            execution_result = {"status": status}
        action_result = build_action_result(
            validated_spec,
            eval_result,
            status=status,
            result_summary=result_summary,
            completed_at=completed_at,
        )
        if record_attempt_func is not None:
            await _record_action_attempt(
                record_attempt_func,
                validated_spec,
                eval_result,
                storage_timestamp_utc=normalized_storage_timestamp_utc,
                execution_result=execution_result,
            )
        action_results.append(action_result)
    return action_results


async def _record_action_attempt(
    record_attempt_func: ActionAttemptRecorder,
    action_spec: dict[str, Any],
    eval_result: dict[str, Any],
    *,
    storage_timestamp_utc: str,
    execution_result: dict[str, Any],
) -> None:
    """Record one action attempt through the existing idempotency ledger."""

    attempt_record = build_action_attempt_record(
        action_spec,
        eval_result,
        recorded_at=storage_timestamp_utc,
        execution_result=execution_result,
    )
    result = record_attempt_func(attempt_record)
    if hasattr(result, "__await__"):
        await result


def _normalize_storage_timestamp(storage_timestamp_utc: str) -> str:
    """Normalize an action execution timestamp before trace/audit use."""

    try:
        normalized_storage_timestamp_utc = normalize_storage_utc_iso(
            storage_timestamp_utc,
        )
    except ValueError as exc:
        raise ActionValidationError(
            f"storage_timestamp_utc: invalid storage UTC timestamp: {exc}"
        ) from exc
    return normalized_storage_timestamp_utc
