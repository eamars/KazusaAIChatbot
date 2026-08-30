"""Shared execution helpers for selected action specs."""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping
from typing import Any

from kazusa_ai_chatbot.action_spec.attempt_ledger import (
    build_action_attempt_record,
)
from kazusa_ai_chatbot.action_spec.evaluator import ActionSpecEvaluator
from kazusa_ai_chatbot.action_spec.handlers.accepted_task import (
    execute_accepted_task_status_check_action,
)
from kazusa_ai_chatbot.action_spec.handlers.background_work import (
    BackgroundWorkEnqueueFunc,
    enqueue_future_speak_action,
)
from kazusa_ai_chatbot.action_spec.handlers.future_cognition import (
    execute_future_cognition_action,
)
from kazusa_ai_chatbot.action_spec.handlers.memory_lifecycle import (
    execute_user_memory_lifecycle_action,
)
from kazusa_ai_chatbot.action_spec.models import (
    ActionAvailabilityContextV1,
    ActionValidationError,
    RuntimeCapabilitySnapshotV1,
)
from kazusa_ai_chatbot.action_spec.registry import (
    ACCEPTED_TASK_STATUS_CHECK_CAPABILITY,
    APPLY_MEMORY_LIFECYCLE_UPDATE_CAPABILITY,
    FUTURE_SPEAK_CAPABILITY,
    SPEAK_CAPABILITY,
    TRIGGER_FUTURE_COGNITION_CAPABILITY,
    build_runtime_capability_snapshot,
    recheck_action_affordance,
)
from kazusa_ai_chatbot.task_resolution.contracts import (
    validate_accepted_task_control,
)


async def execute_accepted_task_control(
    *,
    control: Mapping[str, object],
    affordance: Mapping[str, object] | None = None,
    action_attempt_id: str,
    lifecycle: object,
    advertised_refs: Collection[str] | None = None,
) -> dict[str, object]:
    """Execute one model-selected control against an advertised affordance."""

    validated = validate_accepted_task_control(control)
    affordance_ref = (
        affordance.get("accepted_task_ref")
        if affordance is not None
        else validated["accepted_task_ref"]
    )
    if advertised_refs is not None and validated["accepted_task_ref"] not in {
        str(value) for value in advertised_refs
    }:
        raise ActionValidationError("accepted task control ref is not advertised")
    if validated["accepted_task_ref"] != affordance_ref:
        raise ActionValidationError("accepted task control ref is not advertised")
    effective_affordance = dict(affordance or {
        "accepted_task_ref": validated["accepted_task_ref"],
        "allowed_next_actions": ["continue", "summarize", "cancel"],
    })
    allowed = effective_affordance.get("allowed_next_actions")
    if not isinstance(allowed, list) or validated["operation"] not in allowed:
        raise ActionValidationError("accepted task control operation is unavailable")
    method = lifecycle if callable(lifecycle) else getattr(
        lifecycle,
        "apply_control",
        None,
    )
    if not callable(method):
        method = getattr(lifecycle, "execute", None)
    if not callable(method):
        raise ActionValidationError("accepted task lifecycle is unavailable")
    value = method(
        control=dict(validated),
        affordance=effective_affordance,
        action_attempt_id=action_attempt_id,
        accepted_task_ref=validated["accepted_task_ref"],
        operation=validated["operation"],
        instruction=validated["instruction"],
    )
    if hasattr(value, "__await__"):
        value = await value
    if isinstance(value, Mapping):
        return dict(value)
    return {
        "status": "queued" if validated["operation"] == "continue" else "claimed",
        "accepted_task_ref": validated["accepted_task_ref"],
        "operation": validated["operation"],
    }
from kazusa_ai_chatbot.action_spec.results import (
    ACTION_RESULT_VERSION,
    DEFAULT_ACTION_CONTINUATION,
    ActionResultV1,
    action_attempt_id_from_eval_result,
    build_action_result,
    project_trace_action_result_v2,
)
from kazusa_ai_chatbot.db import DatabaseOperationError
from kazusa_ai_chatbot.time_boundary import normalize_storage_utc_iso

ActionAttemptRecorder = Callable[[dict[str, Any]], Any]
AvailabilitySnapshotFactory = Callable[
    [ActionAvailabilityContextV1],
    RuntimeCapabilitySnapshotV1,
]


async def execute_action_specs_for_trace(
    action_specs: list[dict[str, Any]],
    *,
    storage_timestamp_utc: str,
    executed_action_attempt_ids: set[str] | None = None,
    record_attempt_func: ActionAttemptRecorder | None = None,
    enqueue_background_work_func: BackgroundWorkEnqueueFunc | None = None,
    availability_snapshot_factory: AvailabilitySnapshotFactory | None = None,
    source_llm_trace_id: str = "",
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
        enqueue_background_work_func: Optional queue helper seam for generic
            background-work requests.
        availability_snapshot_factory: Optional fresh runtime snapshot factory.
        source_llm_trace_id: Protected trace owned by the state that selected
            these actions. Empty values are retained for deterministic preview
            paths that do not persist live companion rows.

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
        result_refs = None
        prompt_result_fields: dict[str, Any] = {}
        action_attempt_id = action_attempt_id_from_eval_result(eval_result)
        validated_spec = eval_result["action_spec"] or action_spec
        availability_result = None
        if eval_result["ok"]:
            availability_context = _action_availability_context(validated_spec)
            if availability_snapshot_factory is None:
                fresh_snapshot = build_runtime_capability_snapshot()
            else:
                fresh_snapshot = availability_snapshot_factory(
                    availability_context,
                )
            availability_result = await recheck_action_affordance(
                validated_spec["kind"],
                availability_context,
                fresh_snapshot,
            )
        if not eval_result["ok"]:
            status = "rejected"
            result_summary = "; ".join(eval_result["errors"])
            execution_result = {
                "status": status,
                "errors": list(eval_result["errors"]),
            }
        elif (
            availability_result is not None
            and availability_result["status"] == "unavailable"
        ):
            status = "rejected"
            result_summary = (
                f"{validated_spec['kind']} unavailable: "
                f"{availability_result['reason_code']}"
            )
            execution_result = {
                "status": status,
                "availability": dict(availability_result),
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
                future_kwargs: dict[str, Any] = {
                    "storage_timestamp_utc": normalized_storage_timestamp_utc,
                    "action_attempt_id": action_attempt_id,
                }
                if source_llm_trace_id.strip():
                    future_kwargs["source_llm_trace_id"] = (
                        source_llm_trace_id.strip()
                    )
                future_result = await execute_future_cognition_action(
                    validated_spec,
                    **future_kwargs,
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
                scheduled_count = int(future_result["scheduled_count"])
                trigger_kind = str(future_result["calendar_trigger_kind"])
                result_summary = (
                    "scheduled self-cognition follow-up: "
                    f"scheduled_count={scheduled_count}"
                )
                execution_result = {
                    "status": status,
                    "calendar_trigger_kind": trigger_kind,
                    "calendar_schedule_id": (
                        future_result["calendar_schedule_id"]
                    ),
                    "calendar_run_id": future_result["calendar_run_id"],
                    "scheduled_count": scheduled_count,
                    "future_result": future_result,
                }
        elif validated_spec["kind"] == FUTURE_SPEAK_CAPABILITY:
            try:
                queue_kwargs: dict[str, Any] = {
                    "storage_timestamp_utc": normalized_storage_timestamp_utc,
                    "action_attempt_id": action_attempt_id,
                    "enqueue_background_work_func": enqueue_background_work_func,
                }
                if source_llm_trace_id.strip():
                    queue_kwargs["source_llm_trace_id"] = (
                        source_llm_trace_id.strip()
                    )
                params = validated_spec.get("params")
                if isinstance(params, Mapping):
                    proposal = params.get("scheduled_authority_proposal")
                    if isinstance(proposal, Mapping):
                        queue_kwargs["scheduled_authority_proposal"] = dict(
                            proposal
                        )
                queue_result = await enqueue_future_speak_action(
                    validated_spec,
                    **queue_kwargs,
                )
            except ActionValidationError as exc:
                status = "rejected"
                result_summary = f"future_speak rejected: {exc}"
                execution_result = {
                    "status": status,
                    "error": str(exc),
                }
            except DatabaseOperationError as exc:
                status = "failed"
                result_summary = f"future_speak failed: {exc}"
                execution_result = {
                    "status": status,
                    "error": str(exc),
                }
            except ValueError as exc:
                status = "rejected"
                result_summary = f"future_speak rejected: {exc}"
                execution_result = {
                    "status": status,
                    "error": str(exc),
                }
            else:
                status = _accepted_task_execution_status(queue_result)
                result_summary = queue_result["result_summary"]
                execution_result = {
                    "status": status,
                    "accepted_task_state": (
                        queue_result["accepted_task_state"]
                    ),
                    "accepted_task_summary": (
                        queue_result["accepted_task_summary"]
                    ),
                    "acknowledgement_constraint": (
                        queue_result["acknowledgement_constraint"]
                    ),
                    "wait_guidance": queue_result["wait_guidance"],
                }
                prompt_result_fields = {
                    "accepted_task_state": (
                        queue_result["accepted_task_state"]
                    ),
                    "accepted_task_summary": (
                        queue_result["accepted_task_summary"]
                    ),
                    "acknowledgement_constraint": (
                        queue_result["acknowledgement_constraint"]
                    ),
                    "wait_guidance": queue_result["wait_guidance"],
                }
        elif validated_spec["kind"] == ACCEPTED_TASK_STATUS_CHECK_CAPABILITY:
            try:
                status_result = await execute_accepted_task_status_check_action(
                    validated_spec,
                )
            except ActionValidationError as exc:
                status = "rejected"
                result_summary = f"accepted_task_status_check rejected: {exc}"
                execution_result = {
                    "status": status,
                    "error": str(exc),
                }
            except DatabaseOperationError as exc:
                status = "failed"
                result_summary = f"accepted_task_status_check failed: {exc}"
                execution_result = {
                    "status": status,
                    "error": str(exc),
                }
            else:
                status = "executed"
                completed_at = normalized_storage_timestamp_utc
                prompt_result_fields = _accepted_task_status_prompt_fields(
                    status_result,
                )
                result_summary = prompt_result_fields.pop("result_summary")
                execution_result = {
                    "status": status,
                    **prompt_result_fields,
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
        try:
            action_result = build_action_result(
                validated_spec,
                eval_result,
                status=status,
                result_summary=result_summary,
                result_refs=result_refs,
                completed_at=completed_at,
            )
        except ValueError:
            # Result materialization fails closed when the spec's surface
            # metadata is missing or invalid; return a rejected row instead
            # of crashing the episode trace.
            action_result = _rejected_action_result_fail_closed(
                validated_spec,
                eval_result,
                result_summary=result_summary,
            )
        if prompt_result_fields:
            action_result.update(prompt_result_fields)
        action_result["semantic_result_v2"] = project_trace_action_result_v2(
            action_result,
        )
        if record_attempt_func is not None:
            await _record_action_attempt(
                record_attempt_func,
                validated_spec,
                eval_result,
                storage_timestamp_utc=normalized_storage_timestamp_utc,
                execution_result=execution_result,
                source_llm_trace_id=source_llm_trace_id,
            )
        action_results.append(action_result)
    return action_results


def _rejected_action_result_fail_closed(
    action_spec: dict[str, Any],
    eval_result: dict[str, Any],
    *,
    result_summary: str,
) -> ActionResultV1:
    """Build a prompt-safe rejected result for unreadable surface metadata.

    Deterministic result materialization refuses action specs whose surface
    role and continuation reference are missing or invalid. Execution still
    returns a rejected row so malformed cognition output cannot crash the
    trace and cannot be executed or persisted as a valid action.

    Args:
        action_spec: Raw action spec rejected by deterministic evaluation.
        eval_result: Evaluation output that rejected the action spec.
        result_summary: Prompt-safe rejection summary from the evaluator.

    Returns:
        A fail-closed rejected ``ActionResultV1`` row with the canonical
        no-continuation surface projection.
    """

    handler_owner = eval_result.get("handler_owner")
    if not isinstance(handler_owner, str):
        handler_owner = ""
    kind = action_spec.get("kind")
    action_kind = kind if isinstance(kind, str) and kind else "unknown"
    visibility = action_spec.get("visibility")
    if visibility not in ("private", "preview", "user_visible"):
        visibility = "private"
    action_result: ActionResultV1 = {
        "schema_version": ACTION_RESULT_VERSION,
        "action_attempt_id": action_attempt_id_from_eval_result(eval_result),
        "action_kind": action_kind,
        "handler_owner": handler_owner,
        "status": "rejected",
        "visibility": visibility,
        "result_summary": result_summary,
        "result_refs": [],
        "continuation": dict(DEFAULT_ACTION_CONTINUATION),
        "surface_role": "ordinary",
        "goal_continuation_ref": None,
        "completed_at": None,
    }
    return action_result


def _action_availability_context(
    action_spec: Mapping[str, object],
) -> ActionAvailabilityContextV1:
    """Project trusted action identity into the registry probe context."""

    context: ActionAvailabilityContextV1 = {
        "permission_ref": str(action_spec.get("kind") or ""),
    }
    source_refs = action_spec.get("source_refs")
    if isinstance(source_refs, list) and source_refs:
        source_ref = source_refs[0]
        if isinstance(source_ref, Mapping):
            source_kind = source_ref.get("ref_kind")
            if isinstance(source_kind, str):
                context["source_kind"] = source_kind
    target = action_spec.get("target")
    if isinstance(target, Mapping):
        scope = target.get("scope")
        if isinstance(scope, Mapping):
            context["target_scope"] = dict(scope)
    params = action_spec.get("params")
    if isinstance(params, Mapping):
        requested_work_kind = params.get("work_kind")
        if isinstance(requested_work_kind, str):
            context["requested_work_kind"] = requested_work_kind
    return context


async def _record_action_attempt(
    record_attempt_func: ActionAttemptRecorder,
    action_spec: dict[str, Any],
    eval_result: dict[str, Any],
    *,
    storage_timestamp_utc: str,
    execution_result: dict[str, Any],
    source_llm_trace_id: str = "",
) -> None:
    """Record one action attempt through the existing idempotency ledger."""

    attempt_record = build_action_attempt_record(
        action_spec,
        eval_result,
        recorded_at=storage_timestamp_utc,
        execution_result=execution_result,
        source_llm_trace_id=source_llm_trace_id,
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


def _accepted_task_execution_status(queue_result: dict[str, Any]) -> str:
    """Map accepted-task enqueue outcome into action-result status."""

    if queue_result["status"] == "failed":
        status = "failed"
        return status
    status = "pending"
    return status


def _accepted_task_status_prompt_fields(
    status_result: dict[str, Any],
) -> dict[str, Any]:
    """Project an accepted-task status lookup into prompt-safe fields."""

    if status_result["status"] != "active":
        fields = {
            "result_summary": "No active accepted task found.",
            "accepted_task_state": "delivered",
            "accepted_task_summary": "",
            "acknowledgement_constraint": "progress_report_allowed",
            "wait_guidance": "no_wait",
        }
        return fields
    task = status_result["task"]
    accepted_task_state = _accepted_task_prompt_state(str(task["state"]))
    fields = {
        "result_summary": _accepted_task_result_summary(task),
        "accepted_task_state": accepted_task_state,
        "accepted_task_summary": str(task.get("accepted_task_summary", "")),
        "acknowledgement_constraint": "progress_report_allowed",
        "wait_guidance": _accepted_task_wait_guidance(accepted_task_state),
    }
    return fields


def _accepted_task_prompt_state(task_state: str) -> str:
    """Map lifecycle state to the compact prompt vocabulary."""

    if task_state == "enqueueing" or task_state == "pending":
        prompt_state = "scheduled"
    elif task_state == "running":
        prompt_state = "running"
    elif task_state == "result_ready":
        prompt_state = "result_ready"
    elif task_state == "failure_ready":
        prompt_state = "failed"
    elif task_state == "delivered":
        prompt_state = "delivered"
    elif task_state == "enqueue_failed":
        prompt_state = "enqueue_failed"
    elif task_state in ("delivery_in_progress", "delivery_retryable"):
        prompt_state = "delivery_failed"
    else:
        prompt_state = "failed"
    return prompt_state


def _accepted_task_wait_guidance(accepted_task_state: str) -> str:
    """Return wait guidance for one prompt accepted-task state."""

    if accepted_task_state in ("scheduled", "already_active", "running"):
        guidance = "non_numeric_wait"
    elif accepted_task_state in ("result_ready", "delivered"):
        guidance = "no_wait"
    else:
        guidance = "unavailable"
    return guidance


def _accepted_task_result_summary(task: dict[str, Any]) -> str:
    """Build a compact status-check summary from task state."""

    state = str(task.get("state", ""))
    summary = str(task.get("accepted_task_summary", "")).strip()
    if state in ("result_ready", "delivered"):
        result_summary = str(task.get("result_summary", "")).strip()
        if result_summary:
            summary = f"任务已有结果：{result_summary}"
            return summary
    if state in ("failure_ready", "enqueue_failed", "delivery_retryable"):
        failure_summary = str(task.get("failure_summary", "")).strip()
        if failure_summary:
            summary = f"任务失败：{failure_summary}"
            return summary
    if summary:
        result_summary = f"已接纳任务当前状态为 {state}：{summary}"
    else:
        result_summary = f"已接纳任务当前状态为 {state}。"
    return result_summary
