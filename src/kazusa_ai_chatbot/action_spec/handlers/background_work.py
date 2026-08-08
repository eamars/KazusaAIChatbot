"""Queue handlers for retained future-speak and bound coding lifecycles."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from kazusa_ai_chatbot.accepted_task.models import AcceptedTaskCreateRequest
from kazusa_ai_chatbot.action_spec.models import (
    ActionValidationError,
    validate_action_spec,
)
from kazusa_ai_chatbot.action_spec.registry import (
    ACCEPTED_CODING_TASK_REQUEST_CAPABILITY,
    FUTURE_SPEAK_CAPABILITY,
)
from kazusa_ai_chatbot.background_work import enqueue_background_work_request
from kazusa_ai_chatbot.background_work.models import (
    BACKGROUND_WORK_REQUESTED_DELIVERY,
    TASK_ORCHESTRATOR_WORKER,
    TASK_ORCHESTRATOR_WORKER_PAYLOAD_VERSION,
    BackgroundWorkQueueRequest,
    BackgroundWorkQueueResult,
)
from kazusa_ai_chatbot.config import (
    BACKGROUND_WORK_OUTPUT_CHAR_LIMIT,
    CODING_AGENT_WORKSPACE_ROOT,
)
from kazusa_ai_chatbot.db import DatabaseOperationError
from kazusa_ai_chatbot.time_boundary import local_llm_datetime_to_storage_utc_iso


BackgroundWorkEnqueueFunc = Callable[
    [BackgroundWorkQueueRequest],
    Awaitable[BackgroundWorkQueueResult],
]

_REQUIRED_DELIVERY_TARGET_SCOPE_FIELDS = (
    "source_platform",
    "source_channel_id",
    "source_channel_type",
    "source_message_id",
    "source_platform_bot_id",
    "source_character_name",
    "source_trigger_source",
    "requester_global_user_id",
    "requester_platform_user_id",
    "requester_display_name",
)
_BOUND_CODING_ACTIONS = frozenset((
    "revise_proposal",
    "summarize",
    "status",
    "approve_and_verify",
    "respond_to_blocker",
    "cancel",
))
_BOUND_CODING_PARAM_FIELDS = frozenset((
    "task_brief",
    "coding_action",
    "coding_run_ref",
    "revision_instruction",
    "execution_request",
    "approval_evidence",
    "requested_delivery",
    "max_output_chars",
))


def validate_future_speak_action(
    action_spec: dict[str, Any],
) -> dict[str, Any]:
    """Validate the retained deterministic future-speak queue action."""

    validated = validate_action_spec(action_spec)
    if validated["kind"] != FUTURE_SPEAK_CAPABILITY:
        raise ActionValidationError("kind: expected future_speak")
    _validate_private_background_target(validated)
    params = validated["params"]
    _validate_requested_delivery_and_output_limit(params)
    trigger_at = _required_param_text(params, "trigger_at")
    try:
        local_llm_datetime_to_storage_utc_iso(trigger_at)
    except ValueError as exc:
        raise ActionValidationError(
            f"trigger_at: expected exact local YYYY-MM-DD HH:MM: {exc}"
        ) from exc
    _required_param_text(params, "continuation_objective")
    return validated


def validate_accepted_coding_task_action(
    action_spec: dict[str, Any],
) -> dict[str, Any]:
    """Validate a bound coding-run continuation without widening authority."""

    validated = validate_action_spec(action_spec)
    if validated["kind"] != ACCEPTED_CODING_TASK_REQUEST_CAPABILITY:
        raise ActionValidationError(
            "kind: expected accepted_coding_task_request"
        )
    _validate_private_background_target(validated)
    params = validated["params"]
    unsupported = set(params) - _BOUND_CODING_PARAM_FIELDS
    if unsupported:
        raise ActionValidationError("params: unsupported coding fields")
    _validate_requested_delivery_and_output_limit(params)
    _required_param_text(params, "task_brief")
    coding_action = _required_param_text(params, "coding_action")
    if coding_action not in _BOUND_CODING_ACTIONS:
        raise ActionValidationError("coding_action: unsupported value")
    _coding_run_id(_required_param_text(params, "coding_run_ref"))
    _validate_optional_param_text(params, "revision_instruction")
    _validate_optional_param_text(params, "execution_request")
    if coding_action == "revise_proposal":
        _required_param_text(params, "revision_instruction")
    if coding_action == "approve_and_verify":
        _validate_approval_evidence(params.get("approval_evidence"), validated)
    elif "approval_evidence" in params:
        raise ActionValidationError("approval_evidence: unexpected parameter")
    return validated


async def enqueue_future_speak_action(
    action_spec: dict[str, Any],
    *,
    storage_timestamp_utc: str,
    action_attempt_id: str,
    enqueue_background_work_func: BackgroundWorkEnqueueFunc | None = None,
) -> BackgroundWorkQueueResult:
    """Persist a retained future-speak task and its deterministic worker job."""

    validated = validate_future_speak_action(action_spec)
    params = validated["params"]
    trigger_at = _required_param_text(params, "trigger_at")
    continuation_objective = _required_param_text(
        params,
        "continuation_objective",
    )
    task_summary = f"Schedule future speak for {trigger_at}: {continuation_objective}"
    return await _create_or_queue_accepted_task(
        validated,
        storage_timestamp_utc=storage_timestamp_utc,
        action_attempt_id=action_attempt_id,
        task_kind="future_speak",
        semantic_objective=continuation_objective,
        accepted_task_summary=continuation_objective,
        requested_worker="future_speak",
        worker_payload={
            "trigger_at": trigger_at,
            "continuation_objective": continuation_objective,
        },
        enqueue_background_work_func=enqueue_background_work_func,
        task_brief=task_summary,
    )


async def enqueue_accepted_coding_task_action(
    action_spec: dict[str, Any],
    *,
    storage_timestamp_utc: str,
    action_attempt_id: str,
    enqueue_background_work_func: BackgroundWorkEnqueueFunc | None = None,
) -> BackgroundWorkQueueResult:
    """Persist one reviewed continuation for an existing coding run."""

    validated = validate_accepted_coding_task_action(action_spec)
    params = validated["params"]
    coding_action = _required_param_text(params, "coding_action")
    coding_run_ref = _required_param_text(params, "coding_run_ref")
    task_brief = _required_param_text(params, "task_brief")
    coding_request = _bound_coding_request(
        params,
        validated=validated,
        storage_timestamp_utc=storage_timestamp_utc,
    )
    semantic_objective = _coding_semantic_objective(
        coding_action=coding_action,
        coding_run_ref=coding_run_ref,
        task_brief=task_brief,
    )
    return await _create_or_queue_accepted_task(
        validated,
        storage_timestamp_utc=storage_timestamp_utc,
        action_attempt_id=action_attempt_id,
        task_kind="coding_continuation",
        semantic_objective=semantic_objective,
        accepted_task_summary=semantic_objective,
        requested_worker=TASK_ORCHESTRATOR_WORKER,
        worker_payload={
            "schema_version": TASK_ORCHESTRATOR_WORKER_PAYLOAD_VERSION,
            "operation": "continue_bound_coding_run",
            "checkpoint": None,
            "coding_request": coding_request,
        },
        enqueue_background_work_func=enqueue_background_work_func,
        task_brief=task_brief,
    )


def _validate_private_background_target(validated: Mapping[str, Any]) -> None:
    """Require the shared private user-scope queue boundary."""

    if validated["visibility"] != "private":
        raise ActionValidationError("visibility: expected private")
    if validated["urgency"] != "background":
        raise ActionValidationError("urgency: expected background")
    target = validated["target"]
    if target["owner"] != "background_work":
        raise ActionValidationError("owner: expected background_work")
    if target["target_kind"] != "current_user":
        raise ActionValidationError("target_kind: expected current_user")
    if target["target_id"] is not None:
        raise ActionValidationError("target_id: expected null")
    scope = target["scope"]
    for field_name in _REQUIRED_DELIVERY_TARGET_SCOPE_FIELDS:
        _required_scope_text(scope, field_name)
    if _required_scope_text(scope, "source_trigger_source") != "user_message":
        raise ActionValidationError(
            "scope.source_trigger_source: expected user_message"
        )


def _validate_requested_delivery_and_output_limit(
    params: Mapping[str, object],
) -> None:
    """Validate shared accepted-task queue settings."""

    if params.get("requested_delivery") != BACKGROUND_WORK_REQUESTED_DELIVERY:
        raise ActionValidationError("requested_delivery: unsupported value")
    max_output_chars = params.get("max_output_chars")
    if isinstance(max_output_chars, bool) or not isinstance(
        max_output_chars,
        int,
    ):
        raise ActionValidationError("max_output_chars: expected integer")
    if max_output_chars < 1:
        raise ActionValidationError("max_output_chars: expected positive integer")
    if max_output_chars > BACKGROUND_WORK_OUTPUT_CHAR_LIMIT:
        raise ActionValidationError("max_output_chars: exceeds configured limit")


def _validate_approval_evidence(
    value: object,
    validated: Mapping[str, Any],
) -> None:
    """Require trusted current-turn approval provenance for coding mutation."""

    if not isinstance(value, Mapping):
        raise ActionValidationError("approval_evidence: required for approval")
    scope = validated["target"]["scope"]
    for field_name in (
        "source_message_id",
        "source_trigger_source",
        "requester_global_user_id",
        "quote",
        "storage_timestamp_utc",
    ):
        _required_mapping_text(value, field_name, "approval_evidence")
    if value["source_trigger_source"] != "user_message":
        raise ActionValidationError(
            "approval_evidence.source_trigger_source: expected user_message"
        )
    if value["requester_global_user_id"] != scope[
        "requester_global_user_id"
    ]:
        raise ActionValidationError(
            "approval_evidence.requester_global_user_id: scope mismatch"
        )


def _bound_coding_request(
    params: Mapping[str, object],
    *,
    validated: Mapping[str, Any],
    storage_timestamp_utc: str,
) -> dict[str, object]:
    """Build the frozen public continuation request from reviewed state."""

    workspace_root = CODING_AGENT_WORKSPACE_ROOT.strip()
    if not workspace_root:
        raise ActionValidationError("coding workspace is unavailable")
    coding_action = _required_param_text(params, "coding_action")
    request: dict[str, object] = {
        "workspace_root": workspace_root,
        "run_id": _coding_run_id(_required_param_text(params, "coding_run_ref")),
        "action": coding_action,
        "reason": str(validated["reason"]).strip(),
    }
    revision_instruction = _optional_param_text(params, "revision_instruction")
    execution_request = _optional_param_text(params, "execution_request")
    if revision_instruction:
        request["revision_instruction"] = revision_instruction
    if execution_request:
        request["execution_request"] = execution_request
    if coding_action == "approve_and_verify":
        approval_evidence = params["approval_evidence"]
        if not isinstance(approval_evidence, Mapping):
            raise ActionValidationError("approval_evidence: required for approval")
        request["approval"] = {
            "approved": True,
            "approved_by": _required_mapping_text(
                approval_evidence,
                "requester_global_user_id",
                "approval_evidence",
            ),
            "approved_at": _required_mapping_text(
                approval_evidence,
                "storage_timestamp_utc",
                "approval_evidence",
            ),
            "approval_reason": _required_mapping_text(
                approval_evidence,
                "quote",
                "approval_evidence",
            ),
        }
    return request


async def _create_or_queue_accepted_task(
    validated: Mapping[str, Any],
    *,
    storage_timestamp_utc: str,
    action_attempt_id: str,
    task_kind: str,
    semantic_objective: str,
    accepted_task_summary: str,
    requested_worker: str,
    worker_payload: dict[str, object],
    enqueue_background_work_func: BackgroundWorkEnqueueFunc | None,
    task_brief: str,
) -> BackgroundWorkQueueResult:
    """Create v2 accepted state and enqueue its single approved worker."""

    from kazusa_ai_chatbot.accepted_task.lifecycle import (
        create_or_return_active_accepted_task,
        mark_accepted_task_enqueue_failed,
        mark_accepted_task_pending,
    )

    create_request = _accepted_task_create_request(
        validated,
        storage_timestamp_utc=storage_timestamp_utc,
        task_kind=task_kind,
        semantic_objective=semantic_objective,
        accepted_task_summary=accepted_task_summary,
    )
    create_result = await create_or_return_active_accepted_task(create_request)
    accepted_task = create_result["task"]
    active_state = _task_text(accepted_task, "state")
    if create_result["status"] == "already_active" and active_state not in {
        "enqueueing",
        "pending",
    }:
        return _accepted_task_queue_result(
            accepted_task,
            accepted_task_state="already_active",
            status="pending",
            job_id=_task_text(accepted_task, "executor_ref"),
            result_summary="Accepted task is already active.",
            acknowledgement_constraint="progress_report_allowed",
            wait_guidance="non_numeric_wait",
        )

    queue_request = _queue_request_from_accepted_task(
        validated,
        accepted_task,
        storage_timestamp_utc=storage_timestamp_utc,
        action_attempt_id=action_attempt_id,
        semantic_objective=semantic_objective,
        requested_worker=requested_worker,
        worker_payload=worker_payload,
    )
    job_id = queue_request["job_id"]
    if active_state == "pending":
        existing_job_id = _task_text(accepted_task, "executor_ref")
        if not existing_job_id:
            raise DatabaseOperationError(
                "pending accepted task is missing its background job id"
            )
        queue_request["job_id"] = existing_job_id
        job_id = existing_job_id
    else:
        pending_task = await mark_accepted_task_pending(
            accepted_task_id=_task_text(accepted_task, "accepted_task_id"),
            executor_ref=job_id,
            updated_at=storage_timestamp_utc,
        )
        if pending_task is None:
            await mark_accepted_task_enqueue_failed(
                accepted_task_id=_task_text(
                    accepted_task,
                    "accepted_task_id",
                ),
                failure_summary="Accepted task could not become pending.",
                updated_at=storage_timestamp_utc,
            )
            raise DatabaseOperationError(
                "accepted task pending transition failed before job insert"
            )
    enqueue = enqueue_background_work_func or enqueue_background_work_request
    try:
        queue_result = await enqueue(queue_request)
    except (DatabaseOperationError, ValueError) as exc:
        await mark_accepted_task_enqueue_failed(
            accepted_task_id=_task_text(accepted_task, "accepted_task_id"),
            failure_summary="Background task enqueue failed.",
            updated_at=storage_timestamp_utc,
        )
        raise DatabaseOperationError("accepted task enqueue failed") from exc
    if queue_result["job_id"] != job_id:
        await mark_accepted_task_enqueue_failed(
            accepted_task_id=_task_text(accepted_task, "accepted_task_id"),
            failure_summary="Background task returned an invalid job id.",
            updated_at=storage_timestamp_utc,
        )
        raise DatabaseOperationError(
            "accepted task enqueue returned an unexpected job id"
        )
    return _accepted_task_queue_result(
        accepted_task,
        accepted_task_state="scheduled",
        status="pending",
        job_id=queue_result["job_id"],
        result_summary="Accepted task scheduled.",
        acknowledgement_constraint="promise_allowed",
        wait_guidance="non_numeric_wait",
    )


def _accepted_task_create_request(
    validated: Mapping[str, Any],
    *,
    storage_timestamp_utc: str,
    task_kind: str,
    semantic_objective: str,
    accepted_task_summary: str,
) -> AcceptedTaskCreateRequest:
    """Build one trusted v2 accepted-task request from the action scope."""

    if task_kind not in {"future_speak", "coding_continuation"}:
        raise ActionValidationError("accepted task kind is unsupported")
    params = validated["params"]
    scope = validated["target"]["scope"]
    request: AcceptedTaskCreateRequest = {
        "task_kind": task_kind,
        "semantic_objective": semantic_objective,
        "accepted_task_summary": accepted_task_summary,
        "requested_delivery": BACKGROUND_WORK_REQUESTED_DELIVERY,
        "max_output_chars": int(params["max_output_chars"]),
        "source_trigger_source": _required_scope_text(
            scope,
            "source_trigger_source",
        ),
        "source_platform": _required_scope_text(scope, "source_platform"),
        "source_channel_id": _required_scope_text(scope, "source_channel_id"),
        "source_channel_type": _required_scope_text(scope, "source_channel_type"),
        "source_message_id": _required_scope_text(scope, "source_message_id"),
        "source_platform_bot_id": _required_scope_text(
            scope,
            "source_platform_bot_id",
        ),
        "source_character_name": _required_scope_text(
            scope,
            "source_character_name",
        ),
        "requester_global_user_id": _required_scope_text(
            scope,
            "requester_global_user_id",
        ),
        "requester_platform_user_id": _required_scope_text(
            scope,
            "requester_platform_user_id",
        ),
        "requester_display_name": _required_scope_text(
            scope,
            "requester_display_name",
        ),
        "storage_timestamp_utc": storage_timestamp_utc,
    }
    return request


def _queue_request_from_accepted_task(
    validated: Mapping[str, Any],
    accepted_task: Mapping[str, Any],
    *,
    storage_timestamp_utc: str,
    action_attempt_id: str,
    semantic_objective: str,
    requested_worker: str,
    worker_payload: Mapping[str, object],
) -> BackgroundWorkQueueRequest:
    """Build one internal v2 job request from accepted-task state."""

    params = validated["params"]
    scope = validated["target"]["scope"]
    accepted_task_id = _task_text(accepted_task, "accepted_task_id")
    request: BackgroundWorkQueueRequest = {
        "job_id": f"job-{accepted_task_id.removeprefix('task-')}",
        "source_action_attempt_id": action_attempt_id,
        "idempotency_key": f"background_work:{accepted_task_id}",
        "accepted_task_id": accepted_task_id,
        "task_identity_key": _task_text(accepted_task, "task_identity_key"),
        "semantic_objective": semantic_objective,
        "requested_worker": requested_worker,
        "worker_payload": dict(worker_payload),
        "source_platform": _required_scope_text(scope, "source_platform"),
        "source_channel_id": _required_scope_text(scope, "source_channel_id"),
        "source_channel_type": _required_scope_text(scope, "source_channel_type"),
        "source_message_id": _required_scope_text(scope, "source_message_id"),
        "source_platform_bot_id": _required_scope_text(
            scope,
            "source_platform_bot_id",
        ),
        "source_character_name": _required_scope_text(
            scope,
            "source_character_name",
        ),
        "requester_global_user_id": _required_scope_text(
            scope,
            "requester_global_user_id",
        ),
        "requester_platform_user_id": _required_scope_text(
            scope,
            "requester_platform_user_id",
        ),
        "requester_display_name": _required_scope_text(
            scope,
            "requester_display_name",
        ),
        "requested_delivery": BACKGROUND_WORK_REQUESTED_DELIVERY,
        "max_output_chars": int(params["max_output_chars"]),
        "storage_timestamp_utc": storage_timestamp_utc,
    }
    return request


def _accepted_task_queue_result(
    task: Mapping[str, Any],
    *,
    accepted_task_state: str,
    status: str,
    job_id: str,
    result_summary: str,
    acknowledgement_constraint: str,
    wait_guidance: str,
) -> BackgroundWorkQueueResult:
    """Project durable state into a prompt-safe queue response."""

    result: BackgroundWorkQueueResult = {
        "status": status,
        "job_id": job_id,
        "job_ref": f"background_work_job:{job_id}" if job_id else "",
        "accepted_task_id": _task_text(task, "accepted_task_id"),
        "task_identity_key": _task_text(task, "task_identity_key"),
        "accepted_task_summary": _task_text(task, "accepted_task_summary"),
        "acknowledgement_constraint": acknowledgement_constraint,
        "wait_guidance": wait_guidance,
        "result_summary": result_summary,
    }
    result["accepted_task_state"] = accepted_task_state
    return result


def _coding_semantic_objective(
    *,
    coding_action: str,
    coding_run_ref: str,
    task_brief: str,
) -> str:
    """Build stable task identity material for a bound coding continuation."""

    return f"{coding_action} {coding_run_ref}: {task_brief}"


def _coding_run_id(coding_run_ref: str) -> str:
    """Extract the frozen public run id from its prompt-safe reference."""

    prefix = "coding_run:"
    if not coding_run_ref.startswith(prefix):
        raise ActionValidationError(
            "coding_run_ref: expected prompt-safe coding_run:<run_id>"
        )
    run_id = coding_run_ref[len(prefix):].strip()
    if not run_id:
        raise ActionValidationError("coding_run_ref: expected a run id")
    return run_id


def _required_param_text(params: Mapping[str, object], field_name: str) -> str:
    """Return one required semantic action parameter."""

    return _required_mapping_text(params, field_name, "params")


def _validate_optional_param_text(
    params: Mapping[str, object],
    field_name: str,
) -> None:
    """Validate an optional parameter when the caller supplied it."""

    if field_name in params:
        _required_param_text(params, field_name)


def _optional_param_text(params: Mapping[str, object], field_name: str) -> str:
    """Return one optional semantic action parameter."""

    if field_name not in params:
        return ""
    return _required_mapping_text(params, field_name, "params")


def _required_scope_text(scope: Mapping[str, object], field_name: str) -> str:
    """Return one required trusted delivery-scope field."""

    return _required_mapping_text(scope, field_name, "scope")


def _required_mapping_text(
    value: Mapping[str, object],
    field_name: str,
    label: str,
) -> str:
    """Require one non-empty text value from a typed mapping."""

    field_value = value.get(field_name)
    if not isinstance(field_value, str) or not field_value.strip():
        raise ActionValidationError(f"{label}.{field_name}: expected non-empty string")
    return field_value.strip()


def _task_text(task: Mapping[str, Any], field_name: str) -> str:
    """Read one prompt-safe text field from a durable accepted-task row."""

    value = task.get(field_name)
    if not isinstance(value, str):
        return ""
    return value.strip()
