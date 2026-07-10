"""Action-spec handler for generic background-work queue requests."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from kazusa_ai_chatbot.accepted_task import (
    create_or_return_active_accepted_task,
    mark_accepted_task_enqueue_failed,
    mark_accepted_task_pending,
)
from kazusa_ai_chatbot.accepted_task.models import AcceptedTaskCreateRequest
from kazusa_ai_chatbot.action_spec.models import (
    ActionValidationError,
    validate_action_spec,
)
from kazusa_ai_chatbot.action_spec.registry import (
    ACCEPTED_CODING_TASK_REQUEST_CAPABILITY,
    BACKGROUND_WORK_REQUEST_CAPABILITY,
    FUTURE_SPEAK_CAPABILITY,
)
from kazusa_ai_chatbot.background_work import enqueue_background_work_request
from kazusa_ai_chatbot.background_work.models import (
    BACKGROUND_WORK_REQUESTED_DELIVERY,
    BackgroundWorkQueueRequest,
    BackgroundWorkQueueResult,
)
from kazusa_ai_chatbot.config import BACKGROUND_WORK_OUTPUT_CHAR_LIMIT
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
_FORBIDDEN_WORKER_LOCAL_PARAMS = frozenset((
    "worker",
    "work_kind",
    "task_type",
    "tool_args",
    "artifact_text",
))
_CODING_FORBIDDEN_WORKER_LOCAL_PARAMS = (
    _FORBIDDEN_WORKER_LOCAL_PARAMS | frozenset((
        "approval",
        "command",
        "execution_specs",
        "local_root",
        "shell",
        "workspace_root",
    ))
)
_CODING_ACTIONS_REQUIRING_RUN_REF = frozenset((
    "revise_proposal",
    "summarize",
    "status",
    "approve_and_verify",
    "respond_to_blocker",
    "cancel",
))


def validate_background_work_action(
    action_spec: dict[str, Any],
) -> dict[str, Any]:
    """Validate one private generic background-work queue action."""

    validated = validate_action_spec(action_spec)
    if validated["kind"] != BACKGROUND_WORK_REQUEST_CAPABILITY:
        raise ActionValidationError("kind: expected background_work_request")
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
        _require_non_empty_scope(scope, field_name)
    _require_user_message_source(scope)

    params = validated["params"]
    forbidden_params = sorted(
        field_name
        for field_name in params
        if field_name in _FORBIDDEN_WORKER_LOCAL_PARAMS
    )
    if forbidden_params:
        raise ActionValidationError(
            "params: worker-local fields are not allowed: "
            f"{', '.join(forbidden_params)}"
        )
    requested_delivery = params.get("requested_delivery")
    if requested_delivery != BACKGROUND_WORK_REQUESTED_DELIVERY:
        raise ActionValidationError("requested_delivery: unsupported value")
    _require_non_empty_param(params, "task_brief")
    max_output_chars = params.get("max_output_chars")
    if not isinstance(max_output_chars, int):
        raise ActionValidationError("max_output_chars: expected integer")
    if max_output_chars < 1:
        raise ActionValidationError("max_output_chars: expected positive integer")
    if max_output_chars > BACKGROUND_WORK_OUTPUT_CHAR_LIMIT:
        raise ActionValidationError("max_output_chars: exceeds configured limit")
    return validated


def validate_future_speak_action(
    action_spec: dict[str, Any],
) -> dict[str, Any]:
    """Validate one private future-speak background-work queue action."""

    validated = validate_action_spec(action_spec)
    if validated["kind"] != FUTURE_SPEAK_CAPABILITY:
        raise ActionValidationError("kind: expected future_speak")
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
        _require_non_empty_scope(scope, field_name)
    _require_user_message_source(scope)

    params = validated["params"]
    requested_delivery = params.get("requested_delivery")
    if requested_delivery != BACKGROUND_WORK_REQUESTED_DELIVERY:
        raise ActionValidationError("requested_delivery: unsupported value")
    trigger_at = _param_text(params, "trigger_at")
    try:
        local_llm_datetime_to_storage_utc_iso(trigger_at)
    except ValueError as exc:
        raise ActionValidationError(
            f"trigger_at: expected exact local YYYY-MM-DD HH:MM: {exc}"
        ) from exc
    _require_non_empty_param(params, "continuation_objective")
    max_output_chars = params.get("max_output_chars")
    if not isinstance(max_output_chars, int):
        raise ActionValidationError("max_output_chars: expected integer")
    if max_output_chars < 1:
        raise ActionValidationError("max_output_chars: expected positive integer")
    if max_output_chars > BACKGROUND_WORK_OUTPUT_CHAR_LIMIT:
        raise ActionValidationError("max_output_chars: exceeds configured limit")
    return validated


def validate_accepted_coding_task_action(
    action_spec: dict[str, Any],
) -> dict[str, Any]:
    """Validate one private durable coding-run background action."""

    validated = validate_action_spec(action_spec)
    if validated["kind"] != ACCEPTED_CODING_TASK_REQUEST_CAPABILITY:
        raise ActionValidationError("kind: expected accepted_coding_task_request")
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
        _require_non_empty_scope(scope, field_name)
    _require_user_message_source(scope)

    params = validated["params"]
    forbidden_params = sorted(
        field_name
        for field_name in params
        if field_name in _CODING_FORBIDDEN_WORKER_LOCAL_PARAMS
    )
    if forbidden_params:
        raise ActionValidationError(
            "params: worker-local fields are not allowed: "
            f"{', '.join(forbidden_params)}"
        )
    requested_delivery = params.get("requested_delivery")
    if requested_delivery != BACKGROUND_WORK_REQUESTED_DELIVERY:
        raise ActionValidationError("requested_delivery: unsupported value")
    _require_non_empty_param(params, "task_brief")
    coding_action = _param_text(params, "coding_action")
    if coding_action not in (
        "start",
        "revise_proposal",
        "summarize",
        "status",
        "approve_and_verify",
        "respond_to_blocker",
        "cancel",
    ):
        raise ActionValidationError("coding_action: unsupported value")
    if coding_action in _CODING_ACTIONS_REQUIRING_RUN_REF:
        coding_run_ref = _optional_param_text(params, "coding_run_ref")
        if not coding_run_ref.startswith("coding_run:"):
            raise ActionValidationError(
                "coding_run_ref: expected prompt-safe coding_run:<run_id>"
            )
    approval_evidence = params.get("approval_evidence")
    if coding_action == "approve_and_verify":
        _validate_approval_evidence(approval_evidence, scope)
    elif approval_evidence is not None:
        raise ActionValidationError("approval_evidence: unexpected parameter")
    max_output_chars = params.get("max_output_chars")
    if not isinstance(max_output_chars, int):
        raise ActionValidationError("max_output_chars: expected integer")
    if max_output_chars < 1:
        raise ActionValidationError("max_output_chars: expected positive integer")
    if max_output_chars > BACKGROUND_WORK_OUTPUT_CHAR_LIMIT:
        raise ActionValidationError("max_output_chars: exceeds configured limit")
    return validated


async def enqueue_background_work_action(
    action_spec: dict[str, Any],
    *,
    storage_timestamp_utc: str,
    action_attempt_id: str,
    enqueue_background_work_func: BackgroundWorkEnqueueFunc | None = None,
) -> BackgroundWorkQueueResult:
    """Persist one validated generic background-work queue action."""

    validated = validate_background_work_action(action_spec)
    params = validated["params"]
    task_brief = _param_text(params, "task_brief")
    result = await _create_or_queue_accepted_task(
        validated,
        storage_timestamp_utc=storage_timestamp_utc,
        action_attempt_id=action_attempt_id,
        action_kind=BACKGROUND_WORK_REQUEST_CAPABILITY,
        accepted_task_seed=task_brief,
        accepted_task_detail=task_brief,
        accepted_task_summary=task_brief,
        requested_worker="",
        worker_payload={},
        enqueue_background_work_func=enqueue_background_work_func,
    )
    return result


async def enqueue_accepted_coding_task_action(
    action_spec: dict[str, Any],
    *,
    storage_timestamp_utc: str,
    action_attempt_id: str,
    enqueue_background_work_func: BackgroundWorkEnqueueFunc | None = None,
) -> BackgroundWorkQueueResult:
    """Persist one validated durable coding-run background action."""

    validated = validate_accepted_coding_task_action(action_spec)
    params = validated["params"]
    task_brief = _param_text(params, "task_brief")
    coding_action = _param_text(params, "coding_action")
    coding_run_ref = _optional_param_text(params, "coding_run_ref")
    execution_request = _optional_param_text(params, "execution_request")
    approval_evidence = params.get("approval_evidence")
    worker_payload: dict[str, object] = {
        "schema_version": "coding_agent_worker_payload.v2",
        "operation": coding_action,
        "task_brief": task_brief,
        "coding_run_ref": coding_run_ref,
        "execution_request": execution_request,
    }
    if isinstance(approval_evidence, Mapping):
        worker_payload["approval_evidence"] = dict(approval_evidence)
    result = await _create_or_queue_accepted_task(
        validated,
        storage_timestamp_utc=storage_timestamp_utc,
        action_attempt_id=action_attempt_id,
        action_kind=ACCEPTED_CODING_TASK_REQUEST_CAPABILITY,
        accepted_task_seed=_coding_accepted_task_seed(
            coding_action=coding_action,
            coding_run_ref=coding_run_ref,
            task_brief=task_brief,
        ),
        accepted_task_detail=_coding_accepted_task_detail(
            coding_action=coding_action,
            coding_run_ref=coding_run_ref,
            task_brief=task_brief,
        ),
        accepted_task_summary=_coding_accepted_task_summary(
            coding_action=coding_action,
            coding_run_ref=coding_run_ref,
            task_brief=task_brief,
        ),
        requested_worker="coding_agent",
        worker_payload=worker_payload,
        enqueue_background_work_func=enqueue_background_work_func,
    )
    return result


def _validate_approval_evidence(
    value: object,
    scope: Mapping[str, object],
) -> None:
    """Validate the trusted current-message evidence required for approval."""

    if not isinstance(value, Mapping):
        raise ActionValidationError("approval_evidence: required for approval")
    for field_name in (
        "source_message_id",
        "source_trigger_source",
        "requester_global_user_id",
        "quote",
        "storage_timestamp_utc",
    ):
        field_value = value.get(field_name)
        if not isinstance(field_value, str) or not field_value.strip():
            raise ActionValidationError(
                f"approval_evidence.{field_name}: required"
            )
    if value["source_trigger_source"] != "user_message":
        raise ActionValidationError(
            "approval_evidence.source_trigger_source: expected user_message"
        )
    if value["requester_global_user_id"] != scope["requester_global_user_id"]:
        raise ActionValidationError(
            "approval_evidence.requester_global_user_id: scope mismatch"
        )


async def enqueue_future_speak_action(
    action_spec: dict[str, Any],
    *,
    storage_timestamp_utc: str,
    action_attempt_id: str,
    enqueue_background_work_func: BackgroundWorkEnqueueFunc | None = None,
) -> BackgroundWorkQueueResult:
    """Persist one validated future-speak background-work queue action."""

    validated = validate_future_speak_action(action_spec)
    params = validated["params"]
    trigger_at = _param_text(params, "trigger_at")
    continuation_objective = _param_text(params, "continuation_objective")
    task_summary = (
        f"Schedule future speak for {trigger_at}: "
        f"{continuation_objective}"
    )
    result = await _create_or_queue_accepted_task(
        validated,
        storage_timestamp_utc=storage_timestamp_utc,
        action_attempt_id=action_attempt_id,
        action_kind=FUTURE_SPEAK_CAPABILITY,
        accepted_task_seed=continuation_objective,
        accepted_task_detail=f"{trigger_at} {continuation_objective}",
        accepted_task_summary=continuation_objective,
        requested_worker="future_speak",
        worker_payload={
            "trigger_at": trigger_at,
            "continuation_objective": continuation_objective,
        },
        enqueue_background_work_func=enqueue_background_work_func,
        task_brief=task_summary,
    )
    return result


def _coding_accepted_task_seed(
    *,
    coding_action: str,
    coding_run_ref: str,
    task_brief: str,
) -> str:
    """Build duplicate-suppression seed for one coding task request."""

    if coding_action == "start":
        return task_brief
    return_value = f"{coding_action}:{coding_run_ref}:{task_brief}"
    return return_value


def _coding_accepted_task_detail(
    *,
    coding_action: str,
    coding_run_ref: str,
    task_brief: str,
) -> str:
    """Build prompt-safe accepted coding task detail."""

    if coding_run_ref:
        return_value = f"{coding_action} {coding_run_ref}: {task_brief}"
        return return_value
    return_value = f"{coding_action}: {task_brief}"
    return return_value


def _coding_accepted_task_summary(
    *,
    coding_action: str,
    coding_run_ref: str,
    task_brief: str,
) -> str:
    """Build prompt-safe accepted coding task summary."""

    if coding_action == "start":
        return_value = f"Coding task: {task_brief}"
        return return_value
    if coding_run_ref:
        return_value = f"Coding run {coding_run_ref}: {coding_action}"
        return return_value
    return_value = f"Coding run: {coding_action}"
    return return_value


async def _create_or_queue_accepted_task(
    validated: dict[str, Any],
    *,
    storage_timestamp_utc: str,
    action_attempt_id: str,
    action_kind: str,
    accepted_task_seed: str,
    accepted_task_detail: str,
    accepted_task_summary: str,
    requested_worker: str,
    worker_payload: dict[str, object],
    enqueue_background_work_func: BackgroundWorkEnqueueFunc | None,
    task_brief: str | None = None,
) -> BackgroundWorkQueueResult:
    """Create accepted-task state and enqueue one internal background job."""

    create_request = _accepted_task_create_request(
        validated,
        storage_timestamp_utc=storage_timestamp_utc,
        action_kind=action_kind,
        accepted_task_seed=accepted_task_seed,
        accepted_task_detail=accepted_task_detail,
        accepted_task_summary=accepted_task_summary,
    )
    create_result = await create_or_return_active_accepted_task(create_request)
    accepted_task = create_result["task"]
    if create_result["status"] == "already_active":
        result = _accepted_task_queue_result(
            accepted_task,
            accepted_task_state="already_active",
            status="pending",
            result_summary="Accepted task is already active.",
            acknowledgement_constraint="progress_report_allowed",
            wait_guidance="non_numeric_wait",
        )
        return result

    queue_request = _queue_request_from_accepted_task(
        validated,
        accepted_task,
        storage_timestamp_utc=storage_timestamp_utc,
        action_attempt_id=action_attempt_id,
        source_context=validated["reason"],
        requested_worker=requested_worker,
        worker_payload=worker_payload,
        task_brief=task_brief or accepted_task_summary,
    )
    if enqueue_background_work_func is None:
        enqueue_background_work_func = enqueue_background_work_request
    try:
        queue_result = await enqueue_background_work_func(queue_request)
    except (DatabaseOperationError, ValueError) as exc:
        await mark_accepted_task_enqueue_failed(
            accepted_task_id=_task_text(accepted_task, "accepted_task_id"),
            failure_summary=f"Background work enqueue failed: {exc}",
            updated_at=storage_timestamp_utc,
        )
        result = _accepted_task_queue_result(
            accepted_task,
            accepted_task_state="enqueue_failed",
            status="failed",
            result_summary=f"Accepted task enqueue failed: {exc}",
            acknowledgement_constraint="promise_forbidden_explain_failure",
            wait_guidance="unavailable",
        )
        return result

    pending_task = await mark_accepted_task_pending(
        accepted_task_id=_task_text(accepted_task, "accepted_task_id"),
        executor_ref=str(queue_result.get("job_id", "")),
        updated_at=storage_timestamp_utc,
    )
    if pending_task is None:
        raise DatabaseOperationError(
            "accepted task pending transition failed after job insert"
        )
    result = _accepted_task_queue_result(
        pending_task,
        accepted_task_state="scheduled",
        status="pending",
        result_summary="Accepted task scheduled.",
        acknowledgement_constraint="promise_allowed",
        wait_guidance="non_numeric_wait",
    )
    return result


def _accepted_task_create_request(
    validated: dict[str, Any],
    *,
    storage_timestamp_utc: str,
    action_kind: str,
    accepted_task_seed: str,
    accepted_task_detail: str,
    accepted_task_summary: str,
) -> AcceptedTaskCreateRequest:
    """Build the semantic accepted-task creation request."""

    params = validated["params"]
    scope = validated["target"]["scope"]
    request: AcceptedTaskCreateRequest = {
        "action_kind": action_kind,
        "accepted_task_seed": accepted_task_seed,
        "accepted_task_detail": accepted_task_detail,
        "accepted_task_summary": accepted_task_summary,
        "source_context": str(validated["reason"]).strip(),
        "requested_delivery": BACKGROUND_WORK_REQUESTED_DELIVERY,
        "max_output_chars": int(params["max_output_chars"]),
        "source_trigger_source": _scope_text(scope, "source_trigger_source"),
        "source_platform": _scope_text(scope, "source_platform"),
        "source_channel_id": _scope_text(scope, "source_channel_id"),
        "source_channel_type": _scope_text(scope, "source_channel_type"),
        "source_message_id": _scope_text(scope, "source_message_id"),
        "source_platform_bot_id": _scope_text(
            scope,
            "source_platform_bot_id",
        ),
        "source_character_name": _scope_text(scope, "source_character_name"),
        "requester_global_user_id": _scope_text(
            scope,
            "requester_global_user_id",
        ),
        "requester_platform_user_id": _scope_text(
            scope,
            "requester_platform_user_id",
        ),
        "requester_display_name": _scope_text(scope, "requester_display_name"),
        "storage_timestamp_utc": storage_timestamp_utc,
    }
    return request


def _queue_request_from_accepted_task(
    validated: dict[str, Any],
    accepted_task: dict[str, Any],
    *,
    storage_timestamp_utc: str,
    action_attempt_id: str,
    source_context: str,
    requested_worker: str,
    worker_payload: dict[str, object],
    task_brief: str,
) -> BackgroundWorkQueueRequest:
    """Build the internal queue request from accepted-task state."""

    params = validated["params"]
    scope = validated["target"]["scope"]
    accepted_task_id = _task_text(accepted_task, "accepted_task_id")
    request: BackgroundWorkQueueRequest = {
        "action_attempt_id": action_attempt_id,
        "idempotency_key": f"background_work:{accepted_task_id}",
        "accepted_task_id": accepted_task_id,
        "task_identity_key": _task_text(accepted_task, "task_identity_key"),
        "task_brief": task_brief,
        "source_context": source_context.strip(),
        "source_platform": _scope_text(scope, "source_platform"),
        "source_channel_id": _scope_text(scope, "source_channel_id"),
        "source_channel_type": _scope_text(scope, "source_channel_type"),
        "source_message_id": _scope_text(scope, "source_message_id"),
        "source_platform_bot_id": _scope_text(
            scope,
            "source_platform_bot_id",
        ),
        "source_character_name": _scope_text(scope, "source_character_name"),
        "requester_global_user_id": _scope_text(
            scope,
            "requester_global_user_id",
        ),
        "requester_platform_user_id": _scope_text(
            scope,
            "requester_platform_user_id",
        ),
        "requester_display_name": _scope_text(scope, "requester_display_name"),
        "requested_delivery": BACKGROUND_WORK_REQUESTED_DELIVERY,
        "max_output_chars": int(params["max_output_chars"]),
        "storage_timestamp_utc": storage_timestamp_utc,
    }
    if requested_worker:
        request["requested_worker"] = requested_worker
        request["worker_payload"] = dict(worker_payload)
    return request


def _accepted_task_queue_result(
    task: dict[str, Any],
    *,
    accepted_task_state: str,
    status: str,
    result_summary: str,
    acknowledgement_constraint: str,
    wait_guidance: str,
) -> BackgroundWorkQueueResult:
    """Project accepted-task state into a prompt-safe queue result."""

    result: BackgroundWorkQueueResult = {
        "status": status,
        "accepted_task_id": _task_text(task, "accepted_task_id"),
        "task_identity_key": _task_text(task, "task_identity_key"),
        "accepted_task_state": accepted_task_state,
        "accepted_task_summary": _task_text(task, "accepted_task_summary"),
        "acknowledgement_constraint": acknowledgement_constraint,
        "wait_guidance": wait_guidance,
        "result_summary": result_summary,
    }
    return result


def _require_non_empty_param(params: dict[str, Any], field_name: str) -> None:
    """Require one non-empty string param."""

    value = params.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise ActionValidationError(f"{field_name}: expected non-empty string")


def _require_non_empty_scope(scope: dict[str, Any], field_name: str) -> None:
    """Require one non-empty trusted target-scope string."""

    value = scope.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise ActionValidationError(
            f"scope.{field_name}: expected non-empty string"
        )


def _param_text(params: dict[str, Any], field_name: str) -> str:
    """Return one previously validated text param."""

    value = params[field_name]
    if not isinstance(value, str):
        raise ActionValidationError(f"{field_name}: expected string")
    return_value = value.strip()
    return return_value


def _optional_param_text(params: dict[str, Any], field_name: str) -> str:
    """Return one optional text param."""

    value = params.get(field_name)
    if not isinstance(value, str):
        return_value = ""
        return return_value
    return_value = value.strip()
    return return_value


def _scope_text(scope: dict[str, Any], field_name: str) -> str:
    """Return one trusted scope text field."""

    value = scope.get(field_name)
    if not isinstance(value, str):
        return_value = ""
        return return_value
    return_value = value.strip()
    return return_value


def _task_text(task: dict[str, Any], field_name: str) -> str:
    """Return one accepted-task text field."""

    value = task.get(field_name)
    if not isinstance(value, str):
        return_value = ""
        return return_value
    return_value = value.strip()
    return return_value


def _require_user_message_source(scope: dict[str, Any]) -> None:
    """Reject delayed task creation from autonomous or result-ready sources."""

    trigger_source = _scope_text(scope, "source_trigger_source")
    if trigger_source != "user_message":
        raise ActionValidationError(
            "scope.source_trigger_source: expected user_message"
        )
