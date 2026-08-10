"""Public v2 queue helpers for reviewed background-work payloads."""

from __future__ import annotations

from collections.abc import Mapping
from kazusa_ai_chatbot.background_work.models import (
    BACKGROUND_WORK_JOB_REF_OWNER,
    BACKGROUND_WORK_JOB_SCHEMA_VERSION,
    BACKGROUND_WORK_REQUESTED_DELIVERY,
    FUTURE_SPEAK_WORKER,
    TASK_ORCHESTRATOR_WORKER,
    TASK_ORCHESTRATOR_WORKER_PAYLOAD_VERSION,
    BackgroundWorkJobDoc,
    BackgroundWorkQueueRequest,
    BackgroundWorkQueueResult,
    background_work_job_ref,
)
from kazusa_ai_chatbot.config import (
    BACKGROUND_WORK_OUTPUT_CHAR_LIMIT,
    BACKGROUND_WORK_WORKER_MAX_ATTEMPTS,
)
from kazusa_ai_chatbot.db.background_work_jobs import insert_background_work_job
from kazusa_ai_chatbot.task_resolution.contracts import (
    validate_task_resolution_checkpoint,
    validate_task_resolution_execution_context,
)
from kazusa_ai_chatbot.time_boundary import normalize_storage_utc_iso


_REQUIRED_TEXT_FIELDS = (
    "job_id",
    "source_action_attempt_id",
    "idempotency_key",
    "accepted_task_id",
    "task_identity_key",
    "semantic_objective",
    "storage_timestamp_utc",
    "source_platform",
    "source_channel_type",
    "source_message_id",
    "source_platform_bot_id",
    "source_character_name",
    "requester_global_user_id",
    "requester_platform_user_id",
    "requester_display_name",
)
_BOUND_CODING_REQUEST_FIELDS = frozenset((
    "workspace_root",
    "run_id",
    "action",
    "revision_instruction",
    "approval",
    "execution_specs",
    "execution_request",
    "repair_attempt_limit",
    "reason",
))
_BOUND_CODING_ACTIONS = frozenset((
    "revise_proposal",
    "summarize",
    "status",
    "approve_and_verify",
    "respond_to_blocker",
    "cancel",
))
_PATCH_APPLY_APPROVAL_FIELDS = frozenset((
    "approved",
    "approved_by",
    "approved_at",
    "approval_reason",
))
_CODE_EXECUTION_SPEC_FIELDS = frozenset((
    "tool",
    "paths",
    "pytest_selectors",
    "timeout_seconds",
))


async def enqueue_background_work_request(
    request: BackgroundWorkQueueRequest,
) -> BackgroundWorkQueueResult:
    """Validate and persist one reviewed v2 background-work job."""

    _validate_queue_request(request)
    storage_timestamp_utc = normalize_storage_utc_iso(
        request["storage_timestamp_utc"],
    )
    job_id = request["job_id"].strip()
    job = _build_job_document(
        request,
        job_id=job_id,
        storage_timestamp_utc=storage_timestamp_utc,
    )
    stored_job = await insert_background_work_job(job)
    return _queue_result_from_job(stored_job)


def _validate_queue_request(request: BackgroundWorkQueueRequest) -> None:
    """Validate v2 queue state before it reaches durable persistence."""

    for field_name in _REQUIRED_TEXT_FIELDS:
        value = request.get(field_name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field_name} is required")
    if request["requested_delivery"] != BACKGROUND_WORK_REQUESTED_DELIVERY:
        raise ValueError("requested_delivery is not supported")
    max_output_chars = request.get("max_output_chars")
    if not isinstance(max_output_chars, int) or max_output_chars < 1:
        raise ValueError("max_output_chars must be a positive integer")
    if max_output_chars > BACKGROUND_WORK_OUTPUT_CHAR_LIMIT:
        raise ValueError("max_output_chars exceeds configured output limit")
    requested_worker = request.get("requested_worker")
    worker_payload = request.get("worker_payload")
    if requested_worker == TASK_ORCHESTRATOR_WORKER:
        _validate_task_orchestrator_payload(worker_payload, request)
        return
    if requested_worker == FUTURE_SPEAK_WORKER:
        _validate_future_speak_payload(worker_payload)
        return
    raise ValueError("requested_worker is not supported")


def validate_task_orchestrator_worker_payload(
    value: object,
) -> dict[str, object]:
    """Validate the exact durable payload owned by task orchestration."""

    if not isinstance(value, Mapping):
        raise ValueError("task_orchestrator worker_payload must be a mapping")
    if set(value) != {
        "schema_version",
        "operation",
        "checkpoint",
        "coding_request",
    }:
        raise ValueError("task_orchestrator worker_payload fields are invalid")
    if value.get("schema_version") != TASK_ORCHESTRATOR_WORKER_PAYLOAD_VERSION:
        raise ValueError("task_orchestrator worker_payload schema_version is invalid")
    operation = value.get("operation")
    checkpoint = value.get("checkpoint")
    coding_request = value.get("coding_request")
    if operation == "resume_task_resolution":
        if coding_request is not None:
            raise ValueError("resume_task_resolution cannot contain coding_request")
        validate_task_resolution_checkpoint(checkpoint)
        return dict(value)
    if operation == "continue_bound_coding_run":
        if checkpoint is not None:
            raise ValueError("continue_bound_coding_run cannot contain checkpoint")
        _validate_bound_coding_request(coding_request)
        return dict(value)
    raise ValueError("task_orchestrator worker_payload operation is unsupported")


def _validate_bound_coding_request(value: object) -> None:
    """Validate the frozen public coding continuation request shape.

    This worker boundary accepts only established coding-run actions and passes
    the validated public request unchanged to the retained coding lifecycle.
    """

    if not isinstance(value, Mapping):
        raise ValueError("continue_bound_coding_run requires coding_request")
    request = dict(value)
    unknown_fields = set(request) - _BOUND_CODING_REQUEST_FIELDS
    if unknown_fields:
        raise ValueError("coding_request contains unsupported fields")
    for field_name in ("workspace_root", "run_id", "action"):
        _require_non_empty_mapping_text(request, field_name, "coding_request")
    action = str(request["action"])
    if action not in _BOUND_CODING_ACTIONS:
        raise ValueError("coding_request.action is unsupported")
    _validate_optional_coding_texts(request)
    _validate_optional_coding_execution_specs(request)
    _validate_optional_repair_attempt_limit(request)
    if action == "approve_and_verify":
        _validate_patch_apply_approval(request.get("approval"))
    elif "approval" in request:
        raise ValueError("coding_request.approval is only valid for approval")
    if action == "revise_proposal":
        _require_non_empty_mapping_text(
            request,
            "revision_instruction",
            "coding_request",
        )


def _validate_optional_coding_texts(request: Mapping[str, object]) -> None:
    """Validate optional text fields exposed by the frozen coding request."""

    for field_name in (
        "revision_instruction",
        "execution_request",
        "reason",
    ):
        if field_name not in request:
            continue
        _require_non_empty_mapping_text(request, field_name, "coding_request")


def _validate_optional_coding_execution_specs(
    request: Mapping[str, object],
) -> None:
    """Validate optional public execution specs without executing commands."""

    if "execution_specs" not in request:
        return
    execution_specs = request["execution_specs"]
    if not isinstance(execution_specs, list):
        raise ValueError("coding_request.execution_specs must be a list")
    for execution_spec in execution_specs:
        if not isinstance(execution_spec, Mapping):
            raise ValueError("coding_request.execution_specs must contain objects")
        spec = dict(execution_spec)
        if set(spec) - _CODE_EXECUTION_SPEC_FIELDS:
            raise ValueError(
                "coding_request.execution_specs contain unsupported fields"
            )
        if "tool" in spec:
            _require_non_empty_mapping_text(
                spec,
                "tool",
                "coding_request.execution_specs",
            )
        for list_field in ("paths", "pytest_selectors"):
            if list_field not in spec:
                continue
            values = spec[list_field]
            if (
                not isinstance(values, list)
                or any(
                    not isinstance(item, str) or not item.strip()
                    for item in values
                )
            ):
                raise ValueError(
                    f"coding_request.execution_specs.{list_field} is invalid"
                )
        if "timeout_seconds" in spec:
            timeout_seconds = spec["timeout_seconds"]
            if (
                isinstance(timeout_seconds, bool)
                or not isinstance(timeout_seconds, int)
                or timeout_seconds < 1
            ):
                raise ValueError(
                    "coding_request.execution_specs.timeout_seconds is invalid"
                )


def _validate_optional_repair_attempt_limit(
    request: Mapping[str, object],
) -> None:
    """Validate the optional bounded repair-attempt integer."""

    if "repair_attempt_limit" not in request:
        return
    repair_attempt_limit = request["repair_attempt_limit"]
    if (
        isinstance(repair_attempt_limit, bool)
        or not isinstance(repair_attempt_limit, int)
        or repair_attempt_limit < 1
    ):
        raise ValueError("coding_request.repair_attempt_limit is invalid")


def _validate_patch_apply_approval(value: object) -> None:
    """Require the public approval object before a coding mutation continues."""

    if not isinstance(value, Mapping):
        raise ValueError("coding_request.approval is required")
    approval = dict(value)
    if set(approval) != _PATCH_APPLY_APPROVAL_FIELDS:
        raise ValueError("coding_request.approval fields are invalid")
    if approval["approved"] is not True:
        raise ValueError("coding_request.approval.approved must be true")
    for field_name in ("approved_by", "approved_at", "approval_reason"):
        _require_non_empty_mapping_text(
            approval,
            field_name,
            "coding_request.approval",
        )


def _require_non_empty_mapping_text(
    value: Mapping[str, object],
    field_name: str,
    label: str,
) -> str:
    """Return one required non-empty text value from a trusted mapping."""

    field_value = value.get(field_name)
    if not isinstance(field_value, str) or not field_value.strip():
        raise ValueError(f"{label}.{field_name} is required")
    text = field_value.strip()
    return text


def _validate_task_orchestrator_payload(
    value: object,
    request: BackgroundWorkQueueRequest,
) -> None:
    """Validate a task-resume payload and its persisted execution context."""

    payload = validate_task_orchestrator_worker_payload(value)
    if payload["operation"] != "resume_task_resolution":
        return
    execution_context = request.get("task_execution_context")
    validate_task_resolution_execution_context(execution_context)


def _validate_future_speak_payload(value: object) -> None:
    """Validate the retained deterministic future-speak handoff."""

    if not isinstance(value, Mapping):
        raise ValueError("future_speak worker_payload must be a mapping")
    if set(value) != {"trigger_at", "continuation_objective"}:
        raise ValueError("future_speak worker_payload fields are invalid")
    for field_name in ("trigger_at", "continuation_objective"):
        field_value = value.get(field_name)
        if not isinstance(field_value, str) or not field_value.strip():
            raise ValueError(f"future_speak worker_payload {field_name} is required")


def _build_job_document(
    request: BackgroundWorkQueueRequest,
    *,
    job_id: str,
    storage_timestamp_utc: str,
) -> BackgroundWorkJobDoc:
    """Build one v2 durable row from validated queue material."""

    task_execution_context = request.get("task_execution_context")
    if isinstance(task_execution_context, Mapping):
        context_projection = dict(task_execution_context)
    else:
        context_projection: dict[str, object] = {}
    job: BackgroundWorkJobDoc = {
        "schema_version": BACKGROUND_WORK_JOB_SCHEMA_VERSION,
        "job_id": job_id,
        "idempotency_key": request["idempotency_key"].strip(),
        "source_action_attempt_id": request["source_action_attempt_id"].strip(),
        "source_llm_trace_id": str(
            request.get("source_llm_trace_id") or ""
        ).strip(),
        "accepted_task_id": request["accepted_task_id"].strip(),
        "task_identity_key": request["task_identity_key"].strip(),
        "semantic_objective": request["semantic_objective"].strip(),
        "status": "queued",
        "delivery_state": "queued",
        "requested_delivery": request["requested_delivery"],
        "max_output_chars": int(request["max_output_chars"]),
        "source_platform": request["source_platform"].strip(),
        "source_channel_id": request["source_channel_id"].strip(),
        "source_channel_type": request["source_channel_type"].strip(),
        "source_message_id": request["source_message_id"].strip(),
        "source_platform_bot_id": request["source_platform_bot_id"].strip(),
        "source_character_name": request["source_character_name"].strip(),
        "requester_global_user_id": request["requester_global_user_id"].strip(),
        "requester_platform_user_id": request[
            "requester_platform_user_id"
        ].strip(),
        "requester_display_name": request["requester_display_name"].strip(),
        "created_at": storage_timestamp_utc,
        "updated_at": storage_timestamp_utc,
        "lease_owner": None,
        "lease_expires_at": None,
        "attempt_count": 0,
        "max_attempts": BACKGROUND_WORK_WORKER_MAX_ATTEMPTS,
        "requested_worker": request["requested_worker"],
        "worker_payload": dict(request["worker_payload"]),
        "task_execution_context": context_projection,
        "task_resolution_result": {},
        "artifact_text": "",
        "failure_summary": "",
        "result_summary": "",
        "completed_at": "",
        "delivery_attempt_count": 0,
        "delivery_failure_summary": "",
        "delivery_tracking_id": "",
        "delivered_conversation_message_id": "",
        "delivered_at": "",
    }
    return job


def _queue_result_from_job(job: BackgroundWorkJobDoc) -> BackgroundWorkQueueResult:
    """Project durable creation into a queue-internal confirmation result."""

    job_id = _job_text(job, "job_id")
    return {
        "status": "pending",
        "job_id": job_id,
        "job_ref": background_work_job_ref(job_id),
        "accepted_task_id": _job_text(job, "accepted_task_id"),
        "task_identity_key": _job_text(job, "task_identity_key"),
        "accepted_task_summary": _job_text(job, "semantic_objective"),
        "acknowledgement_constraint": "promise_allowed",
        "wait_guidance": "non_numeric_wait",
        "result_summary": "Accepted task continuation is durable.",
    }


def _job_text(job: Mapping[str, object], field_name: str) -> str:
    """Return one trusted job text field."""

    value = job.get(field_name)
    if not isinstance(value, str):
        return ""
    return value.strip()
