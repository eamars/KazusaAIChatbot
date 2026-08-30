"""Public v2 queue helpers for reviewed background-work payloads."""

from __future__ import annotations

from collections.abc import Mapping

from kazusa_ai_chatbot.background_work.models import (
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
from kazusa_ai_chatbot.cognition_episode import (
    CognitiveEpisodeValidationError,
    GoalContinuationRefV1,
    validate_goal_continuation_ref,
)
from kazusa_ai_chatbot.cognition_shared.contracts import (
    CognitionContractError,
    validate_scheduled_future_speech_authority,
)
from kazusa_ai_chatbot.config import (
    BACKGROUND_WORK_OUTPUT_CHAR_LIMIT,
    BACKGROUND_WORK_WORKER_MAX_ATTEMPTS,
)
from kazusa_ai_chatbot.db.background_work_jobs import insert_background_work_job
from kazusa_ai_chatbot.task_resolution.contracts import (
    validate_accepted_task_control,
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
    continuation_ref = _validate_goal_continuation_ref_field(request)
    if requested_worker == TASK_ORCHESTRATOR_WORKER:
        _validate_task_orchestrator_payload(
            worker_payload,
            request,
            continuation_ref=continuation_ref,
        )
        return
    if requested_worker == FUTURE_SPEAK_WORKER:
        _validate_future_speak_payload(worker_payload)
        _validate_scheduled_authority_carrier(request)
        return
    raise ValueError("requested_worker is not supported")


def _validate_scheduled_authority_carrier(
    request: BackgroundWorkQueueRequest,
) -> None:
    """Require and validate the immutable authority on future-speak jobs."""

    authority = request.get("scheduled_future_speech_authority")
    if authority is None:
        raise ValueError(
            "future_speak jobs require scheduled_future_speech_authority"
        )
    try:
        validate_scheduled_future_speech_authority(authority)
    except (CognitionContractError, ValueError) as exc:
        raise ValueError(
            f"scheduled_future_speech_authority is invalid: {exc}"
        ) from exc


def validate_task_orchestrator_worker_payload(
    value: object,
) -> dict[str, object]:
    """Validate the exact durable payload owned by task orchestration."""

    if not isinstance(value, Mapping):
        raise ValueError("task_orchestrator worker_payload must be a mapping")
    if set(value) != {
        "schema_version",
        "operation",
        "task_session_id",
        "operation_generation",
        "control",
    }:
        raise ValueError("task_orchestrator worker_payload fields are invalid")
    if value.get("schema_version") != TASK_ORCHESTRATOR_WORKER_PAYLOAD_VERSION:
        raise ValueError("task_orchestrator worker_payload schema_version is invalid")
    operation = value.get("operation")
    session_id = value.get("task_session_id")
    generation = value.get("operation_generation")
    control = value.get("control")
    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("task_orchestrator task_session_id is required")
    if (
        isinstance(generation, bool)
        or not isinstance(generation, int)
        or generation < 0
    ):
        raise ValueError("task_orchestrator operation_generation is invalid")
    if operation == "open_dsh_resolution":
        if control is not None or generation != 0:
            raise ValueError("open_dsh_resolution generation/control is invalid")
    elif operation == "continue_dsh_resolution":
        if control is not None:
            if not isinstance(control, Mapping):
                raise ValueError("continue_dsh_resolution control is invalid")
            try:
                validate_accepted_task_control(control)
            except ValueError as exc:
                raise ValueError(
                    f"continue_dsh_resolution control is invalid: {exc}"
                ) from exc
    else:
        raise ValueError("task_orchestrator worker_payload operation is unsupported")
    return dict(value)


def _validate_task_orchestrator_payload(
    value: object,
    request: BackgroundWorkQueueRequest,
    *,
    continuation_ref: GoalContinuationRefV1 | None,
) -> None:
    """Validate a task payload and its persisted typed execution context.

    Every task-resolution job fails closed without the exact validated
    continuation reference and typed task execution context.
    """

    validate_task_orchestrator_worker_payload(value)
    if continuation_ref is None:
        raise ValueError(
            "task-resolution jobs require goal_continuation_ref"
        )
    execution_context = request.get("task_execution_context")
    if not isinstance(execution_context, Mapping):
        raise ValueError(
            "task_execution_context is required for task-resolution jobs"
        )
    validated_context = validate_task_resolution_execution_context(
        execution_context
    )
    if validated_context["goal_continuation_ref"] != continuation_ref:
        raise ValueError(
            "goal_continuation_ref conflicts with task execution context"
        )


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


def _validate_goal_continuation_ref_field(
    request: BackgroundWorkQueueRequest,
) -> GoalContinuationRefV1 | None:
    """Validate the explicit continuation reference on every queue request.

    Task-resolution jobs require a non-null validated reference; future-speak
    and private unrelated actions carry an explicit null reference.
    """

    if "goal_continuation_ref" not in request:
        raise ValueError("goal_continuation_ref is required")
    raw_ref = request["goal_continuation_ref"]
    if raw_ref is None:
        return None
    try:
        continuation_ref = validate_goal_continuation_ref(raw_ref)
    except CognitiveEpisodeValidationError as exc:
        raise ValueError(
            f"goal_continuation_ref: invalid reference: {exc}"
        ) from exc
    return continuation_ref


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
    scheduled_authority = request.get("scheduled_future_speech_authority")
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
        "goal_continuation_ref": request["goal_continuation_ref"],
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
    if isinstance(scheduled_authority, dict):
        job["scheduled_future_speech_authority"] = dict(
            scheduled_authority
        )
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
