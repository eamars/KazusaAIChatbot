"""Public queue helpers for generic background-work requests."""

from __future__ import annotations

from uuid import uuid4

from kazusa_ai_chatbot.background_work.models import (
    BACKGROUND_WORK_JOB_REF_OWNER,
    BACKGROUND_WORK_REQUESTED_DELIVERY,
    BackgroundWorkJobDoc,
    BackgroundWorkQueueRequest,
    BackgroundWorkQueueResult,
    background_work_job_ref,
)
from kazusa_ai_chatbot.config import (
    BACKGROUND_WORK_INPUT_CHAR_LIMIT,
    BACKGROUND_WORK_OUTPUT_CHAR_LIMIT,
    BACKGROUND_WORK_WORKER_MAX_ATTEMPTS,
)
from kazusa_ai_chatbot.db.background_work_jobs import insert_background_work_job
from kazusa_ai_chatbot.time_boundary import normalize_storage_utc_iso

_WORKER_LOCAL_QUEUE_FIELDS = frozenset((
    "worker",
    "work_kind",
    "task_type",
    "tool_args",
    "artifact_text",
))


async def enqueue_background_work_request(
    request: BackgroundWorkQueueRequest,
) -> BackgroundWorkQueueResult:
    """Validate and persist one generic background-work request."""

    _validate_queue_request(request)
    storage_timestamp_utc = normalize_storage_utc_iso(
        request["storage_timestamp_utc"],
    )
    job_id = f"job-{uuid4().hex}"
    job = _build_job_document(
        request,
        job_id=job_id,
        storage_timestamp_utc=storage_timestamp_utc,
    )
    stored_job = await insert_background_work_job(job)
    result = _queue_result_from_job(
        stored_job,
        observed_at=storage_timestamp_utc,
    )
    return result


def _validate_queue_request(request: BackgroundWorkQueueRequest) -> None:
    """Validate live-path queue semantics before persistence."""

    leaked_fields = sorted(
        field_name for field_name in request if field_name in _WORKER_LOCAL_QUEUE_FIELDS
    )
    if leaked_fields:
        raise ValueError(
            "worker-local fields are not allowed in background work queue "
            f"requests: {', '.join(leaked_fields)}"
        )

    for field_name in (
        "action_attempt_id",
        "idempotency_key",
        "task_brief",
        "storage_timestamp_utc",
    ):
        if not request[field_name].strip():
            raise ValueError(f"{field_name} is required")

    if request["requested_delivery"] != BACKGROUND_WORK_REQUESTED_DELIVERY:
        raise ValueError("requested_delivery is not supported")
    if len(request["task_brief"]) > BACKGROUND_WORK_INPUT_CHAR_LIMIT:
        raise ValueError("task_brief exceeds background work input limit")
    if request["max_output_chars"] > BACKGROUND_WORK_OUTPUT_CHAR_LIMIT:
        raise ValueError("max_output_chars exceeds configured output limit")
    if request["max_output_chars"] < 1:
        raise ValueError("max_output_chars must be positive")


def _build_job_document(
    request: BackgroundWorkQueueRequest,
    *,
    job_id: str,
    storage_timestamp_utc: str,
) -> BackgroundWorkJobDoc:
    """Build the durable job row from a validated queue request."""

    job: BackgroundWorkJobDoc = {
        "schema_version": "background_work_job.v1",
        "job_id": job_id,
        "idempotency_key": request["idempotency_key"],
        "source_action_attempt_id": request["action_attempt_id"],
        "status": "queued",
        "delivery_state": "queued",
        "task_brief": request["task_brief"].strip(),
        "requested_delivery": request["requested_delivery"],
        "max_output_chars": int(request["max_output_chars"]),
        "source_platform": request["source_platform"].strip(),
        "source_channel_id": request["source_channel_id"].strip(),
        "source_channel_type": request["source_channel_type"].strip(),
        "source_message_id": request["source_message_id"].strip(),
        "source_platform_bot_id": request["source_platform_bot_id"].strip(),
        "source_character_name": request["source_character_name"].strip(),
        "requester_global_user_id": request["requester_global_user_id"].strip(),
        "requester_platform_user_id": (
            request["requester_platform_user_id"].strip()
        ),
        "requester_display_name": request["requester_display_name"].strip(),
        "created_at": storage_timestamp_utc,
        "updated_at": storage_timestamp_utc,
        "lease_owner": None,
        "lease_expires_at": None,
        "attempt_count": 0,
        "max_attempts": BACKGROUND_WORK_WORKER_MAX_ATTEMPTS,
        "router_action": "",
        "worker": "",
        "routed_task": "",
        "router_reason": "",
        "artifact_text": "",
        "artifact_char_count": 0,
        "failure_summary": "",
        "result_summary": "",
        "worker_metadata": {},
        "completed_at": "",
        "delivery_attempt_count": 0,
        "delivery_tracking_id": "",
        "delivered_conversation_message_id": "",
        "delivered_at": "",
    }
    return job


def _queue_result_from_job(
    job: BackgroundWorkJobDoc,
    *,
    observed_at: str,
) -> BackgroundWorkQueueResult:
    """Project one durable job row into a prompt-safe pending result."""

    job_ref = background_work_job_ref(job["job_id"])
    evidence_ref: dict[str, str] = {
        "schema_version": "evidence_ref.v1",
        "evidence_kind": "system_event",
        "evidence_id": job_ref,
        "owner": BACKGROUND_WORK_JOB_REF_OWNER,
        "excerpt": "queued background work request",
        "observed_at": observed_at,
    }
    result: BackgroundWorkQueueResult = {
        "status": "pending",
        "queue_state": "queued",
        "job_id": job["job_id"],
        "job_ref": job_ref,
        "task_summary": job["task_brief"],
        "result_summary": "Background work job queued.",
        "operational_owner": BACKGROUND_WORK_JOB_REF_OWNER,
        "acknowledgement_constraint": "promise_allowed",
        "evidence_ref": evidence_ref,
    }
    return result
