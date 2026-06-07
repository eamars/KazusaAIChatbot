"""Public queue helpers for background artifact requests."""

from __future__ import annotations

from uuid import uuid4

from kazusa_ai_chatbot.background_artifact.models import (
    BACKGROUND_ARTIFACT_JOB_REF_OWNER,
    BACKGROUND_ARTIFACT_REQUESTED_DELIVERY,
    BACKGROUND_ARTIFACT_WORK_KINDS,
    BackgroundArtifactJobDoc,
    BackgroundArtifactQueueRequest,
    BackgroundArtifactQueueResult,
    background_artifact_job_ref,
)
from kazusa_ai_chatbot.config import (
    BACKGROUND_ARTIFACT_INPUT_CHAR_LIMIT,
    BACKGROUND_ARTIFACT_OUTPUT_CHAR_LIMIT,
    BACKGROUND_ARTIFACT_WORKER_MAX_ATTEMPTS,
)
from kazusa_ai_chatbot.db import insert_background_artifact_job
from kazusa_ai_chatbot.time_boundary import normalize_storage_utc_iso


async def enqueue_background_artifact_request(
    request: BackgroundArtifactQueueRequest,
) -> BackgroundArtifactQueueResult:
    """Validate and persist one background artifact request."""

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
    stored_job = await insert_background_artifact_job(job)
    result = _queue_result_from_job(
        stored_job,
        observed_at=storage_timestamp_utc,
    )
    return result


def _validate_queue_request(request: BackgroundArtifactQueueRequest) -> None:
    """Validate first-scope queue semantics before persistence."""

    for field_name in (
        "action_attempt_id",
        "idempotency_key",
        "objective",
        "input_summary",
        "storage_timestamp_utc",
    ):
        if not request[field_name].strip():
            raise ValueError(f"{field_name} is required")

    if request["work_kind"] not in BACKGROUND_ARTIFACT_WORK_KINDS:
        raise ValueError("work_kind is not supported")
    if request["requested_delivery"] != BACKGROUND_ARTIFACT_REQUESTED_DELIVERY:
        raise ValueError("requested_delivery is not supported")
    if len(request["objective"]) > BACKGROUND_ARTIFACT_INPUT_CHAR_LIMIT:
        raise ValueError("objective exceeds background artifact input limit")
    if len(request["input_summary"]) > BACKGROUND_ARTIFACT_INPUT_CHAR_LIMIT:
        raise ValueError("input_summary exceeds background artifact input limit")
    if request["max_output_chars"] > BACKGROUND_ARTIFACT_OUTPUT_CHAR_LIMIT:
        raise ValueError("max_output_chars exceeds configured output limit")
    if request["max_output_chars"] < 1:
        raise ValueError("max_output_chars must be positive")


def _build_job_document(
    request: BackgroundArtifactQueueRequest,
    *,
    job_id: str,
    storage_timestamp_utc: str,
) -> BackgroundArtifactJobDoc:
    """Build the durable job row from a validated queue request."""

    job: BackgroundArtifactJobDoc = {
        "schema_version": "background_artifact_job.v1",
        "job_id": job_id,
        "idempotency_key": request["idempotency_key"],
        "source_action_attempt_id": request["action_attempt_id"],
        "status": "queued",
        "delivery_state": "queued",
        "work_kind": request["work_kind"],
        "objective": request["objective"].strip(),
        "input_summary": request["input_summary"].strip(),
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
        "max_attempts": BACKGROUND_ARTIFACT_WORKER_MAX_ATTEMPTS,
        "artifact_text": "",
        "artifact_char_count": 0,
        "failure_summary": "",
        "completed_at": "",
        "delivery_attempt_count": 0,
        "delivery_failure_summary": "",
        "delivery_tracking_id": "",
        "delivered_conversation_message_id": "",
        "delivered_at": "",
    }
    return job


def _queue_result_from_job(
    job: BackgroundArtifactJobDoc,
    *,
    observed_at: str,
) -> BackgroundArtifactQueueResult:
    """Project one durable job row into a prompt-safe pending result."""

    job_ref = background_artifact_job_ref(job["job_id"])
    evidence_ref: dict[str, str] = {
        "schema_version": "evidence_ref.v1",
        "evidence_kind": "system_event",
        "evidence_id": job_ref,
        "owner": BACKGROUND_ARTIFACT_JOB_REF_OWNER,
        "excerpt": f"queued {job['work_kind']} artifact request",
        "observed_at": observed_at,
    }
    result: BackgroundArtifactQueueResult = {
        "status": "pending",
        "queue_state": "queued",
        "job_id": job["job_id"],
        "job_ref": job_ref,
        "work_kind": job["work_kind"],
        "objective_summary": job["objective"],
        "result_summary": "Background artifact job queued.",
        "operational_owner": BACKGROUND_ARTIFACT_JOB_REF_OWNER,
        "acknowledgement_constraint": "promise_allowed",
        "evidence_ref": evidence_ref,
    }
    return result
