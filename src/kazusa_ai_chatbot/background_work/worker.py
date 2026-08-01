"""Leased worker loop for reviewed v2 background-work payloads."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from importlib import import_module
from typing import Literal
from uuid import uuid4

from kazusa_ai_chatbot.accepted_task import (
    mark_accepted_task_failure_ready,
    mark_future_speak_accepted_task_delivered,
    mark_accepted_task_running,
    mark_tool_result_ready,
)
from kazusa_ai_chatbot.background_work.models import (
    FUTURE_SPEAK_WORKER,
    TASK_ORCHESTRATOR_WORKER,
)
from kazusa_ai_chatbot.background_work.subagent.task_orchestrator import (
    execute_task_orchestrator_job,
)
from kazusa_ai_chatbot.config import (
    BACKGROUND_WORK_WORKER_CLAIM_LIMIT,
    BACKGROUND_WORK_WORKER_LEASE_SECONDS,
    BACKGROUND_WORK_WORKER_MAX_ATTEMPTS,
)
from kazusa_ai_chatbot.db.background_work_jobs import (
    claim_background_work_job,
    complete_background_work_job,
    fail_background_work_job,
)
from kazusa_ai_chatbot.db.errors import DatabaseOperationError
from kazusa_ai_chatbot.task_resolution.contracts import (
    TaskResolutionContractError,
    TaskResolutionResultV1,
)
from kazusa_ai_chatbot.time_boundary import storage_utc_now_iso


logger = logging.getLogger(__name__)

BACKGROUND_WORK_WORKER_COMPONENT = "background_work.worker"


async def run_background_work_worker_tick(
    *,
    claim_limit: int = BACKGROUND_WORK_WORKER_CLAIM_LIMIT,
    lease_seconds: int = BACKGROUND_WORK_WORKER_LEASE_SECONDS,
    max_attempts: int = BACKGROUND_WORK_WORKER_MAX_ATTEMPTS,
    worker_id: str | None = None,
) -> dict[str, int]:
    """Claim and process a bounded batch of reviewed v2 background jobs.

    Args:
        claim_limit: Maximum jobs processed during this runtime tick.
        lease_seconds: Durable claim lease for each job.
        max_attempts: Maximum process retries allowed by the queue.
        worker_id: Stable test or runtime owner token. A unique token is
            created when the caller does not provide one.

    Returns:
        Counts of claimed jobs that reached a terminal worker disposition.
    """

    lease_token = worker_id or f"background-work-worker-{uuid4().hex}"
    processed_count = 0
    succeeded_count = 0
    failed_count = 0
    for _ in range(max(0, int(claim_limit))):
        claimed_at = storage_utc_now_iso()
        job = await claim_background_work_job(
            lease_owner=lease_token,
            lease_seconds=lease_seconds,
            now_utc=claimed_at,
            max_attempts=max_attempts,
        )
        if job is None:
            break
        processed_count += 1
        accepted_task_id = _job_text(job, "accepted_task_id")
        if accepted_task_id:
            running_task = await mark_accepted_task_running(
                accepted_task_id=accepted_task_id,
                started_at=claimed_at,
            )
            if running_task is None:
                await fail_background_work_job(
                    job_id=job["job_id"],
                    lease_owner=lease_token,
                    failure_summary=(
                        "The accepted task was not ready for execution."
                    ),
                    result_summary=(
                        "The accepted task could not enter running state."
                    ),
                    failed_at=claimed_at,
                    skip_result_delivery=True,
                )
                failed_count += 1
                continue
        try:
            disposition = await _run_claimed_job(
                job,
                lease_owner=lease_token,
            )
        except Exception as exc:
            logger.exception(
                f"Background-work job {job['job_id']} could not complete: {exc}",
            )
            failed_at = storage_utc_now_iso()
            await fail_background_work_job(
                job_id=job["job_id"],
                lease_owner=lease_token,
                failure_summary="The background task could not complete.",
                result_summary="The accepted task could not complete.",
                failed_at=failed_at,
            )
            if accepted_task_id:
                await mark_accepted_task_failure_ready(
                    accepted_task_id=accepted_task_id,
                    failure_summary="The accepted task could not complete.",
                    completed_at=failed_at,
                )
            failed_count += 1
            continue
        if disposition == "completed":
            succeeded_count += 1
        else:
            failed_count += 1

    result = {
        "processed_count": processed_count,
        "succeeded_count": succeeded_count,
        "failed_count": failed_count,
    }
    return result


async def _run_claimed_job(
    job: Mapping[str, object],
    *,
    lease_owner: str,
) -> Literal["completed", "failed"]:
    """Execute one validated v2 payload through its deterministic worker."""

    requested_worker = _job_text(job, "requested_worker")
    if requested_worker == TASK_ORCHESTRATOR_WORKER:
        result = await execute_task_orchestrator_job(
            job,
            lease_owner=lease_owner,
        )
        await _complete_task_orchestrator_job(
            job,
            lease_owner=lease_owner,
            result=result,
        )
        return "completed"
    if requested_worker == FUTURE_SPEAK_WORKER:
        result = await _execute_future_speak_job(job)
        completed_at = storage_utc_now_iso()
        accepted_task_id = _job_text(job, "accepted_task_id")
        if accepted_task_id:
            delivered_task = await mark_future_speak_accepted_task_delivered(
                accepted_task_id=accepted_task_id,
                delivered_at=completed_at,
            )
            if delivered_task is None:
                raise DatabaseOperationError(
                    "future-speak accepted task transition is missing"
                )
        completed_job = await complete_background_work_job(
            job_id=_required_job_text(job, "job_id"),
            lease_owner=lease_owner,
            task_resolution_result=None,
            artifact_text=result["artifact_text"],
            result_summary=result["result_summary"],
            completed_at=completed_at,
            skip_result_delivery=True,
        )
        if completed_job is None:
            raise DatabaseOperationError(
                "future-speak job completion lost its worker lease"
            )
        return "completed"
    raise ValueError("requested_worker is not supported")


async def _complete_task_orchestrator_job(
    job: Mapping[str, object],
    *,
    lease_owner: str,
    result: TaskResolutionResultV1,
) -> None:
    """Persist terminal task resolution and its accepted-task delivery state."""

    completed_at = storage_utc_now_iso()
    summary = result["prompt_safe_summary"]
    artifact_text = _bounded_artifact_text(job, summary)
    result_status = result["status"]
    if result_status not in {
        "resolved",
        "partial",
        "needs_user_input",
        "approval_required",
        "unavailable",
        "failed",
    }:
        raise TaskResolutionContractError(
            "background task resolution must return a terminal result"
        )
    accepted_task_id = _job_text(job, "accepted_task_id")
    if accepted_task_id:
        if result_status in {"resolved", "partial"}:
            accepted_task = await mark_tool_result_ready(
                accepted_task_id=accepted_task_id,
                artifact_text=artifact_text,
                result_summary=summary,
                completed_at=completed_at,
                result_kind=result_status,
                completion_status=result_status,
                remaining_needs=list(result["remaining_needs"]),
                coding_run_context=dict(result["coding_run_context"]),
            )
        else:
            accepted_task = await mark_accepted_task_failure_ready(
                accepted_task_id=accepted_task_id,
                failure_summary=summary,
                completed_at=completed_at,
                result_kind=result_status,
                remaining_needs=list(result["remaining_needs"]),
                coding_run_context=dict(result["coding_run_context"]),
            )
        if accepted_task is None:
            raise DatabaseOperationError(
                "task-resolution accepted task transition is missing"
            )
    completed_job = await complete_background_work_job(
        job_id=_required_job_text(job, "job_id"),
        lease_owner=lease_owner,
        task_resolution_result=result,
        artifact_text=artifact_text,
        result_summary=summary,
        completed_at=completed_at,
    )
    if completed_job is None:
        raise DatabaseOperationError(
            "task-resolution job completion lost its worker lease"
        )


def _bounded_artifact_text(job: Mapping[str, object], summary: str) -> str:
    """Constrain the result artifact to the v2 job's declared output budget."""

    max_output_chars = job.get("max_output_chars")
    if not isinstance(max_output_chars, int) or max_output_chars < 1:
        raise ValueError("background job max_output_chars is invalid")
    artifact_text = summary[:max_output_chars]
    return artifact_text


def _job_text(job: Mapping[str, object], field_name: str) -> str:
    """Read one optional trusted text field from a validated v2 job."""

    value = job.get(field_name)
    if not isinstance(value, str):
        return ""
    text = value.strip()
    return text


def _required_job_text(job: Mapping[str, object], field_name: str) -> str:
    """Require one non-empty trusted text field from a validated v2 job."""

    text = _job_text(job, field_name)
    if not text:
        raise ValueError(f"background job {field_name} is required")
    return text


async def _execute_future_speak_job(
    job: Mapping[str, object],
) -> dict[str, str]:
    """Load deterministic scheduling only for a claimed future-speak job."""

    module = import_module(
        "kazusa_ai_chatbot.background_work.subagent.future_speak",
    )
    execute_job = getattr(module, "execute_future_speak_job")
    result = await execute_job(job)
    return result
