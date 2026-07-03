"""Runtime worker loop for generic background work."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any
from uuid import uuid4

from kazusa_ai_chatbot.accepted_task import (
    mark_accepted_task_delivered,
    mark_accepted_task_failure_ready,
    mark_accepted_task_result_ready,
    mark_accepted_task_running,
)
from kazusa_ai_chatbot.background_work.providers import dispatch_background_work
from kazusa_ai_chatbot.background_work.router import route_background_work
from kazusa_ai_chatbot.background_work.subagent import worker_descriptions
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
    """Claim and process a bounded batch of queued background-work jobs."""

    if worker_id is None:
        worker_id = f"background-work-worker-{uuid4().hex}"
    processed_count = 0
    succeeded_count = 0
    failed_count = 0
    for _ in range(max(0, int(claim_limit))):
        now_utc = storage_utc_now_iso()
        job = await claim_background_work_job(
            lease_owner=worker_id,
            lease_seconds=lease_seconds,
            now_utc=now_utc,
            max_attempts=max_attempts,
        )
        if job is None:
            break
        processed_count += 1
        task_brief = _job_text(job, "task_brief")
        source_summary = _job_text(job, "source_context")
        if not source_summary:
            source_summary = task_brief
        accepted_task_id = _job_text(job, "accepted_task_id")
        if accepted_task_id:
            await mark_accepted_task_running(
                accepted_task_id=accepted_task_id,
                started_at=now_utc,
            )
        try:
            requested_worker = _job_text(job, "requested_worker")
            if requested_worker:
                router_decision = _requested_worker_decision(
                    job,
                    requested_worker=requested_worker,
                    task_brief=task_brief,
                    source_summary=source_summary,
                )
                worker_decision = dict(router_decision)
            else:
                router_decision = await route_background_work(
                    task_brief=task_brief,
                    source_summary=source_summary,
                    worker_descriptions=_routable_worker_descriptions(),
                    max_output_chars=int(job["max_output_chars"]),
                )
                worker_decision = dict(router_decision)
                worker_decision["task_brief"] = task_brief
                worker_decision["source_summary"] = source_summary
            worker_result = await dispatch_background_work(
                worker_decision,
                max_output_chars=int(job["max_output_chars"]),
            )
        except Exception as exc:
            logger.exception(
                f"Background-work job {job['job_id']} failed during "
                f"routing or dispatch: {exc}"
            )
            await fail_background_work_job(
                job_id=job["job_id"],
                lease_owner=worker_id,
                failure_summary=f"Unhandled error during routing or dispatch: {type(exc).__name__}: {str(exc)[:200]}",
                failed_at=storage_utc_now_iso(),
            )
            if accepted_task_id:
                await mark_accepted_task_failure_ready(
                    accepted_task_id=accepted_task_id,
                    failure_summary=(
                        "Unhandled error during routing or dispatch: "
                        f"{type(exc).__name__}: {str(exc)[:200]}"
                    ),
                    completed_at=storage_utc_now_iso(),
                )
            failed_count += 1
            continue
        completed_at = storage_utc_now_iso()
        if worker_result["status"] == "succeeded":
            skip_result_delivery = _skip_result_delivery(worker_result)
            await complete_background_work_job(
                job_id=job["job_id"],
                lease_owner=worker_id,
                router_action=router_decision["action"],
                worker=worker_result["worker"],
                routed_task=source_summary,
                router_reason=router_decision["reason"],
                artifact_text=worker_result["artifact_text"],
                result_summary=worker_result["result_summary"],
                worker_metadata=worker_result["worker_metadata"],
                completed_at=completed_at,
                skip_result_delivery=skip_result_delivery,
            )
            if accepted_task_id:
                if skip_result_delivery:
                    await mark_accepted_task_delivered(
                        accepted_task_id=accepted_task_id,
                        delivered_conversation_message_id="",
                        delivered_at=completed_at,
                    )
                else:
                    await mark_accepted_task_result_ready(
                        accepted_task_id=accepted_task_id,
                        artifact_text=worker_result["artifact_text"],
                        result_summary=worker_result["result_summary"],
                        completed_at=completed_at,
                    )
            succeeded_count += 1
        else:
            await fail_background_work_job(
                job_id=job["job_id"],
                lease_owner=worker_id,
                router_action=router_decision["action"],
                worker=worker_result["worker"],
                routed_task=source_summary,
                router_reason=router_decision["reason"],
                failure_summary=worker_result["failure_summary"],
                result_summary=worker_result["result_summary"],
                worker_metadata=worker_result["worker_metadata"],
                failed_at=completed_at,
            )
            if accepted_task_id:
                await mark_accepted_task_failure_ready(
                    accepted_task_id=accepted_task_id,
                    failure_summary=worker_result["failure_summary"],
                    completed_at=completed_at,
                )
            failed_count += 1

    result = {
        "processed_count": processed_count,
        "succeeded_count": succeeded_count,
        "failed_count": failed_count,
    }
    return result


def _requested_worker_decision(
    job: Mapping[str, Any],
    *,
    requested_worker: str,
    task_brief: str,
    source_summary: str,
) -> dict[str, object]:
    """Build the deterministic worker decision for bound background actions."""

    worker_payload = _job_mapping(job, "worker_payload")
    worker_payload["source_action_attempt_id"] = _job_text(
        job,
        "source_action_attempt_id",
    )
    worker_payload["source_scope"] = _source_scope_from_job(job)
    storage_timestamp_utc = _job_text(job, "updated_at")
    if not storage_timestamp_utc:
        storage_timestamp_utc = _job_text(job, "created_at")
    if storage_timestamp_utc:
        worker_payload["storage_timestamp_utc"] = storage_timestamp_utc
    decision: dict[str, object] = {
        "action": "execute",
        "worker": requested_worker,
        "reason": "Validated background action requested this worker.",
        "task_brief": task_brief,
        "source_summary": source_summary,
        "worker_payload": worker_payload,
    }
    return decision


def _routable_worker_descriptions() -> dict[str, str]:
    """Return workers that can run without deterministic worker payloads."""

    descriptions = worker_descriptions()
    descriptions.pop("future_speak", None)
    return descriptions


def _skip_result_delivery(worker_result: Mapping[str, Any]) -> bool:
    """Return whether a worker result has its own user-visible follow-up path."""

    worker_metadata = worker_result.get("worker_metadata")
    if not isinstance(worker_metadata, Mapping):
        return_value = False
        return return_value
    return_value = worker_metadata.get("skip_result_delivery") is True
    return return_value


def _source_scope_from_job(job: Mapping[str, Any]) -> dict[str, object]:
    """Project trusted delivery scope into the future-cognition source scope."""

    source_scope: dict[str, object] = {
        "source_platform": _job_text(job, "source_platform"),
        "source_channel_id": _job_text(job, "source_channel_id"),
        "source_channel_type": _job_text(job, "source_channel_type"),
        "source_user_id": _job_text(job, "requester_global_user_id"),
        "source_platform_bot_id": _job_text(job, "source_platform_bot_id"),
        "source_character_name": _job_text(job, "source_character_name"),
        "source_message_id": _job_text(job, "source_message_id"),
    }
    return source_scope


def _job_mapping(
    job: Mapping[str, Any],
    field_name: str,
) -> dict[str, object]:
    """Return one trusted job mapping as a plain dict."""

    value = job.get(field_name)
    if not isinstance(value, Mapping):
        return_value: dict[str, object] = {}
        return return_value
    return_value = dict(value)
    return return_value


def _job_text(job: Mapping[str, Any], field_name: str) -> str:
    """Return one trusted job text field."""

    value = job.get(field_name)
    if not isinstance(value, str):
        return_value = ""
        return return_value
    return_value = value.strip()
    return return_value
