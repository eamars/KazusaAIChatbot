"""Runtime worker loop for generic background work."""

from __future__ import annotations

import logging
from typing import Any
from uuid import uuid4

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
        source_summary = job.get("source_context", "").strip() or job["task_brief"]
        try:
            router_decision = await route_background_work(
                task_brief=job["task_brief"],
                source_summary=source_summary,
                worker_descriptions=worker_descriptions(),
                max_output_chars=int(job["max_output_chars"]),
            )
            worker_decision = dict(router_decision)
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
            failed_count += 1
            continue
        completed_at = storage_utc_now_iso()
        if worker_result["status"] == "succeeded":
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
            failed_count += 1

    result = {
        "processed_count": processed_count,
        "succeeded_count": succeeded_count,
        "failed_count": failed_count,
    }
    return result
