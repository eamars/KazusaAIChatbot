"""Text-only background artifact worker."""

from __future__ import annotations

import json
from typing import Any
from uuid import uuid4

from langchain_core.messages import HumanMessage, SystemMessage

from kazusa_ai_chatbot.background_artifact.prompts import (
    BACKGROUND_ARTIFACT_WORKER_PROMPT,
    build_background_artifact_worker_payload,
)
from kazusa_ai_chatbot.config import (
    BACKGROUND_ARTIFACT_LLM_API_KEY,
    BACKGROUND_ARTIFACT_LLM_BASE_URL,
    BACKGROUND_ARTIFACT_LLM_MODEL,
    BACKGROUND_ARTIFACT_OUTPUT_CHAR_LIMIT,
    BACKGROUND_ARTIFACT_WORKER_CLAIM_LIMIT,
    BACKGROUND_ARTIFACT_WORKER_LEASE_SECONDS,
    BACKGROUND_ARTIFACT_WORKER_MAX_ATTEMPTS,
)
from kazusa_ai_chatbot.db import (
    claim_background_artifact_job,
    complete_background_artifact_job,
    fail_background_artifact_job,
)
from kazusa_ai_chatbot.time_boundary import storage_utc_now_iso
from kazusa_ai_chatbot.utils import get_llm, parse_llm_json_output

BACKGROUND_ARTIFACT_WORKER_COMPONENT = "background_artifact.worker"

_background_artifact_llm = get_llm(
    temperature=0.2,
    top_p=0.8,
    model=BACKGROUND_ARTIFACT_LLM_MODEL,
    base_url=BACKGROUND_ARTIFACT_LLM_BASE_URL,
    api_key=BACKGROUND_ARTIFACT_LLM_API_KEY,
)


async def run_background_artifact_worker_tick(
    *,
    claim_limit: int = BACKGROUND_ARTIFACT_WORKER_CLAIM_LIMIT,
    lease_seconds: int = BACKGROUND_ARTIFACT_WORKER_LEASE_SECONDS,
    max_attempts: int = BACKGROUND_ARTIFACT_WORKER_MAX_ATTEMPTS,
    worker_id: str | None = None,
) -> dict[str, int]:
    """Claim and process a bounded batch of queued text artifact jobs."""

    if worker_id is None:
        worker_id = f"background-artifact-worker-{uuid4().hex}"
    processed_count = 0
    succeeded_count = 0
    failed_count = 0
    for _index in range(max(0, int(claim_limit))):
        now_utc = storage_utc_now_iso()
        job = await claim_background_artifact_job(
            lease_owner=worker_id,
            lease_seconds=lease_seconds,
            now_utc=now_utc,
            max_attempts=max_attempts,
        )
        if job is None:
            break
        processed_count += 1
        worker_result = await _run_job(job)
        completed_at = storage_utc_now_iso()
        if worker_result["status"] == "succeeded":
            await complete_background_artifact_job(
                job_id=job["job_id"],
                lease_owner=worker_id,
                artifact_text=worker_result["artifact_text"],
                completed_at=completed_at,
            )
            succeeded_count += 1
        else:
            await fail_background_artifact_job(
                job_id=job["job_id"],
                lease_owner=worker_id,
                failure_summary=worker_result["failure_summary"],
                failed_at=completed_at,
            )
            failed_count += 1

    result = {
        "processed_count": processed_count,
        "succeeded_count": succeeded_count,
        "failed_count": failed_count,
    }
    return result


async def _run_job(job: dict[str, Any]) -> dict[str, str]:
    """Generate one text artifact and normalize the model response."""

    payload = build_background_artifact_worker_payload(
        work_kind=str(job["work_kind"]),
        objective=str(job["objective"]),
        input_summary=str(job["input_summary"]),
        max_output_chars=int(job["max_output_chars"]),
    )
    response = await _background_artifact_llm.ainvoke([
        SystemMessage(content=BACKGROUND_ARTIFACT_WORKER_PROMPT),
        HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
    ])
    parsed = parse_llm_json_output(response.content)
    result = _normalize_worker_result(
        parsed,
        max_output_chars=min(
            int(job["max_output_chars"]),
            BACKGROUND_ARTIFACT_OUTPUT_CHAR_LIMIT,
        ),
    )
    return result


def _normalize_worker_result(
    parsed: object,
    *,
    max_output_chars: int,
) -> dict[str, str]:
    """Normalize text artifact output to the first-scope contract."""

    if not isinstance(parsed, dict):
        result = {
            "status": "failed",
            "artifact_text": "",
            "failure_summary": "Artifact worker returned malformed output.",
        }
        return result

    status = parsed.get("status")
    artifact_text = parsed.get("artifact_text")
    if status == "succeeded" and isinstance(artifact_text, str):
        result = {
            "status": "succeeded",
            "artifact_text": artifact_text[:max_output_chars],
            "failure_summary": "",
        }
        return result

    failure_summary = parsed.get("failure_summary")
    if not isinstance(failure_summary, str) or not failure_summary.strip():
        failure_summary = "Artifact worker could not complete the request."
    result = {
        "status": "failed",
        "artifact_text": "",
        "failure_summary": failure_summary.strip(),
    }
    return result
