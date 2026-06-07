"""Real LLM checks for the background artifact worker."""

from __future__ import annotations

import json
import logging

import httpx
import pytest

from kazusa_ai_chatbot.background_artifact import worker as worker_module
from kazusa_ai_chatbot.background_artifact.prompts import (
    BACKGROUND_ARTIFACT_WORKER_PROMPT,
    build_background_artifact_worker_payload,
)
from kazusa_ai_chatbot.config import BACKGROUND_ARTIFACT_LLM_BASE_URL
from tests.llm_trace import write_llm_trace

pytestmark = [pytest.mark.asyncio, pytest.mark.live_llm]

logger = logging.getLogger(__name__)


async def test_background_artifact_worker_live_case() -> None:
    """Run one bounded artifact job through the real worker LLM."""

    await _skip_if_worker_llm_unavailable()
    job = {
        "job_id": "live-worker-fixture-001",
        "work_kind": "coding_snippet",
        "objective": (
            "Generate a small Python fibonacci(n) function that returns a "
            "list of the first n Fibonacci numbers."
        ),
        "input_summary": (
            "The user asked for a bounded Fibonacci sequence generator. "
            "Provide a self-contained code snippet only."
        ),
        "max_output_chars": 1200,
    }
    prompt_payload = build_background_artifact_worker_payload(
        work_kind=job["work_kind"],
        objective=job["objective"],
        input_summary=job["input_summary"],
        max_output_chars=job["max_output_chars"],
    )

    result = await worker_module._run_job(job)
    trace_path = write_llm_trace(
        "background_artifact_worker_live_llm",
        "coding_snippet_fibonacci",
        {
            "case_id": "coding_snippet_fibonacci",
            "route_base_url": BACKGROUND_ARTIFACT_LLM_BASE_URL,
            "system_prompt": BACKGROUND_ARTIFACT_WORKER_PROMPT,
            "prompt_payload": prompt_payload,
            "normalized_result": result,
            "judgment": "manual_review_required_for_worker_artifact_quality",
        },
    )
    logger.info(
        "BACKGROUND_ARTIFACT_WORKER_LIVE trace=%s result=%s",
        trace_path,
        json.dumps(result, ensure_ascii=True),
    )

    assert result["status"] == "succeeded"
    assert result["artifact_text"].strip()
    assert len(result["artifact_text"]) <= job["max_output_chars"]
    assert result["failure_summary"] == ""


async def _skip_if_worker_llm_unavailable() -> None:
    """Skip when the configured worker LLM endpoint is unavailable."""

    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(
                f"{BACKGROUND_ARTIFACT_LLM_BASE_URL.rstrip('/')}/models"
            )
    except httpx.HTTPError as exc:
        pytest.skip(
            f"LLM endpoint is unavailable: "
            f"{BACKGROUND_ARTIFACT_LLM_BASE_URL}; {exc}"
        )

    if response.status_code >= 500:
        pytest.skip(
            f"LLM endpoint returned server error {response.status_code}: "
            f"{BACKGROUND_ARTIFACT_LLM_BASE_URL}"
        )
