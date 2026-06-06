"""Real LLM quality check for the background-work router."""

from __future__ import annotations

import json
import logging
import sys

import httpx
import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from kazusa_ai_chatbot.background_work.router import (
    BACKGROUND_WORK_ROUTER_PROMPT,
    _background_work_router_llm,
    build_background_work_router_payload,
    normalize_background_work_router_output,
)
from kazusa_ai_chatbot.background_work.subagent.text_artifact import (
    DESCRIPTION as TEXT_ARTIFACT_DESCRIPTION,
)
from kazusa_ai_chatbot.config import BACKGROUND_WORK_LLM_BASE_URL
from kazusa_ai_chatbot.utils import parse_llm_json_output
from tests.llm_trace import write_llm_trace

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

pytestmark = [pytest.mark.asyncio, pytest.mark.live_llm]

logger = logging.getLogger(__name__)


async def test_background_work_router_live_case() -> None:
    """Route one bounded text-artifact request with the real local LLM."""

    await _skip_if_llm_unavailable()
    payload = build_background_work_router_payload(
        task_brief=(
            "Generate a compact Python function that returns the first n "
            "Fibonacci numbers."
        ),
        source_summary=(
            "The user asked for a bounded Fibonacci sequence generator and "
            "expects the result later."
        ),
        worker_descriptions={"text_artifact": TEXT_ARTIFACT_DESCRIPTION},
        max_output_chars=3000,
    )
    response = await _background_work_router_llm.ainvoke([
        SystemMessage(content=BACKGROUND_WORK_ROUTER_PROMPT),
        HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
    ])
    raw_output = str(response.content)
    parsed = parse_llm_json_output(raw_output)
    decision = normalize_background_work_router_output(parsed)
    leakage_errors = _router_leakage_errors(decision)
    trace_path = write_llm_trace(
        "background_work_router_live_llm",
        "fibonacci_text_artifact_route",
        {
            "prompt": BACKGROUND_WORK_ROUTER_PROMPT,
            "prompt_payload": payload,
            "raw_output": raw_output,
            "parsed_output": parsed,
            "normalized_decision": decision,
            "leakage_errors": leakage_errors,
            "judgment": "manual_review_required_for_worker_route_quality",
        },
    )
    logger.info(
        f"BACKGROUND_WORK_ROUTER_LIVE trace={trace_path} "
        f"decision={json.dumps(decision, ensure_ascii=True)}"
    )

    assert decision["action"] == "execute"
    assert decision["worker"] == "text_artifact"
    assert decision["task"]
    assert decision["reason"]
    assert leakage_errors == []


async def _skip_if_llm_unavailable() -> None:
    """Skip when the configured background-work LLM endpoint is unavailable."""

    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(
                f"{BACKGROUND_WORK_LLM_BASE_URL.rstrip('/')}/models"
            )
    except httpx.HTTPError as exc:
        pytest.skip(
            "Background-work LLM endpoint is unavailable: "
            f"{BACKGROUND_WORK_LLM_BASE_URL}; {exc}"
        )

    if response.status_code >= 500:
        pytest.skip(
            "Background-work LLM endpoint returned server error "
            f"{response.status_code}: {BACKGROUND_WORK_LLM_BASE_URL}"
        )


def _router_leakage_errors(decision: dict[str, object]) -> list[str]:
    """Return worker-local fragments that leaked into router output."""

    serialized = json.dumps(decision, ensure_ascii=False).lower()
    errors: list[str] = []
    for fragment in (
        "work_kind",
        "task_type",
        "coding_snippet",
        "text_rewrite",
        "artifact_text",
        "tool_args",
        "adapter_id",
        "job_id",
    ):
        if fragment in serialized:
            errors.append(f"forbidden router fragment leaked: {fragment}")
    return errors
