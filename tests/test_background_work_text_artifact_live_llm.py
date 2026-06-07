"""Real LLM quality check for the text-artifact worker stages."""

from __future__ import annotations

import json
import logging
import sys

import httpx
import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from kazusa_ai_chatbot.background_work.subagent.text_artifact import (
    TEXT_ARTIFACT_GENERATOR_PROMPT,
    TEXT_ARTIFACT_TASK_ROUTER_PROMPT,
    _text_artifact_generator_llm,
    _text_artifact_task_router_llm,
    build_text_artifact_generator_payload,
    build_text_artifact_task_router_payload,
    normalize_text_artifact_generator_output,
    normalize_text_artifact_task_router_output,
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


async def test_background_work_text_artifact_live_case() -> None:
    """Classify and generate one bounded code snippet with real worker LLMs."""

    await _skip_if_llm_unavailable()
    source_summary = (
        "The user asked for a simple Fibonacci sequence generator and expects "
        "a short code answer later."
    )
    max_output_chars = 3000
    task_payload = build_text_artifact_task_router_payload(
        task=(
            "Generate a compact Python function fibonacci(n) that returns the "
            "first n Fibonacci numbers."
        ),
        source_summary=source_summary,
        max_output_chars=max_output_chars,
    )
    task_response = await _text_artifact_task_router_llm.ainvoke([
        SystemMessage(content=TEXT_ARTIFACT_TASK_ROUTER_PROMPT),
        HumanMessage(content=json.dumps(task_payload, ensure_ascii=False)),
    ])
    raw_task_output = str(task_response.content)
    parsed_task_output = parse_llm_json_output(raw_task_output)
    task_decision = normalize_text_artifact_task_router_output(
        parsed_task_output
    )

    generator_payload = build_text_artifact_generator_payload(
        task_type=task_decision["task_type"],
        task=source_summary,
        source_summary=source_summary,
        max_output_chars=max_output_chars,
    )
    generator_response = await _text_artifact_generator_llm.ainvoke([
        SystemMessage(content=TEXT_ARTIFACT_GENERATOR_PROMPT),
        HumanMessage(content=json.dumps(generator_payload, ensure_ascii=False)),
    ])
    raw_generator_output = str(generator_response.content)
    parsed_generator_output = parse_llm_json_output(raw_generator_output)
    generator_result = normalize_text_artifact_generator_output(
        parsed_generator_output,
        max_output_chars=max_output_chars,
    )
    trace_path = write_llm_trace(
        "background_work_text_artifact_live_llm",
        "fibonacci_code_snippet",
        {
            "task_router_prompt": TEXT_ARTIFACT_TASK_ROUTER_PROMPT,
            "task_router_payload": task_payload,
            "task_router_raw_output": raw_task_output,
            "task_router_parsed_output": parsed_task_output,
            "task_router_decision": task_decision,
            "generator_prompt": TEXT_ARTIFACT_GENERATOR_PROMPT,
            "generator_payload": generator_payload,
            "generator_raw_output": raw_generator_output,
            "generator_parsed_output": parsed_generator_output,
            "generator_result": generator_result,
            "judgment": (
                "manual_review_required_for_worker_classification_and_generation"
            ),
        },
    )
    logger.info(
        f"BACKGROUND_WORK_TEXT_ARTIFACT_LIVE trace={trace_path} "
        f"task={json.dumps(task_decision, ensure_ascii=True)} "
        f"result={json.dumps(generator_result, ensure_ascii=True)}"
    )

    assert task_decision["task_type"] == "coding_snippet"
    assert "task" not in task_decision, (
        "task classifier must not emit a cleaned 'task' string"
    )
    assert "artifact_text" not in task_decision
    assert generator_result["status"] == "succeeded"
    assert "def " in generator_result["artifact_text"]
    assert "fib" in generator_result["artifact_text"].lower()
    assert "task_type" not in generator_result


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
