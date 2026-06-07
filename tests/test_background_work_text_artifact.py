"""Tests for the first background-work text-artifact subagent."""

from __future__ import annotations

import importlib

from unittest.mock import AsyncMock

import pytest


def test_text_artifact_exposes_two_separate_llm_stage_contracts() -> None:
    """The worker must classify task type and generate artifact separately."""

    module = importlib.import_module(
        "kazusa_ai_chatbot.background_work.subagent.text_artifact"
    )

    for name in (
        "build_text_artifact_task_router_payload",
        "normalize_text_artifact_task_router_output",
        "build_text_artifact_generator_payload",
        "normalize_text_artifact_generator_output",
        "execute",
    ):
        assert hasattr(module, name)


def test_text_artifact_task_router_does_not_emit_clean_task() -> None:
    """Task classifier must output only task_type and reason.
    It must not manufacture a cleaned task string for the generator."""

    module = importlib.import_module(
        "kazusa_ai_chatbot.background_work.subagent.text_artifact"
    )
    normalize = getattr(module, "normalize_text_artifact_task_router_output")

    decision = normalize({
        "task_type": "coding_snippet",
        "task": "Generate a clean Fibonacci function.",
        "reason": "The task asks for bounded code text.",
    })

    assert "task_type" in decision
    assert "reason" in decision
    assert "task" not in decision, (
        "task classifier must not emit a cleaned 'task' string; "
        "classifiers classify, they do not generate worker parameters"
    )


def test_task_router_normalizer_excludes_artifact_text() -> None:
    """Worker-local classification must not return generated artifacts."""

    module = importlib.import_module(
        "kazusa_ai_chatbot.background_work.subagent.text_artifact"
    )
    normalize = getattr(module, "normalize_text_artifact_task_router_output")

    decision = normalize({
        "task_type": "coding_snippet",
        "task": "Generate a Fibonacci function snippet.",
        "reason": "The task asks for bounded code text.",
        "artifact_text": "def fibonacci(n): ...",
    })

    assert decision == {
        "task_type": "coding_snippet",
        "reason": "The task asks for bounded code text.",
    }


def test_generator_normalizer_excludes_task_type_selection() -> None:
    """Artifact generation must not choose the worker-local task type."""

    module = importlib.import_module(
        "kazusa_ai_chatbot.background_work.subagent.text_artifact"
    )
    normalize = getattr(module, "normalize_text_artifact_generator_output")

    result = normalize({
        "status": "succeeded",
        "artifact_text": "def fibonacci(n): ...",
        "failure_summary": "",
        "result_summary": "Generated a compact Fibonacci snippet.",
        "task_type": "coding_snippet",
    })

    assert result == {
        "status": "succeeded",
        "artifact_text": "def fibonacci(n): ...",
        "failure_summary": "",
        "result_summary": "Generated a compact Fibonacci snippet.",
    }


@pytest.mark.asyncio
async def test_execute_uses_job_max_output_cap_for_worker_stages(
    monkeypatch,
) -> None:
    """Worker execution should honor the cap from the queued job."""

    module = importlib.import_module(
        "kazusa_ai_chatbot.background_work.subagent.text_artifact"
    )
    route_task = AsyncMock(return_value={
        "task_type": "summary",
        "reason": "The task is a bounded summary.",
    })
    generate = AsyncMock(return_value={
        "status": "succeeded",
        "artifact_text": "short summary",
        "failure_summary": "",
        "result_summary": "Generated a short summary.",
    })
    monkeypatch.setattr(module, "_route_text_artifact_task", route_task)
    monkeypatch.setattr(module, "_generate_text_artifact", generate)

    result = await module.execute(
        {
            "action": "execute",
            "worker": "text_artifact",
            "reason": "The task is bounded text artifact work.",
            "source_summary": "Summarize this text.",
        },
        max_output_chars=120,
    )

    assert result["status"] == "succeeded"
    route_task.assert_awaited_once()
    generate.assert_awaited_once()
    assert route_task.await_args.kwargs["max_output_chars"] == 120
    assert generate.await_args.kwargs["max_output_chars"] == 120
