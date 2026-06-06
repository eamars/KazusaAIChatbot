"""Tests for deterministic background-work provider dispatch."""

from __future__ import annotations

import importlib

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


@pytest.mark.asyncio
async def test_provider_dispatches_by_worker_and_action_only(monkeypatch) -> None:
    """Provider dispatch should not pass worker-local params to workers."""

    providers = importlib.import_module("kazusa_ai_chatbot.background_work.providers")
    execute = getattr(providers, "execute_background_work_decision")
    worker_execute = AsyncMock(return_value={
        "status": "succeeded",
        "worker": "text_artifact",
        "artifact_text": "def fibonacci(n): ...",
        "failure_summary": "",
        "result_summary": "Generated a Fibonacci snippet.",
        "worker_metadata": {"task_type": "coding_snippet"},
    })
    worker = SimpleNamespace(execute=worker_execute)
    monkeypatch.setattr(
        providers,
        "load_background_work_workers",
        lambda: {"text_artifact": worker},
    )

    result = await execute({
        "action": "execute",
        "worker": "text_artifact",
        "task": "Generate a Fibonacci function snippet.",
        "reason": "The task is bounded text artifact work.",
        "work_kind": "coding_snippet",
        "tool_args": {"path": "fibonacci.py"},
    })

    assert result["status"] == "succeeded"
    worker_execute.assert_awaited_once()
    worker_decision = worker_execute.await_args.args[0]
    assert worker_decision == {
        "action": "execute",
        "worker": "text_artifact",
        "task": "Generate a Fibonacci function snippet.",
        "reason": "The task is bounded text artifact work.",
    }


@pytest.mark.asyncio
async def test_provider_passes_max_output_cap_as_execution_context(
    monkeypatch,
) -> None:
    """The queued output cap is deterministic context, not router output."""

    providers = importlib.import_module("kazusa_ai_chatbot.background_work.providers")
    dispatch = getattr(providers, "dispatch_background_work")
    worker_execute = AsyncMock(return_value={
        "status": "succeeded",
        "worker": "text_artifact",
        "artifact_text": "short result",
        "failure_summary": "",
        "result_summary": "Generated a bounded result.",
        "worker_metadata": {"task_type": "summary"},
    })
    worker = SimpleNamespace(execute=worker_execute)
    monkeypatch.setattr(
        providers,
        "load_background_work_workers",
        lambda: {"text_artifact": worker},
    )

    result = await dispatch(
        {
            "action": "execute",
            "worker": "text_artifact",
            "task": "Summarize this text.",
            "reason": "The task is bounded text artifact work.",
        },
        max_output_chars=120,
    )

    assert result["status"] == "succeeded"
    worker_execute.assert_awaited_once()
    assert worker_execute.await_args.kwargs["max_output_chars"] == 120
    assert "max_output_chars" not in worker_execute.await_args.args[0]


@pytest.mark.asyncio
async def test_provider_rejects_unknown_worker_without_fallback(monkeypatch) -> None:
    """Unsupported workers should fail closed instead of using a fallback."""

    providers = importlib.import_module("kazusa_ai_chatbot.background_work.providers")
    execute = getattr(providers, "execute_background_work_decision")
    monkeypatch.setattr(providers, "load_background_work_workers", lambda: {})

    result = await execute({
        "action": "execute",
        "worker": "web_research",
        "task": "Research a topic.",
        "reason": "The router selected an unavailable worker.",
    })

    assert result["status"] == "rejected"
    assert result["worker"] == "web_research"
    assert "unsupported" in result["failure_summary"].lower()
