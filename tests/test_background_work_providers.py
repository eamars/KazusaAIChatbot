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
        "reason": "The task is bounded text artifact work.",
        "source_summary": "Generate a Fibonacci function snippet.",
        "work_kind": "coding_snippet",
        "tool_args": {"path": "fibonacci.py"},
    })

    assert result["status"] == "succeeded"
    worker_execute.assert_awaited_once()
    worker_decision = worker_execute.await_args.args[0]
    assert worker_decision["action"] == "execute"
    assert worker_decision["worker"] == "text_artifact"
    assert worker_decision["reason"] == "The task is bounded text artifact work."
    assert worker_decision["source_summary"] == "Generate a Fibonacci function snippet."
    assert "task" not in worker_decision


@pytest.mark.asyncio
async def test_provider_passes_trusted_task_brief_to_worker(monkeypatch) -> None:
    """Workers should receive durable task_brief separate from source summary."""

    providers = importlib.import_module("kazusa_ai_chatbot.background_work.providers")
    dispatch = getattr(providers, "dispatch_background_work")
    worker_execute = AsyncMock(return_value={
        "status": "succeeded",
        "worker": "coding_agent",
        "artifact_text": "Image reading uses attachment descriptors.",
        "failure_summary": "",
        "result_summary": "Answered source-code question.",
        "worker_metadata": {},
    })
    worker = SimpleNamespace(execute=worker_execute)
    monkeypatch.setattr(
        providers,
        "load_background_work_workers",
        lambda: {"coding_agent": worker},
    )

    result = await dispatch(
        {
            "action": "execute",
            "worker": "coding_agent",
            "reason": "The task is bounded source-code reading.",
            "task_brief": "Explain how image reading is implemented.",
            "source_summary": "The user asked for a repository explanation.",
        },
        max_output_chars=120,
    )

    assert result["status"] == "succeeded"
    worker_decision = worker_execute.await_args.args[0]
    assert worker_decision["task_brief"] == (
        "Explain how image reading is implemented."
    )
    assert worker_decision["source_summary"] == (
        "The user asked for a repository explanation."
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "operation",
    [
        "start",
        "revise_proposal",
        "summarize",
        "status",
        "approve_and_verify",
        "respond_to_blocker",
        "cancel",
    ],
)
async def test_provider_preserves_every_coding_worker_payload(
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    """Provider dispatch must pass each bound coding operation unchanged."""

    providers = importlib.import_module("kazusa_ai_chatbot.background_work.providers")
    worker_execute = AsyncMock(return_value={
        "status": "succeeded",
        "worker": "coding_agent",
        "artifact_text": "Coding operation accepted.",
        "failure_summary": "",
        "result_summary": "Coding operation accepted.",
        "worker_metadata": {},
    })
    worker = SimpleNamespace(execute=worker_execute)
    monkeypatch.setattr(
        providers,
        "load_background_work_workers",
        lambda: {"coding_agent": worker},
    )
    coding_run_ref = "" if operation == "start" else "coding_run:run-001"
    worker_payload = {
        "schema_version": "coding_agent_worker_payload.v2",
        "operation": operation,
        "task_brief": "Perform the requested coding operation.",
        "coding_run_ref": coding_run_ref,
        "execution_request": "",
    }

    result = await providers.dispatch_background_work({
        "action": "execute",
        "worker": "coding_agent",
        "reason": "Validated background action requested this worker.",
        "task_brief": "Perform the requested coding operation.",
        "source_summary": "The user requested durable coding work.",
        "worker_payload": worker_payload,
    })

    assert result["status"] == "succeeded"
    worker_execute.assert_awaited_once()
    dispatched_decision = worker_execute.await_args.args[0]
    assert dispatched_decision["worker"] == "coding_agent"
    assert dispatched_decision["worker_payload"] == worker_payload


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
            "reason": "The task is bounded text artifact work.",
            "source_summary": "Summarize this text.",
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
        "reason": "The router selected an unavailable worker.",
    })

    assert result["status"] == "rejected"
    assert result["worker"] == "web_research"
    assert "unsupported" in result["failure_summary"].lower()
