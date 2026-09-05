"""Retained deterministic coverage for generation-bound DSH worker resume."""

from __future__ import annotations

import importlib

import pytest

from tests.task_resolution_test_helpers import (
    InMemoryDshBindingStore,
    _context,
    _resolution_ref,
    recorded_task_checkpoint,
)


def _job(*, operation: str = "open_dsh_resolution") -> dict[str, object]:
    """Build one leased V2 task job for the worker entrypoint."""

    context = _context()
    return {
        "schema_version": "background_work_job.v2",
        "job_id": "job-task-001",
        "semantic_objective": "Resolve one bounded public question.",
        "task_execution_context": context,
        "worker_payload": {
            "schema_version": "task_orchestrator_worker_payload.v2",
            "operation": operation,
            "task_session_id": "session-task-001",
            "operation_generation": 0,
            "control": None,
        },
    }


def test_recorded_result_is_canonical_and_dsh_owned() -> None:
    """The retained fixture follows the one public V1 result carrier."""

    module = importlib.import_module("kazusa_ai_chatbot.task_resolution.contracts")
    _checkpoint, result = recorded_task_checkpoint()
    validated = module.validate_task_resolution_result(result)
    assert validated["evidence"][0]["specialist"] == "dsh"
    assert validated["coding_run_context"] == {}


@pytest.mark.asyncio
async def test_worker_projects_terminal_runtime_result(monkeypatch: pytest.MonkeyPatch) -> None:
    """A claimed V2 open operation reaches the typed task result projector."""

    worker = importlib.import_module(
        "kazusa_ai_chatbot.background_work.subagent.task_orchestrator"
    )
    service = importlib.import_module(
        "kazusa_ai_chatbot.task_resolution.service"
    )
    context = _context()
    objective = "Resolve one bounded public question."
    start_spec = service._build_start_spec(
        {
            "semantic_goal": objective,
            "reason": "The task needs one public source.",
            "evidence_handles": [],
            "start_in_background": True,
        },
        context,
    )
    reference = _resolution_ref()
    binding_store = InMemoryDshBindingStore(preassigned_ref=reference)
    await binding_store.create_task_binding(
        binding={
            "schema_version": "dsh_task_binding.v1",
            "task_session_id": reference["dsh_session_id"],
            "semantic_objective": objective,
            "goal_continuation_ref": context["goal_continuation_ref"],
            "source_scope": {},
            "state": "queued",
            "start_spec": start_spec,
            "resolution_thread_id": None,
            "segment_id": None,
            "resolution_ref": None,
            "operation_generation": 0,
            "current_accepted_task_id": "accepted-task-001",
            "current_background_work_job_id": "job-task-001",
            "latest_task_resolution_result": None,
            "revision": 0,
            "created_at": context["current_timestamp_utc"],
            "updated_at": context["current_timestamp_utc"],
        },
    )

    class Runtime:
        async def open(self, **kwargs: object) -> dict[str, object]:
            before_resolve = kwargs["before_resolve"]
            await before_resolve(reference)
            return {
                "kind": "terminal",
                "terminal": {
                    "status": "resolved",
                    "summary": "A public source resolved the requested fact.",
                    "findings": [
                        {"answer": "A public source resolved the requested fact."},
                    ],
                    "completed_subgoals": ["Resolve one bounded public question."],
                    "remaining_needs": [],
                    "clarification_request": None,
                    "approval_request": None,
                    "artifact_refs": [],
                    "warnings": [],
                },
                "evidence": [{
                    "schema_version": "evidence_receipt.v2",
                    "resolution_thread_id": reference["resolution_thread_id"],
                    "segment_id": reference["segment_id"],
                    "scope_fingerprint": "sha256:scope",
                    "audience_fingerprint": "sha256:audience",
                    "policy_epoch": "dsh-standard-policy-v2",
                    "evidence_id": "public-evidence-1",
                    "source_kind": "semantic",
                    "semantic_ref": "https://example.com/source",
                    "content_digest": "sha256:public-evidence-1",
                    "provenance": {"tool_name": "web_search"},
                }],
            }

    monkeypatch.setattr(worker, "_TASK_RESOLUTION_RUNTIME", Runtime())
    monkeypatch.setattr(worker, "_TASK_RESOLUTION_BINDING_STORE", binding_store)
    result = await worker.execute_task_orchestrator_job(
        _job(),
        lease_owner="worker-a",
    )
    assert result["status"] == "resolved"
    assert result["evidence"][0]["evidence_id"] == "public-evidence-1"
    assert result["evidence"][0]["specialist"] == "dsh"
    assert binding_store.bindings[reference["dsh_session_id"]]["state"] == (
        "terminal"
    )
