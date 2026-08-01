"""Leased background entrypoint for resumable task-resolution sessions."""

from __future__ import annotations

from collections.abc import Mapping
from importlib import import_module

from kazusa_ai_chatbot.background_work.jobs import (
    validate_task_orchestrator_worker_payload,
)
from kazusa_ai_chatbot.db.background_work_jobs import (
    checkpoint_background_work_job,
)
from kazusa_ai_chatbot.task_resolution.contracts import (
    TaskResolutionContractError,
    TaskResolutionExecutionContextV1,
    TaskResolutionResultV1,
    validate_task_resolution_checkpoint,
    validate_task_resolution_execution_context,
    validate_task_resolution_result,
)
from kazusa_ai_chatbot.task_resolution.orchestrator import (
    run_task_orchestrator,
)
from kazusa_ai_chatbot.time_boundary import storage_utc_now_iso


async def execute_task_orchestrator_job(
    job: Mapping[str, object],
    *,
    lease_owner: str,
) -> TaskResolutionResultV1:
    """Resume one task-resolution job while preserving every dispatch update.

    Args:
        job: Claimed v2 background-work row with its durable payload.
        lease_owner: Current worker lease token required for checkpoint writes.

    Returns:
        The terminal prompt-safe result for accepted-task delivery.
    """

    job_id = _required_job_text(job, "job_id")
    worker_payload = _required_job_mapping(job, "worker_payload")
    payload = validate_task_orchestrator_worker_payload(worker_payload)
    if payload["operation"] == "continue_bound_coding_run":
        result = await _continue_bound_coding_run(
            payload,
            semantic_objective=_required_job_text(job, "semantic_objective"),
        )
        return result
    if payload["operation"] != "resume_task_resolution":
        raise TaskResolutionContractError("task-orchestrator operation is invalid")
    checkpoint = validate_task_resolution_checkpoint(payload["checkpoint"])
    execution_context = validate_task_resolution_execution_context(
        _required_job_mapping(job, "task_execution_context"),
    )
    existing_result = _stored_task_result(job)
    if checkpoint["terminal_status"]:
        if existing_result is None:
            raise TaskResolutionContractError(
                "terminal checkpoint is missing its durable task result"
            )
        return existing_result

    async def persist_checkpoint(
        updated_checkpoint: dict[str, object],
        result_snapshot: dict[str, object] | None,
    ) -> None:
        """Write one selection, start, or completion checkpoint under lease."""

        checkpoint_kwargs: dict[str, object] = {
            "job_id": job_id,
            "lease_owner": lease_owner,
            "checkpoint": updated_checkpoint,
            "updated_at": storage_utc_now_iso(),
        }
        if result_snapshot is not None:
            checkpoint_kwargs["task_resolution_result"] = result_snapshot
        updated_job = await checkpoint_background_work_job(**checkpoint_kwargs)
        if updated_job is None:
            raise TaskResolutionContractError(
                "task-resolution checkpoint lost its worker lease"
            )

    prior_result = _deferred_task_result(existing_result)
    result = await run_task_orchestrator(
        checkpoint,
        execution_context,
        inline_deadline=None,
        checkpoint_persist_func=persist_checkpoint,
        prior_result=prior_result,
    )
    return result


async def _continue_bound_coding_run(
    payload: Mapping[str, object],
    *,
    semantic_objective: str,
) -> TaskResolutionResultV1:
    """Continue one already-bound coding run through its frozen public API."""

    coding_request = payload.get("coding_request")
    if not isinstance(coding_request, Mapping):
        raise TaskResolutionContractError(
            "continue_bound_coding_run requires a validated coding request"
        )
    coding_agent = import_module("kazusa_ai_chatbot.coding_agent")
    continue_coding_run = getattr(coding_agent, "continue_coding_run")
    response = await continue_coding_run(dict(coding_request))
    if not isinstance(response, Mapping):
        raise TaskResolutionContractError(
            "coding continuation did not return a public response object"
        )
    coding_specialist = import_module(
        "kazusa_ai_chatbot.task_resolution.specialists.coding",
    )
    project_result = getattr(
        coding_specialist,
        "project_bound_coding_continuation_result",
    )
    result = project_result(
        dict(response),
        semantic_objective=semantic_objective,
    )
    validated_result = validate_task_resolution_result(result)
    return validated_result


def _stored_task_result(
    job: Mapping[str, object],
) -> TaskResolutionResultV1 | None:
    """Read an optional validated task snapshot from a durable v2 job."""

    value = job.get("task_resolution_result")
    if not isinstance(value, Mapping) or not value:
        return None
    result = validate_task_resolution_result(value)
    return result


def _deferred_task_result(
    result: TaskResolutionResultV1 | None,
) -> TaskResolutionResultV1 | None:
    """Pass only a nonterminal durable snapshot back into orchestration."""

    if result is None:
        return None
    if result["status"] != "deferred":
        raise TaskResolutionContractError(
            "nonterminal task checkpoint has a terminal task result"
        )
    return result


def _required_job_mapping(
    job: Mapping[str, object],
    field_name: str,
) -> dict[str, object]:
    """Require one v2 job mapping at the task-worker boundary."""

    value = job.get(field_name)
    if not isinstance(value, Mapping):
        raise TaskResolutionContractError(
            f"background job {field_name}: expected object"
        )
    mapping = dict(value)
    return mapping


def _required_job_text(job: Mapping[str, object], field_name: str) -> str:
    """Require one non-empty v2 job identifier or source field."""

    value = job.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise TaskResolutionContractError(
            f"background job {field_name}: expected non-empty text"
        )
    text = value.strip()
    return text
