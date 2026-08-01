"""Registered task-resolution specialist handlers and shared projections."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from hashlib import sha256

from kazusa_ai_chatbot.task_resolution.contracts import (
    MAX_TASK_RESOLUTION_REASON_CHARS,
    MAX_TASK_RESOLUTION_TEXT_CHARS,
    MAX_TASK_RESOLUTION_TEXT_ITEMS,
    TASK_CODING_SPECIALIST_OBJECTIVE_MODES,
    TASK_RESOLUTION_EVIDENCE_VERSION,
    TASK_SPECIALIST_RESULT_VERSION,
    TaskResolutionEvidenceV1,
    TaskResolutionContractError,
    TaskResolutionExecutionContextV1,
    TaskSpecialistRequestV1,
    TaskSpecialistResultV1,
    validate_task_resolution_execution_context,
    validate_task_specialist_request,
    validate_task_specialist_result,
)


def _validated_handler_inputs(
    request: object,
    execution_context: object,
) -> tuple[TaskSpecialistRequestV1, TaskResolutionExecutionContextV1]:
    """Validate canonical handler inputs before any specialist mapping."""

    validated_request = validate_task_specialist_request(request)
    validated_context = validate_task_resolution_execution_context(
        execution_context,
    )
    return validated_request, validated_context


def _require_handler_coding_objective_mode(
    request: TaskSpecialistRequestV1,
    *,
    specialist: str,
) -> str:
    """Keep coding modes bound to the only handler that owns coding work."""

    coding_objective_mode = request["coding_objective_mode"]
    if specialist == "coding":
        if coding_objective_mode not in TASK_CODING_SPECIALIST_OBJECTIVE_MODES:
            raise TaskResolutionContractError(
                "coding_objective_mode: coding requires read_only or propose_patch"
            )
        return coding_objective_mode
    if coding_objective_mode != "none":
        raise TaskResolutionContractError(
            "coding_objective_mode: non-coding specialists require none"
        )
    return coding_objective_mode


def _specialist_result(
    *,
    specialist: str,
    status: str,
    evidence: Sequence[TaskResolutionEvidenceV1] = (),
    completed_subgoals: Sequence[str] = (),
    remaining_needs: Sequence[str] = (),
    reason: str,
    retryable: bool = False,
    coding_run_context: Mapping[str, object] | None = None,
) -> TaskSpecialistResultV1:
    """Build one validated result without exposing provider implementation data."""

    result: TaskSpecialistResultV1 = {
        "schema_version": TASK_SPECIALIST_RESULT_VERSION,
        "specialist": specialist,
        "status": status,
        "evidence": list(evidence),
        "completed_subgoals": _bounded_text_items(completed_subgoals),
        "remaining_needs": _bounded_text_items(remaining_needs),
        "reason": _bounded_text(reason, maximum=MAX_TASK_RESOLUTION_REASON_CHARS),
        "retryable": retryable,
    }
    if coding_run_context:
        result["coding_run_context"] = dict(coding_run_context)
    validated = validate_task_specialist_result(result)
    return validated


def _specialist_evidence(
    *,
    request: TaskSpecialistRequestV1,
    specialist: str,
    summary: str,
    provenance_refs: Sequence[str],
    limitations: Sequence[str] = (),
) -> TaskResolutionEvidenceV1:
    """Build stable, provenance-bearing evidence for one handler result."""

    bounded_summary = _bounded_text(summary)
    bounded_refs = _bounded_text_items(provenance_refs)
    fingerprint = "\x1f".join((
        request["task_node_id"],
        specialist,
        bounded_summary,
        *bounded_refs,
    ))
    evidence: TaskResolutionEvidenceV1 = {
        "schema_version": TASK_RESOLUTION_EVIDENCE_VERSION,
        "evidence_id": (
            f"task-evidence:{specialist}:"
            f"{sha256(fingerprint.encode('utf-8')).hexdigest()[:16]}"
        ),
        "task_node_id": request["task_node_id"],
        "specialist": specialist,
        "summary": bounded_summary,
        "provenance_refs": bounded_refs,
        "limitations": _bounded_text_items(limitations),
    }
    return evidence


def _prompt_message_text(
    execution_context: TaskResolutionExecutionContextV1,
) -> str:
    """Return supplied prompt-safe text for a text/computation specialist."""

    prompt_context = execution_context["prompt_message_context"]
    for field_name in ("text", "content", "message"):
        value = prompt_context.get(field_name)
        if isinstance(value, str) and value.strip():
            return _bounded_text(value, maximum=8000)
    return ""


def _caller_supplied_expression(
    execution_context: TaskResolutionExecutionContextV1,
) -> str:
    """Return an explicitly structured numeric expression, if supplied."""

    prompt_context = execution_context["prompt_message_context"]
    for field_name in ("numeric_expression", "expression"):
        value = prompt_context.get(field_name)
        if isinstance(value, str) and value.strip():
            return _bounded_text(value, maximum=1200)
    return ""


def _bounded_text(value: object, *, maximum: int = MAX_TASK_RESOLUTION_TEXT_CHARS) -> str:
    """Return non-empty prompt-safe text within a task-resolution contract cap."""

    if not isinstance(value, str):
        raise ValueError("task-resolution text: expected string")
    normalized = value.strip()
    if not normalized:
        raise ValueError("task-resolution text: expected non-empty string")
    return normalized[:maximum]


def _bounded_text_items(values: Sequence[object]) -> list[str]:
    """Project a bounded sequence into the task-resolution text-list shape."""

    items: list[str] = []
    for value in values[:MAX_TASK_RESOLUTION_TEXT_ITEMS]:
        if not isinstance(value, str) or not value.strip():
            continue
        items.append(_bounded_text(value))
    return items
