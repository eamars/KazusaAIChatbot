"""Frozen-public-API coding specialist for task-resolution sessions."""

from __future__ import annotations

from collections.abc import Mapping

from kazusa_ai_chatbot.coding_agent import (
    CodingRunResponse,
    CodingRunStartRequest,
    start_coding_run,
)
from kazusa_ai_chatbot.task_resolution.contracts import (
    TASK_RESOLUTION_RESULT_VERSION,
    TaskResolutionExecutionContextV1,
    TaskResolutionResultV1,
    TaskSpecialistRequestV1,
    TaskSpecialistResultV1,
    validate_task_resolution_result,
)
from kazusa_ai_chatbot.task_resolution.specialists import (
    _bounded_text,
    _bounded_text_items,
    _require_handler_coding_objective_mode,
    _specialist_evidence,
    _specialist_result,
    _validated_handler_inputs,
)


SPECIALIST = "coding"
CODING_RUN_REF_PREFIX = "coding_run:"
MAX_CODING_ANSWER_CHARS = 3000


async def resolve_with_coding(
    request: dict[str, object],
    execution_context: TaskResolutionExecutionContextV1,
) -> TaskSpecialistResultV1:
    """Start one validated coding run through the frozen public entrypoint.

    Task resolution does not invent approval, patch, execution, or continuation
    actions.  The existing coding-run lifecycle owns those operations after a
    public run response exposes an allowed next action.
    """

    task_request, context = _validated_handler_inputs(request, execution_context)
    _require_handler_coding_objective_mode(
        task_request,
        specialist=SPECIALIST,
    )
    workspace_root = context["coding_workspace_root"].strip()
    if not workspace_root:
        return _specialist_result(
            specialist=SPECIALIST,
            status="incompatible",
            remaining_needs=_remaining_needs(task_request),
            reason="Coding work requires a trusted coding workspace.",
        )
    start_request = _start_request(task_request, workspace_root, context)
    response = await start_coding_run(start_request)
    return _response_result(task_request, response)


def project_bound_coding_continuation_result(
    response: CodingRunResponse,
    *,
    semantic_objective: str,
) -> TaskResolutionResultV1:
    """Project one public bound-run continuation into a task result.

    The coding-run lifecycle retains action, approval, execution, and repair
    semantics. Task resolution retains only the bounded public outcome needed
    for accepted-task delivery and later cognition.
    """

    objective = _bounded_text(semantic_objective)
    coding_context = _coding_run_context(response)
    run_status = _response_text(response, "status", maximum=80)
    summary = _response_text(response, "answer_text")
    limitations = _response_text_items(response, "limitations")
    if run_status == "completed":
        return _continuation_task_result(
            status="resolved",
            summary=summary or "The coding run completed its requested work.",
            completed_subgoals=[objective],
            remaining_needs=[],
            coding_run_context=coding_context,
        )
    if run_status == "awaiting_approval":
        return _continuation_task_result(
            status="approval_required",
            summary=summary or "The coding run requires approval to continue.",
            completed_subgoals=[],
            remaining_needs=limitations or [
                "Review the coding run and provide approval if desired.",
            ],
            coding_run_context=coding_context,
        )
    if run_status == "blocked":
        return _continuation_task_result(
            status="needs_user_input",
            summary=summary or "The coding run requires additional input.",
            completed_subgoals=[],
            remaining_needs=limitations or [objective],
            coding_run_context=coding_context,
        )
    if _response_text(response, "operation_outcome", maximum=40) == "busy":
        return _continuation_task_result(
            status="unavailable",
            summary=summary or "The coding run is temporarily unavailable.",
            completed_subgoals=[],
            remaining_needs=[objective],
            coding_run_context=coding_context,
        )
    if run_status in {"rejected", "cancelled"}:
        return _continuation_task_result(
            status="unavailable",
            summary=summary or "The coding run could not continue that action.",
            completed_subgoals=[],
            remaining_needs=limitations or [objective],
            coding_run_context=coding_context,
        )
    return _continuation_task_result(
        status="failed",
        summary=summary or "The coding run did not return a usable result.",
        completed_subgoals=[],
        remaining_needs=limitations or [objective],
        coding_run_context=coding_context,
    )


def _start_request(
    request: TaskSpecialistRequestV1,
    workspace_root: str,
    context: TaskResolutionExecutionContextV1,
) -> CodingRunStartRequest:
    """Map only the validated coding mode into the frozen start request."""

    max_answer_chars = min(context["max_output_chars"], MAX_CODING_ANSWER_CHARS)
    start_request: CodingRunStartRequest = {
        "question": request["objective"],
        "objective_type": request["coding_objective_mode"],
        "workspace_root": workspace_root,
        "local_root_hint": workspace_root,
        "source_scope_hint": "directory",
        "max_answer_chars": max_answer_chars,
        "max_artifact_chars": max_answer_chars * 8,
    }
    return start_request


def _response_result(
    request: TaskSpecialistRequestV1,
    response: CodingRunResponse,
) -> TaskSpecialistResultV1:
    """Project a public coding response without leaking implementation state."""

    run_status = _response_text(response, "status", maximum=80)
    coding_context = _coding_run_context(response)
    if run_status == "completed":
        summary = _response_text(response, "answer_text")
        if not summary:
            summary = "The coding run completed its requested work."
        provenance_ref = _coding_provenance_ref(coding_context)
        evidence = _specialist_evidence(
            request=request,
            specialist=SPECIALIST,
            summary=summary,
            provenance_refs=[provenance_ref],
            limitations=_response_text_items(response, "limitations"),
        )
        return _specialist_result(
            specialist=SPECIALIST,
            status="resolved",
            evidence=[evidence],
            completed_subgoals=[request["objective"]],
            reason="The coding run completed its requested work.",
            coding_run_context=coding_context,
        )
    if run_status == "awaiting_approval":
        return _specialist_result(
            specialist=SPECIALIST,
            status="approval_required",
            remaining_needs=["Review the coding run and provide approval if desired."],
            reason="The coding run requires its existing approval lifecycle.",
            coding_run_context=coding_context,
        )
    if run_status == "blocked":
        return _specialist_result(
            specialist=SPECIALIST,
            status="needs_user_input",
            remaining_needs=_response_text_items(response, "limitations")
            or _remaining_needs(request),
            reason="The coding run requires additional user input.",
            coding_run_context=coding_context,
        )
    if _response_text(response, "operation_outcome", maximum=40) == "busy":
        return _specialist_result(
            specialist=SPECIALIST,
            status="temporarily_unavailable",
            remaining_needs=_remaining_needs(request),
            reason="The coding-run public API is temporarily busy.",
            retryable=True,
            coding_run_context=coding_context,
        )
    if run_status in {"rejected", "cancelled"}:
        return _specialist_result(
            specialist=SPECIALIST,
            status="incompatible",
            remaining_needs=_remaining_needs(request),
            reason="The coding-run public API rejected this coding subgoal.",
            coding_run_context=coding_context,
        )
    return _specialist_result(
        specialist=SPECIALIST,
        status="failed",
        remaining_needs=_remaining_needs(request),
        reason="The coding run did not reach a terminal public result.",
        coding_run_context=coding_context,
    )


def _continuation_task_result(
    *,
    status: str,
    summary: str,
    completed_subgoals: list[str],
    remaining_needs: list[str],
    coding_run_context: Mapping[str, object],
) -> TaskResolutionResultV1:
    """Build one terminal task-resolution projection for a bound coding run."""

    result: TaskResolutionResultV1 = {
        "schema_version": TASK_RESOLUTION_RESULT_VERSION,
        "status": status,
        "prompt_safe_summary": _bounded_text(summary),
        "evidence": [],
        "completed_subgoals": _bounded_text_items(completed_subgoals),
        "remaining_needs": _bounded_text_items(remaining_needs),
        "checkpoint": {},
        "coding_run_context": dict(coding_run_context),
    }
    validated = validate_task_resolution_result(result)
    return validated


def _coding_run_context(response: Mapping[str, object]) -> dict[str, object]:
    """Project only the exact prompt-safe coding handoff contract."""

    run_id = _response_text(response, "run_id", maximum=120)
    if not run_id:
        return {}
    summary = _response_text(response, "answer_text")
    if not summary:
        summary = _response_text(response, "goal")
    if not summary:
        summary = "Coding run state is available through its public reference."
    run_status = _response_text(response, "status", maximum=80)
    context = {
        "schema_version": "coding_run_context.v1",
        "coding_run_ref": _coding_run_ref(run_id),
        "status": run_status or "unknown",
        "summary": summary,
        "limitations": _response_text_items(response, "limitations"),
        "allowed_next_actions": _response_text_items(
            response,
            "allowed_next_actions",
        ),
        "followup_open": run_status in {"awaiting_approval", "blocked"},
    }
    return context


def _coding_provenance_ref(coding_context: Mapping[str, object]) -> str:
    """Return the stable public run reference used as coding evidence provenance."""

    coding_run_ref = coding_context.get("coding_run_ref")
    if isinstance(coding_run_ref, str) and coding_run_ref.strip():
        return coding_run_ref
    return "coding_run:completed"


def _coding_run_ref(run_id: str) -> str:
    """Normalize a public run ID into the task-resolution reference vocabulary."""

    if run_id.startswith(CODING_RUN_REF_PREFIX):
        return run_id
    return f"{CODING_RUN_REF_PREFIX}{run_id}"


def _response_text(
    response: Mapping[str, object],
    field_name: str,
    *,
    maximum: int = 1200,
) -> str:
    """Read one bounded prompt-safe string from a public coding response."""

    value = response.get(field_name)
    if not isinstance(value, str) or not value.strip():
        return ""
    return _bounded_text(value, maximum=maximum)


def _response_text_items(
    response: Mapping[str, object],
    field_name: str,
) -> list[str]:
    """Read one bounded public string-list from a coding response."""

    value = response.get(field_name)
    if not isinstance(value, list):
        return []
    items: list[str] = []
    for raw_item in value[:8]:
        if isinstance(raw_item, str) and raw_item.strip():
            items.append(_bounded_text(raw_item))
    return items


def _remaining_needs(request: TaskSpecialistRequestV1) -> list[str]:
    """Retain canonical unresolved needs after a coding refusal or failure."""

    if request["remaining_needs"]:
        return list(request["remaining_needs"])
    return [request["objective"]]
