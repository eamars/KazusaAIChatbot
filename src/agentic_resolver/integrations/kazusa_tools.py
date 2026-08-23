"""Adapters over the four existing task-resolution specialist handlers."""

from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable, Mapping

from agentic_resolver.tools import ToolDefinition, ToolRegistry
from kazusa_ai_chatbot.task_resolution.contracts import (
    TASK_SPECIALIST_REQUEST_VERSION,
    TaskResolutionExecutionContextV1,
    TaskSpecialistResultV1,
    validate_task_resolution_execution_context,
    validate_task_specialist_result,
)
from kazusa_ai_chatbot.task_resolution.specialists.coding import (
    resolve_with_coding,
)
from kazusa_ai_chatbot.task_resolution.specialists.local_context import (
    resolve_with_local_context,
)
from kazusa_ai_chatbot.task_resolution.specialists.public_research import (
    resolve_with_public_research,
)
from kazusa_ai_chatbot.task_resolution.specialists.text_computation import (
    resolve_with_text_computation,
)

SpecialistHandler = Callable[
    [dict[str, object], TaskResolutionExecutionContextV1],
    Awaitable[TaskSpecialistResultV1],
]

_TOOL_DESCRIPTIONS = {
    "local_context": (
        "Retrieve bounded private or local context through the existing "
        "local-context specialist."
    ),
    "public_research": (
        "Investigate current public evidence through the existing public "
        "research specialist."
    ),
    "coding": (
        "Use the existing coding-run lifecycle for read-only analysis or an "
        "approval-gated patch proposal."
    ),
    "text_computation": (
        "Perform bounded supplied-text transformation or deterministic "
        "computation through the existing specialist."
    ),
}


def build_kazusa_tool_registry(
    execution_context: TaskResolutionExecutionContextV1,
) -> ToolRegistry:
    """Bind the current four public specialist handlers to native tools.

    Args:
        execution_context: Trusted prompt-safe context captured by the caller.

    Returns:
        A frozen ordinary-tool registry that leaves handler contracts intact.
    """

    validated_context = validate_task_resolution_execution_context(
        execution_context
    )
    handlers: dict[str, SpecialistHandler] = {
        "local_context": resolve_with_local_context,
        "public_research": resolve_with_public_research,
        "coding": resolve_with_coding,
        "text_computation": resolve_with_text_computation,
    }
    definitions = [
        _specialist_tool_definition(
            specialist=specialist,
            handler=handler,
            execution_context=validated_context,
        )
        for specialist, handler in handlers.items()
    ]
    registry = ToolRegistry(definitions)
    return registry


def _specialist_tool_definition(
    *,
    specialist: str,
    handler: SpecialistHandler,
    execution_context: TaskResolutionExecutionContextV1,
) -> ToolDefinition:
    """Build one adapter while retaining the current handler as semantic owner."""

    properties: dict[str, object] = {
        "objective": {
            "type": "string",
            "minLength": 1,
            "maxLength": 4000,
        },
    }
    required = ["objective"]
    side_effect_class = "read"
    if specialist == "coding":
        properties["coding_objective_mode"] = {
            "type": "string",
            "enum": ["read_only", "propose_patch"],
        }
        required.append("coding_objective_mode")
        side_effect_class = "approval_gated"

    async def _execute(arguments: Mapping[str, object]) -> object:
        objective = arguments["objective"]
        if not isinstance(objective, str):
            raise TypeError("validated specialist objective must be a string")
        coding_objective_mode = "none"
        if specialist == "coding":
            raw_mode = arguments["coding_objective_mode"]
            if not isinstance(raw_mode, str):
                raise TypeError("validated coding objective mode must be a string")
            coding_objective_mode = raw_mode
        request = {
            "schema_version": TASK_SPECIALIST_REQUEST_VERSION,
            "task_node_id": f"agentic-{uuid.uuid4().hex}",
            "objective": objective,
            "available_evidence": [],
            "remaining_needs": [],
            "trusted_scope": _trusted_scope(execution_context),
            "coding_objective_mode": coding_objective_mode,
        }
        result = await handler(request, execution_context)
        validated_result = validate_task_specialist_result(result)
        return validated_result

    definition = ToolDefinition(
        name=specialist,
        description=_TOOL_DESCRIPTIONS[specialist],
        input_schema={
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        },
        execute=_execute,
        project_result=_project_specialist_result,
        side_effect_class=side_effect_class,
    )
    return definition


def _trusted_scope(
    context: TaskResolutionExecutionContextV1,
) -> dict[str, str]:
    """Project the exact current specialist trusted-scope contract."""

    scope = {
        "trigger_source": "agentic_resolver",
        "platform": context["platform"],
        "channel_id": context["channel_id"],
        "channel_type": context["channel_type"],
        "source_message_id": context["source_message_id"],
        "requester_global_user_id": context["requester_global_user_id"],
        "requester_platform_user_id": context["requester_platform_user_id"],
    }
    return scope


def _project_specialist_result(value: object) -> Mapping[str, object]:
    """Project validated specialist fields into one bounded tool observation."""

    result = validate_task_specialist_result(value)
    evidence_rows = [
        {
            "evidence_id": row["evidence_id"],
            "summary": row["summary"],
            "provenance_refs": list(row["provenance_refs"]),
            "limitations": list(row["limitations"]),
        }
        for row in result["evidence"]
    ]
    summaries = [row["summary"] for row in result["evidence"]]
    summary = "\n".join(summaries) or result["reason"]
    provenance_refs: list[str] = []
    limitations: list[str] = []
    for evidence in result["evidence"]:
        for reference in evidence["provenance_refs"]:
            if reference not in provenance_refs:
                provenance_refs.append(reference)
        for limitation in evidence["limitations"]:
            if limitation not in limitations:
                limitations.append(limitation)
    for remaining_need in result["remaining_needs"]:
        if remaining_need not in limitations:
            limitations.append(remaining_need)
    projected: dict[str, object] = {
        "summary": summary,
        "specialist": result["specialist"],
        "status": result["status"],
        "evidence": evidence_rows,
        "provenance_refs": provenance_refs,
        "limitations": limitations,
        "completed_subgoals": list(result["completed_subgoals"]),
        "remaining_needs": list(result["remaining_needs"]),
        "reason": result["reason"],
        "retryable": result["retryable"],
    }
    coding_run_context = result.get("coding_run_context")
    if isinstance(coding_run_context, dict):
        projected["coding_run_context"] = dict(coding_run_context)
    return projected
