"""Public boundary for inline-first resumable task resolution."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from kazusa_ai_chatbot.task_resolution.contracts import (
    AcceptedTaskControlV1,
    DshResolutionRefV1,
    TaskResolutionAdmissionV1,
    TaskResolutionContractError,
    TaskResolutionExecutionContextV2,
    TaskResolutionResultV1,
    validate_accepted_task_control,
    validate_dsh_resolution_ref,
    validate_task_resolution_admission,
    validate_task_resolution_execution_context,
    validate_task_resolution_result,
)

__all__ = [
    "AcceptedTaskControlV1",
    "DshResolutionRefV1",
    "TaskResolutionAdmissionV1",
    "TaskResolutionContractError",
    "TaskResolutionExecutionContextV2",
    "TaskResolutionResultV1",
    "continue_delivered_task",
    "reconcile_task_resolution_result",
    "resolve_task_inline",
    "resume_task_resolution",
    "start_task_resolution_in_background",
    "validate_accepted_task_control",
    "validate_dsh_resolution_ref",
    "validate_task_resolution_admission",
    "validate_task_resolution_execution_context",
    "validate_task_resolution_result",
]


def __getattr__(name: str) -> Any:
    """Resolve service entrypoints only when a caller requests them.

    Contract validators are imported by accepted-task and background-job
    persistence during application startup.  Delaying the service module keeps
    those schema imports independent from cognition integration imports.
    """

    if name not in {
        "resolve_task_inline",
        "resume_task_resolution",
        "start_task_resolution_in_background",
        "reconcile_task_resolution_result",
        "continue_delivered_task",
    }:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    service_module = import_module("kazusa_ai_chatbot.task_resolution.service")
    resolved_value = getattr(service_module, name)
    globals()[name] = resolved_value
    return resolved_value
