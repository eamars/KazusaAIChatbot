"""Public boundary for inline-first resumable task resolution."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from kazusa_ai_chatbot.task_resolution.contracts import (
    TaskResolutionCheckpointV1,
    TaskResolutionContractError,
    TaskResolutionExecutionContextV1,
    TaskPendingDispatchV1,
    TaskResolutionResultV1,
    TaskSpecialistRequestV1,
    TaskSpecialistResultV1,
    validate_task_resolution_checkpoint,
    validate_task_resolution_result,
    validate_task_specialist_request,
    validate_task_specialist_result,
)
__all__ = [
    "TaskResolutionCheckpointV1",
    "TaskResolutionContractError",
    "TaskResolutionExecutionContextV1",
    "TaskPendingDispatchV1",
    "TaskResolutionResultV1",
    "TaskSpecialistRequestV1",
    "TaskSpecialistResultV1",
    "resolve_task_inline",
    "resume_task_resolution",
    "validate_task_resolution_checkpoint",
    "validate_task_resolution_result",
    "validate_task_specialist_request",
    "validate_task_specialist_result",
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
    }:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    service_module = import_module("kazusa_ai_chatbot.task_resolution.service")
    resolved_value = getattr(service_module, name)
    globals()[name] = resolved_value
    return resolved_value
