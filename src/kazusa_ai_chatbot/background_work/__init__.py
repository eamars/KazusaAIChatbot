"""Public entrypoints for generic background work."""

from __future__ import annotations

from typing import Any

from kazusa_ai_chatbot.background_work.models import (
    FUTURE_SPEAK_WORKER,
    TASK_ORCHESTRATOR_WORKER,
    BackgroundWorkJobDoc,
    BackgroundWorkJobRef,
    BackgroundWorkQueueRequest,
    BackgroundWorkQueueResult,
    FutureSpeakWorkerPayloadV1,
    TaskOrchestratorWorkerPayloadV2,
)

__all__ = [
    "FUTURE_SPEAK_WORKER",
    "TASK_ORCHESTRATOR_WORKER",
    "BackgroundWorkJobDoc",
    "BackgroundWorkJobRef",
    "BackgroundWorkQueueRequest",
    "BackgroundWorkQueueResult",
    "BackgroundWorkRuntimeHandle",
    "FutureSpeakWorkerPayloadV1",
    "TaskOrchestratorWorkerPayloadV2",
    "enqueue_background_work_request",
    "run_background_work_runtime_tick",
    "start_background_work_runtime",
    "stop_background_work_runtime",
]


def __getattr__(name: str) -> Any:
    """Resolve runtime helpers lazily to keep DB imports acyclic."""

    if name == "enqueue_background_work_request":
        module = __import__(
            "kazusa_ai_chatbot.background_work.jobs",
            fromlist=[name],
        )
        resolved_value = getattr(module, name)
        return resolved_value
    if name in (
        "BackgroundWorkRuntimeHandle",
        "run_background_work_runtime_tick",
        "start_background_work_runtime",
        "stop_background_work_runtime",
    ):
        module = __import__(
            "kazusa_ai_chatbot.background_work.runtime",
            fromlist=[name],
        )
        resolved_value = getattr(module, name)
        return resolved_value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
