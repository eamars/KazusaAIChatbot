"""Public entrypoints for generic background work."""

from __future__ import annotations

from typing import Any

from kazusa_ai_chatbot.background_work.models import (
    BackgroundWorkJobDoc,
    BackgroundWorkJobRef,
    BackgroundWorkQueueRequest,
    BackgroundWorkQueueResult,
    BackgroundWorkResult,
    BackgroundWorkRouterDecision,
    BackgroundWorkWorkerDecision,
)

__all__ = [
    "BackgroundWorkJobDoc",
    "BackgroundWorkJobRef",
    "BackgroundWorkQueueRequest",
    "BackgroundWorkQueueResult",
    "BackgroundWorkResult",
    "BackgroundWorkRouterDecision",
    "BackgroundWorkWorkerDecision",
    "BackgroundWorkRuntimeHandle",
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
