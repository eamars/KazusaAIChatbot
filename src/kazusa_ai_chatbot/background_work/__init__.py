"""Public entrypoints for generic background work."""

from kazusa_ai_chatbot.background_work.jobs import (
    enqueue_background_work_request,
)
from kazusa_ai_chatbot.background_work.models import (
    BackgroundWorkJobDoc,
    BackgroundWorkJobRef,
    BackgroundWorkQueueRequest,
    BackgroundWorkQueueResult,
    BackgroundWorkResult,
    BackgroundWorkRouterDecision,
    BackgroundWorkWorkerDecision,
)
from kazusa_ai_chatbot.background_work.runtime import (
    BackgroundWorkRuntimeHandle,
    run_background_work_runtime_tick,
    start_background_work_runtime,
    stop_background_work_runtime,
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
