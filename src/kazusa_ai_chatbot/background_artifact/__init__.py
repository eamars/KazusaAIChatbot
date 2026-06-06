"""Public background artifact runtime boundary."""

from __future__ import annotations

from typing import Any

from kazusa_ai_chatbot.background_artifact.models import (
    BackgroundArtifactJobDoc,
    BackgroundArtifactQueueRequest,
    BackgroundArtifactQueueResult,
)

__all__ = [
    "BackgroundArtifactJobDoc",
    "BackgroundArtifactQueueRequest",
    "BackgroundArtifactQueueResult",
    "BackgroundArtifactRuntimeHandle",
    "enqueue_background_artifact_request",
    "run_background_artifact_runtime_tick",
    "start_background_artifact_runtime",
    "stop_background_artifact_runtime",
]


def __getattr__(name: str) -> Any:
    """Resolve runtime helpers lazily to keep DB imports acyclic."""

    if name == "enqueue_background_artifact_request":
        module = __import__(
            "kazusa_ai_chatbot.background_artifact.jobs",
            fromlist=[name],
        )
        return getattr(module, name)
    if name in (
        "BackgroundArtifactRuntimeHandle",
        "run_background_artifact_runtime_tick",
        "start_background_artifact_runtime",
        "stop_background_artifact_runtime",
    ):
        module = __import__(
            "kazusa_ai_chatbot.background_artifact.runtime",
            fromlist=[name],
        )
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
