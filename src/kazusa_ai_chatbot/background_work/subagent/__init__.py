"""Worker discovery for background-work subagents."""

from __future__ import annotations

from typing import Protocol, TypedDict

from kazusa_ai_chatbot.background_work.models import (
    BackgroundWorkResult,
    BackgroundWorkWorkerDecision,
)
from kazusa_ai_chatbot.background_work.subagent import future_speak, text_artifact


class BackgroundWorkWorker(TypedDict):
    """Public worker registry row."""

    worker: str
    description: str
    execute: "BackgroundWorkExecute"


class BackgroundWorkExecute(Protocol):
    """Callable contract for one background-work worker entrypoint."""

    async def __call__(
        self,
        decision: BackgroundWorkWorkerDecision,
        *,
        max_output_chars: int,
    ) -> BackgroundWorkResult:
        """Execute one route decision with deterministic runtime context."""


def discover_background_work_workers() -> dict[str, BackgroundWorkWorker]:
    """Return the enabled background-work worker registry."""

    workers: dict[str, BackgroundWorkWorker] = {}
    for module in (future_speak, text_artifact):
        worker_name = str(getattr(module, "WORKER"))
        description = str(getattr(module, "DESCRIPTION"))
        execute_func = getattr(module, "execute")
        workers[worker_name] = {
            "worker": worker_name,
            "description": description,
            "execute": execute_func,
        }
    return workers


def worker_descriptions() -> dict[str, str]:
    """Return prompt-safe worker descriptions for the router."""

    workers = discover_background_work_workers()
    descriptions = {
        worker_name: str(worker["description"])
        for worker_name, worker in workers.items()
    }
    return descriptions
