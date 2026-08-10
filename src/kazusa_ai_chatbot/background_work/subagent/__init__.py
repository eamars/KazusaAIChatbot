"""Owned execution entrypoints for reviewed background-work payloads."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "execute_future_speak_job",
    "execute_task_orchestrator_job",
]


_ENTRYPOINT_MODULES = {
    "execute_future_speak_job": (
        "kazusa_ai_chatbot.background_work.subagent.future_speak"
    ),
    "execute_task_orchestrator_job": (
        "kazusa_ai_chatbot.background_work.subagent.task_orchestrator"
    ),
}


def __getattr__(name: str) -> Any:
    """Resolve a worker implementation only when its payload is claimed."""

    module_name = _ENTRYPOINT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name)
    resolved_value = getattr(module, name)
    globals()[name] = resolved_value
    return resolved_value
