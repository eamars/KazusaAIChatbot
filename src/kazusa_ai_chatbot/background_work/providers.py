"""Deterministic dispatch from background-work route decisions to workers."""

from __future__ import annotations

from collections.abc import Mapping

from kazusa_ai_chatbot.background_work.models import (
    BackgroundWorkResult,
    BackgroundWorkRouterDecision,
)
from kazusa_ai_chatbot.background_work.subagent import (
    discover_background_work_workers,
)
from kazusa_ai_chatbot.config import BACKGROUND_WORK_OUTPUT_CHAR_LIMIT


def load_background_work_workers() -> dict[str, object]:
    """Return the current background-work worker registry."""

    workers = discover_background_work_workers()
    return workers


async def execute_background_work_decision(
    decision: Mapping[str, object],
) -> BackgroundWorkResult:
    """Execute one sanitized background-work route decision."""

    sanitized_decision: BackgroundWorkRouterDecision = {
        "action": _decision_text(decision, "action"),
        "worker": _decision_text(decision, "worker"),
        "task": _decision_text(decision, "task"),
        "reason": _decision_text(decision, "reason"),
    }
    result = await dispatch_background_work(sanitized_decision)
    return result


async def dispatch_background_work(
    decision: BackgroundWorkRouterDecision,
    *,
    max_output_chars: int = BACKGROUND_WORK_OUTPUT_CHAR_LIMIT,
) -> BackgroundWorkResult:
    """Execute one route decision through its selected worker."""

    if decision["action"] != "execute":
        result: BackgroundWorkResult = {
            "status": _status_for_route_action(decision["action"]),
            "worker": decision["worker"],
            "artifact_text": "",
            "failure_summary": decision["reason"],
            "result_summary": decision["reason"],
            "worker_metadata": {},
        }
        return result

    workers = load_background_work_workers()
    worker = workers.get(decision["worker"])
    if worker is None:
        result = {
            "status": "rejected",
            "worker": decision["worker"],
            "artifact_text": "",
            "failure_summary": "Unsupported background-work worker selected.",
            "result_summary": "Selected worker unavailable.",
            "worker_metadata": {},
        }
        return result

    execute_func = _worker_execute_func(worker)
    result = await execute_func(
        decision,
        max_output_chars=max_output_chars,
    )
    return result


def _worker_execute_func(worker: object):
    """Return a worker execute callable from a registry row or module object."""

    if isinstance(worker, dict):
        execute_func = worker["execute"]
        return execute_func
    execute_func = getattr(worker, "execute")
    return execute_func


def _decision_text(decision: Mapping[str, object], field_name: str) -> str:
    """Read one route decision text field."""

    value = decision.get(field_name)
    if not isinstance(value, str):
        return ""
    return value.strip()


def _status_for_route_action(action: str) -> str:
    """Map route-only action labels into worker result status labels."""

    if action == "needs_user_input":
        return_value = "needs_user_input"
        return return_value
    if action == "reject":
        return_value = "rejected"
        return return_value
    return_value = "failed"
    return return_value
