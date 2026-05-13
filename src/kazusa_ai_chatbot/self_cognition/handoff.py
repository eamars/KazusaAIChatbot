"""Dispatcher handoff boundary for self-cognition action candidates."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from kazusa_ai_chatbot.dispatcher import TaskDispatcher
from kazusa_ai_chatbot.dispatcher.task import DispatchContext, RawToolCall
from kazusa_ai_chatbot.self_cognition import models


def build_raw_tool_call(action_candidate: dict[str, Any]) -> RawToolCall:
    """Convert a self-cognition candidate to the existing dispatcher shape.

    Args:
        action_candidate: Local action-candidate artifact emitted by tracking.

    Returns:
        Raw `send_message` tool call accepted by `TaskDispatcher`.

    Raises:
        ValueError: If the candidate is not a send-message candidate.
    """

    dispatch_shape = action_candidate.get("dispatch_shape")
    if dispatch_shape != models.ACTION_KIND_SEND_MESSAGE:
        raise ValueError("self-cognition candidate is not send_message")

    args = {
        "target_platform": str(action_candidate.get("target_platform") or ""),
        "target_channel": str(action_candidate.get("target_channel") or ""),
        "target_channel_type": str(
            action_candidate.get("target_channel_type") or ""
        ),
        "text": str(action_candidate.get("text") or ""),
    }
    execute_at = action_candidate.get("execute_at")
    if isinstance(execute_at, str) and execute_at.strip():
        args["execute_at"] = execute_at.strip()

    raw_call = RawToolCall(
        tool=models.ACTION_KIND_SEND_MESSAGE,
        args=args,
    )
    return raw_call


async def dispatch_action_candidate(
    case: models.SelfCognitionCase,
    action_attempt: dict[str, Any],
    action_candidate: dict[str, Any],
    dispatcher: TaskDispatcher,
    *,
    now: datetime,
) -> dict[str, Any]:
    """Hand one action candidate to the existing dispatcher.

    Args:
        case: Self-cognition case that produced the candidate.
        action_attempt: Action-attempt artifact tied to the candidate.
        action_candidate: Local action-candidate artifact.
        dispatcher: Existing service `TaskDispatcher`.
        now: Frozen dispatch time for immediate sends.

    Returns:
        Local dispatch-result artifact. Scheduler rows may be created only by
        the dispatcher, not by this module.
    """

    raw_call = build_raw_tool_call(action_candidate)
    dispatch_context = _build_dispatch_context(case, now=now)
    instruction = f"self_cognition:{_string_field(case, 'case_id')}"
    dispatch_result = await dispatcher.dispatch(
        [raw_call],
        dispatch_context,
        instruction=instruction,
    )
    scheduled_event_ids = [
        event_id
        for _, event_id in dispatch_result.scheduled
    ]
    rejections = [
        f"{raw.tool}: {reason}"
        for raw, reason in dispatch_result.rejected
    ]
    if scheduled_event_ids:
        status = "accepted"
    elif rejections:
        status = "rejected"
    else:
        status = "not_requested"

    result = {
        "attempt_id": action_attempt["attempt_id"],
        "idempotency_key": action_attempt["idempotency_key"],
        "production_handoff": bool(scheduled_event_ids),
        "dispatcher_called": True,
        "scheduled_event_ids": scheduled_event_ids,
        "rejections": rejections,
        "status": status,
    }
    return result


def _build_dispatch_context(
    case: models.SelfCognitionCase,
    *,
    now: datetime,
) -> DispatchContext:
    """Build source context for dispatcher validation and scheduler audit."""

    target_scope = _target_scope(case)
    source_user_id = target_scope["user_id"] or "self_cognition"
    context = DispatchContext(
        source_platform=target_scope["platform"],
        source_channel_id=target_scope["platform_channel_id"],
        source_user_id=source_user_id,
        source_message_id=f"self_cognition:{_string_field(case, 'case_id')}",
        guild_id=None,
        bot_permission_role="user",
        now=now,
        source_channel_type=target_scope["channel_type"],
    )
    return context


def _target_scope(case: models.SelfCognitionCase) -> dict[str, str | None]:
    """Read a case target scope as external input."""

    value = case.get("target_scope")
    if not isinstance(value, dict):
        value = {}
    raw_platform = value.get("platform")
    raw_platform_channel_id = value.get("platform_channel_id")
    raw_channel_type = value.get("channel_type")
    raw_user_id = value.get("user_id")
    scope = {
        "platform": raw_platform if isinstance(raw_platform, str) else "",
        "platform_channel_id": (
            raw_platform_channel_id
            if isinstance(raw_platform_channel_id, str)
            else ""
        ),
        "channel_type": (
            raw_channel_type
            if isinstance(raw_channel_type, str)
            else ""
        ),
        "user_id": raw_user_id if isinstance(raw_user_id, str) else None,
    }
    return scope


def _string_field(case: dict[str, Any], field_name: str) -> str:
    """Read an optional external string field safely."""

    value = case.get(field_name)
    if not isinstance(value, str):
        return_value = ""
        return return_value
    return_value = value
    return return_value
