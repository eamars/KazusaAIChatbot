"""Handler helpers for future self-cognition trigger actions."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from kazusa_ai_chatbot import scheduler
from kazusa_ai_chatbot.action_spec.models import (
    ACTION_CONTINUATION_VERSION,
    ActionValidationError,
    validate_action_spec,
)
from kazusa_ai_chatbot.action_spec.registry import (
    TRIGGER_FUTURE_COGNITION_CAPABILITY,
)

MAX_FUTURE_COGNITION_CONTINUATION_DEPTH = 1
FUTURE_COGNITION_TOOL = "trigger_future_cognition"
_FORBIDDEN_RAW_ID_PARAM_KEYS = frozenset(
    (
        "adapter_id",
        "channel_id",
        "collection",
        "database",
        "event_id",
        "handler_id",
        "platform_channel_id",
        "raw_id",
        "scheduler_id",
        "target_channel",
    )
)


def build_future_cognition_scheduled_event(
    action_spec: dict[str, Any],
    *,
    timestamp: str,
    action_attempt_id: str,
) -> dict:
    """Build a scheduler document for a future self-cognition slot.

    Args:
        action_spec: Selected ``trigger_future_cognition`` action spec.
        timestamp: Current episode timestamp used as creation time and as the
            immediate background slot time when no trigger time was supplied.
        action_attempt_id: Stable action-attempt id for trace correlation.

    Returns:
        Scheduler-shaped event document containing prompt-safe trigger args.

    Raises:
        ActionValidationError: If the action is not a private bounded
            self-cognition trigger contract.
    """

    if not timestamp.strip():
        raise ActionValidationError("timestamp: expected non-empty string")
    if not action_attempt_id.strip():
        raise ActionValidationError("action_attempt_id: expected non-empty string")

    validated = validate_future_cognition_action(action_spec)
    params = validated["params"]
    source_scope = _source_scope(validated["target"]["scope"])
    trigger_at = params["trigger_at"]
    if trigger_at is None:
        execute_at = _normalized_iso_datetime(timestamp)
        normalized_trigger_at = None
    else:
        normalized_trigger_at = _normalized_absolute_iso_datetime(trigger_at)
        execute_at = normalized_trigger_at

    continuation_objective = str(params["continuation_objective"]).strip()
    event = {
        "event_id": f"future_cognition:{action_attempt_id}",
        "tool": FUTURE_COGNITION_TOOL,
        "args": {
            "episode_type": "self_cognition",
            "trigger_at": normalized_trigger_at,
            "continuation_objective": continuation_objective,
            "source_action_attempt_id": action_attempt_id,
            "source_refs": list(validated["source_refs"]),
            "continuation": dict(validated["continuation"]),
        },
        "execute_at": execute_at,
        "created_at": _normalized_iso_datetime(timestamp),
        "status": "pending",
        "source_platform": source_scope["source_platform"],
        "source_channel_id": source_scope["source_channel_id"],
        "source_channel_type": source_scope["source_channel_type"],
        "source_user_id": source_scope["source_user_id"],
        "source_message_id": action_attempt_id,
        "source_platform_bot_id": source_scope["source_platform_bot_id"],
        "source_character_name": source_scope["source_character_name"],
        "guild_id": None,
        "bot_role": "system",
    }
    return event


async def execute_future_cognition_action(
    action_spec: dict[str, Any],
    *,
    timestamp: str,
    action_attempt_id: str,
) -> dict:
    """Persist a future self-cognition slot without running cognition inline.

    Args:
        action_spec: Selected ``trigger_future_cognition`` action spec.
        timestamp: Current episode timestamp.
        action_attempt_id: Stable action-attempt id for trace correlation.

    Returns:
        Prompt-safe execution result containing scheduled event ids.
    """

    event = build_future_cognition_scheduled_event(
        action_spec,
        timestamp=timestamp,
        action_attempt_id=action_attempt_id,
    )
    event_id = await scheduler.schedule_event(event)
    result = {
        "status": "scheduled",
        "scheduled_event_ids": [event_id],
        "episode_type": "self_cognition",
        "trigger_at": event["args"]["trigger_at"],
        "reason": str(action_spec["reason"]),
    }
    return result


def validate_future_cognition_action(
    action_spec: dict[str, Any],
) -> dict[str, Any]:
    """Validate a private bounded future self-cognition trigger action."""

    validated = validate_action_spec(action_spec)
    if validated["kind"] != TRIGGER_FUTURE_COGNITION_CAPABILITY:
        raise ActionValidationError("kind: expected trigger_future_cognition")
    if validated["visibility"] != "private":
        raise ActionValidationError("visibility: expected private")
    if validated["urgency"] not in ("background", "scheduled"):
        raise ActionValidationError("urgency: expected background or scheduled")

    target = validated["target"]
    if target["owner"] != "orchestrator":
        raise ActionValidationError("owner: expected orchestrator")
    if target["target_kind"] != "cognitive_episode":
        raise ActionValidationError("target_kind: expected cognitive_episode")
    if target["target_id"] is not None:
        raise ActionValidationError("target_id: expected null")

    params = validated["params"]
    if params.get("episode_type") != "self_cognition":
        raise ActionValidationError("episode_type: expected self_cognition")
    trigger_at = params.get("trigger_at")
    if trigger_at is not None:
        if not isinstance(trigger_at, str):
            raise ActionValidationError("trigger_at: expected string or null")
        _normalized_absolute_iso_datetime(trigger_at)
    continuation_objective = params.get("continuation_objective")
    if (
        not isinstance(continuation_objective, str)
        or not continuation_objective.strip()
    ):
        raise ActionValidationError(
            "continuation_objective: expected non-empty string"
        )
    _reject_raw_id_params(params)
    _validate_bounded_continuation(validated["continuation"])
    return_value = validated
    return return_value


def _validate_bounded_continuation(continuation: dict[str, Any]) -> None:
    """Require continuation metadata to stay within the bounded contract."""

    if continuation["schema_version"] != ACTION_CONTINUATION_VERSION:
        raise ActionValidationError("continuation.schema_version: unsupported")
    max_depth = continuation["max_depth"]
    if max_depth > MAX_FUTURE_COGNITION_CONTINUATION_DEPTH:
        raise ActionValidationError("max_depth: exceeds future cognition bound")


def _reject_raw_id_params(params: dict[str, Any]) -> None:
    """Reject L2d-authored params that look like delivery or storage ids."""

    for key in params:
        normalized_key = key.strip().casefold()
        if normalized_key in _FORBIDDEN_RAW_ID_PARAM_KEYS:
            raise ActionValidationError(f"{key}: raw id params are not allowed")


def _source_scope(scope: dict[str, Any]) -> dict[str, str]:
    """Return trusted scheduler source fields from deterministic target scope."""

    source_scope = {
        "source_platform": _scope_text(scope, "source_platform") or "orchestrator",
        "source_channel_id": _scope_text(scope, "source_channel_id"),
        "source_channel_type": _scope_text(scope, "source_channel_type")
        or "internal",
        "source_user_id": _scope_text(scope, "source_user_id")
        or "self_cognition",
        "source_platform_bot_id": _scope_text(scope, "source_platform_bot_id"),
        "source_character_name": _scope_text(scope, "source_character_name"),
    }
    return source_scope


def _scope_text(scope: dict[str, Any], field_name: str) -> str:
    """Read one optional text value from deterministic target scope."""

    value = scope.get(field_name)
    if not isinstance(value, str):
        return_value = ""
        return return_value
    return_value = value.strip()
    return return_value


def _normalized_iso_datetime(value: str) -> str:
    """Normalize an ISO timestamp into UTC ISO format."""

    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ActionValidationError(
            f"timestamp: invalid ISO timestamp: {exc}"
        ) from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    normalized = parsed.astimezone(timezone.utc).isoformat()
    return normalized


def _normalized_absolute_iso_datetime(value: str) -> str:
    """Normalize an absolute ISO timestamp or raise a validation error."""

    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ActionValidationError(
            f"trigger_at: invalid ISO timestamp: {exc}"
        ) from exc
    if parsed.tzinfo is None:
        raise ActionValidationError("trigger_at: expected absolute timestamp")
    normalized = parsed.astimezone(timezone.utc).isoformat()
    return normalized
