"""Action-spec handler for accepted-task lifecycle status checks."""

from __future__ import annotations

from typing import Any

from kazusa_ai_chatbot.accepted_task import check_accepted_task_status
from kazusa_ai_chatbot.accepted_task.models import AcceptedTaskStatusResult
from kazusa_ai_chatbot.action_spec.models import (
    ActionValidationError,
    validate_action_spec,
)
from kazusa_ai_chatbot.action_spec.registry import (
    ACCEPTED_TASK_STATUS_CHECK_CAPABILITY,
)

_REQUIRED_STATUS_SCOPE_FIELDS = (
    "source_platform",
    "source_channel_id",
    "source_channel_type",
    "requester_global_user_id",
    "requester_platform_user_id",
)


def validate_accepted_task_status_check_action(
    action_spec: dict[str, Any],
) -> dict[str, Any]:
    """Validate one private accepted-task status-check action."""

    validated = validate_action_spec(action_spec)
    if validated["kind"] != ACCEPTED_TASK_STATUS_CHECK_CAPABILITY:
        raise ActionValidationError("kind: expected accepted_task_status_check")
    if validated["visibility"] != "private":
        raise ActionValidationError("visibility: expected private")

    target = validated["target"]
    if target["owner"] != "accepted_task":
        raise ActionValidationError("owner: expected accepted_task")
    if target["target_kind"] != "current_user":
        raise ActionValidationError("target_kind: expected current_user")
    if target["target_id"] is not None:
        raise ActionValidationError("target_id: expected null")
    scope = target["scope"]
    for field_name in _REQUIRED_STATUS_SCOPE_FIELDS:
        _require_non_empty_scope(scope, field_name)

    params = validated["params"]
    if params:
        raise ActionValidationError("params: expected empty status-check object")
    return validated


async def execute_accepted_task_status_check_action(
    action_spec: dict[str, Any],
) -> AcceptedTaskStatusResult:
    """Look up active accepted-task status without queueing new work."""

    validated = validate_accepted_task_status_check_action(action_spec)
    scope = validated["target"]["scope"]
    result = await check_accepted_task_status({
        "source_platform": _scope_text(scope, "source_platform"),
        "source_channel_id": _scope_text(scope, "source_channel_id"),
        "source_channel_type": _scope_text(scope, "source_channel_type"),
        "requester_global_user_id": _scope_text(
            scope,
            "requester_global_user_id",
        ),
        "requester_platform_user_id": _scope_text(
            scope,
            "requester_platform_user_id",
        ),
    })
    return result


def _require_non_empty_scope(scope: dict[str, Any], field_name: str) -> None:
    """Require one non-empty trusted target-scope string."""

    value = scope.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise ActionValidationError(
            f"scope.{field_name}: expected non-empty string"
        )


def _scope_text(scope: dict[str, Any], field_name: str) -> str:
    """Return one trusted scope text field."""

    value = scope.get(field_name)
    if not isinstance(value, str):
        return_value = ""
        return return_value
    return_value = value.strip()
    return return_value
