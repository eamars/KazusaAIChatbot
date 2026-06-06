"""Action-spec handler for background artifact queue requests."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from kazusa_ai_chatbot.action_spec.models import (
    ActionValidationError,
    validate_action_spec,
)
from kazusa_ai_chatbot.action_spec.registry import (
    BACKGROUND_ARTIFACT_REQUEST_CAPABILITY,
)
from kazusa_ai_chatbot.background_artifact import (
    enqueue_background_artifact_request,
)
from kazusa_ai_chatbot.background_artifact.models import (
    BACKGROUND_ARTIFACT_REQUESTED_DELIVERY,
    BACKGROUND_ARTIFACT_WORK_KINDS,
    BackgroundArtifactQueueRequest,
    BackgroundArtifactQueueResult,
)
from kazusa_ai_chatbot.config import BACKGROUND_ARTIFACT_OUTPUT_CHAR_LIMIT

BackgroundArtifactEnqueueFunc = Callable[
    [BackgroundArtifactQueueRequest],
    Awaitable[BackgroundArtifactQueueResult],
]

_REQUIRED_DELIVERY_TARGET_SCOPE_FIELDS = (
    "source_platform",
    "source_channel_id",
    "source_channel_type",
    "source_message_id",
    "source_platform_bot_id",
    "source_character_name",
    "requester_global_user_id",
    "requester_platform_user_id",
    "requester_display_name",
)


def validate_background_artifact_action(
    action_spec: dict[str, Any],
) -> dict[str, Any]:
    """Validate one private bounded background artifact queue action."""

    validated = validate_action_spec(action_spec)
    if validated["kind"] != BACKGROUND_ARTIFACT_REQUEST_CAPABILITY:
        raise ActionValidationError("kind: expected background_artifact_request")
    if validated["visibility"] != "private":
        raise ActionValidationError("visibility: expected private")
    if validated["urgency"] != "background":
        raise ActionValidationError("urgency: expected background")

    target = validated["target"]
    if target["owner"] != "background_artifact":
        raise ActionValidationError("owner: expected background_artifact")
    if target["target_kind"] != "current_user":
        raise ActionValidationError("target_kind: expected current_user")
    if target["target_id"] is not None:
        raise ActionValidationError("target_id: expected null")
    scope = target["scope"]
    for field_name in _REQUIRED_DELIVERY_TARGET_SCOPE_FIELDS:
        _require_non_empty_scope(scope, field_name)

    params = validated["params"]
    work_kind = params.get("work_kind")
    if work_kind not in BACKGROUND_ARTIFACT_WORK_KINDS:
        raise ActionValidationError("work_kind: unsupported background artifact")
    requested_delivery = params.get("requested_delivery")
    if requested_delivery != BACKGROUND_ARTIFACT_REQUESTED_DELIVERY:
        raise ActionValidationError("requested_delivery: unsupported value")
    _require_non_empty_param(params, "objective")
    _require_non_empty_param(params, "input_summary")
    max_output_chars = params.get("max_output_chars")
    if not isinstance(max_output_chars, int):
        raise ActionValidationError("max_output_chars: expected integer")
    if max_output_chars < 1:
        raise ActionValidationError("max_output_chars: expected positive integer")
    if max_output_chars > BACKGROUND_ARTIFACT_OUTPUT_CHAR_LIMIT:
        raise ActionValidationError("max_output_chars: exceeds configured limit")
    return validated


async def enqueue_background_artifact_action(
    action_spec: dict[str, Any],
    *,
    storage_timestamp_utc: str,
    action_attempt_id: str,
    enqueue_background_artifact_func: BackgroundArtifactEnqueueFunc
    | None = None,
) -> BackgroundArtifactQueueResult:
    """Persist one validated background artifact queue action."""

    validated = validate_background_artifact_action(action_spec)
    params = validated["params"]
    scope = validated["target"]["scope"]
    queue_request: BackgroundArtifactQueueRequest = {
        "action_attempt_id": action_attempt_id,
        "idempotency_key": f"background_artifact:{action_attempt_id}",
        "work_kind": _param_text(params, "work_kind"),
        "objective": _param_text(params, "objective"),
        "input_summary": _param_text(params, "input_summary"),
        "requested_delivery": "send_result_when_done",
        "max_output_chars": int(params["max_output_chars"]),
        "source_platform": _scope_text(scope, "source_platform"),
        "source_channel_id": _scope_text(scope, "source_channel_id"),
        "source_channel_type": _scope_text(scope, "source_channel_type"),
        "source_message_id": _scope_text(scope, "source_message_id"),
        "source_platform_bot_id": _scope_text(scope, "source_platform_bot_id"),
        "source_character_name": _scope_text(scope, "source_character_name"),
        "requester_global_user_id": _scope_text(
            scope,
            "requester_global_user_id",
        ),
        "requester_platform_user_id": _scope_text(
            scope,
            "requester_platform_user_id",
        ),
        "requester_display_name": _scope_text(scope, "requester_display_name"),
        "storage_timestamp_utc": storage_timestamp_utc,
    }
    if enqueue_background_artifact_func is None:
        enqueue_background_artifact_func = enqueue_background_artifact_request
    result = await enqueue_background_artifact_func(queue_request)
    return result


def _require_non_empty_param(params: dict[str, Any], field_name: str) -> None:
    """Require one non-empty string param."""

    value = params.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise ActionValidationError(f"{field_name}: expected non-empty string")


def _require_non_empty_scope(scope: dict[str, Any], field_name: str) -> None:
    """Require one non-empty trusted target-scope string."""

    value = scope.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise ActionValidationError(
            f"scope.{field_name}: expected non-empty string"
        )


def _param_text(params: dict[str, Any], field_name: str) -> str:
    """Return one previously validated text param."""

    value = params[field_name]
    if not isinstance(value, str):
        raise ActionValidationError(f"{field_name}: expected string")
    return_value = value.strip()
    return return_value


def _scope_text(scope: dict[str, Any], field_name: str) -> str:
    """Return one trusted scope text field."""

    value = scope.get(field_name)
    if not isinstance(value, str):
        return_value = ""
        return return_value
    return_value = value.strip()
    return return_value
