"""Normalization contracts for action-router raw model output."""

from __future__ import annotations

import logging

from kazusa_ai_chatbot.action_spec.registry import (
    BACKGROUND_WORK_REQUEST_CAPABILITY,
    MEMORY_LIFECYCLE_UPDATE_CAPABILITY,
    SPEAK_CAPABILITY,
    TRIGGER_FUTURE_COGNITION_CAPABILITY,
)
from kazusa_ai_chatbot.cognition_resolver.contracts import (
    ALLOWED_RESOLVER_CAPABILITIES,
    RESOLVER_CAPABILITY_REQUEST_VERSION,
)

logger = logging.getLogger(__name__)

_ALLOWED_ACTION_CAPABILITIES = frozenset((
    SPEAK_CAPABILITY,
    MEMORY_LIFECYCLE_UPDATE_CAPABILITY,
    TRIGGER_FUTURE_COGNITION_CAPABILITY,
    BACKGROUND_WORK_REQUEST_CAPABILITY,
))

_BACKGROUND_WORK_FORBIDDEN_FIELDS = frozenset((
    "task_brief",
    "worker",
    "task_type",
    "tool_args",
    "work_kind",
    "artifact_text",
    "file_path",
))

_RESOLVER_FORBIDDEN_FIELDS = frozenset((
    "schema_version",
    "resume_id",
    "pending_row_id",
    "resolver_id",
))

ACTION_SPEC_CAP = 3


def normalize_action_router_output(raw: object) -> dict[str, object]:
    """Normalize raw action-router model output into the route-only contract.

    Strips forbidden fields from resolver requests and action requests.
    Returns a dict with resolver_capability_requests, resolver_pending_resolution,
    resolver_goal_progress, and action_requests.
    """

    if not isinstance(raw, dict):
        return_value = {
            "resolver_capability_requests": [],
            "resolver_pending_resolution": None,
            "resolver_goal_progress": None,
            "action_requests": [],
        }
        return return_value

    resolver_capability_requests = _normalize_resolver_requests(
        raw.get("resolver_capability_requests"),
    )
    resolver_pending_resolution = _normalize_pending_resolution(
        raw.get("resolver_pending_resolution"),
    )
    resolver_goal_progress = _normalize_goal_progress(
        raw.get("resolver_goal_progress"),
    )
    action_requests = _normalize_action_requests(
        raw.get("action_requests"),
    )

    return_value = {
        "resolver_capability_requests": resolver_capability_requests,
        "resolver_pending_resolution": resolver_pending_resolution,
        "resolver_goal_progress": resolver_goal_progress,
        "action_requests": action_requests,
    }
    return return_value


def _normalize_resolver_requests(
    raw_requests: object,
) -> list[dict[str, object]]:
    """Strip forbidden fields from resolver capability requests."""

    if not isinstance(raw_requests, list):
        return []

    normalized: list[dict[str, object]] = []
    for raw in raw_requests:
        if not isinstance(raw, dict):
            continue
        capability_kind = _text_field(raw, "capability_kind")
        if capability_kind not in ALLOWED_RESOLVER_CAPABILITIES:
            logger.warning(
                f"Action router dropped unsupported resolver capability: "
                f"{capability_kind}"
            )
            continue
        objective = _text_field(raw, "objective")
        reason = _text_field(raw, "reason")
        priority = _text_field(raw, "priority")
        if not objective or not reason:
            logger.warning(
                "Action router dropped resolver request without objective "
                "or reason"
            )
            continue
        if priority not in ("now", "background"):
            priority = "now"
        cleaned: dict[str, object] = {
            "schema_version": RESOLVER_CAPABILITY_REQUEST_VERSION,
            "capability_kind": capability_kind,
            "objective": objective,
            "reason": reason,
            "priority": priority,
        }
        normalized.append(cleaned)
        if len(normalized) >= ACTION_SPEC_CAP:
            break
    return normalized


def _normalize_pending_resolution(
    raw: object,
) -> dict[str, object] | None:
    """Pass through pending resolution if structurally valid."""

    if not isinstance(raw, dict):
        return None
    decision = _text_field(raw, "decision")
    if not decision:
        return None
    return_value = {
        "decision": decision,
        "reason": _text_field(raw, "reason"),
    }
    return return_value


def _normalize_goal_progress(raw: object) -> dict[str, object] | None:
    """Pass through goal progress without model-supplied schema metadata."""

    if not isinstance(raw, dict):
        return_value = None
        return return_value
    cleaned = {
        key: value
        for key, value in raw.items()
        if key not in _RESOLVER_FORBIDDEN_FIELDS
    }
    return_value: dict[str, object] = cleaned
    return return_value


def _normalize_action_requests(
    raw_requests: object,
) -> list[dict[str, object]]:
    """Strip forbidden fields from action requests."""

    if not isinstance(raw_requests, list):
        return []

    normalized: list[dict[str, object]] = []
    for raw in raw_requests:
        if not isinstance(raw, dict):
            continue
        capability = _text_field(raw, "capability")
        reason = _text_field(raw, "reason")
        if not capability or not reason:
            logger.warning(
                "Action router dropped action request without capability or reason"
            )
            continue
        if capability not in _ALLOWED_ACTION_CAPABILITIES:
            logger.warning(
                f"Action router dropped unsupported action capability: "
                f"{capability}"
            )
            continue
        cleaned: dict[str, object] = {
            "capability": capability,
            "reason": reason,
        }
        decision = _text_field(raw, "decision")
        if decision:
            cleaned["decision"] = decision
        detail = _text_field(raw, "detail")
        if detail:
            cleaned["detail"] = detail
        if capability == "background_work_request":
            for forbidden in _BACKGROUND_WORK_FORBIDDEN_FIELDS:
                cleaned.pop(forbidden, None)
        normalized.append(cleaned)
        if len(normalized) >= ACTION_SPEC_CAP:
            break
    return normalized


def _text_field(obj: dict[str, object], key: str) -> str:
    """Read one optional text field from model output."""

    value = obj.get(key)
    if not isinstance(value, str):
        return_value = ""
        return return_value
    return_value = value.strip()
    return return_value
