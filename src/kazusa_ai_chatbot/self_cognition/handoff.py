"""Dispatcher handoff boundary for self-cognition action candidates."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from kazusa_ai_chatbot.action_spec.evaluator import (
    build_raw_tool_call_from_action_spec,
)
from kazusa_ai_chatbot.action_spec.registry import SEND_MESSAGE_CAPABILITY
from kazusa_ai_chatbot.dispatcher import TaskDispatcher
from kazusa_ai_chatbot.dispatcher.task import DispatchContext
from kazusa_ai_chatbot.self_cognition import models


def build_send_message_action_spec(
    case: models.SelfCognitionCase,
    action_candidate: dict[str, Any],
) -> dict[str, Any]:
    """Convert a legacy self-cognition delivery candidate to ActionSpecV1.

    Args:
        case: Self-cognition trigger case that selected the candidate.
        action_candidate: Legacy local action-candidate artifact.

    Returns:
        A send-message action spec that must pass the shared action evaluator
        before entering the dispatcher bridge.

    Raises:
        ValueError: If the candidate is not a send-message candidate.
    """

    dispatch_shape = action_candidate.get("dispatch_shape")
    if dispatch_shape != models.ACTION_KIND_SEND_MESSAGE:
        raise ValueError("self-cognition candidate is not send_message")

    execute_at = action_candidate.get("execute_at")
    if not isinstance(execute_at, str) or not execute_at.strip():
        execute_at = None
    delivery_mentions = action_candidate.get("delivery_mentions")
    if not isinstance(delivery_mentions, list):
        delivery_mentions = []
    action_spec = {
        "schema_version": "action_spec.v1",
        "kind": SEND_MESSAGE_CAPABILITY,
        "cognition_mode": "deliberative",
        "source_refs": _action_source_refs(case),
        "target": {
            "schema_version": "action_target.v1",
            "target_kind": "current_channel",
            "target_id": None,
            "owner": "dispatcher",
            "scope": {"channel_relation": "same"},
        },
        "params": {
            "target_channel": "same",
            "text": str(action_candidate.get("text") or ""),
            "execute_at": execute_at,
            "delivery_mentions": delivery_mentions,
        },
        "urgency": "scheduled" if execute_at is not None else "now",
        "visibility": "user_visible",
        "deadline": execute_at,
        "continuation": {
            "schema_version": "action_continuation.v1",
            "mode": "none",
            "episode_type": None,
            "max_depth": 0,
            "include_result_as": None,
        },
        "reason": "Self-cognition selected a user-visible follow-up.",
    }
    return action_spec


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

    action_spec = build_send_message_action_spec(case, action_candidate)
    raw_call = build_raw_tool_call_from_action_spec(action_spec)
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
        "action_spec_schema_version": action_spec["schema_version"],
        "action_spec_kind": action_spec["kind"],
    }
    return result


def _build_dispatch_context(
    case: models.SelfCognitionCase,
    *,
    now: datetime,
) -> DispatchContext:
    """Build source context for dispatcher validation and scheduler audit."""

    target_scope = _target_scope(case)
    source_user_id = target_scope["user_id"] or ""
    context = DispatchContext(
        source_platform=target_scope["platform"],
        source_channel_id=target_scope["platform_channel_id"],
        source_user_id=source_user_id,
        source_message_id=f"self_cognition:{_string_field(case, 'case_id')}",
        guild_id=None,
        bot_permission_role="user",
        now=now,
        source_channel_type=target_scope["channel_type"],
        source_platform_bot_id=_string_field(case, "platform_bot_id"),
        source_character_name=_character_name(case),
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


def _action_source_refs(
    case: models.SelfCognitionCase,
) -> list[dict[str, Any]]:
    """Project self-cognition source refs into action-spec references."""

    raw_refs = case.get("source_refs")
    if not isinstance(raw_refs, list):
        raw_refs = []

    source_refs = []
    for raw_ref in raw_refs:
        if not isinstance(raw_ref, dict):
            continue
        source_kind = raw_ref.get("source_kind")
        source_id = raw_ref.get("source_id")
        if not isinstance(source_id, str) or not source_id.strip():
            continue
        if source_kind == "user_memory_unit":
            ref_kind = "memory_unit"
            owner = "user_memory_units"
        else:
            ref_kind = "system_event"
            owner = "self_cognition"
        source_refs.append(
            {
                "schema_version": "action_source_ref.v1",
                "ref_kind": ref_kind,
                "ref_id": source_id,
                "owner": owner,
                "relationship": "basis",
                "evidence_refs": [],
            }
        )

    if not source_refs:
        source_refs.append(
            {
                "schema_version": "action_source_ref.v1",
                "ref_kind": "system_event",
                "ref_id": _string_field(case, "case_id") or "self_cognition",
                "owner": "self_cognition",
                "relationship": "basis",
                "evidence_refs": [],
            }
        )
    return source_refs


def _string_field(case: dict[str, Any], field_name: str) -> str:
    """Read an optional external string field safely."""

    value = case.get(field_name)
    if not isinstance(value, str):
        return_value = ""
        return return_value
    return_value = value
    return return_value


def _character_name(case: models.SelfCognitionCase) -> str:
    """Read the active character name from the case profile."""

    profile = case.get("character_profile")
    if not isinstance(profile, dict):
        return_value = ""
        return return_value
    name = profile.get("name")
    if not isinstance(name, str):
        return_value = ""
        return return_value
    return_value = name
    return return_value
