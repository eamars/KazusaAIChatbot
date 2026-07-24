"""Tracking records and route ownership for self-cognition runs."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from kazusa_ai_chatbot.action_spec.registry import SPEAK_CAPABILITY
from kazusa_ai_chatbot.brain_service.delivery_mentions import (
    build_inline_delivery_mentions,
)
from kazusa_ai_chatbot.self_cognition import models

PRODUCTION_HANDOFF_ENABLED = False


def build_idempotency_key(
    source_kind: str,
    source_id: str,
    due_at: str | None,
    target_scope: dict[str, Any],
    action_kind: str,
) -> str:
    """Build the stable duplicate-suppression identity for one action.

    Args:
        source_kind: Source category that produced the possible action.
        source_id: Stable source identifier, such as a promise id.
        due_at: Due occurrence represented by the trigger, if any.
        target_scope: Platform and channel target for the action.
        action_kind: Outward action kind, currently `send_message`.

    Returns:
        SHA-256 idempotency key that excludes generated message text.
    """

    normalized_scope = _normalized_target_scope(target_scope)
    identity = {
        "source_kind": source_kind,
        "source_id": source_id,
        "due_at": due_at,
        "target_scope": normalized_scope,
        "action_kind": action_kind,
    }
    rendered = json.dumps(identity, sort_keys=True, ensure_ascii=False)
    digest = hashlib.sha256(rendered.encode("utf-8")).hexdigest()
    key = f"sha256:{digest}"
    return key


def build_trigger_record(case: models.SelfCognitionCase) -> dict[str, Any]:
    """Build the local trigger record explaining why the run exists.

    Args:
        case: Self-cognition source case.

    Returns:
        Trigger record for local tracking.
    """

    target_scope = _case_target_scope(case)
    source_refs = _case_source_refs(case)
    trigger_identity = {
        "case_name": _string_field(case, "case_name"),
        "case_id": _string_field(case, "case_id"),
        "trigger_kind": _string_field(case, "trigger_kind"),
        "target_scope": target_scope,
        "source_refs": source_refs,
        "idle_timestamp_utc": _string_field(case, "idle_timestamp_utc"),
    }
    trigger_id = _stable_prefixed_id(
        "self_cognition_trigger",
        trigger_identity,
    )
    trigger_record = {
        "trigger_id": trigger_id,
        "trigger_kind": _string_field(case, "trigger_kind"),
        "target_scope": target_scope,
        "source_refs": source_refs,
        "idle_timestamp_utc": _string_field(case, "idle_timestamp_utc"),
        "semantic_due_state": _optional_string_field(
            case,
            "semantic_due_state",
        ),
        "actionability": _string_field(case, "actionability"),
        "status": "accepted",
    }
    return trigger_record


def build_run_record(
    case: models.SelfCognitionCase,
    trigger_record: dict[str, Any],
    selected_route: str,
    budget: dict[str, int],
) -> dict[str, Any]:
    """Build the run record for the selected route.

    Args:
        case: Self-cognition source case.
        trigger_record: Trigger record returned by `build_trigger_record`.
        selected_route: Route selected by cognition and deterministic tracking.
        budget: Local budget counters.

    Returns:
        Run record for local tracking.
    """

    output_mode = "scheduled_action_request"
    if selected_route != models.ROUTE_ACTION_CANDIDATE:
        output_mode = "silent"

    run_record = {
        "run_id": f"self_cognition_run:{trigger_record['trigger_id']}",
        "trigger_id": trigger_record["trigger_id"],
        "idle_timestamp_utc": _string_field(case, "idle_timestamp_utc"),
        "output_mode": output_mode,
        "selected_route": selected_route,
        "status": "completed",
        "evidence_refs": _case_source_refs(case),
        "budget": {
            "rag_calls": int(budget["rag_calls"]),
            "cognition_calls": int(budget["cognition_calls"]),
            "dialog_calls": int(budget["dialog_calls"]),
            "topic_limit": int(budget["topic_limit"]),
        },
    }
    return run_record


def build_route_effect(
    run_record: dict[str, Any],
    route: str,
    consumer: str,
    effect_summary: str,
    next_topic: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the route-effect record for the selected consumer.

    Args:
        run_record: Run record returned by `build_run_record`.
        route: Selected route name.
        consumer: Component that receives the tracking effect.
        effect_summary: Human-readable effect summary.
        next_topic: Optional bounded follow-up topic descriptor.

    Returns:
        Route effect with `production_write` fixed to false.
    """

    route_effect = {
        "run_id": run_record["run_id"],
        "route": route,
        "consumer": consumer,
        "production_write": False,
        "effect_summary": effect_summary,
        "next_topic": next_topic,
    }
    return route_effect


def build_consolidation_outcome_record(
    consolidation_state: dict[str, Any],
    consolidation_result: dict[str, Any],
) -> dict[str, Any]:
    """Build sanitized metadata for a self-cognition consolidation call.

    Args:
        consolidation_state: Self-cognition state passed to the shared
            consolidator.
        consolidation_result: Result returned by the shared consolidator.

    Returns:
        Structural outcome metadata without prompt, packet, dialog, or DB row
        bodies.
    """

    episode = consolidation_state["cognitive_episode"]
    metadata = consolidation_result["consolidation_metadata"]
    if not isinstance(metadata, dict):
        raise ValueError("consolidation_metadata must be a dict")

    required_metadata_fields = ("write_success", "cache_evicted_count")
    missing_fields = [
        field_name
        for field_name in required_metadata_fields
        if field_name not in metadata
    ]
    if missing_fields:
        joined_fields = ", ".join(missing_fields)
        raise ValueError(
            f"consolidation_metadata missing required fields: {joined_fields}"
        )

    write_success = metadata["write_success"]
    if not isinstance(write_success, dict):
        raise ValueError("write_success must be a dict")
    sanitized_write_success = {
        str(key): bool(value)
        for key, value in write_success.items()
        if isinstance(key, str)
    }

    cache_evicted_count = metadata["cache_evicted_count"]
    if isinstance(cache_evicted_count, bool) or not isinstance(
        cache_evicted_count,
        int,
    ):
        raise ValueError("cache_evicted_count must be an int")

    outcome = {
        "consolidation_called": True,
        "write_success": sanitized_write_success,
        "scheduled_event_count": 0,
        "cache_evicted_count": cache_evicted_count,
        "origin_trigger_source": episode["trigger_source"],
        "origin_episode_id": episode["episode_id"],
    }
    return outcome


def classify_route(
    case: models.SelfCognitionCase,
    cognition_output: dict[str, Any],
    action_attempt: dict[str, Any] | None = None,
) -> str:
    """Derive the self-cognition route without forcing social contact.

    Args:
        case: Self-cognition source case.
        cognition_output: Output returned by the shared cognition graph.
        action_attempt: Optional local action attempt used for duplicate state.

    Returns:
        One supported route name.
    """

    explicit_route = _explicit_route(cognition_output)
    if explicit_route:
        route = _route_with_action_attempt(explicit_route, action_attempt)
        return route

    v2_route = _v2_route_for_due_schedule(case, cognition_output)
    if v2_route:
        route = _route_with_action_attempt(v2_route, action_attempt)
        return route

    action_spec_route = _route_from_action_specs(cognition_output)
    if action_spec_route:
        route = _route_with_action_attempt(action_spec_route, action_attempt)
        return route

    content_plan_route = _route_from_content_plan(cognition_output)
    if content_plan_route:
        route = _route_with_action_attempt(content_plan_route, action_attempt)
        return route

    case_name = _string_field(case, "case_name")
    due_state = _optional_string_field(case, "semantic_due_state")
    if case_name == models.CASE_GROUP_NOISE_REJECTED:
        return_value = models.ROUTE_AUDIT_ONLY
    elif due_state == models.DUE_STATE_FUTURE_DUE:
        return_value = models.ROUTE_PROGRESS_MAINTENANCE
    else:
        return_value = models.ROUTE_AUDIT_ONLY
    return return_value


def build_action_attempt(
    case: models.SelfCognitionCase,
    trigger_record: dict[str, Any],
    existing_attempts: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build or suppress a local action attempt for an outward route.

    Args:
        case: Self-cognition source case.
        trigger_record: Trigger record returned by `build_trigger_record`.
        existing_attempts: Local prior attempts supplied by the case fixture.

    Returns:
        Action-attempt record with candidate, held, duplicate, or closed
        status.
    """

    source_ref = _first_source_ref(case)
    source_kind = _string_field(source_ref, "source_kind")
    source_id = _string_field(source_ref, "source_id")
    due_at = _optional_string_field(source_ref, "due_at")
    target_scope = _case_target_scope(case)
    idempotency_key = build_idempotency_key(
        source_kind,
        source_id,
        due_at,
        target_scope,
        models.ACTION_KIND_SEND_MESSAGE,
    )
    status = _candidate_status(case, idempotency_key, existing_attempts)
    attempt_id = _stable_prefixed_id(
        "self_cognition_attempt",
        {
            "idempotency_key": idempotency_key,
            "trigger_id": trigger_record["trigger_id"],
        },
    )
    action_attempt = {
        "attempt_id": attempt_id,
        "run_id": f"self_cognition_run:{trigger_record['trigger_id']}",
        "trigger_id": trigger_record["trigger_id"],
        "source_kind": source_kind,
        "source_id": source_id,
        "target_scope": target_scope,
        "action_kind": models.ACTION_KIND_SEND_MESSAGE,
        "due_at": due_at,
        "idempotency_key": idempotency_key,
        "status": status,
    }
    return action_attempt


def build_action_candidate(
    case: models.SelfCognitionCase,
    action_attempt: dict[str, Any],
    text: str,
) -> dict[str, Any] | None:
    """Build a send-message candidate for a non-duplicate action attempt.

    Args:
        case: Self-cognition source case.
        action_attempt: Action attempt returned by `build_action_attempt`.
        text: Candidate message text emitted by the cognition output.

    Returns:
        Handoff-shaped local candidate, or `None` when no candidate is allowed.
    """

    if action_attempt["status"] != models.ACTION_ATTEMPT_STATUS_CANDIDATE:
        return_value = None
        return return_value

    clean_text = text.strip()
    if not clean_text:
        return_value = None
        return return_value

    target_scope = _case_target_scope(case)
    action_candidate = {
        "attempt_id": action_attempt["attempt_id"],
        "target_platform": target_scope["platform"],
        "target_channel": target_scope["platform_channel_id"],
        "target_channel_type": target_scope["channel_type"],
        "text": clean_text,
        "execute_at": None,
        "dispatch_shape": models.ACTION_KIND_SEND_MESSAGE,
        "production_handoff": PRODUCTION_HANDOFF_ENABLED,
    }
    delivery_mentions = _delivery_mentions_for_action(case, text=clean_text)
    if delivery_mentions:
        action_candidate["delivery_mentions"] = delivery_mentions
    return action_candidate


def _candidate_status(
    case: models.SelfCognitionCase,
    idempotency_key: str,
    existing_attempts: list[dict[str, Any]],
) -> str:
    """Classify action-attempt status from due state and prior attempts."""

    for attempt in existing_attempts:
        attempt_key = attempt.get("idempotency_key")
        attempt_status = attempt.get("status")
        group_retry_status = (
            _string_field(case, "case_name") == models.CASE_GROUP_CHAT_REVIEW
            and attempt_status == models.ACTION_ATTEMPT_STATUS_DELIVERY_FAILED
        )
        if (
            attempt_key == idempotency_key
            and (
                attempt_status in models.ACTION_ATTEMPT_SUPPRESSING_STATUSES
                or group_retry_status
            )
        ):
            return_value = models.ACTION_ATTEMPT_STATUS_DUPLICATE
            return return_value

    due_state = _optional_string_field(case, "semantic_due_state")
    if due_state in models.CONTACT_DUE_STATES:
        return_value = models.ACTION_ATTEMPT_STATUS_CANDIDATE
    elif _string_field(case, "case_name") == models.CASE_GROUP_CHAT_REVIEW:
        return_value = models.ACTION_ATTEMPT_STATUS_CANDIDATE
    elif _string_field(case, "case_name") == models.CASE_TOPIC_RAG_FOLLOWUP:
        return_value = models.ACTION_ATTEMPT_STATUS_CANDIDATE
    else:
        return_value = models.ACTION_ATTEMPT_STATUS_HELD
    return return_value


def _route_with_action_attempt(
    route: str,
    action_attempt: dict[str, Any] | None,
) -> str:
    """Preserve outward route semantics while honoring duplicate state."""

    if route != models.ROUTE_ACTION_CANDIDATE:
        return_value = route
        return return_value

    attempt_status = _action_attempt_status(action_attempt)
    if attempt_status == models.ACTION_ATTEMPT_STATUS_DUPLICATE:
        return_value = models.ROUTE_ACTION_CANDIDATE
        return return_value

    return_value = route
    return return_value


def _action_attempt_status(action_attempt: dict[str, Any] | None) -> str:
    """Read an optional action-attempt status safely."""

    if not isinstance(action_attempt, dict):
        return_value = ""
        return return_value
    status = action_attempt.get("status")
    if not isinstance(status, str):
        return_value = ""
        return return_value
    return_value = status
    return return_value


def _explicit_route(cognition_output: dict[str, Any]) -> str:
    """Read an explicit self-cognition route from cognition output."""

    route = cognition_output.get("self_cognition_route")
    if isinstance(route, str) and route in models.SUPPORTED_ROUTES:
        return_value = route
        return return_value

    section = cognition_output.get("self_cognition")
    if isinstance(section, dict):
        selected_route = section.get("selected_route")
        if (
            isinstance(selected_route, str)
            and selected_route in models.SUPPORTED_ROUTES
        ):
            return_value = selected_route
            return return_value

    return_value = ""
    return return_value


def _v2_route_for_due_schedule(
    case: models.SelfCognitionCase,
    cognition_output: dict[str, Any],
) -> str:
    """Project native V2 scheduled speech into the delivery owner."""

    core_output = cognition_output.get("cognition_core_output")
    if not isinstance(core_output, dict):
        return ""
    intention = core_output.get("intention")
    if not isinstance(intention, dict):
        return ""
    native_route = intention.get("route")
    source_kind = _first_source_ref(case).get("source_kind")
    due_state = _optional_string_field(case, "semantic_due_state")
    if (
        source_kind == "scheduled_tick"
        and due_state in models.CONTACT_DUE_STATES
        and native_route in {"speech", "action"}
    ):
        return_value = models.ROUTE_ACTION_CANDIDATE
        return return_value
    return_value = ""
    return return_value


def _route_from_action_specs(cognition_output: dict[str, Any]) -> str:
    """Map selected action specs to the self-cognition visible route."""

    action_specs = cognition_output.get("action_specs")
    if not isinstance(action_specs, list) or not action_specs:
        return_value = ""
        return return_value
    for action_spec in action_specs:
        if not isinstance(action_spec, dict):
            continue
        kind = action_spec.get("kind")
        if kind == SPEAK_CAPABILITY:
            return_value = models.ROUTE_ACTION_CANDIDATE
            return return_value
    return_value = models.ROUTE_AUDIT_ONLY
    return return_value


def _route_from_content_plan(cognition_output: dict[str, Any]) -> str:
    """Map explicit content-plan markers to self-cognition routes."""

    plan_values = _content_plan_values(cognition_output)
    for plan_value in plan_values:
        if models.PROGRESS_MAINTENANCE_MARKER in plan_value:
            return_value = models.ROUTE_PROGRESS_MAINTENANCE
            return return_value
        if models.AUDIT_ONLY_MARKER in plan_value:
            return_value = models.ROUTE_AUDIT_ONLY
            return return_value
        if models.SILENT_NO_WRITE_MARKER in plan_value:
            return_value = models.ROUTE_SILENT_NO_WRITE
            return return_value
    return_value = ""
    return return_value


def _content_plan_values(cognition_output: dict[str, Any]) -> list[str]:
    """Read non-empty content-plan values from the native V2 surface."""

    surface_output = cognition_output.get("text_surface_output_v2")
    if not isinstance(surface_output, dict):
        return_value: list[str] = []
        return return_value

    content_plan = surface_output.get("content_plan")
    if not isinstance(content_plan, str) or not content_plan.strip():
        return_value = []
        return return_value
    return [content_plan]


def _case_target_scope(case: models.SelfCognitionCase) -> dict[str, Any]:
    """Normalize a case target scope for tracking artifacts."""

    value = case.get("target_scope")
    if not isinstance(value, dict):
        value = {}
    scope = _normalized_target_scope(value)
    return scope


def _delivery_mentions_for_action(
    case: models.SelfCognitionCase,
    *,
    text: str,
) -> list[dict[str, Any]]:
    """Build optional delivery mention metadata for a self-cognition action."""

    users: list[dict[str, Any]] = []
    value = case.get("target_scope")
    if isinstance(value, dict):
        user_id = value.get("user_id")
        if isinstance(user_id, str) and user_id:
            user = {
                "global_user_id": user_id,
                "platform_user_id": _optional_string_field(
                    value,
                    "platform_user_id",
                ),
                "display_name": _string_field(value, "display_name"),
            }
            users.append(user)

    delivery_users = case.get("delivery_mention_users")
    if isinstance(delivery_users, list):
        for delivery_user in delivery_users:
            if not isinstance(delivery_user, dict):
                continue
            user = {
                "global_user_id": _optional_string_field(
                    delivery_user,
                    "global_user_id",
                ),
                "platform_user_id": _optional_string_field(
                    delivery_user,
                    "platform_user_id",
                ),
                "display_name": _string_field(delivery_user, "display_name"),
            }
            users.append(user)

    if not users:
        return_value: list[dict[str, Any]] = []
        return return_value

    return_value = build_inline_delivery_mentions(text=text, users=users)
    return return_value


def _normalized_target_scope(target_scope: dict[str, Any]) -> dict[str, Any]:
    """Build the stable target-scope identity used in artifacts and keys."""

    platform = target_scope.get("platform")
    platform_channel_id = target_scope.get("platform_channel_id")
    channel_type = target_scope.get("channel_type")
    user_id = target_scope.get("user_id")
    normalized_scope = {
        "platform": platform if isinstance(platform, str) else "",
        "platform_channel_id": (
            platform_channel_id if isinstance(platform_channel_id, str) else ""
        ),
        "channel_type": channel_type if isinstance(channel_type, str) else "",
        "user_id": user_id if isinstance(user_id, str) else None,
    }
    return normalized_scope


def _case_source_refs(
    case: models.SelfCognitionCase,
) -> list[dict[str, Any]]:
    """Copy case source references for tracking artifacts."""

    value = case.get("source_refs")
    if not isinstance(value, list):
        return_value: list[dict[str, Any]] = []
        return return_value
    refs = [
        dict(item)
        for item in value
        if isinstance(item, dict)
    ]
    return refs


def _first_source_ref(case: models.SelfCognitionCase) -> dict[str, Any]:
    """Return the primary source reference used for action idempotency."""

    source_refs = _case_source_refs(case)
    if source_refs:
        return_value = source_refs[0]
    else:
        return_value = {
            "source_kind": "",
            "source_id": "",
            "due_at": None,
        }
    return return_value


def _stable_prefixed_id(prefix: str, value: dict[str, Any]) -> str:
    """Build a stable local artifact identifier from structured data."""

    rendered = json.dumps(value, sort_keys=True, ensure_ascii=False)
    digest = hashlib.sha256(rendered.encode("utf-8")).hexdigest()
    stable_id = f"{prefix}:{digest[:models.STABLE_ID_DIGEST_PREFIX_LENGTH]}"
    return stable_id


def _string_field(case: dict[str, Any], field_name: str) -> str:
    """Read an optional external string field safely."""

    value = case.get(field_name)
    if not isinstance(value, str):
        return_value = ""
        return return_value
    return_value = value
    return return_value


def _optional_string_field(
    case: dict[str, Any],
    field_name: str,
) -> str | None:
    """Read an optional external string-or-null field safely."""

    value = case.get(field_name)
    if value is None:
        return_value = None
        return return_value
    if not isinstance(value, str):
        return_value = None
        return return_value
    return_value = value
    return return_value
