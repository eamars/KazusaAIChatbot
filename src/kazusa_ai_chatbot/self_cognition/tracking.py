"""Tracking artifacts and route ownership for self-cognition dry runs."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from kazusa_ai_chatbot.self_cognition import models


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
    """Build the local trigger artifact explaining why the run exists.

    Args:
        case: External dry-run case.

    Returns:
        Trigger record for local artifact storage.
    """

    target_scope = _case_target_scope(case)
    source_refs = _case_source_refs(case)
    trigger_identity = {
        "case_name": _string_field(case, "case_name"),
        "case_id": _string_field(case, "case_id"),
        "trigger_kind": _string_field(case, "trigger_kind"),
        "target_scope": target_scope,
        "source_refs": source_refs,
        "idle_timestamp": _string_field(case, "idle_timestamp"),
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
    """Build the run artifact recording the selected route.

    Args:
        case: External dry-run case.
        trigger_record: Trigger artifact returned by `build_trigger_record`.
        selected_route: Route selected by cognition and deterministic tracking.
        budget: Local dry-run budget counters.

    Returns:
        Run record for local artifact storage.
    """

    output_mode = "scheduled_action_request"
    if selected_route != models.ROUTE_ACTION_CANDIDATE:
        output_mode = "silent"

    run_record = {
        "run_id": f"self_cognition_run:{trigger_record['trigger_id']}",
        "trigger_id": trigger_record["trigger_id"],
        "idle_timestamp": _string_field(case, "idle_timestamp"),
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
    """Build the route-effect artifact for the selected consumer.

    Args:
        run_record: Run artifact returned by `build_run_record`.
        route: Selected route name.
        consumer: Dry-run consumer that would receive the effect.
        effect_summary: Human-readable dry-run effect summary.
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


def classify_route(
    case: models.SelfCognitionCase,
    cognition_output: dict[str, Any],
    action_attempt: dict[str, Any] | None = None,
) -> str:
    """Derive the self-cognition route without forcing social contact.

    Args:
        case: External dry-run case.
        cognition_output: Output returned by the shared cognition graph.
        action_attempt: Optional local action attempt used for duplicate state.

    Returns:
        One supported route name.
    """

    explicit_route = _explicit_route(cognition_output)
    if explicit_route:
        route = _route_with_action_attempt(explicit_route, action_attempt)
        return route

    anchor_route = _route_from_content_anchors(cognition_output)
    if anchor_route:
        route = _route_with_action_attempt(anchor_route, action_attempt)
        return route

    if _cognition_selects_outward_contact(cognition_output):
        route = _route_with_action_attempt(
            models.ROUTE_ACTION_CANDIDATE,
            action_attempt,
        )
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
        case: External dry-run case.
        trigger_record: Trigger artifact returned by `build_trigger_record`.
        existing_attempts: Local prior attempts supplied by the case fixture.

    Returns:
        Action-attempt artifact with candidate, held, duplicate, or closed
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
        case: External dry-run case.
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
        "production_handoff": False,
    }
    return action_candidate


def extract_action_candidate_text(cognition_output: dict[str, Any]) -> str:
    """Extract candidate message text from model-emitted action anchors.

    Args:
        cognition_output: Output returned by the shared cognition graph.

    Returns:
        Text after the first action-candidate marker, or an empty string.
    """

    for anchor in _content_anchors(cognition_output):
        marker_index = anchor.find(models.ACTION_CANDIDATE_MARKER)
        if marker_index < 0:
            continue
        text_start = marker_index + len(models.ACTION_CANDIDATE_MARKER)
        candidate_text = anchor[text_start:].strip()
        return candidate_text
    return_value = ""
    return return_value


def _candidate_status(
    case: models.SelfCognitionCase,
    idempotency_key: str,
    existing_attempts: list[dict[str, Any]],
) -> str:
    """Classify action-attempt status from due state and prior attempts."""

    for attempt in existing_attempts:
        attempt_key = attempt.get("idempotency_key")
        attempt_status = attempt.get("status")
        if (
            attempt_key == idempotency_key
            and attempt_status in models.ACTION_ATTEMPT_SUPPRESSING_STATUSES
        ):
            return_value = models.ACTION_ATTEMPT_STATUS_DUPLICATE
            return return_value

    due_state = _optional_string_field(case, "semantic_due_state")
    if due_state in models.CONTACT_DUE_STATES:
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


def _route_from_content_anchors(cognition_output: dict[str, Any]) -> str:
    """Map explicit content-anchor markers to self-cognition routes."""

    anchors = _content_anchors(cognition_output)
    for anchor in anchors:
        if models.ACTION_CANDIDATE_MARKER in anchor:
            return_value = models.ROUTE_ACTION_CANDIDATE
            return return_value
        if models.PROGRESS_MAINTENANCE_MARKER in anchor:
            return_value = models.ROUTE_PROGRESS_MAINTENANCE
            return return_value
        if models.AUDIT_ONLY_MARKER in anchor:
            return_value = models.ROUTE_AUDIT_ONLY
            return return_value
        if models.SILENT_NO_WRITE_MARKER in anchor:
            return_value = models.ROUTE_SILENT_NO_WRITE
            return return_value
    return_value = ""
    return return_value


def _cognition_selects_outward_contact(
    cognition_output: dict[str, Any],
) -> bool:
    """Detect when cognition has selected outward contact semantics."""

    intent = cognition_output.get("character_intent")
    if isinstance(intent, str) and intent in models.OUTWARD_CONTACT_INTENTS:
        return_value = True
        return return_value

    stance = cognition_output.get("logical_stance")
    if isinstance(stance, str) and stance in models.OUTWARD_CONTACT_STANCES:
        anchors = _content_anchors(cognition_output)
        has_answer_anchor = any("[ANSWER]" in anchor for anchor in anchors)
        if has_answer_anchor:
            return_value = True
            return return_value

    return_value = False
    return return_value


def _content_anchors(cognition_output: dict[str, Any]) -> list[str]:
    """Read non-empty linguistic content anchors from cognition output."""

    action_directives = cognition_output.get("action_directives")
    if not isinstance(action_directives, dict):
        return_value: list[str] = []
        return return_value

    linguistic_directives = action_directives.get("linguistic_directives")
    if not isinstance(linguistic_directives, dict):
        return_value = []
        return return_value

    content_anchors = linguistic_directives.get("content_anchors")
    if not isinstance(content_anchors, list):
        return_value = []
        return return_value

    anchors = [
        item
        for item in content_anchors
        if isinstance(item, str) and item.strip()
    ]
    return anchors


def _case_target_scope(case: models.SelfCognitionCase) -> dict[str, Any]:
    """Normalize a case target scope for tracking artifacts."""

    value = case.get("target_scope")
    if not isinstance(value, dict):
        value = {}
    scope = _normalized_target_scope(value)
    return scope


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
