"""Sanitizers for prompt-safe event logging documents."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence

from kazusa_ai_chatbot.event_logging.models import (
    CognitionChainEventFields,
    CognitionV2EventFields,
    EventScopeInput,
)
from kazusa_ai_chatbot.event_logging.schemas import EventScopeRecord

_CHANNEL_REF_SALT = "kazusa-event-log-scope-v1"
_CONTROL_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]+")
_MAX_SHORT_TEXT_CHARS = 300
_MAX_LIST_ITEMS = 25

_DENIED_FIELD_NAMES = frozenset(
    {
        "human_" + "prompt",
        "system_" + "prompt",
        "raw_" + "output",
        "base64_" + "data",
        "embed" + "ding",
        "api_" + "key",
        "shared_" + "secret",
        "message_" + "envelope",
        "replacement_state",
        "mutable_state",
        "cognition_state",
        "raw_state",
        "owner_key",
        "source_id",
        "entity_id",
        "target_refs",
        "evidence_handles",
        "prompt_text",
        "primary_bid",
        "supporting_bids",
        "competing_bids",
        "private_bids",
        "raw_intensity",
        "raw_magnitude",
        "activation_score",
        "priority_score",
    }
)


def sanitize_short_text(value: object, *, limit: int = _MAX_SHORT_TEXT_CHARS) -> str:
    """Return a compact single-field string safe for event-log storage.

    Args:
        value: Input value from a runtime caller or exception.
        limit: Maximum returned character count.

    Returns:
        Sanitized string with control bytes removed and length capped.
    """

    text = str(value or "")
    normalized_text = _CONTROL_PATTERN.sub(" ", text).strip()
    if len(normalized_text) > limit:
        clipped_text = normalized_text[:limit].rstrip()
        normalized_text = f"{clipped_text}..."
    return normalized_text


def sanitize_string_list(values: Sequence[object]) -> list[str]:
    """Return capped sanitized strings from a sequence-like caller value."""

    sanitized_values = [
        sanitize_short_text(value)
        for value in list(values)[:_MAX_LIST_ITEMS]
    ]
    return sanitized_values


def build_scope_record(scope: EventScopeInput | None) -> EventScopeRecord:
    """Project caller scope into a persisted scope without raw channel IDs.

    Args:
        scope: Optional caller-provided platform scope.

    Returns:
        Persisted event scope with a stable private channel reference.
    """

    input_scope = scope or {}
    platform = sanitize_short_text(input_scope.get("platform", ""), limit=80)
    channel_value = sanitize_short_text(
        input_scope.get("platform_channel_id", ""),
        limit=160,
    )
    channel_type = sanitize_short_text(input_scope.get("channel_type", ""), limit=40)
    channel_ref = ""
    if channel_value:
        digest_input = f"{_CHANNEL_REF_SALT}:{platform}:{channel_value}"
        digest = hashlib.sha256(digest_input.encode("utf-8")).hexdigest()
        channel_ref = f"ch_{digest[:32]}"
    scope_record = EventScopeRecord(
        platform=platform,
        platform_channel_ref=channel_ref,
        channel_type=channel_type,
    )
    return scope_record


def sanitize_cognition_v2_event_fields(
    value: Mapping[str, object],
) -> CognitionV2EventFields:
    """Project only bounded V2 diagnostics and redact every other field."""

    state_scope = sanitize_short_text(value.get("state_scope", ""), limit=20)
    if state_scope not in {"", "user", "character"}:
        state_scope = ""
    commit_status = sanitize_short_text(
        value.get("state_commit_status", "not_started"),
        limit=24,
    )
    if commit_status not in {"not_started", "committed", "failed", "skipped"}:
        commit_status = "failed"
    stage_status = sanitize_short_text(
        value.get("stage_status", "failed"),
        limit=20,
    )
    if stage_status not in {"started", "completed", "failed", "skipped"}:
        stage_status = "failed"
    fields = CognitionV2EventFields(
        cognition_component=sanitize_short_text(
            value.get("cognition_component", ""),
            limit=120,
        ),
        selected_branch_id=sanitize_short_text(
            value.get("selected_branch_id", ""),
            limit=80,
        ),
        state_scope=state_scope,
        state_commit_status=commit_status,
        stage_status=stage_status,
    )
    return fields


def sanitize_cognition_chain_event_fields(
    value: Mapping[str, object],
) -> CognitionChainEventFields:
    """Project only closed Cognition V3 chain metrics and reject raw content."""

    allowed_strings = {
        "run_id",
        "cognition_invocation_id",
        "terminal_disposition",
        "chain_model_name",
        "sidecar_model_name",
        "session_disposition",
    }
    allowed_ints = {
        "step_count",
        "repair_count",
        "cold_start_count",
        "prompt_chars_total",
        "new_suffix_chars_total",
        "max_estimated_prompt_tokens",
        "max_reserved_completion_tokens",
        "max_estimated_total_context_tokens",
        "active_total_ceiling_tokens",
        "duration_ms",
        "deadline_ms",
        "l1_stream_count",
        "json_repair_call_count",
        "action_auth_attempt_count",
        "resolver_auth_attempt_count",
        "sidecar_queue_wait_ms_total",
        "sidecar_max_in_flight",
        "sidecar_cancellation_count",
    }
    allowed_floats = {
        "prefix_share_ratio",
        "deadline_consumption_ratio",
    }
    allowed_bools = {
        "extension_available",
        "extension_used",
        "reanchor_used",
        "l1_preempted_by_repair",
    }
    allowed_fields = (
        allowed_strings
        | allowed_ints
        | allowed_floats
        | allowed_bools
        | {"warning_codes"}
    )
    if set(value) != allowed_fields:
        raise ValueError("cognition chain event fields must be exact")

    fields: dict[str, object] = {}
    for field_name in allowed_strings:
        fields[field_name] = sanitize_short_text(
            value.get(field_name, ""),
            limit=160,
        )
    for field_name in allowed_ints:
        raw_value = value.get(field_name, 0)
        if (
            not isinstance(raw_value, int)
            or isinstance(raw_value, bool)
            or raw_value < 0
        ):
            raise ValueError(
                f"cognition chain {field_name} must be non-negative integer"
            )
        fields[field_name] = raw_value
    for field_name in allowed_floats:
        raw_value = value.get(field_name, 0.0)
        if (
            isinstance(raw_value, bool)
            or not isinstance(raw_value, (int, float))
            or raw_value < 0
        ):
            raise ValueError(
                f"cognition chain {field_name} must be non-negative number"
            )
        fields[field_name] = float(raw_value)
    for field_name in allowed_bools:
        raw_value = value.get(field_name, False)
        if not isinstance(raw_value, bool):
            raise TypeError(f"cognition chain {field_name} must be boolean")
        fields[field_name] = raw_value
    fields["warning_codes"] = sanitize_string_list(
        value.get("warning_codes", [])
    )
    result = CognitionChainEventFields(
        run_id=str(fields["run_id"]),
        cognition_invocation_id=str(fields["cognition_invocation_id"]),
        terminal_disposition=str(fields["terminal_disposition"]),
        chain_model_name=str(fields["chain_model_name"]),
        sidecar_model_name=str(fields["sidecar_model_name"]),
        step_count=int(fields["step_count"]),
        repair_count=int(fields["repair_count"]),
        cold_start_count=int(fields["cold_start_count"]),
        prompt_chars_total=int(fields["prompt_chars_total"]),
        new_suffix_chars_total=int(fields["new_suffix_chars_total"]),
        prefix_share_ratio=float(fields["prefix_share_ratio"]),
        max_estimated_prompt_tokens=int(
            fields["max_estimated_prompt_tokens"]
        ),
        max_reserved_completion_tokens=int(
            fields["max_reserved_completion_tokens"]
        ),
        max_estimated_total_context_tokens=int(
            fields["max_estimated_total_context_tokens"]
        ),
        active_total_ceiling_tokens=int(
            fields["active_total_ceiling_tokens"]
        ),
        extension_available=bool(fields["extension_available"]),
        extension_used=bool(fields["extension_used"]),
        reanchor_used=bool(fields["reanchor_used"]),
        session_disposition=str(fields["session_disposition"]),
        duration_ms=int(fields["duration_ms"]),
        deadline_ms=int(fields["deadline_ms"]),
        deadline_consumption_ratio=float(
            fields["deadline_consumption_ratio"]
        ),
        l1_stream_count=int(fields["l1_stream_count"]),
        json_repair_call_count=int(fields["json_repair_call_count"]),
        action_auth_attempt_count=int(fields["action_auth_attempt_count"]),
        resolver_auth_attempt_count=int(
            fields["resolver_auth_attempt_count"]
        ),
        sidecar_queue_wait_ms_total=int(
            fields["sidecar_queue_wait_ms_total"]
        ),
        sidecar_max_in_flight=int(fields["sidecar_max_in_flight"]),
        l1_preempted_by_repair=bool(fields["l1_preempted_by_repair"]),
        sidecar_cancellation_count=int(
            fields["sidecar_cancellation_count"]
        ),
        warning_codes=list(fields["warning_codes"]),
    )
    return result


def unsafe_field_paths(value: object, *, prefix: str = "") -> list[str]:
    """Return denied field paths found in an event document candidate.

    Args:
        value: Nested mapping/list/scalar value to inspect.
        prefix: Internal recursion prefix.

    Returns:
        List of denied key paths. Empty means no denied keys were found.
    """

    if isinstance(value, Mapping):
        paths: list[str] = []
        for raw_key, raw_child in value.items():
            key = str(raw_key)
            child_prefix = key if not prefix else f"{prefix}.{key}"
            if key in _DENIED_FIELD_NAMES:
                paths.append(child_prefix)
                continue
            paths.extend(unsafe_field_paths(raw_child, prefix=child_prefix))
        return paths

    if isinstance(value, list):
        paths = []
        for index, child in enumerate(value[:_MAX_LIST_ITEMS]):
            child_prefix = f"{prefix}[{index}]"
            paths.extend(unsafe_field_paths(child, prefix=child_prefix))
        return paths

    return_value: list[str] = []
    return return_value


def sanitized_failure_reason(exc: BaseException) -> str:
    """Return sanitized exception text for a failed telemetry write."""

    reason = sanitize_short_text(f"{type(exc).__name__}: {exc}")
    return reason


def sanitized_rejection_reason(paths: Sequence[str]) -> str:
    """Return a compact rejection reason for denied field paths."""

    preview_paths = ", ".join(paths[:5])
    reason = sanitize_short_text(f"unsafe fields: {preview_paths}")
    return reason
