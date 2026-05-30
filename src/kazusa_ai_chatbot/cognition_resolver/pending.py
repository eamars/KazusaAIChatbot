"""Pending HIL and approval state backed by the action-attempt ledger."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Awaitable, Callable
from datetime import timedelta
from typing import Any

from kazusa_ai_chatbot.action_spec.attempt_ledger import (
    list_action_attempts,
    upsert_action_attempt,
)
from kazusa_ai_chatbot.cognition_resolver.contracts import (
    RESOLVER_PENDING_RESUME_VERSION,
    ResolverObservationV1,
    ResolverPendingResolutionV1,
    ResolverPendingResumeV1,
    ResolverValidationError,
    project_pending_resume_for_cognition,
    validate_resolver_observation,
    validate_resolver_pending_resolution,
    validate_resolver_pending_resume,
)
from kazusa_ai_chatbot.cognition_resolver.state import (
    project_resolver_context,
    validate_resolver_state,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_schema import GlobalPersonaState
from kazusa_ai_chatbot.time_boundary import (
    normalize_storage_utc_iso,
    parse_storage_utc_datetime,
)

RESOLVER_PENDING_HIL_ACTION_KIND = "resolver_pending_hil"
RESOLVER_PENDING_APPROVAL_ACTION_KIND = "resolver_pending_approval"
PENDING_RESUME_TTL_HOURS = 24

ListActionAttemptsFunc = Callable[..., Awaitable[list[dict[str, Any]]]]
UpsertActionAttemptFunc = Callable[[dict[str, Any]], Awaitable[None]]


def build_pending_resume_record(
    state: GlobalPersonaState,
    observation: ResolverObservationV1,
    *,
    expires_at_utc: str | None = None,
) -> dict[str, Any]:
    """Build one pending-resume ledger row from a blocked observation."""

    validated_observation = validate_resolver_observation(observation)
    capability_kind = validated_observation["capability_kind"]
    action_kind = _action_kind_for_capability(capability_kind)
    created_at_utc = _storage_timestamp(state)
    if expires_at_utc is None:
        expires_at_utc = _expiry_timestamp(created_at_utc)
    else:
        expires_at_utc = normalize_storage_utc_iso(expires_at_utc)
    resume_id = _resume_id(state, validated_observation)
    pending_resume = _pending_resume(
        state,
        validated_observation,
        resume_id=resume_id,
        expires_at_utc=expires_at_utc,
    )
    record = {
        "attempt_id": resume_id,
        "run_id": "",
        "trigger_id": _state_text(state, "platform_message_id"),
        "source_kind": "cognitive_episode",
        "source_id": _episode_id(state),
        "target_scope": _target_scope(state),
        "action_kind": action_kind,
        "due_at": expires_at_utc,
        "idempotency_key": resume_id,
        "status": pending_resume["status"],
        "dispatch_status": "",
        "scheduled_event_ids": [],
        "recorded_at": created_at_utc,
        "action_spec_schema_version": RESOLVER_PENDING_RESUME_VERSION,
        "cognition_mode": "deliberative",
        "validation_status": "accepted",
        "handler_owner": "cognition_resolver",
        "continuation_status": pending_resume["status"],
        "execution_result": {
            "status": pending_resume["status"],
            "pending_resume": pending_resume,
            "resolver_pending_resume": pending_resume,
        },
        "resolver_pending_resume": pending_resume,
        "errors": [],
    }
    return_value = record
    return return_value


async def upsert_pending_resume(
    state: GlobalPersonaState,
    observation: ResolverObservationV1,
    *,
    upsert_action_attempt_func: UpsertActionAttemptFunc = upsert_action_attempt,
) -> ResolverPendingResumeV1:
    """Persist and return a pending HIL or approval resume row."""

    record = build_pending_resume_record(state, observation)
    await upsert_action_attempt_func(record)
    pending_resume = record["execution_result"]["pending_resume"]
    return_value = validate_resolver_pending_resume(pending_resume)
    return return_value


async def load_matching_pending_resume(
    state: GlobalPersonaState,
    *,
    list_action_attempts_func: ListActionAttemptsFunc = list_action_attempts,
    upsert_action_attempt_func: UpsertActionAttemptFunc = upsert_action_attempt,
) -> ResolverPendingResumeV1 | None:
    """Load one unexpired pending row matching the current conversation scope."""

    current_timestamp_utc = _storage_timestamp(state)
    rows = await list_action_attempts_func(limit=1000)
    for row in rows:
        if not _is_pending_resolver_row(row):
            continue
        if not _scope_matches_current_turn(row, state):
            continue
        pending_resume = _pending_resume_from_row(row)
        if pending_resume is None:
            continue
        if not _is_open_pending_status(pending_resume):
            continue
        if _is_expired(pending_resume, current_timestamp_utc):
            expired_row = _updated_pending_row(
                row,
                pending_resume,
                status="expired",
                resolution=None,
                recorded_at=current_timestamp_utc,
            )
            await upsert_action_attempt_func(expired_row)
            continue
        return_value = pending_resume
        return return_value
    return_value = None
    return return_value


async def load_matching_pending_resume_into_state(
    state: GlobalPersonaState,
    *,
    list_action_attempts_func: ListActionAttemptsFunc = list_action_attempts,
    upsert_action_attempt_func: UpsertActionAttemptFunc = upsert_action_attempt,
) -> GlobalPersonaState:
    """Attach one prompt-safe pending resume row to resolver state."""

    pending_resume = await load_matching_pending_resume(
        state,
        list_action_attempts_func=list_action_attempts_func,
        upsert_action_attempt_func=upsert_action_attempt_func,
    )
    if pending_resume is None:
        return_value = state
        return return_value

    resolver_state = validate_resolver_state(state["resolver_state"])
    updated_resolver_state = dict(resolver_state)
    updated_resolver_state["pending_resume"] = pending_resume
    updated_state = dict(state)
    updated_state["resolver_state"] = updated_resolver_state
    updated_state["pending_resolver_resume"] = pending_resume
    updated_state["resolver_context"] = project_resolver_context(
        updated_resolver_state,
    )
    return_value = updated_state
    return return_value


async def apply_pending_resolution(
    state: GlobalPersonaState,
    resolution: ResolverPendingResolutionV1,
    *,
    list_action_attempts_func: ListActionAttemptsFunc = list_action_attempts,
    upsert_action_attempt_func: UpsertActionAttemptFunc = upsert_action_attempt,
) -> dict[str, Any] | None:
    """Apply L2d's pending-row decision to the existing ledger row."""

    validated_resolution = validate_resolver_pending_resolution(resolution)
    rows = await list_action_attempts_func(limit=1000)
    recorded_at = _storage_timestamp(state)
    for row in rows:
        pending_resume = _pending_resume_from_row(row)
        if pending_resume is None:
            continue
        if pending_resume["resume_id"] != validated_resolution["resume_id"]:
            continue
        status = _status_for_resolution(
            pending_resume,
            validated_resolution,
        )
        updated_row = _updated_pending_row(
            row,
            pending_resume,
            status=status,
            resolution=validated_resolution,
            recorded_at=recorded_at,
        )
        await upsert_action_attempt_func(updated_row)
        return_value = updated_row
        return return_value
    return_value = None
    return return_value


def _pending_resume(
    state: GlobalPersonaState,
    observation: ResolverObservationV1,
    *,
    resume_id: str,
    expires_at_utc: str,
) -> ResolverPendingResumeV1:
    """Build the typed pending-resume payload stored in a ledger row."""

    capability_kind = observation["capability_kind"]
    pending_status = _pending_status_for_capability(capability_kind)
    question = ""
    approval_summary = ""
    if capability_kind == "human_clarification":
        question = observation["request_objective"]
    elif capability_kind == "approval_preparation":
        approval_summary = observation["request_objective"]
    pending_resume = {
        "schema_version": RESOLVER_PENDING_RESUME_VERSION,
        "resume_id": resume_id,
        "capability_kind": capability_kind,
        "status": pending_status,
        "platform": _state_text(state, "platform"),
        "platform_channel_id": _state_text(state, "platform_channel_id"),
        "global_user_id": _state_text(state, "global_user_id"),
        "source_message_id": _state_text(state, "platform_message_id"),
        "prompt_safe_question": question,
        "prompt_safe_approval_summary": approval_summary,
        "created_at_utc": _storage_timestamp(state),
        "expires_at_utc": expires_at_utc,
    }
    return_value = validate_resolver_pending_resume(pending_resume)
    return return_value


def _pending_resume_from_row(row: dict[str, Any]) -> ResolverPendingResumeV1 | None:
    """Read and validate pending-resume payload from a ledger row."""

    top_level_resume = row.get("resolver_pending_resume")
    if top_level_resume is not None:
        try:
            validated = validate_resolver_pending_resume(top_level_resume)
        except ResolverValidationError:
            validated = None
        if validated is not None:
            return_value = validated
            return return_value

    execution_result = row.get("execution_result")
    if not isinstance(execution_result, dict):
        return_value = None
        return return_value
    pending_resume = execution_result.get("pending_resume")
    if pending_resume is None:
        pending_resume = execution_result.get("resolver_pending_resume")
    if pending_resume is None:
        return_value = None
        return return_value
    try:
        validated = validate_resolver_pending_resume(pending_resume)
    except ResolverValidationError:
        return_value = None
        return return_value
    return_value = validated
    return return_value


def _updated_pending_row(
    row: dict[str, Any],
    pending_resume: ResolverPendingResumeV1,
    *,
    status: str,
    resolution: ResolverPendingResolutionV1 | None,
    recorded_at: str,
) -> dict[str, Any]:
    """Return one updated pending ledger row."""

    updated_pending = dict(pending_resume)
    updated_pending["status"] = status
    execution_result = dict(row.get("execution_result") or {})
    execution_result["status"] = status
    execution_result["pending_resume"] = validate_resolver_pending_resume(
        updated_pending,
    )
    execution_result["resolver_pending_resume"] = execution_result[
        "pending_resume"
    ]
    if resolution is not None:
        execution_result["pending_resolution"] = resolution
        execution_result["resolver_pending_resolution"] = resolution
    updated_row = dict(row)
    updated_row["status"] = status
    updated_row["continuation_status"] = status
    updated_row["execution_result"] = execution_result
    updated_row["resolver_pending_resume"] = execution_result["pending_resume"]
    updated_row["recorded_at"] = recorded_at
    return_value = updated_row
    return return_value


def _action_kind_for_capability(capability_kind: str) -> str:
    """Map a blocked resolver capability to ledger action kind."""

    if capability_kind == "human_clarification":
        return_value = RESOLVER_PENDING_HIL_ACTION_KIND
        return return_value
    if capability_kind == "approval_preparation":
        return_value = RESOLVER_PENDING_APPROVAL_ACTION_KIND
        return return_value
    raise ResolverValidationError("pending resume requires HIL or approval")


def _pending_status_for_capability(capability_kind: str) -> str:
    """Map a blocked capability to pending status."""

    if capability_kind == "human_clarification":
        return_value = "waiting_for_user"
        return return_value
    if capability_kind == "approval_preparation":
        return_value = "waiting_for_approval"
        return return_value
    raise ResolverValidationError("pending resume requires HIL or approval")


def _status_for_resolution(
    pending_resume: ResolverPendingResumeV1,
    resolution: ResolverPendingResolutionV1,
) -> str:
    """Map L2d's pending decision to durable pending-row status."""

    decision = resolution["decision"]
    if decision == "continue_waiting":
        return_value = pending_resume["status"]
        return return_value
    if decision == "superseded":
        return_value = "superseded"
        return return_value
    return_value = "closed"
    return return_value


def _is_pending_resolver_row(row: dict[str, Any]) -> bool:
    """Return whether a ledger row represents a resolver pending item."""

    action_kind = row.get("action_kind")
    return_value = action_kind in {
        RESOLVER_PENDING_HIL_ACTION_KIND,
        RESOLVER_PENDING_APPROVAL_ACTION_KIND,
    }
    return return_value


def _is_open_pending_status(pending_resume: ResolverPendingResumeV1) -> bool:
    """Return whether a pending row should be projected into cognition."""

    return_value = pending_resume["status"] in {
        "waiting_for_user",
        "waiting_for_approval",
    }
    return return_value


def _scope_matches_current_turn(
    row: dict[str, Any],
    state: GlobalPersonaState,
) -> bool:
    """Return whether a pending row belongs to the current platform scope."""

    target_scope = row.get("target_scope")
    if not isinstance(target_scope, dict):
        return_value = False
        return return_value
    return_value = (
        str(target_scope.get("platform", "")) == _state_text(state, "platform")
        and str(target_scope.get("platform_channel_id", ""))
        == _state_text(state, "platform_channel_id")
        and str(target_scope.get("global_user_id", ""))
        == _state_text(state, "global_user_id")
    )
    return return_value


def _target_scope(state: GlobalPersonaState) -> dict[str, str]:
    """Build deterministic pending-row scope for audit and matching."""

    target_scope = {
        "platform": _state_text(state, "platform"),
        "platform_channel_id": _state_text(state, "platform_channel_id"),
        "global_user_id": _state_text(state, "global_user_id"),
        "source_message_id": _state_text(state, "platform_message_id"),
    }
    return target_scope


def _episode_id(state: GlobalPersonaState) -> str:
    """Read the source cognitive episode id for pending-row audit."""

    episode = state["cognitive_episode"]
    if not isinstance(episode, dict):
        raise ResolverValidationError("cognitive_episode: expected mapping")
    episode_id = episode["episode_id"]
    if not isinstance(episode_id, str) or not episode_id.strip():
        raise ResolverValidationError("cognitive_episode.episode_id: expected string")
    return_value = episode_id.strip()
    return return_value


def _resume_id(
    state: GlobalPersonaState,
    observation: ResolverObservationV1,
) -> str:
    """Build a stable pending resume id for one source message and request."""

    payload = {
        "schema_version": RESOLVER_PENDING_RESUME_VERSION,
        "scope": _target_scope(state),
        "capability_kind": observation["capability_kind"],
        "request_objective": observation["request_objective"],
        "request_reason": observation["request_reason"],
    }
    serialized = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    return_value = f"resolver_pending:v1:{digest}"
    return return_value


def _expiry_timestamp(storage_timestamp_utc: str) -> str:
    """Compute the default pending resume expiry timestamp."""

    created_at = parse_storage_utc_datetime(storage_timestamp_utc)
    expires_at = created_at + timedelta(hours=PENDING_RESUME_TTL_HOURS)
    return_value = expires_at.isoformat()
    return return_value


def _is_expired(
    pending_resume: ResolverPendingResumeV1,
    current_timestamp_utc: str,
) -> bool:
    """Return whether a pending row has expired at the current turn."""

    current = parse_storage_utc_datetime(current_timestamp_utc)
    expires_at = parse_storage_utc_datetime(pending_resume["expires_at_utc"])
    return_value = expires_at <= current
    return return_value


def _storage_timestamp(state: GlobalPersonaState) -> str:
    """Read and normalize the current storage timestamp."""

    value = _state_text(state, "storage_timestamp_utc")
    return_value = normalize_storage_utc_iso(value)
    return return_value


def _state_text(state: GlobalPersonaState, field_name: str) -> str:
    """Read a required text field from persona state."""

    value = state[field_name]
    if not isinstance(value, str) or not value.strip():
        raise ResolverValidationError(f"{field_name}: expected non-empty string")
    return_value = value.strip()
    return return_value
