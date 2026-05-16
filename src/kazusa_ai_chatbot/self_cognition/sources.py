"""Visible and actionable source collectors for the self-cognition worker."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from datetime import datetime
from typing import Any

from kazusa_ai_chatbot.config import (
    SELF_COGNITION_TRIGGER_ACTIVE_COMMITMENT_ENABLED,
)
from kazusa_ai_chatbot.db import (
    get_conversation_history,
    get_user_profile,
    list_due_future_cognition_events,
    query_active_commitment_memory_units,
)
from kazusa_ai_chatbot.dispatcher.task import parse_iso_datetime
from kazusa_ai_chatbot.self_cognition import models
from kazusa_ai_chatbot.utils import text_or_empty


async def collect_self_cognition_cases(
    *,
    now: datetime,
    character_profile: dict[str, Any],
    max_cases: int,
) -> list[models.SelfCognitionCase]:
    """Collect bounded worker cases from enabled production source types.

    Args:
        now: Current worker tick time.
        character_profile: Current character state snapshot.
        max_cases: Maximum cases this worker tick may process.

    Returns:
        Self-cognition cases ready for the runner.
    """

    cases: list[models.SelfCognitionCase] = []
    scheduled_cases = await collect_scheduled_future_cognition_cases(
        now=now,
        character_profile=character_profile,
        max_cases=max_cases,
    )
    cases.extend(scheduled_cases)

    remaining_cases = max_cases - len(cases)
    if SELF_COGNITION_TRIGGER_ACTIVE_COMMITMENT_ENABLED:
        if remaining_cases > 0:
            commitment_cases = await collect_active_commitment_cases(
                now=now,
                character_profile=character_profile,
                max_cases=remaining_cases,
            )
            cases.extend(commitment_cases)

    selected_cases = cases[:max_cases]
    return selected_cases


async def collect_scheduled_future_cognition_cases(
    now: datetime,
    character_profile: dict[str, Any],
    max_cases: int,
    list_due_events_func: Callable[..., Any] | None = None,
) -> list[models.SelfCognitionCase]:
    """Build worker cases from due scheduled future-cognition slots.

    Args:
        now: Current worker tick time.
        character_profile: Current character state snapshot.
        max_cases: Maximum due slots to project.
        list_due_events_func: Optional test seam for scheduled-slot reads.

    Returns:
        Prompt-safe self-cognition cases for the standard worker runner.
    """

    if max_cases <= 0:
        return_value: list[models.SelfCognitionCase] = []
        return return_value

    current_now = parse_iso_datetime(now.isoformat())
    due_events_reader = list_due_events_func or list_due_future_cognition_events
    raw_events = due_events_reader(
        current_timestamp=current_now.isoformat(),
        limit=max_cases,
    )
    if inspect.isawaitable(raw_events):
        raw_events = await raw_events

    if raw_events is None:
        return_value = []
        return return_value
    events = raw_events if isinstance(raw_events, list) else list(raw_events)

    cases: list[models.SelfCognitionCase] = []
    for event in events:
        if len(cases) >= max_cases:
            break
        if not isinstance(event, dict):
            continue
        if not _is_due_future_cognition_event(event, current_now):
            continue
        case = _build_scheduled_future_cognition_case(
            event,
            character_profile=character_profile,
            now=current_now,
        )
        if case is None:
            continue
        cases.append(case)

    return cases


async def collect_active_commitment_cases(
    *,
    now: datetime,
    character_profile: dict[str, Any],
    max_cases: int,
    list_active_commitments_func: Callable[..., Any] | None = None,
    get_conversation_history_func: Callable[..., Any] | None = None,
    get_user_profile_func: Callable[..., Any] | None = None,
) -> list[models.SelfCognitionCase]:
    """Build due-check cases from active commitment memory units.

    Args:
        now: Current worker tick time.
        character_profile: Current character state snapshot.
        max_cases: Maximum active commitments to project.
        list_active_commitments_func: Optional test seam for commitment reads.
        get_conversation_history_func: Optional test seam for visible context.
        get_user_profile_func: Optional test seam for user profile reads.

    Returns:
        Bounded self-cognition cases with recent visible context.
    """

    active_commitment_reader = (
        list_active_commitments_func or query_active_commitment_memory_units
    )
    history_reader = get_conversation_history_func or get_conversation_history
    profile_reader = get_user_profile_func or get_user_profile

    units = await active_commitment_reader(
        current_timestamp=now.isoformat(),
        limit=max_cases,
    )
    cases: list[models.SelfCognitionCase] = []
    for unit in units:
        if len(cases) >= max_cases:
            break
        global_user_id = text_or_empty(unit.get("global_user_id"))
        due_at = text_or_empty(unit.get("due_at"))
        if not global_user_id or not due_at:
            continue
        due_state = _due_state(due_at, now)
        if not due_state:
            continue
        rows = await history_reader(
            global_user_id=global_user_id,
            limit=models.SOURCE_VISIBLE_CONTEXT_LIMIT,
        )
        if not rows:
            continue
        user_profile = await profile_reader(global_user_id)
        case = _build_active_commitment_case(
            unit,
            rows,
            user_profile=user_profile,
            character_profile=character_profile,
            now=now,
            due_state=due_state,
        )
        cases.append(case)

    return cases


def _is_due_future_cognition_event(
    event: dict[str, Any],
    now: datetime,
) -> bool:
    """Return whether a scheduler row is an eligible future-cognition slot."""

    if event.get("status") != "pending":
        return_value = False
        return return_value
    if event.get("tool") != "trigger_future_cognition":
        return_value = False
        return return_value

    execute_at = text_or_empty(event.get("execute_at"))
    if not execute_at:
        return_value = False
        return return_value
    try:
        execute_time = parse_iso_datetime(execute_at)
    except ValueError:
        return_value = False
        return return_value

    return_value = execute_time <= now
    return return_value


def _build_scheduled_future_cognition_case(
    event: dict[str, Any],
    *,
    character_profile: dict[str, Any],
    now: datetime,
) -> models.SelfCognitionCase | None:
    """Project one due future-cognition slot into a normal worker case."""

    args = event.get("args")
    if not isinstance(args, dict):
        return_value = None
        return return_value
    if args.get("episode_type") != "self_cognition":
        return_value = None
        return return_value

    continuation_objective = text_or_empty(args.get("continuation_objective"))
    if not continuation_objective:
        return_value = None
        return return_value
    event_id = text_or_empty(event.get("event_id"))
    if not event_id:
        return_value = None
        return return_value

    execute_at = text_or_empty(event.get("execute_at"))
    source_action_attempt_id = text_or_empty(
        args.get("source_action_attempt_id")
    )
    source_platform = text_or_empty(event.get("source_platform"))
    source_channel_id = text_or_empty(event.get("source_channel_id"))
    source_channel_type = text_or_empty(event.get("source_channel_type"))
    source_user_id = text_or_empty(event.get("source_user_id"))
    source_platform_bot_id = text_or_empty(event.get("source_platform_bot_id"))
    source_refs = _scheduled_future_cognition_source_refs(
        args,
        continuation_objective=continuation_objective,
        execute_at=execute_at,
    )
    safe_continuation = _safe_continuation(args.get("continuation"))

    case: models.SelfCognitionCase = {
        "case_name": models.CASE_SCHEDULED_FUTURE_COGNITION,
        "case_id": event_id,
        "idle_timestamp": now.isoformat(),
        "last_evidence_timestamp": _scheduled_event_evidence_timestamp(event),
        "trigger_kind": models.TRIGGER_SCHEDULED_FUTURE_COGNITION,
        "semantic_due_state": models.DUE_STATE_DUE_NOW,
        "actionability": "scheduled_private_followup_ready_no_direct_contact",
        "target_scope": {
            "platform": source_platform or "orchestrator",
            "platform_channel_id": source_channel_id,
            "channel_type": source_channel_type or "internal",
            "user_id": source_user_id or "self_cognition",
            "display_name": source_user_id or "self-cognition",
        },
        "source_refs": source_refs,
        "visible_context": [],
        "conversation_progress": {
            "source": "scheduled_future_cognition",
            "continuation_objective": continuation_objective,
            "continuation": safe_continuation,
        },
        "character_profile": dict(character_profile),
        "user_profile": {
            "affinity": models.DEFAULT_DRY_RUN_AFFINITY,
            "display_name": "self-cognition",
            "last_relationship_insight": "",
        },
        "current_mood": text_or_empty(character_profile.get("mood")),
        "global_vibe": text_or_empty(character_profile.get("global_vibe")),
        "rag_query": continuation_objective,
        "platform_bot_id": source_platform_bot_id,
        "source_scheduled_event_id": event_id,
        "source_action_attempt_id": source_action_attempt_id,
    }
    return case


def _scheduled_future_cognition_source_refs(
    args: dict[str, Any],
    *,
    continuation_objective: str,
    execute_at: str,
) -> list[models.SelfCognitionSourceRef]:
    """Build prompt-safe source references for a scheduled cognition slot."""

    source_refs: list[models.SelfCognitionSourceRef] = [
        {
            "source_kind": "scheduled_event",
            "source_id": "scheduled_future_cognition_slot",
            "due_at": execute_at or None,
            "summary": continuation_objective,
        }
    ]

    raw_source_refs = args.get("source_refs")
    if not isinstance(raw_source_refs, list):
        return source_refs

    for index, raw_ref in enumerate(raw_source_refs):
        if not isinstance(raw_ref, dict):
            continue
        source_ref = {
            "source_kind": _safe_action_source_kind(raw_ref),
            "source_id": f"action_source_ref:{index}",
            "due_at": None,
            "summary": _safe_action_source_summary(raw_ref),
        }
        source_refs.append(source_ref)
    return source_refs


def _safe_action_source_kind(raw_ref: dict[str, Any]) -> str:
    """Project an action source kind without leaking storage identifiers."""

    ref_kind = text_or_empty(raw_ref.get("ref_kind"))
    if ref_kind:
        return_value = f"action_{ref_kind}"
    else:
        return_value = "action_source_ref"
    return return_value


def _safe_action_source_summary(raw_ref: dict[str, Any]) -> str:
    """Summarize action source metadata without raw ids or excerpts."""

    parts: list[str] = []
    for field_name in ("owner", "relationship", "ref_kind"):
        value = text_or_empty(raw_ref.get(field_name))
        if value:
            parts.append(f"{field_name}={value}")
    if not parts:
        return_value = "prior action source reference"
        return return_value
    return_value = "; ".join(parts)
    return return_value


def _safe_continuation(value: object) -> dict[str, Any]:
    """Project bounded continuation metadata for model-visible context."""

    if not isinstance(value, dict):
        return_value: dict[str, Any] = {}
        return return_value

    safe_value: dict[str, Any] = {}
    mode = value.get("mode")
    if isinstance(mode, str):
        safe_value["mode"] = mode
    return safe_value


def _scheduled_event_evidence_timestamp(event: dict[str, Any]) -> str:
    """Choose the visible timestamp for a scheduled slot case."""

    execute_at = text_or_empty(event.get("execute_at"))
    if execute_at:
        return execute_at
    created_at = text_or_empty(event.get("created_at"))
    return created_at


def _build_active_commitment_case(
    unit: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    user_profile: dict[str, Any],
    character_profile: dict[str, Any],
    now: datetime,
    due_state: str,
) -> models.SelfCognitionCase:
    """Project one active commitment and recent transcript into a case."""

    latest_row = rows[-1]
    unit_id = text_or_empty(unit.get("unit_id"))
    due_at = text_or_empty(unit.get("due_at"))
    fact = text_or_empty(unit.get("fact"))
    case_name = models.CASE_COMMITMENT_PAST_DUE
    actionability = "past_due_commitment_contact_socially_available"
    if due_state == models.DUE_STATE_FUTURE_DUE:
        case_name = models.CASE_COMMITMENT_BEFORE_DUE
        actionability = "future_commitment_progress_visible_no_contact_required"

    case: models.SelfCognitionCase = {
        "case_name": case_name,
        "case_id": f"active_commitment:{unit_id}:{due_at}",
        "idle_timestamp": now.isoformat(),
        "last_evidence_timestamp": _last_evidence_timestamp(unit, rows),
        "trigger_kind": models.TRIGGER_ACTIVE_COMMITMENT_DUE_CHECK,
        "semantic_due_state": due_state,
        "actionability": actionability,
        "target_scope": {
            "platform": text_or_empty(latest_row.get("platform")),
            "platform_channel_id": text_or_empty(
                latest_row.get("platform_channel_id")
            ),
            "channel_type": text_or_empty(latest_row.get("channel_type")),
            "user_id": text_or_empty(unit.get("global_user_id")),
            "platform_user_id": text_or_empty(
                latest_row.get("platform_user_id")
            ),
            "display_name": text_or_empty(latest_row.get("display_name")),
        },
        "source_refs": [
            {
                "source_kind": "user_memory_unit",
                "source_id": unit_id,
                "due_at": due_at,
                "summary": fact,
            }
        ],
        "visible_context": _visible_context(rows),
        "character_profile": dict(character_profile),
        "user_profile": dict(user_profile),
        "current_mood": text_or_empty(character_profile.get("mood")),
        "global_vibe": text_or_empty(character_profile.get("global_vibe")),
    }
    return case


def _due_state(due_at: str, now: datetime) -> str:
    """Classify a due timestamp relative to the worker tick."""

    try:
        due_time = parse_iso_datetime(due_at)
    except ValueError:
        return_value = ""
        return return_value
    if due_time > now:
        return_value = models.DUE_STATE_FUTURE_DUE
    elif due_time == now:
        return_value = models.DUE_STATE_DUE_NOW
    else:
        return_value = models.DUE_STATE_PAST_DUE
    return return_value


def _last_evidence_timestamp(
    unit: dict[str, Any],
    rows: list[dict[str, Any]],
) -> str:
    """Choose the most recent visible or memory timestamp for a case."""

    latest_row = rows[-1]
    row_timestamp = text_or_empty(latest_row.get("timestamp"))
    if row_timestamp:
        return_value = row_timestamp
        return return_value
    for field_name in ("updated_at", "last_seen_at", "first_seen_at"):
        timestamp = text_or_empty(unit.get(field_name))
        if timestamp:
            return timestamp
    return_value = ""
    return return_value


def _visible_context(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Project recent conversation rows into runner-visible case context."""

    visible_rows: list[dict[str, str]] = []
    for row in rows:
        body_text = text_or_empty(row.get("body_text"))
        if not body_text:
            continue
        visible_row = {
            "role": text_or_empty(row.get("role")),
            "body_text": body_text,
            "display_name": text_or_empty(row.get("display_name")),
            "timestamp": text_or_empty(row.get("timestamp")),
        }
        visible_rows.append(visible_row)
    return visible_rows
