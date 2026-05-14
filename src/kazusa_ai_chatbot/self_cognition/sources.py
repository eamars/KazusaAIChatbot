"""Visible and actionable source collectors for the self-cognition worker."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import Any

from kazusa_ai_chatbot.config import (
    SELF_COGNITION_TRIGGER_ACTIVE_COMMITMENT_ENABLED,
)
from kazusa_ai_chatbot.db import (
    get_conversation_history,
    get_user_profile,
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
    if SELF_COGNITION_TRIGGER_ACTIVE_COMMITMENT_ENABLED:
        commitment_cases = await collect_active_commitment_cases(
            now=now,
            character_profile=character_profile,
            max_cases=max_cases,
        )
        cases.extend(commitment_cases)

    selected_cases = cases[:max_cases]
    return selected_cases


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
