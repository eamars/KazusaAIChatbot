"""Deterministic scope selection and loading for residue context."""

from __future__ import annotations

import logging

from kazusa_ai_chatbot import db, event_logging
from kazusa_ai_chatbot.config import (
    INTERNAL_MONOLOGUE_RESIDUE_CONTEXT_CHAR_LIMIT,
    INTERNAL_MONOLOGUE_RESIDUE_WINDOW_SIZE,
)
from kazusa_ai_chatbot.db import DatabaseOperationError
from kazusa_ai_chatbot.internal_monologue_residue.models import (
    InternalMonologueResidueRow,
    ResidueLoadResult,
    ResidueScopeCandidate,
    ResidueScopeKind,
    ResidueTriggerScope,
)
from kazusa_ai_chatbot.internal_monologue_residue.projection import (
    project_residue_window,
)

logger = logging.getLogger(__name__)

RESIDUE_COMPONENT = "internal_monologue_residue"


async def load_residue_context(
    *,
    trigger_scope: ResidueTriggerScope,
    current_timestamp_utc: str,
) -> ResidueLoadResult:
    """Load and project the eligible rolling residue window for a trigger.

    Args:
        trigger_scope: Current character, platform, channel, and user scope.
        current_timestamp_utc: Storage UTC timestamp used for projection ages.

    Returns:
        Sanitized load status plus the single L2a prompt-facing context string.
    """

    scope_candidates = build_scope_candidates(trigger_scope)
    scope_keys = [candidate["scope_key"] for candidate in scope_candidates]
    if not scope_keys:
        result = _empty_load_result(status="empty_scope")
        return result

    try:
        rows = await db.list_internal_monologue_residue_rows(
            scope_keys=scope_keys,
            per_scope_limit=INTERNAL_MONOLOGUE_RESIDUE_WINDOW_SIZE,
        )
    except DatabaseOperationError as exc:
        logger.warning(f"Internal monologue residue load failed: {exc}")
        await _record_load_event(
            status="failed",
            selected_count=0,
            candidate_count=0,
        )
        result = _empty_load_result(status="load_failed")
        return result

    selected_rows = select_residue_rows(
        rows=rows,
        scope_candidates=scope_candidates,
        window_size=INTERNAL_MONOLOGUE_RESIDUE_WINDOW_SIZE,
    )
    context = project_residue_window(
        rows=selected_rows,
        current_timestamp_utc=current_timestamp_utc,
        context_char_limit=INTERNAL_MONOLOGUE_RESIDUE_CONTEXT_CHAR_LIMIT,
    )
    status = "loaded" if context else "empty"
    await _record_load_event(
        status=status,
        selected_count=len(selected_rows),
        candidate_count=len(rows),
    )
    result: ResidueLoadResult = {
        "internal_monologue_residue_context": context,
        "selected_count": len(selected_rows),
        "candidate_count": len(rows),
        "scope_order": [
            candidate["scope_kind"]
            for candidate in scope_candidates
        ],
        "status": status,
    }
    return result


def build_scope_candidates(
    trigger_scope: ResidueTriggerScope,
) -> list[ResidueScopeCandidate]:
    """Build deterministic candidate scopes in ownership priority order."""

    character_id = trigger_scope["character_id"]
    platform = trigger_scope["platform"]
    platform_channel_id = trigger_scope["platform_channel_id"]
    channel_type = trigger_scope["channel_type"]
    global_user_id = trigger_scope["global_user_id"]
    candidates: list[ResidueScopeCandidate] = []

    if platform and platform_channel_id and global_user_id:
        candidates.append({
            "scope_kind": "user_thread",
            "scope_key": build_scope_key(
                character_id=character_id,
                scope_kind="user_thread",
                platform=platform,
                platform_channel_id=platform_channel_id,
                global_user_id=global_user_id,
            ),
            "rank": 0,
        })

    if channel_type == "group" and platform and platform_channel_id:
        candidates.append({
            "scope_kind": "group_scene",
            "scope_key": build_scope_key(
                character_id=character_id,
                scope_kind="group_scene",
                platform=platform,
                platform_channel_id=platform_channel_id,
                global_user_id="",
            ),
            "rank": 1,
        })

    candidates.append({
        "scope_kind": "character_global",
        "scope_key": build_scope_key(
            character_id=character_id,
            scope_kind="character_global",
            platform="",
            platform_channel_id="",
            global_user_id="",
        ),
        "rank": 2,
    })
    return candidates


def build_scope_key(
    *,
    character_id: str,
    scope_kind: ResidueScopeKind,
    platform: str,
    platform_channel_id: str,
    global_user_id: str,
) -> str:
    """Return a stable private residue scope key."""

    if scope_kind == "character_global":
        scope_key = f"character_global:{character_id}"
    elif scope_kind == "group_scene":
        scope_key = f"group_scene:{character_id}:{platform}:{platform_channel_id}"
    else:
        scope_key = (
            f"user_thread:{character_id}:{platform}:"
            f"{platform_channel_id}:{global_user_id}"
        )
    return scope_key


def select_residue_window(
    *,
    trigger_scope: ResidueTriggerScope,
    rows: list[InternalMonologueResidueRow],
    window_size: int,
) -> list[InternalMonologueResidueRow]:
    """Select eligible rows for a trigger using production scope priority."""

    scope_candidates = build_scope_candidates(trigger_scope)
    rank_by_scope_kind = {
        candidate["scope_kind"]: candidate["rank"]
        for candidate in scope_candidates
    }
    eligible_rows = [
        row
        for row in rows
        if _row_matches_trigger_scope(
            row=row,
            trigger_scope=trigger_scope,
            rank_by_scope_kind=rank_by_scope_kind,
        )
    ]
    newest_first = sorted(
        eligible_rows,
        key=lambda row: str(row.get("created_at") or ""),
        reverse=True,
    )
    selected_rows = sorted(
        newest_first,
        key=lambda row: rank_by_scope_kind[str(row.get("scope_kind") or "")],
    )[:window_size]
    return selected_rows


def _row_matches_trigger_scope(
    *,
    row: InternalMonologueResidueRow,
    trigger_scope: ResidueTriggerScope,
    rank_by_scope_kind: dict[ResidueScopeKind, int],
) -> bool:
    """Return whether a row is eligible for the trigger scope."""

    row_scope_kind = row.get("scope_kind")
    if row_scope_kind not in rank_by_scope_kind:
        return_value = False
        return return_value

    if str(row.get("character_id") or "") != trigger_scope["character_id"]:
        return_value = False
        return return_value

    if row_scope_kind == "character_global":
        return_value = True
        return return_value

    if str(row.get("platform") or "") != trigger_scope["platform"]:
        return_value = False
        return return_value
    if (
        str(row.get("platform_channel_id") or "")
        != trigger_scope["platform_channel_id"]
    ):
        return_value = False
        return return_value

    if row_scope_kind == "group_scene":
        return_value = True
        return return_value

    return_value = (
        str(row.get("global_user_id") or "") == trigger_scope["global_user_id"]
    )
    return return_value


def select_residue_rows(
    *,
    rows: list[InternalMonologueResidueRow],
    scope_candidates: list[ResidueScopeCandidate],
    window_size: int,
) -> list[InternalMonologueResidueRow]:
    """Rank eligible rows by scope priority, then recency, and cap the window."""

    rank_by_scope = {
        candidate["scope_key"]: candidate["rank"]
        for candidate in scope_candidates
    }
    eligible_rows = [
        row
        for row in rows
        if str(row.get("scope_key") or "") in rank_by_scope
    ]
    newest_first = sorted(
        eligible_rows,
        key=lambda row: str(row.get("created_at") or ""),
        reverse=True,
    )
    selected_rows = sorted(
        newest_first,
        key=lambda row: rank_by_scope[str(row.get("scope_key") or "")],
    )
    window_rows = selected_rows[:window_size]
    return window_rows


def _empty_load_result(*, status: str) -> ResidueLoadResult:
    """Return an empty sanitized load result."""

    result: ResidueLoadResult = {
        "internal_monologue_residue_context": "",
        "selected_count": 0,
        "candidate_count": 0,
        "scope_order": [],
        "status": status,
    }
    return result


async def _record_load_event(
    *,
    status: str,
    selected_count: int,
    candidate_count: int,
) -> None:
    """Record sanitized residue load telemetry."""

    await event_logging.record_database_operation_event(
        component=RESIDUE_COMPONENT,
        collection=db.INTERNAL_MONOLOGUE_RESIDUE_COLLECTION,
        operation_kind="load_residue_context",
        status=status,
        idempotency_result=(
            f"selected:{selected_count};candidates:{candidate_count}"
        ),
        latency_ms=0,
    )
