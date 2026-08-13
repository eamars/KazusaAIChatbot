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
    record_telemetry: bool = True,
) -> ResidueLoadResult:
    """Load and project the eligible rolling residue window for a trigger.

    Args:
        trigger_scope: Current character, platform, channel, and user scope.
        current_timestamp_utc: Storage UTC timestamp used for projection ages.
        record_telemetry: Whether to record the sanitized database-read event.

    Returns:
        Sanitized load status plus the single L2a prompt-facing context string.
    """

    scope_candidates = build_scope_candidates(trigger_scope)
    scope_keys = [candidate["scope_key"] for candidate in scope_candidates]
    if not scope_keys:
        result = _empty_load_result(
            status="empty_scope",
            barrier_disposition="none",
        )
        return result

    try:
        rows = await db.list_internal_monologue_residue_rows(
            scope_keys=scope_keys,
            per_scope_limit=INTERNAL_MONOLOGUE_RESIDUE_WINDOW_SIZE,
        )
    except DatabaseOperationError as exc:
        logger.warning(f"Internal monologue residue load failed: {exc}")
        if record_telemetry:
            await _record_load_event(
                status="failed",
                selected_count=0,
                candidate_count=0,
                scope_kind="targetless",
                barrier_disposition="unknown",
            )
        result = _empty_load_result(
            status="load_failed",
            barrier_disposition="unknown",
        )
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
    barrier_disposition = _barrier_disposition(
        rows=rows,
        scope_candidates=scope_candidates,
    )
    if context:
        status = "loaded"
    elif barrier_disposition == "clear_scope":
        status = "cleared"
    else:
        status = "empty"
    if record_telemetry:
        await _record_load_event(
            status=status,
            selected_count=len(selected_rows),
            candidate_count=len(rows),
            scope_kind=scope_candidates[0]["scope_kind"],
            barrier_disposition=barrier_disposition,
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
        "barrier_disposition": barrier_disposition,
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

    if scope_kind == "group_scene":
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
    eligible_rows = _apply_scope_barriers(eligible_rows)
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

    if not _is_canonical_residue_row(row):
        return False
    row_scope_kind = row.get("scope_kind")
    if row_scope_kind not in rank_by_scope_kind:
        return_value = False
        return return_value

    if str(row.get("character_id") or "") != trigger_scope["character_id"]:
        return_value = False
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
        if _is_canonical_residue_row(row)
        and str(row.get("scope_key") or "") in rank_by_scope
    ]
    eligible_rows = _apply_scope_barriers(eligible_rows)
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


def _apply_scope_barriers(
    rows: list[InternalMonologueResidueRow],
) -> list[InternalMonologueResidueRow]:
    """Drop rows superseded by the newest exact-scope replacement or clear."""

    rows_by_scope: dict[str, list[InternalMonologueResidueRow]] = {}
    for row in rows:
        scope_key = str(row.get("scope_key") or "")
        if scope_key:
            rows_by_scope.setdefault(scope_key, []).append(row)

    eligible: list[InternalMonologueResidueRow] = []
    for scope_rows in rows_by_scope.values():
        ordered = sorted(
            scope_rows,
            key=_residue_order_key,
            reverse=True,
        )
        barrier_index = next(
            (
                index
                for index, row in enumerate(ordered)
                if row.get("disposition") in {"replace_scope", "clear_scope"}
            ),
            None,
        )
        if barrier_index is None:
            eligible.extend(ordered)
        else:
            eligible.extend(ordered[:barrier_index + 1])
    return eligible


def _is_canonical_residue_row(row: InternalMonologueResidueRow) -> bool:
    """Accept only rows written with the canonical v2 storage contract."""

    return (
        row.get("schema_version") == "internal_monologue_residue.v2"
        and bool(str(row.get("operation_id") or ""))
        and row.get("disposition") in {
            "append",
            "replace_scope",
            "clear_scope",
        }
        and "purge_at" in row
    )


def _barrier_disposition(
    *,
    rows: list[InternalMonologueResidueRow],
    scope_candidates: list[ResidueScopeCandidate],
) -> str:
    """Classify the newest barrier in the highest-priority eligible scope."""

    rank_by_scope = {
        candidate["scope_key"]: candidate["rank"]
        for candidate in scope_candidates
    }
    barriers = [
        row
        for row in rows
        if str(row.get("scope_key") or "") in rank_by_scope
        and row.get("disposition") in {"replace_scope", "clear_scope"}
    ]
    if not barriers:
        return "none"
    selected = sorted(
        barriers,
        key=lambda row: (
            rank_by_scope[str(row.get("scope_key") or "")],
            str(row.get("created_at") or ""),
            str(row.get("operation_id") or ""),
        ),
    )[0]
    disposition = selected.get("disposition")
    if disposition in {"replace_scope", "clear_scope"}:
        return disposition
    return "unknown"


def _residue_order_key(
    row: InternalMonologueResidueRow,
) -> tuple[str, str, str]:
    """Provide deterministic newest-first ordering for canonical rows."""

    return (
        str(row.get("created_at") or ""),
        str(row.get("operation_id") or ""),
        str(row.get("residue_id") or ""),
    )


def _empty_load_result(
    *,
    status: str,
    barrier_disposition: str,
) -> ResidueLoadResult:
    """Return an empty sanitized load result."""

    result: ResidueLoadResult = {
        "internal_monologue_residue_context": "",
        "selected_count": 0,
        "candidate_count": 0,
        "scope_order": [],
        "status": status,
        "barrier_disposition": barrier_disposition,
    }
    return result


async def _record_load_event(
    *,
    status: str,
    selected_count: int,
    candidate_count: int,
    scope_kind: str,
    barrier_disposition: str,
) -> None:
    """Record sanitized residue load telemetry."""

    await event_logging.record_continuity_boundary_event(
        component=RESIDUE_COMPONENT,
        boundary="residue_load",
        status=(
            "succeeded"
            if status == "loaded"
            else "empty"
            if status in {"empty", "cleared"}
            else "persistence_failed"
        ),
        scope_kind=(
            scope_kind
            if scope_kind in {"user_thread", "group_scene"}
            else "targetless"
        ),
        candidate_count=candidate_count,
        selected_count=selected_count,
        barrier_disposition=(
            barrier_disposition
            if barrier_disposition in {
                "none",
                "replace_scope",
                "clear_scope",
            }
            else "unknown"
        ),
    )
