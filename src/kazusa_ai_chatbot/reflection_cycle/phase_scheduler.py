"""Pure materialization for reflection phase run intents."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from typing import Literal, TypedDict

from kazusa_ai_chatbot.reflection_cycle.models import ReflectionScopeInput


REFLECTION_PHASE_TRIGGER_KIND = "reflection_phase_slot"
REFLECTION_PHASE_GROUPS_PER_SLOT = 1
REFLECTION_PHASE_ALLOWED_ACTIONS = (
    "reflection_hourly_slot",
    "group_self_cognition_review",
)


class ReflectionPhaseSourceScope(TypedDict):
    """Calendar source identity for one selected reflection scope."""

    scope_ref: str
    platform: str
    platform_channel_id: str
    channel_type: str


class ReflectionPhasePayload(TypedDict):
    """Phase-slot execution metadata carried by a run intent."""

    phase_period_seconds: int
    max_slots_per_period: int
    prompt_version: str
    allowed_actions: list[str]


class ReflectionPhaseRunIntent(TypedDict):
    """Calendar-compatible intent for one composite reflection phase slot."""

    run_id: str
    trigger_kind: Literal["reflection_phase_slot"]
    due_at: str
    period_start_utc: str
    slot_index: int
    offset_seconds: int
    source_scope: ReflectionPhaseSourceScope
    payload: ReflectionPhasePayload
    idempotency_key: str


def build_phase_run_intents(
    *,
    period_start_utc: datetime,
    eligible_scopes: list[ReflectionScopeInput],
    phase_period_seconds: int,
    max_slots_per_period: int,
    min_slot_spacing_seconds: int,
    prompt_version: str,
) -> list[ReflectionPhaseRunIntent]:
    """Build deterministic run intents for one reflection phase period.

    Args:
        period_start_utc: Start of the phase period. Offset-aware datetimes are
            normalized to UTC before intent fields are rendered.
        eligible_scopes: Monitor-eligible channel scopes captured for this
            period.
        phase_period_seconds: Length of the phase period in seconds.
        max_slots_per_period: Maximum channel scopes to schedule in the period.
        min_slot_spacing_seconds: Minimum allowed spacing between slot offsets.
        prompt_version: Reflection prompt contract version carried in payloads.

    Returns:
        Calendar-compatible run-intent dictionaries ordered by due slot.

    Raises:
        ValueError: If period timing or slot counts cannot form valid offsets.
    """

    _validate_phase_config(
        phase_period_seconds=phase_period_seconds,
        max_slots_per_period=max_slots_per_period,
        min_slot_spacing_seconds=min_slot_spacing_seconds,
    )
    normalized_period_start = _normalize_period_start(period_start_utc)
    period_start_iso = normalized_period_start.isoformat()
    sorted_scopes = sorted(eligible_scopes, key=_scope_identity)
    selected_scopes = _select_period_scopes(
        period_start_utc=normalized_period_start,
        sorted_scopes=sorted_scopes,
        phase_period_seconds=phase_period_seconds,
        max_slots_per_period=max_slots_per_period,
    )
    offset_step_seconds = _offset_step_seconds(
        phase_period_seconds=phase_period_seconds,
        max_slots_per_period=max_slots_per_period,
        min_slot_spacing_seconds=min_slot_spacing_seconds,
    )

    intents: list[ReflectionPhaseRunIntent] = []
    for slot_index, scope in enumerate(selected_scopes):
        offset_seconds = slot_index * offset_step_seconds
        due_at = normalized_period_start + timedelta(seconds=offset_seconds)
        due_at_iso = due_at.isoformat()
        source_scope = _source_scope(scope)
        run_id = _phase_run_id(
            period_start_utc=period_start_iso,
            slot_index=slot_index,
            source_scope=source_scope,
        )
        payload = {
            "phase_period_seconds": phase_period_seconds,
            "max_slots_per_period": max_slots_per_period,
            "prompt_version": prompt_version,
            "allowed_actions": list(REFLECTION_PHASE_ALLOWED_ACTIONS),
        }
        intent: ReflectionPhaseRunIntent = {
            "run_id": run_id,
            "trigger_kind": REFLECTION_PHASE_TRIGGER_KIND,
            "due_at": due_at_iso,
            "period_start_utc": period_start_iso,
            "slot_index": slot_index,
            "offset_seconds": offset_seconds,
            "source_scope": source_scope,
            "payload": payload,
            "idempotency_key": run_id,
        }
        intents.append(intent)

    return intents


def _validate_phase_config(
    *,
    phase_period_seconds: int,
    max_slots_per_period: int,
    min_slot_spacing_seconds: int,
) -> None:
    """Validate timing values used to place phase slots inside a period."""

    if phase_period_seconds < 1:
        raise ValueError("phase_period_seconds must be >= 1")
    if max_slots_per_period < 1:
        raise ValueError("max_slots_per_period must be >= 1")
    if min_slot_spacing_seconds < 1:
        raise ValueError("min_slot_spacing_seconds must be >= 1")

    allowed_slot_count = (
        (phase_period_seconds - 1) // min_slot_spacing_seconds
    ) + 1
    if max_slots_per_period > allowed_slot_count:
        raise ValueError(
            "max_slots_per_period cannot fit inside phase_period_seconds "
            "with min_slot_spacing_seconds"
        )


def _normalize_period_start(period_start_utc: datetime) -> datetime:
    """Return the phase period start as an aware UTC datetime."""

    if period_start_utc.tzinfo is None:
        raise ValueError("period_start_utc must be timezone-aware")

    normalized_period_start = period_start_utc.astimezone(timezone.utc)
    return normalized_period_start


def _select_period_scopes(
    *,
    period_start_utc: datetime,
    sorted_scopes: list[ReflectionScopeInput],
    phase_period_seconds: int,
    max_slots_per_period: int,
) -> list[ReflectionScopeInput]:
    """Select scopes for a period, rotating overflow across later periods."""

    if len(sorted_scopes) <= max_slots_per_period:
        selected_scopes = sorted_scopes[:]
        return selected_scopes

    period_index = math.floor(
        period_start_utc.timestamp() / phase_period_seconds
    )
    start_index = (period_index * max_slots_per_period) % len(sorted_scopes)
    selected_scopes = []
    for offset in range(max_slots_per_period):
        selected_index = (start_index + offset) % len(sorted_scopes)
        selected_scopes.append(sorted_scopes[selected_index])

    return selected_scopes


def _offset_step_seconds(
    *,
    phase_period_seconds: int,
    max_slots_per_period: int,
    min_slot_spacing_seconds: int,
) -> int:
    """Return the spacing used between materialized slot offsets."""

    if max_slots_per_period == 1:
        offset_step_seconds = phase_period_seconds
    else:
        even_step_seconds = phase_period_seconds // max_slots_per_period
        offset_step_seconds = max(
            even_step_seconds,
            min_slot_spacing_seconds,
        )

    return offset_step_seconds


def _source_scope(scope: ReflectionScopeInput) -> ReflectionPhaseSourceScope:
    """Project a reflection scope into the calendar source-scope shape."""

    source_scope: ReflectionPhaseSourceScope = {
        "scope_ref": scope.scope_ref,
        "platform": scope.platform,
        "platform_channel_id": scope.platform_channel_id,
        "channel_type": scope.channel_type,
    }
    return source_scope


def _scope_identity(scope: ReflectionScopeInput) -> tuple[str, str, str, str]:
    """Return the stable identity used for deterministic scope ordering."""

    identity = (
        scope.scope_ref,
        scope.platform,
        scope.platform_channel_id,
        scope.channel_type,
    )
    return identity


def _phase_run_id(
    *,
    period_start_utc: str,
    slot_index: int,
    source_scope: ReflectionPhaseSourceScope,
) -> str:
    """Build the stable duplicate-suppression id for one phase slot."""

    identity = {
        "trigger_kind": REFLECTION_PHASE_TRIGGER_KIND,
        "period_start_utc": period_start_utc,
        "slot_index": slot_index,
        "source_scope": source_scope,
    }
    serialized = json.dumps(
        identity,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    run_id = f"reflection_phase_slot_{digest[:32]}"
    return run_id
