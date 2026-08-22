"""Semantic projection from canonical state into bounded model context."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from zoneinfo import ZoneInfoNotFoundError

from kazusa_ai_chatbot.cognition_shared.prompt_budget import (
    CHARACTER_OPERATIONAL_CONSUMER_ROLES,
    CHARACTER_OPERATIONAL_CONTEXT_DIGEST_CHARS,
    MAX_CONTEXT_AFFECT_ROWS,
    MAX_CONTEXT_PRESSURE_ROWS,
    MAX_RELATIONSHIP_AFFECT_ROWS,
    MAX_RELATIONSHIP_CAUSAL_ROWS,
    MAX_RELATIONSHIP_CAUSAL_SUMMARY_CHARS,
    canonical_digest,
    fit_character_operational_context,
    fit_relationship_operational_context,
)
from kazusa_ai_chatbot.cognition_shared.state_reducers import (
    apply_character_elapsed_decay,
)
from kazusa_ai_chatbot.time_boundary import (
    local_minutes_in_zone,
    local_period_bounds,
)

RAW_STATE_KEYS = frozenset({
    "entity_id",
    "owner_user_id",
    "created_at",
    "updated_at",
    "started_at",
    "last_reinforced_at",
    "primary_root",
    "root_refs",
    "evidence_refs",
    "state_scope",
    "schema_version",
    "scope",
    "kind",
})

CHARACTER_OPERATIONAL_STATE_VIEW_SCHEMA = "character_operational_state_view.v1"
CHARACTER_OPERATIONAL_CONTEXT_SCHEMA = "character_operational_context.v1"
RELATIONSHIP_OPERATIONAL_CONTEXT_SCHEMA = "relationship_operational_context.v1"
RELATIONSHIP_AXIS_FIELDS = (
    "familiarity",
    "positive_regard",
    "trust",
    "attachment",
    "desired_closeness",
    "perceived_closeness",
    "care",
    "boundary_safety",
    "exclusivity",
    "unresolved_injury",
    "salience",
)
_SIGNED_RELATIONSHIP_AXES = frozenset({
    "positive_regard",
    "trust",
    "boundary_safety",
})
MAX_CHARACTER_OPERATIONAL_AFFECT_ROWS = 21
MAX_CHARACTER_OPERATIONAL_PRESSURE_ROWS = 8
OPERATIONAL_PRESSURE_THRESHOLD = 40
# New causal entities are retained natively at salience >= 25, so any committed
# active entity above that floor is a durable operational pressure and must
# stay observable even when it is not yet significant enough for a typed class.
OPERATIONAL_ENTITY_PRESSURE_SALIENCE_FLOOR = 25
MINUTES_PER_DAY = 24 * 60

CHARACTER_SLEEP_PHASE_OUTSIDE = "清醒时段"
CHARACTER_SLEEP_PHASE_IN_WINDOW = "睡眠中"
CHARACTER_SLEEP_PHASE_WAKE_PREP = "即将醒来"

_RELATIONSHIP_REQUIRED_EMOTION_IDS = frozenset({
    "love_attachment",
    "jealousy",
    "loneliness",
})
_ENTITY_PRESSURE_STATUSES = {
    "goals": {"pursuing", "blocked"},
    "threats": {"active"},
    "active_events": {"active"},
    "knowledge_gaps": {"open", "reduced"},
}
_ENTITY_KIND_BY_FIELD = {
    "goals": "goal",
    "threats": "threat",
    "active_events": "event",
    "knowledge_gaps": "knowledge_gap",
}


def evidence_source_identity(
    evidence_ref: Mapping[str, Any],
) -> tuple[str, str]:
    """Return the exact provenance identity carried by one evidence ref."""

    source_kind = evidence_ref.get("source_kind")
    source_id = evidence_ref.get("source_id")
    if (
        not isinstance(source_kind, str)
        or not source_kind
        or not isinstance(source_id, str)
        or not source_id
    ):
        raise ValueError("evidence provenance identity is invalid")
    identity = (source_kind, source_id)
    return identity


@dataclass(frozen=True)
class PromptProjectionV2:
    """Hold prompt-safe values and private handle bindings separately."""

    payload: dict[str, Any]
    handle_to_ref: dict[str, dict[str, str]]
    identity_by_question: dict[str, dict[str, object]] = field(
        default_factory=dict,
    )


def project_numeric_band(value: int, *, signed: bool = False) -> str:
    """Translate a bounded scalar into the frozen semantic band vocabulary."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("projection value must be an integer")
    if signed:
        if not -100 <= value <= 100:
            raise ValueError("signed projection value is out of range")
        if value <= -61:
            return "强烈负向"
        if value <= -21:
            return "负向"
        if value <= 20:
            return "中性或混合"
        if value <= 60:
            return "正向"
        return "强烈正向"
    if not 0 <= value <= 100:
        raise ValueError("unsigned projection value is out of range")
    if value == 0:
        return "无"
    if value <= 20:
        return "极低"
    if value <= 40:
        return "低"
    if value <= 60:
        return "中等"
    if value <= 80:
        return "高"
    return "极高"


def project_relationship_axis(field_name: str, value: int) -> str:
    """Return the domain-specific semantic meaning of one native axis.

    Each axis keeps its own zero and band meanings so that unestablished
    trust, unproven boundary history, and absent care are never conflated
    with neutral permission. Values are validated against the native axis
    range and never enter the model as numbers.

    Args:
        field_name: One of the eleven native relationship axis names.
        value: Integer axis value inside the native range.

    Returns:
        A bounded Simplified Chinese semantic descriptor.

    Raises:
        ValueError: For an unknown axis, a Boolean, or an out-of-range value.
    """

    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("relationship axis value must be an integer")
    if field_name not in RELATIONSHIP_AXIS_FIELDS:
        raise ValueError(f"unknown relationship axis: {field_name}")
    if field_name in _SIGNED_RELATIONSHIP_AXES:
        if not -100 <= value <= 100:
            raise ValueError("relationship signed axis value is out of range")
    elif not 0 <= value <= 100:
        raise ValueError("relationship axis value is out of range")
    descriptor = _RELATIONSHIP_AXIS_BANDS[field_name]
    if value == 0:
        return descriptor.zero
    if field_name in _SIGNED_RELATIONSHIP_AXES:
        if value <= -61:
            return descriptor.strong_negative
        if value <= -21:
            return descriptor.negative
        if value <= -1:
            return descriptor.mild_negative
        if value <= 20:
            return descriptor.mild_positive
        if value <= 40:
            return descriptor.developing_positive
        if value <= 60:
            return descriptor.established_positive
        if value <= 80:
            return descriptor.strong_positive
        return descriptor.very_strong_positive
    if value <= 20:
        return descriptor.emerging
    if value <= 40:
        return descriptor.low
    if value <= 60:
        return descriptor.medium
    if value <= 80:
        return descriptor.high
    return descriptor.very_high


class _AxisSemantics:
    """Hold one axis' bounded semantic bands as stable model-facing text."""

    def __init__(
        self,
        *,
        zero: str,
        emerging: str,
        low: str,
        medium: str,
        high: str,
        very_high: str,
        strong_negative: str = "",
        negative: str = "",
        mild_negative: str = "",
        mild_positive: str = "",
        developing_positive: str = "",
        established_positive: str = "",
        strong_positive: str = "",
        very_strong_positive: str = "",
    ) -> None:
        self.zero = zero
        self.emerging = emerging
        self.low = low
        self.medium = medium
        self.high = high
        self.very_high = very_high
        self.strong_negative = strong_negative
        self.negative = negative
        self.mild_negative = mild_negative
        self.mild_positive = mild_positive
        self.developing_positive = developing_positive
        self.established_positive = established_positive
        self.strong_positive = strong_positive
        self.very_strong_positive = very_strong_positive


_RELATIONSHIP_AXIS_BANDS: dict[str, _AxisSemantics] = {
    "familiarity": _AxisSemantics(
        zero="完全不认识",
        emerging="几乎不认识，仅有浅淡印象",
        low="只有有限的初步了解",
        medium="有一定熟悉度",
        high="比较熟悉",
        very_high="非常熟悉",
    ),
    "positive_regard": _AxisSemantics(
        zero="尚未形成明确好坏观感",
        emerging="观感轻度正面",
        low="正面观感正在形成",
        medium="观感明显正面",
        high="观感强烈正面",
        very_high="观感非常强烈正面",
        strong_negative="观感强烈负面",
        negative="观感明显负面",
        mild_negative="观感轻度负面",
        mild_positive="观感轻度正面",
        developing_positive="正面观感正在形成",
        established_positive="观感明显正面",
        strong_positive="观感强烈正面",
        very_strong_positive="观感非常强烈正面",
    ),
    "trust": _AxisSemantics(
        zero="信任尚未建立",
        emerging="信任刚刚开始建立",
        low="信任正在建立",
        medium="已有相当信任",
        high="高度信任",
        very_high="非常深厚的信任",
        strong_negative="信任被严重破坏",
        negative="信任明显受损",
        mild_negative="信任轻度受损",
        mild_positive="信任刚刚开始建立",
        developing_positive="信任正在建立",
        established_positive="已有相当信任",
        strong_positive="高度信任",
        very_strong_positive="非常深厚的信任",
    ),
    "attachment": _AxisSemantics(
        zero="尚未形成依恋",
        emerging="依恋刚刚萌芽",
        low="依恋正在形成",
        medium="已有稳定依恋",
        high="依恋深厚",
        very_high="依恋非常深厚",
    ),
    "desired_closeness": _AxisSemantics(
        zero="对更亲近没有任何愿望",
        emerging="对更亲近几乎没有愿望",
        low="有少量亲近愿望",
        medium="明确希望更亲近",
        high="强烈希望更亲近",
        very_high="非常渴望更亲近",
    ),
    "perceived_closeness": _AxisSemantics(
        zero="完全没有亲近感",
        emerging="几乎没有亲近感",
        low="亲近感较低",
        medium="有中等亲近感",
        high="亲近感较强",
        very_high="非常亲近",
    ),
    "care": _AxisSemantics(
        zero="尚未投入关心",
        emerging="关心刚开始投入",
        low="关心正在积累",
        medium="已有实质关心",
        high="关心深厚",
        very_high="非常深厚的关心",
    ),
    "boundary_safety": _AxisSemantics(
        zero="边界历史尚未建立",
        emerging="边界相处刚刚开始",
        low="边界相处正在建立",
        medium="已有安全的边界相处",
        high="边界相处很安全",
        very_high="边界相处非常安全",
        strong_negative="边界历史受到严重侵害",
        negative="边界历史明显受损",
        mild_negative="边界历史轻度受损",
        mild_positive="边界相处刚刚开始",
        developing_positive="边界相处正在建立",
        established_positive="已有安全的边界相处",
        strong_positive="边界相处很安全",
        very_strong_positive="边界相处非常安全",
    ),
    "exclusivity": _AxisSemantics(
        zero="没有任何排他性",
        emerging="排他性很弱",
        low="有少量排他倾向",
        medium="明确排他",
        high="排他性强烈",
        very_high="完全排他",
    ),
    "unresolved_injury": _AxisSemantics(
        zero="没有未化解的伤害",
        emerging="几乎无未化解伤害",
        low="有少量未化解伤害",
        medium="有未化解的伤害",
        high="有较重的未化解伤害",
        very_high="有严重的未化解伤害",
    ),
    "salience": _AxisSemantics(
        zero="当前关系没有浮现",
        emerging="关系几乎不被注意",
        low="关系关注度较低",
        medium="关系处于中等关注",
        high="关系当前显著",
        very_high="关系当前非常突出",
    ),
}


def project_duration(started_at: str, now: str) -> str:
    """Translate elapsed UTC time into the frozen semantic duration labels."""

    elapsed = _parse_utc(now) - _parse_utc(started_at)
    seconds = max(0.0, elapsed.total_seconds())
    if seconds < 10 * 60:
        return "即时"
    if seconds < 2 * 3600:
        return "近期"
    if seconds < 24 * 3600:
        return "较早"
    if seconds < 7 * 24 * 3600:
        return "最近几天内"
    return "较久以前"


def project_character_sleep_phase(
    now: datetime,
    *,
    sleep_local_period: str,
    character_time_zone: str,
    wake_prep_minutes: int,
) -> str:
    """Translate the configured sleep window into a frozen phase label.

    The window is the same half-open local period that
    ``is_self_cognition_sleep_period`` reports: inclusive start, exclusive
    end, with overnight windows wrapping past midnight. The two in-window
    labels partition that window, with the wake-prep label covering the final
    ``wake_prep_minutes`` before the exclusive end. An empty
    ``sleep_local_period`` disables the window and returns the outside label.

    Args:
        now: Timezone-aware instant to project.
        sleep_local_period: Exact ``HH:MM-HH:MM`` local period or empty text.
        character_time_zone: IANA timezone used for the local projection.
        wake_prep_minutes: Positive minutes before the window end covered by
            the wake-prep label.

    Returns:
        One frozen label: ``CHARACTER_SLEEP_PHASE_OUTSIDE``,
        ``CHARACTER_SLEEP_PHASE_IN_WINDOW``, or
        ``CHARACTER_SLEEP_PHASE_WAKE_PREP``.

    Raises:
        ValueError: If ``now`` is timezone-naive, the period or timezone is
            invalid, or ``wake_prep_minutes`` is not a positive integer.
    """

    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    if not isinstance(sleep_local_period, str):
        raise ValueError("sleep_local_period must be a string")
    if (
        isinstance(wake_prep_minutes, bool)
        or not isinstance(wake_prep_minutes, int)
        or wake_prep_minutes < 1
    ):
        raise ValueError("wake_prep_minutes must be a positive integer")

    clean_period = sleep_local_period.strip()
    if not clean_period:
        return CHARACTER_SLEEP_PHASE_OUTSIDE

    try:
        start_minutes, end_minutes = local_period_bounds(clean_period)
        current_minutes = local_minutes_in_zone(
            now,
            time_zone=character_time_zone,
        )
    except (TypeError, ZoneInfoNotFoundError) as exc:
        raise ValueError(
            f"invalid character time zone: {character_time_zone!r}"
        ) from exc

    if start_minutes < end_minutes:
        in_window = start_minutes <= current_minutes < end_minutes
        end_minutes_in_day = end_minutes
        projected_minutes = current_minutes
    else:
        in_window = (
            current_minutes >= start_minutes
            or current_minutes < end_minutes
        )
        end_minutes_in_day = end_minutes + MINUTES_PER_DAY
        projected_minutes = (
            current_minutes + MINUTES_PER_DAY
            if current_minutes < start_minutes
            else current_minutes
        )

    if not in_window:
        return CHARACTER_SLEEP_PHASE_OUTSIDE
    if projected_minutes >= end_minutes_in_day - wake_prep_minutes:
        return CHARACTER_SLEEP_PHASE_WAKE_PREP
    return CHARACTER_SLEEP_PHASE_IN_WINDOW


def project_relationship_context(
    user_state: Mapping[str, Any],
    *,
    effective_at: str | None = None,
) -> dict[str, Any]:
    """Project current-user relationship state for its permitted consumer.

    A complete user cognition state produces the canonical bounded operational
    projection. Existing prompt-only callers may pass a relationship mapping;
    they retain the established qualitative projection and do not receive
    causal rows or raw operational values.
    """

    if user_state.get("state_scope") == "user":
        if effective_at is None:
            raise ValueError(
                "relationship operational projection requires effective_at"
            )
        projected_context = _project_user_relationship_context(
            user_state,
            effective_at=effective_at,
        )
        return projected_context
    if effective_at is not None:
        raise ValueError(
            "relationship mappings do not support an operational effective_at"
        )
    prompt_context = _project_relationship_prompt_context(user_state)
    return prompt_context


def project_character_operational_state(
    state: Mapping[str, Any],
    *,
    effective_at: str,
) -> dict[str, Any]:
    """Build the complete redacted operational view from character state.

    The persisted document remains untouched. Ordinary elapsed fading is
    applied only to the in-memory effective view, while source and view hashes
    retain auditable identities for the persisted source and redacted result.
    """

    if state["state_scope"] != "character":
        raise ValueError(
            "character operational projection requires character state"
        )
    source_updated_at = state["updated_at"]
    elapsed_seconds = _elapsed_seconds(source_updated_at, effective_at)
    if _has_complete_activation_rows(state["affect_activations"]):
        effective_state = apply_character_elapsed_decay(
            state,
            elapsed_seconds=elapsed_seconds,
        )
    else:
        effective_state = deepcopy(dict(state))
    affect_rows = _project_character_affect_rows(
        effective_state,
        effective_at=effective_at,
    )
    pressure_rows = _project_character_pressure_rows(
        effective_state,
        effective_at=effective_at,
    )
    source_digest = canonical_digest(dict(state))
    view_without_digest = {
        "schema_version": CHARACTER_OPERATIONAL_STATE_VIEW_SCHEMA,
        "source_updated_at": source_updated_at,
        "effective_at": effective_at,
        "source_digest": source_digest,
        "affect": affect_rows,
        "pressures": pressure_rows,
    }
    view = {
        **view_without_digest,
        "view_digest": canonical_digest(view_without_digest),
    }
    return view


def select_character_operational_context(
    state_view: Mapping[str, Any],
    *,
    consumer_role: str,
) -> dict[str, Any]:
    """Select one bounded model-facing context from a full operational view."""

    if consumer_role not in CHARACTER_OPERATIONAL_CONSUMER_ROLES:
        raise ValueError(f"unsupported operational consumer role: {consumer_role}")
    if state_view["schema_version"] != CHARACTER_OPERATIONAL_STATE_VIEW_SCHEMA:
        raise ValueError("unsupported character operational state view")
    affect_rows = [
        deepcopy(dict(row))
        for row in state_view["affect"][:MAX_CONTEXT_AFFECT_ROWS]
        if isinstance(row, Mapping)
    ]
    pressure_limit = (
        0
        if consumer_role == "surface"
        else MAX_CONTEXT_PRESSURE_ROWS
    )
    pressure_rows = [
        deepcopy(dict(row))
        for row in state_view["pressures"][:pressure_limit]
        if isinstance(row, Mapping)
    ]
    context_without_digest = {
        "schema_version": CHARACTER_OPERATIONAL_CONTEXT_SCHEMA,
        "source_updated_at": state_view["source_updated_at"],
        "effective_at": state_view["effective_at"],
        "view_digest": state_view["view_digest"],
        "consumer_role": consumer_role,
        "affect": affect_rows,
        "pressures": pressure_rows,
    }
    context_with_placeholder = {
        **context_without_digest,
        "context_digest": "0" * CHARACTER_OPERATIONAL_CONTEXT_DIGEST_CHARS,
    }
    fit_result = fit_character_operational_context(context_with_placeholder)
    context = fit_result.payload
    return context


def _has_complete_activation_rows(activations: Sequence[Any]) -> bool:
    """Return whether every activation can undergo deterministic fading."""

    return all(_is_complete_activation(activation) for activation in activations)


def _is_complete_activation(activation: Any) -> bool:
    """Return whether one activation has the lifecycle fields projection needs."""

    required_fields = {
        "emotion_id",
        "primary_root",
        "root_refs",
        "phase",
        "score",
        "peak_score",
        "trend",
        "cause_status",
        "started_at",
        "updated_at",
        "last_reinforced_at",
    }
    is_complete = (
        isinstance(activation, Mapping)
        and required_fields.issubset(activation)
    )
    return is_complete


def _project_relationship_prompt_context(
    relationship: Mapping[str, Any],
) -> dict[str, Any]:
    """Project a relationship mapping into the established prompt-safe bands."""

    axes: dict[str, str] = {}
    for field_name in RELATIONSHIP_AXIS_FIELDS:
        value = relationship.get(field_name)
        if isinstance(value, int) and not isinstance(value, bool):
            axes[field_name] = project_relationship_axis(field_name, value)
    context = {
        "relationship_summary": "当前关系背景",
        "axes": axes,
    }
    return context


def _project_user_relationship_context(
    user_state: Mapping[str, Any],
    *,
    effective_at: str,
) -> dict[str, Any]:
    """Project one user's native relationship rows without cross-user state."""

    relationship = user_state["relationship"]
    relationship_id = relationship["relationship_id"]
    axes = {
        field_name: relationship[field_name]
        for field_name in (
            "familiarity",
            "positive_regard",
            "trust",
            "attachment",
            "desired_closeness",
            "perceived_closeness",
            "care",
            "boundary_safety",
            "exclusivity",
            "unresolved_injury",
            "salience",
        )
    }
    causal_rows = _project_relationship_causal_rows(
        user_state,
        relationship_id=relationship_id,
        effective_at=effective_at,
    )
    affect_rows = _project_relationship_affect_rows(
        user_state,
        relationship_id=relationship_id,
        effective_at=effective_at,
    )
    relationship_context = {
        "schema_version": RELATIONSHIP_OPERATIONAL_CONTEXT_SCHEMA,
        "relationship_id": relationship_id,
        "axes": axes,
        "causal_context": causal_rows,
        "affect": affect_rows,
        "relationship_freshness": project_duration(
            relationship["updated_at"],
            effective_at,
        ),
        "evidence_freshness": _relationship_evidence_freshness(
            relationship["evidence_refs"],
            effective_at=effective_at,
        ),
    }
    fit_result = fit_relationship_operational_context(relationship_context)
    return fit_result.payload


def _project_character_affect_rows(
    state: Mapping[str, Any],
    *,
    effective_at: str,
) -> list[dict[str, str]]:
    """Project active character affect rows without root identifiers."""

    sortable_rows: list[tuple[int, int, str, dict[str, str]]] = []
    for activation in state["affect_activations"]:
        if not _is_complete_activation(activation):
            continue
        emotion_id = activation["emotion_id"]
        primary_root = activation["primary_root"]
        if (
            emotion_id in _RELATIONSHIP_REQUIRED_EMOTION_IDS
            or not isinstance(primary_root, Mapping)
            or primary_root["scope"] != "character"
            or activation["phase"] not in {"active", "fading"}
        ):
            continue
        score = activation["score"]
        if isinstance(score, bool) or not isinstance(score, int):
            continue
        row = {
            "emotion_id": emotion_id,
            "intensity": project_numeric_band(score),
            "phase": activation["phase"],
            "trend": activation["trend"],
            "root_kind": primary_root["kind"],
            "cause_class": _activation_cause_class(
                state,
                primary_root,
                emotion_id=emotion_id,
            ),
            "freshness": project_duration(
                activation["updated_at"],
                effective_at,
            ),
        }
        phase_rank = 0 if activation["phase"] == "active" else 1
        sortable_rows.append((
            phase_rank,
            -score,
            activation["updated_at"],
            row,
        ))
    sortable_rows.sort(key=lambda item: item[:3])
    affect_rows = [
        row
        for _, _, _, row in sortable_rows[:MAX_CHARACTER_OPERATIONAL_AFFECT_ROWS]
    ]
    return affect_rows


def _project_character_pressure_rows(
    state: Mapping[str, Any],
    *,
    effective_at: str,
) -> list[dict[str, str]]:
    """Project eligible character pressure rows in stable urgency order."""

    sortable_rows: list[tuple[int, str, dict[str, str]]] = []
    for field_name, statuses in _ENTITY_PRESSURE_STATUSES.items():
        entity_kind = _ENTITY_KIND_BY_FIELD[field_name]
        for entity in state[field_name]:
            if not isinstance(entity, Mapping):
                continue
            salience = entity["salience"]
            if (
                entity["status"] not in statuses
                or isinstance(salience, bool)
                or not isinstance(salience, int)
                or salience < OPERATIONAL_ENTITY_PRESSURE_SALIENCE_FLOOR
            ):
                continue
            row = {
                "kind": entity_kind,
                "salience": project_numeric_band(salience),
                "lifecycle": entity["status"],
                "cause_class": _cause_class_for_entity(
                    entity_kind,
                    entity,
                ),
                "freshness": project_duration(
                    entity["updated_at"],
                    effective_at,
                ),
            }
            sortable_rows.append((-salience, entity["updated_at"], row))
    for drive_id, drive in state["drives"].items():
        pressure = drive["pressure"]
        if pressure < OPERATIONAL_PRESSURE_THRESHOLD:
            continue
        row = {
            "kind": "drive",
            "salience": project_numeric_band(pressure),
            "lifecycle": "active",
            "cause_class": _cause_class_for_drive(drive_id, pressure),
            "freshness": project_duration(state["updated_at"], effective_at),
        }
        sortable_rows.append((-pressure, drive_id, row))
    meaning = state["meaning_state"]
    meaning_pressure = max(
        0,
        100 - min(meaning["purpose_coherence"], meaning["agency"]),
    )
    if meaning_pressure >= OPERATIONAL_PRESSURE_THRESHOLD:
        row = {
            "kind": "meaning",
            "salience": project_numeric_band(meaning_pressure),
            "lifecycle": "active",
            "cause_class": "meaning_pressure",
            "freshness": project_duration(state["updated_at"], effective_at),
        }
        sortable_rows.append((-meaning_pressure, "meaning", row))
    sortable_rows.sort(key=lambda item: item[:2])
    pressure_rows = [
        row
        for _, _, row in sortable_rows[:MAX_CHARACTER_OPERATIONAL_PRESSURE_ROWS]
    ]
    return pressure_rows


def _project_relationship_causal_rows(
    user_state: Mapping[str, Any],
    *,
    relationship_id: str,
    effective_at: str,
) -> list[dict[str, str]]:
    """Select the two strongest native rows targeting one relationship."""

    sortable_rows: list[tuple[int, str, dict[str, str]]] = []
    for field_name, entity_kind in _ENTITY_KIND_BY_FIELD.items():
        for entity in user_state[field_name]:
            if (
                not isinstance(entity, Mapping)
                or not _targets_relationship(entity, relationship_id)
            ):
                continue
            summary = _normalized_summary(entity["description"])
            row = {
                "entity_kind": entity_kind,
                "semantic_summary": summary,
                "salience": project_numeric_band(entity["salience"]),
                "lifecycle": entity["status"],
                "freshness": project_duration(entity["updated_at"], effective_at),
            }
            sortable_rows.append((
                -entity["salience"],
                entity["updated_at"],
                row,
            ))
    sortable_rows.sort(key=lambda item: item[:2])
    causal_rows = [
        row
        for _, _, row in sortable_rows[:MAX_RELATIONSHIP_CAUSAL_ROWS]
    ]
    return causal_rows


def _project_relationship_affect_rows(
    user_state: Mapping[str, Any],
    *,
    relationship_id: str,
    effective_at: str,
) -> list[dict[str, str]]:
    """Select the two strongest affects rooted in one relationship."""

    sortable_rows: list[tuple[int, str, dict[str, str]]] = []
    for activation in user_state["affect_activations"]:
        if not isinstance(activation, Mapping):
            continue
        root = activation["primary_root"]
        if (
            not isinstance(root, Mapping)
            or root["kind"] != "relationship"
            or root["entity_id"] != relationship_id
            or activation["phase"] not in {"active", "fading"}
        ):
            continue
        row = {
            "emotion_id": activation["emotion_id"],
            "intensity": project_numeric_band(activation["score"]),
            "phase": activation["phase"],
            "trend": activation["trend"],
            "freshness": project_duration(
                activation["updated_at"],
                effective_at,
            ),
        }
        sortable_rows.append((
            -activation["score"],
            activation["updated_at"],
            row,
        ))
    sortable_rows.sort(key=lambda item: item[:2])
    affect_rows = [
        row
        for _, _, row in sortable_rows[:MAX_RELATIONSHIP_AFFECT_ROWS]
    ]
    return affect_rows


def _activation_cause_class(
    state: Mapping[str, Any],
    root: Mapping[str, Any],
    *,
    emotion_id: str,
) -> str:
    """Map one activation root to its closed source-free cause class."""

    root_kind = root["kind"]
    if root_kind == "meaning":
        return "meaning_pressure"
    if root_kind == "drive":
        drive = state["drives"][root["entity_id"]]
        return _cause_class_for_drive(root["entity_id"], drive["pressure"])
    entity = _root_entity(state, root)
    if entity is None:
        return (
            "connection_warmth"
            if emotion_id in {"joy", "gratitude", "compassion_empathy"}
            else "general_activation"
        )
    cause_class = _cause_class_for_entity(
        root_kind,
        entity,
        emotion_id=emotion_id,
    )
    return cause_class


def _cause_class_for_entity(
    entity_kind: str,
    entity: Mapping[str, Any],
    *,
    emotion_id: str | None = None,
) -> str:
    """Classify native pressure with the frozen first-match predicate order."""

    if entity_kind == "threat":
        return "safety_pressure"
    if entity_kind == "knowledge_gap":
        return "uncertainty_pressure"
    if entity_kind == "goal":
        goal_kind = entity["goal_kind"]
        if goal_kind == "autonomy_boundary":
            return "boundary_pressure"
        if goal_kind == "moral_repair":
            return "repair_pressure"
        if goal_kind == "safety":
            return "safety_pressure"
        if goal_kind == "loss_recovery":
            return "loss_pressure"
        if goal_kind == "epistemic_exploration":
            return "uncertainty_pressure"
        if goal_kind == "meaning_reconstruction":
            return "meaning_pressure"
        if goal_kind == "self_improvement":
            return "competence_pressure"
        return "goal_pressure"
    if entity_kind == "event":
        if _event_axis_at_least(
            entity,
            ("unfairness", "norm_violation", "exposure", "identity_threat"),
        ):
            return "boundary_pressure"
        if (
            entity["repair_need"] >= OPERATIONAL_PRESSURE_THRESHOLD
            and _event_axis_at_least(entity, ("harm", "norm_violation"))
        ):
            return "repair_pressure"
        if entity["harm"] >= OPERATIONAL_PRESSURE_THRESHOLD:
            return "safety_pressure"
        if entity["temporal_loss"] >= OPERATIONAL_PRESSURE_THRESHOLD:
            return "loss_pressure"
        if entity["memory_warmth"] >= OPERATIONAL_PRESSURE_THRESHOLD:
            return "connection_warmth"
        if (
            entity["expectation_mismatch"] >= OPERATIONAL_PRESSURE_THRESHOLD
            and _has_group_or_third_party_role(entity)
        ):
            return "relationship_strain"
    if emotion_id in {"joy", "gratitude", "compassion_empathy"}:
        return "connection_warmth"
    return "general_activation"


def _cause_class_for_drive(drive_id: str, pressure: int) -> str:
    """Map one active drive pressure to its closed operational class."""

    if (
        drive_id == "competence"
        and pressure >= OPERATIONAL_PRESSURE_THRESHOLD
    ):
        return "competence_pressure"
    return "goal_pressure"


def _root_entity(
    state: Mapping[str, Any],
    root: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    """Find the native entity referenced by one character activation root."""

    field_name = {
        "goal": "goals",
        "threat": "threats",
        "event": "active_events",
        "knowledge_gap": "knowledge_gaps",
    }.get(root["kind"])
    if field_name is None:
        return None
    for entity in state[field_name]:
        if entity["entity_id"] == root["entity_id"]:
            matched_entity = entity
            return matched_entity
    return None


def _event_axis_at_least(
    entity: Mapping[str, Any],
    field_names: tuple[str, ...],
) -> bool:
    """Return whether any frozen event axis reaches the projection threshold."""

    return any(
        entity[field_name] >= OPERATIONAL_PRESSURE_THRESHOLD
        for field_name in field_names
    )


def _has_group_or_third_party_role(entity: Mapping[str, Any]) -> bool:
    """Return whether an event's retained role is group or third-party scoped."""

    return any(
        isinstance(role_ref, Mapping)
        and role_ref["entity_kind"] in {"group", "third_party"}
        for role_ref in entity["role_refs"]
    )


def _targets_relationship(entity: Mapping[str, Any], relationship_id: str) -> bool:
    """Return whether an entity keeps a relationship-scoped causal role."""

    return any(
        isinstance(role_ref, Mapping)
        and role_ref["entity_kind"] == "relationship"
        and role_ref["entity_id"] == relationship_id
        for role_ref in entity["role_refs"]
    )


def _relationship_evidence_freshness(
    evidence_refs: Sequence[Mapping[str, Any]],
    *,
    effective_at: str,
) -> str:
    """Return the latest relationship evidence age without exposing evidence."""

    occurred_at_values = [
        evidence_ref["occurred_at"]
        for evidence_ref in evidence_refs
        if isinstance(evidence_ref, Mapping)
        and isinstance(evidence_ref.get("occurred_at"), str)
    ]
    if not occurred_at_values:
        return "无证据"
    latest_occurred_at = max(occurred_at_values)
    freshness = project_duration(latest_occurred_at, effective_at)
    return freshness


def _normalized_summary(value: str) -> str:
    """Normalize and bound one relationship-scoped causal summary."""

    normalized = " ".join(value.split())
    if len(normalized) <= MAX_RELATIONSHIP_CAUSAL_SUMMARY_CHARS:
        return normalized
    head_length = MAX_RELATIONSHIP_CAUSAL_SUMMARY_CHARS // 2
    tail_length = MAX_RELATIONSHIP_CAUSAL_SUMMARY_CHARS - head_length - 1
    summary = (
        f"{normalized[:head_length]}…{normalized[-tail_length:]}"
    )
    return summary


def _elapsed_seconds(source_updated_at: str, effective_at: str) -> int:
    """Return non-negative elapsed UTC seconds for a read-time state view."""

    elapsed = _parse_utc(effective_at) - _parse_utc(source_updated_at)
    elapsed_seconds = max(0, int(elapsed.total_seconds()))
    return elapsed_seconds


def project_trend(previous: int, current: int) -> str:
    """Return direction using the fixed four-point change rule."""

    difference = current - previous
    if difference >= 4:
        return "上升"
    if difference <= -4:
        return "下降"
    return "稳定"


def project_state_for_prompt(
    state: Mapping[str, Any],
    *,
    character_constraints: Mapping[str, Any],
    character_identity_context: Mapping[str, Mapping[str, object]],
    relationship_context: Mapping[str, Any] | None = None,
    character_operational_context: Mapping[str, Any] | None = None,
    scene_context: Mapping[str, Any] | None = None,
    evidence: Sequence[Mapping[str, Any]] = (),
) -> PromptProjectionV2:
    """Project all prompt-visible state into semantic descriptors.

    Persistent ids and raw scalar values stay in ``handle_to_ref`` for
    deterministic mapping and are never included in ``payload``. Persisted
    standards remain available to raw state validation and storage, while
    the live semantic projection keeps their model-facing list empty.
    """

    handle_to_ref: dict[str, dict[str, str]] = {}
    goal_identity = deepcopy(dict(character_identity_context["goal_cognition"]))
    if isinstance(goal_identity.get("boundaries"), Mapping):
        goal_identity["boundaries"] = _project_boundary_profile(
            goal_identity["boundaries"],
        )
    payload: dict[str, Any] = {
        "goals": [],
        "threats": [],
        "events": [],
        "knowledge_gaps": [],
        "affect": [],
        "causal_candidates": [],
        "evidence": [],
        "character_constraints": _project_constraints(character_constraints),
        "character_identity": goal_identity,
    }
    evidence_identity_by_handle: dict[str, tuple[str, str]] = {}
    for row in evidence:
        evidence_handle = row.get("evidence_handle")
        evidence_ref = row.get("evidence_ref")
        if (
            isinstance(evidence_handle, str)
            and isinstance(evidence_ref, Mapping)
        ):
            evidence_identity_by_handle[evidence_handle] = (
                evidence_source_identity(evidence_ref)
            )

    matched_native_evidence: dict[tuple[str, str], str] = {}
    for field_name, prefix in (
        ("threats", "t"),
        ("active_events", "ev"),
        ("knowledge_gaps", "k"),
    ):
        entity_kind = _ENTITY_KIND_BY_FIELD[field_name]
        eligible_statuses = _ENTITY_PRESSURE_STATUSES[field_name]
        for evidence_handle, identity in evidence_identity_by_handle.items():
            matching_handles = [
                f"{prefix}{index}"
                for index, entity in enumerate(
                    state[field_name],
                    start=1,
                )
                if entity["status"] in eligible_statuses
                and any(
                    evidence_source_identity(evidence_ref) == identity
                    for evidence_ref in entity.get("evidence_refs", [])
                )
            ]
            if len(matching_handles) > 1:
                raise ValueError(
                    "ambiguous same-source native entity for "
                    f"{entity_kind}:{evidence_handle}"
                )
            if matching_handles:
                matched_native_evidence[(entity_kind, evidence_handle)] = (
                    matching_handles[0]
                )

    for field_name, prompt_name, prefix in (
        ("goals", "goals", "g"),
        ("threats", "threats", "t"),
        ("active_events", "events", "ev"),
        ("knowledge_gaps", "knowledge_gaps", "k"),
    ):
        for index, entity in enumerate(state[field_name], start=1):
            handle = f"{prefix}{index}"
            handle_to_ref[handle] = {
                "scope": state["state_scope"],
                "kind": _kind_for_field(field_name),
                "entity_id": entity["entity_id"],
            }
            payload[prompt_name].append(
                _project_entity(
                    handle,
                    entity,
                    state["updated_at"],
                    evidence_handles=[
                        evidence_handle
                        for (matched_kind, evidence_handle), matched_handle
                        in matched_native_evidence.items()
                        if matched_kind == _ENTITY_KIND_BY_FIELD[field_name]
                        and matched_handle == handle
                    ],
                )
            )
    relationship = state.get("relationship")
    if (
        isinstance(relationship_context, Mapping)
        and relationship_context.get("schema_version")
        == RELATIONSHIP_OPERATIONAL_CONTEXT_SCHEMA
    ):
        relationship = relationship_context
    if isinstance(relationship, Mapping):
        relationship_id = relationship.get("relationship_id")
        if not isinstance(relationship_id, str) or not relationship_id:
            raise ValueError("relationship projection requires relationship id")
        handle_to_ref["r1"] = {
            "scope": "user",
            "kind": "relationship",
            "entity_id": relationship_id,
        }
        if relationship.get("schema_version") == (
            RELATIONSHIP_OPERATIONAL_CONTEXT_SCHEMA
        ):
            payload["relationship"] = project_operational_relationship_context(
                relationship,
            )
        else:
            payload["relationship"] = _project_relationship(relationship)
    if isinstance(character_operational_context, Mapping):
        payload["character_operational_context"] = (
            _project_operational_character_context(
                character_operational_context,
            )
        )
    for activation in state["affect_activations"]:
        payload["affect"].append(_project_activation(activation, state))
    for index, drive_id in enumerate(state.get("drives", {}), start=1):
        handle_to_ref[f"d{index}"] = {
            "scope": state["state_scope"],
            "kind": "drive",
            "entity_id": drive_id,
        }
    if isinstance(state.get("meaning_state"), Mapping):
        handle_to_ref["m1"] = {
            "scope": state["state_scope"],
            "kind": "meaning",
            "entity_id": "meaning:character",
        }
    owner_user_id = state.get("owner_user_id")
    handle_to_ref["self"] = {
        "scope": "character",
        "kind": "meaning",
        "entity_id": "meaning:character",
    }
    if isinstance(owner_user_id, str) and owner_user_id:
        handle_to_ref["current_user"] = {
            "scope": "user",
            "kind": "relationship",
            "entity_id": f"relationship:user:{owner_user_id}",
        }
    participant_bindings = (
        scene_context.get("participant_bindings")
        if isinstance(scene_context, Mapping)
        else None
    )
    if participant_bindings is not None:
        if not isinstance(participant_bindings, list):
            raise ValueError("scene participant bindings must be a list")
        for binding in participant_bindings:
            if not isinstance(binding, Mapping):
                raise ValueError("scene participant binding must be a mapping")
            handle = binding.get("handle")
            if not isinstance(handle, str) or not handle.strip():
                raise ValueError("scene participant handle is invalid")
            if handle in handle_to_ref:
                raise ValueError("scene participant handle collides")
            handle_to_ref[handle] = {
                "scope": "episode",
                "kind": "third_party",
                "entity_id": f"scene:{handle}",
            }
    for index, row in enumerate(evidence, start=1):
        evidence_handle = row.get("evidence_handle")
        if not isinstance(evidence_handle, str):
            continue
        evidence_ref = row.get("evidence_ref")
        if isinstance(evidence_ref, Mapping):
            payload["evidence"].append({
                "handle": evidence_handle,
                "source_kind": evidence_ref.get("source_kind", "unknown"),
                "semantic_summary": row.get(
                    "semantic_text",
                    evidence_ref.get("semantic_summary", ""),
                ),
            })
        for kind, prefix, description in (
            ("event", "ce", "当前事件"),
            ("threat", "ct", "可能的当前威胁"),
            ("knowledge_gap", "ck", "可能的当前知识缺口"),
        ):
            if (kind, evidence_handle) in matched_native_evidence:
                continue
            handle = f"{prefix}{index}"
            handle_to_ref[handle] = {
                "scope": state["state_scope"],
                "kind": kind,
                "entity_id": f"candidate:{kind}:{evidence_handle}",
            }
            payload["causal_candidates"].append({
                "handle": handle,
                "candidate_kind": kind,
                "evidence_handle": evidence_handle,
                "description": description,
                "lifecycle": "候选，等待有依据的评估",
            })
    validate_prompt_projection(payload)
    identity_by_question: dict[str, dict[str, object]] = {}
    for question_kind, context in character_identity_context.items():
        projected_context = deepcopy(dict(context))
        if isinstance(projected_context.get("boundaries"), Mapping):
            projected_context["boundaries"] = _project_boundary_profile(
                projected_context["boundaries"],
            )
        identity_by_question[question_kind] = projected_context
    return PromptProjectionV2(
        payload=payload,
        handle_to_ref=handle_to_ref,
        identity_by_question=identity_by_question,
    )


def validate_prompt_projection(payload: Mapping[str, Any]) -> None:
    """Reject raw state fields except canonical semantic pressure kinds."""

    def visit(value: Any, path: tuple[object, ...] = ()) -> None:
        if isinstance(value, Mapping):
            for key, nested in value.items():
                is_semantic_pressure_kind = (
                    key == "kind"
                    and len(path) == 3
                    and path[0] == "character_operational_context"
                    and path[1] == "pressures"
                    and isinstance(path[2], int)
                )
                is_relational_decision_schema = (
                    key == "schema_version"
                    and path == ("relational_willingness",)
                )
                if (
                    key in RAW_STATE_KEYS
                    and not is_semantic_pressure_kind
                    and not is_relational_decision_schema
                ):
                    raise ValueError(
                        f"raw state key leaked into prompt: {key}"
                    )
                visit(nested, (*path, key))
        elif isinstance(value, list):
            for index, nested in enumerate(value):
                visit(nested, (*path, index))

    visit(payload)


def _project_entity(
    handle: str,
    entity: Mapping[str, Any],
    now: str,
    *,
    evidence_handles: Sequence[str],
) -> dict[str, Any]:
    """Project one causal entity without ids, timestamps, or raw axes."""

    result: dict[str, Any] = {
        "handle": handle,
        "description": entity["description"],
        "lifecycle": _lifecycle_label(entity["status"]),
        "salience": project_numeric_band(entity["salience"]),
        "duration": project_duration(entity["created_at"], now),
        "causal_roles": _project_roles(entity.get("role_refs", [])),
    }
    if evidence_handles:
        result["evidence_handles"] = list(evidence_handles)
    for field_name, signed in (
        ("importance", False),
        ("progress", False),
        ("obstruction", False),
        ("urgency", False),
        ("residual_pressure", False),
        ("harm", False),
        ("responsibility", False),
        ("uncertainty", False),
        ("relevance", False),
        ("trust", False),
        ("attachment", False),
        ("positive_regard", True),
    ):
        if field_name in entity:
            result[field_name] = project_numeric_band(
                entity[field_name],
                signed=signed,
            )
    return result


def _project_roles(value: Any) -> list[str]:
    """Project structured causal roles into semantic relationship phrases."""

    if not isinstance(value, list):
        return []
    role_labels = {
        "actor": "行动者",
        "experiencer": "体验者",
        "target": "对象",
        "object": "客体",
        "affected_goal": "受影响目标",
        "affected_relationship": "受影响关系",
    }
    labels: list[str] = []
    for role in value:
        if not isinstance(role, Mapping):
            continue
        role_name = role.get("role")
        if isinstance(role_name, str) and role_name.strip():
            role_label = role_labels.get(role_name.strip(), "语义")
            labels.append(f"{role_label}角色在因果上具有相关性")
    return labels


def _project_relationship(relationship: Mapping[str, Any]) -> dict[str, Any]:
    """Project relationship axes into semantic labels."""
    return {
        "handle": "r1",
        **project_relationship_context(relationship),
    }


def project_operational_relationship_context(
    relationship_context: Mapping[str, Any],
) -> dict[str, Any]:
    """Project the canonical relationship context without its durable id."""

    axes = relationship_context["axes"]
    if not isinstance(axes, Mapping):
        raise ValueError("relationship operational axes must be a mapping")
    projected_axes = {
        field_name: project_relationship_axis(field_name, value)
        for field_name, value in axes.items()
    }
    return {
        "handle": "r1",
        "axes": projected_axes,
        "causal_context": [
            dict(row)
            for row in relationship_context["causal_context"]
            if isinstance(row, Mapping)
        ],
        "affect": [
            dict(row)
            for row in relationship_context["affect"]
            if isinstance(row, Mapping)
        ],
        "relationship_freshness": relationship_context[
            "relationship_freshness"
        ],
        "evidence_freshness": relationship_context[
            "evidence_freshness"
        ],
    }


def _project_operational_character_context(
    context: Mapping[str, Any],
) -> dict[str, list[dict[str, str]]]:
    """Strip audit metadata before a selected posture reaches a model."""

    if context.get("schema_version") != CHARACTER_OPERATIONAL_CONTEXT_SCHEMA:
        raise ValueError("unsupported character operational context schema")
    affect = context.get("affect")
    pressures = context.get("pressures")
    if not isinstance(affect, list) or not isinstance(pressures, list):
        raise ValueError("character operational context rows are invalid")
    return {
        "affect": [
            {
                field_name: row[field_name]
                for field_name in (
                    "emotion_id",
                    "intensity",
                    "phase",
                    "trend",
                    "root_kind",
                    "cause_class",
                    "freshness",
                )
            }
            for row in affect
            if isinstance(row, Mapping)
        ],
        "pressures": [
            {
                field_name: row[field_name]
                for field_name in (
                    "kind",
                    "salience",
                    "lifecycle",
                    "cause_class",
                    "freshness",
                )
            }
            for row in pressures
            if isinstance(row, Mapping)
        ],
    }


def _project_boundary_profile(boundaries: Mapping[str, Any]) -> dict[str, str]:
    """Translate one identity boundary profile into compact semantics.

    Numeric boundary storage stays unchanged. The model receives only bounded
    Chinese descriptions of the character's pressure-response style.
    """

    projected: dict[str, str] = {
        "self_integrity": _boundary_float_semantic(
            boundaries["self_integrity"],
            "边界",
        ),
        "control_sensitivity": _boundary_float_semantic(
            boundaries["control_sensitivity"],
            "控制敏感",
        ),
        "compliance_strategy": _compliance_strategy_semantic(
            boundaries["compliance_strategy"],
        ),
        "relational_override": _boundary_float_semantic(
            boundaries["relational_override"],
            "关系覆盖",
        ),
        "control_intimacy_misread": _boundary_float_semantic(
            boundaries["control_intimacy_misread"],
            "亲密误读",
        ),
        "boundary_recovery": _boundary_recovery_semantic(
            boundaries["boundary_recovery"],
        ),
        "authority_skepticism": _boundary_float_semantic(
            boundaries["authority_skepticism"],
            "权威怀疑",
        ),
    }
    return projected


def _boundary_float_semantic(value: object, subject: str) -> str:
    """Translate one 0.0..1.0 boundary float into a bounded descriptor."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("boundary float projection requires a number")
    band = (
        "弱"
        if value < 0.2
        else "偏弱" if value < 0.4
        else "中" if value < 0.6
        else "偏强" if value < 0.8
        else "强"
    )
    return f"{subject}{band}"


def _compliance_strategy_semantic(value: object) -> str:
    """Describe the character's pressure-response style."""

    compliance_labels = {
        "resist": "压力下抵抗",
        "evade": "压力下回避",
        "comply": "压力下顺从",
    }
    label = compliance_labels.get(value)
    if label is None:
        raise ValueError("compliance strategy is invalid")
    return label


def _boundary_recovery_semantic(value: object) -> str:
    """Describe the stable recovery posture after a boundary event."""

    recovery_labels = {
        "rebound": "受损后快恢复",
        "delayed_rebound": "受损后慢恢复",
        "decay": "受损后逐渐弱化",
        "detach": "受损后倾向抽离",
    }
    label = recovery_labels.get(value)
    if label is None:
        raise ValueError("boundary recovery is invalid")
    return label


def _project_constraints(constraints: Mapping[str, Any]) -> dict[str, Any]:
    """Project character constraints separately from mutable user state."""

    drives = {
        drive_id: {
            "importance": project_numeric_band(row["importance"]),
            "pressure": project_numeric_band(row["pressure"]),
        }
        for drive_id, row in constraints["drives"].items()
    }
    meaning = {
        field_name: project_numeric_band(constraints["meaning_state"][field_name])
        for field_name in (
            "purpose_coherence",
            "agency",
            "identity_continuity",
            "salience",
        )
    }
    personality = {
        field_name: constraints["personality_judgment"][field_name]
        for field_name in ("logic", "defense", "quirks", "taboos")
    }
    return {
        "drives": drives,
        "standards": [],
        "meaning_state": meaning,
        "personality_judgment": personality,
    }


def _project_activation(
    activation: Mapping[str, Any],
    state: Mapping[str, Any],
) -> dict[str, str]:
    """Project activation lifecycle controls into natural language."""

    return {
        "emotion": activation["emotion_id"],
        "phase": (
            "原因仍然存在"
            if activation["cause_status"] == "active"
            else "该感受在问题解决后逐渐减弱"
        ),
        "intensity": project_numeric_band(activation["score"]),
        "trend": _trend_label(activation["trend"]),
        "cause_summary": _activation_cause_summary(activation, state),
    }


def _activation_cause_summary(
    activation: Mapping[str, Any],
    state: Mapping[str, Any],
) -> str:
    """Describe the actual primary cause without exposing its identifier."""

    root = activation.get("primary_root")
    if not isinstance(root, Mapping):
        return "有依据的原因仍在当前语境中"
    fields = {
        "goal": "goals",
        "threat": "threats",
        "event": "active_events",
        "knowledge_gap": "knowledge_gaps",
    }
    field_name = fields.get(root.get("kind"))
    if field_name is not None:
        for entity in state.get(field_name, []):
            if (
                isinstance(entity, Mapping)
                and entity.get("entity_id") == root.get("entity_id")
            ):
                description = entity.get("description")
                if isinstance(description, str) and description.strip():
                    return description[:500]
    if root.get("kind") == "relationship":
        return "当前关系带来激活情绪的社会压力"
    if root.get("kind") == "meaning":
        return "目标感和能动性持续偏低"
    return "有依据的原因仍在当前语境中"


def _kind_for_field(field_name: str) -> str:
    """Return the canonical singular entity kind."""

    return {
        "goals": "goal",
        "threats": "threat",
        "active_events": "event",
        "knowledge_gaps": "knowledge_gap",
    }[field_name]


def _lifecycle_label(status: str) -> str:
    """Translate deterministic status into a model-facing descriptor."""

    return {
        "pursuing": "进行中",
        "blocked": "受阻，等待解决",
        "satisfied": "已完成",
        "failed": "失败，等待恢复",
        "abandoned": "已放下",
        "active": "活跃且未解决",
        "resolved": "已解决",
        "replaced": "已被替代",
        "open": "开放且不确定",
        "reduced": "部分减弱但仍不确定",
    }.get(status, status)


def _trend_label(value: str) -> str:
    """Translate a persisted activation trend for model-facing context."""

    return {
        "rising": "上升",
        "stable": "稳定",
        "falling": "下降",
    }.get(value, value)


def _parse_utc(value: str) -> datetime:
    """Parse a required UTC Z timestamp."""

    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError("projection timestamp must end in Z")
    parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    if parsed.tzinfo is None:
        raise ValueError("projection timestamp must be timezone aware")
    return parsed.astimezone(timezone.utc)
