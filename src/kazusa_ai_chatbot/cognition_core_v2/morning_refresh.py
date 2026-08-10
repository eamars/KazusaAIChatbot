"""Owned deterministic character morning-refresh transition."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal, TypedDict

from kazusa_ai_chatbot.cognition_core_v2.state_models import (
    CognitionStateError,
    validate_cognition_state,
)
from kazusa_ai_chatbot.cognition_core_v2.state_reducers import (
    apply_sleep_recovery,
)


CHARACTER_MORNING_REFRESH_RESULT_SCHEMA = (
    "character_morning_refresh_result.v2"
)

_ENTITY_FIELDS = ("goals", "threats", "active_events", "knowledge_gaps")


class CharacterMorningRefreshResultV2(TypedDict):
    """Typed deterministic morning-refresh result for the reflection audit."""

    schema_version: Literal["character_morning_refresh_result.v2"]
    recovered_state: dict[str, Any]
    applied_elapsed_sleep_seconds: int
    reduced_drive_count: int
    settled_entity_count: int
    retained_activation_count: int
    removed_activation_count: int


def run_character_morning_refresh(
    state: Mapping[str, Any],
    *,
    elapsed_sleep_seconds: int,
    updated_at: str,
) -> CharacterMorningRefreshResultV2:
    """Own the complete deterministic character morning-refresh transition.

    The entrypoint applies the character-scope sleep-recovery reducer and
    validates the recovered state before returning it, so callers never
    receive an unvalidated replacement. It knows nothing about local dates,
    run identifiers, or persistence; the reflection cycle owns those along
    with the audit row.

    Args:
        state: Character-scoped cognition state to recover.
        elapsed_sleep_seconds: Non-negative configured sleep-window length.
        updated_at: Replacement ``updated_at`` timestamp for the recovered
            state, formatted as UTC ISO-8601 text ending in ``Z``.

    Returns:
        A typed result carrying the validated recovered state, the applied
        elapsed seconds, and bounded deterministic transition counts.

    Raises:
        ValueError: If ``elapsed_sleep_seconds`` is not a non-negative
            integer.
        CognitionStateError: If the state is not character scoped or the
            recovered state fails validation.
    """

    if state["state_scope"] != "character":
        raise CognitionStateError(
            "morning refresh requires character cognition scope"
        )
    if (
        isinstance(elapsed_sleep_seconds, bool)
        or not isinstance(elapsed_sleep_seconds, int)
    ):
        raise ValueError("elapsed_sleep_seconds must be an integer")
    if elapsed_sleep_seconds < 0:
        raise ValueError("elapsed_sleep_seconds must be non-negative")

    recovered_state = apply_sleep_recovery(
        state,
        elapsed_sleep_seconds=elapsed_sleep_seconds,
        updated_at=updated_at,
    )
    validated_state = validate_cognition_state(recovered_state)
    reduced_drive_count = _count_reduced_drives(state, validated_state)
    settled_entity_count = _count_settled_entities(state, validated_state)
    retained_activation_count = len(validated_state["affect_activations"])
    removed_activation_count = _count_removed_activations(
        state,
        validated_state,
    )
    result: CharacterMorningRefreshResultV2 = {
        "schema_version": CHARACTER_MORNING_REFRESH_RESULT_SCHEMA,
        "recovered_state": validated_state,
        "applied_elapsed_sleep_seconds": elapsed_sleep_seconds,
        "reduced_drive_count": reduced_drive_count,
        "settled_entity_count": settled_entity_count,
        "retained_activation_count": retained_activation_count,
        "removed_activation_count": removed_activation_count,
    }
    return result


def _count_reduced_drives(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> int:
    """Count drives whose pressure was reduced by the recovery pass."""

    before_drives = before.get("drives")
    after_drives = after.get("drives")
    if not isinstance(before_drives, Mapping) or not isinstance(
        after_drives,
        Mapping,
    ):
        return 0

    reduced_count = 0
    for drive_id, before_drive in before_drives.items():
        after_drive = after_drives.get(drive_id)
        if (
            isinstance(before_drive, Mapping)
            and isinstance(after_drive, Mapping)
            and isinstance(before_drive.get("pressure"), int)
            and isinstance(after_drive.get("pressure"), int)
            and after_drive["pressure"] < before_drive["pressure"]
        ):
            reduced_count += 1
    return reduced_count


def _count_settled_entities(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> int:
    """Count entity rows whose transient salience or pressure changed."""

    settled_count = 0
    for field_name in _ENTITY_FIELDS:
        before_rows = before.get(field_name)
        after_rows = after.get(field_name)
        if not isinstance(before_rows, list) or not isinstance(
            after_rows,
            list,
        ):
            continue
        before_by_id = {
            row.get("entity_id"): row
            for row in before_rows
            if isinstance(row, Mapping)
        }
        after_by_id = {
            row.get("entity_id"): row
            for row in after_rows
            if isinstance(row, Mapping)
        }
        for entity_id, before_row in before_by_id.items():
            after_row = after_by_id.get(entity_id)
            if not isinstance(after_row, Mapping):
                continue
            if (
                before_row.get("salience") != after_row.get("salience")
                or before_row.get("residual_pressure")
                != after_row.get("residual_pressure")
            ):
                settled_count += 1
    return settled_count


def _count_removed_activations(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> int:
    """Count affect activations removed by the recovery pass."""

    before_activations = before.get("affect_activations")
    after_activations = after.get("affect_activations")
    if not isinstance(before_activations, list) or not isinstance(
        after_activations,
        list,
    ):
        return 0
    removed_count = max(0, len(before_activations) - len(after_activations))
    return removed_count
