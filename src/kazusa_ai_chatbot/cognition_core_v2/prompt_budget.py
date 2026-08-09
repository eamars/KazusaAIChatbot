"""Deterministic aggregate prompt budgeting for semantic evidence."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
import logging
from typing import Any


logger = logging.getLogger(__name__)


class PromptBudgetError(ValueError):
    """Required prompt structure cannot fit after permitted reduction."""


def fit_evidence_texts_to_budget(
    payload: dict[str, Any] | list[dict[str, Any]],
    evidence_rows: list[dict[str, Any]] | None = None,
    *,
    text_field: str,
    maximum_chars: int | None = None,
    minimum_text_chars: int = 1,
    budget: int | None = None,
) -> str | list[dict[str, Any]]:
    """Serialize or middle-truncate low-priority evidence until it fits.

    Args:
        payload: Complete prompt payload containing ``evidence_rows``, or an
            ordered row list for standalone deterministic fitting.
        evidence_rows: Ordered, caller-owned evidence rows inside ``payload``.
        text_field: Semantic text field eligible for bounded truncation.
        maximum_chars: Maximum serialized payload length for aggregate mode.
        minimum_text_chars: Minimum retained text length for each evidence row.
        budget: Maximum serialized row-list length for standalone mode.

    Returns:
        The maximally retained JSON serialization in aggregate mode, or a
        copied fitted row list in standalone mode.

    Raises:
        PromptBudgetError: If required structure still exceeds the cap after
            every evidence text reaches its permitted floor.
    """

    standalone_mode = isinstance(payload, list)
    if standalone_mode:
        if evidence_rows is not None or maximum_chars is not None or budget is None:
            raise ValueError("standalone prompt fitting requires only budget")
        copied_rows = deepcopy(payload)
        standalone_payload = {"evidence": copied_rows}
        fitted_serialization = _fit_payload_evidence_texts(
            standalone_payload,
            copied_rows,
            text_field=text_field,
            maximum_chars=budget,
            minimum_text_chars=minimum_text_chars,
        )
        fitted_payload = json.loads(fitted_serialization)
        fitted_rows = fitted_payload["evidence"]
        return fitted_rows
    if evidence_rows is None or maximum_chars is None or budget is not None:
        raise ValueError(
            "aggregate prompt fitting requires payload, rows, and maximum_chars"
        )
    if maximum_chars <= 0:
        raise ValueError("maximum prompt characters must be positive")
    if minimum_text_chars <= 0:
        raise ValueError("minimum evidence text characters must be positive")
    if not isinstance(payload, dict):
        raise TypeError("aggregate prompt payload must be a mapping")
    fitted_payload = _fit_payload_evidence_texts(
        payload,
        evidence_rows,
        text_field=text_field,
        maximum_chars=maximum_chars,
        minimum_text_chars=minimum_text_chars,
    )
    return fitted_payload


def _fit_payload_evidence_texts(
    payload: dict[str, Any],
    evidence_rows: list[dict[str, Any]],
    *,
    text_field: str,
    maximum_chars: int,
    minimum_text_chars: int,
) -> str:
    """Serialize or middle-truncate ordered evidence within one fixed budget."""

    serialized_payload = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
    )
    if len(serialized_payload) <= maximum_chars:
        return serialized_payload

    for row in reversed(evidence_rows):
        semantic_text = row[text_field]
        if not isinstance(semantic_text, str):
            raise TypeError("prompt evidence semantic text must be a string")
        if len(semantic_text) <= minimum_text_chars:
            continue

        row[text_field] = middle_truncate_text(
            semantic_text,
            minimum_text_chars,
        )
        floor_serialization = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
        )
        if len(floor_serialization) > maximum_chars:
            continue

        lower_bound = minimum_text_chars
        upper_bound = len(semantic_text) - 1
        retained_chars = minimum_text_chars
        while lower_bound <= upper_bound:
            candidate_chars = (lower_bound + upper_bound) // 2
            row[text_field] = middle_truncate_text(
                semantic_text,
                candidate_chars,
            )
            candidate_serialization = json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
            )
            if len(candidate_serialization) <= maximum_chars:
                retained_chars = candidate_chars
                lower_bound = candidate_chars + 1
            else:
                upper_bound = candidate_chars - 1

        row[text_field] = middle_truncate_text(
            semantic_text,
            retained_chars,
        )
        fitted_payload = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
        )
        return fitted_payload

    raise PromptBudgetError(
        "required prompt structure exceeds the aggregate character cap"
    )


IDENTITY_TEXT_FLOORS: tuple[tuple[tuple[str, ...], int], ...] = (
    (("core", "backstory"), 600),
    (("core", "description"), 400),
    (("self_image", "self_concept"), 400),
    (("personality", "quirks"), 300),
    (("personality", "taboos"), 300),
    (("personality", "logic"), 300),
    (("personality", "tempo"), 300),
    (("personality", "defense"), 300),
)
MAX_REDUCED_GROWTH_EDGES = 2
MAX_REDUCED_STANDARD_DESCRIPTION_CHARS = 120
SCENE_TEXT_FLOORS: tuple[tuple[str, int], ...] = (
    ("public_group_scene", 400),
    ("conversation_continuity", 400),
    ("semantic_scene", 300),
    ("semantic_temporal_context", 200),
)


def middle_truncate_text(value: str, maximum_chars: int) -> str:
    """Retain both semantic ends while removing the middle of long text."""

    if len(value) <= maximum_chars:
        return value
    if maximum_chars == 1:
        return value[:1]

    marker = "..."
    if maximum_chars <= len(marker) + 1:
        head_chars = maximum_chars // 2
        tail_chars = maximum_chars - head_chars
        bounded_text = value[:head_chars] + value[-tail_chars:]
        return bounded_text

    retained_chars = maximum_chars - len(marker)
    head_chars = (retained_chars + 1) // 2
    tail_chars = retained_chars - head_chars
    bounded_text = (
        value[:head_chars]
        + marker
        + value[-tail_chars:]
    )
    return bounded_text


MAX_RELATIONSHIP_OPERATIONAL_CONTEXT_CHARS = 900
MAX_CHARACTER_OPERATIONAL_CONTEXT_CHARS = 1200
CHARACTER_OPERATIONAL_CONTEXT_DIGEST_CHARS = 64
MAX_RELATIONSHIP_CAUSAL_ROWS = 2
MAX_RELATIONSHIP_AFFECT_ROWS = 2
MAX_CONTEXT_AFFECT_ROWS = 3
MAX_CONTEXT_PRESSURE_ROWS = 4
MAX_SCENE_PARTICIPANT_BINDINGS = 8
MAX_RELATIONSHIP_CAUSAL_SUMMARY_CHARS = 160
MIN_RELATIONSHIP_CAUSAL_SUMMARY_CHARS = 80
RELATIONSHIP_CAUSAL_ENTITY_KINDS = frozenset({
    "goal",
    "threat",
    "event",
    "knowledge_gap",
})
RELATIONSHIP_AFFECT_PHASES = frozenset({"active", "fading"})
CHARACTER_OPERATIONAL_ROOT_KINDS = frozenset({
    "goal",
    "threat",
    "event",
    "knowledge_gap",
    "drive",
    "meaning",
})
OPERATIONAL_CAUSE_CLASSES = frozenset({
    "safety_pressure",
    "uncertainty_pressure",
    "meaning_pressure",
    "boundary_pressure",
    "repair_pressure",
    "loss_pressure",
    "competence_pressure",
    "connection_warmth",
    "relationship_strain",
    "goal_pressure",
    "general_activation",
})
CHARACTER_OPERATIONAL_CONSUMER_ROLES = frozenset({
    "settled_relevance",
    "appraisal branch",
    "goal",
    "surface",
})
RELATIONSHIP_AXIS_NAMES = frozenset({
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
})
_BOUNDED_TEXT_MAX_CHARS = 500


@dataclass(frozen=True)
class ContextFitResult:
    """Describe one deterministic operational-context fit without raw text."""

    payload: dict[str, Any]
    owner: str
    limit: int
    original_size: int
    final_size: int
    trimmed_fields: tuple[str, ...]
    dropped_rows: tuple[str, ...]
    fallback_level: int


def serialized_character_count(value: Mapping[str, Any]) -> int:
    """Measure the canonical compact-JSON representation in decoded chars."""

    serialized = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return len(serialized)


def canonical_digest(value: Mapping[str, Any]) -> str:
    """Hash one mapping with the stable contract JSON encoding."""

    serialized = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    return digest


def fit_relationship_operational_context(
    payload: Mapping[str, Any],
) -> ContextFitResult:
    """Fit one relationship operational context to its fixed packet cap.

    The packet is deep-copied before fitting, so the caller's mapping is
    never mutated. Free-text causal summaries are middle-truncated from the
    lowest-priority row upward, then optional causal rows and affect rows are
    dropped from the end of their stable order when the cap still requires
    it. Relationship identity, axes, freshness labels, and schema fields are
    preserved. A packet whose optional rows violate their row contract is
    returned unchanged so the strict validator remains the authority for
    shape and type failures.

    Returns:
        A ContextFitResult holding the fitted copy and bounded diagnostics.

    Raises:
        CognitionContextLimitError: If required fields alone still exceed the
            fixed packet cap after every permitted reduction.
    """

    fitted = deepcopy(dict(payload))
    limit = MAX_RELATIONSHIP_OPERATIONAL_CONTEXT_CHARS
    if not _fittable_relationship_rows(fitted):
        return _unchanged_fit_result(
            payload=fitted,
            owner="relationship",
            limit=limit,
        )
    original_size = serialized_character_count(fitted)
    trimmed_fields: list[str] = []
    dropped_rows: list[str] = []
    while serialized_character_count(fitted) > limit:
        causal_rows = fitted.get("causal_context")
        if isinstance(causal_rows, list):
            trimmed = False
            for row_index in range(len(causal_rows) - 1, -1, -1):
                row = causal_rows[row_index]
                summary = row["semantic_summary"]
                if len(summary) > MIN_RELATIONSHIP_CAUSAL_SUMMARY_CHARS:
                    row["semantic_summary"] = _fit_summary_to_budget(
                        fitted,
                        summary,
                        row,
                        limit=limit,
                    )
                    trimmed_fields.append(
                        f"causal_context[{row_index}].semantic_summary"
                    )
                    trimmed = True
                    break
            if trimmed:
                continue
            if causal_rows:
                causal_rows.pop()
                dropped_rows.append("causal_context")
                continue
        affect = fitted.get("affect")
        if isinstance(affect, list) and affect:
            affect.pop()
            dropped_rows.append("affect")
            continue
        _raise_context_limit("relationship")
    final_size = serialized_character_count(fitted)
    fallback_level = _fit_fallback_level(trimmed_fields, dropped_rows)
    _log_fit_result(
        owner="relationship",
        limit=limit,
        original_size=original_size,
        final_size=final_size,
        trimmed_fields=trimmed_fields,
        dropped_rows=dropped_rows,
        fallback_level=fallback_level,
    )
    result = ContextFitResult(
        payload=fitted,
        owner="relationship",
        limit=limit,
        original_size=original_size,
        final_size=final_size,
        trimmed_fields=tuple(trimmed_fields),
        dropped_rows=tuple(dropped_rows),
        fallback_level=fallback_level,
    )
    return result


def fit_character_operational_context(
    payload: Mapping[str, Any],
) -> ContextFitResult:
    """Fit one character operational context including its digest field.

    The packet is deep-copied before fitting and must carry a
    ``context_digest`` key; producers pass a fixed placeholder and consumers
    pass the persisted digest. The digest space is reserved while optional
    pressure rows and affect rows are dropped from the end of their stable
    order, then the digest is recomputed over the fitted body so the returned
    packet is self-consistent. A packet that violates the row contracts is
    returned unchanged so the strict validator remains the authority for
    shape and type failures.

    Returns:
        A ContextFitResult holding the fitted copy and bounded diagnostics.

    Raises:
        CognitionContextLimitError: If required fields alone still exceed the
            fixed packet cap after every permitted reduction.
    """

    fitted = deepcopy(dict(payload))
    limit = MAX_CHARACTER_OPERATIONAL_CONTEXT_CHARS
    if not _fittable_character_context(fitted):
        return _unchanged_fit_result(
            payload=fitted,
            owner="character",
            limit=limit,
        )
    original_size = serialized_character_count(fitted)
    body = {
        key: value
        for key, value in fitted.items()
        if key != "context_digest"
    }
    working = {
        **body,
        "context_digest": "0" * CHARACTER_OPERATIONAL_CONTEXT_DIGEST_CHARS,
    }
    dropped_rows: list[str] = []
    while serialized_character_count(working) > limit:
        pressures = working.get("pressures")
        if isinstance(pressures, list) and pressures:
            pressures.pop()
            dropped_rows.append("pressures")
            continue
        affect = working.get("affect")
        if isinstance(affect, list) and affect:
            affect.pop()
            dropped_rows.append("affect")
            continue
        _raise_context_limit("character")
    fitted_body = {
        key: value
        for key, value in working.items()
        if key != "context_digest"
    }
    incoming_digest = fitted["context_digest"]
    computed_digest = canonical_digest(fitted_body)
    if (
        dropped_rows
        or incoming_digest == "0" * CHARACTER_OPERATIONAL_CONTEXT_DIGEST_CHARS
    ):
        final_digest = computed_digest
    else:
        final_digest = incoming_digest
    final_payload = {
        **fitted_body,
        "context_digest": final_digest,
    }
    final_size = serialized_character_count(final_payload)
    if final_size > limit:
        _raise_context_limit("character")
    fallback_level = _fit_fallback_level((), dropped_rows)
    _log_fit_result(
        owner="character",
        limit=limit,
        original_size=original_size,
        final_size=final_size,
        trimmed_fields=(),
        dropped_rows=dropped_rows,
        fallback_level=fallback_level,
    )
    result = ContextFitResult(
        payload=final_payload,
        owner="character",
        limit=limit,
        original_size=original_size,
        final_size=final_size,
        trimmed_fields=(),
        dropped_rows=tuple(dropped_rows),
        fallback_level=fallback_level,
    )
    return result


def _fittable_relationship_rows(payload: Mapping[str, Any]) -> bool:
    """Return whether every optional relationship row matches its contract.

    A structurally invalid row must never be dropped by fitting, because that
    would convert a strict-contract failure into an accepted packet.
    """

    if set(payload) != {
        "schema_version",
        "relationship_id",
        "axes",
        "causal_context",
        "affect",
        "relationship_freshness",
        "evidence_freshness",
    }:
        return False
    if payload["schema_version"] != "relationship_operational_context.v1":
        return False
    relationship_id = payload["relationship_id"]
    if (
        not isinstance(relationship_id, str)
        or not relationship_id.strip()
        or len(relationship_id) > 200
    ):
        return False
    if not _is_relationship_axes_valid(payload.get("axes")):
        return False
    causal_context = payload.get("causal_context")
    affect = payload.get("affect")
    if not isinstance(causal_context, list) or not isinstance(affect, list):
        return False
    if (
        len(causal_context) > MAX_RELATIONSHIP_CAUSAL_ROWS
        or len(affect) > MAX_RELATIONSHIP_AFFECT_ROWS
    ):
        return False
    if not _is_bounded_text(payload.get("relationship_freshness")):
        return False
    if not _is_bounded_text(payload.get("evidence_freshness")):
        return False
    return all(
        _is_relationship_causal_row_valid(row)
        for row in causal_context
    ) and all(
        _is_relationship_affect_row_valid(row)
        for row in affect
    )


def _fittable_character_context(payload: Mapping[str, Any]) -> bool:
    """Return whether the character packet may be safely row-reduced.

    The packet must already satisfy the exact keys, row contracts, and row
    caps that fitting could otherwise mask; the strict validator remains
    authoritative for every remaining shape and type check.
    """

    if set(payload) != {
        "schema_version",
        "source_updated_at",
        "effective_at",
        "view_digest",
        "consumer_role",
        "affect",
        "pressures",
        "context_digest",
    }:
        return False
    digest = payload["context_digest"]
    if (
        not isinstance(digest, str)
        or not digest.strip()
        or len(digest) > CHARACTER_OPERATIONAL_CONTEXT_DIGEST_CHARS
    ):
        return False
    if not _is_utc_timestamp(payload.get("source_updated_at")):
        return False
    if not _is_utc_timestamp(payload.get("effective_at")):
        return False
    if not _is_digest(payload.get("view_digest")):
        return False
    consumer_role = payload["consumer_role"]
    if consumer_role not in CHARACTER_OPERATIONAL_CONSUMER_ROLES:
        return False
    affect = payload["affect"]
    pressures = payload["pressures"]
    if not isinstance(affect, list) or not isinstance(pressures, list):
        return False
    if (
        len(affect) > MAX_CONTEXT_AFFECT_ROWS
        or len(pressures) > MAX_CONTEXT_PRESSURE_ROWS
        or (consumer_role == "surface" and pressures)
    ):
        return False
    body = {
        key: value
        for key, value in payload.items()
        if key != "context_digest"
    }
    if (
        digest != "0" * CHARACTER_OPERATIONAL_CONTEXT_DIGEST_CHARS
        and digest != canonical_digest(body)
    ):
        return False
    return all(
        _is_character_affect_row_valid(row)
        for row in affect
    ) and all(
        _is_character_pressure_row_valid(row)
        for row in pressures
    )


def _is_relationship_causal_row_valid(value: Any) -> bool:
    """Return whether one causal row passes the strict row contract."""

    if not isinstance(value, Mapping):
        return False
    if set(value) != {
        "entity_kind",
        "semantic_summary",
        "salience",
        "lifecycle",
        "freshness",
    }:
        return False
    if value["entity_kind"] not in RELATIONSHIP_CAUSAL_ENTITY_KINDS:
        return False
    if (
        not isinstance(value["semantic_summary"], str)
        or not value["semantic_summary"].strip()
        or len(value["semantic_summary"]) > MAX_RELATIONSHIP_CAUSAL_SUMMARY_CHARS
    ):
        return False
    return all(
        _is_bounded_text(value[field_name])
        for field_name in ("salience", "lifecycle", "freshness")
    )


def _is_relationship_affect_row_valid(value: Any) -> bool:
    """Return whether one relationship affect row passes its contract."""

    if not isinstance(value, Mapping):
        return False
    if set(value) != {
        "emotion_id",
        "intensity",
        "phase",
        "trend",
        "freshness",
    }:
        return False
    if value["phase"] not in RELATIONSHIP_AFFECT_PHASES:
        return False
    return all(
        _is_bounded_text(value[field_name])
        for field_name in ("emotion_id", "intensity", "trend", "freshness")
    )


def _is_character_affect_row_valid(value: Any) -> bool:
    """Return whether one character affect row passes its contract."""

    if not isinstance(value, Mapping):
        return False
    if set(value) != {
        "emotion_id",
        "intensity",
        "phase",
        "trend",
        "root_kind",
        "cause_class",
        "freshness",
    }:
        return False
    if value["phase"] not in RELATIONSHIP_AFFECT_PHASES:
        return False
    if value["root_kind"] not in CHARACTER_OPERATIONAL_ROOT_KINDS:
        return False
    if value["cause_class"] not in OPERATIONAL_CAUSE_CLASSES:
        return False
    return all(
        _is_bounded_text(value[field_name])
        for field_name in ("emotion_id", "intensity", "trend", "freshness")
    )


def _is_character_pressure_row_valid(value: Any) -> bool:
    """Return whether one character pressure row passes its contract."""

    if not isinstance(value, Mapping):
        return False
    if set(value) != {
        "kind",
        "salience",
        "lifecycle",
        "cause_class",
        "freshness",
    }:
        return False
    if value["kind"] not in CHARACTER_OPERATIONAL_ROOT_KINDS:
        return False
    if value["cause_class"] not in OPERATIONAL_CAUSE_CLASSES:
        return False
    return all(
        _is_bounded_text(value[field_name])
        for field_name in ("salience", "lifecycle", "freshness")
    )


def _is_bounded_text(value: Any) -> bool:
    """Return whether one value passes the default bounded-text contract."""

    return (
        isinstance(value, str)
        and bool(value.strip())
        and len(value) <= _BOUNDED_TEXT_MAX_CHARS
    )


def _is_digest(value: Any) -> bool:
    """Return whether one digest field has its strict text bound."""

    return (
        isinstance(value, str)
        and bool(value.strip())
        and len(value) <= CHARACTER_OPERATIONAL_CONTEXT_DIGEST_CHARS
    )


def _is_utc_timestamp(value: Any) -> bool:
    """Return whether one timestamp is a parseable UTC-Z string."""

    if not isinstance(value, str) or not value.endswith("Z"):
        return False
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError:
        return False
    return parsed.tzinfo is not None


def _is_relationship_axes_valid(value: Any) -> bool:
    """Return whether relationship axes satisfy the strict numeric shape."""

    if not isinstance(value, Mapping) or set(value) != RELATIONSHIP_AXIS_NAMES:
        return False
    for axis_name, axis_value in value.items():
        if axis_name in {"positive_regard", "trust", "boundary_safety"}:
            if (
                isinstance(axis_value, bool)
                or not isinstance(axis_value, int)
                or not -100 <= axis_value <= 100
            ):
                return False
        elif (
            isinstance(axis_value, bool)
            or not isinstance(axis_value, int)
            or not 0 <= axis_value <= 100
        ):
            return False
    return True


def _unchanged_fit_result(
    *,
    payload: dict[str, Any],
    owner: str,
    limit: int,
) -> ContextFitResult:
    """Return an unchanged packet so strict validation owns malformed input."""

    try:
        size = serialized_character_count(payload)
    except (TypeError, ValueError):
        size = 0
    return ContextFitResult(
        payload=payload,
        owner=owner,
        limit=limit,
        original_size=size,
        final_size=size,
        trimmed_fields=(),
        dropped_rows=(),
        fallback_level=0,
    )


def _fit_fallback_level(
    trimmed_fields: list[str] | tuple[str, ...],
    dropped_rows: list[str] | tuple[str, ...],
) -> int:
    """Return the deepest reduction stage actually applied by one fit."""

    if dropped_rows:
        return 2
    if trimmed_fields:
        return 1
    return 0


def _fit_summary_to_budget(
    payload: dict[str, Any],
    original: str,
    row: dict[str, Any],
    *,
    limit: int,
) -> str:
    """Retain the longest valid middle-truncated summary that fits."""

    lower_bound = MIN_RELATIONSHIP_CAUSAL_SUMMARY_CHARS
    upper_bound = len(original) - 1
    best_length: int | None = None
    while lower_bound <= upper_bound:
        candidate_length = (lower_bound + upper_bound) // 2
        row["semantic_summary"] = middle_truncate_text(
            original,
            candidate_length,
        )
        if serialized_character_count(payload) <= limit:
            best_length = candidate_length
            lower_bound = candidate_length + 1
        else:
            upper_bound = candidate_length - 1
    if best_length is None:
        best_length = MIN_RELATIONSHIP_CAUSAL_SUMMARY_CHARS
    fitted_summary = middle_truncate_text(original, best_length)
    row["semantic_summary"] = fitted_summary
    return fitted_summary


def _log_fit_result(
    *,
    owner: str,
    limit: int,
    original_size: int,
    final_size: int,
    trimmed_fields: list[str] | tuple[str, ...],
    dropped_rows: list[str] | tuple[str, ...],
    fallback_level: int,
) -> None:
    """Emit one bounded size-recovery diagnostic when a fit reduced a packet."""

    if original_size == final_size:
        return
    logger.info(
        f"{owner} operational context fitted: {original_size}->{final_size} "
        f"chars (limit={limit}); trimmed={trimmed_fields!r}; "
        f"dropped={dropped_rows!r}; fallback_level={fallback_level}"
    )


def _raise_context_limit(owner: str) -> None:
    """Raise the typed post-fit invariant without a module-level import cycle."""

    from kazusa_ai_chatbot.cognition_core_v2.contracts import (
        CognitionContextLimitError,
    )

    raise CognitionContextLimitError(
        f"required {owner} operational context exceeds the fixed cap"
    )


def reduce_identity_projection(identity: dict[str, Any]) -> bool:
    """Apply the next bounded identity reduction step for one prompt packet.

    One call applies the first text floor whose field is still above it, or
    truncates growth edges once every text floor is reached. Fields that are
    permission- or role-relevant (name, gender, age, birthday, mbti, and every
    boundary value) are never reduced, and missing keys are skipped.

    Args:
        identity: Prompt-visible identity partition mutated in place.

    Returns:
        True when one bounded reduction step was applied, False at the floor.
    """

    for path, floor in IDENTITY_TEXT_FLOORS:
        owner = identity
        for key in path[:-1]:
            nested = owner.get(key)
            if not isinstance(nested, Mapping):
                owner = None
                break
            owner = nested
        if owner is None:
            continue
        leaf_key = path[-1]
        value = owner.get(leaf_key)
        if isinstance(value, str) and len(value) > floor:
            owner[leaf_key] = middle_truncate_text(value, floor)
            return True
    self_image = identity.get("self_image")
    if isinstance(self_image, Mapping):
        growth_edges = self_image.get("current_growth_edges")
        if (
            isinstance(growth_edges, list)
            and len(growth_edges) > MAX_REDUCED_GROWTH_EDGES
        ):
            self_image["current_growth_edges"] = (
                growth_edges[:MAX_REDUCED_GROWTH_EDGES]
            )
            return True
    return False


def reduce_constraints_projection(constraints: dict[str, Any]) -> bool:
    """Apply the next bounded character-constraint reduction step.

    One call middle-truncates every standard description above its floor.
    Drives, standards rows, and meaning state are never removed.

    Args:
        constraints: Prompt-visible character constraints mutated in place.

    Returns:
        True when at least one description was reduced, False at the floor.
    """

    standards = constraints.get("standards")
    if not isinstance(standards, list):
        return False
    reduced = False
    for standard in standards:
        if not isinstance(standard, Mapping):
            continue
        description = standard.get("description")
        if (
            isinstance(description, str)
            and len(description) > MAX_REDUCED_STANDARD_DESCRIPTION_CHARS
        ):
            standard["description"] = middle_truncate_text(
                description,
                MAX_REDUCED_STANDARD_DESCRIPTION_CHARS,
            )
            reduced = True
    return reduced


def reduce_scene_context_projection(scene_context: dict[str, Any]) -> bool:
    """Apply the next bounded scene-context reduction step.

    Args:
        scene_context: Prompt-visible scene context mutated in place.

    Returns:
        True when one scene text field was reduced, False at the floor.
    """

    for key, floor in SCENE_TEXT_FLOORS:
        value = scene_context.get(key)
        if isinstance(value, str) and len(value) > floor:
            scene_context[key] = middle_truncate_text(value, floor)
            return True
    return False
