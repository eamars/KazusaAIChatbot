"""Deterministic aggregate prompt budgeting for semantic evidence."""

from __future__ import annotations

import json
from collections.abc import Mapping
from copy import deepcopy
from typing import Any


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
