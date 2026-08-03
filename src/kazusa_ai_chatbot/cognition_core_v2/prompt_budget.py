"""Deterministic aggregate prompt budgeting for semantic evidence."""

from __future__ import annotations

import json
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

        row[text_field] = _middle_truncate_text(
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
            row[text_field] = _middle_truncate_text(
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

        row[text_field] = _middle_truncate_text(
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


def _middle_truncate_text(value: str, maximum_chars: int) -> str:
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
