"""Deterministic aggregate prompt budgeting for semantic evidence."""

from __future__ import annotations

import json
from typing import Any


class PromptBudgetError(ValueError):
    """Required prompt structure cannot fit after permitted reduction."""


def fit_evidence_texts_to_budget(
    payload: dict[str, Any],
    evidence_rows: list[dict[str, Any]],
    *,
    text_field: str,
    maximum_chars: int,
    minimum_text_chars: int,
) -> str:
    """Serialize or middle-truncate low-priority evidence until it fits.

    Args:
        payload: Complete prompt payload containing ``evidence_rows``.
        evidence_rows: Ordered, caller-owned evidence rows inside ``payload``.
        text_field: Semantic text field eligible for bounded truncation.
        maximum_chars: Maximum serialized payload length.
        minimum_text_chars: Minimum retained text length for each evidence row.

    Returns:
        The maximally retained deterministic JSON serialization within budget.

    Raises:
        PromptBudgetError: If required structure still exceeds the cap after
            every evidence text reaches its permitted floor.
    """

    if maximum_chars <= 0:
        raise ValueError("maximum prompt characters must be positive")
    if minimum_text_chars <= 0:
        raise ValueError("minimum evidence text characters must be positive")

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
