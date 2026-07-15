"""Retired consolidation reviewer boundary.

The native V2 cognition state is committed by the cognition owner. Legacy
consolidation reviewers cannot author character or relationship state.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


_CHARACTER_STATE_REVIEW_PROMPT = (
    "The native cognition-state owner handles character-state replacement. "
    "This retired consolidation lane accepts no write candidate."
)

_RELATIONSHIP_PROFILE_REVIEW_PROMPT = (
    "The native cognition-state owner handles relationship replacement. "
    "This retired consolidation lane accepts no write candidate."
)


def retired_review_result(candidate: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Return the deterministic result for a retired reviewer boundary."""

    del candidate
    return {
        "decision": "reject",
        "reason": "retired_consolidation_lane",
    }
