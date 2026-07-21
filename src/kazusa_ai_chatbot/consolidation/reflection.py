"""Retired consolidation reviewer boundary.

The native V2 cognition state is committed by the cognition owner. Legacy
consolidation reviewers cannot author character or relationship state.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


_CHARACTER_STATE_REVIEW_PROMPT = (
    "原生 cognition state owner 负责替换角色状态。"
    "这个已经停用的 consolidation lane 不接受写入候选。"
)

_RELATIONSHIP_PROFILE_REVIEW_PROMPT = (
    "原生 cognition state owner 负责替换关系状态。"
    "这个已经停用的 consolidation lane 不接受写入候选。"
)


def retired_review_result(candidate: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Return the deterministic result for a retired reviewer boundary."""

    del candidate
    return {
        "decision": "reject",
        "reason": "retired_consolidation_lane",
    }
