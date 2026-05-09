"""Deterministic write policy for consolidation origin metadata."""

from __future__ import annotations

from typing import Literal, TypedDict

from kazusa_ai_chatbot.nodes.persona_supervisor2_consolidator_origin import (
    ConsolidationOriginMetadata,
)

WritePolicyKey = Literal[
    "character_state",
    "relationship_insight",
    "user_memory_units",
    "task_dispatch",
    "affinity",
    "character_image",
    "cache_invalidation",
]


class WritePolicyDecision(TypedDict):
    """Allow/deny decision for one durable write or side effect."""

    allowed: bool
    reason: str


class ConsolidationWritePolicy(TypedDict):
    """Write-policy decisions for every consolidation persistence category."""

    character_state: WritePolicyDecision
    relationship_insight: WritePolicyDecision
    user_memory_units: WritePolicyDecision
    task_dispatch: WritePolicyDecision
    affinity: WritePolicyDecision
    character_image: WritePolicyDecision
    cache_invalidation: WritePolicyDecision


_TEXT_CHAT_INPUT_SOURCES = ["dialog_text"]
_TEXT_CHAT_OUTPUT_MODES = frozenset(("visible_reply", "think_only", "silent"))
_ALLOWED_REASON = "user_message_dialog_text"
_DENIED_REASON = "origin_not_enabled"


def build_consolidation_write_policy(
    *,
    origin: ConsolidationOriginMetadata,
) -> ConsolidationWritePolicy:
    """Build write/effect permissions from identifier-only origin metadata.

    Args:
        origin: Consolidation origin metadata projected from a cognitive
            episode.

    Returns:
        Policy decisions for every current durable write and side-effect
        category.
    """
    origin_is_allowed = (
        origin["trigger_source"] == "user_message"
        and origin["input_sources"] == _TEXT_CHAT_INPUT_SOURCES
        and origin["output_mode"] in _TEXT_CHAT_OUTPUT_MODES
    )
    if origin_is_allowed:
        allowed = True
        reason = _ALLOWED_REASON
    else:
        allowed = False
        reason = _DENIED_REASON

    policy: ConsolidationWritePolicy = {
        "character_state": {"allowed": allowed, "reason": reason},
        "relationship_insight": {"allowed": allowed, "reason": reason},
        "user_memory_units": {"allowed": allowed, "reason": reason},
        "task_dispatch": {"allowed": allowed, "reason": reason},
        "affinity": {"allowed": allowed, "reason": reason},
        "character_image": {"allowed": allowed, "reason": reason},
        "cache_invalidation": {"allowed": allowed, "reason": reason},
    }
    return policy
