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
    affinity: WritePolicyDecision
    character_image: WritePolicyDecision
    cache_invalidation: WritePolicyDecision


_SUPPORTED_USER_MESSAGE_INPUT_SOURCE_PROFILES = {
    ("dialog_text",),
    ("dialog_text", "image_observation"),
    ("dialog_text", "audio_observation"),
    ("dialog_text", "image_observation", "audio_observation"),
}
_TEXT_CHAT_OUTPUT_MODES = frozenset(("visible_reply", "think_only", "silent"))
_ALLOWED_REASON = "user_message_chat_input"
_INTERNAL_THOUGHT_ALLOWED_REASON = "internal_thought_same_path"
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
    user_message_origin_is_allowed = (
        origin["trigger_source"] == "user_message"
        and (
            tuple(origin["input_sources"])
            in _SUPPORTED_USER_MESSAGE_INPUT_SOURCE_PROFILES
        )
        and origin["output_mode"] in _TEXT_CHAT_OUTPUT_MODES
    )
    internal_thought_origin_is_allowed = (
        origin["trigger_source"] == "internal_thought"
        and tuple(origin["input_sources"]) == ("internal_monologue",)
        and origin["output_mode"] == "preview"
    )
    if user_message_origin_is_allowed:
        allowed = True
        reason = _ALLOWED_REASON
    elif internal_thought_origin_is_allowed:
        allowed = True
        reason = _INTERNAL_THOUGHT_ALLOWED_REASON
    else:
        allowed = False
        reason = _DENIED_REASON

    policy: ConsolidationWritePolicy = {
        "character_state": {"allowed": allowed, "reason": reason},
        "relationship_insight": {"allowed": allowed, "reason": reason},
        "user_memory_units": {"allowed": allowed, "reason": reason},
        "affinity": {"allowed": allowed, "reason": reason},
        "character_image": {"allowed": allowed, "reason": reason},
        "cache_invalidation": {"allowed": allowed, "reason": reason},
    }
    return policy
