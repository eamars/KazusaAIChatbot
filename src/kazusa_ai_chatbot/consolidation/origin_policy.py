"""Deterministic write policy for consolidation origin metadata."""

from __future__ import annotations

from typing import Literal, TypedDict

from kazusa_ai_chatbot.consolidation.origin import (
    ConsolidationOriginMetadata,
)

WritePolicyKey = Literal[
    "user_memory_units",
    "group_channel_style_image",
    "character_self_guidance",
    "cache_invalidation",
]


class WritePolicyDecision(TypedDict):
    """Allow/deny decision for one durable write or side effect."""

    allowed: bool
    reason: str


class ConsolidationWritePolicy(TypedDict):
    """Write-policy decisions for every consolidation persistence category."""

    user_memory_units: WritePolicyDecision
    group_channel_style_image: WritePolicyDecision
    character_self_guidance: WritePolicyDecision
    cache_invalidation: WritePolicyDecision


_SUPPORTED_SOURCE_PROFILES = {
    "user_message": frozenset({
        ("dialog", "system_event"),
        ("dialog", "image_observation", "system_event"),
        ("dialog", "audio_observation", "system_event"),
        (
            "dialog",
            "image_observation",
            "audio_observation",
            "system_event",
        ),
    }),
    "internal_thought": frozenset({
        ("internal_thought", "system_event"),
    }),
    "self_cognition": frozenset({
        ("self_cognition", "system_event"),
    }),
    "scheduled_tick": frozenset({
        ("scheduled_event", "system_event"),
    }),
    "tool_result": frozenset({
        ("tool_result", "system_event"),
    }),
}
_EXPECTED_OUTPUT_MODES = {
    "user_message": frozenset({"visible_reply"}),
    "internal_thought": frozenset({"preview"}),
    "self_cognition": frozenset({"preview"}),
    "scheduled_tick": frozenset({"preview"}),
    "tool_result": frozenset({"visible_reply"}),
}
_ALLOWED_REASON = "user_message_chat_input"
_INTERNAL_THOUGHT_ALLOWED_REASON = "internal_thought_same_path"
_SELF_COGNITION_ALLOWED_REASON = "self_cognition_same_path"
_SCHEDULED_TICK_ALLOWED_REASON = "scheduled_tick_source"
_TOOL_RESULT_ALLOWED_REASON = "tool_result_source"
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
    trigger_source = origin["trigger_source"]
    source_profiles = _SUPPORTED_SOURCE_PROFILES.get(trigger_source, frozenset())
    expected_output_modes = _EXPECTED_OUTPUT_MODES.get(
        trigger_source,
        frozenset(),
    )
    allowed = (
        tuple(origin["input_sources"]) in source_profiles
        and origin["output_mode"] in expected_output_modes
    )
    allowed_reasons = {
        "user_message": _ALLOWED_REASON,
        "internal_thought": _INTERNAL_THOUGHT_ALLOWED_REASON,
        "self_cognition": _SELF_COGNITION_ALLOWED_REASON,
        "scheduled_tick": _SCHEDULED_TICK_ALLOWED_REASON,
        "tool_result": _TOOL_RESULT_ALLOWED_REASON,
    }
    if allowed:
        reason = allowed_reasons[trigger_source]
    else:
        reason = _DENIED_REASON

    policy: ConsolidationWritePolicy = {
        "user_memory_units": {"allowed": allowed, "reason": reason},
        "group_channel_style_image": {"allowed": allowed, "reason": reason},
        "character_self_guidance": {"allowed": allowed, "reason": reason},
        "cache_invalidation": {"allowed": allowed, "reason": reason},
    }
    return policy
