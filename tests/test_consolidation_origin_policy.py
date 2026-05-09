"""Tests for deterministic consolidation origin write policy."""

from __future__ import annotations

from kazusa_ai_chatbot.nodes.persona_supervisor2_consolidator_origin import (
    ConsolidationOriginMetadata,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_consolidator_origin_policy import (
    ConsolidationWritePolicy,
    WritePolicyKey,
    build_consolidation_write_policy,
)


_POLICY_KEYS: tuple[WritePolicyKey, ...] = (
    "character_state",
    "relationship_insight",
    "user_memory_units",
    "task_dispatch",
    "affinity",
    "character_image",
    "cache_invalidation",
)


def _origin(
    *,
    trigger_source: str = "user_message",
    input_sources: list[str] | None = None,
    output_mode: str = "visible_reply",
) -> ConsolidationOriginMetadata:
    """Build identifier-only origin metadata for policy tests.

    Args:
        trigger_source: Trigger source stored on the origin metadata.
        input_sources: Input-source list stored on the origin metadata.
        output_mode: Output mode stored on the origin metadata.

    Returns:
        Consolidation origin metadata with no percept or prompt content.
    """
    active_input_sources = list(input_sources or ["dialog_text"])
    origin: ConsolidationOriginMetadata = {
        "episode_id": "episode-1",
        "trigger_source": trigger_source,
        "input_sources": active_input_sources,
        "output_mode": output_mode,
        "timestamp": "2026-05-10T09:00:00+12:00",
        "platform": "qq",
        "platform_channel_id": "channel-1",
        "channel_type": "group",
        "platform_message_id": "message-1",
        "active_turn_platform_message_ids": ["message-1"],
        "active_turn_conversation_row_ids": ["conversation-row-1"],
        "current_platform_user_id": "platform-user-1",
        "current_global_user_id": "global-user-1",
        "current_display_name": "Test User",
    }
    return origin


def _assert_all_decisions(
    policy: ConsolidationWritePolicy,
    *,
    allowed: bool,
    reason: str,
) -> None:
    """Assert every write/effect category returns the same policy decision.

    Args:
        policy: Policy returned by the policy builder.
        allowed: Expected allow flag for every category.
        reason: Expected reason for every category.
    """
    assert tuple(policy) == _POLICY_KEYS
    for key in _POLICY_KEYS:
        assert policy[key] == {"allowed": allowed, "reason": reason}


def test_user_message_dialog_text_allows_all_write_categories() -> None:
    """Current text-chat origins should preserve all existing writer effects."""
    policy = build_consolidation_write_policy(origin=_origin())

    _assert_all_decisions(
        policy,
        allowed=True,
        reason="user_message_dialog_text",
    )


def test_reflection_signal_origin_denies_all_write_categories() -> None:
    """Reflection origins should be denied until a later plan enables writes."""
    policy = build_consolidation_write_policy(
        origin=_origin(
            trigger_source="reflection_signal",
            input_sources=["reflection_artifact"],
            output_mode="think_only",
        ),
    )

    _assert_all_decisions(policy, allowed=False, reason="origin_not_enabled")


def test_internal_thought_origin_denies_all_write_categories() -> None:
    """Internal-thought origins should be denied by the shared write policy."""
    policy = build_consolidation_write_policy(
        origin=_origin(
            trigger_source="internal_thought",
            input_sources=["internal_monologue"],
            output_mode="think_only",
        ),
    )

    _assert_all_decisions(policy, allowed=False, reason="origin_not_enabled")


def test_non_dialog_text_input_denies_all_write_categories() -> None:
    """User-message origins with unsupported inputs should be denied."""
    policy = build_consolidation_write_policy(
        origin=_origin(input_sources=["dialog_text", "image_observation"]),
    )

    _assert_all_decisions(policy, allowed=False, reason="origin_not_enabled")


def test_preview_output_denies_all_write_categories() -> None:
    """Preview outputs should not create durable consolidation effects."""
    policy = build_consolidation_write_policy(
        origin=_origin(output_mode="preview"),
    )

    _assert_all_decisions(policy, allowed=False, reason="origin_not_enabled")
