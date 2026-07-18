"""Tests for deterministic consolidation origin write policy."""

from __future__ import annotations

from kazusa_ai_chatbot.consolidation.origin import (
    ConsolidationOriginMetadata,
)
from kazusa_ai_chatbot.consolidation.origin_policy import (
    ConsolidationWritePolicy,
    WritePolicyKey,
    build_consolidation_write_policy,
)


_POLICY_KEYS: tuple[WritePolicyKey, ...] = (
    "user_memory_units",
    "group_channel_style_image",
    "character_self_guidance",
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
    if input_sources is None:
        input_sources = ["dialog", "system_event"]
    active_input_sources = list(input_sources)
    origin: ConsolidationOriginMetadata = {
        "episode_id": "episode-1",
        "trigger_source": trigger_source,
        "input_sources": active_input_sources,
        "output_mode": output_mode,
        "storage_timestamp_utc": "2026-05-10T09:00:00+12:00",
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
    """Canonical text-chat origins preserve all existing writer effects."""
    policy = build_consolidation_write_policy(origin=_origin())

    _assert_all_decisions(
        policy,
        allowed=True,
        reason="user_message_chat_input",
    )


def test_user_message_multimodal_inputs_allow_all_write_categories() -> None:
    """Image/audio observations remain user-message chat inputs."""
    supported_inputs = [
        ["dialog", "image_observation", "system_event"],
        ["dialog", "audio_observation", "system_event"],
        [
            "dialog",
            "image_observation",
            "audio_observation",
            "system_event",
        ],
    ]

    for input_sources in supported_inputs:
        policy = build_consolidation_write_policy(
            origin=_origin(input_sources=input_sources),
        )

        _assert_all_decisions(
            policy,
            allowed=True,
            reason="user_message_chat_input",
        )


def test_scheduled_tick_wrong_profile_denies_all_write_categories() -> None:
    """A scheduled source with the wrong percept profile fails closed."""
    policy = build_consolidation_write_policy(
        origin=_origin(
            trigger_source="scheduled_tick",
            input_sources=["reflection_run", "system_event"],
            output_mode="think_only",
        ),
    )

    _assert_all_decisions(policy, allowed=False, reason="origin_not_enabled")


def test_internal_thought_origin_denies_all_write_categories() -> None:
    """Non-preview internal-thought origins should remain denied."""
    policy = build_consolidation_write_policy(
        origin=_origin(
            trigger_source="internal_thought",
            input_sources=["internal_thought", "system_event"],
            output_mode="think_only",
        ),
    )

    _assert_all_decisions(policy, allowed=False, reason="origin_not_enabled")


def test_internal_thought_preview_origin_allows_all_write_categories() -> None:
    """Self-cognition preview origins should use the same durable lanes."""
    policy = build_consolidation_write_policy(
        origin=_origin(
            trigger_source="internal_thought",
            input_sources=["internal_thought", "system_event"],
            output_mode="preview",
        ),
    )

    _assert_all_decisions(
        policy,
        allowed=True,
        reason="internal_thought_same_path",
    )


def test_internal_thought_preview_rejects_extra_input_sources() -> None:
    """Self-cognition consolidation rejects extra source profiles."""
    policy = build_consolidation_write_policy(
        origin=_origin(
            trigger_source="internal_thought",
            input_sources=[
                "internal_thought",
                "system_event",
                "rag_memory_evidence",
            ],
            output_mode="preview",
        ),
    )

    _assert_all_decisions(policy, allowed=False, reason="origin_not_enabled")


def test_unsupported_input_denies_all_write_categories() -> None:
    """User-message origins with unsupported inputs should be denied."""
    policy = build_consolidation_write_policy(
        origin=_origin(
            input_sources=["dialog", "system_event", "rag_memory_evidence"],
        ),
    )

    _assert_all_decisions(policy, allowed=False, reason="origin_not_enabled")


def test_private_canonical_sources_allow_all_write_categories() -> None:
    """Native private source episodes enter the same consolidation lanes."""

    cases = [
        ("self_cognition", ["self_cognition", "system_event"]),
        ("scheduled_tick", ["scheduled_event", "system_event"]),
        ("tool_result", ["tool_result", "system_event"]),
    ]

    for trigger_source, input_sources in cases:
        policy = build_consolidation_write_policy(
            origin=_origin(
                trigger_source=trigger_source,
                input_sources=input_sources,
                output_mode=(
                    "preview"
                    if trigger_source != "tool_result"
                    else "visible_reply"
                ),
            ),
        )
        reason = {
            "self_cognition": "self_cognition_same_path",
            "scheduled_tick": "scheduled_tick_source",
            "tool_result": "tool_result_source",
        }[trigger_source]
        _assert_all_decisions(policy, allowed=True, reason=reason)


def test_preview_output_denies_all_write_categories() -> None:
    """Preview outputs should not create durable consolidation effects."""
    policy = build_consolidation_write_policy(
        origin=_origin(output_mode="preview"),
    )

    _assert_all_decisions(policy, allowed=False, reason="origin_not_enabled")
