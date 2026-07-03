"""Tests for deterministic consolidation target planning."""

from __future__ import annotations

import inspect
from typing import Any

import pytest

from kazusa_ai_chatbot.consolidation import target as target_module
from kazusa_ai_chatbot.consolidation.target import (
    ConsolidationTargetValidationError,
    build_consolidation_target_plan,
    validate_write_intent,
)


def _base_state() -> dict[str, Any]:
    """Build a minimal consolidation state with a real user target."""

    state: dict[str, Any] = {
        "storage_timestamp_utc": "2026-05-20T08:00:00+00:00",
        "global_user_id": "global-user-1",
        "user_name": "Test User",
        "user_profile": {
            "global_user_id": "global-user-1",
            "affinity": 500,
            "display_name": "Test User",
        },
        "platform": "qq",
        "platform_channel_id": "private-1",
        "channel_type": "private",
        "platform_message_id": "message-1",
        "internal_monologue": "",
        "emotional_appraisal": "",
        "interaction_subtext": "",
        "final_dialog": [],
        "action_directives": {},
        "cognitive_episode": {
            "episode_id": "episode-1",
            "trigger_source": "user_message",
            "input_sources": ["dialog_text"],
            "output_mode": "visible_reply",
            "target_scope": {
                "platform": "qq",
                "platform_channel_id": "private-1",
                "channel_type": "private",
                "current_global_user_id": "global-user-1",
                "current_display_name": "Test User",
                "target_broadcast": False,
            },
        },
    }
    return state


def _targets_by_kind(plan: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Group target-plan rows by target kind for readable assertions."""

    targets_by_kind: dict[str, list[dict[str, Any]]] = {}
    for target in plan["targets"]:
        target_kind = target["target_kind"]
        targets_by_kind.setdefault(target_kind, []).append(target)
    return targets_by_kind


def test_target_planner_has_no_llm_dependency() -> None:
    """Target construction must remain deterministic."""

    source_text = inspect.getsource(target_module)

    forbidden_tokens = (
        "CONSOLIDATION_LLM",
        "get_llm",
        "SystemMessage",
        "HumanMessage",
        "ainvoke",
        "invoke",
        "prompt",
    )
    for token in forbidden_tokens:
        assert token not in source_text


def test_target_plan_builds_user_target_for_valid_chat() -> None:
    """A normal private user message should keep the existing user lane."""

    plan = build_consolidation_target_plan(_base_state())
    targets_by_kind = _targets_by_kind(plan)

    assert plan["origin_kind"] == "user_message"
    assert len(targets_by_kind["user"]) == 1
    user_target = targets_by_kind["user"][0]
    assert user_target["target_id"]["global_user_id"] == "global-user-1"
    assert "affinity" in user_target["write_lanes"]
    assert "user_memory_units" in user_target["write_lanes"]
    assert "group_channel" not in targets_by_kind


def test_group_chat_target_plan_includes_group_and_current_author() -> None:
    """A group chat turn should keep group state separate from user state."""

    state = _base_state()
    state["platform_channel_id"] = "group-1"
    state["channel_type"] = "group"
    state["cognitive_episode"]["target_scope"] = {
        "platform": "qq",
        "platform_channel_id": "group-1",
        "channel_type": "group",
        "current_global_user_id": "global-user-1",
        "current_display_name": "Test User",
        "target_broadcast": True,
    }

    plan = build_consolidation_target_plan(state)
    targets_by_kind = _targets_by_kind(plan)

    assert len(targets_by_kind["group_channel"]) == 1
    assert targets_by_kind["group_channel"][0]["target_id"] == {
        "platform": "qq",
        "platform_channel_id": "group-1",
    }
    assert len(targets_by_kind["user"]) == 1
    assert targets_by_kind["user"][0]["target_id"]["global_user_id"] == (
        "global-user-1"
    )


def test_group_review_target_plan_does_not_create_synthetic_user() -> None:
    """Group review without a real user should not fabricate user identity."""

    state = _base_state()
    state["global_user_id"] = "self_cognition"
    state["user_name"] = "self-cognition"
    state["platform_channel_id"] = "group-1"
    state["channel_type"] = "group"
    state["cognitive_episode"] = {
        "episode_id": "self-cognition:group-review",
        "trigger_source": "internal_thought",
        "input_sources": ["internal_monologue"],
        "output_mode": "preview",
        "target_scope": {
            "platform": "qq",
            "platform_channel_id": "group-1",
            "channel_type": "group",
            "current_global_user_id": "",
            "current_display_name": "group audience",
            "target_broadcast": True,
        },
    }

    plan = build_consolidation_target_plan(state)
    targets_by_kind = _targets_by_kind(plan)

    assert len(targets_by_kind["group_channel"]) == 1
    assert "user" not in targets_by_kind
    assert "character" not in targets_by_kind
    serialized_plan = repr(plan)
    assert "self_cognition" not in serialized_plan


def test_reflection_signal_target_plan_adds_internal_promotion_lane() -> None:
    """Reflection origins should expose only internal shared promotion target."""

    state = _base_state()
    state["global_user_id"] = ""
    state["user_profile"] = {}
    state["cognitive_episode"] = {
        "episode_id": "reflection-case-1",
        "trigger_source": "reflection_signal",
        "input_sources": ["reflection_artifact"],
        "output_mode": "visible_reply",
        "target_scope": {
            "platform": "qq",
            "platform_channel_id": "private-1",
            "channel_type": "private",
            "current_global_user_id": "",
            "current_display_name": "reflection",
            "target_broadcast": False,
        },
    }

    plan = build_consolidation_target_plan(state)
    targets_by_kind = _targets_by_kind(plan)
    internal_target = targets_by_kind["internal"][0]

    assert "user" not in targets_by_kind
    assert "character" not in targets_by_kind
    assert "shared_memory_promotion" in internal_target["write_lanes"]


def test_real_user_target_requires_validated_profile_shape() -> None:
    """A real user id with a malformed profile should fail at target planning."""

    state = _base_state()
    state["user_profile"] = {"display_name": "Test User"}

    with pytest.raises(ConsolidationTargetValidationError, match="affinity"):
        build_consolidation_target_plan(state)


def test_real_user_target_requires_matching_profile_identity() -> None:
    """A real user target must prove the same validated user-profile row."""

    state = _base_state()
    state["user_profile"] = {
        "global_user_id": "other-user",
        "affinity": 500,
    }

    with pytest.raises(ConsolidationTargetValidationError, match="mismatched"):
        build_consolidation_target_plan(state)


def test_validate_write_intent_rejects_forbidden_group_user_lane() -> None:
    """Group-channel targets must not reach user DB lanes."""

    state = _base_state()
    state["platform_channel_id"] = "group-1"
    state["channel_type"] = "group"
    state["cognitive_episode"]["target_scope"]["channel_type"] = "group"
    state["cognitive_episode"]["target_scope"]["platform_channel_id"] = "group-1"
    plan = build_consolidation_target_plan(state)
    group_target = _targets_by_kind(plan)["group_channel"][0]

    with pytest.raises(ConsolidationTargetValidationError, match="affinity"):
        validate_write_intent(
            {
                "target_alias": group_target["target_alias"],
                "write_lane": "affinity",
                "payload": {"delta": 1},
            },
            plan,
        )


def test_validate_write_intent_accepts_real_user_lane() -> None:
    """A real user target may use existing user lanes after validation."""

    plan = build_consolidation_target_plan(_base_state())
    user_target = _targets_by_kind(plan)["user"][0]
    intent = {
        "target_alias": user_target["target_alias"],
        "write_lane": "affinity",
        "payload": {"delta": 1},
    }

    validated_intent = validate_write_intent(intent, plan)

    assert validated_intent == intent
