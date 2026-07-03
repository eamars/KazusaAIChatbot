"""Focused tests for the consolidator lane-router contract."""

from __future__ import annotations

import importlib
from typing import Any

import pytest

from kazusa_ai_chatbot.consolidation.target import (
    build_consolidation_target_plan,
)


EXPECTED_LANES = {
    "character_state",
    "relationship_profile",
    "user_memory_units",
    "active_commitment",
    "character_self_guidance",
    "interaction_style_image",
    "shared_memory_promotion",
}


def _lane_router_module() -> Any:
    """Import the planned lane-router module with a clear failure."""

    try:
        module = importlib.import_module(
            "kazusa_ai_chatbot.consolidation.lane_router"
        )
    except ModuleNotFoundError as exc:
        pytest.fail(
            "Missing consolidation.lane_router module required by the "
            "lane-router bigbang plan."
        )
        raise exc
    return module


def _base_state() -> dict[str, Any]:
    """Build a normal private-chat state for target planning."""

    state: dict[str, Any] = {
        "global_user_id": "global-user-1",
        "user_name": "Test User",
        "user_profile": {
            "global_user_id": "global-user-1",
            "affinity": 500,
        },
        "platform": "qq",
        "platform_channel_id": "private-1",
        "channel_type": "private",
        "character_profile": {"name": "Kazusa"},
        "cognitive_episode": {
            "episode_id": "episode-router-contract-1",
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


def _target_plan_for_group_without_user() -> dict[str, Any]:
    """Build a group-channel target plan without a real user target."""

    state = _base_state()
    state["global_user_id"] = "self_cognition"
    state["user_profile"] = {}
    state["platform_channel_id"] = "group-1"
    state["channel_type"] = "group"
    state["cognitive_episode"] = {
        "episode_id": "episode-router-group-1",
        "trigger_source": "internal_thought",
        "input_sources": ["internal_monologue"],
        "output_mode": "preview",
        "target_scope": {
            "platform": "qq",
            "platform_channel_id": "group-1",
            "channel_type": "group",
            "current_global_user_id": "",
            "current_display_name": "group",
            "target_broadcast": True,
        },
    }
    return build_consolidation_target_plan(state)


def _reflection_target_plan_without_user() -> dict[str, Any]:
    """Build a reflection target plan without a real current user."""

    state = _base_state()
    state["global_user_id"] = ""
    state["user_profile"] = {}
    state["cognitive_episode"] = {
        "episode_id": "episode-router-reflection-1",
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
    return build_consolidation_target_plan(state)


def test_allowed_lane_names_are_exact_contract() -> None:
    """The router must expose only the seven planned consolidation lanes."""

    module = _lane_router_module()

    assert set(module.CONSOLIDATION_LANE_NAMES) == EXPECTED_LANES


def test_new_lane_prompts_avoid_fixture_examples_and_negative_accretion() -> None:
    """Runtime prompts should not encode the live-gating examples directly."""

    lane_router_module = _lane_router_module()
    self_guidance_module = importlib.import_module(
        "kazusa_ai_chatbot.consolidation.character_self_guidance"
    )
    reflection_module = importlib.import_module(
        "kazusa_ai_chatbot.consolidation.reflection"
    )
    prompts = {
        "router": lane_router_module._ROUTER_PROMPT,
        "self_guidance_specialist": self_guidance_module._SPECIALIST_PROMPT,
        "self_guidance_reviewer": self_guidance_module._REVIEW_PROMPT,
        "character_state_reviewer": (
            reflection_module._CHARACTER_STATE_REVIEW_PROMPT
        ),
        "relationship_profile_reviewer": (
            reflection_module._RELATIONSHIP_PROFILE_REVIEW_PROMPT
        ),
    }
    fixture_fragments = (
        '复读',
        '收到',
        '猫娘',
        '蓝星大陆',
        '七个王国',
        '小李',
        '低糖奶茶',
        '阿然',
        '奥克兰',
        '羽毛球',
        '早睡',
        '发报告',
        '接龙',
        "Aran",
        "Xiao Li",
        "milk tea",
        "roleplay tone",
        "seven kingdoms",
    )
    negative_markers = (
        "Do not",
        "Never",
        "Forbidden",
        "forbidden",
        "不要",
        "禁止",
    )

    for prompt_name, prompt in prompts.items():
        assert "# Boundary Examples" not in prompt, prompt_name
        for fragment in fixture_fragments:
            assert fragment not in prompt, (prompt_name, fragment)
        negative_count = sum(
            prompt.count(marker)
            for marker in negative_markers
        )
        assert negative_count <= 1, prompt_name


def test_lane_roster_prunes_impossible_user_lanes() -> None:
    """Target planning should prune impossible lanes before LLM routing."""

    module = _lane_router_module()

    target_plan = _target_plan_for_group_without_user()
    roster = module.build_lane_roster(target_plan)
    roster_lanes = {entry["lane"] for entry in roster}

    assert "interaction_style_image" in roster_lanes
    assert "user_memory_units" not in roster_lanes
    assert "active_commitment" not in roster_lanes
    assert "relationship_profile" not in roster_lanes


def test_reflection_roster_prunes_live_chat_character_and_user_lanes() -> None:
    """Reflection-origin routing should not consider normal chat lanes."""

    module = _lane_router_module()

    target_plan = _reflection_target_plan_without_user()
    roster = module.build_lane_roster(target_plan)
    roster_lanes = {entry["lane"] for entry in roster}

    assert roster_lanes == {"shared_memory_promotion"}


def test_lane_roster_includes_character_self_guidance_for_chat() -> None:
    """Normal user-message chats can route accepted character self-guidance."""

    module = _lane_router_module()

    target_plan = build_consolidation_target_plan(_base_state())
    roster = module.build_lane_roster(target_plan)
    roster_lanes = {entry["lane"] for entry in roster}

    assert "character_self_guidance" in roster_lanes
    assert "character_state" in roster_lanes
    assert "user_memory_units" in roster_lanes
    assert "active_commitment" in roster_lanes


def test_router_output_accepts_only_coarse_lane_tasks() -> None:
    """Router output should contain lane tasks, not memory text or DB ops."""

    module = _lane_router_module()
    target_plan = build_consolidation_target_plan(_base_state())
    roster = module.build_lane_roster(target_plan)
    output = {
        "lane_tasks": [
            {
                "lane": "user_memory_units",
                "reason": "user stated a durable personal fact",
                "source_keys": ["current_turn_user_message"],
            }
        ]
    }

    validated = module.validate_lane_router_output(output, roster)

    assert validated == output


@pytest.mark.parametrize(
    "bad_task",
    [
        {"lane": "not_a_lane", "reason": "x", "source_keys": []},
        {
            "lane": "user_memory_units",
            "reason": "x",
            "source_keys": [],
            "target_id": {"global_user_id": "global-user-1"},
        },
        {
            "lane": "user_memory_units",
            "reason": "x",
            "source_keys": [],
            "write_lane": "user_memory_units",
        },
        {
            "lane": "user_memory_units",
            "reason": "x",
            "source_keys": [],
            "payload": {"fact": "memory text belongs to a specialist"},
        },
        {
            "lane": "user_memory_units",
            "reason": "x",
            "source_keys": [],
            "fact": "memory text belongs to a specialist",
        },
    ],
)
def test_router_output_rejects_non_coarse_fields(
    bad_task: dict[str, Any],
) -> None:
    """Router validation should fail closed on DB or memory payload fields."""

    module = _lane_router_module()
    target_plan = build_consolidation_target_plan(_base_state())
    roster = module.build_lane_roster(target_plan)

    with pytest.raises(ValueError):
        module.validate_lane_router_output({"lane_tasks": [bad_task]}, roster)
