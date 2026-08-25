"""Focused tests for the consolidator lane-router contract."""

from __future__ import annotations

import importlib
import json
from typing import Any

import pytest

from kazusa_ai_chatbot.consolidation.target import (
    build_consolidation_target_plan,
)
from kazusa_ai_chatbot.cognition_shared.state_models import (
    build_acquaintance_user_state,
)


EXPECTED_LANES = {
    "user_memory_units",
    "active_commitment",
    "character_identity_growth",
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
            "cognition_state": build_acquaintance_user_state(
                global_user_id="global-user-1",
                updated_at="2026-07-03T00:00:00Z",
            ),
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
    state.pop("cognitive_episode")
    state["origin_kind"] = "reflection_run"
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


def test_scheduled_tick_user_roster_excludes_reflection_style_lanes() -> None:
    """Scheduled user targets keep calendar lanes, not reflection owners."""

    module = _lane_router_module()
    state = _base_state()
    state["consolidation_origin"] = {
        "trigger_source": "scheduled_tick",
        "current_global_user_id": "global-user-1",
        "storage_timestamp_utc": "2026-07-03T00:00:00+00:00",
    }
    target_plan = build_consolidation_target_plan(state)
    roster_lanes = {
        entry["lane"] for entry in module.build_lane_roster(target_plan)
    }

    assert {"user_memory_units", "active_commitment"}.issubset(
        roster_lanes
    )
    assert "interaction_style_image" not in roster_lanes
    assert "shared_memory_promotion" not in roster_lanes


def test_lane_roster_includes_character_self_guidance_for_chat() -> None:
    """Normal user-message chats can route accepted character self-guidance."""

    module = _lane_router_module()

    target_plan = build_consolidation_target_plan(_base_state())
    roster = module.build_lane_roster(target_plan)
    roster_lanes = {entry["lane"] for entry in roster}

    assert "character_self_guidance" in roster_lanes
    assert "character_identity_growth" in roster_lanes
    assert "user_memory_units" in roster_lanes
    assert "active_commitment" in roster_lanes


def test_lane_router_distinguishes_global_user_group_and_transient_behavior_rules(
) -> None:
    """The coarse roster keeps global, scoped, and transient meanings separate."""

    module = _lane_router_module()
    target_plan = build_consolidation_target_plan(_base_state())
    roster = module.build_lane_roster(target_plan)
    roster_lanes = {entry["lane"] for entry in roster}

    assert {
        "user_memory_units",
        "active_commitment",
        "character_self_guidance",
    }.issubset(roster_lanes)
    user_memory_taxonomy = ("持久事实", "偏好", "模式", "变化", "里程碑")
    assert all(
        term in module._LANE_DESCRIPTIONS["user_memory_units"]
        for term in user_memory_taxonomy
    )
    assert all(term in module._ROUTER_PROMPT for term in user_memory_taxonomy)
    group_roster = module.build_lane_roster(
        _target_plan_for_group_without_user()
    )
    assert "interaction_style_image" in {
        entry["lane"] for entry in group_roster
    }
    assert set(module._ROUTER_TASK_KEYS) == {
        "lane",
        "reason",
        "source_keys",
    }
    assert set(module.CONSOLIDATION_LANE_NAMES) == EXPECTED_LANES

    captured_case_fragments = (
        "复读",
        "猫娘",
        "接龙",
        "收到",
        "阿然",
    )
    for fragment in captured_case_fragments:
        assert fragment not in module._ROUTER_PROMPT


def test_relationship_experience_can_route_character_owned_identity() -> None:
    """Close relationships may shape identity without globalizing details."""

    module = _lane_router_module()
    prompt = module._ROUTER_PROMPT

    assert "亲密关系经历也可能促成角色自己的持久变化" in prompt
    assert "关系对象、关系事实与私密细节仍归原有作用域" in prompt
    assert "角色自己的抽象变化" in prompt


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
        ],
        "character_operational_state_task": None,
    }

    validated = module.validate_lane_router_output(output, roster)

    assert validated == output


def test_identity_route_requires_one_closed_semantic_evidence_card() -> None:
    """Identity routing owns summaries while repositories own opaque roots."""

    module = _lane_router_module()
    target_plan = build_consolidation_target_plan(_base_state())
    roster = module.build_lane_roster(target_plan)
    output = {
        "lane_tasks": [{
            "lane": "character_identity_growth",
            "reason": "the character expressed a potentially durable change",
            "source_keys": [
                "assistant_final_dialog",
                "internal_thought",
            ],
            "identity_evidence": {
                "decontextualized_event": (
                    "The character reconsidered a recurring response pattern."
                ),
                "character_cognition_summary": (
                    "The character framed the change as her own judgment."
                ),
                "visible_self_expression_summary": (
                    "The character explicitly described a changed self-view."
                ),
            },
        }],
        "character_operational_state_task": None,
    }

    validated = module.validate_lane_router_output(output, roster)

    assert validated == output
    assert "episode_id" not in str(validated["lane_tasks"][0][
        "identity_evidence"
    ])


@pytest.mark.asyncio
async def test_router_prompt_excludes_repository_refs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The lane model receives semantic source views without raw identifiers."""

    module = _lane_router_module()
    captured_messages: list[Any] = []

    class _Response:
        content = (
            '{"lane_tasks":[],"character_operational_state_task":null}'
        )

    async def _invoke(messages, *, config):
        del config
        captured_messages.extend(messages)
        return _Response()

    monkeypatch.setattr(module._lane_router_llm, "ainvoke", _invoke)
    state = _base_state()
    state["consolidation_target_plan"] = build_consolidation_target_plan(state)
    await module.call_lane_router_llm(
        state,
        source_views=[{
            "source_key": "assistant_final_dialog",
            "source_kind": "assistant_final_dialog",
            "summary": "A bounded visible response.",
            "source_refs": [{
                "episode_id": "episode-private-root",
                "platform_message_id": "private-message",
            }],
        }],
        roster=module.build_lane_roster(
            state["consolidation_target_plan"]
        ),
    )

    human_prompt = str(captured_messages[1].content)
    assert "source_refs" not in human_prompt
    assert "episode-private-root" not in human_prompt
    assert "private-message" not in human_prompt


@pytest.mark.asyncio
async def test_router_prompt_projects_nonempty_source_role_without_remapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prompt projection preserves source ownership metadata verbatim."""

    module = _lane_router_module()
    captured_messages: list[Any] = []

    class _Response:
        content = (
            '{"lane_tasks":[],"character_operational_state_task":null}'
        )

    async def _invoke(messages, *, config):
        del config
        captured_messages.extend(messages)
        return _Response()

    monkeypatch.setattr(module._lane_router_llm, "ainvoke", _invoke)
    state = _base_state()
    state["consolidation_target_plan"] = build_consolidation_target_plan(state)
    await module.call_lane_router_llm(
        state,
        source_views=[
            {
                "source_key": "reflection_user_style_signal",
                "source_kind": "reflection_run",
                "summary": "A structured style signal.",
                "source_role": "user_style_signal",
                "source_refs": [{"reflection_run_id": "run-private-root"}],
            },
            {
                "source_key": "plain_source",
                "source_kind": "user_message",
                "summary": "A plain source.",
                "source_role": "",
            },
        ],
        roster=module.build_lane_roster(
            state["consolidation_target_plan"]
        ),
    )

    payload = json.loads(str(captured_messages[1].content))
    projected_views = payload["source_views"]
    assert projected_views[0]["source_role"] == "user_style_signal"
    assert "lane" not in projected_views[0]
    assert "source_role" not in projected_views[1]
    assert "source_refs" not in str(payload)
    assert "run-private-root" not in str(payload)


def test_router_prompt_declares_target_plan_and_roster_contracts() -> None:
    """Stable prompt identifiers distinguish eligibility from output words."""

    module = _lane_router_module()
    prompt = module._ROUTER_PROMPT

    assert "target_plan.write_lanes" in prompt
    assert "lane_tasks[].lane" in prompt
    assert "lane_roster[].lane" in prompt


def test_router_rejects_write_lane_token_outside_offered_roster() -> None:
    """Router output accepts only exact values offered by the roster."""

    module = _lane_router_module()
    roster = [{
        "lane": "interaction_style_image",
        "description": "style",
    }]
    output = {
        "lane_tasks": [{
            "lane": "user_style_image",
            "reason": "x",
            "source_keys": [],
        }],
        "character_operational_state_task": None,
    }

    with pytest.raises(ValueError, match="unknown consolidation lane"):
        module.validate_lane_router_output(output, roster)


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
        module.validate_lane_router_output(
            {
                "lane_tasks": [bad_task],
                "character_operational_state_task": None,
            },
            roster,
        )
