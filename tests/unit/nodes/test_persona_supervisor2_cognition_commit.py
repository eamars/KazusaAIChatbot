"""Focused commit and continuation-lineage checks for canonical cognition."""

from __future__ import annotations

import pytest

from kazusa_ai_chatbot.cognition_shared.state_models import (
    build_acquaintance_user_state,
    build_character_production_state,
)
from kazusa_ai_chatbot.nodes import persona_supervisor2_cognition as cognition_node
from tests.unit.cognition_core_v3.test_handleless_contract import _input


@pytest.mark.asyncio
async def test_resolver_recurrence_commits_against_original_user_base(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _input()
    original = payload["mutable_state"]
    replacement = dict(original)
    replacement["relationship"] = dict(original["relationship"])
    replacement["relationship"]["trust"] = 10
    captured: dict[str, object] = {}

    async def replace_user(owner: str, expected: dict, next_state: dict) -> bool:
        captured["owner"] = owner
        captured["expected"] = expected
        captured["replacement"] = next_state
        return True

    monkeypatch.setattr(
        cognition_node,
        "compare_and_replace_user_cognition_state",
        replace_user,
    )
    output = {
        "schema_version": "cognition_output.v3",
        "state_projection": {
            "state_scope": "user",
            "owner_key": "user-1",
            "expected_previous_state": replacement,
            "original_persisted_state": original,
            "replacement_state": replacement,
        },
    }
    await cognition_node.commit_cognition_output(output)
    assert captured["owner"] == "user-1"
    assert captured["expected"] == original
    assert captured["replacement"] == replacement


def test_current_continuation_uses_exact_private_goal_ref() -> None:
    payload = _input()
    replacement = build_acquaintance_user_state(
        global_user_id="user-1",
        updated_at=payload["mutable_state"]["updated_at"],
    )
    output = {
        "state_projection": {
            "continuation_goal_ref": {
                "scope": "user",
                "kind": "goal",
                "entity_id": "goal:ordinary_response:user:current",
            },
        },
    }
    continuation = cognition_node._canonical_goal_continuation_ref(
        output,
        {"cognitive_episode": payload["episode"]},
        replacement,
    )
    assert continuation["goal_ref"]["entity_id"] == (
        "goal:ordinary_response:user:current"
    )


def test_global_projection_preserves_exact_private_monologue() -> None:
    """Global residue state receives G subjectivity rather than goal analysis."""

    payload = _input()
    caller_state = {
        **payload,
        "global_user_id": "user-1",
        "cognitive_episode": payload["episode"],
    }
    output = {
        "schema_version": "cognition_output.v3",
        "active_character_goal": {
            "goal_kind": "clarify",
            "intent": "understand the unfamiliar object",
            "reason": "the observation does not identify the object",
            "cause_summary": "an unfamiliar object appeared",
        },
        "private_monologue": (
            "I am curious, but I do not want to pretend I recognize it."
        ),
        "response_plan": {
            "goal_resolution": "answerable_now",
            "response_goal": "ask what the unfamiliar object is",
            "action_requests": [],
            "resolver_requests": [],
            "epistemic_boundary": (
                "Describe its visible form and leave its identity unknown."
            ),
        },
        "state_projection": {
            "replacement_state": payload["mutable_state"],
        },
        "affect_projection": [],
        "relationship_projection": {},
        "relational_willingness": {},
        "cause_provenance": [],
    }

    projected = cognition_node._project_output_to_global_state(
        output,
        caller_state,
        available_actions=payload["available_actions"],
        available_resolver_capabilities=(
            payload["available_resolver_capabilities"]
        ),
    )

    assert projected["internal_monologue"] == output["private_monologue"]
    assert projected["internal_monologue"] != (
        output["active_character_goal"]["reason"]
    )


def test_global_projection_supplies_consolidation_interaction_subtext() -> None:
    """Project the goal reason and private monologue into separate fields."""

    payload = _input()
    caller_state = {
        **payload,
        "global_user_id": "user-1",
        "cognitive_episode": payload["episode"],
    }
    reason = "the compass reacts to a direction I cannot see"
    private_monologue = "I should ask what makes the needle move before guessing."
    output = {
        "schema_version": "cognition_output.v3",
        "active_character_goal": {
            "goal_kind": "clarify",
            "intent": "understand the unfamiliar compass",
            "reason": reason,
            "cause_summary": "the compass needle moved without an obvious cause",
        },
        "private_monologue": private_monologue,
        "response_plan": {
            "goal_resolution": "answerable_now",
            "response_goal": "ask what makes the compass needle move",
            "action_requests": [],
            "resolver_requests": [],
            "epistemic_boundary": (
                "Describe the movement and leave its cause unknown."
            ),
        },
        "state_projection": {
            "replacement_state": payload["mutable_state"],
        },
        "affect_projection": [],
        "relationship_projection": {},
        "relational_willingness": {},
        "cause_provenance": [],
    }

    projected = cognition_node._project_output_to_global_state(
        output,
        caller_state,
        available_actions=payload["available_actions"],
        available_resolver_capabilities=(
            payload["available_resolver_capabilities"]
        ),
    )

    assert projected["interaction_subtext"] == reason
    assert projected["internal_monologue"] == private_monologue
    assert projected["interaction_subtext"] != projected["internal_monologue"]


@pytest.mark.asyncio
async def test_persona_character_commit_reads_canonical_state_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = build_character_production_state(updated_at="2026-01-01T00:00:00Z")
    replacement = build_character_production_state(updated_at="2026-01-01T00:00:00.000001Z")
    captured: dict[str, object] = {}

    async def replace_character(*, expected_updated_at: str, replacement: dict) -> bool:
        captured["expected"] = expected_updated_at
        captured["replacement"] = replacement
        return True

    monkeypatch.setattr(
        cognition_node,
        "compare_and_replace_character_cognition_state",
        replace_character,
    )
    output = {
        "schema_version": "cognition_output.v3",
        "state_projection": {
            "state_scope": "character",
            "owner_key": "character",
            "expected_previous_state": replacement,
            "original_persisted_state": original,
            "replacement_state": replacement,
        },
    }
    await cognition_node.commit_cognition_output(
        output,
        expected_character_updated_at=original["updated_at"],
    )
    assert captured["expected"] == original["updated_at"]
    assert captured["replacement"] == replacement
