"""Shared typed surface fixtures for active owner tests."""

from __future__ import annotations

from copy import deepcopy

from tests.cognition_test_helpers import (
    canonical_character_identity,
    canonical_episode,
)


def build_relational_decision(
    *,
    stance: str = "reject",
) -> dict[str, object]:
    """Build one complete relational decision for surface tests."""

    return {
        "applicable": True,
        "stance": stance,
        "reason": "当前证据支持角色作出该判断",
        "cause_summary": "当前证据支持该关系立场",
    }


def build_surface_state(
    decision: dict[str, object],
) -> dict[str, object]:
    """Build one committed cognition packet for surface owner tests."""

    output = {
        "schema_version": "cognition_output.v3",
        "active_character_goal": {
            "goal_kind": "ordinary_response",
            "intent": "acknowledge the grounded episode",
            "reason": "the current episode establishes the selected route",
            "cause_summary": "grounded current episode",
        },
        "response_plan": {
            "response_goal": "acknowledge the grounded episode",
            "goal_resolution": "answerable_now",
            "action_requests": [],
            "resolver_requests": [],
        },
        "affect_projection": [],
        "relationship_projection": None,
        "relational_willingness": deepcopy(decision),
    }
    return {
        "storage_timestamp_utc": "2026-07-14T00:00:00Z",
        "user_input": "current turn input",
        "cognitive_episode": canonical_episode(
            episode_id="relational-surface-episode",
            content="current turn input",
        ),
        "cognition_core_output": output,
        "pre_surface_action_results": [],
        "character_profile": canonical_character_identity(marker="surface"),
    }
