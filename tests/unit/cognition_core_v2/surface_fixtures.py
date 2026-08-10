"""Shared typed surface fixtures for canonical owner tests."""

from __future__ import annotations

from copy import deepcopy

from tests.cognition_core_v2_test_helpers import (
    canonical_character_identity,
    canonical_cognition_output,
    canonical_episode,
)


def build_relational_decision(
    *,
    stance: str = "reject",
) -> dict[str, object]:
    """Build one complete V2 relational decision for surface tests."""

    relationship_state = {
        "accept": "established",
        "conditional_accept": "developing_or_uncertain",
        "reject": "unestablished",
    }[stance]
    return {
        "schema_version": "relational_willingness.v2",
        "applicability": "relationship_sensitive",
        "stance": stance,
        "current_user_relationship_state": relationship_state,
        "reason": '当前证据支持角色作出该判断',
        "evidence_handles": ["e1"],
    }


def build_surface_state(
    decision: dict[str, object],
) -> dict[str, object]:
    """Build one committed cognition packet for surface owner tests."""

    output = deepcopy(canonical_cognition_output())
    output["relational_willingness"] = deepcopy(decision)
    admitted_bid = output.get("admitted_bid")
    if isinstance(admitted_bid, dict):
        admitted_bid["relational_willingness"] = deepcopy(decision)
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
