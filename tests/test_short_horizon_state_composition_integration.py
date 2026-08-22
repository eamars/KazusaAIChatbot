"""Focused connector tests for global state, relationship, and style wiring."""

from __future__ import annotations

from kazusa_ai_chatbot.cognition_shared.state_models import (
    build_acquaintance_user_state,
    build_character_production_state,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_cognition import (
    build_cognition_input_from_global_state,
)
from tests.cognition_test_helpers import (
    canonical_character_identity,
    canonical_episode,
    canonical_identity_context,
)


NOW = "2026-08-02T00:00:00Z"


def _state() -> dict[str, object]:
    """Build the adapter-neutral state required by the connector."""

    profile = canonical_character_identity(marker="composition")
    profile["global_user_id"] = "character-global"
    return {
        "character_profile": profile,
        "character_identity_context": canonical_identity_context(
            marker="composition",
        ),
        "character_cognition_state": build_character_production_state(
            updated_at=NOW,
        ),
        "cognitive_episode": canonical_episode(
            episode_id="episode-composition",
            current_global_user_id="user-composition",
            content="a grounded current episode",
        ),
        "global_user_id": "user-composition",
        "user_name": "Current User",
        "public_group_scene": "",
        "channel_type": "private",
        "trigger_source": "user_message",
        "rag_result": {"memory_evidence": []},
    }


def test_connector_projects_character_and_relationship_context_by_role() -> None:
    """Cognition receives projections, not a second persistence read."""

    payload = build_cognition_input_from_global_state(
        _state(),
        mutable_state=build_acquaintance_user_state(
            global_user_id="user-composition",
            updated_at=NOW,
        ),
    )

    assert payload["state_scope"] == "user"
    assert payload["mutable_state"]["state_scope"] == "user"
    assert payload["character_operational_context"]["consumer_role"] == (
        "appraisal branch"
    )
    assert payload["relationship_context"]["relationship_id"] == (
        "relationship:user:user-composition"
    )
    assert "character_cognition_state" not in repr(payload["evidence"])


def test_current_episode_and_progress_keep_topic_authority_over_global_state() -> None:
    """Global affect can alter stance but cannot create a new scene topic."""

    state = _state()
    state["public_group_scene"] = "current public topic"
    payload = build_cognition_input_from_global_state(
        state,
        mutable_state=build_acquaintance_user_state(
            global_user_id="user-composition",
            updated_at=NOW,
        ),
    )

    assert payload["scene_context"]["public_group_scene"] == (
        "current public topic"
    )
    assert payload["character_operational_context"]["consumer_role"] == (
        "appraisal branch"
    )
