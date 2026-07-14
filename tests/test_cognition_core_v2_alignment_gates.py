"""Checkpoint D anti-drift alignment and two-phase boundary tests."""

from __future__ import annotations

import pytest

from kazusa_ai_chatbot.cognition_core_v2.contracts import (
    CognitionContractError,
)
from kazusa_ai_chatbot.cognition_core_v2.semantic_source_planner import (
    QUESTION_KINDS,
    plan_semantic_questions,
)
from kazusa_ai_chatbot.cognition_core_v2.state_models import (
    build_acquaintance_user_state,
    build_character_production_state,
)
from kazusa_ai_chatbot.cognition_core_v2.contracts import (
    validate_cognition_core_input,
)


NOW = "2026-07-14T00:00:00Z"


def _constraints() -> dict[str, object]:
    """Return the character state as a separate read-only constraint object."""

    state = build_character_production_state(updated_at=NOW)
    return {
        "drives": state["drives"],
        "standards": state["standards"],
        "meaning_state": state["meaning_state"],
    }


def _evidence(source_kind: str = "episode") -> list[dict[str, object]]:
    """Build one typed semantic evidence row."""

    return [{
        "evidence_handle": "e1",
        "evidence_ref": {
            "source_kind": source_kind,
            "source_id": "episode-d",
            "occurred_at": NOW,
            "semantic_summary": "bounded episode evidence",
        },
        "semantic_text": "A bounded semantic observation is available.",
        "visible_to": [],
    }]


def test_episode_evidence_selects_each_family_once_with_unique_path_owners() -> None:
    """Planner coverage is provenance-driven and has exactly six families."""

    state = build_acquaintance_user_state(
        global_user_id="user-d",
        updated_at=NOW,
    )
    questions = plan_semantic_questions(
        _evidence(),
        state,
        _constraints(),
    )

    assert [question["question_kind"] for question in questions] == list(
        QUESTION_KINDS
    )
    assert len({path for question in questions for path in question["permitted_delta_paths"]}) == sum(
        len(question["permitted_delta_paths"]) for question in questions
    )


def test_scheduler_evidence_selects_only_goal_threat_outcome() -> None:
    """Source provenance, not user-language keywords, selects question families."""

    state = build_acquaintance_user_state(
        global_user_id="user-d",
        updated_at=NOW,
    )
    questions = plan_semantic_questions(
        _evidence("scheduler_event"),
        state,
        _constraints(),
    )

    assert [question["question_kind"] for question in questions] == [
        "goal_threat_outcome",
    ]


def test_v2_input_rejects_scope_mismatch_before_any_model_call() -> None:
    """Invalid scope ownership fails at the public boundary."""

    state = build_acquaintance_user_state(
        global_user_id="user-d",
        updated_at=NOW,
    )
    payload = {
        "schema_version": "cognition_core_input.v2",
        "episode": {},
        "state_scope": "character",
        "mutable_state": state,
        "character_constraints": _constraints(),
        "evidence": _evidence(),
        "direct_facts": [],
        "available_actions": [],
        "available_resolver_capabilities": [],
        "scene_context": {
            "channel_scope": "internal",
            "character_role": "character",
            "semantic_scene": "test",
            "semantic_temporal_context": "now",
        },
    }

    with pytest.raises(CognitionContractError, match="scope"):
        validate_cognition_core_input(payload)
