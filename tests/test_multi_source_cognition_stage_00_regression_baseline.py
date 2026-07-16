"""V2-only cognition regression baseline."""

import json
from pathlib import Path

import pytest

from kazusa_ai_chatbot.cognition_core_v2.contracts import (
    CognitionContractError,
    validate_cognition_core_input,
)
from kazusa_ai_chatbot.cognition_core_v2.state_models import (
    build_acquaintance_user_state,
    build_character_production_state,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_rag_projection import (
    project_known_facts,
)
from tests.cognition_core_v2_test_helpers import canonical_episode


NOW = "2026-07-14T00:00:00Z"
FIXTURE_PATH = (
    Path(__file__).parent
    / "fixtures"
    / "multi_source_cognition_stage_00_cases.json"
)


def _fixture() -> dict[str, object]:
    """Load the frozen Stage 00 response-path and RAG cases."""

    with FIXTURE_PATH.open("r", encoding="utf-8") as file_handle:
        return json.load(file_handle)


def _payload() -> dict[str, object]:
    character = build_character_production_state(updated_at=NOW)
    return {
        "schema_version": "cognition_core_input.v2",
        "episode": canonical_episode(
            episode_id="v2-regression-episode",
            current_global_user_id="v2-regression-user",
            content="quiet private greeting",
        ),
        "state_scope": "user",
        "mutable_state": build_acquaintance_user_state(
            global_user_id="v2-regression-user",
            updated_at=NOW,
        ),
        "character_constraints": {
            "drives": character["drives"],
            "standards": character["standards"],
            "meaning_state": character["meaning_state"],
        },
        "evidence": [],
        "direct_facts": [],
        "available_actions": [],
        "available_resolver_capabilities": [],
        "scene_context": {
            "channel_scope": "private",
            "character_role": "companion",
            "semantic_scene": "quiet private greeting",
            "conversation_continuity": "No unresolved public commitment.",
            "semantic_temporal_context": "immediate",
        },
        "private_continuity_context": "I remain calmly attentive.",
    }


def test_v2_regression_baseline_accepts_exact_public_contract() -> None:
    """The retained baseline is the frozen native V2 input shape."""

    validated = validate_cognition_core_input(_payload())

    assert validated["schema_version"] == "cognition_core_input.v2"
    assert validated["state_scope"] == "user"


def test_v2_regression_baseline_rejects_legacy_prompt_state() -> None:
    """Legacy prose-stage fields cannot re-enter through regression fixtures."""

    payload = _payload()
    payload["internal_monologue"] = "forbidden legacy prompt state"

    with pytest.raises(CognitionContractError):
        validate_cognition_core_input(payload)


def test_stage_00_fixture_preserves_required_response_path_cases() -> None:
    """The reusable baseline keeps every retained current-chat scenario."""

    fixture_data = _fixture()
    case_ids = {
        item["case_id"]
        for item in fixture_data["cases"]
    }

    assert {
        "private_text",
        "group_text",
        "reply",
        "silence",
        "rag_skip",
        "rag_hit",
        "think_only",
        "no_remember",
        "listen_only",
    } <= case_ids
    assert len(fixture_data["frozen_evidence_corpus"]["known_facts"]) == 3


def test_stage_00_frozen_evidence_projects_current_rag_shape() -> None:
    """The frozen RAG corpus still reaches typed current evidence lanes."""

    corpus = _fixture()["frozen_evidence_corpus"]
    expected = corpus["expected_projection"]

    rag_result = project_known_facts(
        corpus["known_facts"],
        current_user_id=expected["current_user_id"],
        character_user_id=expected["character_user_id"],
        answer=expected["answer"],
        unknown_slots=[],
        loop_count=3,
    )

    assert rag_result["answer"] == expected["answer"]
    assert "Prior discussion settled on a short plan." in (
        rag_result["conversation_evidence"][0]
    )
    assert rag_result["memory_evidence"][0]["source_system"] == (
        "user_memory_units"
    )
    assert rag_result["user_memory_unit_candidates"][0]["unit_id"] == (
        "unit-plan-1"
    )
    assert rag_result["external_evidence"][0]["url"] == (
        "https://example.com/stage-00"
    )
    assert len(rag_result["supervisor_trace"]["dispatched"]) == 3
