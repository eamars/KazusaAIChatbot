"""Deterministic ownership test for src/kazusa_ai_chatbot/cognition_core_v2/state_projection.py."""

from __future__ import annotations

from importlib import import_module
from typing import Any

import pytest

from kazusa_ai_chatbot.cognition_core_v2.state_models import (
    build_acquaintance_user_state,
    build_character_production_state,
)
from kazusa_ai_chatbot.cognition_core_v2.state_projection import (
    project_relationship_context,
    project_state_for_prompt,
)
from tests.cognition_core_v2_test_helpers import canonical_identity_context

MODULE_PATH = "kazusa_ai_chatbot.cognition_core_v2.state_projection"
EXPECTED_SYMBOLS = ["project_state_for_prompt"]


def test_state_projection_exposes_owned_contract() -> None:
    """Keep the module's named owner contract discoverable."""

    module = import_module(MODULE_PATH)
    missing_symbols = [
        symbol
        for symbol in EXPECTED_SYMBOLS
        if not hasattr(module, symbol)
    ]

    assert not missing_symbols, (
        f"{MODULE_PATH} is missing owner symbols: {missing_symbols}"
    )


def test_relationship_maintenance_metadata_is_not_projected() -> None:
    """Keep maintenance bookkeeping outside the model-facing projection."""

    state = build_acquaintance_user_state(
        global_user_id="projection-maintenance-user",
        updated_at="2026-08-18T00:00:00Z",
    )
    relationship = state["relationship"]
    relationship["relationship_maintenance"] = {
        "schema_version": "relationship_maintenance.v1",
        "last_interaction_date_utc": "2026-08-18",
        "last_bonus_date_utc": None,
        "last_source_id": "episode:projection",
        "processed_source_ids": ["episode:projection"],
    }

    projected = project_relationship_context(relationship)

    assert "relationship_maintenance" not in projected
    assert "relationship_maintenance" not in projected["axes"]


def _constraints() -> dict[str, Any]:
    """Build the character constraint projection for identity tests."""

    character_state = build_character_production_state(
        updated_at="2026-08-18T00:00:00Z",
    )
    constraints = {
        "drives": character_state["drives"],
        "standards": character_state["standards"],
        "meaning_state": character_state["meaning_state"],
        "personality_judgment": {
            "logic": "evidence-led",
            "defense": "reserved",
            "quirks": "brief hesitation",
            "taboos": "preserve character agency",
        },
    }
    return constraints


def _causal_state() -> dict[str, Any]:
    """Build a minimal projection state with three causal entity kinds."""

    state = build_acquaintance_user_state(
        global_user_id="projection-source-user",
        updated_at="2026-08-18T00:00:00Z",
    )
    evidence_refs = {
        "event": {
            "source_kind": "episode",
            "source_id": "source-event",
            "occurred_at": "2026-08-18T00:00:00Z",
            "semantic_summary": "event source",
        },
        "threat": {
            "source_kind": "episode",
            "source_id": "source-threat",
            "occurred_at": "2026-08-18T00:00:00Z",
            "semantic_summary": "threat source",
        },
        "knowledge_gap": {
            "source_kind": "episode",
            "source_id": "source-gap",
            "occurred_at": "2026-08-18T00:00:00Z",
            "semantic_summary": "gap source",
        },
    }
    state["active_events"] = [{
        "entity_id": "event:custom-source",
        "description": "Stored event",
        "salience": 50,
        "role_refs": [],
        "evidence_refs": [evidence_refs["event"]],
        "created_at": "2026-08-18T00:00:00Z",
        "status": "active",
    }]
    state["threats"] = [{
        "entity_id": "threat:custom-source",
        "description": "Stored threat",
        "salience": 50,
        "role_refs": [],
        "evidence_refs": [evidence_refs["threat"]],
        "created_at": "2026-08-18T00:00:00Z",
        "status": "active",
    }]
    state["knowledge_gaps"] = [{
        "entity_id": "knowledge_gap:custom-source",
        "description": "Stored gap",
        "salience": 50,
        "role_refs": [],
        "evidence_refs": [evidence_refs["knowledge_gap"]],
        "created_at": "2026-08-18T00:00:00Z",
        "status": "open",
    }]
    return state


def _evidence_row(handle: str, evidence_ref: dict[str, str]) -> dict[str, Any]:
    """Build one prompt evidence row for a source identity."""

    return {
        "evidence_handle": handle,
        "evidence_ref": evidence_ref,
        "semantic_text": evidence_ref["semantic_summary"],
    }


def test_source_identity_binds_current_candidates_to_native_handles() -> None:
    """Reuse one native handle for each exact current causal source."""

    state = _causal_state()
    evidence = [
        _evidence_row("e1", state["active_events"][0]["evidence_refs"][0]),
        _evidence_row("e2", state["threats"][0]["evidence_refs"][0]),
        _evidence_row(
            "e3",
            state["knowledge_gaps"][0]["evidence_refs"][0],
        ),
    ]
    projection = project_state_for_prompt(
        state,
        character_constraints=_constraints(),
        character_identity_context=canonical_identity_context(),
        evidence=evidence,
    )

    assert {"ev1", "t1", "k1"} <= set(projection.handle_to_ref)
    assert {"ce1", "ct2", "ck3"}.isdisjoint(projection.handle_to_ref)
    assert projection.payload["events"][0]["evidence_handles"] == ["e1"]
    assert projection.payload["threats"][0]["evidence_handles"] == ["e2"]
    assert projection.payload["knowledge_gaps"][0]["evidence_handles"] == [
        "e3"
    ]


def test_source_identity_keeps_distinct_candidate_and_native_entity() -> None:
    """Keep a candidate when its source differs from stored native evidence."""

    state = _causal_state()
    transit_ref = {
        "source_kind": "episode",
        "source_id": "source-transit",
        "occurred_at": "2026-08-18T00:00:00Z",
        "semantic_summary": "transit source",
    }
    projection = project_state_for_prompt(
        state,
        character_constraints=_constraints(),
        character_identity_context=canonical_identity_context(),
        evidence=[_evidence_row("e1", transit_ref)],
    )

    assert "ev1" in projection.handle_to_ref
    assert "ce1" in projection.handle_to_ref
    assert "evidence_handles" not in projection.payload["events"][0]


def test_source_identity_rejects_ambiguous_native_matches() -> None:
    """Fail closed when two eligible native rows share one source identity."""

    state = _causal_state()
    state["active_events"].append({
        **state["active_events"][0],
        "entity_id": "event:duplicate-source",
    })
    evidence_ref = state["active_events"][0]["evidence_refs"][0]

    with pytest.raises(ValueError, match="ambiguous same-source"):
        project_state_for_prompt(
            state,
            character_constraints=_constraints(),
            character_identity_context=canonical_identity_context(),
            evidence=[_evidence_row("e1", evidence_ref)],
        )


def test_source_identity_allows_multiple_goals_with_shared_source() -> None:
    """Preserve independently valid goals that cite one source identity."""

    state = _causal_state()
    shared_ref = {
        "source_kind": "episode",
        "source_id": "shared-goal-source",
        "occurred_at": "2026-08-18T00:00:00Z",
        "semantic_summary": "shared goal source",
    }
    state["goals"] = [
        {
            "entity_id": "goal:first-shared-source",
            "description": "First pursuing goal",
            "salience": 50,
            "role_refs": [],
            "evidence_refs": [shared_ref],
            "created_at": "2026-08-18T00:00:00Z",
            "updated_at": "2026-08-18T00:00:00Z",
            "status": "pursuing",
            "goal_kind": "ordinary_response",
            "importance": 50,
            "progress": 0,
            "obstruction": 0,
            "expected_success": 50,
            "controllability": 50,
            "recoverability": 50,
            "urgency": 50,
        },
        {
            "entity_id": "goal:second-shared-source",
            "description": "Second blocked goal",
            "salience": 50,
            "role_refs": [],
            "evidence_refs": [shared_ref],
            "created_at": "2026-08-18T00:00:00Z",
            "updated_at": "2026-08-18T00:00:00Z",
            "status": "blocked",
            "goal_kind": "safety",
            "importance": 50,
            "progress": 0,
            "obstruction": 20,
            "expected_success": 50,
            "controllability": 50,
            "recoverability": 50,
            "urgency": 50,
        },
    ]

    projection = project_state_for_prompt(
        state,
        character_constraints=_constraints(),
        character_identity_context=canonical_identity_context(),
        evidence=[_evidence_row("e1", shared_ref)],
    )

    assert [row["handle"] for row in projection.payload["goals"]] == [
        "g1",
        "g2",
    ]
    assert [row["description"] for row in projection.payload["goals"]] == [
        "First pursuing goal",
        "Second blocked goal",
    ]
    assert "evidence_handles" not in projection.payload["goals"][0]
    assert "evidence_handles" not in projection.payload["goals"][1]
