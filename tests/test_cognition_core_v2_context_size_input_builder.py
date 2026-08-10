"""Deterministic regression for the production Cognition V2 input builder."""

from __future__ import annotations

import json
from typing import Any

from kazusa_ai_chatbot.cognition_core_v2.state_models import (
    EVENT_FIELDS,
    build_acquaintance_user_state,
    build_character_production_state,
    validate_cognition_state,
)
from kazusa_ai_chatbot.cognition_core_v2.prompt_budget import canonical_digest
from kazusa_ai_chatbot.nodes.persona_supervisor2_cognition import (
    build_cognition_input_from_global_state,
)
from tests.cognition_core_v2_test_helpers import (
    canonical_character_identity,
    canonical_episode,
)


_NOW = "2026-08-06T00:00:00Z"
_RELATIONSHIP_CONTEXT_LIMIT = 900
_CHARACTER_CONTEXT_LIMIT = 1200


def _serialized_size(value: dict[str, Any]) -> int:
    """Measure one packet with the canonical contract serialization."""

    serialized = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return len(serialized)


def _summary(prefix: str, length: int) -> str:
    """Build deterministic fixture text with an exact character length."""

    return prefix + "x" * max(0, length - len(prefix))


def _relationship_event(
    *,
    entity_id: str,
    relationship_id: str,
    description: str,
    salience: int,
) -> dict[str, Any]:
    """Build one valid relationship-scoped event for the RCA shape."""

    event = {
        "entity_id": entity_id,
        "description": description,
        "salience": salience,
        "role_refs": [{
            "role": "affected_relationship",
            "entity_kind": "relationship",
            "entity_id": relationship_id,
        }],
        "evidence_refs": [{
            "source_kind": "episode",
            "source_id": entity_id,
            "occurred_at": _NOW,
            "semantic_summary": description,
        }],
        "created_at": _NOW,
        "updated_at": _NOW,
        "status": "active",
        "outcome_impact": 0,
    }
    event.update({
        field_name: 0
        for field_name in EVENT_FIELDS - {"status", "outcome_impact"}
    })
    return event


def _incident_state() -> dict[str, Any]:
    """Build the synthetic two-row relationship shape from the RCA."""

    state = dict(build_acquaintance_user_state(
        global_user_id="context-limit-user",
        updated_at=_NOW,
    ))
    relationship_id = state["relationship"]["relationship_id"]
    state["active_events"] = [
        _relationship_event(
            entity_id="event-1",
            relationship_id=relationship_id,
            description=_summary("incident event A: ", 160),
            salience=90,
        ),
        _relationship_event(
            entity_id="event-2",
            relationship_id=relationship_id,
            description=_summary("incident event B: ", 155),
            salience=80,
        ),
    ]
    return validate_cognition_state(state)


def test_production_input_builder_fits_relationship_context() -> None:
    """The real graph builder must return a bounded relationship packet."""

    mutable_state = _incident_state()
    user_id = mutable_state["owner_user_id"]
    payload = build_cognition_input_from_global_state(
        {
            "cognitive_episode": canonical_episode(
                episode_id="context-size-builder-episode",
                content="context-size integration event",
                current_global_user_id=user_id,
            ),
            "global_user_id": user_id,
            "user_input": "context-size integration event",
            "decontextualized_input": "context-size integration event",
            "user_multimedia_input": [],
            "rag_result": {"memory_evidence": []},
            "character_profile": canonical_character_identity(
                marker="context-size-builder",
            ),
            "public_group_scene": "",
        },
        mutable_state=mutable_state,
        character_state=build_character_production_state(
            updated_at=_NOW,
        ),
    )

    relationship_context = payload["relationship_context"]
    assert isinstance(relationship_context, dict)
    assert _serialized_size(relationship_context) <= (
        _RELATIONSHIP_CONTEXT_LIMIT
    )
    character_context = payload["character_operational_context"]
    assert isinstance(character_context, dict)
    assert _serialized_size(character_context) <= _CHARACTER_CONTEXT_LIMIT
    character_body = {
        key: value
        for key, value in character_context.items()
        if key != "context_digest"
    }
    assert character_context["context_digest"] == (
        canonical_digest(character_body)
    )
