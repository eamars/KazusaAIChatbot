"""Direct ownership tests for semantic appraisal projection."""

from __future__ import annotations

import json

from kazusa_ai_chatbot.cognition_core_v2.semantic_appraisal import (
    _project_question_state,
    appraise_semantic_question,
)
from kazusa_ai_chatbot.cognition_core_v2.state_projection import (
    PromptProjectionV2,
)


def test_character_constraint_projection_excludes_standard_handles() -> None:
    """Do not expose persisted standards through one appraisal question."""

    projection = PromptProjectionV2(
        payload={
            "character_constraints": {
                "standards": [{
                    "handle": "s1",
                    "description": "repository default",
                }],
            },
        },
        handle_to_ref={
            "s1": {
                "kind": "standard",
                "entity_id": "s1",
            },
        },
    )
    question = {
        "question_id": "q:moral_identity",
        "question_kind": "moral_identity",
        "semantic_question": "Inspect the bounded question.",
        "evidence_handles": ["e1"],
        "permitted_role_handles": ["s1"],
        "permitted_delta_paths": [],
        "dependencies": [],
    }

    projected_state = _project_question_state(projection, question)

    assert "s1" not in json.dumps(projected_state, sort_keys=True)
    assert "character_constraints" not in projected_state


def test_semantic_appraisal_exposes_owned_contract() -> None:
    """Keep the semantic appraisal entrypoint attached to this source owner."""

    assert callable(appraise_semantic_question)
