"""Direct ownership tests for semantic appraisal projection."""

from __future__ import annotations

import json

import pytest

from kazusa_ai_chatbot.cognition_core_v2.semantic_appraisal import (
    _fit_appraisal_payload,
    _project_question_state,
    appraise_semantic_question,
    validate_semantic_appraisal_result,
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
        "permitted_role_assignment_handles": [],
        "permitted_delta_paths": [],
        "dependencies": [],
    }

    projected_state = _project_question_state(projection, question)

    assert "s1" not in json.dumps(projected_state, sort_keys=True)
    assert "character_constraints" not in projected_state


def test_semantic_appraisal_exposes_owned_contract() -> None:
    """Keep the semantic appraisal entrypoint attached to this source owner."""

    assert callable(appraise_semantic_question)


def test_causal_candidate_is_rejected_as_role_assignment_handle() -> None:
    """A ceN handle cannot appear in role assignment entity handles."""

    question = {
        "question_id": "q:event_agency",
        "question_kind": "event_agency",
        "semantic_question": "Assess responsibility and intentionality.",
        "evidence_handles": ["e1"],
        "permitted_role_handles": ["ce1", "current_user", "self"],
        "permitted_role_assignment_handles": ["current_user", "self"],
        "permitted_delta_paths": ["active_events.ce1.intentionality"],
        "dependencies": [],
    }
    parsed = {
        "question_id": "q:event_agency",
        "selected_evidence_handles": ["e1"],
        "selected_role_handles": ["ce1", "self"],
        "propositions": [{
            "proposition_kind": "intentionality",
            "subject_handle": "ce1",
            "evidence_handles": ["e1"],
            "role_assignments": [{
                "role": "target",
                "entity_handle": "ce1",
            }],
            "semantic_value": "The group event candidate appears deliberate.",
        }],
        "deltas": [],
        "explanation": "The bounded evidence supports the event claim.",
    }

    with pytest.raises(
        ValueError,
        match=r"role_assignments\[\*\]\.entity_handle must be one of "
        r'\["current_user", "self"\]',
    ):
        validate_semantic_appraisal_result(
            parsed,
            question,
            {"e1"},
            {
                "ce1": {
                    "scope": "user",
                    "kind": "event",
                    "entity_id": "candidate:event:e1",
                },
                "current_user": {
                    "scope": "user",
                    "kind": "relationship",
                    "entity_id": "relationship:user:unit",
                },
                "self": {
                    "scope": "character",
                    "kind": "meaning",
                    "entity_id": "meaning:character",
                },
            },
        )


def test_appraisal_fitting_prunes_causal_and_assignment_domains_independently(
) -> None:
    """Removing causal rows never strips the assignment survivor domain."""

    causal_handles = [f"ce{index}" for index in range(1, 41)]
    payload = {
        "question": {
            "question_id": "q:event_agency",
            "question_kind": "event_agency",
            "semantic_question": "Identify the current event agency.",
            "permitted_role_handles": [*causal_handles, "self"],
            "permitted_role_assignment_handles": [
                "self",
                "current_user",
            ],
            "candidate_origin_evidence": {
                handle: "e1" for handle in causal_handles
            },
            "permitted_delta_path_domains": [{
                "state_field": "events",
                "handles": list(causal_handles),
                "axes": ["salience"],
                "delta_limit": 40,
            }],
            "permitted_proposition_kinds": ["event"],
            "proposition_kind_semantics": {
                "event": "one event proposition",
            },
            "handle_field_domains": {
                "subject_handle": [*causal_handles, "self"],
                "object_handle": [*causal_handles, "self"],
                "entity_handle": ["self", "current_user"],
                "evidence_handles": ["e1"],
            },
            "role_handle_semantics": {},
            "micro_appraisal": {
                "item_index": 1,
                "maximum_items": 8,
            },
        },
        "evidence": [],
        "state": {
            "events": [
                {
                    "handle": handle,
                    "semantic_text": "x" * 800,
                }
                for handle in causal_handles
            ],
        },
    }

    fitted_text, surviving_roles, surviving_assignments = (
        _fit_appraisal_payload(
            payload,
            system_prompt_chars=0,
        )
    )

    assert surviving_assignments == frozenset({"self", "current_user"})
    assert "ce1" in surviving_roles
    assert "ce40" not in surviving_roles
    fitted_question = json.loads(fitted_text)["question"]
    assert set(
        fitted_question["handle_field_domains"]["entity_handle"]
    ) == {"self", "current_user"}
    assert "ce40" not in fitted_question["permitted_role_handles"]
    assert fitted_question["handle_field_domains"][
        "subject_handle"
    ] != fitted_question["handle_field_domains"]["entity_handle"]
