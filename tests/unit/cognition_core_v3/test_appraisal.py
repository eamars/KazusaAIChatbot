"""Deterministic tests for the fixed A1/A2 appraisal product contract."""

from __future__ import annotations

import pytest

from kazusa_ai_chatbot.cognition_core_v2.semantic_source_planner import (
    question_proposition_kind_semantics,
    question_proposition_kinds,
)
from kazusa_ai_chatbot.cognition_core_v3 import appraisal


def _question() -> dict[str, object]:
    """Build one canonical V2 question for bridge tests."""

    return {
        "question_id": "q:event_agency",
        "question_kind": "event_agency",
        "semantic_question": "判断责任和意图",
        "evidence_handles": ["e1"],
        "permitted_role_handles": ["ce1", "current_user"],
        "permitted_role_assignment_handles": ["self", "current_user"],
        "permitted_delta_paths": ["active_events.ce1.responsibility"],
        "dependencies": [],
    }


def test_appraisal_family_vocabulary_uses_the_canonical_v2_owner() -> None:
    """V3 exposes no copied family vocabulary or semantics registry."""

    for family in (
        "event_agency",
        "goal_threat_outcome",
        "epistemic_comparison_memory",
        "relationship_social",
        "moral_identity",
        "existential_drive",
    ):
        assert appraisal.family_proposition_kinds(family) == (
            question_proposition_kinds(family)
        )
        assert appraisal.family_proposition_kind_semantics(family) == (
            question_proposition_kind_semantics(family)
        )
    assert not hasattr(appraisal, "FAMILY_PROPOSITION_KINDS")
    assert not hasattr(appraisal, "FAMILY_PROPOSITION_KIND_SEMANTICS")


def test_appraisal_stage_requires_exact_family_objects_and_empty_arrays() -> None:
    """Each planned family owns exactly two bounded product arrays."""

    valid = {
        "event_agency": {"propositions": [], "deltas": []},
    }
    assert appraisal.validate_appraisal_stage_output(
        valid,
        planned_families=("event_agency",),
    ) == valid

    with pytest.raises(ValueError, match="exactly"):
        appraisal.validate_appraisal_stage_output(
            {"event_agency": {"propositions": [], "deltas": []}, "extra": {}},
            planned_families=("event_agency",),
        )
    with pytest.raises(ValueError, match="only propositions and deltas"):
        appraisal.validate_appraisal_stage_output(
            {"event_agency": {"propositions": [], "deltas": [], "extra": []}},
            planned_families=("event_agency",),
        )


def test_appraisal_nullable_object_handle_is_normalized_before_v2_bridge() -> None:
    """Only the optional object handle accepts null, which is then omitted."""

    proposition = {
        "proposition_kind": "meaning_relevance",
        "subject_handle": "self",
        "object_handle": None,
        "evidence_handles": ["e1"],
        "role_assignments": [{
            "role": "experiencer",
            "entity_handle": "self",
        }],
        "semantic_value": "当前事件与角色的意义判断相关。",
    }
    normalized = appraisal.validate_appraisal_stage_output(
        {"existential_drive": {
            "propositions": [proposition],
            "deltas": [],
        }},
        planned_families=("existential_drive",),
    )
    assert normalized["existential_drive"]["propositions"][0] == {
        key: value
        for key, value in proposition.items()
        if key != "object_handle"
    }
    with pytest.raises(TypeError, match="object handle"):
        appraisal.validate_appraisal_stage_output(
            {"existential_drive": {
                "propositions": [{**proposition, "object_handle": 3}],
                "deltas": [],
            }},
            planned_families=("existential_drive",),
        )


def test_appraisal_nullable_existential_proposition_survives_reduction() -> None:
    """The canonical existential self proposition remains one accepted product."""

    question = {
        "question_id": "q:existential_drive",
        "question_kind": "existential_drive",
        "semantic_question": "判断意义相关性",
        "evidence_handles": ["e1"],
        "permitted_role_handles": ["self"],
        "permitted_role_assignment_handles": ["self"],
        "permitted_delta_paths": [],
        "dependencies": [],
    }
    reduced = appraisal.reduce_appraisal_stage_output(
        {"existential_drive": {
            "propositions": [{
                "proposition_kind": "meaning_relevance",
                "subject_handle": "self",
                "object_handle": None,
                "evidence_handles": ["e1"],
                "role_assignments": [{
                    "role": "experiencer",
                    "entity_handle": "self",
                }],
                "semantic_value": "当前事件与角色的意义判断相关。",
            }],
            "deltas": [],
        }},
        planned_families=("existential_drive",),
        questions_by_family={"existential_drive": question},
        evidence_handles=["e1"],
        handle_to_ref={"self": {
            "entity_id": "character-1",
            "kind": "self",
        }},
    )
    assert len(reduced[0]["propositions"]) == 1
    assert "object_handle" not in reduced[0]["propositions"][0]


def test_appraisal_role_assignments_are_authorized_per_cited_evidence() -> None:
    """Retrieved evidence cannot grant current-user or self role ownership."""

    proposition = {
        "proposition_kind": "relationship_threat",
        "subject_handle": "self",
        "evidence_handles": ["e2"],
        "role_assignments": [{
            "role": "actor",
            "entity_handle": "current_user",
        }],
        "semantic_value": "关系含义判断。",
    }
    raw = {"relationship_social": {
        "propositions": [proposition],
        "deltas": [],
    }}
    role_domains = {
        "e1": ["current_user", "self"],
        "e2": [],
    }
    with pytest.raises(ValueError, match="authorized"):
        appraisal.validate_appraisal_stage_output(
            raw,
            planned_families=("relationship_social",),
            role_assignment_handles_by_evidence=role_domains,
        )
    current_event = {
        **proposition,
        "evidence_handles": ["e1"],
    }
    accepted = appraisal.validate_appraisal_stage_output(
        {"relationship_social": {
            "propositions": [current_event],
            "deltas": [],
        }},
        planned_families=("relationship_social",),
        role_assignment_handles_by_evidence=role_domains,
    )
    assert accepted["relationship_social"]["propositions"] == [
        current_event
    ]
def test_appraisal_stage_preserves_v2_proposition_and_delta_products() -> None:
    """The bridge retains both product kinds and invokes the V2 validator."""

    question = _question()
    raw = {
        "event_agency": {
            "propositions": [{
                "proposition_kind": "responsibility",
                "subject_handle": "ce1",
                "evidence_handles": ["e1"],
                "role_assignments": [{
                    "role": "actor",
                    "entity_handle": "current_user",
                }],
                "semantic_value": "当前用户对事件结果负有责任。",
            }],
            "deltas": [{
                "target_path": "active_events.ce1.responsibility",
                "delta": 2,
                "evidence_handles": ["e1"],
                "reason": "当前证据支持责任变化。",
            }],
        }
    }
    reduced = appraisal.reduce_appraisal_stage_output(
        raw,
        planned_families=("event_agency",),
        questions_by_family={"event_agency": question},
        evidence_handles=["e1"],
        handle_to_ref={
            "ce1": {"entity_id": "candidate:event:e1", "kind": "event"},
            "current_user": {"entity_id": "user-1", "kind": "user"},
            "self": {"entity_id": "character-1", "kind": "self"},
        },
    )
    assert reduced[0]["propositions"] == raw["event_agency"]["propositions"]
    assert reduced[0]["deltas"] == raw["event_agency"]["deltas"]
    assert reduced[0]["question_id"] == "q:event_agency"


def test_appraisal_reduction_merges_every_product_and_preserves_ordered_metadata() -> None:
    """Canonical V2 merge retains every proposition, delta, and explanation."""

    question = _question()
    question["evidence_handles"] = ["e1", "e2"]
    question["permitted_delta_paths"] = [
        "active_events.ce1.responsibility",
        "active_events.ce1.intentionality",
    ]
    raw = {
        "event_agency": {
            "propositions": [
                {
                    "proposition_kind": "responsibility",
                    "subject_handle": "ce1",
                    "evidence_handles": ["e2", "e1"],
                    "role_assignments": [],
                    "semantic_value": "第一责任判断。",
                },
                {
                    "proposition_kind": "intentionality",
                    "subject_handle": "ce1",
                    "evidence_handles": ["e1", "e2"],
                    "role_assignments": [],
                    "semantic_value": "第二意图判断。",
                },
            ],
            "deltas": [
                {
                    "target_path": "active_events.ce1.responsibility",
                    "delta": 1,
                    "evidence_handles": ["e1", "e2"],
                    "reason": "第一变化依据。",
                },
                {
                    "target_path": "active_events.ce1.intentionality",
                    "delta": -1,
                    "evidence_handles": ["e2", "e1"],
                    "reason": "第二变化依据。",
                },
            ],
        }
    }

    reduced = appraisal.reduce_appraisal_stage_output(
        raw,
        planned_families=("event_agency",),
        questions_by_family={"event_agency": question},
        evidence_handles=["e1", "e2"],
        handle_to_ref={
            "ce1": {"entity_id": "candidate:event:e1", "kind": "event"},
            "current_user": {"entity_id": "user-1", "kind": "user"},
            "self": {"entity_id": "character-1", "kind": "self"},
        },
    )

    result = reduced[0]
    assert result["propositions"] == raw["event_agency"]["propositions"]
    assert result["deltas"] == raw["event_agency"]["deltas"]
    assert result["selected_evidence_handles"] == ["e2", "e1"]
    assert result["selected_role_handles"] == ["ce1"]
    assert result["explanation"] == (
        "第一责任判断。 第二意图判断。 第一变化依据。 第二变化依据。"
    )


def test_appraisal_stage_rejects_over_capacity_and_invalid_product_fields() -> None:
    """Eight products fit while a ninth or malformed row fails closed."""

    proposition = {
        "proposition_kind": "responsibility",
        "subject_handle": "ce1",
        "evidence_handles": ["e1"],
        "role_assignments": [],
        "semantic_value": "支持的责任判断。",
    }
    valid = {
        "event_agency": {
            "propositions": [dict(proposition) for _ in range(8)],
            "deltas": [{
                "target_path": f"active_events.ce1.responsibility_{index}",
                "delta": 1,
                "evidence_handles": ["e1"],
                "reason": "支持的责任变化。",
            } for index in range(8)],
        }
    }
    assert appraisal.validate_appraisal_stage_output(
        valid,
        planned_families=("event_agency",),
    ) == valid
    over_capacity = {
        "event_agency": {
            "propositions": [dict(proposition) for _ in range(9)],
            "deltas": [],
        }
    }
    with pytest.raises(ValueError, match="at most eight"):
        appraisal.validate_appraisal_stage_output(
            over_capacity,
            planned_families=("event_agency",),
        )
    over_delta_capacity = {
        "event_agency": {
            "propositions": [],
            "deltas": [{
                "target_path": f"active_events.ce1.responsibility_{index}",
                "delta": 1,
                "evidence_handles": ["e1"],
                "reason": "支持的责任变化。",
            } for index in range(9)],
        }
    }
    with pytest.raises(ValueError, match="at most eight"):
        appraisal.validate_appraisal_stage_output(
            over_delta_capacity,
            planned_families=("event_agency",),
        )
    malformed = {
        "event_agency": {
            "propositions": [{**proposition, "unknown": True}],
            "deltas": [],
        }
    }
    with pytest.raises(ValueError, match="fields are not exact"):
        appraisal.validate_appraisal_stage_output(
            malformed,
            planned_families=("event_agency",),
        )
    with pytest.raises(TypeError, match="evidence is invalid"):
        appraisal.validate_appraisal_stage_output(
            {
                "event_agency": {
                    "propositions": [{
                        **proposition,
                        "evidence_handles": "e1",
                    }],
                    "deltas": [],
                }
            },
            planned_families=("event_agency",),
        )
