"""Regression matrix for trace-database failure families."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from kazusa_ai_chatbot.cognition_core_v2.action_selection import (
    _validate_action_plan_decision,
)
from kazusa_ai_chatbot.cognition_core_v2.contracts import (
    validate_relational_willingness,
)
from kazusa_ai_chatbot.cognition_core_v2.goal_cognition import (
    validate_goal_bid_draft,
)
from kazusa_ai_chatbot.cognition_core_v2.semantic_appraisal import (
    _canonicalize_semantic_appraisal_item,
    validate_semantic_appraisal_result,
)
from kazusa_ai_chatbot.cognition_core_v2.state_models import (
    CognitionStateError,
)
from kazusa_ai_chatbot.cognition_core_v2.transition_guards import (
    transition_event,
    transition_knowledge_gap,
)


_ROOT = Path(__file__).resolve().parents[1]
_TRACE_INVENTORY_PATH = (
    _ROOT
    / "test_artifacts"
    / "diagnostics"
    / "trace_failure_mode_inventory_2026-08-04.json"
)

_EXPECTED_TRACE_FAMILIES = frozenset({
    "evidence_handles_not_permitted",
    "relational_willingness_evidence_unavailable",
    "semantic_delta_path_not_owned",
    "selected_evidence_unknown_handle",
    "terminal_event_transition_rejected",
    "goal_bid_fields_not_exact",
    "candidate_origin_evidence_missing",
    "semantic_role_value_invalid",
    "relational_willingness_episode_evidence_missing",
    "resolved_knowledge_gap_transition_rejected",
    "selected_roles_unknown_handle",
    "semantic_proposition_subject_kind_mismatch",
    "pending_resolution_invalid",
    "semantic_proposition_object_handle_not_permitted",
    "delta_reason_invalid",
    "role_handles_not_permitted",
    "goal_bid_consequences_invalid",
    "relational_willingness_fields_not_exact",
    "semantic_delta_type_invalid",
    "semantic_micro_appraisal_fields_not_exact",
})

_EXPECTED_FAILED_STAGE_GROUPS = frozenset({
    "goal_cognition.ordinary_response.initial",
    "goal_cognition.autonomy_boundary.initial",
    "goal_cognition.ordinary_response.selection_initial",
    "goal_cognition.autonomy_boundary.selection_initial",
    "goal_cognition.ordinary_response.selection_regeneration_1",
    "goal_cognition.ordinary_response.repair_1",
    "goal_cognition.ordinary_response.repair_2",
    "goal_cognition.ordinary_response.selection_regeneration_2",
    "goal_cognition.autonomy_boundary.repair_1",
    "persona_relevance_agent.initial",
    "character_identity_growth.review",
    "action_planning.repair",
    "action_planning",
    "dialog_generator",
    "goal_cognition.autonomy_boundary.selection_regeneration_1",
})

_EXPECTED_NON_SUCCESS_EVENT_GROUPS = frozenset({
    (
        "runtime_error",
        "self_cognition.worker",
        "failed",
        "KeyError",
        "'current_thread'",
        "self_cognition_case_processing",
    ),
    ("tick", "self_cognition.worker", "failed", "", "", ""),
    ("tick", "reflection_cycle.worker", "failed", "", "", ""),
    (
        "semantic_dialog_misalignment",
        "nodes.dialog_agent",
        "retrying",
        "",
        "",
        "",
    ),
    (
        "validation_warning",
        "reflection_cycle.worker",
        "warning",
        "",
        "",
        "",
    ),
    ("turn", "brain_service", "failed", "", "", ""),
    (
        "runtime_error",
        "brain_service",
        "failed",
        "CognitionExecutionError",
        "required cognition branch failed: ordinary_response",
        "d082d501044831dc",
    ),
    (
        "internal_monologue_residue_recorder",
        "internal_monologue_residue",
        "retry",
        "",
        "",
        "",
    ),
    (
        "runtime_error",
        "brain_service",
        "failed",
        "CognitionExecutionError",
        "required cognition branch failed: ordinary_response",
        "6cbfaea54839c2fb",
    ),
    (
        "runtime_error",
        "reflection_cycle.worker",
        "failed",
        "ValueError",
        "recursive reflection root metadata is inconsistent",
        "reflection_worker_tick",
    ),
})

_WILLINGNESS_REASON = "\u597d\u7684"


def _semantic_question(
    *,
    question_kind: str = "relationship_social",
    question_id: str = "q:relationship_social",
    roles: list[str] | None = None,
    paths: list[str] | None = None,
) -> dict[str, Any]:
    """Build the smallest prompt-owned semantic question for one validator."""

    return {
        "question_id": question_id,
        "question_kind": question_kind,
        "semantic_question": "bounded semantic question",
        "evidence_handles": ["e1", "e2"],
        "permitted_role_handles": roles or ["current_user", "self", "r1"],
        "permitted_delta_paths": paths or [
            "relationship.r1.axes.perceived_closeness",
        ],
        "dependencies": [],
    }


def _semantic_handle_refs() -> dict[str, dict[str, str]]:
    """Build canonical-looking refs for the handles used by the matrix."""

    return {
        "current_user": {
            "scope": "user",
            "kind": "user",
            "entity_id": "user:current",
        },
        "self": {
            "scope": "character",
            "kind": "character",
            "entity_id": "character:global",
        },
        "r1": {
            "scope": "user",
            "kind": "relationship",
            "entity_id": "relationship:r1",
        },
        "ce1": {
            "scope": "user",
            "kind": "event",
            "entity_id": "candidate:event:e1",
        },
        "ck1": {
            "scope": "user",
            "kind": "knowledge_gap",
            "entity_id": "candidate:knowledge_gap:e1",
        },
    }


def _empty_semantic_result(
    question_id: str = "q:relationship_social",
) -> dict[str, Any]:
    """Return an otherwise valid empty aggregate appraisal."""

    return {
        "question_id": question_id,
        "selected_evidence_handles": [],
        "selected_role_handles": [],
        "propositions": [],
        "deltas": [],
        "explanation": "bounded explanation",
    }


def _semantic_proposition(
    *,
    proposition_kind: str = "social_meaning",
    subject_handle: str = "current_user",
    evidence_handles: list[str] | None = None,
    role: str = "actor",
    entity_handle: str = "current_user",
    object_handle: str | None = None,
) -> dict[str, Any]:
    """Build one valid-shaped proposition with one switchable defect."""

    proposition: dict[str, Any] = {
        "proposition_kind": proposition_kind,
        "subject_handle": subject_handle,
        "evidence_handles": (
            ["e1"] if evidence_handles is None else evidence_handles
        ),
        "role_assignments": [{
            "role": role,
            "entity_handle": entity_handle,
        }],
        "semantic_value": "bounded semantic value",
    }
    if object_handle is not None:
        proposition["object_handle"] = object_handle
    return proposition


def _semantic_delta(
    *,
    target_path: str = "relationship.r1.axes.perceived_closeness",
    delta: object = 10,
    evidence_handles: list[str] | None = None,
    reason: str = "bounded reason",
) -> dict[str, Any]:
    """Build one valid-shaped delta with one switchable defect."""

    return {
        "target_path": target_path,
        "delta": delta,
        "evidence_handles": (
            ["e1"] if evidence_handles is None else evidence_handles
        ),
        "reason": reason,
    }


def _assert_value_error(
    operation: Callable[[], object],
    expected_prefix: str,
) -> None:
    """Assert the owning validator reports the observed error family."""

    with pytest.raises(ValueError) as caught:
        operation()
    assert str(caught.value).startswith(expected_prefix)


def _base_goal_bid() -> dict[str, Any]:
    """Return a complete ordinary goal draft for goal-owned defects."""

    return {
        "intention": "bounded intention",
        "desired_outcome": "bounded outcome",
        "concrete_detail": "bounded detail",
        "reason": "bounded reason",
        "private_monologue": "bounded monologue",
        "target_role_handles": ["self"],
        "evidence_handles": ["e1"],
        "expected_consequences": ["bounded consequence"],
        "confidence": "high",
    }


def _base_willingness(
    evidence_handles: list[str] | None = None,
) -> dict[str, Any]:
    """Return a valid relational-willingness decision."""

    return {
        "schema_version": "relational_willingness.v2",
        "applicability": "relationship_sensitive",
        "stance": "reject",
        "current_user_relationship_state": "unestablished",
        "reason": _WILLINGNESS_REASON,
        "evidence_handles": (
            ["e1"] if evidence_handles is None else evidence_handles
        ),
    }


def test_trace_inventory_contains_all_observed_contract_families() -> None:
    """Keep the test matrix synchronized with the frozen day-wide inventory."""

    if not _TRACE_INVENTORY_PATH.exists():
        pytest.skip("day-wide trace inventory is not present")
    inventory = json.loads(_TRACE_INVENTORY_PATH.read_text(encoding="utf-8"))
    observed = {
        row["family"]
        for row in inventory["capsule_validation_error_families"]
    }
    assert observed == _EXPECTED_TRACE_FAMILIES


def test_trace_inventory_contains_all_failed_stage_and_runtime_groups(
) -> None:
    """Keep non-capsule failures attached to the rebuilt test collection."""

    if not _TRACE_INVENTORY_PATH.exists():
        pytest.skip("day-wide trace inventory is not present")
    inventory = json.loads(_TRACE_INVENTORY_PATH.read_text(encoding="utf-8"))
    observed_stages = {
        row["stage_name"] for row in inventory["failed_step_groups"]
    }
    assert observed_stages == _EXPECTED_FAILED_STAGE_GROUPS
    observed_events = {
        (
            row["event_type"],
            row["component"],
            row["status"],
            row["error_class"],
            row["error_preview"],
            row["stack_fingerprint"],
        )
        for row in inventory["non_success_event_groups"]
    }
    assert observed_events == _EXPECTED_NON_SUCCESS_EVENT_GROUPS


def test_selected_evidence_unknown_handle_is_rejected() -> None:
    """The semantic owner rejects evidence handles outside its prompt."""

    question = _semantic_question()
    result = _empty_semantic_result()
    result["selected_evidence_handles"] = ["e9"]
    _assert_value_error(
        lambda: validate_semantic_appraisal_result(
            result,
            question,
            {"e1", "e2"},
            _semantic_handle_refs(),
        ),
        "selected evidence contains unknown handles",
    )


def test_selected_roles_unknown_handle_is_rejected() -> None:
    """The semantic owner rejects role handles outside its question."""

    question = _semantic_question()
    result = _empty_semantic_result()
    result["selected_role_handles"] = ["r9"]
    _assert_value_error(
        lambda: validate_semantic_appraisal_result(
            result,
            question,
            {"e1", "e2"},
            _semantic_handle_refs(),
        ),
        "selected roles contains unknown handles",
    )


def test_semantic_delta_path_not_owned_is_rejected() -> None:
    """The semantic owner enforces the question's delta-path allowlist."""

    question = _semantic_question()
    result = _empty_semantic_result()
    result["deltas"] = [_semantic_delta(
        target_path="relationship.r1.axes.boundary_pressure",
    )]
    _assert_value_error(
        lambda: validate_semantic_appraisal_result(
            result,
            question,
            {"e1", "e2"},
            _semantic_handle_refs(),
        ),
        "semantic delta path 'relationship.r1.axes.boundary_pressure' "
        "is not owned by question",
    )


def test_candidate_origin_evidence_missing_is_rejected() -> None:
    """Candidate entities must retain their originating evidence binding."""

    question = _semantic_question(
        roles=["ce1", "current_user", "self"],
        paths=["active_events.ce1.intentionality"],
    )
    result = _empty_semantic_result()
    result["propositions"] = [_semantic_proposition(
        proposition_kind="intentionality",
        subject_handle="ce1",
        evidence_handles=[],
        entity_handle="current_user",
    )]
    _assert_value_error(
        lambda: validate_semantic_appraisal_result(
            result,
            question,
            {"e1", "e2"},
            _semantic_handle_refs(),
        ),
        "causal candidates must cite originating evidence: ce1->e1",
    )


def test_semantic_role_value_invalid_is_rejected() -> None:
    """Semantic role assignments accept only the frozen role vocabulary."""

    question = _semantic_question()
    result = _empty_semantic_result()
    result["selected_evidence_handles"] = ["e1"]
    result["propositions"] = [_semantic_proposition(role="unsupported_role")]
    _assert_value_error(
        lambda: validate_semantic_appraisal_result(
            result,
            question,
            {"e1", "e2"},
            _semantic_handle_refs(),
        ),
        "semantic role value is invalid",
    )


def test_semantic_proposition_subject_kind_mismatch_is_rejected() -> None:
    """Terminal proposition kinds must address their owned entity kind."""

    question = _semantic_question(
        question_kind="goal_threat_outcome",
        question_id="q:goal_threat_outcome",
        roles=["ce1", "current_user", "self"],
    )
    result = _empty_semantic_result("q:goal_threat_outcome")
    result["propositions"] = [_semantic_proposition(
        proposition_kind="knowledge_answered",
        subject_handle="ce1",
        entity_handle="self",
    )]
    _assert_value_error(
        lambda: validate_semantic_appraisal_result(
            result,
            question,
            {"e1", "e2"},
            _semantic_handle_refs(),
        ),
        "semantic proposition kind requires subject kind 'knowledge_gap'; "
        "received 'event'",
    )


def test_semantic_proposition_object_handle_not_permitted_is_rejected(
) -> None:
    """Proposition object handles remain inside the question-owned roles."""

    question = _semantic_question()
    result = _empty_semantic_result()
    result["propositions"] = [_semantic_proposition(
        object_handle="r9",
    )]
    _assert_value_error(
        lambda: validate_semantic_appraisal_result(
            result,
            question,
            {"e1", "e2"},
            _semantic_handle_refs(),
        ),
        "semantic proposition object handle 'r9' is not permitted",
    )


def test_delta_reason_invalid_is_rejected() -> None:
    """Every semantic delta carries bounded non-empty reasoning text."""

    question = _semantic_question()
    result = _empty_semantic_result()
    result["selected_evidence_handles"] = ["e1"]
    result["deltas"] = [_semantic_delta(reason="")]
    _assert_value_error(
        lambda: validate_semantic_appraisal_result(
            result,
            question,
            {"e1", "e2"},
            _semantic_handle_refs(),
        ),
        "reason must be non-empty text up to 300 characters",
    )


def test_semantic_delta_type_invalid_is_rejected() -> None:
    """Semantic numeric deltas reject values outside the integer bound."""

    question = _semantic_question()
    result = _empty_semantic_result()
    result["deltas"] = [_semantic_delta(delta=41)]
    _assert_value_error(
        lambda: validate_semantic_appraisal_result(
            result,
            question,
            {"e1", "e2"},
            _semantic_handle_refs(),
        ),
        "semantic delta must be a JSON integer from -40 through 40; "
        "received int",
    )


def test_semantic_micro_appraisal_fields_not_exact_is_rejected() -> None:
    """The generation boundary accepts only the singular item contract."""

    _assert_value_error(
        lambda: _canonicalize_semantic_appraisal_item({
            "question_id": "q:relationship_social",
            "proposition": None,
        }),
        "semantic micro-appraisal fields must be exactly question_id, "
        "proposition, and delta",
    )


def test_terminal_event_transition_is_rejected() -> None:
    """A terminal event cannot be transitioned again by semantic reduction."""

    event = {
        "status": "resolved",
        "repair_need": 0,
    }
    with pytest.raises(
        CognitionStateError,
        match="terminal event cannot transition",
    ):
        transition_event(
            event,
            transition="resolved",
            evidence={"outcome_kind": "completion"},
        )


def test_resolved_knowledge_gap_transition_is_rejected() -> None:
    """A resolved knowledge gap cannot accept another semantic transition."""

    gap = {
        "status": "resolved",
        "uncertainty": 0,
    }
    with pytest.raises(
        CognitionStateError,
        match="resolved knowledge gap cannot transition",
    ):
        transition_knowledge_gap(
            gap,
            transition="resolved",
            evidence={"outcome_kind": "answer"},
        )


def test_goal_evidence_handles_not_permitted_is_rejected() -> None:
    """Goal drafts cannot cite evidence outside the admitted prompt set."""

    draft = _base_goal_bid()
    draft["evidence_handles"] = ["e9"]
    _assert_value_error(
        lambda: validate_goal_bid_draft(
            draft,
            evidence_handles={"e1"},
            role_handles={"self"},
        ),
        "evidence handles are not permitted",
    )


def test_goal_bid_fields_not_exact_is_rejected() -> None:
    """Goal generation keeps its exact field boundary."""

    draft = _base_goal_bid()
    draft["unexpected"] = "field"
    _assert_value_error(
        lambda: validate_goal_bid_draft(
            draft,
            evidence_handles={"e1"},
            role_handles={"self"},
        ),
        "goal bid draft fields are not exact",
    )


def test_goal_bid_consequences_invalid_is_rejected() -> None:
    """A goal bid must contain one to eight bounded consequences."""

    draft = _base_goal_bid()
    draft["expected_consequences"] = []
    _assert_value_error(
        lambda: validate_goal_bid_draft(
            draft,
            evidence_handles={"e1"},
            role_handles={"self"},
        ),
        "goal bid consequences are invalid",
    )


def test_goal_role_handles_not_permitted_is_rejected() -> None:
    """Goal drafts cannot target roles outside the admitted role set."""

    draft = _base_goal_bid()
    draft["target_role_handles"] = ["r9"]
    _assert_value_error(
        lambda: validate_goal_bid_draft(
            draft,
            evidence_handles={"e1"},
            role_handles={"self"},
        ),
        "role handles are not permitted",
    )


def test_relational_willingness_fields_not_exact_is_rejected() -> None:
    """The relational decision retains its exact six-field shape."""

    draft = _base_goal_bid()
    willingness = _base_willingness()
    willingness.pop("reason")
    draft["relational_willingness"] = willingness
    _assert_value_error(
        lambda: validate_goal_bid_draft(
            draft,
            evidence_handles={"e1"},
            role_handles={"self"},
            require_relational_willingness=True,
            episode_handles={"e1"},
        ),
        "relational willingness fields are not exact",
    )


def test_relational_willingness_evidence_unavailable_is_rejected() -> None:
    """Relational decisions cannot cite handles absent from the prompt."""

    _assert_value_error(
        lambda: validate_relational_willingness(
            _base_willingness(["e9"]),
            evidence_handles={"e1"},
            episode_handles={"e1"},
        ),
        "relational willingness cites an unavailable evidence handle",
    )


def test_relational_willingness_episode_evidence_missing_is_rejected() -> None:
    """Relationship-sensitive decisions must cite current-episode evidence."""

    _assert_value_error(
        lambda: validate_relational_willingness(
            _base_willingness(["e2"]),
            evidence_handles={"e1", "e2"},
            episode_handles={"e1"},
        ),
        "relational willingness must cite current episode evidence",
    )


def test_pending_resolution_invalid_is_rejected() -> None:
    """Reject pending-resolution decisions outside the action enum."""

    parsed = {
        "action_requests": [],
        "resolver_requests": [],
        "goal_resolution": "answerable_now",
        "resolver_pending_resolution": {
            "decision": "unsupported",
            "reason": "bounded reason",
        },
        "resolver_goal_progress": None,
    }
    _assert_value_error(
        lambda: _validate_action_plan_decision(
            parsed,
            bid_handles={},
            action_handles={},
            resolver_handles={},
        ),
        "pending resolution decision is invalid",
    )
