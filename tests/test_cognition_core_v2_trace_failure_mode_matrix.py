"""Regression matrix for trace-database failure families."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Mapping
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from kazusa_ai_chatbot.cognition_core_v2 import facade
from kazusa_ai_chatbot.cognition_core_v2.action_selection import (
    _validate_action_plan_decision,
)
from kazusa_ai_chatbot.cognition_core_v2.contracts import (
    validate_cognition_core_input,
    validate_relational_willingness,
)
from kazusa_ai_chatbot.cognition_core_v2.diagnostics import (
    write_diagnostic_artifact,
)
from kazusa_ai_chatbot.cognition_core_v2.goal_cognition import (
    CONTINUITY_AUTHORITY_INSTRUCTIONS,
    GOAL_COGNITION_PROMPT,
    _fit_goal_prompt_payload,
    validate_goal_bid_draft,
)
from kazusa_ai_chatbot.cognition_core_v2.parallel_executor import (
    ParallelExecutionResult,
)
from kazusa_ai_chatbot.cognition_core_v2.semantic_appraisal import (
    _canonicalize_semantic_appraisal_item,
    validate_semantic_appraisal_result,
)
from kazusa_ai_chatbot.cognition_core_v2.state_models import (
    CognitionStateError,
    validate_cognition_state,
)
from kazusa_ai_chatbot.cognition_core_v2.transition_guards import (
    transition_event,
    transition_knowledge_gap,
)
from kazusa_ai_chatbot.conversation_progress.projection import (
    _progress_age_descriptor,
)

_ROOT = Path(__file__).resolve().parents[1]
_TRACE_INVENTORY_PATH = (
    _ROOT
    / "test_artifacts"
    / "diagnostics"
    / "trace_failure_mode_inventory_2026-08-04.json"
)
_CAPACITY_REPLAY_EXPORTS = {
    "8d0d4295": _ROOT
    / "test_artifacts"
    / "diagnostics"
    / "llmtrace_8d0d42952b76450c9e1dc32574f9fd44_replay_export.json",
    "9164e957": _ROOT
    / "test_artifacts"
    / "diagnostics"
    / "llmtrace_9164e957298e4cffb68db7911bcd28b1_replay_export.json",
}

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


def test_captured_s8_group_noise_replay_keeps_injury_foreground() -> None:
    """The all-lane S8 replay keeps crisis evidence above stale competition."""

    fixture_path = (
        _ROOT / "tests" / "fixtures" / "qq_group_topic_continuity_regression.json"
    )
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    s8 = next(case for case in fixture["cases"] if case["case_id"] == "S8")

    evidence = [
        {
            "handle": "episode_current",
            "semantic_text": s8["current_event"],
            "authority": "current_event",
        },
        {
            "handle": "public_scene",
            "semantic_text": "；".join(s8["public_scene"]),
            "authority": "public_scene",
        },
        {
            "handle": "participant_progress",
            "semantic_text": s8["participant_progress"],
            "authority": "participant_continuity",
        },
        {
            "handle": "private_residue",
            "semantic_text": "；".join(s8["residue"]),
            "authority": "private_motive_only",
        },
        {
            "handle": "promoted_guidance",
            "semantic_text": s8["promoted_guidance"],
            "authority": "conditional_character_guidance",
        },
    ]
    payload = {
        "branch": {"goal_kind": "ordinary_response"},
        "semantic_context": {
            "scene_context": {
                "semantic_scene": s8["current_event"],
                "public_group_scene": "；".join(s8["public_scene"]),
            },
            "private_continuity_context": "；".join(s8["residue"]),
        },
        "evidence": evidence,
    }
    system_prompt = GOAL_COGNITION_PROMPT + CONTINUITY_AUTHORITY_INSTRUCTIONS
    prompt_text = _fit_goal_prompt_payload(
        payload,
        system_prompt=system_prompt,
    )

    current_authority_handles = {
        row["handle"]
        for row in evidence
        if row["authority"] in {"current_event", "public_scene"}
    }
    non_current_authority_handles = {
        row["handle"]
        for row in evidence
        if row["authority"] in {
            "participant_continuity",
            "private_motive_only",
            "conditional_character_guidance",
        }
    }
    assert "injured participant" in prompt_text
    assert "current-user injury" in prompt_text
    assert "recovery" in prompt_text
    assert "一个当前目标" in system_prompt
    assert current_authority_handles == {"episode_current", "public_scene"}
    assert non_current_authority_handles.isdisjoint(current_authority_handles)
    assert "stale reward residue" in prompt_text
    assert "conditional_character_guidance" in prompt_text


def _semantic_question(
    *,
    question_kind: str = "relationship_social",
    question_id: str = "q:relationship_social",
    roles: list[str] | None = None,
    paths: list[str] | None = None,
    assignment_roles: list[str] | None = None,
) -> dict[str, Any]:
    """Build the smallest prompt-owned semantic question for one validator."""

    return {
        "question_id": question_id,
        "question_kind": question_kind,
        "semantic_question": "bounded semantic question",
        "evidence_handles": ["e1", "e2"],
        "permitted_role_handles": roles or ["current_user", "self", "r1"],
        "permitted_role_assignment_handles": (
            assignment_roles
            if assignment_roles is not None
            else (roles or ["current_user", "self", "r1"])
        ),
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
        assignment_roles=["current_user", "self"],
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
        assignment_roles=["current_user", "self"],
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
        "semantic delta must be a JSON integer from -10 through 10; "
        "received int",
    )


def test_relationship_delta_narrow_bound_is_enforced() -> None:
    """A real relationship path rejects deltas beyond its per-event limit."""

    question = _semantic_question(
        paths=["relationship.r1.attachment"],
    )
    rejected = _empty_semantic_result()
    rejected["selected_evidence_handles"] = ["e1"]
    rejected["deltas"] = [_semantic_delta(
        target_path="relationship.r1.attachment",
        delta=11,
    )]
    _assert_value_error(
        lambda: validate_semantic_appraisal_result(
            rejected,
            question,
            {"e1", "e2"},
            _semantic_handle_refs(),
        ),
        "semantic delta must be a JSON integer from -10 through 10; "
        "received int",
    )

    accepted = _empty_semantic_result()
    accepted["selected_evidence_handles"] = ["e1"]
    accepted["selected_role_handles"] = ["r1"]
    accepted["deltas"] = [_semantic_delta(
        target_path="relationship.r1.attachment",
        delta=10,
    )]
    validated = validate_semantic_appraisal_result(
        accepted,
        question,
        {"e1", "e2"},
        _semantic_handle_refs(),
    )
    assert validated["deltas"][0]["delta"] == 10


def test_candidate_role_assignment_handle_is_rejected() -> None:
    """A causal candidate cannot be selected as a role-assignment entity."""

    question = _semantic_question(
        question_kind="event_agency",
        question_id="q:event_agency",
        roles=["ce1", "current_user", "self"],
        paths=["active_events.ce1.intentionality"],
        assignment_roles=["current_user", "self"],
    )
    result = _empty_semantic_result("q:event_agency")
    result["selected_evidence_handles"] = ["e1"]
    result["propositions"] = [_semantic_proposition(
        proposition_kind="intentionality",
        subject_handle="ce1",
        entity_handle="ce1",
    )]
    _assert_value_error(
        lambda: validate_semantic_appraisal_result(
            result,
            question,
            {"e1", "e2"},
            _semantic_handle_refs(),
        ),
        'role_assignments[*].entity_handle must be one of '
        '["current_user", "self"]',
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


def _load_capacity_replay_input(
    short_trace_id: str,
) -> tuple[dict[str, Any], dict[str, Any], Path, str]:
    """Load one protected capacity trace and its first valid candidate."""

    export_path = _CAPACITY_REPLAY_EXPORTS[short_trace_id]
    if not export_path.exists():
        pytest.skip(f"protected replay export is missing: {export_path}")
    export = json.loads(export_path.read_text(encoding="utf-8"))
    query = export.get("query")
    assert isinstance(query, Mapping)
    expected_trace_id = query.get("trace_id")
    assert isinstance(expected_trace_id, str)
    capsules = export.get("cognition_failure_capsules")
    assert isinstance(capsules, list)
    capsule = next(
        capsule
        for capsule in capsules
        if isinstance(capsule, Mapping)
        and capsule.get("trace_id") == expected_trace_id
    )
    input_payload = capsule.get("input_payload")
    assert isinstance(input_payload, Mapping)
    replay_input = deepcopy(dict(input_payload))
    mutable_state = replay_input.get("mutable_state")
    if isinstance(mutable_state, dict):
        relationship = mutable_state.get("relationship")
        if (
            isinstance(relationship, dict)
            and "relationship_maintenance" not in relationship
        ):
            relationship["relationship_maintenance"] = {
                "schema_version": "relationship_maintenance.v1",
                "last_interaction_date_utc": None,
                "last_bonus_date_utc": None,
                "last_source_id": None,
                "processed_source_ids": [],
            }
    current_event_occurred_at = next(
        evidence_row["evidence_ref"]["occurred_at"]
        for evidence_row in replay_input["evidence"]
        if evidence_row["evidence_ref"]["source_kind"] == "episode"
    )
    authority_by_source_kind = {
        "episode": "current_event",
        "media_observation": "current_event",
        "action_result": "current_event",
        "scheduler_event": "current_event",
        "tool_result": "current_event",
        "conversation_evidence": "participant_continuity",
        "recall_evidence": "contextual_fact_only",
        "resolver_observation": "contextual_fact_only",
    }
    for evidence_row in replay_input["evidence"]:
        evidence_ref = evidence_row["evidence_ref"]
        source_kind = evidence_ref["source_kind"]
        if source_kind == "promoted_memory":
            memory_scope = evidence_row["memory_scope"]
            if memory_scope == "current_user_continuity":
                authority = "participant_continuity"
            elif memory_scope == "shared_character_or_world":
                authority = "character_world_context"
            else:
                raise AssertionError(
                    f"unsupported replay memory scope: {memory_scope}"
                )
        elif source_kind == "promoted_reflection":
            authority = (
                "conditional_character_guidance"
                if ":self_guidance:" in evidence_ref["source_id"]
                else "character_world_context"
            )
        else:
            authority = authority_by_source_kind[source_kind]
        evidence_row["authority"] = authority
        if (
            source_kind == "conversation_evidence"
            and evidence_ref["source_id"].startswith(
                "conversation-progress-event:"
            )
        ):
            occurred_at = evidence_ref["occurred_at"]
            evidence_row["temporal_provenance"] = {
                "occurred_at": occurred_at,
                "age_descriptor": _progress_age_descriptor(
                    occurred_at,
                    current_event_occurred_at,
                ),
            }
    assert all(
        "authority" in evidence_row
        for evidence_row in replay_input["evidence"]
    )
    payload = validate_cognition_core_input(replay_input)
    assert {
        (
            evidence_row["evidence_ref"]["source_kind"],
            evidence_row["authority"],
        )
        for evidence_row in payload["evidence"]
    } == {
        ("episode", "current_event"),
        ("conversation_evidence", "participant_continuity"),
        ("promoted_memory", "character_world_context"),
        ("promoted_reflection", "character_world_context"),
        ("promoted_reflection", "conditional_character_guidance"),
    }
    attempts = capsule.get("attempts")
    assert isinstance(attempts, list)
    candidate = next(
        attempt["parsed_output"]
        for attempt in attempts
        if isinstance(attempt, Mapping)
        and str(attempt.get("stage_name", "")).startswith(
            "semantic_appraisal.q:event_agency."
        )
        and attempt.get("parse_status") == "succeeded"
        and isinstance(attempt.get("parsed_output"), Mapping)
        and (
            attempt["parsed_output"].get("propositions")
            or attempt["parsed_output"].get("deltas")
        )
        and any(
            isinstance(delta, Mapping)
            and isinstance(delta.get("delta"), int)
            and abs(delta["delta"]) >= 25
            for delta in attempt["parsed_output"].get("deltas", [])
        )
    )
    assert isinstance(candidate, Mapping)
    payload_value = dict(payload)
    candidate_value = dict(candidate)
    return_value = (
        payload_value,
        candidate_value,
        export_path,
        expected_trace_id,
    )
    return return_value


def _replay_capacity_trace(
    short_trace_id: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run a captured candidate through the public cognition facade."""

    payload, candidate, export_path, source_trace_id = (
        _load_capacity_replay_input(short_trace_id)
    )
    state = validate_cognition_state(payload["mutable_state"])
    persisted_capsules: list[dict[str, Any]] = []

    async def captured_appraisal(
        *_args: Any,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        """Return the protected candidate at the semantic appraisal seam."""

        candidate_copy = deepcopy(candidate)
        return candidate_copy

    async def empty_dependency_graph(
        *_args: Any,
        **_kwargs: Any,
    ) -> ParallelExecutionResult:
        """Keep downstream branch execution deterministic and side-effect free."""

        execution = ParallelExecutionResult()
        return execution

    async def silence_action_plan(**_: Any) -> dict[str, Any]:
        """Return the canonical no-admitted-bid action decision."""

        return {
            "intention": {
                "route": "silence",
            "intention": "remain silent",
            "target_roles": [],
            "reason": "no valid admitted bid",
            "goal_continuation_ref": None,
        },
            "action_requests": [],
            "resolver_requests": [],
            "goal_resolution": "blocked",
            "resolver_pending_resolution": None,
            "resolver_goal_progress": None,
        }

    monkeypatch.setattr(
        facade,
        "plan_semantic_questions",
        lambda *_args, **_kwargs: [{
            "question_id": candidate["question_id"],
            "question_kind": "event_agency",
            "semantic_question": "Assess the captured event candidate.",
        }],
    )
    monkeypatch.setattr(
        facade,
        "appraise_semantic_question",
        captured_appraisal,
    )
    monkeypatch.setattr(
        facade,
        "execute_dependency_graph",
        empty_dependency_graph,
    )
    monkeypatch.setattr(facade, "plan_actions", silence_action_plan)
    monkeypatch.setattr(
        facade.failure_capsule,
        "LLM_TRACE_CAPTURE_MODE",
        "metadata",
    )
    monkeypatch.setattr(
        facade.llm_tracing,
        "current_trace_id",
        lambda: source_trace_id,
    )
    monkeypatch.setattr(
        facade.failure_capsule,
        "_schedule_persistence",
        lambda document: persisted_capsules.append(deepcopy(document)),
    )
    output = asyncio.run(
        facade.run_cognition(
            payload,
            SimpleNamespace(),
        )
    )
    updated = output["state_update"]["replacement_state"]
    assert len(state["active_events"]) == 32
    assert len(updated["active_events"]) <= 32
    assert output["schema_version"] == "cognition_core_output.v2"
    assert output["diagnostics"]["stage_status"]["final_reduction"] == (
        "completed"
    )
    before_event_ids = {
        event["entity_id"] for event in state["active_events"]
    }
    after_event_ids = {
        event["entity_id"] for event in updated["active_events"]
    }
    assert before_event_ids <= after_event_ids
    assert persisted_capsules
    assert len(persisted_capsules) == 1
    capsule = persisted_capsules[0]["capsule"]
    assert capsule["outcome"] == "partial_failure"
    assert capsule["exception"] is None
    question_id = str(candidate["question_id"])
    rejection_event = next(
        event
        for event in capsule["failure_events"]
        if (
            isinstance(event.get("details"), Mapping)
            and event["details"].get("failure_code")
            == "semantic_appraisal_reduction_rejected"
        )
    )
    rejection = rejection_event["details"]
    assert rejection["question_id"] == question_id
    assert rejection["failure_code"] == (
        "semantic_appraisal_reduction_rejected"
    )
    assert len(rejection["exception_text"]) <= 500
    appraisal_rows = output["cognition_observability"]["appraisals"]
    assert appraisal_rows == [{
        "question_kind": "event_agency",
        "semantic_question": "Assess the captured event candidate.",
        "status": "failed",
        "failure_code": "semantic_appraisal_reduction_rejected",
    }]
    artifact = {
        "schema_version": "cognition_failure_replay.v2",
        "source_trace_id": source_trace_id,
        "source_export": str(export_path),
        "replay_mode": "captured_candidate_through_run_cognition",
        "candidate": {
            "question_id": question_id,
            "proposition_count": len(candidate.get("propositions", [])),
            "delta_count": len(candidate.get("deltas", [])),
        },
        "accepted_prefix": {
            "question_ids": [],
            "comparison_count": 0,
        },
        "rejection": dict(rejection),
        "state_counts": {
            "active_events_before": len(state["active_events"]),
            "active_events_after": len(updated["active_events"]),
            "active_events_cap": 32,
        },
        "final_output_status": output["schema_version"],
        "failure_capsule_outcome": capsule["outcome"],
        "terminal_capsule_status": (
            "terminal_failure"
            if capsule["outcome"] == "terminal_failure"
            else "none"
        ),
    }
    artifact_path = write_diagnostic_artifact(
        f"cognition_failure_replay_{short_trace_id}",
        artifact,
        artifact_root=_ROOT / "test_artifacts" / "diagnostics",
    )
    assert artifact_path.exists()


def test_captured_trace_8d0d4295_stays_within_active_event_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The first protected capacity failure is contained by admission."""

    _replay_capacity_trace("8d0d4295", monkeypatch)


def test_captured_trace_9164e957_stays_within_active_event_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The second protected capacity failure is contained by admission."""

    _replay_capacity_trace("9164e957", monkeypatch)
