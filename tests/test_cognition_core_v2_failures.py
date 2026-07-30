"""Checkpoint C deterministic failure and transition contracts."""

from __future__ import annotations

import asyncio
from copy import deepcopy
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest

import kazusa_ai_chatbot.cognition_core_v2.facade as facade_module
import kazusa_ai_chatbot.cognition_core_v2.surface as surface_module
import kazusa_ai_chatbot.llm_tracing as tracing
from kazusa_ai_chatbot.cognition_core_v2.contracts import (
    CognitionExecutionError,
)
from kazusa_ai_chatbot.llm_tracing import failure_capsule
from kazusa_ai_chatbot.cognition_core_v2.semantic_appraisal import (
    validate_semantic_appraisal_result,
)
from kazusa_ai_chatbot.cognition_core_v2.state_models import (
    CognitionStateError,
    build_acquaintance_user_state,
    build_character_production_state,
)
from kazusa_ai_chatbot.cognition_core_v2.state_reducers import (
    _validate_goal_supersession,
    apply_elapsed_decay,
    apply_sleep_recovery,
    apply_state_update,
    canonical_event_entity_id,
    reduce_causal_event,
)
from kazusa_ai_chatbot.cognition_core_v2.transition_guards import (
    apply_direct_fact,
    apply_semantic_deltas,
    compare_event,
    transition_event,
    transition_goal,
    transition_knowledge_gap,
    transition_threat,
)
from tests.cognition_core_v2_test_helpers import canonical_episode


def _evidence(source_kind: str = "action_result") -> dict[str, str]:
    """Build one complete typed evidence record."""

    return {
        "source_kind": source_kind,
        "source_id": "action-c",
        "occurred_at": "2026-07-14T00:00:00Z",
        "semantic_summary": "typed result evidence",
    }


def _goal(
    *,
    status: str = "pursuing",
    obstruction: int = 0,
    recoverability: int = 50,
) -> dict[str, object]:
    """Build one complete bounded goal fixture."""

    return {
        "entity_id": "goal:ordinary_response:user:user-c:root-1",
        "description": "complete the bounded fixture goal",
        "status": status,
        "goal_kind": "ordinary_response",
        "importance": 70,
        "progress": 0,
        "obstruction": obstruction,
        "expected_success": 50,
        "controllability": 50,
        "recoverability": recoverability,
        "urgency": 40,
        "salience": 70,
        "role_refs": [],
        "evidence_refs": [_evidence()],
        "created_at": "2026-07-14T00:00:00Z",
        "updated_at": "2026-07-14T00:00:00Z",
    }


def _state_with_goal(**goal_kwargs: object) -> dict[str, object]:
    """Build one user state with a single mutable goal."""

    state = build_acquaintance_user_state(
        global_user_id="user-c",
        updated_at="2026-07-14T00:00:00Z",
    )
    state["goals"] = [_goal(**goal_kwargs)]
    return state


def _fact(
    fact_kind: str,
    target_kind: str,
    entity_id: str,
    *,
    observed_progress: int | None = None,
    source_kind: str = "action_result",
) -> dict[str, object]:
    """Build one canonical typed direct fact."""

    fact: dict[str, object] = {
        "fact_id": f"fact:{fact_kind}:{entity_id}",
        "fact_kind": fact_kind,
        "target_refs": [{
            "scope": "user",
            "kind": target_kind,
            "entity_id": entity_id,
        }],
        "evidence_ref": _evidence(source_kind),
    }
    if observed_progress is not None:
        fact["observed_progress"] = observed_progress
    return fact


def _appraisal_handle_refs(
    *evidence_handles: str,
) -> dict[str, dict[str, str]]:
    """Build canonical prompt refs for direct appraisal validation tests."""

    refs = {
        "self": {"entity_id": "meaning:character"},
        "current_user": {"entity_id": "relationship:user:user-c"},
    }
    for index, evidence_handle in enumerate(evidence_handles, start=1):
        refs[f"ce{index}"] = {
            "entity_id": f"candidate:event:{evidence_handle}",
        }
    return refs


def _surface_input() -> dict[str, object]:
    """Build one valid surface input for failure-capsule integration."""

    return {
        "schema_version": "text_surface_input.v2",
        "episode": canonical_episode(content="Preserve this exact input."),
        "intention": {
            "route": "speech",
            "intention": "answer the current request",
            "target_roles": [],
            "reason": "the current request expects an answer",
        },
        "goal_resolution": "answerable_now",
        "supporting_bids": [],
        "expression_policy": {
            "visibility": "visible",
            "emotional_tone": "neutral",
            "intensity": "restrained",
            "directness": "balanced",
        },
        "semantic_affect": [],
        "permitted_action_results": [],
        "interaction_style_context": "brief and clear",
        "character_expression_context": {
            "tempo": "steady",
            "linguistic_texture": "Concise clauses.",
        },
        "visual_character_context": "reserved and attentive",
    }


@pytest.mark.asyncio
async def test_run_cognition_capsule_preserves_original_exception(
    monkeypatch,
) -> None:
    """Terminal capture re-raises the exact cognition exception object."""

    written: list[dict[str, Any]] = []
    persisted = asyncio.Event()
    expected = CognitionExecutionError("original cognition failure")

    def reject_input(payload: object) -> None:
        del payload
        raise expected

    async def insert_step(document: dict[str, Any]) -> str:
        written.append(document)
        persisted.set()
        return str(document["step_id"])

    monkeypatch.setattr(tracing, "LLM_TRACE_CAPTURE_MODE", "metadata")
    monkeypatch.setattr(
        failure_capsule,
        "LLM_TRACE_CAPTURE_MODE",
        "metadata",
    )
    monkeypatch.setattr(
        facade_module,
        "validate_cognition_core_input",
        reject_input,
    )
    monkeypatch.setattr(tracing.db_llm_tracing, "insert_trace_step", insert_step)
    trace_token = tracing.bind_trace_id("trace-terminal")
    input_payload = {
        "schema_version": "cognition_core_input.v2",
        "nested": {"value": "before"},
    }
    try:
        with pytest.raises(CognitionExecutionError) as exc_info:
            await facade_module.run_cognition(
                cast(Any, input_payload),
                cast(Any, None),
            )
    finally:
        tracing.reset_trace_id(trace_token)
    input_payload["nested"]["value"] = "after"
    await asyncio.wait_for(persisted.wait(), timeout=1)

    assert exc_info.value is expected
    assert len(written) == 1
    capsule = written[0]["capsule"]
    assert capsule["input_payload"]["nested"]["value"] == "before"
    assert capsule["outcome"] == "terminal_failure"
    assert capsule["exception"] == {
        "type": "CognitionExecutionError",
        "message": "original cognition failure",
    }


@pytest.mark.asyncio
async def test_degraded_text_surface_schedules_partial_failure_capsule(
    monkeypatch,
) -> None:
    """Recovered surface failure returns its fallback and promotes evidence."""

    written: list[dict[str, Any]] = []
    persisted = asyncio.Event()

    async def fail_content(payload: object, services: object) -> None:
        del payload
        del services
        raise CognitionExecutionError("content stage exhausted")

    async def preference(
        payload: object,
        services: object,
    ) -> tuple[list[str], list[str]]:
        del payload
        del services
        return [], []

    async def insert_step(document: dict[str, Any]) -> str:
        written.append(document)
        persisted.set()
        return str(document["step_id"])

    monkeypatch.setattr(tracing, "LLM_TRACE_CAPTURE_MODE", "metadata")
    monkeypatch.setattr(
        failure_capsule,
        "LLM_TRACE_CAPTURE_MODE",
        "metadata",
    )
    monkeypatch.setattr(
        surface_module,
        "run_content_plan_stage",
        fail_content,
    )
    monkeypatch.setattr(
        surface_module,
        "run_preference_stage",
        preference,
    )
    monkeypatch.setattr(tracing.db_llm_tracing, "insert_trace_step", insert_step)
    trace_token = tracing.bind_trace_id("trace-partial-surface")
    input_payload = _surface_input()
    try:
        output = await surface_module.run_text_surface_planning(
            cast(Any, input_payload),
            cast(Any, None),
        )
    finally:
        tracing.reset_trace_id(trace_token)
    await asyncio.wait_for(persisted.wait(), timeout=1)

    assert output == surface_module.build_degraded_text_surface(input_payload)
    assert len(written) == 1
    capsule = written[0]["capsule"]
    assert capsule["entrypoint"] == "run_text_surface_planning"
    assert capsule["outcome"] == "partial_failure"
    assert capsule["failure_events"] == [{
        "failure_kind": "degraded_surface",
        "stage_name": "run_text_surface_planning",
        "details": {
            "failed_stages": ["content_plan"],
        },
    }]


@pytest.mark.asyncio
async def test_clean_text_surface_output_has_no_capsule_write(
    monkeypatch,
) -> None:
    """A successful first attempt returns normally without capsule storage."""

    insert_step = AsyncMock()

    async def content(
        payload: object,
        services: object,
    ) -> tuple[str, list[str], dict[str, str]]:
        del payload
        del services
        return (
            "answer the current request",
            ["preserve current meaning"],
            {
                "lexical_register": "plain",
                "sentence_shape": "concise",
                "rhythm": "steady",
                "hesitation": "light",
                "punctuation": "restrained",
            },
        )

    async def preference(
        payload: object,
        services: object,
    ) -> tuple[list[str], list[str]]:
        del payload
        del services
        return [], []

    monkeypatch.setattr(tracing, "LLM_TRACE_CAPTURE_MODE", "metadata")
    monkeypatch.setattr(
        failure_capsule,
        "LLM_TRACE_CAPTURE_MODE",
        "metadata",
    )
    monkeypatch.setattr(
        surface_module,
        "run_content_plan_stage",
        content,
    )
    monkeypatch.setattr(
        surface_module,
        "run_preference_stage",
        preference,
    )
    monkeypatch.setattr(tracing.db_llm_tracing, "insert_trace_step", insert_step)
    trace_token = tracing.bind_trace_id("trace-clean-surface")
    try:
        output = await surface_module.run_text_surface_planning(
            cast(Any, _surface_input()),
            cast(Any, None),
        )
    finally:
        tracing.reset_trace_id(trace_token)
    await asyncio.sleep(0)

    assert output["content_plan"] == "answer the current request"
    assert output["content_requirements"] == ["preserve current meaning"]
    insert_step.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "entrypoint",
    [
        "repair_text_surface_planning",
        "run_visual_surface_planning",
    ],
)
async def test_remaining_surface_entrypoints_capture_terminal_failure(
    monkeypatch,
    entrypoint: str,
) -> None:
    """Repair and visual entrypoints capture before input validation."""

    written: list[dict[str, Any]] = []
    persisted = asyncio.Event()
    expected = ValueError(f"{entrypoint} input failed")

    def reject_input(payload: object) -> None:
        del payload
        raise expected

    async def insert_step(document: dict[str, Any]) -> str:
        written.append(document)
        persisted.set()
        return str(document["step_id"])

    monkeypatch.setattr(tracing, "LLM_TRACE_CAPTURE_MODE", "metadata")
    monkeypatch.setattr(
        failure_capsule,
        "LLM_TRACE_CAPTURE_MODE",
        "metadata",
    )
    monkeypatch.setattr(
        surface_module,
        "validate_text_surface_input",
        reject_input,
    )
    monkeypatch.setattr(tracing.db_llm_tracing, "insert_trace_step", insert_step)
    trace_token = tracing.bind_trace_id(f"trace-{entrypoint}")
    repair_input = {"exact": "repair input"}
    verified_issues = ["preserve the verified issue"]
    visual_input = {"exact": "visual input"}
    try:
        with pytest.raises(ValueError) as exc_info:
            if entrypoint == "repair_text_surface_planning":
                await surface_module.repair_text_surface_planning(
                    cast(Any, repair_input),
                    verified_issues,
                    cast(Any, None),
                )
            else:
                await surface_module.run_visual_surface_planning(
                    cast(Any, visual_input),
                    cast(Any, None),
                )
    finally:
        tracing.reset_trace_id(trace_token)
    await asyncio.wait_for(persisted.wait(), timeout=1)

    assert exc_info.value is expected
    assert len(written) == 1
    capsule = written[0]["capsule"]
    assert capsule["entrypoint"] == entrypoint
    assert capsule["outcome"] == "terminal_failure"
    if entrypoint == "repair_text_surface_planning":
        assert capsule["input_payload"] == {
            "input_payload": repair_input,
            "verified_hard_issues": verified_issues,
        }
    else:
        assert capsule["input_payload"] == visual_input


def test_unsupported_appraisal_can_select_no_evidence() -> None:
    """Accept an explicit no-claim result without forcing false evidence use."""

    question = {
        "question_id": "q:event_agency",
        "question_kind": "event_agency",
        "semantic_question": "Assess responsibility and intentionality.",
        "evidence_handles": ["e1"],
        "permitted_role_handles": ["ce1", "current_user", "self"],
        "permitted_delta_paths": ["active_events.ce1.intentionality"],
        "dependencies": [],
    }
    result = validate_semantic_appraisal_result(
        {
            "question_id": "q:event_agency",
            "selected_evidence_handles": [],
            "selected_role_handles": [],
            "propositions": [],
            "deltas": [],
            "explanation": "The supplied evidence does not support a claim.",
        },
        question,
        {"e1"},
        _appraisal_handle_refs("e1"),
    )

    assert result["selected_evidence_handles"] == []
    assert result["propositions"] == []
    assert result["deltas"] == []


def test_candidate_proposition_rejects_mismatched_evidence() -> None:
    """Keep a candidate proposition bound to its originating evidence row."""

    question = {
        "question_id": "q:event_agency",
        "question_kind": "event_agency",
        "semantic_question": "Assess responsibility and intentionality.",
        "evidence_handles": ["e1", "e2"],
        "permitted_role_handles": ["ce1", "ce2", "current_user", "self"],
        "permitted_delta_paths": ["active_events.ce1.intentionality"],
        "dependencies": [],
    }
    parsed = {
        "question_id": "q:event_agency",
        "selected_evidence_handles": ["e2"],
        "selected_role_handles": ["ce1"],
        "propositions": [{
            "proposition_kind": "intentionality",
            "subject_handle": "ce1",
            "evidence_handles": ["e2"],
            "role_assignments": [],
            "semantic_value": "The action appears deliberate.",
        }],
        "deltas": [],
        "explanation": "The second row does not own the first candidate.",
    }

    with pytest.raises(ValueError, match="originating evidence"):
        validate_semantic_appraisal_result(
            parsed,
            question,
            {"e1", "e2"},
            _appraisal_handle_refs("e1", "e2"),
        )


def test_candidate_delta_rejects_mismatched_evidence() -> None:
    """Keep a candidate delta bound to its originating evidence row."""

    question = {
        "question_id": "q:event_agency",
        "question_kind": "event_agency",
        "semantic_question": "Assess responsibility and intentionality.",
        "evidence_handles": ["e1", "e2"],
        "permitted_role_handles": ["ce1", "ce2", "current_user", "self"],
        "permitted_delta_paths": ["active_events.ce1.intentionality"],
        "dependencies": [],
    }
    parsed = {
        "question_id": "q:event_agency",
        "selected_evidence_handles": ["e2"],
        "selected_role_handles": [],
        "propositions": [],
        "deltas": [{
            "target_path": "active_events.ce1.intentionality",
            "delta": 20,
            "evidence_handles": ["e2"],
            "reason": "The second row does not own the first candidate.",
        }],
        "explanation": "The candidate path and evidence must stay paired.",
    }

    with pytest.raises(ValueError, match="originating evidence"):
        validate_semantic_appraisal_result(
            parsed,
            question,
            {"e1", "e2"},
            _appraisal_handle_refs("e1", "e2"),
        )


def test_candidate_binding_uses_canonical_projection_reference() -> None:
    """Resolve sparse evidence ids from projection instead of handle suffixes."""

    question = {
        "question_id": "q:event_agency",
        "question_kind": "event_agency",
        "semantic_question": "Assess responsibility and intentionality.",
        "evidence_handles": ["e7"],
        "permitted_role_handles": ["ce1", "current_user", "self"],
        "permitted_delta_paths": ["active_events.ce1.intentionality"],
        "dependencies": [],
    }
    result = validate_semantic_appraisal_result(
        {
            "question_id": "q:event_agency",
            "selected_evidence_handles": ["e7"],
            "selected_role_handles": ["ce1"],
            "propositions": [{
                "proposition_kind": "intentionality",
                "subject_handle": "ce1",
                "evidence_handles": ["e7"],
                "role_assignments": [],
                "semantic_value": "The current action appears deliberate.",
            }],
            "deltas": [],
            "explanation": "The current evidence supports intentionality.",
        },
        question,
        {"e7"},
        _appraisal_handle_refs("e7"),
    )

    assert result["propositions"][0]["subject_handle"] == "ce1"


def test_invalid_direct_fact_leaves_state_unchanged() -> None:
    """Reject an untrusted producer before any state mutation."""

    state = _state_with_goal()
    before = deepcopy(state)

    with pytest.raises(CognitionStateError):
        apply_direct_fact(
            state,
            _fact("goal_completed", "goal", state["goals"][0]["entity_id"]),
            producer="dialog_text",
        )

    assert state == before


def test_direct_fact_rejects_extra_fields_and_terminal_mutation() -> None:
    """Keep the direct-fact lane closed to invented fields and terminal rows."""

    state = _state_with_goal()
    extra = _fact("goal_completed", "goal", state["goals"][0]["entity_id"])
    extra["observed_progress"] = 100
    with pytest.raises(CognitionStateError):
        apply_direct_fact(state, extra, producer="action_result")

    terminal_state = _state_with_goal(status="satisfied")
    with pytest.raises(CognitionStateError):
        apply_direct_fact(
            terminal_state,
            _fact("goal_completed", "goal", terminal_state["goals"][0]["entity_id"]),
            producer="action_result",
        )


def test_progress_observation_at_full_progress_completes_goal() -> None:
    """Treat trusted observed progress at 100 as deterministic completion."""

    state = _state_with_goal()
    updated = apply_direct_fact(
        state,
        _fact(
            "goal_progress_observed",
            "goal",
            state["goals"][0]["entity_id"],
            observed_progress=100,
            source_kind="resolver_observation",
        ),
        producer="resolver_observation",
    )

    assert updated["goals"][0]["progress"] == 100
    assert updated["goals"][0]["status"] == "satisfied"
    assert updated["goals"][0]["evidence_refs"][1]["source_kind"] == (
        "resolver_observation"
    )


def test_direct_fact_rejects_evidence_producer_mismatch() -> None:
    """Keep evidence provenance aligned with the trusted producer."""

    state = _state_with_goal()
    fact = _fact(
        "goal_progress_observed",
        "goal",
        state["goals"][0]["entity_id"],
        observed_progress=20,
        source_kind="action_result",
    )
    with pytest.raises(CognitionStateError):
        apply_direct_fact(state, fact, producer="resolver_observation")


def test_direct_fact_source_occurrence_preserves_full_evidence() -> None:
    """Allow scheduler occurrence facts without reducing provenance to a string."""

    state = _state_with_goal()
    goal_id = state["goals"][0]["entity_id"]
    fact = _fact(
        "source_occurred",
        "goal",
        goal_id,
        source_kind="scheduler_event",
    )

    scheduled = apply_direct_fact(state, fact, producer="scheduler_event")

    assert scheduled["goals"][0]["evidence_refs"][-1] == fact["evidence_ref"]


def test_source_occurrence_can_address_one_structured_role_target() -> None:
    """Append occurrence evidence through a unique role-owned causal entity."""

    state = _state_with_goal()
    goal_id = state["goals"][0]["entity_id"]
    state["active_events"] = [{
        "entity_id": "event:role-target",
        "description": "an event carrying a target role",
        "status": "active",
        "outcome_impact": 0,
        "responsibility": 0,
        "intentionality": 0,
        "harm": 0,
        "unfairness": 0,
        "exposure": 0,
        "repair_need": 0,
        "reparability": 100,
        "expectation_mismatch": 0,
        "norm_violation": 0,
        "contamination_risk": 0,
        "identity_threat": 0,
        "comparison_gap": 0,
        "vastness": 0,
        "memory_warmth": 0,
        "temporal_loss": 0,
        "salience": 40,
        "role_refs": [{
            "role": "affected_goal",
            "entity_kind": "goal",
            "entity_id": goal_id,
        }],
        "evidence_refs": [_evidence("episode")],
        "created_at": "2026-07-14T00:00:00Z",
        "updated_at": "2026-07-14T00:00:00Z",
    }]
    fact = _fact(
        "source_occurred",
        "goal",
        goal_id,
        source_kind="scheduler_event",
    )
    fact["target_refs"] = [{
        "role": "affected_goal",
        "entity_kind": "goal",
        "entity_id": goal_id,
    }]

    updated = apply_direct_fact(state, fact, producer="scheduler_event")

    assert updated["active_events"][0]["evidence_refs"][-1] == fact["evidence_ref"]


@pytest.mark.parametrize(
    "axis",
    ["importance", "progress", "salience"],
)
def test_semantic_deltas_reject_reducer_owned_goal_axes(axis: str) -> None:
    """Keep reducer-owned goal axes outside the semantic delta lane."""

    state = _state_with_goal()
    goal_id = state["goals"][0]["entity_id"]
    with pytest.raises(CognitionStateError):
        apply_semantic_deltas(
            state,
            [{
                "target_path": f"goals.{goal_id}.{axis}",
                "delta": 10,
                "evidence_handles": ["action-c"],
                "reason": "forbidden reducer-owned update",
            }],
        )


def test_causal_event_identity_and_salience_are_reducer_owned() -> None:
    """Use the frozen evidence digest and accepted-axis magnitude on create."""

    state = build_acquaintance_user_state(
        global_user_id="user-c",
        updated_at="2026-07-14T00:00:00Z",
    )
    primary_evidence = _evidence("episode")
    event = {
        "description": "a reducer-owned causal event",
        "role_refs": [{
            "role": "actor",
            "entity_kind": "user",
            "entity_id": "user-c",
        }],
        "outcome_impact": 0,
        "responsibility": 0,
        "intentionality": 0,
        "harm": 40,
        "unfairness": 0,
        "exposure": 0,
        "repair_need": 0,
        "reparability": 100,
        "expectation_mismatch": 0,
        "norm_violation": 0,
        "contamination_risk": 0,
        "identity_threat": 0,
        "comparison_gap": 0,
        "vastness": 0,
        "memory_warmth": 0,
        "temporal_loss": 0,
        "salience": 99,
    }

    updated, outcome = reduce_causal_event(
        state,
        event,
        accepted_deltas={"harm": 30},
        primary_evidence=primary_evidence,
    )

    assert outcome == "create"
    stored = updated["active_events"][0]
    assert stored["entity_id"] == canonical_event_entity_id(state, primary_evidence)
    assert stored["salience"] == 30


def test_duplicate_semantic_targets_do_not_block_unique_targets() -> None:
    """Invalidate duplicate targets while applying unaffected unique paths."""

    state = _state_with_goal()
    target_id = state["goals"][0]["entity_id"]
    before = deepcopy(state)
    updated = apply_semantic_deltas(
        state,
        [
            {
                "target_path": f"goals.{target_id}.obstruction",
                "delta": 10,
                "evidence_handles": ["action-c"],
                "reason": "duplicate candidate one",
            },
            {
                "target_path": f"goals.{target_id}.obstruction",
                "delta": -10,
                "evidence_handles": ["action-c"],
                "reason": "duplicate candidate two",
            },
            {
                "target_path": f"goals.{target_id}.urgency",
                "delta": 10,
                "evidence_handles": ["action-c"],
                "reason": "independent urgency update",
            },
        ],
    )

    assert state == before
    assert updated["goals"][0]["obstruction"] == 0
    assert updated["goals"][0]["urgency"] == 50


def test_event_comparison_uses_refs_and_reports_unrelated_text() -> None:
    """Keep event comparison structural rather than description-driven."""

    current = {
        "entity_id": "event-current",
        "role_refs": [
            {
                "role": "actor",
                "entity_kind": "character",
                "entity_id": "character:global",
            },
            {"role": "target", "entity_kind": "user", "entity_id": "user-1"},
        ],
        "axis_deltas": {"harm": 10},
        "salience": 50,
    }
    stored = {
        "entity_id": "event-1",
        "status": "active",
        "role_refs": current["role_refs"],
        "harm": 30,
    }

    assert compare_event(current, stored, {"harm": 5}) == "reinforce"
    assert compare_event(
        {**current, "description": "same words, different refs"},
        {**stored, "role_refs": [{
            "role": "target",
            "entity_kind": "user",
            "entity_id": "user-2",
        }]},
        {"harm": 5},
    ) == "unrelated"


def test_goal_threat_event_and_gap_fsms_require_frozen_guards() -> None:
    """Allow transitions only with threshold and typed evidence guards."""

    blocked = transition_goal(_goal(obstruction=40), transition="blocked")
    assert blocked["status"] == "blocked"
    pursuing = transition_goal(
        _goal(status="blocked", obstruction=24),
        transition="pursuing",
    )
    assert pursuing["status"] == "pursuing"
    with pytest.raises(CognitionStateError):
        transition_goal(_goal(status="satisfied"), transition="pursuing")
    with pytest.raises(CognitionStateError):
        transition_goal(_goal(), transition="abandoned")

    threat = transition_threat(
        {"status": "active", "residual_pressure": 20},
        transition="resolved",
        evidence={"outcome_kind": "resolve"},
    )
    assert threat["status"] == "resolved"
    event = transition_event(
        {"status": "active", "repair_need": 0},
        transition="resolved",
        evidence={"outcome_kind": "repair"},
    )
    assert event["status"] == "resolved"
    gap = transition_knowledge_gap(
        {"status": "open", "uncertainty": 50},
        transition="reduced",
        previous_uncertainty=80,
    )
    assert gap["status"] == "reduced"
    resolved_gap = transition_knowledge_gap(
        {"status": "reduced", "uncertainty": 0},
        transition="resolved",
        previous_uncertainty=50,
        evidence={"outcome_kind": "answer"},
    )
    assert resolved_gap["status"] == "resolved"


def test_semantic_uncertainty_decrease_drives_the_frozen_gap_fsm() -> None:
    """Reduce at twenty points and resolve at the twenty-point threshold."""

    gap = {
        "entity_id": "gap:semantic",
        "status": "open",
        "uncertainty": 60,
        "evidence_refs": [_evidence()],
        "updated_at": "2026-07-14T00:00:00Z",
    }
    state = {
        "updated_at": "2026-07-14T00:00:00Z",
        "knowledge_gaps": [gap],
    }
    reduced = apply_semantic_deltas(
        state,
        [{
            "target_path": "knowledge_gaps.gap:semantic.uncertainty",
            "delta": -20,
            "evidence_handles": ["action-c"],
            "reason": "accepted evidence materially narrows the gap",
        }],
    )
    assert reduced["knowledge_gaps"][0]["status"] == "reduced"
    assert reduced["knowledge_gaps"][0]["uncertainty"] == 40

    resolved = apply_semantic_deltas(
        reduced,
        [{
            "target_path": "knowledge_gaps.gap:semantic.uncertainty",
            "delta": -20,
            "evidence_handles": ["action-c"],
            "reason": "accepted evidence reaches the resolution threshold",
        }],
    )
    assert resolved["knowledge_gaps"][0]["status"] == "resolved"
    assert resolved["knowledge_gaps"][0]["uncertainty"] == 0


def test_goal_supersession_requires_a_distinct_pursuing_goal() -> None:
    """Reject self, missing, and non-pursuing replacement goal handles."""

    old_goal = _goal()
    replacement = {**_goal(), "entity_id": "goal:replacement"}
    state = {"goals": [old_goal, replacement]}
    proposition = {
        "proposition_kind": "goal_supersession",
        "object_handle": "g2",
    }
    handle_to_ref = {
        "g1": {"kind": "goal", "entity_id": old_goal["entity_id"]},
        "g2": {"kind": "goal", "entity_id": replacement["entity_id"]},
    }

    assert _validate_goal_supersession(
        state,
        old_goal,
        proposition,
        handle_to_ref,
    )
    replacement["status"] = "blocked"
    with pytest.raises(CognitionStateError, match="pursuing goal"):
        _validate_goal_supersession(
            state,
            old_goal,
            proposition,
            handle_to_ref,
        )


def test_event_repair_axis_alone_does_not_auto_resolve() -> None:
    """Keep repair_need zero insufficient without typed completion evidence."""

    state = _state_with_goal()
    event = {
        "status": "active",
        "repair_need": 0,
    }
    with pytest.raises(CognitionStateError):
        transition_event(event, transition="resolved")
    assert state["goals"][0]["status"] == "pursuing"


def test_state_update_uses_fixed_order_and_derives_cache() -> None:
    """Run elapsed, facts, deltas, guarded lifecycle, cache, and retention."""

    state = _state_with_goal()
    goal_id = state["goals"][0]["entity_id"]
    state["active_events"] = [{
        "entity_id": "event:blocked-goal",
        "description": "the blocked goal was obstructed by an unfair action",
        "status": "active",
        "outcome_impact": 0,
        "responsibility": 0,
        "intentionality": 70,
        "harm": 0,
        "unfairness": 70,
        "exposure": 0,
        "repair_need": 0,
        "reparability": 80,
        "expectation_mismatch": 0,
        "norm_violation": 0,
        "contamination_risk": 0,
        "identity_threat": 0,
        "comparison_gap": 0,
        "vastness": 0,
        "memory_warmth": 0,
        "temporal_loss": 0,
        "salience": 70,
        "role_refs": [{
            "role": "affected_goal",
            "entity_kind": "goal",
            "entity_id": goal_id,
        }],
        "evidence_refs": [_evidence("episode")],
        "created_at": "2026-07-14T00:00:00Z",
        "updated_at": "2026-07-14T00:00:00Z",
    }]
    updated = apply_state_update(
        state,
        direct_facts=[(
            "action_result",
            _fact(
                "goal_progress_observed",
                "goal",
                goal_id,
                observed_progress=20,
            ),
        )],
        semantic_deltas=[{
            "target_path": f"goals.{goal_id}.obstruction",
            "delta": 40,
            "evidence_handles": ["action-c"],
            "reason": "the result confirms obstruction",
        }],
        elapsed_seconds=3600,
    )

    assert updated["goals"][0]["progress"] == 20
    assert updated["goals"][0]["obstruction"] == 40
    assert updated["goals"][0]["status"] == "blocked"
    assert [row["emotion_id"] for row in updated["affect_activations"]] == [
        "anger",
    ]


def test_elapsed_decay_and_sleep_recovery_are_scope_specific() -> None:
    """Apply user elapsed evolution and character sleep recovery exactly once."""

    user_state = _state_with_goal()
    user_state["affect_activations"] = [{
        "activation_id": "emotion:joy",
        "emotion_id": "joy",
        "primary_root": {
            "scope": "user",
            "kind": "goal",
            "entity_id": user_state["goals"][0]["entity_id"],
        },
        "root_refs": [{
            "scope": "user",
            "kind": "goal",
            "entity_id": user_state["goals"][0]["entity_id"],
        }],
        "phase": "active",
        "score": 60,
        "peak_score": 60,
        "trend": "stable",
        "cause_status": "active",
        "started_at": "2026-07-14T00:00:00Z",
        "updated_at": "2026-07-14T00:00:00Z",
        "last_reinforced_at": "2026-07-14T00:00:00Z",
    }]
    decayed = apply_elapsed_decay(
        user_state,
        elapsed_seconds=3600,
        rate_per_hour=4,
    )
    assert decayed["goals"][0]["salience"] == 66
    assert decayed["affect_activations"][0]["score"] == 56

    character_state = build_character_production_state(
        updated_at="2026-07-14T00:00:00Z",
    )
    character_state["goals"] = [deepcopy(user_state["goals"][0])]
    character_state["goals"][0]["role_refs"] = [{
        "role": "actor",
        "entity_kind": "character",
        "entity_id": "character:global",
    }]
    character_state["affect_activations"] = [{
        **user_state["affect_activations"][0],
        "primary_root": {
            "scope": "character",
            "kind": "goal",
            "entity_id": user_state["goals"][0]["entity_id"],
        },
        "root_refs": [{
            "scope": "character",
            "kind": "goal",
            "entity_id": user_state["goals"][0]["entity_id"],
        }],
    }]
    recovered = apply_sleep_recovery(
        character_state,
        elapsed_sleep_seconds=7200,
    )

    assert recovered["goals"][0]["salience"] == 42
    assert recovered["affect_activations"][0]["score"] == 32
