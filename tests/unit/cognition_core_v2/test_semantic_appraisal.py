"""Direct ownership tests for semantic appraisal projection."""

from __future__ import annotations

import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from kazusa_ai_chatbot.cognition_core_v2 import (
    semantic_appraisal as semantic_appraisal_module,
)
from kazusa_ai_chatbot.cognition_core_v2.contracts import (
    CognitionContextLimitError,
    CognitionExecutionError,
)
from kazusa_ai_chatbot.cognition_core_v2.semantic_appraisal import (
    SEMANTIC_APPRAISAL_ATTEMPT_LIMIT,
    SEMANTIC_APPRAISAL_PROMPT_CAP,
    _canonicalize_semantic_appraisal_item,
    _fit_appraisal_payload,
    _normalize_structural_semantic_appraisal_result,
    _project_question_state,
    _SemanticBoundaryValidationError,
    appraise_semantic_question,
    validate_semantic_appraisal_result,
)
from kazusa_ai_chatbot.cognition_core_v2.state_models import (
    build_acquaintance_user_state,
)
from kazusa_ai_chatbot.cognition_core_v2.state_projection import (
    PromptProjectionV2,
)


class _ScriptedSemanticLLM:
    """Return scripted semantic responses and expose exact call counts."""

    def __init__(self, responses: list[object]) -> None:
        self.responses = list(responses)
        self.calls = 0

    async def ainvoke(self, *_args: object, **_kwargs: object) -> object:
        """Return the next response or raise its scripted provider error."""

        self.calls += 1
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return SimpleNamespace(content=response)


def _execution_question() -> dict[str, object]:
    """Build a minimal event-agency question for stage execution tests."""

    return {
        "question_id": "q:event_agency",
        "question_kind": "event_agency",
        "semantic_question": "Assess the current event agency.",
        "evidence_handles": ["e1"],
        "permitted_role_handles": ["ce1", "current_user", "self"],
        "permitted_role_assignment_handles": ["current_user", "self"],
        "permitted_delta_paths": [],
        "dependencies": [],
    }


def _execution_evidence() -> list[dict[str, object]]:
    """Build one current-event evidence row for stage execution tests."""

    return [{
        "evidence_handle": "e1",
        "evidence_ref": {
            "source_kind": "episode",
            "source_id": "episode:semantic-execution",
            "occurred_at": "2026-08-18T00:00:00Z",
            "semantic_summary": "The current event supplies bounded evidence.",
        },
        "semantic_text": "The current event supplies bounded evidence.",
        "visible_to": ["q:event_agency"],
        "authority": "current_event",
    }]


def _execution_domain_question() -> dict[str, object]:
    """Build an execution question with room for one invalid evidence handle."""

    question = deepcopy(_execution_question())
    question["evidence_handles"] = ["e1", "e2"]
    return question


def _execution_domain_evidence() -> list[dict[str, object]]:
    """Build two evidence rows for a producer-handle-domain repair."""

    evidence = deepcopy(_execution_evidence())
    second = deepcopy(evidence[0])
    second["evidence_handle"] = "e2"
    second["evidence_ref"]["source_id"] = "episode:semantic-execution-2"
    second["evidence_ref"]["semantic_summary"] = (
        "The second event supplies bounded evidence."
    )
    second["semantic_text"] = "The second event supplies bounded evidence."
    evidence.append(second)
    return evidence


def _execution_projection() -> PromptProjectionV2:
    """Build the prompt projection and private handles for stage tests."""

    return PromptProjectionV2(
        payload={
            "events": [{
                "handle": "ce1",
                "description": "The current event.",
            }],
        },
        handle_to_ref={
            "ce1": {
                "scope": "user",
                "kind": "event",
                "entity_id": "candidate:event:e1",
            },
            "current_user": {
                "scope": "user",
                "kind": "relationship",
                "entity_id": "relationship:user:semantic-execution",
            },
            "self": {
                "scope": "character",
                "kind": "meaning",
                "entity_id": "meaning:character",
            },
        },
    )


def _execution_services(llm: object) -> SimpleNamespace:
    """Build the stage-specific service surface used by appraisal execution."""

    config = SimpleNamespace(route_name="test.semantic_appraisal")
    return SimpleNamespace(
        llm=llm,
        appraisal_event_agency_config=config,
        appraisal_relationship_social_config=config,
        appraisal_moral_identity_config=config,
        appraisal_goal_threat_outcome_config=config,
        appraisal_epistemic_comparison_memory_config=config,
        appraisal_existential_drive_config=config,
    )


def _empty_item_response() -> str:
    """Return the canonical empty semantic item that ends a family."""

    return json.dumps({
        "question_id": "q:event_agency",
        "proposition": None,
        "delta": None,
    })


def _origin_repair_question() -> dict[str, object]:
    """Build a knowledge-answer question with one candidate origin."""

    return {
        "question_id": "q:goal_threat_outcome",
        "question_kind": "goal_threat_outcome",
        "semantic_question": "Assess whether the knowledge gap was answered.",
        "evidence_handles": ["e1", "e3"],
        "permitted_role_handles": ["ck1", "current_user", "self"],
        "permitted_role_assignment_handles": ["current_user", "self"],
        "permitted_delta_paths": [],
        "dependencies": [],
    }


def _origin_repair_evidence() -> list[dict[str, object]]:
    """Build origin and resolution evidence for the repair regression."""

    return [
        {
            "evidence_handle": "e1",
            "evidence_ref": {
                "source_kind": "conversation",
                "source_id": "conversation:e1",
                "occurred_at": "2026-08-18T00:00:00Z",
                "semantic_summary": "The original knowledge question was asked.",
            },
            "semantic_text": "The original knowledge question was asked.",
            "visible_to": ["q:goal_threat_outcome"],
            "authority": "conversation",
        },
        {
            "evidence_handle": "e3",
            "evidence_ref": {
                "source_kind": "conversation",
                "source_id": "conversation:e3",
                "occurred_at": "2026-08-18T00:01:00Z",
                "semantic_summary": "The knowledge question received an answer.",
            },
            "semantic_text": "The knowledge question received an answer.",
            "visible_to": ["q:goal_threat_outcome"],
            "authority": "conversation",
        },
    ]


def _origin_repair_projection() -> PromptProjectionV2:
    """Build canonical handles for the candidate-origin repair case."""

    return PromptProjectionV2(
        payload={
            "knowledge_gaps": [{
                "handle": "ck1",
                "description": "The current knowledge gap.",
            }],
        },
        handle_to_ref={
            "ck1": {
                "scope": "user",
                "kind": "knowledge_gap",
                "entity_id": "candidate:knowledge_gap:e1",
            },
            "current_user": {
                "scope": "user",
                "kind": "relationship",
                "entity_id": "relationship:user:semantic-origin",
            },
            "self": {
                "scope": "character",
                "kind": "meaning",
                "entity_id": "meaning:character",
            },
        },
    )


def _origin_repair_item(evidence_handles: list[str]) -> str:
    """Build one scripted knowledge-answer candidate."""

    return json.dumps({
        "question_id": "q:goal_threat_outcome",
        "proposition": {
            "proposition_kind": "knowledge_answered",
            "subject_handle": "ck1",
            "evidence_handles": evidence_handles,
            "role_assignments": [
                {"role": "actor", "entity_handle": "self"},
                {"role": "target", "entity_handle": "current_user"},
            ],
            "semantic_value": "The knowledge question has an answer.",
        },
        "delta": None,
    })


async def _run_execution_appraisal(
    llm: object,
    *,
    question: dict[str, object] | None = None,
    evidence: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    """Run one direct semantic-appraisal execution against a scripted model."""

    return await appraise_semantic_question(
        question or _execution_question(),
        evidence or _execution_evidence(),
        _execution_projection(),
        _execution_services(llm),
        validation_state=build_acquaintance_user_state(
            global_user_id="semantic-execution",
            updated_at="2026-08-18T00:00:00Z",
        ),
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


@pytest.mark.asyncio
async def test_provider_failure_exhausts_the_stage_attempt_budget() -> None:
    """Classify provider failure after the bounded real stage calls."""

    llm = _ScriptedSemanticLLM([
        RuntimeError("provider unavailable")
        for _ in range(SEMANTIC_APPRAISAL_ATTEMPT_LIMIT)
    ])

    with pytest.raises(CognitionExecutionError) as error_info:
        await _run_execution_appraisal(llm)

    assert error_info.value.error_code == (
        "semantic_appraisal_provider_exhausted"
    )
    assert error_info.value.attempt_count == SEMANTIC_APPRAISAL_ATTEMPT_LIMIT
    assert llm.calls == SEMANTIC_APPRAISAL_ATTEMPT_LIMIT


@pytest.mark.asyncio
async def test_prompt_cap_stops_before_the_first_provider_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preserve the zero-call disposition when prompt fitting hits the cap."""

    def reject_prompt(*_args: object, **_kwargs: object) -> object:
        """Raise the external prompt-budget failure at the fitting boundary."""

        raise CognitionContextLimitError("prompt cap reached")

    monkeypatch.setattr(
        semantic_appraisal_module,
        "_fit_appraisal_payload",
        reject_prompt,
    )
    llm = _ScriptedSemanticLLM([])

    with pytest.raises(CognitionContextLimitError):
        await _run_execution_appraisal(llm)

    assert llm.calls == 0


@pytest.mark.asyncio
async def test_recoverable_structure_uses_one_replacement_then_completes() -> None:
    """Repair one malformed producer object and finish on the replacement."""

    llm = _ScriptedSemanticLLM([
        json.dumps({"unexpected": True}),
        _empty_item_response(),
    ])

    result = await _run_execution_appraisal(llm)

    assert result["question_id"] == "q:event_agency"
    assert result["propositions"] == []
    assert result["deltas"] == []
    assert llm.calls == 2


@pytest.mark.asyncio
async def test_unrecoverable_structure_exhausts_replacement_budget() -> None:
    """Retain the bounded contract disposition for repeated malformed output."""

    llm = _ScriptedSemanticLLM([
        json.dumps({"unexpected": True})
        for _ in range(SEMANTIC_APPRAISAL_ATTEMPT_LIMIT)
    ])

    with pytest.raises(CognitionExecutionError) as error_info:
        await _run_execution_appraisal(llm)

    assert error_info.value.error_code == (
        "semantic_appraisal_contract_exhausted"
    )
    assert error_info.value.attempt_count == SEMANTIC_APPRAISAL_ATTEMPT_LIMIT
    assert llm.calls == SEMANTIC_APPRAISAL_ATTEMPT_LIMIT


@pytest.mark.asyncio
async def test_candidate_origin_mismatch_omits_family_without_retry() -> None:
    """A candidate-origin mismatch ends only this family after one call."""

    question = _origin_repair_question()
    state = build_acquaintance_user_state(
        global_user_id="semantic-origin",
        updated_at="2026-08-18T00:00:00Z",
    )
    original_state = deepcopy(state)
    llm = _ScriptedSemanticLLM([_origin_repair_item(["e3"])])

    result = await appraise_semantic_question(
        question,
        _origin_repair_evidence(),
        _origin_repair_projection(),
        _execution_services(llm),
        validation_state=state,
    )

    assert result["question_id"] == "q:goal_threat_outcome"
    assert result["propositions"] == []
    assert result["deltas"] == []
    assert llm.calls == 1
    assert state == original_state


@pytest.mark.asyncio
async def test_boundary_rejection_after_accepted_item_retains_prefix_without_retry(
) -> None:
    """Keep a valid prefix when the next item fails a carrier boundary."""

    question = _origin_repair_question()
    accepted_item = _origin_repair_item(["e1"])
    rejected_item = _origin_repair_item(["e1", "e9"])
    llm = _ScriptedSemanticLLM([accepted_item, rejected_item])
    state = build_acquaintance_user_state(
        global_user_id="semantic-prefix",
        updated_at="2026-08-18T00:00:00Z",
    )
    original_state = deepcopy(state)

    result = await appraise_semantic_question(
        question,
        _origin_repair_evidence(),
        _origin_repair_projection(),
        _execution_services(llm),
        validation_state=state,
    )

    assert result["question_id"] == "q:goal_threat_outcome"
    assert len(result["propositions"]) == 1
    assert result["propositions"][0]["evidence_handles"] == ["e1"]
    assert result["deltas"] == []
    assert llm.calls == 2
    assert state == original_state


@pytest.mark.asyncio
async def test_handle_domain_violation_omits_family_without_retry() -> None:
    """An unknown evidence handle ends only this family after one call."""

    question = _execution_domain_question()
    initial = {
        "question_id": question["question_id"],
        "proposition": {
            "proposition_kind": "intentionality",
            "subject_handle": "ce1",
            "evidence_handles": ["e1", "e9"],
            "role_assignments": [{
                "role": "actor",
                "entity_handle": "current_user",
            }],
            "semantic_value": "The event appears intentional.",
        },
        "delta": None,
    }
    llm = _ScriptedSemanticLLM([json.dumps(initial)])

    result = await _run_execution_appraisal(
        llm,
        question=question,
        evidence=_execution_domain_evidence(),
    )

    assert result["question_id"] == "q:event_agency"
    assert result["propositions"] == []
    assert result["deltas"] == []
    assert llm.calls == 1


@pytest.mark.asyncio
async def test_unknown_role_enum_omits_family_without_retry() -> None:
    """An unknown role enum ends one family without state mutation."""

    question = _execution_question()
    state = build_acquaintance_user_state(
        global_user_id="semantic-role-enum",
        updated_at="2026-08-18T00:00:00Z",
    )
    original_state = deepcopy(state)
    candidate = json.dumps({
        "question_id": question["question_id"],
        "proposition": {
            "proposition_kind": "intentionality",
            "subject_handle": "ce1",
            "evidence_handles": ["e1"],
            "role_assignments": [{
                "role": "unknown_role",
                "entity_handle": "current_user",
            }],
            "semantic_value": "The event appears intentional.",
        },
        "delta": None,
    })
    llm = _ScriptedSemanticLLM([candidate])

    result = await appraise_semantic_question(
        question,
        _execution_evidence(),
        _execution_projection(),
        _execution_services(llm),
        validation_state=state,
    )

    assert result == {
        "question_id": "q:event_agency",
        "selected_evidence_handles": [],
        "selected_role_handles": [],
        "propositions": [],
        "deltas": [],
        "explanation": "No additional supported semantic item.",
    }
    assert llm.calls == 1
    assert state == original_state


@pytest.mark.asyncio
async def test_unmapped_boundary_error_propagates_without_structural_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unexpected boundary defects propagate after their first model call."""

    def raise_unmapped_boundary_error(
        *_args: object,
        **_kwargs: object,
    ) -> object:
        raise ValueError("injected unmapped boundary defect")

    monkeypatch.setattr(
        semantic_appraisal_module,
        "_validate_semantic_boundary_candidate",
        raise_unmapped_boundary_error,
    )
    llm = _ScriptedSemanticLLM([_empty_item_response()])

    with pytest.raises(
        ValueError,
        match="injected unmapped boundary defect",
    ):
        await _run_execution_appraisal(llm)

    assert llm.calls == 1


@pytest.mark.asyncio
async def test_unowned_delta_is_blocked_without_retry() -> None:
    """An unowned target path ends only this family after one call."""

    llm = _ScriptedSemanticLLM([json.dumps({
        "question_id": "q:event_agency",
        "proposition": None,
        "delta": {
            "target_path": "knowledge_gaps.k7.uncertainty",
            "delta": 1,
            "evidence_handles": ["e1"],
            "reason": "The evidence supports a bounded change.",
        },
    })])

    result = await _run_execution_appraisal(llm)

    assert result["question_id"] == "q:event_agency"
    assert result["propositions"] == []
    assert result["deltas"] == []
    assert llm.calls == 1


@pytest.mark.asyncio
async def test_runtime_admission_does_not_call_strict_semantic_validator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Runtime admission stays independent from the strict contract helper."""

    def raise_if_called(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("strict semantic validator was invoked")

    monkeypatch.setattr(
        semantic_appraisal_module,
        "validate_semantic_appraisal_result",
        raise_if_called,
    )

    candidate = json.dumps({
        "question_id": "q:event_agency",
        "proposition": {
            "proposition_kind": "unfamiliar_semantic_label",
            "subject_handle": "ce1",
            "evidence_handles": ["e1"],
            "role_assignments": [{
                "role": "actor",
                "entity_handle": "current_user",
            }],
            "semantic_value": "The authored meaning remains opaque.",
        },
        "delta": None,
    })
    llm = _ScriptedSemanticLLM([candidate, _empty_item_response()])

    result = await _run_execution_appraisal(llm)

    assert result["propositions"][0]["proposition_kind"] == (
        "unfamiliar_semantic_label"
    )
    assert llm.calls == 2


def test_boundary_failure_metadata_is_typed_without_raw_output() -> None:
    """Typed boundary errors expose fields needed by metadata-only capture."""

    error = _SemanticBoundaryValidationError(
        "semantic delta path is not owned by question",
        failure_kind="semantic_boundary_terminal",
        field_path="delta.target_path",
    )

    assert error.failure_kind == "semantic_boundary_terminal"
    assert error.field_path == "delta.target_path"


@pytest.mark.asyncio
async def test_structurally_usable_semantic_content_passes_runtime_without_retry(
) -> None:
    """Admit a structurally valid unfamiliar proposition label unchanged."""

    candidate = json.dumps({
        "question_id": "q:event_agency",
        "proposition": {
            "proposition_kind": "unfamiliar_semantic_label",
            "subject_handle": "ce1",
            "evidence_handles": ["e1"],
            "role_assignments": [],
            "semantic_value": "The authored meaning remains opaque.",
        },
        "delta": None,
    })
    llm = _ScriptedSemanticLLM([candidate, _empty_item_response()])

    result = await _run_execution_appraisal(llm)

    assert result["propositions"][0]["proposition_kind"] == (
        "unfamiliar_semantic_label"
    )
    assert llm.calls == 2


def test_recoverable_structure_does_not_trigger_producer_retry() -> None:
    """A canonical producer object continues through structural admission."""

    canonical = _canonicalize_semantic_appraisal_item({
        "question_id": "q:event_agency",
        "proposition": None,
        "delta": None,
    })
    normalized = _normalize_structural_semantic_appraisal_result(
        canonical,
        question_id="q:event_agency",
        maximum_propositions=1,
        maximum_deltas=1,
        maximum_explanation_chars=120,
    )

    assert normalized["propositions"] == []
    assert normalized["deltas"] == []


def test_unrecoverable_structure_retries_within_appraisal_budget() -> None:
    """Keep unrecoverable producer structure inside the existing cap."""

    with pytest.raises(ValueError, match="fields are not exact"):
        _normalize_structural_semantic_appraisal_result(
            {
                "question_id": "q:event_agency",
                "unexpected": True,
            },
            question_id="q:event_agency",
            maximum_propositions=1,
            maximum_deltas=1,
            maximum_explanation_chars=120,
        )

    assert SEMANTIC_APPRAISAL_ATTEMPT_LIMIT > 1


def test_provider_failure_preserves_appraisal_attempt_cap() -> None:
    """Provider failures retain the bounded appraisal owner cap."""

    assert SEMANTIC_APPRAISAL_ATTEMPT_LIMIT >= 1


def test_prompt_cap_preserves_zero_call_disposition() -> None:
    """The prompt cap remains explicit before any provider call."""

    assert SEMANTIC_APPRAISAL_PROMPT_CAP > 0


def test_boundary_validation_preserves_existing_rejection_behavior() -> None:
    """Boundary ownership rejects unauthorized structured targets directly."""

    question = _execution_question()
    parsed = {
        "question_id": question["question_id"],
        "selected_evidence_handles": ["e1"],
        "selected_role_handles": [],
        "propositions": [],
        "deltas": [{
            "target_path": "knowledge_gaps.k7.uncertainty",
            "delta": 1,
            "evidence_handles": ["e1"],
            "reason": "The evidence supports a bounded change.",
        }],
        "explanation": "The candidate uses an unowned path.",
    }

    with pytest.raises(_SemanticBoundaryValidationError) as error_info:
        semantic_appraisal_module._validate_semantic_boundary_candidate(
            parsed,
            None,
            question,
            question,
            {"e1"},
            _execution_projection().handle_to_ref,
        )

    assert error_info.value.failure_kind == "semantic_boundary_terminal"
