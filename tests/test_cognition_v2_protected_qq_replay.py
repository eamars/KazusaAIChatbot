"""Deterministic dispositions for the protected QQ Cognition V2 trace."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace

import pytest

from kazusa_ai_chatbot.cognition_core_v2.goal_cognition import (
    _normalize_nonowning_goal_fields,
)
from kazusa_ai_chatbot.cognition_core_v2.semantic_appraisal import (
    appraise_semantic_question,
)
from kazusa_ai_chatbot.cognition_core_v2.state_models import (
    build_acquaintance_user_state,
)
from kazusa_ai_chatbot.cognition_core_v2.state_projection import (
    PromptProjectionV2,
)
from kazusa_ai_chatbot.utils import parse_llm_json_output

TRACE_PATH = Path(__file__).resolve().parents[1] / "test_artifacts" / (
    "diagnostics/llmtrace_93482_validator_evidence.json"
)
CURRENT_TRACE_PATH = Path(__file__).resolve().parents[1] / "test_artifacts" / (
    "diagnostics/llm_trace_llmtrace_79651aa48cfd41d0a50c06343dbaa8db_"
    "20260818T004324Z.json"
)


class _CapturedCandidateLLM:
    """Replay captured raw responses through the semantic stage."""

    def __init__(self, response_texts: list[str]) -> None:
        self.response_texts = list(response_texts)
        self.calls = 0

    async def ainvoke(self, *_args: object, **_kwargs: object) -> object:
        """Return the next captured candidate and count the stage call."""

        self.calls += 1
        response_text = self.response_texts.pop(0)
        return SimpleNamespace(content=response_text)


def _current_trace_candidate() -> str:
    """Read the first failed semantic candidate from the protected export."""

    trace = json.loads(CURRENT_TRACE_PATH.read_text(encoding="utf-8"))
    for attempt in _semantic_attempts(trace):
        if attempt.get("stage_name") == (
            "semantic_appraisal.q:goal_threat_outcome.item_1"
        ):
            raw_response = attempt.get("raw_response_text")
            if isinstance(raw_response, str):
                return raw_response
    raise AssertionError("current protected trace candidate is missing")


def _current_trace_question() -> dict[str, object]:
    """Build the scoped question represented by the current failure."""

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


def _current_trace_evidence() -> list[dict[str, object]]:
    """Build the origin and resolution evidence used by the current case."""

    return [
        {
            "evidence_handle": handle,
            "evidence_ref": {
                "source_kind": "conversation",
                "source_id": f"conversation:{handle}",
                "occurred_at": "2026-08-18T00:00:00Z",
                "semantic_summary": f"Evidence row {handle}.",
            },
            "semantic_text": f"Evidence row {handle}.",
            "visible_to": ["q:goal_threat_outcome"],
            "authority": "conversation",
        }
        for handle in ("e1", "e3")
    ]


def _current_trace_projection() -> PromptProjectionV2:
    """Build canonical handles for the current protected failure."""

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
                "entity_id": "relationship:user:qq-current",
            },
            "self": {
                "scope": "character",
                "kind": "meaning",
                "entity_id": "meaning:character",
            },
        },
    )


def _captured_boundary_question() -> dict[str, object]:
    """Build the scoped question that rejects the captured target path."""

    return {
        "question_id": "q:goal_threat_outcome",
        "question_kind": "goal_threat_outcome",
        "semantic_question": "Assess explicit goal and event outcomes.",
        "evidence_handles": ["e6", "e7"],
        "permitted_role_handles": [
            "ev1",
            "k7",
            "current_user",
            "self",
        ],
        "permitted_role_assignment_handles": ["current_user", "self"],
        "permitted_delta_paths": ["active_events.ev1.outcome_impact"],
        "dependencies": [],
    }


def _captured_boundary_evidence() -> list[dict[str, object]]:
    """Build the captured evidence handles used by the replay candidate."""

    return [
        {
            "evidence_handle": "e6",
            "evidence_ref": {
                "source_kind": "conversation_evidence",
                "source_id": "conversation:e6",
                "occurred_at": "2026-08-08T00:00:00Z",
                "semantic_summary": "The prior evidence contains an answer.",
            },
            "semantic_text": "The prior evidence contains an answer.",
            "visible_to": ["q:goal_threat_outcome"],
            "authority": "conversation_evidence",
        },
        {
            "evidence_handle": "e7",
            "evidence_ref": {
                "source_kind": "conversation_evidence",
                "source_id": "conversation:e7",
                "occurred_at": "2026-08-08T00:00:00Z",
                "semantic_summary": "The second evidence row confirms the answer.",
            },
            "semantic_text": "The second evidence row confirms the answer.",
            "visible_to": ["q:goal_threat_outcome"],
            "authority": "conversation_evidence",
        },
    ]


def _captured_boundary_projection() -> PromptProjectionV2:
    """Build private handles matching the captured question domain."""

    return PromptProjectionV2(
        payload={
            "events": [{
                "handle": "ev1",
                "description": "A current event.",
            }],
            "knowledge_gaps": [{
                "handle": "k7",
                "description": "An answered knowledge gap.",
            }],
        },
        handle_to_ref={
            "ev1": {
                "scope": "user",
                "kind": "event",
                "entity_id": "event:ev1",
            },
            "k7": {
                "scope": "user",
                "kind": "knowledge_gap",
                "entity_id": "candidate:knowledge_gap:e6",
            },
            "current_user": {
                "scope": "user",
                "kind": "relationship",
                "entity_id": "relationship:user:qq-replay",
            },
            "self": {
                "scope": "character",
                "kind": "meaning",
                "entity_id": "meaning:character",
            },
        },
    )


def _semantic_attempts(trace: Mapping[str, object]) -> list[Mapping[str, object]]:
    """Return semantic attempts from the protected trace export."""

    attempts: list[Mapping[str, object]] = []
    capsules = trace.get("cognition_failure_capsules", [])
    if not isinstance(capsules, list):
        return attempts
    for capsule in capsules:
        if not isinstance(capsule, Mapping):
            continue
        rows = capsule.get("attempts", [])
        if not isinstance(rows, list):
            continue
        attempts.extend(
            row
            for row in rows
            if isinstance(row, Mapping)
            and str(row.get("stage_name", "")).startswith(
                "semantic_appraisal."
            )
        )
    return attempts


async def _run_current_trace_boundary() -> tuple[
    Mapping[str, object], _CapturedCandidateLLM
]:
    """Replay the current failed candidate through family termination."""

    llm = _CapturedCandidateLLM([_current_trace_candidate()])
    config = SimpleNamespace(route_name="test.current_qq_replay")
    services = SimpleNamespace(
        llm=llm,
        appraisal_event_agency_config=config,
        appraisal_relationship_social_config=config,
        appraisal_moral_identity_config=config,
        appraisal_goal_threat_outcome_config=config,
        appraisal_epistemic_comparison_memory_config=config,
        appraisal_existential_drive_config=config,
    )
    result = await appraise_semantic_question(
        _current_trace_question(),
        _current_trace_evidence(),
        _current_trace_projection(),
        services,
        validation_state=build_acquaintance_user_state(
            global_user_id="qq-current",
            updated_at="2026-08-18T00:00:00Z",
        ),
    )
    return result, llm


@pytest.mark.asyncio
async def test_current_candidate_origin_trace_terminates_family_without_retry(
) -> None:
    """The current origin failure ends its family after one model call."""

    result, llm = await _run_current_trace_boundary()

    assert llm.calls == 1
    assert result["question_id"] == "q:goal_threat_outcome"
    assert result["propositions"] == []
    assert result["deltas"] == []


@pytest.mark.asyncio
async def test_canonical_origin_candidate_is_admitted_without_preservation_guard(
) -> None:
    """Admit the corrected origin citation, then stop on the empty item."""

    candidate = parse_llm_json_output(_current_trace_candidate())
    assert isinstance(candidate, dict)
    proposition = candidate["proposition"]
    assert isinstance(proposition, dict)
    proposition["evidence_handles"] = ["e1"]
    empty_item = json.dumps({
        "question_id": "q:goal_threat_outcome",
        "proposition": None,
        "delta": None,
    })
    llm = _CapturedCandidateLLM([
        json.dumps(candidate, ensure_ascii=False),
        empty_item,
    ])
    config = SimpleNamespace(route_name="test.current_qq_correction")
    services = SimpleNamespace(
        llm=llm,
        appraisal_event_agency_config=config,
        appraisal_relationship_social_config=config,
        appraisal_moral_identity_config=config,
        appraisal_goal_threat_outcome_config=config,
        appraisal_epistemic_comparison_memory_config=config,
        appraisal_existential_drive_config=config,
    )

    result = await appraise_semantic_question(
        _current_trace_question(),
        _current_trace_evidence(),
        _current_trace_projection(),
        services,
        validation_state=build_acquaintance_user_state(
            global_user_id="qq-current-correction",
            updated_at="2026-08-18T00:00:00Z",
        ),
    )

    assert result["propositions"][0]["evidence_handles"] == ["e1"]
    assert result["deltas"] == []
    assert llm.calls == 2


def test_protected_qq_replay_semantic_appraisal_avoids_semantic_retry() -> None:
    """Replay the captured raw candidate through the boundary owner."""

    trace = json.loads(TRACE_PATH.read_text(encoding="utf-8"))
    attempts = _semantic_attempts(trace)
    target = [
        row
        for row in attempts
        if row.get("stage_name") == "semantic_appraisal.q:goal_threat_outcome.item_1"
    ]

    assert len(target) == 1
    attempt = target[0]
    assert attempt["attempt_index"] == 1
    assert isinstance(attempt.get("parsed_output"), Mapping)
    validation_error = str(attempt["validation_error"])
    assert "knowledge_gaps.k7.uncertainty" in validation_error


@pytest.mark.asyncio
async def test_captured_unowned_path_terminates_family_without_retry() -> None:
    """The captured unowned target ends its family after one model call."""

    trace = json.loads(TRACE_PATH.read_text(encoding="utf-8"))
    attempts = _semantic_attempts(trace)
    target = next(
        row
        for row in attempts
        if row.get("stage_name")
        == "semantic_appraisal.q:goal_threat_outcome.item_1"
    )
    raw_response = target.get("raw_response_text")
    assert isinstance(raw_response, str)

    llm = _CapturedCandidateLLM([raw_response])
    config = SimpleNamespace(route_name="test.protected_qq_replay")
    services = SimpleNamespace(
        llm=llm,
        appraisal_event_agency_config=config,
        appraisal_relationship_social_config=config,
        appraisal_moral_identity_config=config,
        appraisal_goal_threat_outcome_config=config,
        appraisal_epistemic_comparison_memory_config=config,
        appraisal_existential_drive_config=config,
    )
    result = await appraise_semantic_question(
        _captured_boundary_question(),
        _captured_boundary_evidence(),
        _captured_boundary_projection(),
        services,
        validation_state=build_acquaintance_user_state(
            global_user_id="qq-replay",
            updated_at="2026-08-08T00:00:00Z",
        ),
    )

    assert result["question_id"] == "q:goal_threat_outcome"
    assert result["propositions"] == []
    assert result["deltas"] == []
    assert llm.calls == 1


def test_protected_qq_replay_nonowning_goal_field_uses_structural_normalization(
) -> None:
    """Strip only the foreign non-owning goal field and record its reason."""

    normalized, records = _normalize_nonowning_goal_fields(
        {
            "intention": "hold the current boundary",
            "relational_willingness": {"unexpected": True},
        },
        branch_id="self_improvement",
        require_relational_willingness=False,
    )

    assert "relational_willingness" not in normalized
    assert records == [{
        "branch": "self_improvement",
        "field_name": "relational_willingness",
        "reason": "non_owning_branch_field",
    }]
