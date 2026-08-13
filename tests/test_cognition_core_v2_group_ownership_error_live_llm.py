"""Replay the captured group ownership failure through live Cognition V2."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from time import time_ns
from typing import Any

import pytest

from kazusa_ai_chatbot.utils import parse_llm_json_output
from tests.cognition_core_v2_appraisal_replay_harness import (
    _run_boundary_once,
)
from tests.test_cognition_core_v2_semantic_appraisal_exhaustion_live_llm import (
    _build_appraisal_context,
)

pytestmark = [pytest.mark.asyncio, pytest.mark.live_llm]

_ROOT = Path(__file__).resolve().parents[1]
_TRACE_PATH = (
    _ROOT
    / "test_artifacts"
    / "diagnostics"
    / "llm_trace_llmtrace_cb507d084dc64436b4a5cdc3232013b9.json"
)
_TRACE_ID = "llmtrace_cb507d084dc64436b4a5cdc3232013b9"
_STAGE_NAME = "semantic_appraisal.q:goal_threat_outcome.item_1"
_QUESTION_ID = "q:goal_threat_outcome"
_LEGACY_ERROR = "terminal proposition postcondition target is unknown"
_ROLE_ASSIGNMENT_DOMAIN_ERROR = (
    "role_assignments[*].entity_handle must be one of"
)
_ARTIFACT_ROOT = (
    _ROOT / "test_artifacts" / "llm_debug" / "group_ownership_error"
)


def _load_captured_failure() -> tuple[dict[str, Any], str, str]:
    """Load the protected input and exact failed appraisal candidate."""

    if not _TRACE_PATH.exists():
        raise AssertionError(
            f"fetched trace export is missing: {_TRACE_PATH}"
        )
    trace_bytes = _TRACE_PATH.read_bytes()
    trace = json.loads(trace_bytes.decode("utf-8"))
    query = trace.get("query")
    if not isinstance(query, Mapping) or query.get("trace_id") != _TRACE_ID:
        raise AssertionError("trace export query does not match the requested id")
    capsules = [
        capsule
        for capsule in trace.get("cognition_failure_capsules", [])
        if (
            isinstance(capsule, Mapping)
            and capsule.get("trace_id") == _TRACE_ID
        )
    ]
    if len(capsules) != 1:
        raise AssertionError("trace export must contain one matching failure capsule")
    input_payload = capsules[0].get("input_payload")
    if not isinstance(input_payload, dict):
        raise TypeError("captured Cognition input is not an object")
    failures = capsules[0].get("failure_events")
    if not isinstance(failures, list) or not any(
        isinstance(failure, Mapping)
        and failure.get("stage_name") == "semantic_appraisal_reduction"
        and isinstance(failure.get("details"), Mapping)
        and failure["details"].get("exception_text") == _LEGACY_ERROR
        for failure in failures
    ):
        raise AssertionError(
            "captured trace does not contain the expected reduction failure"
        )
    attempts = [
        attempt
        for attempt in capsules[0].get("attempts", [])
        if (
            isinstance(attempt, Mapping)
            and attempt.get("stage_name") == _STAGE_NAME
        )
    ]
    if len(attempts) != 1:
        raise AssertionError("captured trace must contain one target appraisal")
    raw_response = attempts[0].get("raw_response_text")
    if not isinstance(raw_response, str) or not raw_response.strip():
        raise AssertionError("captured target appraisal output is missing")
    parsed = parse_llm_json_output(
        raw_response,
        deterministic_only=True,
    )
    if not isinstance(parsed, Mapping):
        raise TypeError("captured appraisal output is not an object")
    proposition = parsed.get("proposition")
    delta = parsed.get("delta")
    if (
        not isinstance(proposition, Mapping)
        or proposition.get("proposition_kind") != "event_completed"
        or proposition.get("subject_handle") != "ce1"
        or not isinstance(delta, Mapping)
        or delta.get("target_path") != "active_events.ce1.outcome_impact"
    ):
        raise AssertionError(
            "captured appraisal candidate does not preserve the ownership case"
        )
    return (
        input_payload,
        raw_response,
        hashlib.sha256(trace_bytes).hexdigest(),
    )


def _write_case_artifact(payload: Mapping[str, Any]) -> Path:
    """Write raw structured evidence for post-run inspection."""

    _ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    artifact_path = _ARTIFACT_ROOT / f"group_ownership_error_{time_ns()}.json"
    artifact_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    return artifact_path


def _assert_repaired_boundary(candidate_run: Mapping[str, Any]) -> None:
    """Verify bounded contract repair followed by a valid group reduction."""

    output = candidate_run["output"]
    if not isinstance(output, Mapping):
        raise TypeError("replay output is not an object")
    observability = output.get("cognition_observability")
    if not isinstance(observability, Mapping):
        raise TypeError("cognition observability is not an object")
    appraisals = observability.get("appraisals")
    if not isinstance(appraisals, list):
        raise TypeError("appraisal observability rows are not a list")
    target_rows = [
        row
        for row in appraisals
        if (
            isinstance(row, Mapping)
            and row.get("question_kind") == "goal_threat_outcome"
        )
    ]
    if len(target_rows) != 1:
        raise AssertionError(
            "the replay must expose one target appraisal row"
        )
    assert target_rows[0].get("status") == "completed", (
        "target appraisal did not complete; failure_code="
        f"{target_rows[0].get('failure_code')}"
    )
    assert not target_rows[0].get("failure_code")
    assert candidate_run["run_error"] is None
    persisted_capsules = candidate_run["persisted_capsules"]
    if not isinstance(persisted_capsules, list):
        raise TypeError("persisted failure capsules are not a list")
    for document in persisted_capsules:
        capsule = document.get("capsule")
        if not isinstance(capsule, Mapping):
            continue
        failure_events = capsule.get("failure_events")
        if not isinstance(failure_events, list):
            continue
        assert not any(
            isinstance(failure, Mapping)
            and failure.get("failure_kind")
            == "semantic_appraisal_reduction_failure"
            and isinstance(failure.get("details"), Mapping)
            and failure["details"].get("question_id") == _QUESTION_ID
            and failure["details"].get("failure_code")
            == "semantic_appraisal_reduction_rejected"
            and failure["details"].get("exception_text") == _LEGACY_ERROR
            for failure in failure_events
        ), "the repaired run must not reproduce the stale-target failure"


async def test_live_group_ownership_error_repairs_terminal_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Repair the captured group terminal-target failure with a live owner."""

    input_payload, raw_response, source_sha256 = _load_captured_failure()
    episode = input_payload.get("episode")
    if not isinstance(episode, Mapping):
        raise TypeError("captured episode is missing")
    target_scope = episode.get("target_scope")
    if not isinstance(target_scope, Mapping):
        raise TypeError("captured target scope is missing")
    if target_scope.get("channel_type") != "group":
        raise AssertionError("captured reproduction is not a group case")
    percepts = episode.get("percepts")
    if not isinstance(percepts, list) or not percepts:
        raise AssertionError("captured group percept is missing")
    dialog_content = percepts[0].get("content")
    if not isinstance(dialog_content, Mapping):
        raise TypeError("captured dialog percept is invalid")
    operation = dialog_content.get("response_operation")
    if not isinstance(operation, Mapping):
        raise TypeError("captured response operation is missing")
    assert operation.get("response_owner_role") == "当前角色"
    assert operation.get("embedded_actor_role") == "当前用户"
    assert operation.get("embedded_target_role") == "其他参与者"
    assert operation.get("selection_required") is False
    addressed_users = target_scope.get("target_addressed_user_ids")
    assert isinstance(addressed_users, list)
    assert len(addressed_users) >= 2

    _, _, _, questions = _build_appraisal_context(input_payload)
    question = next(
        question
        for question in questions
        if question.get("question_id") == _QUESTION_ID
    )
    assignment_handles = set(
        question["permitted_role_assignment_handles"]
    )
    assert {"p1", "p2", "p3", "p4", "p5"} <= assignment_handles
    assert not any(
        handle.startswith(("ce", "ct", "ck", "ev"))
        for handle in assignment_handles
    )
    assert "ce1" in question["permitted_role_handles"]
    candidate_run = await _run_boundary_once(
        input_payload=input_payload,
        question=question,
        planned_questions=questions,
        question_id=_QUESTION_ID,
        mode="candidate",
        candidate_result=None,
        first_response_text=raw_response,
        artifact_id="group_ownership_error_terminal_target",
        monkeypatch=monkeypatch,
    )
    calls = candidate_run["candidate_calls"]
    assert calls
    assert calls[0]["response_source"] == (
        "preserved_or_controlled_candidate"
    )
    assert any(
        call.get("response_source") == "live_model"
        for call in calls
    )
    target_calls = [
        call for call in calls if call.get("question_id") == _QUESTION_ID
    ]
    assert len(target_calls) >= 2
    assert target_calls[1].get("response_source") == "live_model"
    evidence_text = json.dumps(
        {
            "capture": candidate_run["capture"],
            "persisted_capsules": candidate_run["persisted_capsules"],
        },
        ensure_ascii=False,
        default=str,
    )
    candidate_capture = candidate_run["capture"]
    artifact_path = _write_case_artifact({
        "schema_version": "group_ownership_error_live_llm.v1",
        "case_id": "group_ownership_error_terminal_target",
        "source": {
            "trace_id": _TRACE_ID,
            "trace_path": str(_TRACE_PATH),
            "trace_sha256": source_sha256,
            "stage_name": _STAGE_NAME,
        },
        "input_summary": {
            "channel_type": target_scope["channel_type"],
            "response_operation": dict(operation),
            "addressed_user_count": len(addressed_users),
        },
        "captured_candidate": {
            "question_id": _QUESTION_ID,
            "proposition_kind": "event_completed",
            "subject_handle": "ce1",
            "target_path": "active_events.ce1.outcome_impact",
            "raw_response": raw_response,
            "invalid_role_assignment": {
                "role": "target",
                "entity_handle": "ce1",
            },
        },
        "reproduction": {
            "legacy_error": _LEGACY_ERROR,
            "bounded_contract_error": _ROLE_ASSIGNMENT_DOMAIN_ERROR,
            "planned_question_ids": [
                planned_question.get("question_id")
                for planned_question in questions
            ],
            "assignment_handles": sorted(assignment_handles),
            "candidate_calls": calls,
            "capture": candidate_capture,
            "persisted_capsules": candidate_run["persisted_capsules"],
            "output": candidate_run["output"],
            "run_error": candidate_run["run_error"],
            "candidate_snapshot": candidate_run["snapshot"],
        },
    })
    assert _ROLE_ASSIGNMENT_DOMAIN_ERROR in evidence_text
    assert _LEGACY_ERROR not in evidence_text
    _assert_repaired_boundary(candidate_run)
    print(f"group ownership error artifact: {artifact_path}")
