"""Deterministic boundaries for targetless group self-cognition replies."""

from __future__ import annotations

import inspect
import json
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
from pymongo.errors import DuplicateKeyError

import kazusa_ai_chatbot.db.self_cognition as db_self_cognition
import kazusa_ai_chatbot.event_logging.recording as recording_module
from kazusa_ai_chatbot import event_logging
from kazusa_ai_chatbot.cognition_core_v2.action_selection import plan_actions
from kazusa_ai_chatbot.cognition_core_v2.contracts import (
    CognitionContractError,
    validate_self_cognition_response_decision,
)
from kazusa_ai_chatbot.self_cognition import models, runner, tracking, worker


def _proposal_response(
    *,
    evidence_handles: list[str] | None = None,
    semantic_target_handle: str = "current_group_scene",
    participation_basis: str = "grounded_scene_intervention",
    response_goal: str = "answer the current group scene",
) -> dict[str, Any]:
    """Build one valid visible-reply semantic decision."""

    return {
        "decision": "propose_visible_reply",
        "evidence_handles": evidence_handles or ["e1"],
        "semantic_target_handle": semantic_target_handle,
        "participation_basis": participation_basis,
        "response_goal": response_goal,
        "reason": "The current scene supports a bounded intervention.",
    }


def _silent_response() -> dict[str, Any]:
    """Build one valid semantic silence decision."""

    return {
        "decision": "stay_silent",
        "evidence_handles": [],
        "semantic_target_handle": "",
        "participation_basis": "",
        "response_goal": "",
        "reason": "No grounded reason to intervene.",
    }


def _evidence(
    *,
    source_kind: str = "episode",
    evidence_handle: str = "e1",
) -> dict[str, Any]:
    """Build one prompt-safe evidence row."""

    row: dict[str, Any] = {
        "evidence_handle": evidence_handle,
        "evidence_ref": {
            "source_kind": source_kind,
            "source_id": "group-episode-1",
            "occurred_at": "2026-05-18T04:10:00+00:00",
            "semantic_summary": "The current group scene contains one recent message.",
        },
        "semantic_text": "The current group scene contains one recent message.",
        "visible_to": ["q:event_agency"],
        "authority": (
            "current_event"
            if source_kind == "episode"
            else "character_world_context"
        ),
    }
    if source_kind == "promoted_memory":
        row["memory_scope"] = "shared_character_or_world"
    return row


def _primary_bid() -> dict[str, Any]:
    """Build one complete admitted bid for action-planning tests."""

    return {
        "branch_id": "ordinary_response",
        "goal_ref": {"scope": "character", "kind": "goal", "entity_id": "g1"},
        "intention": "intervene in the current group scene",
        "desired_outcome": "preserve a grounded interaction",
        "concrete_detail": "answer only from the current scene",
        "reason": "The admitted evidence supports this motive.",
        "private_monologue": "The current scene may warrant a brief reply.",
        "target_roles": [],
        "evidence_handles": ["e1"],
        "expected_consequences": ["the interaction remains coherent"],
        "confidence": "high",
    }


def _group_case(
    *,
    labels: dict[str, str] | None = None,
    existing_attempts: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a valid targetless group-review case."""

    active_labels = {
        "assistant_presence": "present",
        "bot_addressing": "ambient_group_context",
        "message_recency": "recent",
        "response_risk": "low",
    }
    if labels is not None:
        active_labels.update(labels)
    source_id = (
        "scope_group:2026-05-18T04:00:00+00:00:"
        "2026-05-18T04:15:00+00:00"
    )
    case = {
        "case_name": models.CASE_GROUP_CHAT_REVIEW,
        "case_id": source_id,
        "idle_timestamp_utc": "2026-05-18T04:15:00+00:00",
        "last_evidence_timestamp_utc": "2026-05-18T04:10:00+00:00",
        "trigger_kind": models.TRIGGER_GROUP_CHAT_REVIEW,
        "semantic_due_state": None,
        "actionability": "active_group_review_same_channel_no_fallback",
        "target_scope": {
            "platform": "qq",
            "platform_channel_id": "54369546",
            "channel_type": "group",
            "user_id": None,
        },
        "source_refs": [{
            "source_kind": "reflection_activity_window",
            "source_id": source_id,
            "due_at": None,
            "summary": "A recent group activity window is ready for review.",
        }],
        "visible_context": [{
            "role": "participant",
            "body_text": "A recent group message.",
            "timestamp": "2026-05-18T04:10:00+00:00",
        }],
        "conversation_progress": None,
        "source_context": {
            "schema_version": "self_cognition_group_source_context.v1",
            "context_kind": "group_chat_review",
            "group_activity_window": {
                "source": "reflection_activity_window",
                "window_start": "2026-05-18T04:00:00+00:00",
                "window_end": "2026-05-18T04:15:00+00:00",
                "semantic_labels": active_labels,
            },
            "conversation_evidence": [],
        },
        "target_binding_status": "bound",
        "delivery_target": {
            "schema_version": "self_cognition_delivery_target.v1",
            "platform": "qq",
            "platform_channel_id": "54369546",
            "channel_type": "group",
            "user_id": None,
        },
        "existing_attempts": existing_attempts or [],
    }
    return case


def _group_cognition_output(
    response: dict[str, Any],
) -> dict[str, Any]:
    """Build the bounded cognition output consumed by group policy."""

    return {
        "cognition_core_output": {
            "intention": {
                "route": "speech",
                "intention": "intervene in the current group scene",
                "target_roles": [],
                "reason": "The current scene supports a bounded intervention.",
            },
            "admitted_bid": {"evidence_handles": ["e1"]},
            "self_cognition_response": response,
        },
        "action_specs": [],
    }


def _episode() -> dict[str, Any]:
    """Build a targetless group episode envelope."""

    return {
        "episode_id": "group-episode-1",
        "trigger_source": "self_cognition",
        "output_mode": "think_only",
        "target_scope": {
            "channel_type": "group",
            "current_global_user_id": "",
            "current_platform_user_id": "",
        },
    }


class _PlannerLLM:
    """Return one deterministic action-planning response."""

    def __init__(self, response: dict[str, Any]) -> None:
        self.response = response

    async def ainvoke(
        self,
        messages: list[object],
        *,
        config: object,
    ) -> SimpleNamespace:
        del messages, config
        return SimpleNamespace(content=json.dumps(self.response))


async def _run_group_case(
    monkeypatch: pytest.MonkeyPatch,
    case: dict[str, Any],
    cognition_output: dict[str, Any],
    *,
    reserve_action_attempt_func: Callable[..., Any] | None = None,
    use_default_reserver: bool = False,
    sequence: list[str] | None = None,
) -> tuple[dict[str, Any], list[str], list[dict[str, Any]]]:
    """Run one group case with deterministic residue, style, and dialog seams."""

    async def load_residue(_case: dict[str, Any]) -> str:
        return ""

    async def prepare_style(_case: dict[str, Any]) -> None:
        return None

    monkeypatch.setattr(runner, "_load_residue_context_for_case", load_residue)
    monkeypatch.setattr(
        runner,
        "_prepare_interaction_style_context",
        prepare_style,
    )
    dialog_calls: list[str] = []
    reservation_calls: list[dict[str, Any]] = []

    async def build_dialog_state(
        cognition_state: dict[str, Any],
        output: dict[str, Any],
        *,
        usage_mode: str,
    ) -> dict[str, Any]:
        del usage_mode
        state = dict(cognition_state)
        state.update(output)
        return state

    async def dialog_client(_state: dict[str, Any]) -> dict[str, Any]:
        if sequence is not None:
            sequence.append("dialog")
        dialog_calls.append("called")
        return {"final_dialog": ["A grounded visible reply."]}

    async def default_reserver(attempt: dict[str, Any]) -> bool:
        if sequence is not None:
            sequence.append("reserve")
        reservation_calls.append(dict(attempt))
        return True

    monkeypatch.setattr(
        runner,
        "_build_dialog_state_with_text_surface",
        build_dialog_state,
    )
    kwargs: dict[str, Any] = {
        "cognition_client": lambda _state: cognition_output,
        "dialog_client": dialog_client,
    }
    if use_default_reserver:
        pass
    elif reserve_action_attempt_func is None:
        kwargs["reserve_action_attempt_func"] = default_reserver
    else:
        async def wrapped_reserver(attempt: dict[str, Any]) -> bool:
            if sequence is not None:
                sequence.append("reserve")
            reservation_calls.append(dict(attempt))
            result = reserve_action_attempt_func(attempt)
            if inspect.isawaitable(result):
                result = await result
            return bool(result)

        kwargs["reserve_action_attempt_func"] = wrapped_reserver
    artifacts = await runner.build_self_cognition_case_artifacts_async(
        case,
        **kwargs,
    )
    return artifacts, dialog_calls, reservation_calls


def test_group_self_cognition_response_contract_requires_exact_decision_shape() -> None:
    """The response decision accepts only its exact bounded object shape."""

    valid = _proposal_response()
    assert validate_self_cognition_response_decision(valid) == valid

    with pytest.raises(CognitionContractError):
        validate_self_cognition_response_decision({
            **valid,
            "unexpected": "field",
        })
    with pytest.raises(CognitionContractError):
        validate_self_cognition_response_decision({
            key: value
            for key, value in valid.items()
            if key != "reason"
        })


def test_group_self_cognition_response_proposal_requires_current_episode_evidence() -> None:
    """Visible proposals cannot be grounded only in promoted memory."""

    with pytest.raises(CognitionContractError):
        validate_self_cognition_response_decision(
            _proposal_response(),
            evidence=[_evidence(source_kind="promoted_memory")],
        )

    assert validate_self_cognition_response_decision(
        _proposal_response(),
        evidence=[_evidence()],
    )["decision"] == "propose_visible_reply"


def test_group_self_cognition_response_proposal_requires_target_and_goal() -> None:
    """Visible proposals require both a semantic target and response goal."""

    no_target = _proposal_response(semantic_target_handle="")
    no_goal = _proposal_response(response_goal="")

    with pytest.raises(CognitionContractError):
        validate_self_cognition_response_decision(no_target)
    with pytest.raises(CognitionContractError):
        validate_self_cognition_response_decision(no_goal)

    with pytest.raises(CognitionContractError):
        validate_self_cognition_response_decision({
            **_silent_response(),
            "participation_basis": "grounded_scene_intervention",
        })
    with pytest.raises(CognitionContractError):
        validate_self_cognition_response_decision(
            _proposal_response(response_goal=" ")
        )


@pytest.mark.asyncio
async def test_group_action_planning_requires_explicit_silence_or_reply_decision() -> None:
    """Think-only planning must carry an explicit semantic response decision."""

    evidence = [_evidence()]
    services_base = {
        "action_planning_config": object(),
        "action_authorization_config": object(),
        "resolver_authorization_config": object(),
    }
    for response, expected_route in (
        (_silent_response(), "silence"),
        (_proposal_response(), "speech"),
    ):
        services = SimpleNamespace(
            llm=_PlannerLLM({
                "action_requests": [],
                "resolver_requests": [],
                "resolver_pending_resolution": None,
                "resolver_goal_progress": None,
                "goal_resolution": "answerable_now",
                "self_cognition_response": response,
            }),
            **services_base,
        )
        result = await plan_actions(
            primary_bid=_primary_bid(),
            supporting_bids=[],
            episode=_episode(),
            evidence=evidence,
            available_actions=[],
            available_resolvers=[],
            resolver_context="resolver_status=idle",
            scene_context={"participant_bindings": []},
            services=services,
        )

        assert result["self_cognition_response"]["decision"] == response[
            "decision"
        ]
        assert result["intention"]["route"] == expected_route


def test_response_outcome_values_are_closed() -> None:
    """Response telemetry normalizes every field to its closed contract."""

    outcome = tracking._bounded_response_outcome({
        "semantic_disposition": "unknown",
        "policy_disposition": "unknown",
        "execution_disposition": "unknown",
        "policy_reason": "unknown",
        "response_gate_codes": ["approved_for_dialog", "raw_content"],
    })

    assert outcome == {
        "semantic_disposition": (
            models.SEMANTIC_DISPOSITION_COGNITION_CONTRACT_FAILED
        ),
        "policy_disposition": models.POLICY_DISPOSITION_NOT_EVALUATED,
        "execution_disposition": models.EXECUTION_DISPOSITION_NOT_REQUESTED,
        "policy_reason": "",
        "response_gate_codes": ["approved_for_dialog"],
    }


def test_group_proposal_requires_structured_participation_grounding() -> None:
    """Ambient group context without participation grounding remains silent."""

    case = _group_case(labels={
        "assistant_presence": "not_in_window",
        "bot_addressing": "ambient_group_context",
    })
    outcome = tracking.evaluate_group_response_policy(
        case,
        _group_cognition_output(_proposal_response()),
    )

    assert outcome is not None
    assert outcome["semantic_disposition"] == (
        models.SEMANTIC_DISPOSITION_REPLY_PROPOSED
    )
    assert outcome["policy_disposition"] == (
        models.POLICY_DISPOSITION_REJECTED
    )
    assert outcome["policy_reason"] == "unresolved_target"
    assert outcome["response_gate_codes"][-1] == "participation_grounding"


def test_missing_group_response_is_a_contract_failure() -> None:
    """The legacy missing-response shape fails closed with explicit evidence."""

    output = _group_cognition_output(_proposal_response())
    del output["cognition_core_output"]["self_cognition_response"]
    outcome = tracking.evaluate_group_response_policy(_group_case(), output)

    assert outcome is not None
    assert outcome["semantic_disposition"] == (
        models.SEMANTIC_DISPOSITION_COGNITION_CONTRACT_FAILED
    )
    assert outcome["policy_disposition"] == (
        models.POLICY_DISPOSITION_NOT_EVALUATED
    )
    assert outcome["response_gate_codes"] == ["response_contract"]
    assert tracking.classify_route(_group_case(), output) == (
        models.ROUTE_AUDIT_ONLY
    )


def test_high_risk_label_alone_does_not_reject_grounded_proposal() -> None:
    """Risk metadata informs the scene while grounding owns participation."""

    case = _group_case(labels={"response_risk": "high"})
    outcome = tracking.evaluate_group_response_policy(
        case,
        _group_cognition_output(_proposal_response()),
    )

    assert outcome is not None
    assert outcome["policy_disposition"] == (
        models.POLICY_DISPOSITION_APPROVED
    )
    assert outcome["policy_reason"] == ""


def test_recent_grounded_group_proposal_is_eligible() -> None:
    """A recent grounded proposal reaches the dialog-eligible policy state."""

    outcome = tracking.evaluate_group_response_policy(
        _group_case(),
        _group_cognition_output(_proposal_response()),
    )

    assert outcome is not None
    assert outcome["semantic_disposition"] == (
        models.SEMANTIC_DISPOSITION_REPLY_PROPOSED
    )
    assert outcome["policy_disposition"] == (
        models.POLICY_DISPOSITION_APPROVED
    )
    assert outcome["response_gate_codes"][-1] == "approved_for_dialog"


def test_group_proposal_requires_window_provenance_and_channel_only_target() -> None:
    """Source identity and target binding must describe the reviewed group."""

    wrong_source = _group_case()
    wrong_source["source_context"]["group_activity_window"]["source"] = (
        "untrusted_source"
    )
    outcome = tracking.evaluate_group_response_policy(
        wrong_source,
        _group_cognition_output(_proposal_response()),
    )
    assert outcome is not None
    assert outcome["policy_reason"] == "invalid_provenance"

    participant_target = _group_case()
    participant_target["delivery_target"]["target_global_user_id"] = "p1"
    outcome = tracking.evaluate_group_response_policy(
        participant_target,
        _group_cognition_output(_proposal_response()),
    )
    assert outcome is not None
    assert outcome["policy_reason"] == "unresolved_target"
    assert outcome["response_gate_codes"][-1] == "bound_group_target"


def test_failed_group_delivery_remains_duplicate_suppressed() -> None:
    """A failed group delivery cannot create a second source-window reply."""

    case = _group_case()
    source_ref = case["source_refs"][0]
    idempotency_key = tracking.build_idempotency_key(
        source_ref["source_kind"],
        source_ref["source_id"],
        source_ref["due_at"],
        case["target_scope"],
        models.ACTION_KIND_SEND_MESSAGE,
    )
    case["existing_attempts"] = [{
        "idempotency_key": idempotency_key,
        "status": models.ACTION_ATTEMPT_STATUS_DELIVERY_FAILED,
    }]

    outcome = tracking.evaluate_group_response_policy(
        case,
        _group_cognition_output(_proposal_response()),
    )
    assert outcome is not None
    assert outcome["policy_disposition"] == (
        models.POLICY_DISPOSITION_REJECTED
    )
    assert outcome["policy_reason"] == "duplicate"
    assert tracking.classify_route(
        case,
        _group_cognition_output(_proposal_response()),
    ) == models.ROUTE_AUDIT_ONLY


def test_duplicate_group_window_remains_suppressed() -> None:
    """A previously reserved source window cannot become a second reply."""

    case = _group_case()
    source_ref = case["source_refs"][0]
    idempotency_key = tracking.build_idempotency_key(
        source_ref["source_kind"],
        source_ref["source_id"],
        source_ref["due_at"],
        case["target_scope"],
        models.ACTION_KIND_SEND_MESSAGE,
    )
    case["existing_attempts"] = [{
        "idempotency_key": idempotency_key,
        "status": models.ACTION_ATTEMPT_STATUS_CANDIDATE,
    }]

    outcome = tracking.evaluate_group_response_policy(
        case,
        _group_cognition_output(_proposal_response()),
    )
    assert outcome is not None
    assert outcome["policy_disposition"] == (
        models.POLICY_DISPOSITION_REJECTED
    )
    assert outcome["policy_reason"] == "duplicate"
    assert tracking.classify_route(
        case,
        _group_cognition_output(_proposal_response()),
    ) == models.ROUTE_AUDIT_ONLY


@pytest.mark.asyncio
async def test_policy_rejection_skips_action_attempt_dialog_and_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rejected proposal ends before action, dialog, and dispatch seams."""

    case = _group_case(labels={
        "assistant_presence": "not_in_window",
        "bot_addressing": "ambient_group_context",
    })
    artifacts, dialog_calls, reservation_calls = await _run_group_case(
        monkeypatch,
        case,
        _group_cognition_output(_proposal_response()),
    )
    outcome = artifacts[models.ARTIFACT_RESPONSE_OUTCOME]
    route_effect = artifacts[models.ARTIFACT_ROUTE_EFFECT]

    assert outcome["policy_disposition"] == (
        models.POLICY_DISPOSITION_REJECTED
    )
    assert outcome["policy_reason"] == "unresolved_target"
    assert route_effect["route"] == models.ROUTE_AUDIT_ONLY
    assert models.ARTIFACT_ACTION_ATTEMPT not in artifacts
    assert models.ARTIFACT_DISPATCH_RESULT not in artifacts
    assert dialog_calls == []
    assert reservation_calls == []


@pytest.mark.asyncio
async def test_eligible_group_proposal_calls_dialog_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An eligible proposal reaches the existing dialog surface exactly once."""

    artifacts, dialog_calls, reservation_calls = await _run_group_case(
        monkeypatch,
        _group_case(labels={"response_risk": "high"}),
        _group_cognition_output(_proposal_response()),
    )

    assert len(dialog_calls) == 1
    assert len(reservation_calls) == 1
    assert artifacts[models.ARTIFACT_ACTION_ATTEMPT]["status"] == (
        models.ACTION_ATTEMPT_STATUS_CANDIDATE
    )
    assert artifacts[models.ARTIFACT_ACTION_CANDIDATE]["text"] == (
        "A grounded visible reply."
    )
    assert artifacts[models.ARTIFACT_RUN_RECORD]["selected_route"] == (
        models.ROUTE_ACTION_CANDIDATE
    )


@pytest.mark.asyncio
async def test_dialog_runs_only_after_atomic_duplicate_reservation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The reservation completes before the dialog client is entered."""

    sequence: list[str] = []
    artifacts, dialog_calls, _reservation_calls = await _run_group_case(
        monkeypatch,
        _group_case(),
        _group_cognition_output(_proposal_response()),
        sequence=sequence,
    )

    assert sequence == ["reserve", "dialog"]
    assert len(dialog_calls) == 1
    assert artifacts[models.ARTIFACT_ACTION_CANDIDATE]["text"]


@pytest.mark.asyncio
async def test_worker_event_contains_three_dispositions_and_gate_codes_without_raw_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Worker telemetry carries bounded outcomes without source or text bodies."""

    replacement = AsyncMock()
    monkeypatch.setattr(
        worker.event_logging,
        "record_self_cognition_event",
        replacement,
    )
    artifacts = {
        models.ARTIFACT_TRIGGER_RECORD: {
            "trigger_id": "trigger-1",
            "trigger_kind": models.TRIGGER_GROUP_CHAT_REVIEW,
        },
        models.ARTIFACT_RUN_RECORD: {
            "run_id": "run-1",
            "selected_route": models.ROUTE_ACTION_CANDIDATE,
            "output_mode": "visible_reply_candidate",
            "status": "completed",
            "budget": {
                "rag_calls": 0,
                "cognition_calls": 1,
                "dialog_calls": 1,
                "topic_limit": 1,
            },
        },
        models.ARTIFACT_RESPONSE_OUTCOME: {
            "semantic_disposition": models.SEMANTIC_DISPOSITION_REPLY_PROPOSED,
            "policy_disposition": models.POLICY_DISPOSITION_APPROVED,
            "execution_disposition": models.EXECUTION_DISPOSITION_NOT_REQUESTED,
            "policy_reason": "",
            "response_gate_codes": ["approved_for_dialog"],
            "source_text": "raw source that must stay out",
            "candidate_text": "raw candidate that must stay out",
        },
    }

    await worker._record_self_cognition_event_from_artifacts(
        case={"case_id": "case-1"},
        artifact_payloads=artifacts,
        dispatch_status="not_requested",
    )

    kwargs = replacement.await_args.kwargs
    assert kwargs["semantic_disposition"] == (
        models.SEMANTIC_DISPOSITION_REPLY_PROPOSED
    )
    assert kwargs["policy_disposition"] == models.POLICY_DISPOSITION_APPROVED
    assert kwargs["execution_disposition"] == (
        models.EXECUTION_DISPOSITION_NOT_REQUESTED
    )
    assert kwargs["response_gate_codes"] == ["approved_for_dialog"]
    serialized = json.dumps(kwargs, ensure_ascii=False, sort_keys=True)
    assert "raw source that must stay out" not in serialized
    assert "raw candidate that must stay out" not in serialized


@pytest.mark.asyncio
async def test_worker_event_does_not_infer_response_failure_without_response_outcome(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Scheduled and binding-only runs must not become fake contract failures."""

    replacement = AsyncMock()
    monkeypatch.setattr(
        worker.event_logging,
        "record_self_cognition_event",
        replacement,
    )
    await worker._record_self_cognition_event_from_artifacts(
        case={"case_id": "case-1"},
        artifact_payloads={
            models.ARTIFACT_TRIGGER_RECORD: {
                "trigger_id": "trigger-1",
                "trigger_kind": models.TRIGGER_ACTIVE_COMMITMENT_DUE_CHECK,
            },
            models.ARTIFACT_RUN_RECORD: {
                "run_id": "run-1",
                "selected_route": models.ROUTE_ACTION_CANDIDATE,
                "output_mode": "scheduled_action_request",
                "status": "completed",
                "budget": {
                    "rag_calls": 0,
                    "cognition_calls": 1,
                    "dialog_calls": 0,
                    "topic_limit": 1,
                },
            },
        },
        dispatch_status="not_requested",
    )

    kwargs = replacement.await_args.kwargs
    assert "semantic_disposition" not in kwargs
    assert "policy_disposition" not in kwargs
    assert "execution_disposition" not in kwargs


@pytest.mark.asyncio
async def test_duplicate_reservation_is_atomic_before_dialog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A unique-key reservation failure prevents any dialog invocation."""

    captured: dict[str, Any] = {}

    class _Collection:
        async def update_one(
            self,
            key_filter: dict[str, Any],
            update: dict[str, Any],
            *,
            upsert: bool,
        ) -> SimpleNamespace:
            captured["key_filter"] = key_filter
            captured["update"] = update
            captured["upsert"] = upsert
            raise DuplicateKeyError("source window already reserved")

    class _Database:
        self_cognition_action_attempts = _Collection()

    async def get_db() -> _Database:
        return _Database()

    monkeypatch.setattr(db_self_cognition, "get_db", get_db)
    sequence: list[str] = []
    artifacts, dialog_calls, reservation_calls = await _run_group_case(
        monkeypatch,
        _group_case(),
        _group_cognition_output(_proposal_response()),
        use_default_reserver=True,
        sequence=sequence,
    )

    assert captured["upsert"] is True
    assert captured["key_filter"]["idempotency_key"]
    assert "idempotency_key" in captured["update"]["$setOnInsert"]
    assert sequence == []
    assert dialog_calls == []
    assert reservation_calls == []
    assert artifacts[models.ARTIFACT_ACTION_ATTEMPT]["status"] == (
        models.ACTION_ATTEMPT_STATUS_DUPLICATE
    )
    outcome = artifacts[models.ARTIFACT_RESPONSE_OUTCOME]
    assert outcome["policy_reason"] == "duplicate"


@pytest.mark.asyncio
async def test_self_cognition_event_contract_rejects_raw_response_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Event logging keeps raw response content outside the telemetry payload."""

    captured: dict[str, Any] = {}

    async def write_event(document: dict[str, Any]) -> str:
        captured.update(document)
        return str(document["event_id"])

    monkeypatch.setattr(recording_module.repository, "write_event", write_event)
    result = await event_logging.record_self_cognition_event(
        component="self_cognition.worker",
        case_id="case-1",
        trigger_kind=models.TRIGGER_GROUP_CHAT_REVIEW,
        selected_route=models.ROUTE_AUDIT_ONLY,
        output_mode="silent",
        budget={
            "rag_calls": 0,
            "cognition_calls": 1,
            "dialog_calls": 0,
            "topic_limit": 1,
        },
        dispatch_status="not_requested",
        status="completed",
        semantic_disposition="invalid_semantic_value",
        policy_disposition="invalid_policy_value",
        execution_disposition="invalid_execution_value",
        policy_reason="raw_policy_text",
        response_gate_codes=["approved_for_dialog", "raw_response_content"],
        consolidation_outcome={
            "consolidation_called": True,
            "write_success": {"character_state": True},
            "scheduled_event_count": 0,
            "cache_evicted_count": 0,
            "origin_trigger_source": "group_chat_review",
            "origin_episode_id": "episode-1",
            "raw_output": "raw model response",
            "source_packet_text": "raw source packet",
            "candidate_text": "raw candidate text",
        },
    )

    assert result["status"] == "recorded"
    payload = captured["payload"]
    assert payload["semantic_disposition"] == (
        models.SEMANTIC_DISPOSITION_COGNITION_CONTRACT_FAILED
    )
    assert payload["policy_disposition"] == (
        models.POLICY_DISPOSITION_NOT_EVALUATED
    )
    assert payload["execution_disposition"] == (
        models.EXECUTION_DISPOSITION_NOT_REQUESTED
    )
    assert payload["policy_reason"] == ""
    assert payload["response_gate_codes"] == ["approved_for_dialog"]
    serialized = json.dumps(captured, ensure_ascii=False, sort_keys=True)
    assert "raw model response" not in serialized
    assert "raw source packet" not in serialized
    assert "raw candidate text" not in serialized
