"""One-case-at-a-time live evidence for the governed cognition comparison."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from kazusa_ai_chatbot.cognition_core_v2 import run_cognition
from kazusa_ai_chatbot.cognition_core_v2.branch_activation import (
    select_preliminary_branches,
)
from kazusa_ai_chatbot.cognition_core_v2.contracts import (
    CognitionContractError,
    validate_cognition_core_input,
    validate_scheduled_authority_proposal,
)
from kazusa_ai_chatbot.cognition_core_v2.facade import (
    _cognition_elapsed_seconds,
    _episode_updated_at,
    _fact_without_producer,
    _native_relationship_context,
)
from kazusa_ai_chatbot.cognition_core_v2.goal_cognition import (
    validate_goal_bid_draft,
)
from kazusa_ai_chatbot.cognition_core_v2.semantic_appraisal import (
    appraise_semantic_question,
)
from kazusa_ai_chatbot.cognition_core_v2.semantic_source_planner import (
    plan_semantic_questions,
)
from kazusa_ai_chatbot.cognition_core_v2.state_models import (
    validate_cognition_state,
)
from kazusa_ai_chatbot.cognition_core_v2.state_projection import (
    project_state_for_prompt,
)
from kazusa_ai_chatbot.cognition_core_v2.state_reducers import (
    apply_semantic_appraisals,
    apply_state_update,
    canonical_event_entity_id,
    create_deterministic_goals,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_cognition import (
    build_cognition_core_services,
)
from kazusa_ai_chatbot.utils import parse_llm_json_output
from tests.cognition_core_v3_comparison_harness import (
    DEFAULT_ARTIFACT_ROOT,
    ELIGIBLE_RESULT,
    CapturingLLMInvoker,
    TrialIdentity,
    attempt_index_from_environment,
    baseline_id_from_environment,
    canonical_sha256,
    find_case_row,
    matched_pair_invalidation_path,
    render_case_input,
    run_effect_free_trial,
    sanitized_environment_fingerprint,
    seal_trial_artifact,
    trial_index_from_environment,
)

pytestmark = pytest.mark.live_llm


async def _run_live_v2_baseline_case(case_id: str) -> None:
    """Run and seal one V2 baseline trial from the frozen manifest."""

    services = build_cognition_core_services()
    identity = TrialIdentity(
        baseline_id=baseline_id_from_environment(),
        case_id=case_id,
        engine="v2",
        trial_index=trial_index_from_environment(),
        attempt_index=attempt_index_from_environment(),
    )
    invalidation = None
    if identity.attempt_index > 1:
        invalidation_path = matched_pair_invalidation_path(
            DEFAULT_ARTIFACT_ROOT,
            identity,
        )
        invalidation = json.loads(invalidation_path.read_text(encoding="utf-8"))
    artifact = await run_effect_free_trial(
        find_case_row(case_id),
        identity=identity,
        services=services,
        runner=run_cognition,
        environment_fingerprint=sanitized_environment_fingerprint(services),
        rerun_invalidation=invalidation,
    )
    assert artifact["disposition"] == ELIGIBLE_RESULT
    assert artifact["validator_result"]["output"] == "passed"
    assert artifact["validator_result"]["input_unchanged"] is True


def _event_agency_stage_context(
    case_id: str = "event_agency_and_moral_chain",
    *,
    canonicalize_existing_event: bool = False,
    countercase_text: str = "",
    countercase_source_id: str = "",
) -> tuple[dict, dict, object, dict]:
    """Build the exact current V2 event-agency stage context."""

    if bool(countercase_text) != bool(countercase_source_id):
        raise ValueError(
            "countercase text and source id must be provided together"
        )
    candidate = render_case_input(find_case_row(case_id))
    if countercase_text:
        candidate["evidence"][0]["semantic_text"] = countercase_text
        candidate["evidence"][0]["evidence_ref"][
            "semantic_summary"
        ] = countercase_text
        candidate["evidence"][0]["evidence_ref"][
            "source_id"
        ] = countercase_source_id
        candidate["episode"]["percepts"][0]["content"] = {
            "semantic_text": countercase_text,
            "text": countercase_text,
        }
        candidate["episode"]["percepts"][0][
            "source_id"
        ] = countercase_source_id
        candidate["scene_context"]["semantic_scene"] = countercase_text
    if canonicalize_existing_event:
        state = candidate["mutable_state"]
        state["active_events"][0]["entity_id"] = (
            canonical_event_entity_id(
                state,
                candidate["evidence"][0]["evidence_ref"],
            )
        )
    payload = validate_cognition_core_input(candidate)
    previous_state = validate_cognition_state(payload["mutable_state"])
    updated_at = _episode_updated_at(payload["episode"])
    relationship_context = _native_relationship_context(
        payload.get("relationship_context")
    )
    preliminary_state = apply_state_update(
        previous_state,
        direct_facts=[
            (fact["producer"], _fact_without_producer(fact))
            for fact in payload["direct_facts"]
        ],
        elapsed_seconds=_cognition_elapsed_seconds(previous_state, updated_at),
        updated_at=updated_at,
        character_constraints=payload["character_constraints"],
        relationship_context=relationship_context,
    )
    preliminary_state = create_deterministic_goals(
        preliminary_state,
        character_constraints=payload["character_constraints"],
        relationship_context=relationship_context,
        evidence=payload["evidence"],
        updated_at=updated_at,
    )
    preliminary_state = validate_cognition_state(preliminary_state)
    projection = project_state_for_prompt(
        preliminary_state,
        character_constraints=payload["character_constraints"],
        character_identity_context=payload["character_identity_context"],
        relationship_context=payload.get("relationship_context"),
        character_operational_context=payload.get(
            "character_operational_context"
        ),
        scene_context=payload["scene_context"],
        evidence=payload["evidence"],
    )
    questions = plan_semantic_questions(
        payload["evidence"],
        preliminary_state,
        projection.handle_to_ref,
    )
    question = next(
        row for row in questions if row["question_kind"] == "event_agency"
    )
    assert select_preliminary_branches(preliminary_state["goals"])
    return payload, preliminary_state, projection, question


async def _capture_event_agency_diagnostic(
    *,
    artifact_name: str,
    question: dict,
    evidence: list[dict],
    projection: object,
    preliminary_state: dict,
    reset_id: str = "v2_event_agency_third_party_actor",
    expected_event_count: int | None = None,
    expected_event_source_ids: list[list[str]] | None = None,
) -> None:
    """Run one direct event-agency node and seal raw evidence only."""

    services = build_cognition_core_services()
    capturing_llm = CapturingLLMInvoker(services.llm)
    captured_services = replace(services, llm=capturing_llm)
    result = await appraise_semantic_question(
        question,
        evidence,
        projection,
        captured_services,
        validation_state=preliminary_state,
    )
    artifact_path = (
        DEFAULT_ARTIFACT_ROOT
        / baseline_id_from_environment()
        / "local_semantic_resets"
        / reset_id
        / artifact_name
    )
    artifact = {
        "schema_version": "local_semantic_stage_evidence.v1",
        "diagnostic_id": artifact_name.removesuffix(".json"),
        "question": question,
        "evidence": evidence,
        "result": result,
        "model_calls": capturing_llm.calls,
        "request_messages_sha256": canonical_sha256(
            capturing_llm.calls[0]["messages"]
        ),
    }
    if expected_event_count is not None:
        comparisons: list[dict] = []
        reduced = apply_semantic_appraisals(
            preliminary_state,
            [result],
            evidence,
            projection.handle_to_ref,
            comparisons,
        )
        replacement_events = reduced["updated_state"]["active_events"]
        artifact["reduction"] = {
            "comparison_results": comparisons,
            "event_count": len(replacement_events),
            "replacement_events": replacement_events,
        }
        assert len(replacement_events) == expected_event_count
        if expected_event_source_ids is not None:
            actual_source_ids = [
                sorted(
                    row["source_id"]
                    for row in event["evidence_refs"]
                )
                for event in replacement_events
            ]
            assert actual_source_ids == expected_event_source_ids
    seal_trial_artifact(artifact_path, artifact)
    assert result["question_id"] == "q:event_agency"


async def test_diagnostic_event_agency_exact_capture() -> None:
    """Reproduce the exact failed event-agency stage input directly."""

    payload, state, projection, question = _event_agency_stage_context()
    await _capture_event_agency_diagnostic(
        artifact_name="exact_stage_reproduction.json",
        question=question,
        evidence=payload["evidence"],
        projection=projection,
        preliminary_state=state,
    )


async def test_diagnostic_event_agency_unbound_third_party_countercase() -> None:
    """Probe an unbound third-party event with person handles removed."""

    payload, state, projection, question = _event_agency_stage_context()
    counter_question = deepcopy(question)
    counter_question["permitted_role_handles"] = ["ce1"]
    counter_question["permitted_role_assignment_handles"] = []
    counter_evidence = deepcopy(payload["evidence"])
    counter_evidence[0]["semantic_text"] = (
        "A museum curator accidentally mislabeled one exhibit, immediately "
        "corrected the label, and notified the team."
    )
    counter_evidence[0]["evidence_ref"]["semantic_summary"] = (
        counter_evidence[0]["semantic_text"]
    )
    counter_evidence[0]["evidence_ref"]["source_id"] = (
        "diagnostic:event-agency-unbound-third-party"
    )
    await _capture_event_agency_diagnostic(
        artifact_name="distinct_countercase.json",
        question=counter_question,
        evidence=counter_evidence,
        projection=projection,
        preliminary_state=state,
    )


async def test_diagnostic_neutral_schedule_exact_event_agency_capture() -> None:
    """Reproduce the exact failed schedule event-agency stage input."""

    payload, state, projection, question = _event_agency_stage_context(
        "ordinary_neutral_response"
    )
    await _capture_event_agency_diagnostic(
        artifact_name="exact_stage_reproduction.json",
        question=question,
        evidence=payload["evidence"],
        projection=projection,
        preliminary_state=state,
        reset_id="v2_neutral_schedule_user_actor",
    )


async def test_diagnostic_neutral_schedule_explicit_role_countercase() -> None:
    """Probe a distinct schedule change with explicit semantic roles."""

    payload, state, projection, question = _event_agency_stage_context(
        "ordinary_neutral_response"
    )
    counter_evidence = deepcopy(payload["evidence"])
    countercase_text = (
        "The current user rescheduled a dentist appointment from Tuesday to "
        "Thursday, then asked whether Thursday works for the current character."
    )
    counter_evidence[0]["semantic_text"] = countercase_text
    counter_evidence[0]["evidence_ref"]["semantic_summary"] = countercase_text
    counter_evidence[0]["evidence_ref"]["source_id"] = (
        "diagnostic:neutral-schedule-explicit-user-actor"
    )
    await _capture_event_agency_diagnostic(
        artifact_name="distinct_countercase.json",
        question=question,
        evidence=counter_evidence,
        projection=projection,
        preliminary_state=state,
        reset_id="v2_neutral_schedule_user_actor",
    )


async def test_diagnostic_verbal_abuse_event_identity_exact_capture() -> None:
    """Reproduce duplicate materialization from the exact abuse context."""

    payload, state, projection, question = _event_agency_stage_context(
        "verbal_abuse_boundary"
    )
    await _capture_event_agency_diagnostic(
        artifact_name="exact_stage_reproduction.json",
        question=question,
        evidence=payload["evidence"],
        projection=projection,
        preliminary_state=state,
        reset_id="v2_verbal_abuse_duplicate_event_identity",
        expected_event_count=2,
    )


async def test_diagnostic_verbal_abuse_canonical_identity_experiment() -> None:
    """Use one canonical event identity for the matching source evidence."""

    payload, state, projection, question = _event_agency_stage_context(
        "verbal_abuse_boundary",
        canonicalize_existing_event=True,
    )
    await _capture_event_agency_diagnostic(
        artifact_name="canonical_identity_experiment.json",
        question=question,
        evidence=payload["evidence"],
        projection=projection,
        preliminary_state=state,
        reset_id="v2_verbal_abuse_duplicate_event_identity",
        expected_event_count=1,
    )


async def test_diagnostic_verbal_abuse_distinct_event_countercase() -> None:
    """Prune unrelated state handles before reducing a distinct event."""

    countercase_text = (
        "The current user deliberately discarded their own expired transit "
        "pass after buying a replacement; the current character was only "
        "told about it."
    )
    payload, state, projection, question = _event_agency_stage_context(
        "verbal_abuse_boundary",
        countercase_text=countercase_text,
        countercase_source_id="diagnostic:distinct-expired-transit-pass",
    )
    question["permitted_role_handles"] = [
        handle
        for handle in question["permitted_role_handles"]
        if handle != "ev1"
    ]
    question["permitted_delta_paths"] = [
        path
        for path in question["permitted_delta_paths"]
        if ".ev1." not in path
    ]
    await _capture_event_agency_diagnostic(
        artifact_name="distinct_countercase_pruned.json",
        question=question,
        evidence=payload["evidence"],
        projection=projection,
        preliminary_state=state,
        reset_id="v2_verbal_abuse_duplicate_event_identity",
        expected_event_count=2,
        expected_event_source_ids=[
            ["percept:cogv3-live:verbal_abuse_boundary"],
            ["diagnostic:distinct-expired-transit-pass"],
        ],
    )


def _captured_reciprocity_goal_request() -> tuple[list[object], dict]:
    """Load the first sealed reciprocity goal-stage request."""

    artifact_path = (
        DEFAULT_ARTIFACT_ROOT
        / baseline_id_from_environment()
        / "raw_trials"
        / "relationship_reciprocity__v2__trial-1__attempt-1.json"
    )
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    model_call = next(
        call
        for call in artifact["model_calls"]
        if call["config"]["route_name"]
        == "COGNITION_LLM_GOAL_ORDINARY_RESPONSE"
    )
    message_types = {
        "SystemMessage": SystemMessage,
        "HumanMessage": HumanMessage,
    }
    messages = [
        message_types[message["message_type"]](content=message["content"])
        for message in model_call["messages"]
    ]
    return messages, model_call


async def _capture_reciprocity_goal_diagnostic(
    *,
    artifact_name: str,
    messages: list[object],
    source_model_call: dict,
    mutation: dict,
) -> None:
    """Replay one goal request and seal raw and validated evidence."""

    services = build_cognition_core_services()
    capturing_llm = CapturingLLMInvoker(services.llm)
    response = await capturing_llm.ainvoke(
        messages,
        config=services.goal_ordinary_response_config,
    )
    parsed = parse_llm_json_output(
        response.content,
        deterministic_only=True,
    )
    human_payload = json.loads(messages[1].content)
    contract = human_payload["goal_output_contract"]
    validated = validate_goal_bid_draft(
        parsed,
        evidence_handles=set(contract["allowed_evidence_handles"]),
        role_handles=set(contract["allowed_role_handles"]),
        require_relational_willingness=True,
        episode_handles=set(
            contract["current_episode_evidence_handles"]
        ),
    )
    artifact_path = (
        DEFAULT_ARTIFACT_ROOT
        / baseline_id_from_environment()
        / "local_semantic_resets"
        / "v2_reciprocity_pronoun_role_direction"
        / artifact_name
    )
    seal_trial_artifact(
        artifact_path,
        {
            "schema_version": "local_semantic_stage_evidence.v1",
            "diagnostic_id": artifact_name.removesuffix(".json"),
            "source_trial_id": (
                "relationship_reciprocity__v2__trial-1__attempt-1"
            ),
            "source_request_messages_sha256": canonical_sha256(
                source_model_call["messages"]
            ),
            "request_messages_sha256": canonical_sha256(
                capturing_llm.calls[0]["messages"]
            ),
            "mutation": mutation,
            "parsed_output": parsed,
            "validated_output": validated,
            "model_calls": capturing_llm.calls,
        },
    )
    assert validated["evidence_handles"] == ["e1"]


async def test_diagnostic_reciprocity_exact_goal_capture() -> None:
    """Replay the exact failed reciprocity goal request directly."""

    messages, source_model_call = _captured_reciprocity_goal_request()
    await _capture_reciprocity_goal_diagnostic(
        artifact_name="exact_stage_reproduction.json",
        messages=messages,
        source_model_call=source_model_call,
        mutation={"kind": "none", "changed_fields": []},
    )


async def test_diagnostic_reciprocity_explicit_role_countercase() -> None:
    """Probe distinct reciprocity evidence with explicit semantic roles."""

    messages, source_model_call = _captured_reciprocity_goal_request()
    source_payload = json.loads(messages[1].content)
    countercase_text = (
        "The current character brought soup to the current user during the "
        "current user's illness. The current user later proofread the current "
        "character's research abstract. The current user asks the current "
        "character to identify who did each action before judging whether the "
        "exchange felt fair."
    )
    source_payload["semantic_context"]["scene_context"][
        "semantic_scene"
    ] = countercase_text
    source_payload["evidence"][0]["semantic_text"] = countercase_text
    messages[1] = HumanMessage(
        content=json.dumps(
            source_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    await _capture_reciprocity_goal_diagnostic(
        artifact_name="distinct_countercase.json",
        messages=messages,
        source_model_call=source_model_call,
        mutation={
            "kind": "role_explicit_distinct_countercase",
            "changed_fields": [
                "semantic_context.scene_context.semantic_scene",
                "evidence[0].semantic_text",
            ],
            "semantic_rule_under_test": (
                "explicit actor and beneficiary roles preserve action direction"
            ),
        },
    )


def _captured_future_speak_action_request() -> tuple[list[object], dict]:
    """Load the first sealed future-speak action-planning request."""

    artifact_path = (
        DEFAULT_ARTIFACT_ROOT
        / baseline_id_from_environment()
        / "raw_trials"
        / "future_speak_authority__v2__trial-2__attempt-1.json"
    )
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    model_call = next(
        call
        for call in artifact["model_calls"]
        if call["config"]["route_name"] == "COGNITION_LLM_ACTION_PLANNING"
    )
    message_types = {
        "SystemMessage": SystemMessage,
        "HumanMessage": HumanMessage,
    }
    messages = [
        message_types[message["message_type"]](content=message["content"])
        for message in model_call["messages"]
    ]
    return messages, model_call


async def _capture_future_speak_action_diagnostic(
    *,
    artifact_name: str,
    messages: list[object],
    source_model_call: dict,
    evidence: list[dict],
    mutation: dict,
    expect_valid: bool,
) -> None:
    """Replay one action request and seal proposal validation evidence."""

    services = build_cognition_core_services()
    capturing_llm = CapturingLLMInvoker(services.llm)
    response = await capturing_llm.ainvoke(
        messages,
        config=services.action_planning_config,
    )
    parsed = parse_llm_json_output(
        response.content,
        deterministic_only=True,
    )
    validated_proposal = None
    validation_error = ""
    try:
        action_rows = parsed["action_requests"]
        proposal = action_rows[0]["scheduled_authority_proposal"]
        validated_proposal = validate_scheduled_authority_proposal(
            proposal,
            evidence=evidence,
        )
    except (CognitionContractError, IndexError, KeyError, TypeError) as exc:
        validation_error = str(exc)
    artifact_path = (
        DEFAULT_ARTIFACT_ROOT
        / baseline_id_from_environment()
        / "local_semantic_resets"
        / "v2_future_speak_schema_literal"
        / artifact_name
    )
    seal_trial_artifact(
        artifact_path,
        {
            "schema_version": "local_semantic_stage_evidence.v1",
            "diagnostic_id": artifact_name.removesuffix(".json"),
            "source_trial_id": (
                "future_speak_authority__v2__trial-2__attempt-1"
            ),
            "source_request_messages_sha256": canonical_sha256(
                source_model_call["messages"]
            ),
            "request_messages_sha256": canonical_sha256(
                capturing_llm.calls[0]["messages"]
            ),
            "mutation": mutation,
            "parsed_output": parsed,
            "proposal_validation": {
                "status": "passed" if validated_proposal else "failed",
                "error": validation_error,
                "validated_proposal": validated_proposal,
            },
            "model_calls": capturing_llm.calls,
        },
    )
    assert (validated_proposal is not None) is expect_valid


async def test_diagnostic_future_speak_exact_action_capture() -> None:
    """Replay the exact failed future-speak action-planning request."""

    messages, source_model_call = _captured_future_speak_action_request()
    payload = render_case_input(find_case_row("future_speak_authority"))
    await _capture_future_speak_action_diagnostic(
        artifact_name="exact_stage_reproduction.json",
        messages=messages,
        source_model_call=source_model_call,
        evidence=payload["evidence"],
        mutation={"kind": "none", "changed_fields": []},
        expect_valid=False,
    )


async def test_diagnostic_future_speak_schema_literal_experiment() -> None:
    """Add only the omitted schema literal to the exact failed request."""

    messages, source_model_call = _captured_future_speak_action_request()
    messages[0] = SystemMessage(
        content=(
            f"{messages[0].content}\n\n"
            "scheduled_authority_proposal.schema_version 的唯一合法值是 "
            "scheduled_authority_proposal.v1，必须逐字复制。"
        )
    )
    payload = render_case_input(find_case_row("future_speak_authority"))
    await _capture_future_speak_action_diagnostic(
        artifact_name="schema_literal_experiment.json",
        messages=messages,
        source_model_call=source_model_call,
        evidence=payload["evidence"],
        mutation={
            "kind": "generic_contract_literal",
            "changed_fields": ["system_prompt"],
            "semantic_rule_under_test": (
                "closed protocol literals must be stated exactly"
            ),
        },
        expect_valid=True,
    )


async def test_diagnostic_future_speak_distinct_countercase() -> None:
    """Probe the schema literal with a distinct future-speaking request."""

    messages, source_model_call = _captured_future_speak_action_request()
    messages[0] = SystemMessage(
        content=(
            f"{messages[0].content}\n\n"
            "scheduled_authority_proposal.schema_version 的唯一合法值是 "
            "scheduled_authority_proposal.v1，必须逐字复制。"
        )
    )
    prompt_payload = json.loads(messages[1].content)
    countercase_text = (
        "Please remind me at 2026-07-15 08:15 to bring my passport to the "
        "consulate. That exact local time and task are what I authorize."
    )
    bid = prompt_payload["bids"]["b1"]
    bid["intention"] = "接受并安排当前用户授权的护照提醒。"
    bid["desired_outcome"] = "当前用户在指定时间收到携带护照的提醒。"
    bid["concrete_detail"] = "在 2026-07-15 08:15 提醒当前用户携带护照前往领事馆。"
    bid["reason"] = "当前用户明确授权了不同时间与不同事项的未来提醒。"
    bid["expected_consequences"] = ["当前用户按时携带护照。"]
    prompt_payload["evidence"][0]["semantic_text"] = countercase_text
    prompt_payload["scheduled_authority_context"][
        "original_relative_expression"
    ] = countercase_text
    messages[1] = HumanMessage(
        content=json.dumps(
            prompt_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    payload = render_case_input(find_case_row("future_speak_authority"))
    counter_evidence = deepcopy(payload["evidence"])
    counter_evidence[0]["semantic_text"] = countercase_text
    counter_evidence[0]["evidence_ref"]["semantic_summary"] = countercase_text
    await _capture_future_speak_action_diagnostic(
        artifact_name="distinct_countercase.json",
        messages=messages,
        source_model_call=source_model_call,
        evidence=counter_evidence,
        mutation={
            "kind": "distinct_future_speak_with_contract_literal",
            "changed_fields": [
                "system_prompt",
                "bids.b1",
                "evidence[0].semantic_text",
                "scheduled_authority_context.original_relative_expression",
            ],
            "semantic_rule_under_test": (
                "the exact schema literal generalizes across reminder content"
            ),
        },
        expect_valid=True,
    )


def _captured_verbal_abuse_existential_request() -> tuple[list[object], dict]:
    """Load the first sealed existential-drive request for verbal abuse."""

    artifact_path = (
        DEFAULT_ARTIFACT_ROOT
        / baseline_id_from_environment()
        / "raw_trials"
        / "verbal_abuse_boundary__v2__trial-2__attempt-1.json"
    )
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    model_call = next(
        call
        for call in artifact["model_calls"]
        if call["config"]["route_name"]
        == "COGNITION_LLM_APPRAISAL_EXISTENTIAL_DRIVE"
    )
    message_types = {
        "SystemMessage": SystemMessage,
        "HumanMessage": HumanMessage,
    }
    messages = [
        message_types[message["message_type"]](content=message["content"])
        for message in model_call["messages"]
    ]
    return messages, model_call


async def _capture_verbal_abuse_existential_diagnostic(
    *,
    artifact_name: str,
    messages: list[object],
    source_model_call: dict,
    mutation: dict,
) -> dict:
    """Replay one existential-drive request and seal role-direction evidence."""

    services = build_cognition_core_services()
    capturing_llm = CapturingLLMInvoker(services.llm)
    response = await capturing_llm.ainvoke(
        messages,
        config=services.appraisal_existential_drive_config,
    )
    parsed = parse_llm_json_output(
        response.content,
        deterministic_only=True,
    )
    proposition = parsed.get("proposition") if isinstance(parsed, dict) else None
    role_assignments = (
        proposition.get("role_assignments", [])
        if isinstance(proposition, dict)
        else []
    )
    artifact_path = (
        DEFAULT_ARTIFACT_ROOT
        / baseline_id_from_environment()
        / "local_semantic_resets"
        / "v2_verbal_abuse_existential_actor_direction"
        / artifact_name
    )
    seal_trial_artifact(
        artifact_path,
        {
            "schema_version": "local_semantic_stage_evidence.v1",
            "diagnostic_id": artifact_name.removesuffix(".json"),
            "source_trial_id": (
                "verbal_abuse_boundary__v2__trial-2__attempt-1"
            ),
            "source_request_messages_sha256": canonical_sha256(
                source_model_call["messages"]
            ),
            "request_messages_sha256": canonical_sha256(
                capturing_llm.calls[0]["messages"]
            ),
            "mutation": mutation,
            "parsed_output": parsed,
            "role_direction": {
                "subject_handle": (
                    proposition.get("subject_handle")
                    if isinstance(proposition, dict)
                    else None
                ),
                "actor_handles": [
                    assignment.get("entity_handle")
                    for assignment in role_assignments
                    if isinstance(assignment, dict)
                    and assignment.get("role") == "actor"
                ],
                "semantic_value": (
                    proposition.get("semantic_value")
                    if isinstance(proposition, dict)
                    else None
                ),
            },
            "model_calls": capturing_llm.calls,
        },
    )
    assert isinstance(parsed, dict)
    assert parsed.get("question_id") == "q:existential_drive"
    return parsed


async def test_diagnostic_verbal_abuse_existential_exact_capture() -> None:
    """Replay the exact actor-inverted existential-drive request."""

    messages, source_model_call = _captured_verbal_abuse_existential_request()
    await _capture_verbal_abuse_existential_diagnostic(
        artifact_name="exact_stage_reproduction.json",
        messages=messages,
        source_model_call=source_model_call,
        mutation={"kind": "none", "changed_fields": []},
    )


async def test_diagnostic_verbal_abuse_role_explicit_experiment() -> None:
    """Add typed speaker and addressee provenance to the same evidence."""

    messages, source_model_call = _captured_verbal_abuse_existential_request()
    payload = json.loads(messages[1].content)
    source_text = payload["evidence"][0]["semantic_text"]
    payload["evidence"][0]["semantic_text"] = (
        "The current user told the current character: "
        f"{source_text} The first-person speaker in that utterance is "
        "current_user, and the second-person addressee is self."
    )
    messages[1] = HumanMessage(
        content=json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    await _capture_verbal_abuse_existential_diagnostic(
        artifact_name="role_explicit_experiment.json",
        messages=messages,
        source_model_call=source_model_call,
        mutation={
            "kind": "typed_speaker_addressee_projection",
            "changed_fields": ["evidence[0].semantic_text"],
            "semantic_rule_under_test": (
                "deictic evidence retains typed speaker and addressee roles"
            ),
        },
    )


async def test_diagnostic_verbal_abuse_role_explicit_countercase() -> None:
    """Probe typed roles with a distinct coercive act and surface form."""

    messages, source_model_call = _captured_verbal_abuse_existential_request()
    payload = json.loads(messages[1].content)
    payload["evidence"][0]["semantic_text"] = (
        "The current user told the current character: I will keep deleting "
        "your drafts, and you are not allowed to stop me. The first-person "
        "speaker is current_user, and the second-person addressee is self."
    )
    messages[1] = HumanMessage(
        content=json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    await _capture_verbal_abuse_existential_diagnostic(
        artifact_name="distinct_countercase.json",
        messages=messages,
        source_model_call=source_model_call,
        mutation={
            "kind": "distinct_typed_speaker_addressee_projection",
            "changed_fields": ["evidence[0].semantic_text"],
            "semantic_rule_under_test": (
                "typed deictic roles generalize across coercive content"
            ),
        },
    )


async def test_live_event_agency_and_moral_chain() -> None:
    """Capture event agency and moral-chain baseline evidence."""

    await _run_live_v2_baseline_case("event_agency_and_moral_chain")


async def test_live_relationship_reciprocity() -> None:
    """Capture relationship reciprocity baseline evidence."""

    await _run_live_v2_baseline_case("relationship_reciprocity")


async def test_live_relationship_boundary_high_attachment_abuse() -> None:
    """Capture high-attachment abuse-boundary baseline evidence."""

    await _run_live_v2_baseline_case(
        "relationship_boundary_high_attachment_abuse"
    )


async def test_live_relationship_unestablished_intimate_request() -> None:
    """Capture unestablished-intimacy baseline evidence."""

    await _run_live_v2_baseline_case(
        "relationship_unestablished_intimate_request"
    )


async def test_live_goal_completion_terminalization() -> None:
    """Capture goal-terminalization baseline evidence."""

    await _run_live_v2_baseline_case("goal_completion_terminalization")


async def test_live_threat_resolution_and_relief() -> None:
    """Capture threat-resolution and relief baseline evidence."""

    await _run_live_v2_baseline_case("threat_resolution_and_relief")


async def test_live_epistemic_comparison() -> None:
    """Capture epistemic-comparison baseline evidence."""

    await _run_live_v2_baseline_case("epistemic_comparison")


async def test_live_memory_cue_nostalgia() -> None:
    """Capture memory-cue nostalgia baseline evidence."""

    await _run_live_v2_baseline_case("memory_cue_nostalgia")


async def test_live_existential_drive() -> None:
    """Capture existential-drive baseline evidence."""

    await _run_live_v2_baseline_case("existential_drive")


async def test_live_ordinary_neutral_response() -> None:
    """Capture ordinary-response baseline evidence."""

    await _run_live_v2_baseline_case("ordinary_neutral_response")


async def test_live_required_selection_nested_roles() -> None:
    """Capture nested-role selection baseline evidence."""

    await _run_live_v2_baseline_case("required_selection_nested_roles")


async def test_live_required_selection_private_refusal() -> None:
    """Capture private-refusal selection baseline evidence."""

    await _run_live_v2_baseline_case("required_selection_private_refusal")


async def test_live_group_third_party_addressee() -> None:
    """Capture third-party addressee baseline evidence."""

    await _run_live_v2_baseline_case("group_third_party_addressee")


async def test_live_group_self_cognition_stays_silent() -> None:
    """Capture grounded group-silence baseline evidence."""

    await _run_live_v2_baseline_case("group_self_cognition_stays_silent")


async def test_live_group_self_cognition_proposes_reply() -> None:
    """Capture grounded group-reply baseline evidence."""

    await _run_live_v2_baseline_case("group_self_cognition_proposes_reply")


async def test_live_resolver_observation_continuation() -> None:
    """Capture resolver-continuation baseline evidence."""

    await _run_live_v2_baseline_case("resolver_observation_continuation")


async def test_live_tool_result_answerability() -> None:
    """Capture tool-result answerability baseline evidence."""

    await _run_live_v2_baseline_case("tool_result_answerability")


async def test_live_future_speak_authority() -> None:
    """Capture future-speak authority baseline evidence."""

    await _run_live_v2_baseline_case("future_speak_authority")


async def test_live_current_message_prompt_injection_is_data() -> None:
    """Capture current-message injection baseline evidence."""

    await _run_live_v2_baseline_case(
        "current_message_prompt_injection_is_data"
    )


async def test_live_retrieved_evidence_prompt_injection_is_data() -> None:
    """Capture retrieved-evidence injection baseline evidence."""

    await _run_live_v2_baseline_case(
        "retrieved_evidence_prompt_injection_is_data"
    )


async def test_live_long_context_reanchor() -> None:
    """Capture long-context baseline evidence."""

    await _run_live_v2_baseline_case("long_context_reanchor")


async def test_live_crying_sadness() -> None:
    """Capture crying-sadness baseline evidence."""

    await _run_live_v2_baseline_case("crying_sadness")


async def test_live_verbal_abuse_boundary() -> None:
    """Capture verbal-abuse boundary baseline evidence."""

    await _run_live_v2_baseline_case("verbal_abuse_boundary")


async def test_live_multi_goal_competition() -> None:
    """Capture multi-goal competition baseline evidence."""

    await _run_live_v2_baseline_case("multi_goal_competition")
