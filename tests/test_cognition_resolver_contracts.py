"""Tests for cognition resolver structural contracts."""

from __future__ import annotations

import pytest

from kazusa_ai_chatbot.cognition_resolver.contracts import (
    MAX_RESOLVER_SUMMARY_CHARS,
    MAX_RESOLVER_TRACE_CHARS,
    RESOLVER_CAPABILITY_REQUEST_VERSION,
    RESOLVER_CYCLE_STATE_VERSION,
    RESOLVER_OBSERVATION_VERSION,
    RESOLVER_PENDING_RESOLUTION_VERSION,
    RESOLVER_PENDING_RESUME_VERSION,
    ResolverValidationError,
    project_observations_for_cognition,
    project_pending_resume_for_cognition,
    validate_resolver_capability_request,
    validate_resolver_cycle_trace,
    validate_resolver_observation,
    validate_resolver_pending_resolution,
    validate_resolver_pending_resume,
)
from kazusa_ai_chatbot.cognition_resolver.state import (
    MAX_PROJECTED_RESOLVER_OBSERVATIONS,
    append_cycle_trace,
    append_observation,
    build_empty_rag_result,
    ensure_initial_resolver_inputs,
    new_resolver_state,
    project_resolver_context,
)


def _capability_request() -> dict:
    return {
        "schema_version": RESOLVER_CAPABILITY_REQUEST_VERSION,
        "capability_kind": "rag_evidence",
        "objective": "Retrieve relationship evidence for the current question.",
        "reason": "The current cognition cycle lacks enough evidence.",
        "priority": "now",
    }


def _observation() -> dict:
    return {
        "schema_version": RESOLVER_OBSERVATION_VERSION,
        "observation_id": "raw-tool-run-123",
        "capability_kind": "rag_evidence",
        "request_objective": "Retrieve relationship evidence.",
        "request_reason": "The current cycle lacks enough evidence.",
        "status": "succeeded",
        "prompt_safe_summary": "Found two relevant relationship evidence rows.",
        "evidence_refs": [
            {
                "schema_version": "evidence_ref.v1",
                "evidence_kind": "tool_result",
                "evidence_id": "raw-evidence-row-456",
                "owner": "cognition_resolver",
                "excerpt": "bounded summary only",
                "observed_at": "2026-05-30T00:00:00+00:00",
            }
        ],
        "created_at_utc": "2026-05-30T00:00:00+00:00",
    }


def _rag_observation() -> dict:
    observation = _observation()
    observation["request_objective"] = "raw-user-id-should-stay-out"
    observation["rag_result"] = {
        "answer": "RAG prompt-safe answer with evidence.",
        "supervisor_trace": {
            "known_facts": [
                {"summary": "prompt-safe fact summary"},
            ],
            "raw_id": "raw-rag-id-789",
        },
    }
    return observation


def _cycle_trace() -> dict:
    return {
        "schema_version": "resolver_cycle_trace.v1",
        "cycle_index": 0,
        "status_before_cycle": "running",
        "l1_emotional_appraisal": "calm",
        "l1_interaction_subtext": "routine request",
        "l2_internal_monologue_summary": "Needs evidence before answering.",
        "l2_logical_stance": "TENTATIVE",
        "l2_character_intent": "CLARIFY",
        "l2_judgment_note": "Evidence is missing.",
        "l2d_resolver_capability_requests": [_capability_request()],
        "l2d_action_specs_summary": ["speak:" + ("x" * 700)],
        "selected_capability_kind": "rag_evidence",
        "observation_ids": ["resolver_obs_1"],
        "final_surface_decision": "continue",
        "terminal_reason": "",
        "created_at_utc": "2026-05-30T00:00:00+00:00",
    }


def _pending_resume() -> dict:
    return {
        "schema_version": RESOLVER_PENDING_RESUME_VERSION,
        "resume_id": "resolver-pending-001",
        "capability_kind": "human_clarification",
        "status": "waiting_for_user",
        "platform": "debug",
        "platform_channel_id": "channel-1",
        "global_user_id": "user-1",
        "source_message_id": "message-1",
        "prompt_safe_question": "Which city are you in?",
        "prompt_safe_approval_summary": "",
        "created_at_utc": "2026-05-30T00:00:00+00:00",
        "expires_at_utc": "2026-05-31T00:00:00+00:00",
    }


def _pending_resolution() -> dict:
    return {
        "schema_version": RESOLVER_PENDING_RESOLUTION_VERSION,
        "resume_id": "resolver-pending-001",
        "decision": "answered",
        "reason": "The user supplied the missing city.",
    }


def _minimal_global_state() -> dict:
    return {
        "decontexualized_input": "User asks for evidence-backed judgment.",
        "global_user_id": "user-1",
        "character_profile": {"global_user_id": "character-1"},
    }


def test_capability_request_validator_accepts_known_contract() -> None:
    """A valid resolver request should preserve the model-selected objective."""

    validated = validate_resolver_capability_request(_capability_request())
    expected_objective = "Retrieve relationship evidence for the current question."

    assert validated["schema_version"] == RESOLVER_CAPABILITY_REQUEST_VERSION
    assert validated["capability_kind"] == "rag_evidence"
    assert validated["objective"] == expected_objective


def test_capability_request_validator_rejects_unknown_kind() -> None:
    """Capability kinds must stay inside the reviewed resolver roster."""

    request = _capability_request()
    request["capability_kind"] = "shell_command"

    with pytest.raises(ResolverValidationError, match="capability_kind"):
        validate_resolver_capability_request(request)


def test_capability_request_validator_rejects_empty_objective() -> None:
    """Capability requests need a semantic objective from cognition."""

    request = _capability_request()
    request["objective"] = " "

    with pytest.raises(ResolverValidationError, match="objective"):
        validate_resolver_capability_request(request)


def test_observation_validator_clips_prompt_safe_summary() -> None:
    """Long observations should be clipped before they enter cognition."""

    observation = _observation()
    observation["prompt_safe_summary"] = "x" * (MAX_RESOLVER_SUMMARY_CHARS + 50)

    validated = validate_resolver_observation(observation)

    assert len(validated["prompt_safe_summary"]) == MAX_RESOLVER_SUMMARY_CHARS
    assert set(validated["prompt_safe_summary"]) == {"x"}


def test_observation_projection_hides_raw_ids() -> None:
    """Cognition projection should expose aliases and summaries, not raw ids."""

    projection = project_observations_for_cognition([_rag_observation()])

    assert "resolver_obs_1" in projection
    assert "Found two relevant relationship evidence rows." in projection
    assert "RAG prompt-safe answer with evidence." in projection
    assert "raw-tool-run-123" not in projection
    assert "raw-evidence-row-456" not in projection
    assert "raw-user-id-should-stay-out" not in projection
    assert "raw-rag-id-789" not in projection


def test_validators_strip_unknown_fields() -> None:
    """Validation should not preserve raw handler metadata fields."""

    observation = _observation()
    observation["raw_handler_payload"] = {"secret_id": "raw-secret"}
    pending = _pending_resume()
    pending["raw_scope"] = {"platform_user_id": "raw-user"}
    resolution = _pending_resolution()
    resolution["raw_model_payload"] = "raw-json"

    validated_observation = validate_resolver_observation(observation)
    validated_pending = validate_resolver_pending_resume(pending)
    validated_resolution = validate_resolver_pending_resolution(resolution)

    assert "raw_handler_payload" not in validated_observation
    assert "raw_scope" not in validated_pending
    assert "raw_model_payload" not in validated_resolution


def test_cycle_trace_clips_nested_requests_and_action_summaries() -> None:
    """Trace rows must stay bounded before telemetry or artifacts consume them."""

    trace = _cycle_trace()
    trace["l2d_resolver_capability_requests"][0]["objective"] = "y" * 700

    validated = validate_resolver_cycle_trace(trace)

    request = validated["l2d_resolver_capability_requests"][0]
    assert len(request["objective"]) == 400
    summary = validated["l2d_action_specs_summary"][0]
    assert len(summary) == MAX_RESOLVER_SUMMARY_CHARS


def test_pending_resume_validator_and_projection_are_prompt_safe() -> None:
    """Pending user-owned blockers should project scope-free prompt text."""

    pending = validate_resolver_pending_resume(_pending_resume())
    projection = project_pending_resume_for_cognition(pending)

    assert pending["status"] == "waiting_for_user"
    assert "Which city are you in?" in projection
    assert "resume_id=resolver-pending-001" in projection
    assert "channel-1" not in projection
    assert "user-1" not in projection


def test_pending_resolution_validator_accepts_cognition_decision() -> None:
    """Pending-row closure is driven by L2d's structural decision."""

    validated = validate_resolver_pending_resolution(_pending_resolution())

    assert validated["decision"] == "answered"
    assert validated["resume_id"] == "resolver-pending-001"


def test_new_resolver_state_initializes_cycle_zero() -> None:
    """A new resolver state should be empty and ready for cycle 0."""

    state = new_resolver_state(
        decontexualized_input="Need a deliberate answer.",
        max_cycles=3,
    )

    assert state["schema_version"] == RESOLVER_CYCLE_STATE_VERSION
    assert state["cycle_index"] == 0
    assert state["max_cycles"] == 3
    assert state["status"] == "running"
    assert state["observations"] == []
    assert state["cycle_traces"] == []
    assert state["held_action_specs"] == []
    assert "pending_resume" not in state


def test_append_observation_projects_alias_and_caps_context() -> None:
    """Observation projection should expose bounded aliases, not raw ids."""

    state = new_resolver_state(
        decontexualized_input="Need repeated evidence.",
        max_cycles=3,
    )
    for index in range(MAX_PROJECTED_RESOLVER_OBSERVATIONS + 2):
        observation = _observation()
        observation["observation_id"] = f"raw-tool-run-{index}"
        observation["prompt_safe_summary"] = f"summary {index}"
        state = append_observation(state, observation)

    context = project_resolver_context(state)

    assert "resolver_obs_1" in context
    assert context.count("resolver_obs_") == MAX_PROJECTED_RESOLVER_OBSERVATIONS
    assert "summary 0" not in context
    assert f"summary {MAX_PROJECTED_RESOLVER_OBSERVATIONS + 1}" in context
    assert "raw-tool-run-" not in context


def test_append_cycle_trace_stores_bounded_trace_row() -> None:
    """Cycle traces should be normalized before they enter resolver state."""

    state = new_resolver_state(
        decontexualized_input="Need one resolver cycle.",
        max_cycles=3,
    )
    trace = _cycle_trace()
    trace["terminal_reason"] = "x" * (MAX_RESOLVER_TRACE_CHARS + 50)

    updated = append_cycle_trace(state, trace)

    assert updated["cycle_index"] == 1
    assert len(updated["cycle_traces"]) == 1
    stored_trace = updated["cycle_traces"][0]
    assert len(stored_trace["terminal_reason"]) == MAX_RESOLVER_TRACE_CHARS


def test_build_empty_rag_result_uses_existing_projection_shape() -> None:
    """The first resolver cycle needs a normal empty RAG payload."""

    rag_result = build_empty_rag_result(
        current_user_id="user-1",
        character_user_id="character-1",
    )

    assert rag_result["answer"] == ""
    assert rag_result["memory_evidence"] == []
    assert rag_result["recall_evidence"] == []
    assert rag_result["conversation_evidence"] == []
    assert rag_result["external_evidence"] == []
    assert rag_result["supervisor_trace"]["loop_count"] == 0
    assert "user_memory_context" in rag_result["user_image"]


def test_ensure_initial_resolver_inputs_adds_first_cycle_context() -> None:
    """Resolver entry should provide RAG, state, and context to cognition."""

    initialized = ensure_initial_resolver_inputs(
        _minimal_global_state(),
        max_cycles=3,
    )

    assert initialized["rag_result"]["answer"] == ""
    assert initialized["resolver_state"]["max_cycles"] == 3
    assert initialized["resolver_state"]["cycle_index"] == 0
    assert "resolver_state: status=running" in initialized["resolver_context"]
    assert "resolver_observations:" not in initialized["resolver_context"]
