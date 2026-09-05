"""Direct ownership tests for the L3 surface handoff."""

from __future__ import annotations

import asyncio
import json
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest

from kazusa_ai_chatbot.background_work.result_source import (
    build_result_ready_episode_from_job,
)
from kazusa_ai_chatbot.cognition_resolver.contracts import (
    TERMINAL_RESOLVER_SURFACE_DECISION,
    ResolverValidationError,
)
from kazusa_ai_chatbot.cognition_shared import surface, surface_stages
from kazusa_ai_chatbot.cognition_shared.contracts import (
    CognitionContractError,
    CognitionExecutionError,
    TextSurfaceServicesV2,
    validate_text_surface_input_canonical,
)
from kazusa_ai_chatbot.cognition_shared.state_projection import (
    validate_prompt_projection,
)
from kazusa_ai_chatbot.llm_interface import LLMCallConfig, LLMThinkingConfig
from kazusa_ai_chatbot.nodes import dialog_agent
from kazusa_ai_chatbot.nodes import persona_supervisor2_l3_surface as l3_surface
from kazusa_ai_chatbot.nodes.persona_supervisor2_schema import GlobalPersonaState
from tests.task_resolution_test_helpers import (
    accepted_task_completed_job,
    resolver_task_observation,
)
from tests.unit.nodes.surface_fixtures import (
    build_relational_decision,
    build_surface_state,
)


def _tool_result_surface_state(
    *,
    evidence_excerpt: str = "PLAN3_E2E_BETA_SELECTED",
    semantic_summary: str = "The beta marker is ready.",
) -> tuple[dict[str, object], dict[str, object]]:
    """Build one L3 state carrying a validated completed task result."""

    job = deepcopy(accepted_task_completed_job())
    task_result = job["task_resolution_result"]
    assert isinstance(task_result, dict)
    task_result["evidence_excerpts"] = [evidence_excerpt]
    task_result["prompt_safe_summary"] = semantic_summary
    episode = build_result_ready_episode_from_job(job)
    continuation_ref = task_result["goal_continuation_ref"]
    state = build_surface_state(build_relational_decision())
    state["cognitive_episode"] = episode
    action_specs = state["action_specs"]
    assert isinstance(action_specs, list)
    assert isinstance(action_specs[0], dict)
    action_specs[0].update({
        "surface_role": "task_result",
        "goal_continuation_ref": continuation_ref,
    })
    return state, task_result


def test_degraded_tool_result_surface_preserves_semantic_result() -> None:
    """A succeeded typed result remains deliverable after content exhaustion."""

    marker = "PLAN3_E2E_BETA_SELECTED"
    state, _task_result = _tool_result_surface_state(
        evidence_excerpt="A separate provenance finding.",
        semantic_summary=f"Resolved result: {marker}",
    )
    payload = l3_surface.build_text_surface_input_from_global_state(
        state,
        interaction_style_context="brief and natural",
    )
    resolver_result = payload["resolver_result"]
    selected_intent = payload["response_plan"]["response_goal"]

    degraded = surface.build_degraded_text_surface(payload)

    assert resolver_result["semantic_result"] == f"Resolved result: {marker}"
    assert resolver_result["evidence_excerpts"] == [
        "A separate provenance finding.",
    ]
    assert marker not in selected_intent
    assert degraded["content_plan"] == resolver_result["semantic_result"]
    assert degraded["selected_surface_intent"] == selected_intent
    assert degraded["resolver_result"] == resolver_result


def test_degraded_ordinary_surface_keeps_selected_intent() -> None:
    """Ordinary surface degradation preserves the selected response intent."""

    payload = l3_surface.build_text_surface_input_from_global_state(
        build_surface_state(build_relational_decision()),
        interaction_style_context="brief and natural",
    )

    degraded = surface.build_degraded_text_surface(payload)

    assert degraded["content_plan"] == payload["response_plan"]["response_goal"]


def test_degraded_blocked_tool_result_keeps_selected_intent() -> None:
    """A non-succeeded resolver result cannot replace selected intent."""

    state, _task_result = _tool_result_surface_state()
    payload = l3_surface.build_text_surface_input_from_global_state(
        state,
        interaction_style_context="brief and natural",
    )
    blocked_result = dict(payload["resolver_result"])
    blocked_result.update({
        "status": "blocked",
        "semantic_result": "Blocked result must not become delivery text.",
        "evidence_state": "blocked",
        "evidence_excerpts": [],
        "evidence_handles": [],
        "remaining_needs": ["One required dependency is unavailable."],
    })
    payload["resolver_result"] = blocked_result

    degraded = surface.build_degraded_text_surface(payload)

    assert degraded["content_plan"] == payload["response_plan"]["response_goal"]
    assert degraded["selected_surface_intent"] == payload["response_plan"][
        "response_goal"
    ]


@pytest.mark.asyncio
async def test_terminal_dialog_preserves_degraded_tool_result_semantic_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both exhausted model stages retain validated tool-result semantics."""

    marker = "PLAN3_E2E_BETA_SELECTED"
    state, _task_result = _tool_result_surface_state(
        evidence_excerpt="A separate provenance finding.",
        semantic_summary=f"Resolved result: {marker}",
    )
    payload = l3_surface.build_text_surface_input_from_global_state(
        state,
        interaction_style_context="brief and natural",
    )

    async def fail_content_stage(*_args: object, **_kwargs: object) -> object:
        raise CognitionExecutionError(
            "content-plan candidates exhausted",
            error_code="surface_content_plan_contract_exhausted",
            stage="surface.content_plan",
            attempt_count=3,
            safe_checkpoint="post_cognition_commit",
        )

    async def record_event(**_kwargs: object) -> dict[str, object]:
        return {}

    class _InvalidDialogInvoker:
        def __init__(self) -> None:
            self.calls: list[object] = []

        async def ainvoke(self, messages, *, config):
            del config
            self.calls.append(messages)
            return SimpleNamespace(content='{"unexpected": "field"}')

    monkeypatch.setattr(surface, "run_content_plan_stage", fail_content_stage)
    surface_output = await surface._run_text_surface_planning(
        payload,
        SimpleNamespace(llm=object(), content_plan_config=object()),
        session=None,
    )
    invoker = _InvalidDialogInvoker()
    monkeypatch.setattr(dialog_agent, "_dialog_generator_llm", invoker)
    monkeypatch.setattr(
        dialog_agent.llm_tracing,
        "record_llm_trace_step",
        record_event,
    )
    monkeypatch.setattr(
        dialog_agent.event_logging,
        "record_llm_stage_event",
        record_event,
    )
    monkeypatch.setattr(
        dialog_agent.event_logging,
        "record_model_contract_event",
        record_event,
    )
    monkeypatch.setattr(
        dialog_agent.event_logging,
        "record_dialog_quality_event",
        record_event,
    )
    dialog_state = build_surface_state(build_relational_decision())
    dialog_state["cognitive_episode"] = payload["episode"]
    dialog_state["text_surface_output_v2"] = surface_output
    dialog_state["dialog_usage_mode"] = "live_visible_reply"

    dialog_output = await dialog_agent.dialog_generator(dialog_state)

    assert surface_output["content_plan"] == f"Resolved result: {marker}"
    assert surface_output["selected_surface_intent"] == payload["response_plan"][
        "response_goal"
    ]
    assert dialog_output["final_dialog"] == [f"Resolved result: {marker}"]
    assert len(invoker.calls) == 3


def test_l3_surface_preserves_relational_willingness() -> None:
    """L3 carries the exact selected stance into the surface input."""

    decision = build_relational_decision(stance="conditional_accept")
    payload = l3_surface.build_text_surface_input_from_global_state(
        build_surface_state(decision),
        interaction_style_context="brief and natural",
    )

    assert payload["relational_willingness"] == decision
    assert "relationship_willingness" not in payload


def test_l3_surface_preserves_selected_response_operation() -> None:
    """L3 carries the canonical response goal beside the semantic goal."""

    state = build_surface_state(build_relational_decision())
    payload = l3_surface.build_text_surface_input_from_global_state(
        state,
        interaction_style_context="brief and natural",
    )

    assert payload["response_plan"]["response_goal"] == (
        "acknowledge the grounded episode"
    )


def test_l3_surface_projects_typed_tool_result_with_evidence_lineage() -> None:
    """A result-ready episode keeps its exact source-owned evidence in L3."""

    state, task_result = _tool_result_surface_state()

    payload = l3_surface.build_text_surface_input_from_global_state(
        state,
        interaction_style_context="brief and natural",
    )
    resolver_result = payload["resolver_result"]

    episode = state["cognitive_episode"]
    assert isinstance(episode, dict)
    percepts = episode["percepts"]
    assert isinstance(percepts, list)
    assert isinstance(percepts[0], dict)
    content = percepts[0]["content"]
    assert isinstance(content, dict)
    cognition_source = content["cognition_source"]
    assert isinstance(cognition_source, dict)
    assert resolver_result["semantic_result"] == cognition_source["semantic_summary"]
    assert resolver_result["evidence_excerpts"] == ["PLAN3_E2E_BETA_SELECTED"]
    assert resolver_result["evidence_handles"] == ["https://example.com/source"]
    continuation_ref = task_result["goal_continuation_ref"]
    assert episode["origin_metadata"]["goal_continuation_ref"] == continuation_ref
    assert cognition_source["goal_continuation_ref"] == continuation_ref
    assert state["action_specs"][0]["goal_continuation_ref"] == continuation_ref
    projected = surface._project_surface_payload(payload)
    assert projected["resolver_result"] == resolver_result
    degraded = surface.build_degraded_text_surface(payload)
    assert degraded["resolver_result"] == resolver_result


@pytest.mark.asyncio
async def test_l3_tool_result_reaches_the_dialog_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The dialog call receives the exact validated L3 resolver result."""

    state, _task_result = _tool_result_surface_state()
    payload = l3_surface.build_text_surface_input_from_global_state(
        state,
        interaction_style_context="brief and natural",
    )
    surface_output = surface.build_degraded_text_surface(payload)

    class _DialogInvoker:
        def __init__(self) -> None:
            self.calls: list[object] = []

        async def ainvoke(self, messages, *, config):
            del config
            self.calls.append(messages)
            return SimpleNamespace(content='{"final_dialog": ["Result ready."]}')

    async def record_event(**_kwargs: object) -> dict[str, object]:
        return {}

    invoker = _DialogInvoker()
    monkeypatch.setattr(dialog_agent, "_dialog_generator_llm", invoker)
    monkeypatch.setattr(
        dialog_agent.llm_tracing,
        "record_llm_trace_step",
        record_event,
    )
    monkeypatch.setattr(
        dialog_agent.event_logging,
        "record_llm_stage_event",
        record_event,
    )

    final_dialog, disposition = await dialog_agent._render_dialog_candidate(
        surface_output=surface_output,
        user_name="Test User",
        repair_issues=[],
        attempt_number=1,
        llm_trace_id="l3-resolver-result-test",
    )

    assert final_dialog == ["Result ready."]
    assert disposition is None
    assert len(invoker.calls) == 1
    messages = invoker.calls[0]
    assert isinstance(messages, list)
    dialog_payload = json.loads(messages[1].content)
    assert dialog_payload["text_surface_output_v2"]["resolver_result"] == (
        payload["resolver_result"]
    )


def test_l3_surface_rejects_mismatched_tool_result_speak_continuation() -> None:
    """A visible task result must retain its exact cognition-owned lineage."""

    state, _task_result = _tool_result_surface_state()
    action_specs = state["action_specs"]
    assert isinstance(action_specs, list)
    assert isinstance(action_specs[0], dict)
    action_specs[0]["goal_continuation_ref"] = None

    with pytest.raises(ValueError, match="speak continuation reference"):
        l3_surface.build_text_surface_input_from_global_state(
            state,
            interaction_style_context="brief and natural",
        )


def test_l3_surface_omits_resolver_result_for_ordinary_episode() -> None:
    """Ordinary L3 input keeps the resolver lane absent."""

    payload = l3_surface.build_text_surface_input_from_global_state(
        build_surface_state(build_relational_decision()),
        interaction_style_context="brief and natural",
    )

    assert "resolver_result" not in payload


def test_persona_graph_retains_canonical_resolver_recurrence_fields() -> None:
    """Keep resolver closure available to downstream persona graph nodes."""

    assert {
        "resolver_state",
        "resolver_context",
        "resolver_capability_requests",
        "resolver_cycle_trace",
        "resolver_goal_progress",
    } <= GlobalPersonaState.__optional_keys__


def test_l3_terminal_resolver_surface_closes_stale_pending_plan() -> None:
    """A resolver terminal speak overrides the superseded pending P plan."""

    state = build_surface_state(build_relational_decision())
    output = state["cognition_core_output"]
    assert isinstance(output, dict)
    response_plan = output["response_plan"]
    assert isinstance(response_plan, dict)
    response_plan.update({
        "response_goal": "I will retry the blocked lookup.",
        "goal_resolution": "requires_required_evidence",
        "resolver_requests": [{"capability": "task_resolution_request"}],
    })
    observation = resolver_task_observation()
    observation.update({
        "status": "failed",
        "prompt_safe_summary": "The evidence path is terminally blocked.",
        "evidence_refs": [],
        "task_resolution_evidence_state": {
            "schema_version": "resolver_evidence_state.v1",
            "state": "blocked",
            "remaining_needs": ["current Christchurch weather"],
        },
    })
    terminal_detail = (
        "Explain that current evidence retrieval is blocked, keep current "
        "weather unknown, and close this turn without retry or deferred work."
    )
    terminal_spec = {
        "kind": "speak",
        "source_refs": [{"owner": "cognition_resolver"}],
        "params": {
            "surface_requirements": {
                "decision": TERMINAL_RESOLVER_SURFACE_DECISION,
                "detail": terminal_detail,
            },
        },
        "cognition_provenance": {
            "target_roles": [{
                "role": "target",
                "entity_kind": "user",
                "entity_id": "user-1",
            }],
        },
    }
    state["action_specs"] = [terminal_spec]
    state["resolver_state"] = {
        "schema_version": "resolver_cycle_state.v1",
        "cycle_index": 2,
        "max_cycles": 3,
        "status": "blocked",
        "original_decontextualized_input": "Get current Christchurch weather.",
        "observations": [observation],
        "cycle_traces": [],
        "held_action_specs": [terminal_spec],
        "required_resolver_evidence_dependency": {
            "schema_version": "required_resolver_evidence_dependency.v2",
            "accepted_request_handle": "resolver_request_0_1",
            "observation_id": observation["observation_id"],
        },
        "terminal_reason": "duplicate request converted to terminal surface",
    }

    payload = l3_surface.build_text_surface_input_from_global_state(
        state,
        interaction_style_context="brief and natural",
    )

    assert payload["resolver_result"]["status"] == "failed"
    assert payload["resolver_result"]["evidence_state"] == "blocked"
    assert payload["response_plan"] == {
        "response_goal": terminal_detail,
        "goal_resolution": "blocked",
        "epistemic_boundary": (
            "Assert the visible turn and keep unsupported details unknown."
        ),
        "action_requests": [],
        "resolver_requests": [],
        "surface_requirements": {
            "decision": TERMINAL_RESOLVER_SURFACE_DECISION,
            "detail": terminal_detail,
        },
        "terminal_work_disposition": "closed",
    }
    assert payload["subjective_expression_context"]["epistemic_boundary"] == (
        terminal_detail
    )


def test_l3_surface_projects_current_task_resolver_dependency() -> None:
    """A current task observation retains validated source-owned evidence."""

    observation = resolver_task_observation()
    continuation_ref = observation["goal_continuation_ref"]

    resolver_result = l3_surface._task_resolver_result(
        observation,
        required=True,
        continuation_ref=continuation_ref,
    )

    assert resolver_result["capability_kind"] == "task_resolution_request"
    assert resolver_result["status"] == "succeeded"
    assert resolver_result["evidence_handles"] == [
        "resolver_evidence_raw-tool-run-123_1"
    ]
    assert resolver_result["evidence_excerpts"] == ["bounded summary only"]


def test_l3_surface_rejects_missing_required_task_observation_before_planning(
) -> None:
    """L3 rejects a dangling V2 dependency before building prompt input."""

    state = build_surface_state(build_relational_decision())
    state["resolver_state"] = {
        "schema_version": "resolver_cycle_state.v1",
        "cycle_index": 0,
        "max_cycles": 3,
        "status": "terminal",
        "original_decontextualized_input": "Resolve one evidence goal.",
        "observations": [],
        "cycle_traces": [],
        "held_action_specs": [],
        "required_resolver_evidence_dependency": {
            "schema_version": "required_resolver_evidence_dependency.v2",
            "accepted_request_handle": "resolver_request_0_1",
            "observation_id": "missing-observation",
        },
        "terminal_reason": "invalid required evidence reference",
    }

    with pytest.raises(ResolverValidationError, match="unavailable"):
        l3_surface.build_text_surface_input_from_global_state(
            state,
            interaction_style_context="brief and natural",
        )


def test_l3_surface_omits_brain_owned_dsh_decision_from_prompt_projection() -> None:
    """Keep the DSH decision for Brain enactment outside L3 prompt input."""

    state = build_surface_state(build_relational_decision())
    cognition_output = state["cognition_core_output"]
    assert isinstance(cognition_output, dict)
    response_plan = cognition_output["response_plan"]
    assert isinstance(response_plan, dict)
    ordinary_response_plan = dict(response_plan)
    dsh_decision = {
        "interaction_id": "dsh-surface-interaction",
        "kind": "approval",
        "decision": "allow_once",
        "answer": None,
        "response_goal": None,
        "relay_mode": None,
        "reason": "The requested operation is permitted once.",
    }
    response_plan["dsh_interaction_decision"] = dsh_decision

    payload = l3_surface.build_text_surface_input_from_global_state(
        state,
        interaction_style_context="brief and natural",
    )

    assert payload["response_plan"] == ordinary_response_plan
    assert "dsh_interaction_decision" not in payload["response_plan"]
    assert response_plan["dsh_interaction_decision"] == dsh_decision
    prompt_payload = surface._project_surface_payload(payload)
    validate_prompt_projection(prompt_payload)


def test_l3_surface_excludes_pending_control_plane_fields() -> None:
    """L3 keeps pending disposition and timing outside visible semantics."""

    state = build_surface_state(build_relational_decision())
    cognition_output = state["cognition_core_output"]
    assert isinstance(cognition_output, dict)
    response_plan = cognition_output["response_plan"]
    assert isinstance(response_plan, dict)
    ordinary_response_plan = dict(response_plan)
    response_plan["pending_resolution"] = {
        "decision": "answered",
        "reason": "The current observation answered the clarification.",
    }
    response_plan["pending_task_continuation"] = {
        "schema_version": "pending_task_continuation.v1",
        "on_answered_clarification": "background_task_admission",
    }

    payload = l3_surface.build_text_surface_input_from_global_state(
        state,
        interaction_style_context="brief and natural",
    )

    assert payload["response_plan"] == ordinary_response_plan
    assert "pending_resolution" not in payload["response_plan"]
    assert "pending_task_continuation" not in payload["response_plan"]
    assert response_plan["pending_resolution"]["decision"] == "answered"
    assert response_plan["pending_task_continuation"][
        "on_answered_clarification"
    ] == "background_task_admission"
    prompt_payload = surface._project_surface_payload(payload)
    validate_prompt_projection(prompt_payload)


def test_l3_surface_projects_subjective_context_and_authoritative_addressee() -> None:
    """L3 receives exact G/P subjectivity and caller-owned target rows."""

    state = build_surface_state(build_relational_decision())
    payload = l3_surface.build_text_surface_input_from_global_state(
        state,
        interaction_style_context="brief and natural",
    )

    assert payload["subjective_expression_context"] == {
        "private_monologue": (
            "I feel attentive because this turn asks for a grounded response."
        ),
        "epistemic_boundary": (
            "Assert the visible turn and keep unsupported details unknown."
        ),
    }
    assert payload["addressee_plan"] == [{
        "handle": "current_user",
        "display_name": "Test User",
        "semantic_role": "direct_recipient",
        "wording_policy": "second_person_allowed",
    }]


def test_persona_supervisor2_l3_surface_exposes_owned_contract() -> None:
    """Keep the L3 surface builder attached to this source owner."""

    assert callable(l3_surface.build_text_surface_input_from_global_state)


def test_text_surface_input_requires_exact_bounded_overused_moves() -> None:
    """Require the single canonical L3 field with no alias or fallback."""

    state = build_surface_state(build_relational_decision())
    state["conversation_progress"] = {
        "overused_moves": [
            "the character already used a visible response maneuver",
            "the character already used a second response maneuver",
        ],
    }
    payload = l3_surface.build_text_surface_input_from_global_state(
        state,
        interaction_style_context="brief and natural",
    )

    assert payload["overused_moves"] == state["conversation_progress"]["overused_moves"]
    missing = dict(payload)
    missing.pop("overused_moves")
    with pytest.raises(CognitionContractError):
        validate_text_surface_input_canonical(missing)


def test_surface_payload_projects_exact_overused_moves() -> None:
    """Copy the accepted move list into the one content-planning packet."""

    state = build_surface_state(build_relational_decision())
    state["conversation_progress"] = {
        "overused_moves": ["first observed move", "second observed move"],
    }
    payload = l3_surface.build_text_surface_input_from_global_state(
        state,
        interaction_style_context="brief and natural",
    )

    projected = surface._project_surface_payload(payload)

    assert projected["overused_moves"] == payload["overused_moves"]


def test_l3_surface_receives_exact_current_participant_overused_moves() -> None:
    """L3 copies the current participant's bounded rows without rewriting."""

    state = build_surface_state(build_relational_decision())
    state["conversation_progress"] = {
        "overused_moves": [
            "first observed response move",
            "second observed response move",
            "third observed response move",
            "fourth observed response move",
        ],
    }

    payload = l3_surface.build_text_surface_input_from_global_state(
        state,
        interaction_style_context="brief and natural",
    )

    assert payload["overused_moves"] == (
        state["conversation_progress"]["overused_moves"]
    )


@pytest.mark.asyncio
async def test_text_surface_uses_one_content_call_and_deterministic_preference_projection(
    monkeypatch,
) -> None:
    """Text surface makes one content call and copies deterministic fields."""

    trace_rows: list[dict[str, object]] = []

    async def record_trace_step(**kwargs: object) -> dict[str, object]:
        trace_rows.append(kwargs)
        return {
            "accepted": True,
            "trace_id": "surface-trace",
            "status": "recorded",
            "reason": "",
        }

    def unexpected_direct_append(**kwargs: object) -> None:
        raise AssertionError("surface must not append model attempts directly")

    monkeypatch.setattr(
        surface_stages.llm_tracing,
        "record_llm_trace_step",
        record_trace_step,
    )
    monkeypatch.setattr(
        surface_stages.failure_capsule,
        "append_model_attempt",
        unexpected_direct_append,
    )

    class _Invoker:
        def __init__(self) -> None:
            self.calls: list[str] = []

        async def ainvoke(self, messages, *, config):
            self.calls.append(config.stage_name)
            result = {
                "content_plan": "acknowledge the current turn",
                "content_requirements": ["keep the answer grounded"],
                "delivery_profile": {
                    "lexical_register": "natural",
                    "sentence_shape": "complete",
                    "rhythm": "steady",
                    "hesitation": "none",
                    "punctuation": "clear",
                },
                "lexical_avoidances": [],
            }
            return SimpleNamespace(content=json.dumps(result))

    config = LLMCallConfig(
        stage_name="surface",
        route_name="test-surface",
        base_url="http://test",
        api_key="test",
        model="test",
        temperature=0.0,
        top_p=1.0,
        top_k=None,
        max_completion_tokens=512,
        presence_penalty=None,
        thinking=LLMThinkingConfig(enabled=False),
        context_window_tokens=4096,
    )
    invoker = _Invoker()
    services = TextSurfaceServicesV2(
        llm=invoker,
        content_plan_config=replace(config, stage_name="content-plan"),
    )
    state = build_surface_state(build_relational_decision())
    payload = l3_surface.build_text_surface_input_from_global_state(
        state,
        interaction_style_context="brief and natural",
    )
    trace_token = surface_stages.llm_tracing.bind_trace_id("surface-trace")
    try:
        output = await surface.run_text_surface_planning(payload, services)
    finally:
        surface_stages.llm_tracing.reset_trace_id(trace_token)

    assert invoker.calls == ["content-plan"]
    assert [row["stage_name"] for row in trace_rows] == [
        "surface.content_plan",
    ]
    assert [row["trace_id"] for row in trace_rows] == ["surface-trace"]
    assert all(row["status"] == "succeeded" for row in trace_rows)
    assert all(row["parsed_output"] for row in trace_rows)
    assert output["epistemic_boundary"] == (
        payload["subjective_expression_context"]["epistemic_boundary"]
    )
    assert output["visible_boundaries"] == []
    assert output["addressee_plan"] == payload["addressee_plan"]
    assert "private_monologue" not in output


@pytest.mark.asyncio
async def test_successful_content_plan_is_not_discarded_by_deterministic_projection(
    monkeypatch,
) -> None:
    """A valid content product reaches the validated public surface unchanged."""

    async def run_content(payload, services):
        return (
            "preserve this successful content plan",
            ["retain the grounded response"],
            {
                "lexical_register": "natural",
                "sentence_shape": "complete",
                "rhythm": "steady",
                "hesitation": "none",
                "punctuation": "clear",
            },
            [],
        )

    monkeypatch.setattr(surface, "run_content_plan_stage", run_content)
    state = build_surface_state(build_relational_decision())
    payload = l3_surface.build_text_surface_input_from_global_state(
        state,
        interaction_style_context="brief and natural",
    )
    services = SimpleNamespace(llm=object(), content_plan_config=object())

    output = await surface._run_text_surface_planning(
        payload,
        services,
        session=None,
    )

    assert output["content_plan"] == "preserve this successful content plan"
    assert output["epistemic_boundary"] == (
        payload["subjective_expression_context"]["epistemic_boundary"]
    )
    assert output["visible_boundaries"] == []
    assert output["addressee_plan"] == payload["addressee_plan"]
    assert "private_monologue" not in output


def test_text_surface_services_have_one_semantic_stage() -> None:
    """The injected text owner exposes only the content-plan configuration."""

    assert set(TextSurfaceServicesV2.__dataclass_fields__) == {
        "llm",
        "content_plan_config",
    }


@pytest.mark.asyncio
async def test_l3_handler_binds_state_trace_for_text_and_visual_calls(
    monkeypatch,
) -> None:
    """The L3 boundary shares and then restores the state trace context."""

    willingness = build_relational_decision()
    input_payload = {
        "episode": {"origin_metadata": {"debug_modes": {}}},
        "relational_willingness": willingness,
    }
    seen_trace_ids: dict[str, str] = {}

    async def load_style_context(state) -> str:
        return "style"

    def build_input(state, *, interaction_style_context):
        return input_payload

    async def run_text(payload, services):
        seen_trace_ids["text"] = surface_stages.llm_tracing.current_trace_id()
        return {"relational_willingness": payload["relational_willingness"]}

    async def run_visual(payload, services):
        seen_trace_ids["visual"] = surface_stages.llm_tracing.current_trace_id()
        return {"visual_directives": "a grounded visual direction"}

    monkeypatch.setattr(
        l3_surface,
        "_load_interaction_style_context",
        load_style_context,
    )
    monkeypatch.setattr(
        l3_surface,
        "build_text_surface_input_from_global_state",
        build_input,
    )
    monkeypatch.setattr(l3_surface, "run_text_surface_planning", run_text)
    monkeypatch.setattr(l3_surface, "run_visual_surface_planning", run_visual)

    outer_token = l3_surface.llm_tracing.bind_trace_id("outer-trace")
    try:
        result = await l3_surface.call_l3_text_surface_handler({
            "llm_trace_id": "state-trace",
        })
        assert result["visual_surface_output_v2"]["visual_directives"]
        assert seen_trace_ids == {
            "text": "state-trace",
            "visual": "state-trace",
        }
        assert l3_surface.llm_tracing.current_trace_id() == "outer-trace"
    finally:
        l3_surface.llm_tracing.reset_trace_id(outer_token)


@pytest.mark.asyncio
async def test_unexpected_visual_failure_is_omitted_and_text_surface_is_returned(
    monkeypatch,
) -> None:
    """Any non-cancellation visual failure leaves the text surface usable."""

    willingness = build_relational_decision()
    input_payload = {
        "episode": {"origin_metadata": {"debug_modes": {}}},
        "relational_willingness": willingness,
    }

    async def load_style_context(state) -> str:
        del state
        return "style"

    def build_input(state, *, interaction_style_context):
        del state, interaction_style_context
        return input_payload

    async def run_text(payload, services):
        del services
        return {"relational_willingness": payload["relational_willingness"]}

    async def run_visual(payload, services):
        del payload, services
        raise ValueError("visual contract failed")

    monkeypatch.setattr(
        l3_surface,
        "_load_interaction_style_context",
        load_style_context,
    )
    monkeypatch.setattr(
        l3_surface,
        "build_text_surface_input_from_global_state",
        build_input,
    )
    monkeypatch.setattr(l3_surface, "run_text_surface_planning", run_text)
    monkeypatch.setattr(l3_surface, "run_visual_surface_planning", run_visual)

    result = await l3_surface.call_l3_text_surface_handler({
        "llm_trace_id": "visual-failure-trace",
    })

    assert result["text_surface_output_v2"]["relational_willingness"] == (
        willingness
    )
    assert "visual_surface_output_v2" not in result
    assert result["attempt_diagnostics"] == [{
        "schema_version": "episode_attempt_diagnostic.v1",
        "stage": "surface.visual",
        "error_code": "surface_visual_omitted",
        "attempt_count": 3,
        "safe_checkpoint": "post_cognition_commit",
        "retryable": False,
        "final_status": "accepted_degraded",
    }]


@pytest.mark.asyncio
async def test_visual_cancellation_still_propagates(monkeypatch) -> None:
    """Cancellation remains fatal to the in-flight surface operation."""

    willingness = build_relational_decision()
    input_payload = {
        "episode": {"origin_metadata": {"debug_modes": {}}},
        "relational_willingness": willingness,
    }

    async def load_style_context(state) -> str:
        del state
        return "style"

    def build_input(state, *, interaction_style_context):
        del state, interaction_style_context
        return input_payload

    async def run_text(payload, services):
        del services
        return {"relational_willingness": payload["relational_willingness"]}

    async def run_visual(payload, services):
        del payload, services
        raise asyncio.CancelledError()

    monkeypatch.setattr(
        l3_surface,
        "_load_interaction_style_context",
        load_style_context,
    )
    monkeypatch.setattr(
        l3_surface,
        "build_text_surface_input_from_global_state",
        build_input,
    )
    monkeypatch.setattr(l3_surface, "run_text_surface_planning", run_text)
    monkeypatch.setattr(l3_surface, "run_visual_surface_planning", run_visual)

    with pytest.raises(asyncio.CancelledError):
        await l3_surface.call_l3_text_surface_handler({
            "llm_trace_id": "visual-cancel-trace",
        })
