"""Direct ownership tests for the L3 surface handoff."""

from __future__ import annotations

import asyncio
import inspect
import json
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest

from kazusa_ai_chatbot.background_work.result_source import (
    build_result_ready_episode_from_job,
)
from kazusa_ai_chatbot.cognition_shared import surface, surface_stages
from kazusa_ai_chatbot.cognition_shared.contracts import (
    CognitionContractError,
    TextSurfaceServicesV2,
    validate_text_surface_input_canonical,
)
from kazusa_ai_chatbot.cognition_shared.state_projection import (
    validate_prompt_projection,
)
from kazusa_ai_chatbot.llm_interface import LLMCallConfig, LLMThinkingConfig
from kazusa_ai_chatbot.nodes import dialog_agent
from kazusa_ai_chatbot.nodes import persona_supervisor2_l3_surface as l3_surface
from tests.test_background_work_delivery import _accepted_task_completed_job
from tests.test_cognition_resolver_contracts import _observation
from tests.unit.nodes.surface_fixtures import (
    build_relational_decision,
    build_surface_state,
)


def _tool_result_surface_state() -> tuple[dict[str, object], dict[str, object]]:
    """Build one L3 state carrying a validated completed task result."""

    job = deepcopy(_accepted_task_completed_job())
    task_result = job["task_resolution_result"]
    assert isinstance(task_result, dict)
    task_result["evidence_excerpts"] = ["PLAN3_E2E_BETA_SELECTED"]
    task_result["prompt_safe_summary"] = "The beta marker is ready."
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


def test_surface_prompts_are_complete_literals_with_owned_authority() -> None:
    """Keep surface prompt authority local to each complete literal."""

    source = inspect.getsource(surface_stages)
    prompt = surface_stages.CONTENT_PLAN_SYSTEM_PROMPT

    assert "VISIBLE_CONTENT_AUTHORITY_GUIDANCE" not in source
    assert "_CONTENT_PLAN_SYSTEM_PROMPT_TEMPLATE" not in source
    assert "SURFACE_REPAIR_INSTRUCTION" not in source
    assert ".format(" not in source
    assert prompt.count("可见语义的选择权属于") == 1
    assert "dialog 必须服从已选 content_plan" in prompt
    assert "# 输出格式" not in prompt
    assert "字段恰好是" not in prompt
    assert "# 输出格式" not in surface_stages.VISUAL_SYSTEM_PROMPT
    assert "visual_directives 只描述终端图像表面的可见角色特征" in (
        surface_stages.VISUAL_SYSTEM_PROMPT
    )


def test_content_plan_keeps_goal_reason_outside_visible_semantic_authority() -> None:
    """Interpretation fields cannot become an independent visible semantic source."""

    prompt = surface_stages.CONTENT_PLAN_SYSTEM_PROMPT
    for field in (
        "active_character_goal.reason",
        "active_character_goal.cause_summary",
        "intention.reason",
        "relational_willingness",
        "private_monologue",
    ):
        assert field in prompt
    assert prompt.count("可见语义的选择权属于") == 1
    assert "intention.reason 为语义锚点" not in (
        surface_stages.CONTENT_PLAN_SYSTEM_PROMPT
    )


def test_visible_content_authority_keeps_epistemic_examples_outside_content_selection() -> None:
    """Epistemic interpretations remain bounded context, not content sources."""

    prompt = surface_stages.CONTENT_PLAN_SYSTEM_PROMPT
    required_invariants = (
        "epistemic_boundary",
        "已选语义的断言强度",
        "解释或未知的内容不是可见内容候选",
        "不能独立进入 content_plan 或 dialog",
    )

    assert all(invariant in prompt for invariant in required_invariants)
    assert prompt.count("可见语义的选择权属于") == 1


def test_visible_content_authority_blocks_relationship_meaning_laundering_as_delivery() -> None:
    """Delivery fields cannot smuggle unselected relationship semantics."""

    prompt = surface_stages.CONTENT_PLAN_SYSTEM_PROMPT
    required_invariants = (
        "未选中的关系解释",
        "改写成感受、姿态、理由",
        "content_requirements",
        "delivery_profile",
        "不能绕过",
    )

    assert all(invariant in prompt for invariant in required_invariants)
    assert prompt.count("可见语义的选择权属于") == 1


def test_surface_repair_packet_projects_runtime_diagnostics_only() -> None:
    """Keep repair data separate from the unchanged complete prompt."""

    prompt_text, fitted_payload = surface_stages._surface_prompt_packet(
        {"observation": "current"},
        stage_name="content_plan",
        safe_checkpoint="pre_state_commit",
        system_prompt_chars=len(surface_stages.CONTENT_PLAN_SYSTEM_PROMPT),
    )
    messages = surface_stages._surface_repair_messages(
        payload=fitted_payload,
        system_prompt=surface_stages.CONTENT_PLAN_SYSTEM_PROMPT,
        invalid_candidate='{"unexpected": "field"}',
        reason="contract_error",
        contract_error="content-plan stage fields are not exact",
        stage_name="content_plan",
        safe_checkpoint="pre_state_commit",
        attempt_count=1,
    )
    first_payload = json.loads(prompt_text)
    payload = json.loads(messages[1].content)

    assert messages[0].content == surface_stages.CONTENT_PLAN_SYSTEM_PROMPT
    assert payload["surface"] == first_payload["surface"]
    assert payload["surface"]["output_contract"] == (
        surface_stages._CONTENT_PLAN_OUTPUT_CONTRACT
    )
    assert set(payload["contract_repair"]) == {
        "reason",
        "contract_error",
        "invalid_candidate",
    }
    assert payload["contract_repair"]["reason"] == "contract_error"
    assert "repair_instruction" not in json.dumps(payload)


@pytest.mark.asyncio
async def test_content_plan_rejects_nested_candidate_and_reuses_output_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Content planning regenerates from its unchanged typed packet contract."""

    trace_rows: list[dict[str, object]] = []

    async def record_trace_step(**kwargs: object) -> dict[str, object]:
        trace_rows.append(kwargs)
        return {}

    class _Invoker:
        def __init__(self) -> None:
            self.calls: list[object] = []
            self.outcomes = [
                {
                    "content_plan": [{"goal": "nested plan"}],
                    "delivery_profile": {
                        "lexical_register": "natural",
                        "sentence_shape": "complete",
                        "rhythm": "steady",
                        "hesitation": "none",
                        "punctuation": "clear",
                    },
                    "lexical_avoidances": [],
                    "dialog": "invalid nested candidate",
                },
                {
                    "content_plan": "Report PLAN3_E2E_BETA_SELECTED.",
                    "content_requirements": [
                        "Preserve PLAN3_E2E_BETA_SELECTED.",
                    ],
                    "delivery_profile": {
                        "lexical_register": "natural",
                        "sentence_shape": "complete",
                        "rhythm": "steady",
                        "hesitation": "none",
                        "punctuation": "clear",
                    },
                    "lexical_avoidances": [],
                },
            ]

        async def ainvoke(self, messages, *, config):
            del config
            self.calls.append(messages)
            outcome = self.outcomes.pop(0)
            return SimpleNamespace(content=json.dumps(outcome))

    config = LLMCallConfig(
        stage_name="content-plan",
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
    monkeypatch.setattr(
        surface_stages.llm_tracing,
        "record_llm_trace_step",
        record_trace_step,
    )
    invoker = _Invoker()

    result = await surface_stages._run_surface_stage(
        payload={"observation": "PLAN3_E2E_BETA_SELECTED"},
        system_prompt=surface_stages.CONTENT_PLAN_SYSTEM_PROMPT,
        llm=invoker,
        config=config,
        stage_name="content_plan",
        validator=surface_stages._validate_content_plan_result,
        safe_checkpoint="pre_state_commit",
    )

    first_payload = json.loads(invoker.calls[0][1].content)
    repair_payload = json.loads(invoker.calls[1][1].content)
    assert result[0] == "Report PLAN3_E2E_BETA_SELECTED."
    assert len(invoker.calls) == 2
    assert first_payload["surface"]["output_contract"] == (
        surface_stages._CONTENT_PLAN_OUTPUT_CONTRACT
    )
    assert repair_payload["surface"] == first_payload["surface"]
    assert set(repair_payload["contract_repair"]) == {
        "reason",
        "contract_error",
        "invalid_candidate",
    }
    assert "guidance" not in json.dumps(first_payload)
    assert "instruction" not in json.dumps(first_payload)
    assert all(row["parse_status"] == "contract_error" for row in trace_rows[:1])
    assert trace_rows[1]["parse_status"] == "succeeded"


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
    assert resolver_result["evidence_handles"] == ["public-evidence-1"]
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


def test_l3_surface_projects_current_task_resolver_dependency() -> None:
    """A current task observation retains validated source-owned evidence."""

    observation = _observation()
    continuation_ref = observation["goal_continuation_ref"]
    dependency = {
        "goal_continuation_ref": continuation_ref,
        "prompt_safe_observation_handle": "resolver_observation_0_1",
        "evidence_handles": ["evidence-handle-1"],
        "state": "complete",
        "remaining_needs": [],
    }

    resolver_result = l3_surface._task_resolver_result(
        observation,
        dependency=dependency,
        continuation_ref=continuation_ref,
    )

    assert resolver_result["capability_kind"] == "task_resolution_request"
    assert resolver_result["status"] == "succeeded"
    assert resolver_result["evidence_handles"] == ["evidence-handle-1"]
    assert resolver_result["evidence_excerpts"] == ["bounded summary only"]


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
    assert "content_requirements" in (
        surface_stages.CONTENT_PLAN_SYSTEM_PROMPT
    )
    assert "epistemic_boundary" in (
        surface_stages.CONTENT_PLAN_SYSTEM_PROMPT
    )
    assert "Preserve every caller-owned addressee_plan row exactly" not in (
        surface_stages.CONTENT_PLAN_SYSTEM_PROMPT
    )
    assert "direction and forms of address" not in (
        surface_stages.CONTENT_PLAN_SYSTEM_PROMPT
    )
    assert "动作舞台提示、拟声、感官反馈和结果反问" in (
        surface_stages.CONTENT_PLAN_SYSTEM_PROMPT
    )
    assert "没有 executed 行时" in surface_stages.CONTENT_PLAN_SYSTEM_PROMPT
    assert "空列表或没有 executed 行时" in (
        surface_stages.CONTENT_PLAN_SYSTEM_PROMPT
    )
    assert "action_kind=speak 只授权说出或发送文字" in (
        surface_stages.CONTENT_PLAN_SYSTEM_PROMPT
    )
    assert "同一类型、同一效果的 executed 行精确支持" in (
        surface_stages.CONTENT_PLAN_SYSTEM_PROMPT
    )
    assert "# 语义审计" in (
        surface_stages.CONTENT_PLAN_SYSTEM_PROMPT
    )
    assert "已表达的回应模式只属于背景连续性" in (
        surface_stages.CONTENT_PLAN_SYSTEM_PROMPT
    )
    assert "对未来外部效果的具体承诺也属于行动主张" in (
        surface_stages.CONTENT_PLAN_SYSTEM_PROMPT
    )
    assert "pending、scheduled 或 executed 行" in (
        surface_stages.CONTENT_PLAN_SYSTEM_PROMPT
    )
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


def test_text_surface_progression_contract_keeps_one_semantic_call() -> None:
    """Progression context stays inside the existing one-call surface stage."""

    assert set(TextSurfaceServicesV2.__dataclass_fields__) == {
        "llm",
        "content_plan_config",
    }
    assert "overused_moves" in surface_stages.CONTENT_PLAN_SYSTEM_PROMPT
    assert "已表达的回应模式只属于背景连续性" in (
        surface_stages.CONTENT_PLAN_SYSTEM_PROMPT
    )


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
