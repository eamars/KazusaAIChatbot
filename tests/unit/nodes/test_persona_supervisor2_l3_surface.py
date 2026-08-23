"""Direct ownership tests for the L3 surface handoff."""

from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from kazusa_ai_chatbot.cognition_shared import surface, surface_stages
from kazusa_ai_chatbot.cognition_shared.contracts import TextSurfaceServicesV2
from kazusa_ai_chatbot.llm_interface import LLMCallConfig, LLMThinkingConfig
from kazusa_ai_chatbot.nodes import persona_supervisor2_l3_surface as l3_surface
from tests.unit.nodes.surface_fixtures import (
    build_relational_decision,
    build_surface_state,
)


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
    assert "same content requirement" in (
        surface_stages.CONTENT_PLAN_SYSTEM_PROMPT
    )
    assert "cannot support an exclusion" in (
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
    assert "输出前不可跳过的合同检查" in (
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
