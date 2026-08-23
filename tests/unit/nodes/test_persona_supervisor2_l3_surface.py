"""Direct ownership tests for the L3 surface handoff."""

from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from kazusa_ai_chatbot.cognition_shared import surface_stages
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


def test_persona_supervisor2_l3_surface_exposes_owned_contract() -> None:
    """Keep the L3 surface builder attached to this source owner."""

    assert callable(l3_surface.build_text_surface_input_from_global_state)


@pytest.mark.asyncio
async def test_surface_attempts_use_trace_recorder_once_without_direct_capsule_append(
    monkeypatch,
) -> None:
    """Successful content and preference calls share the protected recorder."""

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
        async def ainvoke(self, messages, *, config):
            if config.stage_name == "content-plan":
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
            else:
                result = {"visible_boundaries": [], "addressee_plan": []}
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
    services = TextSurfaceServicesV2(
        llm=_Invoker(),
        content_plan_config=replace(config, stage_name="content-plan"),
        preference_config=replace(config, stage_name="preference"),
    )
    trace_token = surface_stages.llm_tracing.bind_trace_id("surface-trace")
    try:
        await surface_stages.run_content_plan_stage(
            {"semantic": "turn"},
            services,
        )
        await surface_stages.run_preference_stage(
            {"semantic": "turn"},
            services,
        )
    finally:
        surface_stages.llm_tracing.reset_trace_id(trace_token)

    assert [row["stage_name"] for row in trace_rows] == [
        "surface.content_plan",
        "surface.preference",
    ]
    assert [row["trace_id"] for row in trace_rows] == ["surface-trace"] * 2
    assert all(row["status"] == "succeeded" for row in trace_rows)
    assert all(row["parsed_output"] for row in trace_rows)


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
