"""Checkpoint F V2 cognition-to-surface handoff tests."""

import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from kazusa_ai_chatbot.cognition_core_v2.contracts import (
    CognitionContractError,
)
from kazusa_ai_chatbot.nodes import persona_supervisor2 as persona_module
from kazusa_ai_chatbot.nodes import persona_supervisor2_l3_surface as l3_surface
from tests.cognition_core_v2_test_helpers import (
    canonical_cognition_output,
    canonical_episode,
)


class _LLM:
    """Return one bounded semantic result for every surface stage."""

    async def ainvoke(self, messages: list[object], *, config: object) -> SimpleNamespace:
        del config
        system = str(getattr(messages[0], "content", ""))
        json.loads(str(getattr(messages[-1], "content", "{}")))
        if "exactly style_guidance" in system:
            result = {"style_guidance": "style"}
        elif "exactly content_plan" in system:
            result = {"content_plan": "content plan"}
        elif "exactly visible_boundaries" in system:
            result = {
                "visible_boundaries": ["visible boundary"],
                "addressee_plan": ["current participant"],
            }
        elif "exactly pacing_guidance" in system:
            result = {"pacing_guidance": "pacing"}
        else:
            raise AssertionError("unexpected surface stage")
        return SimpleNamespace(content=json.dumps(result))


def _state() -> dict[str, object]:
    """Build a committed V2 output packet for the connector."""

    cognition_output = canonical_cognition_output()
    return {
        "storage_timestamp_utc": "2026-07-14T00:00:00Z",
        "user_input": "hello",
        "cognitive_episode": canonical_episode(
            episode_id="surface-episode",
            content="private exchange",
        ),
        "cognition_core_output": cognition_output,
        "action_results": [],
    }


def _services() -> object:
    from kazusa_ai_chatbot.cognition_core_v2.contracts import TextSurfaceServicesV2

    return TextSurfaceServicesV2(
        llm=_LLM(),
        style_config=object(),
        content_plan_config=object(),
        preference_config=object(),
        visual_config=object(),
    )


def test_l3_builder_uses_only_committed_v2_surface_fields() -> None:
    """The surface input carries intention and complete bid projections."""

    payload = l3_surface.build_text_surface_input_from_global_state(
        _state(),
        interaction_style_context="brief and natural",
    )

    assert payload["schema_version"] == "text_surface_input.v2"
    assert payload["primary_bid"]["desired_outcome"] == "maintain continuity"
    assert "entity_id" not in json.dumps(payload)


def test_l3_builder_projects_trace_status_into_exact_v2_action_result() -> None:
    """Legacy execution trace vocabulary stays outside the V2 surface API."""

    state = _state()
    state["action_results"] = [{
        "action_kind": "background_work_request",
        "status": "pending",
        "result_summary": "The accepted task was scheduled.",
    }]

    payload = l3_surface.build_text_surface_input_from_global_state(
        state,
        interaction_style_context="brief and natural",
    )

    assert payload["permitted_action_results"] == [{
        "action_kind": "background_work_request",
        "status": "completed",
        "semantic_result": "The accepted task was scheduled.",
        "target_roles": [],
    }]


def test_l3_builder_rejects_partial_cognition_output() -> None:
    """Surface planning requires the complete committed V2 result."""

    state = _state()
    state["cognition_core_output"] = {
        "intention": {
            "route": "speech",
            "intention": "partial output",
            "target_roles": [],
            "reason": "missing required V2 fields",
        },
    }

    with pytest.raises(CognitionContractError):
        l3_surface.build_text_surface_input_from_global_state(
            state,
            interaction_style_context="brief and natural",
        )


def test_l3_builder_projects_complete_supporting_bid_without_private_refs() -> None:
    """Supporting semantic content survives while internal ids are removed."""

    state = _state()
    output = state["cognition_core_output"]
    supporting = deepcopy(output["admitted_bid"])
    supporting.update({
        "branch_id": "social_care",
        "intention": "acknowledge the participant's effort",
        "desired_outcome": "add a small note of recognition",
        "concrete_detail": "the participant persisted with the task",
        "reason": "supporting evidence warrants recognition",
    })
    output["supporting_bids"] = [supporting]

    payload = l3_surface.build_text_surface_input_from_global_state(
        state,
        interaction_style_context="brief and natural",
    )

    assert payload["supporting_bids"][0]["permitted_detail"] == (
        "the participant persisted with the task"
    )
    assert "branch_id" not in payload["supporting_bids"][0]
    assert "goal_ref" not in payload["supporting_bids"][0]


@pytest.mark.asyncio
async def test_l3_surface_returns_semantic_plan_for_dialog() -> None:
    """Dialog receives an expression plan while retaining final wording ownership."""

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(l3_surface, "_build_surface_services", _services)
    async def _style_context(**kwargs: object) -> dict[str, object]:
        del kwargs
        return {
            "user_style": {
                "speech_guidelines": [],
                "social_guidelines": [],
                "pacing_guidelines": [],
                "engagement_guidelines": [],
                "confidence": "",
            },
            "application_order": ["user_style"],
        }

    monkeypatch.setattr(
        l3_surface,
        "build_interaction_style_context",
        _style_context,
    )
    try:
        update = await l3_surface.call_l3_text_surface_handler(_state())
    finally:
        monkeypatch.undo()

    assert update["text_surface_output_v2"]["schema_version"] == (
        "text_surface_output.v2"
    )
    assert update["text_surface_output_v2"]["content_plan"]
    assert "action_directives" not in update


def test_background_work_no_handoff_result_retains_semantic_task_brief() -> None:
    """A rejected queue handoff still explains the admitted semantic task."""

    result = persona_module._background_no_handoff_result(
        {
            "kind": "background_work_request",
            "params": {"task_brief": "Generate a Fibonacci function."},
        },
        _state(),
    )

    assert result["status"] == "failed"
    assert result["task_summary"] == "Generate a Fibonacci function."
    assert result["acknowledgement_constraint"] == (
        "promise_forbidden_explain_failure"
    )
