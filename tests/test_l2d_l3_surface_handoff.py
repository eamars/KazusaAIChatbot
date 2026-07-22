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
        if "style_guidance" in system and "content_plan" not in system:
            result = {"style_guidance": "style"}
        elif "content_plan" in system and "content_requirements" in system:
            result = {
                "content_plan": "content plan",
                "content_requirements": ["preserve the current addressee"],
            }
        elif "visible_boundaries" in system and "addressee_plan" in system:
            result = {
                "visible_boundaries": ["visible boundary"],
                "addressee_plan": ["current participant"],
            }
        elif "visual_directives" in system:
            result = {"visual_directives": "private image composition"}
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
        "character_profile": _character_profile(),
    }


def _character_profile() -> dict[str, object]:
    """Build the required wording-only character voice source."""

    return {
        "name": "Kazusa",
        "personality_brief": {
            "logic": "analytical",
            "tempo": "moderate",
            "defense": "reserved",
            "quirks": "occasional hesitation",
            "taboos": "stay in character",
        },
        "linguistic_texture_profile": {
            "hesitation_density": 0.4,
            "fragmentation": 0.4,
            "emotional_leakage": 0.4,
            "rhythmic_bounce": 0.4,
            "direct_assertion": 0.4,
            "softener_density": 0.4,
            "counter_questioning": 0.4,
            "formalism_avoidance": 0.4,
            "abstraction_reframing": 0.4,
            "self_deprecation": 0.4,
        },
    }


def _services() -> object:
    from kazusa_ai_chatbot.cognition_core_v2.contracts import TextSurfaceServicesV2

    return TextSurfaceServicesV2(
        llm=_LLM(),
        style_config=object(),
        content_plan_config=object(),
        preference_config=object(),
    )


def _visual_services() -> object:
    from kazusa_ai_chatbot.cognition_core_v2.contracts import (
        VisualSurfaceServicesV2,
    )

    return VisualSurfaceServicesV2(
        llm=_LLM(),
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
        "status": "pending",
        "semantic_result": "The accepted task was scheduled.",
        "target_roles": [],
    }]


@pytest.mark.asyncio
async def test_text_surface_output_carries_exact_action_result_authority() -> None:
    """Dialog receives deterministic action lifecycle truth, not L3 inference."""

    state = _state()
    state["action_results"] = [{
        "action_kind": "background_work_request",
        "status": "scheduled",
        "result_summary": "The accepted task is scheduled.",
    }]
    input_payload = l3_surface.build_text_surface_input_from_global_state(
        state,
        interaction_style_context="brief and natural",
    )

    from kazusa_ai_chatbot.cognition_core_v2.surface import (
        run_text_surface_planning,
    )

    output = await run_text_surface_planning(input_payload, _services())

    assert output["permitted_action_results"] == [{
        "action_kind": "background_work_request",
        "status": "scheduled",
        "semantic_result": "The accepted task is scheduled.",
        "target_roles": [],
    }]


def test_surface_prompt_projects_action_roles_without_identity_leak() -> None:
    """L3 sees target semantics while the exact output ledger retains ids."""

    from kazusa_ai_chatbot.cognition_core_v2.surface import (
        _project_action_results_for_prompt,
    )

    projected = _project_action_results_for_prompt([{
        "action_kind": "future_speak",
        "status": "scheduled",
        "semantic_result": "The reminder is scheduled.",
        "target_roles": [{
            "role": "target",
            "entity_kind": "user",
            "entity_id": "private-user-id",
        }],
    }])

    assert projected == [{
        "action_kind": "future_speak",
        "status": "scheduled",
        "semantic_result": "The reminder is scheduled.",
        "target_roles": [{"role": "target", "entity_kind": "user"}],
    }]
    assert "private-user-id" not in json.dumps(projected)


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
    monkeypatch.setattr(l3_surface, "_build_text_surface_services", _services)
    monkeypatch.setattr(
        l3_surface,
        "_build_visual_surface_services",
        _visual_services,
    )
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
    assert update["visual_surface_output_v2"] == {
        "schema_version": "visual_surface_output.v2",
        "visual_directives": "private image composition",
        "selected_surface_intent": "acknowledge the grounded episode",
    }
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

    assert result["action_kind"] == "background_work_request"
    assert result["status"] == "failed"
    assert result["handler_owner"] == "background_work"
    assert result["task_summary"] == "Generate a Fibonacci function."
    assert result["acknowledgement_constraint"] == (
        "promise_forbidden_explain_failure"
    )
