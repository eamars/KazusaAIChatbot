"""Checkpoint E integration tests for V2 facade and surface handoff."""

import json
from types import SimpleNamespace

import pytest

from kazusa_ai_chatbot.cognition_core_v2 import (
    run_cognition,
    run_text_surface_planning,
)
from kazusa_ai_chatbot.cognition_core_v2.contracts import (
    CognitionCoreServicesV2,
    TextSurfaceServicesV2,
)
from kazusa_ai_chatbot.cognition_core_v2.state_models import (
    build_acquaintance_user_state,
    build_character_production_state,
)

from llm_test_helpers import make_llm_call_config


NOW = "2026-07-14T00:00:00Z"


class _ScriptedLLM:
    """Return exact contract-shaped responses for each V2 stage."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def ainvoke(
        self,
        messages: list[object],
        *,
        config: object,
    ) -> SimpleNamespace:
        del config
        system = str(getattr(messages[0], "content", ""))
        human = str(getattr(messages[-1], "content", "{}"))
        payload = json.loads(human)
        if "scoped semantic question" in system:
            question = payload["question"]
            roles = question["permitted_role_handles"]
            result = {
                "question_id": question["question_id"],
                "selected_evidence_handles": ["ev1"],
                "selected_role_handles": roles[:1],
                "propositions": [],
                "deltas": [],
                "explanation": "the evidence is accepted without a new delta",
            }
        elif "independent goal cognition branch" in system:
            result = {
                "intention": "acknowledge the grounded episode",
                "desired_outcome": "maintain a coherent exchange",
                "concrete_detail": "use the current episode only",
                "reason": "the episode supplies bounded evidence",
                "target_role_handles": [],
                "evidence_handles": ["ev1"],
                "expected_consequences": ["preserve continuity"],
                "confidence": "high",
                "requested_route": "speech",
            }
        elif "Collapse complete goal bids" in system:
            handles = sorted(payload["bids"])
            result = {
                "primary_bid_handle": handles[0],
                "supporting_bid_handles": handles[1:],
                "suppressed_bid_handles": [],
            }
        elif "Select only a route" in system:
            result = {
                "selected_bid_handle": "b1",
                "route": "speech",
            }
        else:
            stage = payload["stage"]
            result = {"result": f"bounded {stage} guidance"}
        self.calls.append(system)
        return SimpleNamespace(content=json.dumps(result))


class _Logger:
    """No-op logger for injected V2 services."""

    def debug(self, message: str, *args: object, **kwargs: object) -> None:
        del message, args, kwargs

    info = debug
    warning = debug
    error = debug


def _core_services(llm: _ScriptedLLM) -> CognitionCoreServicesV2:
    """Build all core stage bindings from one scripted model."""

    return CognitionCoreServicesV2(
        llm=llm,
        appraisal_config=make_llm_call_config("v2_appraisal"),
        goal_cognition_config=make_llm_call_config("v2_goal"),
        collapse_config=make_llm_call_config("v2_collapse"),
        action_selection_config=make_llm_call_config("v2_route"),
        parse_json=json.loads,
        logger=_Logger(),
    )


def _surface_services(llm: _ScriptedLLM) -> TextSurfaceServicesV2:
    """Build all four surface stage bindings."""

    return TextSurfaceServicesV2(
        llm=llm,
        style_config=make_llm_call_config("v2_style"),
        content_plan_config=make_llm_call_config("v2_content"),
        preference_config=make_llm_call_config("v2_preference"),
        visual_config=make_llm_call_config("v2_visual"),
        parse_json=json.loads,
        logger=_Logger(),
    )


def _input() -> dict[str, object]:
    """Build one evidence-grounded user episode."""

    character = build_character_production_state(updated_at=NOW)
    return {
        "schema_version": "cognition_core_input.v2",
        "episode": {
            "episode_id": "e-integration",
            "semantic_scene": "private evidence-grounded exchange",
            "semantic_temporal_context": "immediate",
        },
        "state_scope": "user",
        "mutable_state": build_acquaintance_user_state(
            global_user_id="integration-user",
            updated_at=NOW,
        ),
        "character_constraints": {
            "drives": character["drives"],
            "standards": character["standards"],
            "meaning_state": character["meaning_state"],
        },
        "evidence": [{
            "evidence_handle": "ev1",
            "evidence_ref": {
                "source_kind": "episode",
                "source_id": "episode:e-integration",
                "occurred_at": NOW,
                "semantic_summary": "the user supplied a direct bounded episode",
            },
            "semantic_text": "the user supplied a direct bounded episode",
            "visible_to": ["cognition", "surface"],
        }],
        "direct_facts": [],
        "available_actions": [],
        "available_resolver_capabilities": [],
        "scene_context": {
            "channel_scope": "private",
            "character_role": "companion",
            "semantic_scene": "private evidence-grounded exchange",
            "semantic_temporal_context": "immediate",
        },
    }


@pytest.mark.asyncio
async def test_v2_facade_commits_before_surface_and_preserves_complete_bid() -> None:
    """The core returns one committed state update and complete admitted bid."""

    llm = _ScriptedLLM()
    output = await run_cognition(_input(), _core_services(llm))

    assert output["state_update"]["state_scope"] == "user"
    assert output["intention"]["route"] == "speech"
    assert output["admitted_bid"]["reason"] == "the episode supplies bounded evidence"
    assert output["diagnostics"]["completed_branch_count"] >= 1
    assert output["state_update"]["replacement_state"]["state_scope"] == "user"


@pytest.mark.asyncio
async def test_v2_surface_receives_semantic_handoff_only() -> None:
    """The four surface stages emit a bounded plan from non-private fields."""

    llm = _ScriptedLLM()
    input_payload = {
        "schema_version": "text_surface_input.v2",
        "episode": {
            "episode_id": "surface-episode",
            "semantic_scene": "private scene",
            "semantic_temporal_context": "immediate",
        },
        "intention": {
            "route": "speech",
            "intention": "acknowledge the episode",
            "target_roles": [],
            "reason": "grounded evidence",
        },
        "primary_bid": {
            "motive": "continuity",
            "intention": "acknowledge the episode",
            "desired_outcome": "maintain exchange",
            "permitted_detail": "current episode only",
            "target_summaries": [],
            "expected_consequences": ["preserve continuity"],
        },
        "supporting_bids": [],
        "expression_policy": {
            "visibility": "visible",
            "emotional_tone": "neutral",
            "intensity": "restrained",
            "directness": "balanced",
        },
        "semantic_affect": [],
        "permitted_action_results": [],
        "interaction_style_context": "calm and concise",
    }

    output = await run_text_surface_planning(input_payload, _surface_services(llm))

    assert output["schema_version"] == "text_surface_output.v2"
    assert output["content_plan"] == "bounded content_plan guidance"
    assert len(llm.calls) == 4
