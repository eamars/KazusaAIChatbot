"""Checkpoint F V2 cognition-to-surface handoff tests."""

import json
from types import SimpleNamespace

import pytest

from kazusa_ai_chatbot.nodes import persona_supervisor2_l3_surface as l3_surface


class _LLM:
    """Return one bounded semantic result for every surface stage."""

    async def ainvoke(self, messages: list[object], *, config: object) -> SimpleNamespace:
        del config
        payload = json.loads(str(getattr(messages[-1], "content", "{}")))
        return SimpleNamespace(content=json.dumps({"result": payload["stage"]}))


class _Logger:
    """No-op logger for the surface service fixture."""

    def debug(self, message: str, *args: object, **kwargs: object) -> None:
        del message, args, kwargs

    info = debug
    warning = debug
    error = debug


def _state() -> dict[str, object]:
    """Build a committed V2 output packet for the connector."""

    return {
        "storage_timestamp_utc": "2026-07-14T00:00:00Z",
        "user_input": "hello",
        "cognitive_episode": {
            "episode_id": "surface-episode",
            "semantic_scene": "private exchange",
        },
        "cognition_core_output": {
            "intention": {
                "selected_branch_id": "ordinary_response",
                "route": "speech",
                "intention": "acknowledge the exchange",
                "target_roles": [],
                "reason": "grounded episode",
            },
            "supporting_bids": [],
            "affect_projection": [],
            "expression_policy": {
                "visibility": "visible",
                "emotional_tone": "composed",
                "intensity": "restrained",
                "directness": "balanced",
            },
            "action_requests": [],
            "resolver_requests": [],
            "resolver_progress": {
                "status": "not_requested",
                "semantic_summary": "none",
            },
            "residue": "bounded residue",
            "admitted_bid": {
                "branch_id": "ordinary_response",
                "goal_ref": {
                    "scope": "user",
                    "kind": "episode",
                    "entity_id": "surface-episode",
                },
                "intention": "acknowledge the exchange",
                "desired_outcome": "maintain continuity",
                "concrete_detail": "current episode",
                "reason": "grounded episode",
                "target_roles": [],
                "evidence_handles": ["ev1"],
                "expected_consequences": ["continuity"],
                "confidence": "high",
                "requested_route": "speech",
            },
        },
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
        parse_json=json.loads,
        logger=_Logger(),
    )


def test_l3_builder_uses_only_committed_v2_surface_fields() -> None:
    """The surface input carries intention and complete bid projections."""

    payload = l3_surface.build_text_surface_input_from_global_state(_state())

    assert payload["schema_version"] == "text_surface_input.v2"
    assert payload["primary_bid"]["desired_outcome"] == "maintain continuity"
    assert "entity_id" not in json.dumps(payload)


@pytest.mark.asyncio
async def test_l3_surface_returns_semantic_plan_for_dialog() -> None:
    """Dialog receives an expression plan while retaining final wording ownership."""

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(l3_surface, "_build_surface_services", _services)
    try:
        update = await l3_surface.call_l3_text_surface_handler(_state())
    finally:
        monkeypatch.undo()

    assert update["text_surface_output_v2"]["schema_version"] == (
        "text_surface_output.v2"
    )
    assert update["text_surface_output_v2"]["content_plan"]
    assert "action_directives" not in update
