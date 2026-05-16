"""Tests for action result and episode trace helpers."""

from __future__ import annotations

import json

from kazusa_ai_chatbot.action_spec.evaluator import ActionSpecEvaluator
from kazusa_ai_chatbot.action_spec.results import (
    build_action_result,
    build_episode_trace,
    build_private_surface_output,
    build_text_surface_output,
    has_consolidatable_output,
    project_episode_trace_for_consolidation,
)


def _speak_action_spec() -> dict:
    return {
        "schema_version": "action_spec.v1",
        "kind": "speak",
        "cognition_mode": "deliberative",
        "source_refs": [
            {
                "schema_version": "action_source_ref.v1",
                "ref_kind": "cognitive_episode",
                "ref_id": "episode-001",
                "owner": "cognition_episode",
                "relationship": "basis",
                "evidence_refs": [],
            }
        ],
        "target": {
            "schema_version": "action_target.v1",
            "target_kind": "current_channel",
            "target_id": None,
            "owner": "l3_text",
            "scope": {"surface": "text"},
        },
        "params": {
            "delivery_mode": "visible_reply",
            "execute_at": None,
            "surface_requirements": {
                "decision": "visible_reply",
                "detail": "brief response",
            },
        },
        "urgency": "now",
        "visibility": "user_visible",
        "deadline": None,
        "continuation": {
            "schema_version": "action_continuation.v1",
            "mode": "none",
            "episode_type": None,
            "max_depth": 0,
            "include_result_as": None,
        },
        "reason": "The character selected a visible text surface.",
    }


def test_action_result_uses_evaluator_identity_without_raw_params() -> None:
    """Action results should be traceable without exposing action params."""

    action_spec = _speak_action_spec()
    eval_result = ActionSpecEvaluator().evaluate(action_spec)

    result = build_action_result(
        action_spec,
        eval_result,
        status="executed",
        result_summary="Text surface rendered.",
        completed_at="2026-05-16T00:00:00+00:00",
    )

    assert result["action_attempt_id"].startswith("action_attempt:")
    assert result["handler_owner"] == "l3_text"
    assert result["action_kind"] == "speak"
    assert result["status"] == "executed"
    assert "params" not in result
    assert "handler_id" not in result


def test_episode_trace_projection_omits_handler_ids_and_raw_params() -> None:
    """Consolidator projection should be prompt-safe action evidence."""

    action_spec = _speak_action_spec()
    eval_result = ActionSpecEvaluator().evaluate(action_spec)
    action_result = build_action_result(
        action_spec,
        eval_result,
        status="executed",
        result_summary="Text surface rendered.",
    )
    surface_output = build_text_surface_output(
        fragments=["hello"],
        created_at="2026-05-16T00:00:00+00:00",
        action_attempt_id=action_result["action_attempt_id"],
    )
    trace = build_episode_trace(
        episode_id="episode-001",
        trigger_source="user_message",
        created_at="2026-05-16T00:00:00+00:00",
        action_specs=[action_spec],
        action_results=[action_result],
        surface_outputs=[surface_output],
    )

    projection = project_episode_trace_for_consolidation(trace)
    serialized = json.dumps(projection, ensure_ascii=False)

    assert projection["action_results"][0]["action_kind"] == "speak"
    assert projection["surface_outputs"][0]["fragments"] == ["hello"]
    assert "handler_id" not in serialized
    assert "dispatcher.send_message" not in serialized
    assert "params" not in serialized
    assert "target_channel" not in serialized
    assert "mongodb" not in serialized.lower()


def test_private_surface_and_action_results_are_consolidatable() -> None:
    """Private or action-only episodes should not require visible dialog."""

    private_surface = build_private_surface_output(
        summary="Private finalization only.",
        created_at="2026-05-16T00:00:00+00:00",
    )

    assert has_consolidatable_output({"final_dialog": []}) is False
    assert has_consolidatable_output({
        "final_dialog": [],
        "surface_outputs": [private_surface],
    }) is True
    assert has_consolidatable_output({
        "final_dialog": [],
        "action_results": [{"status": "validated"}],
    }) is True
