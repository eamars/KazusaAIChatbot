"""Tests for action-spec capability registry and evaluator behavior."""

from __future__ import annotations

import json

from kazusa_ai_chatbot.action_spec.evaluator import ActionSpecEvaluator
from kazusa_ai_chatbot.action_spec.registry import (
    build_initial_action_capabilities,
    project_prompt_affordances,
)


def _source_ref() -> dict:
    return {
        "schema_version": "action_source_ref.v1",
        "ref_kind": "memory_unit",
        "ref_id": "promise-001",
        "owner": "user_memory_units",
        "relationship": "basis",
        "evidence_refs": [],
    }


def _target_for_kind(kind: str) -> dict:
    if kind == "memory_lifecycle_update":
        return {
            "schema_version": "action_target.v1",
            "target_kind": "memory_unit",
            "target_id": "promise-001",
            "owner": "user_memory_units",
            "scope": {"unit_type": "active_commitment"},
        }
    if kind == "speak":
        return {
            "schema_version": "action_target.v1",
            "target_kind": "current_channel",
            "target_id": None,
            "owner": "l3_text",
            "scope": {"surface": "text"},
        }
    if kind == "trigger_future_cognition":
        return {
            "schema_version": "action_target.v1",
            "target_kind": "cognitive_episode",
            "target_id": None,
            "owner": "orchestrator",
            "scope": {"episode_type": "self_cognition"},
        }
    raise AssertionError(f"unsupported action kind in test: {kind}")


def _no_continuation() -> dict:
    return {
        "schema_version": "action_continuation.v1",
        "mode": "none",
        "episode_type": None,
        "max_depth": 0,
        "include_result_as": None,
    }


def _params_for_kind(kind: str) -> dict:
    if kind == "memory_lifecycle_update":
        return {
            "memory_kind": "user_memory_unit",
            "unit_type": "active_commitment",
            "unit_id": "promise-001",
            "lifecycle_decision": "abandoned",
            "due_at": "2026-05-07T00:00:00+00:00",
        }
    if kind == "speak":
        return {
            "delivery_mode": "visible_reply",
            "execute_at": None,
            "surface_requirements": {"tone": "brief"},
        }
    if kind == "trigger_future_cognition":
        return {
            "episode_type": "self_cognition",
            "trigger_at": "2026-05-16T00:30:00+00:00",
            "continuation_objective": "Re-evaluate the promise after a natural pause.",
        }
    raise AssertionError(f"unsupported action kind in test: {kind}")


def _action_spec(kind: str) -> dict:
    return {
        "schema_version": "action_spec.v1",
        "kind": kind,
        "cognition_mode": "deliberative",
        "source_refs": [_source_ref()],
        "target": _target_for_kind(kind),
        "params": _params_for_kind(kind),
        "urgency": "scheduled" if kind == "trigger_future_cognition" else "now",
        "visibility": (
            "private"
            if kind in ("memory_lifecycle_update", "trigger_future_cognition")
            else "user_visible"
        ),
        "deadline": None,
        "continuation": _no_continuation(),
        "reason": "The character selected this action from cognition.",
    }


def test_initial_registry_contains_only_approved_runtime_capabilities() -> None:
    """The first registry slice must not expose deferred future tools."""

    capabilities = build_initial_action_capabilities()

    assert set(capabilities) == {
        "memory_lifecycle_update",
        "speak",
        "trigger_future_cognition",
    }
    assert capabilities["memory_lifecycle_update"]["owner_module"] == "memory_lifecycle"
    assert capabilities["speak"]["owner_module"] == "l3_text"
    assert capabilities["trigger_future_cognition"]["owner_module"] == "orchestrator"
    assert "send_message" not in capabilities
    assert "web_research" not in capabilities
    assert "schedule_self_check" not in capabilities
    assert "note_open_loop" not in capabilities


def test_memory_lifecycle_schema_and_vocabulary_match_plan() -> None:
    """The lifecycle capability should expose the exact approved vocabulary."""

    capabilities = build_initial_action_capabilities()
    capability = capabilities["memory_lifecycle_update"]
    schema = capability["input_schema"]
    properties = schema["properties"]

    assert schema["required"] == [
        "memory_kind",
        "unit_type",
        "unit_id",
        "lifecycle_decision",
        "due_at",
    ]
    assert properties["memory_kind"]["enum"] == ["user_memory_unit"]
    assert properties["unit_type"]["enum"] == ["active_commitment"]
    assert properties["lifecycle_decision"]["enum"] == [
        "fulfilled",
        "abandoned",
        "obsolete",
        "deferred",
    ]
    assert capability["category"] == "action"


def test_prompt_affordance_projection_excludes_runtime_internals() -> None:
    """Prompt-visible affordances must not leak handlers, storage, or raw IDs."""

    capabilities = build_initial_action_capabilities()
    projection = project_prompt_affordances(capabilities)
    serialized = json.dumps(projection, sort_keys=True).lower()

    assert "memory_lifecycle_update" in serialized
    assert "speak" in serialized
    assert "trigger_future_cognition" in serialized
    assert "send_message" not in serialized
    for forbidden in (
        "handler_id",
        "dispatcher.send_message",
        "l3_text",
        "self_cognition_action_attempts",
        "user_memory_units",
        "mongodb",
        "mongo",
        "credential",
        "adapter_id",
        "platform_channel_id",
        "channel_id",
        "raw_channel",
    ):
        assert forbidden not in serialized


def test_evaluator_rejects_reflex_for_all_current_capabilities() -> None:
    """Reflex mode is represented in schema but remains disabled at runtime."""

    evaluator = ActionSpecEvaluator(build_initial_action_capabilities())

    for kind in (
        "memory_lifecycle_update",
        "speak",
        "trigger_future_cognition",
    ):
        action_spec = _action_spec(kind)
        action_spec["cognition_mode"] = "reflex"
        result = evaluator.evaluate(action_spec)
        assert result["ok"] is False
        assert any("reflex" in error for error in result["errors"])


def test_evaluator_accepts_speak_surface_action_without_dispatcher_bridge() -> None:
    """Text-surface selection is an L3 action, not a send-message tool call."""

    evaluator = ActionSpecEvaluator(build_initial_action_capabilities())

    result = evaluator.evaluate(_action_spec("speak"))

    assert result["ok"] is True
    assert result["handler_owner"] == "l3_text"


def test_evaluator_accepts_private_future_cognition_trigger() -> None:
    """Future cognition is a private orchestration request, not a tool call."""

    evaluator = ActionSpecEvaluator(build_initial_action_capabilities())

    action_spec = _action_spec("trigger_future_cognition")
    action_spec["visibility"] = "private"
    result = evaluator.evaluate(action_spec)

    assert result["ok"] is True
    assert result["handler_owner"] == "orchestrator"


def test_evaluator_validates_continuation_contract() -> None:
    """Continuation requests must be structurally bounded before execution."""

    evaluator = ActionSpecEvaluator(build_initial_action_capabilities())
    action_spec = _action_spec("trigger_future_cognition")
    action_spec["continuation"] = {
        "schema_version": "action_continuation.v1",
        "mode": "immediate_followup",
        "episode_type": None,
        "max_depth": 1,
        "include_result_as": "tool_result",
    }

    result = evaluator.evaluate(action_spec)

    assert result["ok"] is False
    assert any("episode_type" in error for error in result["errors"])
