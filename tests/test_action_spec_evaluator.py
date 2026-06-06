"""Tests for action-spec capability registry and evaluator behavior."""

from __future__ import annotations

import json

import pytest

from kazusa_ai_chatbot.action_spec.evaluator import ActionSpecEvaluator
from kazusa_ai_chatbot.action_spec.registry import (
    APPLY_MEMORY_LIFECYCLE_UPDATE_CAPABILITY,
    MEMORY_LIFECYCLE_UPDATE_CAPABILITY,
    SPEAK_CAPABILITY,
    TRIGGER_FUTURE_COGNITION_CAPABILITY,
    build_initial_action_capabilities,
    project_prompt_affordances,
)

BACKGROUND_ARTIFACT_REQUEST_CAPABILITY = "background_artifact_request"


def _cognitive_source_ref() -> dict:
    return {
        "schema_version": "action_source_ref.v1",
        "ref_kind": "cognitive_episode",
        "ref_id": "current_cognitive_episode",
        "owner": "cognition_episode",
        "relationship": "basis",
        "evidence_refs": [],
    }


def _memory_source_ref() -> dict:
    return {
        "schema_version": "action_source_ref.v1",
        "ref_kind": "memory_unit",
        "ref_id": "promise-001",
        "owner": "user_memory_units",
        "relationship": "target",
        "evidence_refs": [],
    }


def _source_refs_for_kind(kind: str) -> list[dict]:
    if kind == APPLY_MEMORY_LIFECYCLE_UPDATE_CAPABILITY:
        return [_cognitive_source_ref(), _memory_source_ref()]
    return [_cognitive_source_ref()]


def _target_for_kind(kind: str) -> dict:
    if kind == MEMORY_LIFECYCLE_UPDATE_CAPABILITY:
        return {
            "schema_version": "action_target.v1",
            "target_kind": "cognitive_episode",
            "target_id": None,
            "owner": "memory_lifecycle_specialist",
            "scope": {"unit_type": "active_commitment"},
        }
    if kind == APPLY_MEMORY_LIFECYCLE_UPDATE_CAPABILITY:
        return {
            "schema_version": "action_target.v1",
            "target_kind": "memory_unit",
            "target_id": "promise-001",
            "owner": "user_memory_units",
            "scope": {"unit_type": "active_commitment"},
        }
    if kind == SPEAK_CAPABILITY:
        return {
            "schema_version": "action_target.v1",
            "target_kind": "current_channel",
            "target_id": None,
            "owner": "l3_text",
            "scope": {"surface": "text"},
        }
    if kind == TRIGGER_FUTURE_COGNITION_CAPABILITY:
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
    if kind == MEMORY_LIFECYCLE_UPDATE_CAPABILITY:
        return {
            "review_kind": "active_commitment_lifecycle",
            "detail": "Review active commitments for lifecycle changes.",
        }
    if kind == APPLY_MEMORY_LIFECYCLE_UPDATE_CAPABILITY:
        return {
            "memory_kind": "user_memory_unit",
            "unit_type": "active_commitment",
            "unit_id": "promise-001",
            "lifecycle_decision": "abandoned",
            "due_at": "2026-05-07T00:00:00+00:00",
        }
    if kind == SPEAK_CAPABILITY:
        return {
            "delivery_mode": "visible_reply",
            "execute_at": None,
            "surface_requirements": {"tone": "brief"},
        }
    if kind == TRIGGER_FUTURE_COGNITION_CAPABILITY:
        return {
            "episode_type": "self_cognition",
            "trigger_at": "2026-05-16 00:30",
            "continuation_objective": "Re-evaluate the promise after a natural pause.",
        }
    raise AssertionError(f"unsupported action kind in test: {kind}")


def _action_spec(kind: str) -> dict:
    return {
        "schema_version": "action_spec.v1",
        "kind": kind,
        "cognition_mode": "deliberative",
        "source_refs": _source_refs_for_kind(kind),
        "target": _target_for_kind(kind),
        "params": _params_for_kind(kind),
        "urgency": (
            "scheduled"
            if kind == TRIGGER_FUTURE_COGNITION_CAPABILITY
            else "now"
        ),
        "visibility": (
            "private"
            if kind in (
                MEMORY_LIFECYCLE_UPDATE_CAPABILITY,
                APPLY_MEMORY_LIFECYCLE_UPDATE_CAPABILITY,
                TRIGGER_FUTURE_COGNITION_CAPABILITY,
            )
            else "user_visible"
        ),
        "deadline": None,
        "continuation": _no_continuation(),
        "reason": "The character selected this action from cognition.",
    }


def _background_artifact_action_spec() -> dict:
    return {
        "schema_version": "action_spec.v1",
        "kind": BACKGROUND_ARTIFACT_REQUEST_CAPABILITY,
        "cognition_mode": "deliberative",
        "source_refs": [_cognitive_source_ref()],
        "target": {
            "schema_version": "action_target.v1",
            "target_kind": "current_user",
            "target_id": None,
            "owner": "background_artifact",
            "scope": {
                "source_platform": "debug",
                "source_channel_id": "debug:user:test-user",
                "source_channel_type": "private",
                "source_message_id": "message-001",
                "source_platform_bot_id": "debug-bot-001",
                "source_character_name": "Test Character",
                "requester_global_user_id": (
                    "00000000-0000-4000-8000-000000000002"
                ),
                "requester_platform_user_id": "debug-user-001",
                "requester_display_name": "Test User",
            },
        },
        "params": {
            "work_kind": "coding_snippet",
            "objective": "Generate a Fibonacci function snippet.",
            "input_summary": "The user asked for a simple Fibonacci generator.",
            "requested_delivery": "send_result_when_done",
            "max_output_chars": 3000,
        },
        "urgency": "background",
        "visibility": "private",
        "deadline": None,
        "continuation": _no_continuation(),
        "reason": "The user requested bounded async snippet work.",
    }


def test_initial_registry_contains_only_approved_runtime_capabilities() -> None:
    """The first registry slice must not expose deferred future tools."""

    capabilities = build_initial_action_capabilities()

    assert set(capabilities) == {
        BACKGROUND_ARTIFACT_REQUEST_CAPABILITY,
        MEMORY_LIFECYCLE_UPDATE_CAPABILITY,
        APPLY_MEMORY_LIFECYCLE_UPDATE_CAPABILITY,
        SPEAK_CAPABILITY,
        TRIGGER_FUTURE_COGNITION_CAPABILITY,
    }
    assert (
        capabilities[BACKGROUND_ARTIFACT_REQUEST_CAPABILITY]["owner_module"]
        == "background_artifact"
    )
    assert (
        capabilities[MEMORY_LIFECYCLE_UPDATE_CAPABILITY]["owner_module"]
        == "memory_lifecycle_specialist"
    )
    assert (
        capabilities[APPLY_MEMORY_LIFECYCLE_UPDATE_CAPABILITY]["owner_module"]
        == "memory_lifecycle"
    )
    assert capabilities[SPEAK_CAPABILITY]["owner_module"] == "l3_text"
    assert (
        capabilities[TRIGGER_FUTURE_COGNITION_CAPABILITY]["owner_module"]
        == "orchestrator"
    )
    assert "send_message" not in capabilities
    assert "web_research" not in capabilities
    assert "schedule_self_check" not in capabilities
    assert "note_open_loop" not in capabilities


def test_background_artifact_route_schema_matches_router_contract() -> None:
    """L2d should see semantic async-artifact fields, not queue internals."""

    capabilities = build_initial_action_capabilities()
    capability = capabilities[BACKGROUND_ARTIFACT_REQUEST_CAPABILITY]
    schema = capability["input_schema"]
    properties = schema["properties"]

    assert schema["required"] == [
        "work_kind",
        "objective",
        "input_summary",
        "requested_delivery",
        "max_output_chars",
    ]
    assert properties["work_kind"]["enum"] == [
        "coding_snippet",
        "text_rewrite",
        "summary",
    ]
    assert properties["requested_delivery"]["enum"] == [
        "send_result_when_done",
    ]
    assert "job_id" not in properties
    assert "source_channel_id" not in properties
    assert "adapter_id" not in properties
    assert capability["category"] == "action"


def test_memory_lifecycle_route_schema_matches_router_contract() -> None:
    """The L2d route capability must not expose DB mutation parameters."""

    capabilities = build_initial_action_capabilities()
    capability = capabilities[MEMORY_LIFECYCLE_UPDATE_CAPABILITY]
    schema = capability["input_schema"]
    properties = schema["properties"]

    assert schema["required"] == [
        "review_kind",
        "detail",
    ]
    assert properties["review_kind"]["enum"] == [
        "active_commitment_lifecycle",
    ]
    assert "unit_id" not in properties
    assert "target_alias" not in properties
    assert "lifecycle_decision" not in properties
    assert capability["category"] == "action"


def test_apply_memory_lifecycle_schema_and_vocabulary_match_plan() -> None:
    """The executable lifecycle action should expose DB update vocabulary."""

    capabilities = build_initial_action_capabilities()
    capability = capabilities[APPLY_MEMORY_LIFECYCLE_UPDATE_CAPABILITY]
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
    assert "apply_memory_lifecycle_update" not in serialized
    assert "background_artifact_request" in serialized
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
        "job_id",
        "lease",
        "retry",
    ):
        assert forbidden not in serialized


def test_evaluator_rejects_reflex_for_all_current_capabilities() -> None:
    """Reflex mode is represented in schema but remains disabled at runtime."""

    evaluator = ActionSpecEvaluator(build_initial_action_capabilities())

    for kind in (
        MEMORY_LIFECYCLE_UPDATE_CAPABILITY,
        APPLY_MEMORY_LIFECYCLE_UPDATE_CAPABILITY,
        SPEAK_CAPABILITY,
        TRIGGER_FUTURE_COGNITION_CAPABILITY,
    ):
        action_spec = _action_spec(kind)
        action_spec["cognition_mode"] = "reflex"
        result = evaluator.evaluate(action_spec)
        assert result["ok"] is False
        assert any("reflex" in error for error in result["errors"])


def test_evaluator_accepts_speak_surface_action_without_dispatcher_bridge() -> None:
    """Text-surface selection is an L3 action, not a send-message tool call."""

    evaluator = ActionSpecEvaluator(build_initial_action_capabilities())

    result = evaluator.evaluate(_action_spec(SPEAK_CAPABILITY))

    assert result["ok"] is True
    assert result["handler_owner"] == "l3_text"


def test_evaluator_accepts_memory_lifecycle_route_intent_without_db_target() -> None:
    """L2d memory lifecycle specs route to the specialist, not the DB."""

    evaluator = ActionSpecEvaluator(build_initial_action_capabilities())

    result = evaluator.evaluate(_action_spec(MEMORY_LIFECYCLE_UPDATE_CAPABILITY))

    assert result["ok"] is True
    assert result["handler_owner"] == "memory_lifecycle_specialist"
    assert result["action_spec"]["target"]["target_kind"] == "cognitive_episode"
    assert (
        result["action_spec"]["target"]["owner"]
        == "memory_lifecycle_specialist"
    )
    assert all(
        source_ref["ref_kind"] != "memory_unit"
        for source_ref in result["action_spec"]["source_refs"]
    )
    assert "unit_id" not in result["action_spec"]["params"]


def test_evaluator_rejects_memory_lifecycle_route_with_memory_unit_ref() -> None:
    """Route intents must not smuggle a DB memory-unit binding."""

    evaluator = ActionSpecEvaluator(build_initial_action_capabilities())
    action_spec = _action_spec(MEMORY_LIFECYCLE_UPDATE_CAPABILITY)
    action_spec["source_refs"].append(_memory_source_ref())

    result = evaluator.evaluate(action_spec)

    assert result["ok"] is False
    assert any("memory_unit" in error for error in result["errors"])


def test_evaluator_accepts_apply_memory_lifecycle_executable_action() -> None:
    """Only the apply action carries a trusted user-memory unit target."""

    evaluator = ActionSpecEvaluator(build_initial_action_capabilities())

    result = evaluator.evaluate(
        _action_spec(APPLY_MEMORY_LIFECYCLE_UPDATE_CAPABILITY)
    )

    assert result["ok"] is True
    assert result["handler_owner"] == "memory_lifecycle"
    assert result["action_spec"]["target"]["target_kind"] == "memory_unit"
    assert result["action_spec"]["params"]["unit_id"] == "promise-001"


def test_evaluator_accepts_private_future_cognition_trigger() -> None:
    """Future cognition is a private orchestration request, not a tool call."""

    evaluator = ActionSpecEvaluator(build_initial_action_capabilities())

    action_spec = _action_spec(TRIGGER_FUTURE_COGNITION_CAPABILITY)
    action_spec["visibility"] = "private"
    result = evaluator.evaluate(action_spec)

    assert result["ok"] is True
    assert result["handler_owner"] == "orchestrator"


def test_background_artifact_request_validates_bounded_params() -> None:
    """Accepted async artifact work should validate before durable enqueue."""

    evaluator = ActionSpecEvaluator(build_initial_action_capabilities())

    result = evaluator.evaluate(_background_artifact_action_spec())

    assert result["ok"] is True
    assert result["handler_owner"] == "background_artifact"
    assert result["action_spec"]["target"]["target_kind"] == "current_user"
    assert result["action_spec"]["params"]["work_kind"] == "coding_snippet"
    assert result["action_spec"]["params"]["max_output_chars"] == 3000


def test_background_artifact_request_rejects_shell_scope() -> None:
    """Snippet work must not expand into shell or filesystem execution."""

    evaluator = ActionSpecEvaluator(build_initial_action_capabilities())
    action_spec = _background_artifact_action_spec()
    action_spec["params"]["work_kind"] = "coding_repo_edit"

    result = evaluator.evaluate(action_spec)

    assert result["ok"] is False
    assert any("work_kind" in error for error in result["errors"])


@pytest.mark.parametrize(
    "scope_field",
    (
        "source_channel_id",
        "requester_global_user_id",
    ),
)
def test_background_artifact_request_rejects_missing_delivery_target_scope(
    scope_field: str,
) -> None:
    """Async artifact enqueue must not acknowledge undeliverable jobs."""

    evaluator = ActionSpecEvaluator(build_initial_action_capabilities())
    action_spec = _background_artifact_action_spec()
    del action_spec["target"]["scope"][scope_field]

    result = evaluator.evaluate(action_spec)

    assert result["ok"] is False
    assert any(scope_field in error for error in result["errors"])


def test_evaluator_validates_continuation_contract() -> None:
    """Continuation requests must be structurally bounded before execution."""

    evaluator = ActionSpecEvaluator(build_initial_action_capabilities())
    action_spec = _action_spec(TRIGGER_FUTURE_COGNITION_CAPABILITY)
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
