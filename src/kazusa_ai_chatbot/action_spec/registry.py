"""Action capability registry and prompt-safe affordance projection."""

from __future__ import annotations

from collections.abc import Mapping

from kazusa_ai_chatbot.action_spec.models import (
    CapabilitySpecV1,
    validate_capability_spec,
)

MEMORY_LIFECYCLE_UPDATE_CAPABILITY = "memory_lifecycle_update"
SPEAK_CAPABILITY = "speak"
TRIGGER_FUTURE_COGNITION_CAPABILITY = "trigger_future_cognition"


def build_initial_action_capabilities() -> dict[str, CapabilitySpecV1]:
    """Return the approved initial action capability registry."""

    capabilities = {
        MEMORY_LIFECYCLE_UPDATE_CAPABILITY: _memory_lifecycle_capability(),
        SPEAK_CAPABILITY: _speak_capability(),
        TRIGGER_FUTURE_COGNITION_CAPABILITY: _future_cognition_capability(),
    }
    for capability in capabilities.values():
        validate_capability_spec(capability)
    return capabilities


def project_prompt_affordances(
    capabilities: Mapping[str, CapabilitySpecV1],
) -> list[dict[str, object]]:
    """Project capability metadata into prompt-safe semantic affordances."""

    projection: list[dict[str, object]] = []
    for capability_kind in sorted(capabilities):
        if capability_kind == MEMORY_LIFECYCLE_UPDATE_CAPABILITY:
            projection.append(_memory_lifecycle_projection())
        elif capability_kind == SPEAK_CAPABILITY:
            projection.append(_speak_projection())
        elif capability_kind == TRIGGER_FUTURE_COGNITION_CAPABILITY:
            projection.append(_future_cognition_projection())
    return projection


def _memory_lifecycle_capability() -> CapabilitySpecV1:
    """Build the memory-lifecycle capability spec."""

    return_value: CapabilitySpecV1 = {
        "schema_version": "capability_spec.v1",
        "capability_kind": MEMORY_LIFECYCLE_UPDATE_CAPABILITY,
        "category": "action",
        "owner_module": "memory_lifecycle",
        "input_schema": {
            "type": "object",
            "required": [
                "memory_kind",
                "unit_type",
                "unit_id",
                "lifecycle_decision",
                "due_at",
            ],
            "properties": {
                "memory_kind": {"type": "string", "enum": ["user_memory_unit"]},
                "unit_type": {"type": "string", "enum": ["active_commitment"]},
                "unit_id": {"type": "string"},
                "lifecycle_decision": {
                    "type": "string",
                    "enum": ["fulfilled", "abandoned", "obsolete", "deferred"],
                },
                "due_at": {"type": ["string", "null"]},
            },
        },
        "output_schema": {
            "type": "object",
            "properties": {
                "status": {"type": "string"},
                "unit_id": {"type": "string"},
            },
        },
        "handler_id": "action_spec.memory_lifecycle.user_memory_unit.v1",
        "lifecycle_hooks": ["validate", "repository_update"],
        "permission_policy": "policy:memory_lifecycle.private_commitment.v1",
        "rate_limit_policy": "policy:action.default_rate_limit.v1",
        "audit_policy": "policy:action.audit.v1",
        "prompt_projection_policy": "policy:prompt.action_safe.v1",
    }
    return return_value


def _speak_capability() -> CapabilitySpecV1:
    """Build the L3-text-owned speak surface capability spec."""

    return_value: CapabilitySpecV1 = {
        "schema_version": "capability_spec.v1",
        "capability_kind": SPEAK_CAPABILITY,
        "category": "action",
        "owner_module": "l3_text",
        "input_schema": {
            "type": "object",
            "required": [
                "delivery_mode",
                "execute_at",
                "surface_requirements",
            ],
            "properties": {
                "delivery_mode": {
                    "type": "string",
                    "enum": [
                        "visible_reply",
                        "private_finalization",
                        "delayed",
                        "scheduled",
                    ],
                },
                "execute_at": {"type": ["string", "null"]},
                "surface_requirements": {"type": "object"},
            },
        },
        "output_schema": {
            "type": "object",
            "properties": {
                "surface": {"type": "string"},
                "status": {"type": "string"},
            },
        },
        "handler_id": "l3_text.speak",
        "lifecycle_hooks": ["validate", "route_to_text_surface"],
        "permission_policy": "policy:l3_text.speak.v1",
        "rate_limit_policy": "policy:action.default_rate_limit.v1",
        "audit_policy": "policy:action.audit.v1",
        "prompt_projection_policy": "policy:prompt.action_safe.v1",
    }
    return return_value


def _future_cognition_capability() -> CapabilitySpecV1:
    """Build the orchestrator-owned future-cognition request capability."""

    return_value: CapabilitySpecV1 = {
        "schema_version": "capability_spec.v1",
        "capability_kind": TRIGGER_FUTURE_COGNITION_CAPABILITY,
        "category": "action",
        "owner_module": "orchestrator",
        "input_schema": {
            "type": "object",
            "required": [
                "episode_type",
                "trigger_at",
                "continuation_objective",
            ],
            "properties": {
                "episode_type": {"type": "string", "enum": ["self_cognition"]},
                "trigger_at": {"type": ["string", "null"]},
                "continuation_objective": {"type": "string"},
            },
        },
        "output_schema": {
            "type": "object",
            "properties": {
                "status": {"type": "string"},
                "episode_type": {"type": "string"},
            },
        },
        "handler_id": "orchestrator.future_cognition.request.v1",
        "lifecycle_hooks": ["validate", "enqueue_cognition_request"],
        "permission_policy": "policy:orchestrator.future_cognition.v1",
        "rate_limit_policy": "policy:action.default_rate_limit.v1",
        "audit_policy": "policy:action.audit.v1",
        "prompt_projection_policy": "policy:prompt.action_safe.v1",
    }
    return return_value


def _future_cognition_projection() -> dict[str, object]:
    """Return prompt-safe future-cognition affordance metadata."""

    return_value = {
        "capability": TRIGGER_FUTURE_COGNITION_CAPABILITY,
        "available": True,
        "visibility": "private",
        "semantic_input_summary": [
            "Use when the character wants a later private cognition cycle.",
            "Provide the semantic reason and any ordinary-language timing hint.",
        ],
        "execution_boundary": "downstream scheduler builds the executable request",
    }
    return return_value


def _speak_projection() -> dict[str, object]:
    """Return prompt-safe text-surface affordance metadata."""

    return_value = {
        "capability": SPEAK_CAPABILITY,
        "available": True,
        "visibility": "user_visible",
        "semantic_input_summary": [
            "Use when the character wants a text surface to exist.",
            "Provide the semantic surface intent, not final wording.",
        ],
        "execution_boundary": "downstream text surface builds the surface details",
    }
    return return_value


def _memory_lifecycle_projection() -> dict[str, object]:
    """Return prompt-safe memory-lifecycle affordance metadata."""

    return_value = {
        "capability": MEMORY_LIFECYCLE_UPDATE_CAPABILITY,
        "available": True,
        "visibility": "private",
        "allowed_lifecycle_decisions": [
            "fulfilled",
            "abandoned",
            "obsolete",
            "deferred",
        ],
        "semantic_input_summary": [
            "Use when the character semantically changes a commitment lifecycle.",
            "Choose the lifecycle decision and describe the affected commitment.",
        ],
        "execution_boundary": "downstream lifecycle owner resolves persistence fields",
    }
    return return_value
