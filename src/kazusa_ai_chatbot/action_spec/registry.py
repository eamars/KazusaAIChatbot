"""Action capability registry and prompt-safe affordance projection."""

from __future__ import annotations

from collections.abc import Mapping

from kazusa_ai_chatbot.action_spec.models import (
    CapabilitySpecV1,
    validate_capability_spec,
)

SEND_MESSAGE_CAPABILITY = "send_message"
MEMORY_LIFECYCLE_UPDATE_CAPABILITY = "memory_lifecycle_update"


def build_initial_action_capabilities() -> dict[str, CapabilitySpecV1]:
    """Return the approved initial action capability registry."""

    capabilities = {
        SEND_MESSAGE_CAPABILITY: _send_message_capability(),
        MEMORY_LIFECYCLE_UPDATE_CAPABILITY: _memory_lifecycle_capability(),
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
        capability = capabilities[capability_kind]
        if capability_kind == SEND_MESSAGE_CAPABILITY:
            projection.append(_send_message_projection())
        elif capability_kind == MEMORY_LIFECYCLE_UPDATE_CAPABILITY:
            projection.append(_memory_lifecycle_projection())
    return projection


def _send_message_capability() -> CapabilitySpecV1:
    """Build the dispatcher-owned send-message capability spec."""

    return_value: CapabilitySpecV1 = {
        "schema_version": "capability_spec.v1",
        "capability_kind": SEND_MESSAGE_CAPABILITY,
        "category": "action",
        "owner_module": "dispatcher",
        "input_schema": {
            "type": "object",
            "required": [
                "target_channel",
                "text",
                "execute_at",
                "delivery_mentions",
            ],
            "properties": {
                "target_channel": {"type": "string"},
                "text": {"type": "string"},
                "execute_at": {"type": ["string", "null"]},
                "delivery_mentions": {"type": "array"},
            },
        },
        "output_schema": {
            "type": "object",
            "properties": {
                "scheduled_event_ids": {"type": "array"},
                "status": {"type": "string"},
            },
        },
        "handler_id": "dispatcher.send_message",
        "lifecycle_hooks": ["validate", "bridge_to_dispatcher"],
        "permission_policy": "policy:dispatcher.send_message.v1",
        "rate_limit_policy": "policy:action.default_rate_limit.v1",
        "audit_policy": "policy:action.audit.v1",
        "prompt_projection_policy": "policy:prompt.action_safe.v1",
    }
    return return_value


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


def _send_message_projection() -> dict[str, object]:
    """Return prompt-safe send-message affordance metadata."""

    return_value = {
        "capability": SEND_MESSAGE_CAPABILITY,
        "available": True,
        "visibility": "user_visible",
        "parameter_summary": [
            "target_channel: same or a permitted conversation surface",
            "text: final message text selected for delivery",
            "execute_at: ISO-8601 UTC time or null",
            "delivery_mentions: optional semantic mention requests",
        ],
        "continuation": "may request a bounded follow-up contract",
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
        "parameter_summary": [
            "memory_kind: user_memory_unit",
            "unit_type: active_commitment",
            "unit_id: commitment identifier supplied in evidence",
            "due_at: ISO-8601 UTC time or null",
        ],
        "continuation": "may request a bounded follow-up contract",
    }
    return return_value
