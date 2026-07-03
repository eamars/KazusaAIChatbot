"""Action capability registry and prompt-safe affordance projection."""

from __future__ import annotations

from collections.abc import Mapping

from kazusa_ai_chatbot.action_spec.models import (
    CapabilitySpecV1,
    validate_capability_spec,
)

MEMORY_LIFECYCLE_UPDATE_CAPABILITY = "memory_lifecycle_update"
APPLY_MEMORY_LIFECYCLE_UPDATE_CAPABILITY = "apply_memory_lifecycle_update"
SPEAK_CAPABILITY = "speak"
TRIGGER_FUTURE_COGNITION_CAPABILITY = "trigger_future_cognition"
FUTURE_SPEAK_CAPABILITY = "future_speak"
ACCEPTED_TASK_STATUS_CHECK_CAPABILITY = "accepted_task_status_check"
BACKGROUND_ARTIFACT_REQUEST_CAPABILITY = "background_artifact_request"
BACKGROUND_WORK_REQUEST_CAPABILITY = "background_work_request"


def build_initial_action_capabilities() -> dict[str, CapabilitySpecV1]:
    """Return the approved initial action capability registry."""

    capabilities = {
        MEMORY_LIFECYCLE_UPDATE_CAPABILITY: _memory_lifecycle_capability(),
        APPLY_MEMORY_LIFECYCLE_UPDATE_CAPABILITY: (
            _apply_memory_lifecycle_capability()
        ),
        SPEAK_CAPABILITY: _speak_capability(),
        TRIGGER_FUTURE_COGNITION_CAPABILITY: _future_cognition_capability(),
        FUTURE_SPEAK_CAPABILITY: _future_speak_capability(),
        ACCEPTED_TASK_STATUS_CHECK_CAPABILITY: (
            _accepted_task_status_check_capability()
        ),
        BACKGROUND_ARTIFACT_REQUEST_CAPABILITY: (
            _background_artifact_capability()
        ),
        BACKGROUND_WORK_REQUEST_CAPABILITY: _background_work_capability(),
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
        elif capability_kind == FUTURE_SPEAK_CAPABILITY:
            projection.append(_future_speak_projection())
        elif capability_kind == ACCEPTED_TASK_STATUS_CHECK_CAPABILITY:
            projection.append(_accepted_task_status_check_projection())
        elif capability_kind == BACKGROUND_ARTIFACT_REQUEST_CAPABILITY:
            continue
        elif capability_kind == BACKGROUND_WORK_REQUEST_CAPABILITY:
            projection.append(_background_work_projection())
    return projection


def _memory_lifecycle_capability() -> CapabilitySpecV1:
    """Build the memory-lifecycle capability spec."""

    return_value: CapabilitySpecV1 = {
        "schema_version": "capability_spec.v1",
        "capability_kind": MEMORY_LIFECYCLE_UPDATE_CAPABILITY,
        "category": "action",
        "owner_module": "memory_lifecycle_specialist",
        "input_schema": {
            "type": "object",
            "required": [
                "review_kind",
                "detail",
            ],
            "properties": {
                "review_kind": {
                    "type": "string",
                    "enum": ["active_commitment_lifecycle"],
                },
                "detail": {"type": "string"},
            },
        },
        "output_schema": {
            "type": "object",
            "properties": {
                "status": {"type": "string"},
            },
        },
        "handler_id": "memory_lifecycle.specialist.route.v1",
        "lifecycle_hooks": ["route_to_specialist"],
        "permission_policy": "policy:memory_lifecycle.private_review.v1",
        "rate_limit_policy": "policy:action.default_rate_limit.v1",
        "audit_policy": "policy:action.audit.v1",
        "prompt_projection_policy": "policy:prompt.action_safe.v1",
    }
    return return_value


def _apply_memory_lifecycle_capability() -> CapabilitySpecV1:
    """Build the executable memory-lifecycle capability spec."""

    return_value: CapabilitySpecV1 = {
        "schema_version": "capability_spec.v1",
        "capability_kind": APPLY_MEMORY_LIFECYCLE_UPDATE_CAPABILITY,
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


def _future_speak_capability() -> CapabilitySpecV1:
    """Build the background-work-owned future-speak request capability."""

    return_value: CapabilitySpecV1 = {
        "schema_version": "capability_spec.v1",
        "capability_kind": FUTURE_SPEAK_CAPABILITY,
        "category": "action",
        "owner_module": "background_work",
        "input_schema": {
            "type": "object",
            "required": [
                "trigger_at",
                "continuation_objective",
                "requested_delivery",
                "max_output_chars",
            ],
            "properties": {
                "trigger_at": {"type": "string"},
                "continuation_objective": {"type": "string"},
                "requested_delivery": {
                    "type": "string",
                    "enum": ["send_result_when_done"],
                },
                "max_output_chars": {"type": "integer"},
            },
        },
        "output_schema": {
            "type": "object",
            "properties": {
                "status": {"type": "string"},
                "queue_state": {"type": "string"},
                "task_summary": {"type": "string"},
            },
        },
        "handler_id": "background_work.future_speak.enqueue.v1",
        "lifecycle_hooks": ["validate", "enqueue_background_work"],
        "permission_policy": "policy:background_work.future_speak.v1",
        "rate_limit_policy": "policy:action.default_rate_limit.v1",
        "audit_policy": "policy:action.audit.v1",
        "prompt_projection_policy": "policy:prompt.action_safe.v1",
    }
    return return_value


def _background_artifact_capability() -> CapabilitySpecV1:
    """Build the text-only background artifact request capability."""

    return_value: CapabilitySpecV1 = {
        "schema_version": "capability_spec.v1",
        "capability_kind": BACKGROUND_ARTIFACT_REQUEST_CAPABILITY,
        "category": "action",
        "owner_module": "background_artifact",
        "input_schema": {
            "type": "object",
            "required": [
                "work_kind",
                "objective",
                "input_summary",
                "requested_delivery",
                "max_output_chars",
            ],
            "properties": {
                "work_kind": {
                    "type": "string",
                    "enum": ["coding_snippet", "text_rewrite", "summary"],
                },
                "objective": {"type": "string"},
                "input_summary": {"type": "string"},
                "requested_delivery": {
                    "type": "string",
                    "enum": ["send_result_when_done"],
                },
                "max_output_chars": {"type": "integer"},
            },
        },
        "output_schema": {
            "type": "object",
            "properties": {
                "status": {"type": "string"},
                "queue_state": {"type": "string"},
                "work_kind": {"type": "string"},
            },
        },
        "handler_id": "background_artifact.enqueue.v1",
        "lifecycle_hooks": ["validate", "enqueue_background_artifact"],
        "permission_policy": "policy:background_artifact.enqueue.v1",
        "rate_limit_policy": "policy:action.default_rate_limit.v1",
        "audit_policy": "policy:action.audit.v1",
        "prompt_projection_policy": "policy:prompt.action_safe.v1",
    }
    return return_value


def _background_work_capability() -> CapabilitySpecV1:
    """Build the generic background-work request capability."""

    return_value: CapabilitySpecV1 = {
        "schema_version": "capability_spec.v1",
        "capability_kind": BACKGROUND_WORK_REQUEST_CAPABILITY,
        "category": "action",
        "owner_module": "background_work",
        "input_schema": {
            "type": "object",
            "required": [
                "task_brief",
                "requested_delivery",
                "max_output_chars",
            ],
            "properties": {
                "task_brief": {"type": "string"},
                "requested_delivery": {
                    "type": "string",
                    "enum": ["send_result_when_done"],
                },
                "max_output_chars": {"type": "integer"},
            },
        },
        "output_schema": {
            "type": "object",
            "properties": {
                "status": {"type": "string"},
                "queue_state": {"type": "string"},
                "task_summary": {"type": "string"},
            },
        },
        "handler_id": "background_work.enqueue.v1",
        "lifecycle_hooks": ["validate", "enqueue_background_work"],
        "permission_policy": "policy:background_work.enqueue.v1",
        "rate_limit_policy": "policy:action.default_rate_limit.v1",
        "audit_policy": "policy:action.audit.v1",
        "prompt_projection_policy": "policy:prompt.action_safe.v1",
    }
    return return_value


def _background_artifact_projection() -> dict[str, object]:
    """Return prompt-safe background artifact affordance metadata."""

    return_value = {
        "capability": BACKGROUND_ARTIFACT_REQUEST_CAPABILITY,
        "available": True,
        "visibility": "private",
        "semantic_input_summary": [
            "Use only for accepted bounded text artifact work.",
            (
                "Allowed work_kind values are coding_snippet, text_rewrite, "
                "and summary."
            ),
            "Pair this private request with a visible speak acknowledgement.",
        ],
        "execution_boundary": (
            "durable artifact queue creates later source-bound cognition"
        ),
    }
    return return_value


def _background_work_projection() -> dict[str, object]:
    """Return prompt-safe accepted delayed-task affordance metadata."""

    return_value = {
        "capability": BACKGROUND_WORK_REQUEST_CAPABILITY,
        "available": True,
        "visibility": "private",
        "semantic_input_summary": [
            (
                "Use when the character accepts bounded delayed text, code, "
                "or repository-analysis work."
            ),
            (
                "Repository or source-code analysis stays in this delayed "
                "work path even when it needs public source evidence."
            ),
            "Provide a task reason and detail, not execution internals.",
            "Pair this private request with a visible speak acknowledgement.",
        ],
        "execution_boundary": (
            "durable accepted-task lifecycle records the task before chat acknowledgement"
        ),
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


def _accepted_task_status_check_capability() -> CapabilitySpecV1:
    """Build the accepted-task status-check capability spec."""

    return_value: CapabilitySpecV1 = {
        "schema_version": "capability_spec.v1",
        "capability_kind": ACCEPTED_TASK_STATUS_CHECK_CAPABILITY,
        "category": "action",
        "owner_module": "accepted_task",
        "input_schema": {
            "type": "object",
            "required": [],
            "properties": {},
        },
        "output_schema": {
            "type": "object",
            "properties": {
                "status": {"type": "string"},
                "accepted_task_state": {"type": "string"},
                "accepted_task_summary": {"type": "string"},
            },
        },
        "handler_id": "accepted_task.status_check.v1",
        "lifecycle_hooks": ["validate", "status_lookup"],
        "permission_policy": "policy:accepted_task.status_check.v1",
        "rate_limit_policy": "policy:action.default_rate_limit.v1",
        "audit_policy": "policy:action.audit.v1",
        "prompt_projection_policy": "policy:prompt.action_safe.v1",
    }
    return return_value


def _future_speak_projection() -> dict[str, object]:
    """Return prompt-safe future-speak affordance metadata."""

    return_value = {
        "capability": FUTURE_SPEAK_CAPABILITY,
        "available": True,
        "visibility": "private",
        "semantic_input_summary": [
            (
                "Use when the character accepts a future reminder or delayed "
                "follow-up message."
            ),
            (
                "Put the exact configured-local trigger time in decision as "
                "YYYY-MM-DD HH:MM."
            ),
            (
                "Put only the semantic future-speaking objective in detail; "
                "do not write final message text."
            ),
            "Pair this private request with a visible speak acknowledgement.",
        ],
        "execution_boundary": (
            "durable delayed-task executor schedules a future cognition slot"
        ),
    }
    return return_value


def _accepted_task_status_check_projection() -> dict[str, object]:
    """Return prompt-safe accepted-task status affordance metadata."""

    return_value = {
        "capability": ACCEPTED_TASK_STATUS_CHECK_CAPABILITY,
        "available": True,
        "visibility": "private",
        "semantic_input_summary": [
            "Use when the user asks about already accepted delayed work.",
            "Do not include worker, queue, or job parameters.",
            "Pair this private check with a visible speak progress answer.",
        ],
        "execution_boundary": "durable accepted-task lifecycle status lookup",
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
        "semantic_input_summary": [
            "Use when active commitments need semantic lifecycle review.",
            "Provide review_kind=active_commitment_lifecycle and a short detail.",
        ],
        "execution_boundary": "memory lifecycle specialist chooses aliases and decisions",
    }
    return return_value
