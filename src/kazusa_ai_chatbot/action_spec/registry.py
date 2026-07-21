"""Action capability registry and prompt-safe affordance projection."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timedelta, timezone

from kazusa_ai_chatbot.action_spec.models import (
    ActionAvailabilityContextV1,
    AffordanceSpecV1,
    AvailabilityProbeResultV1,
    CapabilitySpecV1,
    RuntimeCapabilitySnapshotV1,
    validate_capability_spec,
)

MEMORY_LIFECYCLE_UPDATE_CAPABILITY = "memory_lifecycle_update"
APPLY_MEMORY_LIFECYCLE_UPDATE_CAPABILITY = "apply_memory_lifecycle_update"
SPEAK_CAPABILITY = "speak"
TRIGGER_FUTURE_COGNITION_CAPABILITY = "trigger_future_cognition"
FUTURE_SPEAK_CAPABILITY = "future_speak"
ACCEPTED_TASK_REQUEST_CAPABILITY = "accepted_task_request"
ACCEPTED_TASK_STATUS_CHECK_CAPABILITY = "accepted_task_status_check"
ACCEPTED_CODING_TASK_REQUEST_CAPABILITY = "accepted_coding_task_request"
BACKGROUND_WORK_REQUEST_CAPABILITY = "background_work_request"
_QUEUE_ONLY_CAPABILITIES = frozenset({
    TRIGGER_FUTURE_COGNITION_CAPABILITY,
    FUTURE_SPEAK_CAPABILITY,
    ACCEPTED_TASK_REQUEST_CAPABILITY,
    ACCEPTED_CODING_TASK_REQUEST_CAPABILITY,
    BACKGROUND_WORK_REQUEST_CAPABILITY,
})
_USER_MESSAGE_ONLY_CAPABILITIES = frozenset({
    FUTURE_SPEAK_CAPABILITY,
    ACCEPTED_TASK_REQUEST_CAPABILITY,
    ACCEPTED_CODING_TASK_REQUEST_CAPABILITY,
    BACKGROUND_WORK_REQUEST_CAPABILITY,
})
_AVAILABILITY_SNAPSHOT_TTL_SECONDS = 5


def build_runtime_capability_snapshot(
    *,
    route_health: Mapping[str, str] | None = None,
    repository_access: Mapping[str, str] | None = None,
    worker_status: Mapping[str, str] | None = None,
    scheduler_status: str = "healthy",
    adapter_target_status: Mapping[str, str] | None = None,
    coding_workspace_status: str = "healthy",
    permissions: Mapping[str, bool] | None = None,
) -> RuntimeCapabilitySnapshotV1:
    """Build one short-lived, side-effect-free runtime health snapshot."""

    checked_at_datetime = datetime.now(timezone.utc)
    expires_at_datetime = checked_at_datetime + timedelta(
        seconds=_AVAILABILITY_SNAPSHOT_TTL_SECONDS,
    )
    return {
        "checked_at": checked_at_datetime.isoformat(),
        "expires_at": expires_at_datetime.isoformat(),
        "route_health": dict(route_health or {}),
        "repository_access": dict(repository_access or {}),
        "worker_status": dict(worker_status or {}),
        "scheduler_status": scheduler_status,
        "adapter_target_status": dict(adapter_target_status or {}),
        "coding_workspace_status": coding_workspace_status,
        "permissions": dict(permissions or {}),
    }


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
        ACCEPTED_TASK_REQUEST_CAPABILITY: _accepted_task_request_capability(),
        ACCEPTED_CODING_TASK_REQUEST_CAPABILITY: (
            _accepted_coding_task_capability()
        ),
        ACCEPTED_TASK_STATUS_CHECK_CAPABILITY: (
            _accepted_task_status_check_capability()
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
        elif capability_kind == ACCEPTED_TASK_REQUEST_CAPABILITY:
            projection.append(_accepted_task_request_projection())
        elif capability_kind == ACCEPTED_CODING_TASK_REQUEST_CAPABILITY:
            projection.append(_accepted_coding_task_projection())
        elif capability_kind == ACCEPTED_TASK_STATUS_CHECK_CAPABILITY:
            projection.append(_accepted_task_status_check_projection())
        elif capability_kind == BACKGROUND_WORK_REQUEST_CAPABILITY:
            projection.append(_background_work_projection())
    return projection


def is_capability_allowed_for_source(
    capability_kind: str,
    source_kind: str,
) -> bool:
    """Return whether a capability can be created from this source event."""

    if capability_kind not in _USER_MESSAGE_ONLY_CAPABILITIES:
        return True
    return source_kind == "user_message"


def _probe_capability_availability(
    capability_kind: str,
    context: ActionAvailabilityContextV1,
    snapshot: RuntimeCapabilitySnapshotV1,
) -> AvailabilityProbeResultV1:
    """Project deterministic health facts into one capability result."""

    checked_at = snapshot["checked_at"]
    expires_at = snapshot["expires_at"]
    permissions = snapshot["permissions"]
    permission_ref = str(
        context.get("permission_ref") or capability_kind
    )
    if permissions.get(permission_ref) is False:
        return {
            "status": "unavailable",
            "reason_code": "permission_denied",
            "checked_at": checked_at,
            "expires_at": expires_at,
        }

    requested_work_kind = context.get("requested_work_kind")
    if (
        capability_kind in {
            ACCEPTED_TASK_REQUEST_CAPABILITY,
            ACCEPTED_CODING_TASK_REQUEST_CAPABILITY,
            BACKGROUND_WORK_REQUEST_CAPABILITY,
        }
        and requested_work_kind == "unsupported"
    ):
        return {
            "status": "unavailable",
            "reason_code": "unsupported_work_kind",
            "checked_at": checked_at,
            "expires_at": expires_at,
        }

    capability = build_initial_action_capabilities()[capability_kind]
    owner = capability["owner_module"]
    route_status = snapshot["route_health"].get(owner, "healthy")
    if route_status in {"down", "unavailable"}:
        return {
            "status": "unavailable",
            "reason_code": "route_unavailable",
            "checked_at": checked_at,
            "expires_at": expires_at,
        }
    if route_status == "degraded":
        return {
            "status": "unavailable",
            "reason_code": "route_unavailable",
            "checked_at": checked_at,
            "expires_at": expires_at,
        }

    repository_status = snapshot["repository_access"].get(owner, "read_write")
    if repository_status == "down" or (
        repository_status == "read_only"
        and capability_kind != ACCEPTED_TASK_STATUS_CHECK_CAPABILITY
    ):
        return {
            "status": "unavailable",
            "reason_code": "repository_unavailable",
            "checked_at": checked_at,
            "expires_at": expires_at,
        }

    if capability_kind == SPEAK_CAPABILITY:
        target_scope = context.get("target_scope")
        target_keys = ["default"]
        if isinstance(target_scope, Mapping):
            platform = str(target_scope.get("platform") or "")
            channel_id = str(
                target_scope.get("platform_channel_id") or ""
            )
            channel_type = str(target_scope.get("channel_type") or "")
            target_keys = [
                f"{platform}:{channel_id}",
                f"{platform}:{channel_type}",
                platform,
                "default",
            ]
        target_status = "healthy"
        for target_key in target_keys:
            if target_key in snapshot["adapter_target_status"]:
                target_status = snapshot["adapter_target_status"][target_key]
                break
        if target_status in {"down", "unavailable"}:
            return {
                "status": "unavailable",
                "reason_code": "target_unavailable",
                "checked_at": checked_at,
                "expires_at": expires_at,
            }

    if capability_kind in {
        TRIGGER_FUTURE_COGNITION_CAPABILITY,
        FUTURE_SPEAK_CAPABILITY,
    } and snapshot["scheduler_status"] in {"down", "unavailable"}:
        return {
            "status": "unavailable",
            "reason_code": "worker_unavailable",
            "checked_at": checked_at,
            "expires_at": expires_at,
        }
    if (
        capability_kind in {
            TRIGGER_FUTURE_COGNITION_CAPABILITY,
            FUTURE_SPEAK_CAPABILITY,
        }
        and snapshot["scheduler_status"] == "degraded"
        and repository_status == "read_write"
    ):
        return {
            "status": "degraded",
            "reason_code": "queue_only",
            "checked_at": checked_at,
            "expires_at": expires_at,
        }

    if capability_kind == ACCEPTED_CODING_TASK_REQUEST_CAPABILITY:
        if snapshot["coding_workspace_status"] in {"down", "unavailable"}:
            return {
                "status": "unavailable",
                "reason_code": "workspace_unavailable",
                "checked_at": checked_at,
                "expires_at": expires_at,
            }

    worker_status = snapshot["worker_status"].get(owner, "healthy")
    if capability_kind == ACCEPTED_TASK_STATUS_CHECK_CAPABILITY:
        worker_status = "healthy"
    if worker_status in {"down", "unavailable", "degraded"}:
        if repository_status == "read_write" and capability_kind in _QUEUE_ONLY_CAPABILITIES:
            return {
                "status": "degraded",
                "reason_code": "queue_only",
                "checked_at": checked_at,
                "expires_at": expires_at,
            }
        return {
            "status": "unavailable",
            "reason_code": "worker_unavailable",
            "checked_at": checked_at,
            "expires_at": expires_at,
        }

    return {
        "status": "available",
        "reason_code": "ready",
        "checked_at": checked_at,
        "expires_at": expires_at,
    }


def build_episode_affordances(
    capabilities: Mapping[str, CapabilitySpecV1],
    context: ActionAvailabilityContextV1,
    snapshot: RuntimeCapabilitySnapshotV1,
) -> list[AffordanceSpecV1]:
    """Build prompt-safe affordances from one runtime snapshot."""

    affordances: list[AffordanceSpecV1] = []
    for capability_kind in sorted(capabilities):
        if capability_kind == APPLY_MEMORY_LIFECYCLE_UPDATE_CAPABILITY:
            continue
        source_kind = context.get("source_kind")
        if (
            isinstance(source_kind, str)
            and not is_capability_allowed_for_source(
                capability_kind,
                source_kind,
            )
        ):
            continue
        capability = capabilities[capability_kind]
        probe = _probe_capability_availability(
            capability_kind,
            context,
            snapshot,
        )
        if probe["status"] == "unavailable":
            continue
        projections = project_prompt_affordances({capability_kind: capability})
        projection = projections[0] if projections else {}
        semantic_summary = projection.get("semantic_input_summary", [])
        prompt_affordance = (
            semantic_summary[0]
            if isinstance(semantic_summary, list) and semantic_summary
            else capability_kind
        )
        properties = capability["input_schema"].get("properties", {})
        params_summary = {
            str(field_name): "parameter"
            for field_name in properties
            if isinstance(field_name, str)
        }
        visibility = projection.get("visibility", "private")
        if visibility not in {"private", "preview", "user_visible"}:
            visibility = "private"
        latency_tier = "live"
        if capability_kind in {
            FUTURE_SPEAK_CAPABILITY,
            ACCEPTED_TASK_REQUEST_CAPABILITY,
            ACCEPTED_CODING_TASK_REQUEST_CAPABILITY,
            BACKGROUND_WORK_REQUEST_CAPABILITY,
        }:
            latency_tier = "background"
        if capability_kind == TRIGGER_FUTURE_COGNITION_CAPABILITY:
            latency_tier = "scheduled"
        affordance: AffordanceSpecV1 = {
            "schema_version": "affordance_spec.v1",
            "capability_kind": capability_kind,
            "owner": capability["owner_module"],
            "surface": str(projection.get("execution_boundary", "")),
            "availability": probe["status"],
            "visibility": visibility,
            "latency_tier": latency_tier,
            "cost_tier": "low",
            "risk_tier": "medium",
            "allowed_cognition_modes": ["deliberative", "reflex"],
            "allowed_continuation_modes": [
                "none",
                "immediate_followup",
                "scheduled_followup",
            ],
            "permission_policy": capability["permission_policy"],
            "params_summary": params_summary,
            "prompt_affordance": str(prompt_affordance),
        }
        affordances.append(affordance)
    return affordances


async def recheck_action_affordance(
    capability_kind: str,
    context: ActionAvailabilityContextV1,
    fresh_snapshot: RuntimeCapabilitySnapshotV1,
) -> AvailabilityProbeResultV1:
    """Recheck one action against the supplied fresh snapshot."""

    capabilities = build_initial_action_capabilities()
    if capability_kind not in capabilities:
        return {
            "status": "unavailable",
            "reason_code": "unsupported_work_kind",
            "checked_at": fresh_snapshot["checked_at"],
            "expires_at": fresh_snapshot["expires_at"],
        }
    return _probe_capability_availability(
        capability_kind,
        context,
        fresh_snapshot,
    )


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


def _accepted_coding_task_capability() -> CapabilitySpecV1:
    """Build the accepted coding-task durable-run request capability."""

    return_value: CapabilitySpecV1 = {
        "schema_version": "capability_spec.v1",
        "capability_kind": ACCEPTED_CODING_TASK_REQUEST_CAPABILITY,
        "category": "action",
        "owner_module": "background_work",
        "input_schema": {
            "type": "object",
            "required": [
                "task_brief",
                "coding_action",
                "requested_delivery",
                "max_output_chars",
            ],
            "properties": {
                "task_brief": {"type": "string"},
                "coding_action": {
                    "type": "string",
                    "enum": [
                        "start",
                        "revise_proposal",
                        "summarize",
                        "status",
                        "approve_and_verify",
                        "respond_to_blocker",
                        "cancel",
                    ],
                },
                "coding_run_ref": {"type": "string"},
                "execution_request": {"type": "string"},
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
                "coding_run_ref": {"type": "string"},
            },
        },
        "handler_id": "background_work.accepted_coding_task.enqueue.v1",
        "lifecycle_hooks": ["validate", "enqueue_background_work"],
        "permission_policy": "policy:background_work.accepted_coding_task.v1",
        "rate_limit_policy": "policy:action.default_rate_limit.v1",
        "audit_policy": "policy:action.audit.v1",
        "prompt_projection_policy": "policy:prompt.action_safe.v1",
    }
    return return_value


def _accepted_task_request_capability() -> CapabilitySpecV1:
    """Build the accepted-task durable request capability."""

    return_value: CapabilitySpecV1 = {
        "schema_version": "capability_spec.v1",
        "capability_kind": ACCEPTED_TASK_REQUEST_CAPABILITY,
        "category": "action",
        "owner_module": "accepted_task",
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
                "accepted_task_id": {"type": "string"},
            },
        },
        "handler_id": "accepted_task.request.v1",
        "lifecycle_hooks": ["validate", "enqueue_accepted_task"],
        "permission_policy": "policy:accepted_task.request.v1",
        "rate_limit_policy": "policy:action.default_rate_limit.v1",
        "audit_policy": "policy:action.audit.v1",
        "prompt_projection_policy": "policy:prompt.action_safe.v1",
    }
    return return_value


def _background_work_projection() -> dict[str, object]:
    """Return prompt-safe accepted delayed-task affordance metadata."""

    return_value = {
        "capability": BACKGROUND_WORK_REQUEST_CAPABILITY,
        "available": True,
        "availability_context": "",
        "visibility": "private",
        "decision_mode": "closed",
        "allowed_decisions": ["enqueue"],
        "default_decision": "enqueue",
        "decision_pattern": "",
        "context_ref": "",
        "semantic_input_summary": [
            (
                "Use only for explicitly accepted delayed work: bounded text, "
                "code, or repository analysis produced out of turn."
            ),
            (
                "Repository analysis stays here even with public evidence."
            ),
            (
                "Never use for current-turn reasoning, local context recall, "
                "reply preparation, rehearsal, or wording."
            ),
            (
                "Never execute a physical action or generate, store, or later "
                "present an action description."
            ),
            (
                "Provide a task reason and detail without execution internals; "
                "pair it with a visible acknowledgement."
            ),
        ],
        "execution_boundary": (
            "durable accepted-task lifecycle records the task before chat acknowledgement"
        ),
    }
    return return_value


def _accepted_coding_task_projection() -> dict[str, object]:
    """Return prompt-safe durable coding-run affordance metadata."""

    return_value = {
        "capability": ACCEPTED_CODING_TASK_REQUEST_CAPABILITY,
        "available": True,
        "availability_context": "",
        "visibility": "private",
        "decision_mode": "closed",
        "allowed_decisions": [
            "start",
            "revise_proposal",
            "summarize",
            "status",
            "approve_and_verify",
            "respond_to_blocker",
            "cancel",
        ],
        "default_decision": "start",
        "decision_pattern": "",
        "context_ref": "",
        "semantic_input_summary": [
            "Accepted coding work is managed as a durable run.",
            (
                "allowed_next_actions: start, revise_proposal, summarize, "
                "status, approve_and_verify, respond_to_blocker, cancel."
            ),
            (
                "start creates a run; other decisions continue the "
                "matching open run."
            ),
            (
                "Use the contextual affordance decision; ask visibly when no "
                "run is distinguishable."
            ),
            "Put verification or blocker answers in detail.",
            "Pair this private request with visible acknowledgement.",
        ],
        "execution_boundary": (
            "durable accepted-task lifecycle queues the coding-run worker"
        ),
    }
    return return_value


def _accepted_task_request_projection() -> dict[str, object]:
    """Return prompt-safe accepted-task request affordance metadata."""

    return_value = {
        "capability": ACCEPTED_TASK_REQUEST_CAPABILITY,
        "available": True,
        "availability_context": "",
        "visibility": "private",
        "decision_mode": "required_text",
        "allowed_decisions": ["enqueue"],
        "default_decision": "enqueue",
        "decision_pattern": "",
        "context_ref": "",
        "semantic_input_summary": [
            "Use for explicitly accepted bounded delayed work.",
            "Provide the accepted task objective without execution internals.",
            "Pair the private request with a visible acknowledgement.",
        ],
        "execution_boundary": "accepted-task lifecycle queues the durable worker",
    }
    return return_value


def _future_cognition_projection() -> dict[str, object]:
    """Return prompt-safe future-cognition affordance metadata."""

    return_value = {
        "capability": TRIGGER_FUTURE_COGNITION_CAPABILITY,
        "available": True,
        "availability_context": "private_cognition_source",
        "visibility": "private",
        "decision_mode": "closed",
        "allowed_decisions": ["schedule"],
        "default_decision": "schedule",
        "decision_pattern": "",
        "context_ref": "",
        "semantic_input_summary": [
            (
                "Use only when a concrete unresolved private task requires "
                "another cognition cycle after the current turn."
            ),
            (
                "Do not use for rehearsing the current reply, preserving "
                "persona or style, or adding thought that can finish now."
            ),
            "Put the unresolved objective and any timing hint in semantic_goal.",
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
        "availability_context": "",
        "visibility": "private",
        "decision_mode": "required_text",
        "allowed_decisions": [],
        "default_decision": "",
        "decision_pattern": r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}",
        "context_ref": "",
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
        "availability_context": "",
        "visibility": "private",
        "decision_mode": "closed",
        "allowed_decisions": ["check"],
        "default_decision": "check",
        "decision_pattern": "",
        "context_ref": "",
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
        "availability_context": "",
        "visibility": "user_visible",
        "decision_mode": "optional",
        "allowed_decisions": [],
        "default_decision": "visible_reply",
        "decision_pattern": "",
        "context_ref": "",
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
        "availability_context": "active_commitment",
        "visibility": "private",
        "decision_mode": "closed",
        "allowed_decisions": ["active_commitment_lifecycle"],
        "default_decision": "active_commitment_lifecycle",
        "decision_pattern": "",
        "context_ref": "",
        "semantic_input_summary": [
            "Use when active commitments need semantic lifecycle review.",
            "Use the fixed active_commitment_lifecycle decision and put the "
            "concrete review objective in semantic_goal.",
        ],
        "execution_boundary": "memory lifecycle specialist chooses aliases and decisions",
    }
    return return_value
