"""Action result, surface output, and episode trace helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal, NotRequired, TypeAlias, TypedDict

from kazusa_ai_chatbot.cognition_core_v2.contracts import RoleRefV2

from kazusa_ai_chatbot.action_spec.models import (
    ACTION_SPEC_VERSION,
    ActionContinuationV1,
    ActionSpecV1,
    EvidenceRefV1,
)

ACTION_RESULT_VERSION = "action_result.v1"
SURFACE_OUTPUT_VERSION = "surface_output.v1"
EPISODE_TRACE_V2_VERSION = "episode_trace.v2"
CONSOLIDATION_ACTION_PROJECTION_VERSION = "consolidation_action_projection.v1"
DEFAULT_ACTION_CONTINUATION: ActionContinuationV1 = {
    "schema_version": "action_continuation.v1",
    "mode": "none",
    "episode_type": None,
    "max_depth": 0,
    "include_result_as": None,
}

ActionResultStatus: TypeAlias = Literal[
    "rejected",
    "validated",
    "executed",
    "scheduled",
    "pending",
    "failed",
    "cancelled",
]


class ActionResultV1(TypedDict):
    """Auditable result produced after validating or executing one action."""

    schema_version: Literal["action_result.v1"]
    action_attempt_id: str
    action_kind: str
    handler_owner: str
    status: ActionResultStatus
    visibility: Literal["private", "preview", "user_visible"]
    result_summary: str
    result_refs: list[EvidenceRefV1]
    continuation: ActionContinuationV1
    completed_at: str | None
    queue_state: NotRequired[str]
    work_kind: NotRequired[str]
    objective_summary: NotRequired[str]
    operational_owner: NotRequired[str]
    job_ref: NotRequired[str]
    accepted_task_state: NotRequired[str]
    accepted_task_summary: NotRequired[str]
    wait_guidance: NotRequired[str]
    acknowledgement_constraint: NotRequired[str]
    semantic_result_v2: NotRequired["SemanticActionResultV2"]


class SemanticActionResultV2(TypedDict):
    """Typed execution outcome allowed back into later cognition."""

    action_kind: str
    status: Literal[
        "executed",
        "scheduled",
        "pending",
        "failed",
        "unavailable",
    ]
    semantic_result: str
    target_roles: list[RoleRefV2]


def project_semantic_action_result_v2(
    *,
    action_kind: str,
    status: str,
    semantic_result: str,
    target_roles: list[RoleRefV2],
) -> SemanticActionResultV2:
    """Project an action outcome without returning handlers or raw parameters."""

    if status not in {
        "executed",
        "scheduled",
        "pending",
        "failed",
        "unavailable",
    }:
        raise ValueError("V2 action result status is invalid")
    if not isinstance(semantic_result, str) or not semantic_result.strip():
        raise ValueError("V2 action result summary is invalid")
    return {
        "action_kind": action_kind,
        "status": status,
        "semantic_result": semantic_result,
        "target_roles": list(target_roles),
    }


def project_trace_action_result_v2(
    row: Mapping[str, object],
) -> SemanticActionResultV2:
    """Project one executed trace row into the exact V2 surface outcome."""

    trace_status = _string_field(row, "status")
    if trace_status in {"executed", "scheduled", "pending", "failed"}:
        status = trace_status
    else:
        status = "unavailable"
    semantic_result = _string_field(row, "result_summary").strip()
    if not semantic_result:
        semantic_result = f"{_string_field(row, 'action_kind')} {trace_status}"
    target_roles = row.get("target_roles")
    if not isinstance(target_roles, list):
        target_roles = []
    return project_semantic_action_result_v2(
        action_kind=_string_field(row, "action_kind") or "unknown",
        status=status,
        semantic_result=semantic_result,
        target_roles=[
            role for role in target_roles if isinstance(role, dict)
        ],
    )


class SurfaceOutputV1(TypedDict):
    """Surface artifact produced by a selected surface handler."""

    schema_version: Literal["surface_output.v1"]
    surface_kind: Literal["text", "image", "audio", "motor", "tool", "private"]
    visibility: Literal["private", "preview", "user_visible"]
    action_attempt_id: str | None
    fragments: list[str]
    artifact_refs: list[EvidenceRefV1]
    delivery_intent: Literal["deliver_now", "deliver_later", "do_not_deliver"]
    created_at: str


EpisodeTerminalStatusV1 = Literal[
    "completed_visible",
    "completed_private",
    "completed_action",
    "scheduled",
    "failed",
    "cancelled",
]


class EpisodeAttemptDiagnosticV1(TypedDict, total=False):
    """Typed stage-attempt outcome stored in a settled trace."""

    schema_version: Literal["episode_attempt_diagnostic.v1"]
    stage: str
    error_code: str
    attempt_count: int
    safe_checkpoint: str
    retryable: bool
    final_status: str


class DeliveryCorrelationV1(TypedDict, total=False):
    """Delivery intent and receipt correlation for one settled episode."""

    schema_version: Literal["delivery_correlation.v1"]
    delivery_intent: str
    tracking_id: str
    receipt_status: Literal[
        "not_applicable",
        "pending",
        "delivered",
        "failed",
        "unknown",
    ]
    receipt_ref: str


class EpisodeTraceV2(TypedDict):
    """Immutable settled audit trace for one cognitive episode."""

    schema_version: Literal["episode_trace.v2"]
    episode_id: str
    trigger_source: str
    terminal_status: EpisodeTerminalStatusV1
    cognition_refs: list[EvidenceRefV1]
    action_specs: list[ActionSpecV1]
    action_results: list[ActionResultV1]
    surface_outputs: list[SurfaceOutputV1]
    attempt_diagnostics: list[EpisodeAttemptDiagnosticV1]
    delivery_correlation: DeliveryCorrelationV1
    created_at: str
    settled_at: str


def validate_episode_trace_v2(value: object) -> EpisodeTraceV2:
    """Validate the exact immutable trace envelope and its outcome links."""

    if not isinstance(value, Mapping):
        raise ValueError("episode trace must be an object")
    required_fields = {
        "schema_version",
        "episode_id",
        "trigger_source",
        "terminal_status",
        "cognition_refs",
        "action_specs",
        "action_results",
        "surface_outputs",
        "attempt_diagnostics",
        "delivery_correlation",
        "created_at",
        "settled_at",
    }
    if set(value) != required_fields:
        raise ValueError("episode trace fields are not exact")
    if value["schema_version"] != EPISODE_TRACE_V2_VERSION:
        raise ValueError("unsupported episode trace schema version")
    for field_name in (
        "episode_id",
        "trigger_source",
        "created_at",
        "settled_at",
    ):
        field_value = value[field_name]
        if not isinstance(field_value, str) or not field_value.strip():
            raise ValueError(f"episode trace {field_name} is required")

    terminal_status = value["terminal_status"]
    if terminal_status not in {
        "completed_visible",
        "completed_private",
        "completed_action",
        "scheduled",
        "failed",
        "cancelled",
    }:
        raise ValueError("episode trace terminal status is invalid")
    for field_name in (
        "cognition_refs",
        "action_specs",
        "action_results",
        "surface_outputs",
        "attempt_diagnostics",
    ):
        if not isinstance(value[field_name], list):
            raise ValueError(f"episode trace {field_name} must be a list")

    action_attempt_ids: set[str] = set()
    for spec in value["action_specs"]:
        if not isinstance(spec, Mapping):
            raise ValueError("episode trace action spec must be an object")
        if spec.get("schema_version") != ACTION_SPEC_VERSION:
            raise ValueError("episode trace action spec version is invalid")
    for result in value["action_results"]:
        if not isinstance(result, Mapping):
            raise ValueError("episode trace action result must be an object")
        if result.get("schema_version") != ACTION_RESULT_VERSION:
            raise ValueError("episode trace action result version is invalid")
        action_attempt_id = result.get("action_attempt_id")
        if not isinstance(action_attempt_id, str) or not action_attempt_id:
            raise ValueError("episode trace action attempt id is required")
        if action_attempt_id in action_attempt_ids:
            raise ValueError("episode trace contains duplicate action attempts")
        action_attempt_ids.add(action_attempt_id)

    for output in value["surface_outputs"]:
        if not isinstance(output, Mapping):
            raise ValueError("episode trace surface output must be an object")
        if output.get("schema_version") != SURFACE_OUTPUT_VERSION:
            raise ValueError("episode trace surface output version is invalid")
    diagnostic_fields = {
        "schema_version",
        "stage",
        "error_code",
        "attempt_count",
        "safe_checkpoint",
        "retryable",
        "final_status",
    }
    for diagnostic in value["attempt_diagnostics"]:
        if not isinstance(diagnostic, Mapping):
            raise ValueError("episode trace diagnostic must be an object")
        if not diagnostic_fields.issubset(diagnostic):
            raise ValueError("episode trace diagnostic fields are incomplete")
        if diagnostic.get("schema_version") != "episode_attempt_diagnostic.v1":
            raise ValueError("episode trace diagnostic version is invalid")
        if not isinstance(diagnostic.get("attempt_count"), int):
            raise ValueError("episode trace diagnostic attempt count is invalid")
        if not isinstance(diagnostic.get("retryable"), bool):
            raise ValueError("episode trace diagnostic retryability is invalid")

    delivery = value["delivery_correlation"]
    if not isinstance(delivery, Mapping):
        raise ValueError("episode trace delivery correlation must be an object")
    delivery_fields = {
        "schema_version",
        "delivery_intent",
        "tracking_id",
        "receipt_status",
        "receipt_ref",
    }
    if set(delivery) != delivery_fields:
        raise ValueError("episode trace delivery fields are not exact")
    if delivery.get("schema_version") != "delivery_correlation.v1":
        raise ValueError("episode trace delivery version is invalid")
    delivery_intent = delivery.get("delivery_intent")
    if delivery_intent not in {"deliver_now", "deliver_later", "do_not_deliver"}:
        raise ValueError("episode trace delivery intent is invalid")
    receipt_status = delivery.get("receipt_status")
    if receipt_status not in {
        "not_applicable",
        "pending",
        "delivered",
        "failed",
        "unknown",
    }:
        raise ValueError("episode trace delivery receipt status is invalid")
    tracking_id = delivery.get("tracking_id")
    if not isinstance(tracking_id, str):
        raise ValueError("episode trace delivery tracking id is invalid")
    if receipt_status == "not_applicable" and tracking_id:
        raise ValueError("not_applicable delivery cannot have a tracking id")
    if receipt_status in {"pending", "delivered"} and not tracking_id:
        raise ValueError("delivery receipt requires a tracking id")
    if delivery_intent == "do_not_deliver" and receipt_status != "not_applicable":
        raise ValueError("do_not_deliver requires a not_applicable receipt")
    visible_surface = any(
        isinstance(output, Mapping)
        and output.get("visibility") == "user_visible"
        and output.get("delivery_intent") == "deliver_now"
        for output in value["surface_outputs"]
    )
    if terminal_status == "completed_visible" and not visible_surface:
        raise ValueError("completed_visible trace requires a visible surface")
    if terminal_status == "completed_private" and visible_surface:
        raise ValueError("completed_private trace cannot contain a visible surface")
    return dict(value)  # type: ignore[return-value]


class ConsolidationActionProjectionV1(TypedDict):
    """Prompt-safe action evidence consumed by consolidation."""

    schema_version: Literal["consolidation_action_projection.v1"]
    action_kind: str
    status: str
    visibility: Literal["private", "preview", "user_visible"]
    semantic_decision: str
    result_summary: str
    evidence_refs: list[EvidenceRefV1]


def action_attempt_id_from_eval_result(eval_result: dict[str, object]) -> str:
    """Return the action-attempt identifier implied by validation output."""

    idempotency_key = eval_result.get("idempotency_key")
    if not isinstance(idempotency_key, str) or not idempotency_key:
        return_value = ""
        return return_value
    suffix = idempotency_key.removeprefix("action_spec:v1:")
    return_value = f"action_attempt:{suffix}"
    return return_value


def build_action_result(
    action_spec: dict[str, object],
    eval_result: dict[str, object],
    *,
    status: ActionResultStatus = "validated",
    result_summary: str = "",
    result_refs: list[EvidenceRefV1] | None = None,
    completed_at: str | None = None,
) -> ActionResultV1:
    """Build a result row for one validated, rejected, or pending action.

    Args:
        action_spec: Materialized action selected for the episode.
        eval_result: Deterministic evaluation result for ``action_spec``.
        status: Result status recorded for trace and consolidation.
        result_summary: Prompt-safe summary of what happened.
        result_refs: Optional prompt-safe evidence references for handler
            results.
        completed_at: Optional completion timestamp.

    Returns:
        An ``ActionResultV1`` row without handler IDs or raw params.
    """

    handler_owner = eval_result.get("handler_owner")
    if not isinstance(handler_owner, str):
        handler_owner = ""
    summary = result_summary.strip()
    if not summary:
        summary = _default_result_summary(action_spec, status)
    result: ActionResultV1 = {
        "schema_version": ACTION_RESULT_VERSION,
        "action_attempt_id": action_attempt_id_from_eval_result(eval_result),
        "action_kind": _action_kind(action_spec),
        "handler_owner": handler_owner,
        "status": status,
        "visibility": _action_visibility(action_spec),
        "result_summary": summary,
        "result_refs": list(result_refs or []),
        "continuation": _action_continuation(action_spec),
        "completed_at": completed_at,
    }
    return result


def build_text_surface_output(
    *,
    fragments: list[str],
    created_at: str,
    action_attempt_id: str | None = None,
    visibility: Literal["private", "preview", "user_visible"] = "user_visible",
    delivery_intent: Literal[
        "deliver_now",
        "deliver_later",
        "do_not_deliver",
    ] = "deliver_now",
) -> SurfaceOutputV1:
    """Build a text surface artifact from final dialog fragments."""

    output: SurfaceOutputV1 = {
        "schema_version": SURFACE_OUTPUT_VERSION,
        "surface_kind": "text",
        "visibility": visibility,
        "action_attempt_id": action_attempt_id,
        "fragments": [fragment for fragment in fragments if fragment],
        "artifact_refs": [],
        "delivery_intent": delivery_intent,
        "created_at": created_at,
    }
    return output


def build_visual_surface_output(
    *,
    fragments: list[str],
    created_at: str,
) -> SurfaceOutputV1:
    """Build audit-only terminal image directives for the raw episode trace."""

    output: SurfaceOutputV1 = {
        "schema_version": SURFACE_OUTPUT_VERSION,
        "surface_kind": "image",
        "visibility": "private",
        "action_attempt_id": None,
        "fragments": list(fragments),
        "artifact_refs": [],
        "delivery_intent": "do_not_deliver",
        "created_at": created_at,
    }
    return output


def build_private_surface_output(
    *,
    summary: str,
    created_at: str,
    action_attempt_id: str | None = None,
) -> SurfaceOutputV1:
    """Build a private surface artifact for no-visible-output episodes."""

    output = build_text_surface_output(
        fragments=[summary],
        created_at=created_at,
        action_attempt_id=action_attempt_id,
        visibility="private",
        delivery_intent="do_not_deliver",
    )
    output["surface_kind"] = "private"
    return output


def project_episode_trace_for_consolidation(
    trace: EpisodeTraceV2 | dict[str, object] | None,
) -> dict[str, object]:
    """Project model-facing evidence without terminal visual directives."""

    if not isinstance(trace, dict) or trace.get("schema_version") != "episode_trace.v2":
        projection = _empty_projection()
        return projection

    raw_results = trace.get("action_results")
    action_results = raw_results if isinstance(raw_results, list) else []
    raw_outputs = trace.get("surface_outputs")
    surface_outputs = raw_outputs if isinstance(raw_outputs, list) else []
    projected_surface_outputs = []
    for row in surface_outputs:
        if not isinstance(row, dict):
            continue
        surface_kind = row.get("surface_kind")
        visibility = row.get("visibility")
        delivery_intent = row.get("delivery_intent")
        if (
            surface_kind == "image"
            and visibility == "private"
            and delivery_intent == "do_not_deliver"
        ):
            continue
        projected_surface_outputs.append(_project_surface_output(row))

    projection = {
        "schema_version": "episode_trace_projection.v1",
        "episode_id": _string_field(trace, "episode_id"),
        "trigger_source": _string_field(trace, "trigger_source"),
        "action_results": [
            _project_action_result(row)
            for row in action_results
            if isinstance(row, dict)
        ],
        "surface_outputs": projected_surface_outputs,
    }
    return projection


def has_consolidatable_output(trace: dict[str, object]) -> bool:
    """Return whether one settled trace contains consolidation evidence."""

    if trace.get("schema_version") != "episode_trace.v2":
        return False
    for field_name in ("surface_outputs", "action_results"):
        value = trace.get(field_name)
        if isinstance(value, list) and value:
            return True
    return trace.get("terminal_status") in {
        "completed_visible",
        "completed_private",
        "completed_action",
        "scheduled",
    }


def _default_result_summary(action_spec: dict[str, object], status: str) -> str:
    """Build a compact summary when a handler has no richer result text."""

    reason = _string_field(action_spec, "reason").strip()
    if reason:
        return_value = f"{_action_kind(action_spec)} {status}: {reason}"
        return return_value
    return_value = f"{_action_kind(action_spec)} {status}"
    return return_value


def _action_kind(action_spec: dict[str, object]) -> str:
    """Read an action kind from a valid or rejected action spec."""

    kind = _string_field(action_spec, "kind")
    if not kind:
        kind = "unknown"
    return kind


def _action_visibility(
    action_spec: dict[str, object],
) -> Literal["private", "preview", "user_visible"]:
    """Read action visibility with a private default for invalid specs."""

    value = action_spec.get("visibility")
    if value in ("private", "preview", "user_visible"):
        return_value = value
        return return_value
    return_value = "private"
    return return_value


def _action_continuation(action_spec: dict[str, object]) -> ActionContinuationV1:
    """Read continuation metadata with a no-continuation default."""

    value = action_spec.get("continuation")
    if isinstance(value, dict):
        mode = value.get("mode")
        max_depth = value.get("max_depth")
        include_result_as = value.get("include_result_as")
        if (
            value.get("schema_version") == "action_continuation.v1"
            and mode in (
                "none",
                "immediate_followup",
                "scheduled_followup",
                "background_followup",
            )
            and isinstance(max_depth, int)
            and (
                include_result_as in (None, "scheduled_event", "action_result")
            )
        ):
            continuation: ActionContinuationV1 = {
                "schema_version": "action_continuation.v1",
                "mode": mode,
                "episode_type": value.get("episode_type"),
                "max_depth": max_depth,
                "include_result_as": include_result_as,
            }
            return continuation
    return_value = dict(DEFAULT_ACTION_CONTINUATION)
    return return_value


def _empty_projection() -> dict[str, object]:
    """Return an empty trace projection with the current projection version."""

    projection = {
        "schema_version": "episode_trace_projection.v1",
        "episode_id": "",
        "trigger_source": "",
        "action_results": [],
        "surface_outputs": [],
    }
    return projection


def _project_action_result(row: dict[str, object]) -> ConsolidationActionProjectionV1:
    """Project one action result without handler ids or raw params."""

    projection: ConsolidationActionProjectionV1 = {
        "schema_version": CONSOLIDATION_ACTION_PROJECTION_VERSION,
        "action_kind": _string_field(row, "action_kind"),
        "status": _string_field(row, "status"),
        "visibility": _visibility_field(row),
        "semantic_decision": _string_field(row, "result_summary"),
        "result_summary": _string_field(row, "result_summary"),
        "evidence_refs": [],
    }
    raw_refs = row.get("result_refs")
    if isinstance(raw_refs, list):
        projection["evidence_refs"] = [
            ref for ref in raw_refs if isinstance(ref, dict)
        ]
    return projection


def _project_surface_output(row: dict[str, object]) -> dict[str, object]:
    """Project one surface output without raw delivery or adapter fields."""

    raw_fragments = row.get("fragments")
    fragments = raw_fragments if isinstance(raw_fragments, list) else []
    projection = {
        "surface_kind": _string_field(row, "surface_kind"),
        "visibility": _visibility_field(row),
        "delivery_intent": _string_field(row, "delivery_intent"),
        "fragments": [
            fragment for fragment in fragments if isinstance(fragment, str)
        ],
    }
    return projection


def _visibility_field(
    row: dict[str, object],
) -> Literal["private", "preview", "user_visible"]:
    """Return a valid visibility value from a trace row."""

    value = row.get("visibility")
    if value in ("private", "preview", "user_visible"):
        return_value = value
        return return_value
    return_value = "private"
    return return_value


def _string_field(row: dict[str, object], field_name: str) -> str:
    """Read a string field from an external trace row."""

    value = row.get(field_name)
    if not isinstance(value, str):
        return_value = ""
        return return_value
    return_value = value
    return return_value
