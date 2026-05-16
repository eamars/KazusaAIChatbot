"""Action result, surface output, and episode trace helpers."""

from __future__ import annotations

from typing import Literal, TypeAlias, TypedDict

from kazusa_ai_chatbot.action_spec.models import (
    ActionContinuationV1,
    ActionSpecV1,
    EvidenceRefV1,
)

ACTION_RESULT_VERSION = "action_result.v1"
SURFACE_OUTPUT_VERSION = "surface_output.v1"
EPISODE_TRACE_VERSION = "episode_trace.v1"
CONSOLIDATION_ACTION_PROJECTION_VERSION = "consolidation_action_projection.v1"

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


class EpisodeTraceV1(TypedDict):
    """Consolidation-facing record of cognition, actions, and surfaces."""

    schema_version: Literal["episode_trace.v1"]
    episode_id: str
    trigger_source: str
    cognition_refs: list[EvidenceRefV1]
    action_specs: list[ActionSpecV1]
    action_results: list[ActionResultV1]
    surface_outputs: list[SurfaceOutputV1]
    created_at: str


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
    action_spec: ActionSpecV1,
    eval_result: dict[str, object],
    *,
    status: ActionResultStatus = "validated",
    result_summary: str = "",
    completed_at: str | None = None,
) -> ActionResultV1:
    """Build a result row for one validated, rejected, or pending action.

    Args:
        action_spec: Materialized action selected for the episode.
        eval_result: Deterministic evaluation result for ``action_spec``.
        status: Result status recorded for trace and consolidation.
        result_summary: Prompt-safe summary of what happened.
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
        "action_kind": action_spec["kind"],
        "handler_owner": handler_owner,
        "status": status,
        "visibility": action_spec["visibility"],
        "result_summary": summary,
        "result_refs": [],
        "continuation": action_spec["continuation"],
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


def build_episode_trace(
    *,
    episode_id: str,
    trigger_source: str,
    created_at: str,
    action_specs: list[ActionSpecV1],
    action_results: list[ActionResultV1],
    surface_outputs: list[SurfaceOutputV1],
    cognition_refs: list[EvidenceRefV1] | None = None,
) -> EpisodeTraceV1:
    """Build the episode trace consumed by post-turn consolidation."""

    trace: EpisodeTraceV1 = {
        "schema_version": EPISODE_TRACE_VERSION,
        "episode_id": episode_id,
        "trigger_source": trigger_source,
        "cognition_refs": cognition_refs or [],
        "action_specs": action_specs,
        "action_results": action_results,
        "surface_outputs": surface_outputs,
        "created_at": created_at,
    }
    return trace


def project_episode_trace_for_consolidation(
    trace: EpisodeTraceV1 | dict[str, object] | None,
) -> dict[str, object]:
    """Return prompt-safe action and surface evidence for consolidation."""

    if not isinstance(trace, dict):
        projection = _empty_projection()
        return projection

    raw_results = trace.get("action_results")
    action_results = raw_results if isinstance(raw_results, list) else []
    raw_outputs = trace.get("surface_outputs")
    surface_outputs = raw_outputs if isinstance(raw_outputs, list) else []

    projection = {
        "schema_version": "episode_trace_projection.v1",
        "episode_id": _string_field(trace, "episode_id"),
        "trigger_source": _string_field(trace, "trigger_source"),
        "action_results": [
            _project_action_result(row)
            for row in action_results
            if isinstance(row, dict)
        ],
        "surface_outputs": [
            _project_surface_output(row)
            for row in surface_outputs
            if isinstance(row, dict)
        ],
    }
    return projection


def has_consolidatable_output(state: dict[str, object]) -> bool:
    """Return whether a turn has text, surfaces, action results, or finalization."""

    final_dialog = state.get("final_dialog")
    if isinstance(final_dialog, list) and bool(final_dialog):
        return_value = True
        return return_value

    for field_name in ("surface_outputs", "action_results"):
        value = state.get(field_name)
        if isinstance(value, list) and bool(value):
            return_value = True
            return return_value

    dialog_usage_mode = state.get("dialog_usage_mode")
    return_value = dialog_usage_mode == "self_cognition_private_finalization"
    return return_value


def _default_result_summary(action_spec: ActionSpecV1, status: str) -> str:
    """Build a compact summary when a handler has no richer result text."""

    reason = action_spec["reason"].strip()
    if reason:
        return_value = f"{action_spec['kind']} {status}: {reason}"
        return return_value
    return_value = f"{action_spec['kind']} {status}"
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
