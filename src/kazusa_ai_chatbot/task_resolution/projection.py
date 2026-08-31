"""Deterministic projections at the DSH/task-resolution boundary.

The sidecar owns semantic resolution.  This module owns the narrow, typed
projection into the task-resolution result consumed by persistence and the
brain, plus construction of the model-hidden fact list sent at start.
"""

from __future__ import annotations

from collections.abc import Mapping

from agentic_resolver.contracts import DSHResolutionExhaustV2
from kazusa_ai_chatbot.dsh_tool_gateway.contracts import canonical_json
from kazusa_ai_chatbot.task_resolution.contracts import (
    MAX_TASK_RESOLUTION_LIST_ITEMS,
    MAX_TASK_RESOLUTION_TEXT_CHARS,
    TaskResolutionContractError,
    TaskResolutionResultV1,
    validate_dsh_resolution_ref,
    validate_dsh_task_start_spec,
    validate_task_resolution_result,
)

_TERMINAL_RESULT_PROJECTIONS = {
    "resolved": ("resolved", "complete"),
    "partial": ("partial", "partial"),
    "needs_user_input": ("needs_user_input", "pending"),
    "approval_required": ("approval_required", "pending"),
    "unavailable": ("unavailable", "missing"),
    "failed": ("failed", "blocked"),
}


def build_model_facts(context: Mapping[str, object]) -> list[str]:
    """Build the exactly ten ordered facts accepted by the DSH start contract."""

    def encoded(value: object) -> str:
        return canonical_json(value).decode("utf-8")

    return [
        "character_and_scene=" + encoded({
            "character_name": context.get("character_name", ""),
            "scene_context": context.get("scene_context", {}),
        }),
        "local_time=" + encoded(context.get("local_time_context", {})),
        "current_message_context=" + encoded(
            context.get("prompt_message_context", {}),
        ),
        "recent_conversation=" + encoded(
            context.get("chat_history_recent", []),
        ),
        "wide_conversation=" + encoded(context.get("chat_history_wide", [])),
        "conversation_progress=" + encoded(
            context.get("conversation_progress", {}),
        ),
        "persona_summary=" + encoded(context.get("persona_summary", "")),
        "conversation_summary=" + encoded(
            context.get("conversation_summary", ""),
        ),
        "active_turn_lineage=" + encoded({
            "conversation_row_ids": context.get(
                "active_turn_conversation_row_ids", [],
            ),
            "platform_message_ids": context.get(
                "active_turn_platform_message_ids", [],
            ),
        }),
        "attached_media_refs=" + encoded(context.get("session_media_refs", [])),
    ]


def project_dsh_exhaust(
    exhaust: object,
    start_spec: Mapping[str, object],
) -> TaskResolutionResultV1:
    """Project a typed DSH exhaust without semantic reclassification."""

    try:
        validated_spec = validate_dsh_task_start_spec(start_spec)
    except (TypeError, ValueError) as exc:
        raise TaskResolutionContractError(
            f"DSH projection start specification is invalid: {exc}",
        ) from exc
    request_mapping = validated_spec["resolver_request"]
    context_mapping = validated_spec["execution_context"]
    if isinstance(exhaust, DSHResolutionExhaustV2):
        try:
            validated_exhaust = DSHResolutionExhaustV2.from_mapping(
                exhaust.to_dict(),
            )
        except (TypeError, ValueError) as exc:
            raise TaskResolutionContractError(
                f"DSH exhaust is invalid: {exc}",
            ) from exc
    elif isinstance(exhaust, Mapping):
        try:
            validated_exhaust = DSHResolutionExhaustV2.from_mapping(exhaust)
        except (TypeError, ValueError) as exc:
            raise TaskResolutionContractError(
                f"DSH exhaust is invalid: {exc}",
            ) from exc
    else:
        raise TaskResolutionContractError("DSH exhaust must be a typed object")
    return _project_exhaust_result(
        validated_exhaust,
        request_mapping["semantic_goal"],
        context_mapping,
    )


def _project_exhaust_result(
    exhaust: object,
    semantic_objective: str,
    context: Mapping[str, object],
) -> TaskResolutionResultV1:
    """Project one terminal/checkpoint/fault/cancel exhaust."""

    if not isinstance(exhaust, DSHResolutionExhaustV2):
        raise TaskResolutionContractError("DSH exhaust must be validated")
    kind = exhaust.kind
    payload = exhaust.to_dict()

    terminal = payload.get("terminal")
    if kind == "terminal":
        if not isinstance(terminal, Mapping):
            raise TaskResolutionContractError(
                "terminal DSH exhaust is missing its terminal result",
            )
        try:
            status, evidence_state = _TERMINAL_RESULT_PROJECTIONS[
                str(terminal["status"])
            ]
        except KeyError as exc:
            raise TaskResolutionContractError(
                "DSH terminal status is unsupported",
            ) from exc
    else:
        status, evidence_state = {
            "checkpointed": ("deferred", "pending"),
            "runtime_fault": ("unavailable", "missing"),
            "canceled": ("failed", "blocked"),
        }[kind]
    checkpoint: dict[str, object] = {}
    if isinstance(terminal, Mapping):
        terminal_status = terminal.get("status")
        try:
            status, evidence_state = _TERMINAL_RESULT_PROJECTIONS[
                str(terminal_status)
            ]
        except KeyError as exc:
            raise TaskResolutionContractError(
                "DSH terminal status is unsupported",
            ) from exc
        summary = _bounded_text(terminal.get("summary", ""))
        excerpts = _bounded_findings(terminal.get("findings"))
        completed = [
            _bounded_text(item)
            for item in terminal.get("completed_subgoals", [])
            if isinstance(item, str)
        ][:MAX_TASK_RESOLUTION_LIST_ITEMS]
        remaining = [
            _bounded_text(item)
            for item in terminal.get("remaining_needs", [])
            if isinstance(item, str)
        ][:MAX_TASK_RESOLUTION_LIST_ITEMS]
        warnings = [
            _bounded_text(item)
            for item in terminal.get("warnings", [])
            if isinstance(item, str)
        ][:MAX_TASK_RESOLUTION_LIST_ITEMS]
        if warnings:
            summary = _summary_with_warnings(summary, warnings)
        artifact_refs = terminal.get("artifact_refs")
        artifact_handles = [
            item for item in artifact_refs if isinstance(item, str)
        ] if isinstance(artifact_refs, list) else []
        handles: list[str] = []
        evidence_rows: list[dict[str, object]] = []
        evidence_value = payload.get("evidence")
        if isinstance(evidence_value, (list, tuple)):
            for row in evidence_value:
                if not isinstance(row, Mapping):
                    continue
                semantic_ref = row.get("semantic_ref")
                if not isinstance(semantic_ref, str):
                    continue
                evidence_rows.append({
                    "schema_version": "task_resolution_evidence.v1",
                    "evidence_id": str(row["evidence_id"]),
                    "task_node_id": "dsh",
                    "specialist": "dsh",
                    "summary": semantic_ref,
                    "provenance_refs": [
                        item for item in (
                            row["evidence_id"],
                            row["content_digest"],
                        )
                    ],
                    "limitations": [],
                })
                if semantic_ref not in handles:
                    handles.append(semantic_ref)
        for artifact_ref in artifact_handles:
            if artifact_ref not in handles:
                handles.append(artifact_ref)
            if len(handles) >= MAX_TASK_RESOLUTION_LIST_ITEMS:
                break
        evidence_rows = evidence_rows[:MAX_TASK_RESOLUTION_LIST_ITEMS]
        if not evidence_rows and artifact_handles and excerpts:
            evidence_rows = [
                {
                    "schema_version": "task_resolution_evidence.v1",
                    "evidence_id": artifact_ref,
                    "task_node_id": "dsh",
                    "specialist": "dsh",
                    "summary": artifact_ref,
                    "provenance_refs": [artifact_ref],
                    "limitations": [],
                }
                for artifact_ref in artifact_handles[:
                    MAX_TASK_RESOLUTION_LIST_ITEMS
                ]
            ]
        if evidence_rows and not excerpts:
            excerpts = [
                str(row["summary"])
                for row in evidence_rows
            ][:MAX_TASK_RESOLUTION_LIST_ITEMS]
        prompt_safe_summary = summary
    elif kind == "checkpointed":
        raw_checkpoint = payload.get("checkpoint")
        if raw_checkpoint is None or raw_checkpoint == {}:
            raw_identity = payload.get("identity")
            if isinstance(raw_identity, Mapping):
                raw_checkpoint = {
                    "schema_version": "dsh_resolution_ref.v1",
                    **dict(raw_identity),
                }
            elif raw_checkpoint is None:
                raw_checkpoint = payload
        checkpoint = _validated_checkpoint(raw_checkpoint)
        evidence_rows = []
        handles = []
        excerpts = []
        completed = []
        remaining = ["DSH resolution continuation"]
        prompt_safe_summary = "The DSH task needs durable continuation."
    else:
        checkpoint = {}
        evidence_rows = []
        handles = []
        excerpts = []
        completed = []
        remaining = ["DSH resolution retry"]
        fault = payload.get("fault")
        prompt_safe_summary = (
            _bounded_text(fault.get("code", "DSH runtime unavailable"))
            if isinstance(fault, Mapping)
            else "The DSH runtime was unavailable."
        )

    result: dict[str, object] = {
        "schema_version": "task_resolution_result.v1",
        "semantic_objective": _bounded_text(semantic_objective),
        "status": status,
        "scene_context": dict(context.get("scene_context", {})),
        "goal_continuation_ref": context.get("goal_continuation_ref", {}),
        "evidence_state": evidence_state,
        "evidence_excerpts": excerpts[:MAX_TASK_RESOLUTION_LIST_ITEMS],
        "evidence_handles": handles[:MAX_TASK_RESOLUTION_LIST_ITEMS],
        "prompt_safe_summary": prompt_safe_summary,
        "evidence": evidence_rows,
        "completed_subgoals": completed,
        "remaining_needs": remaining,
        "checkpoint": checkpoint,
        "coding_run_context": {},
    }
    return validate_task_resolution_result(result)


def _bounded_text(value: object) -> str:
    """Keep one DSH text value inside the outward result bound."""

    return str(value)[:MAX_TASK_RESOLUTION_TEXT_CHARS]


def _bounded_findings(value: object) -> list[str]:
    """Serialize at most the bounded canonical finding values."""

    if not isinstance(value, list):
        return []
    excerpts: list[str] = []
    for item in value[:MAX_TASK_RESOLUTION_LIST_ITEMS]:
        serialized = canonical_json(item).decode("utf-8")
        if len(serialized) <= MAX_TASK_RESOLUTION_TEXT_CHARS:
            excerpts.append(serialized)
    return excerpts


def _summary_with_warnings(summary: str, warnings: list[str]) -> str:
    """Preserve DSH warnings in the prompt-safe summary without new fields."""

    warning_text = "Warnings: " + "; ".join(warnings)
    if not summary:
        return warning_text[:MAX_TASK_RESOLUTION_TEXT_CHARS]
    return (summary + "\n" + warning_text)[:MAX_TASK_RESOLUTION_TEXT_CHARS]


def _validated_checkpoint(value: object) -> dict[str, object]:
    """Project only the exact durable DSH reference from a checkpoint."""

    if not isinstance(value, Mapping):
        raise TaskResolutionContractError(
            "DSH checkpoint does not contain a resolution reference",
        )
    try:
        return dict(validate_dsh_resolution_ref(value))
    except (TypeError, ValueError) as exc:
        raise TaskResolutionContractError(
            "DSH checkpoint does not contain a valid resolution reference",
        ) from exc
