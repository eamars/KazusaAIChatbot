"""Deterministic checkpoint construction and update helpers."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any
from uuid import uuid4

from kazusa_ai_chatbot.cognition_episode import (
    CognitiveEpisodeValidationError,
    GoalContinuationRefV1,
    validate_goal_continuation_ref,
)
from kazusa_ai_chatbot.task_resolution.contracts import (
    MAX_TASK_RESOLUTION_DISPATCHES,
    MAX_TASK_RESOLUTION_NODES,
    MAX_TASK_RESOLUTION_ORCHESTRATOR_CALLS,
    MAX_TASK_RESOLUTION_ROUTE_CORRECTIONS,
    MAX_TASK_RESOLUTION_SPECIALIST_INVOCATIONS,
    TASK_CODING_OBJECTIVE_MODES,
    TASK_CODING_SPECIALIST_OBJECTIVE_MODES,
    TASK_RESOLUTION_CHECKPOINT_VERSION,
    TASK_RESOLUTION_NODE_VERSION,
    TASK_PENDING_DISPATCH_VERSION,
    TASK_RESOLUTION_RESULT_VERSION,
    TASK_SPECIALISTS,
    TaskResolutionCheckpointV1,
    TaskResolutionContractError,
    TaskResolutionExecutionContextV1,
    TaskResolutionResultV1,
    TaskSpecialistResultV1,
    validate_task_resolution_checkpoint,
    validate_task_resolution_execution_context,
    validate_task_resolution_result,
    validate_task_specialist_request,
    validate_task_specialist_result,
)


def create_task_resolution_checkpoint(
    request: Mapping[str, object],
    execution_context: TaskResolutionExecutionContextV1,
) -> TaskResolutionCheckpointV1:
    """Create the first bounded checkpoint from one authorized resolver goal.

    Args:
        request: Authorized V2 resolver request with the semantic objective.
        execution_context: Trusted prompt-safe caller context for the task.

    Returns:
        A validated checkpoint with one pending root node and no dispatches.
    """

    context = validate_task_resolution_execution_context(execution_context)
    capability = _request_text(request, "capability")
    if capability != "task_resolution_request":
        raise TaskResolutionContractError(
            "capability: expected task_resolution_request"
        )
    semantic_objective = _request_text(request, "semantic_goal")
    continuation_ref = _request_goal_continuation_ref(request)
    if continuation_ref != context["goal_continuation_ref"]:
        raise TaskResolutionContractError(
            "goal_continuation_ref: request conflicts with execution context"
        )
    checkpoint: TaskResolutionCheckpointV1 = {
        "schema_version": TASK_RESOLUTION_CHECKPOINT_VERSION,
        "session_id": f"task_resolution:{uuid4().hex}",
        "semantic_objective": semantic_objective,
        "scene_context": context["scene_context"],
        "goal_continuation_ref": continuation_ref,
        "source_scope": {
            "trigger_source": "user_message",
            "platform": context["platform"],
            "channel_id": context["channel_id"],
            "channel_type": context["channel_type"],
            "source_message_id": context["source_message_id"],
            "requester_global_user_id": context["requester_global_user_id"],
            "requester_platform_user_id": context[
                "requester_platform_user_id"
            ],
        },
        "nodes": [{
            "schema_version": TASK_RESOLUTION_NODE_VERSION,
            "node_id": "node-1",
            "objective": semantic_objective,
            "status": "pending",
            "depends_on": [],
        }],
        "active_node_id": "node-1",
        "evidence": [],
        "remaining_needs": [semantic_objective],
        "attempted_specialists": [],
        "dispatch_count": 0,
        "orchestrator_call_count": 0,
        "route_correction_count": 0,
        "specialist_invocation_counts": [],
        "pending_dispatch": None,
        "terminal_status": "",
        "trace_summary": [],
    }
    validated = validate_task_resolution_checkpoint(checkpoint)
    return validated


def build_specialist_request(
    checkpoint: TaskResolutionCheckpointV1,
    *,
    semantic_subgoal: str | None = None,
    coding_objective_mode: str | None = None,
) -> dict[str, object]:
    """Project one selected semantic subgoal into a handler request.

    Args:
        checkpoint: Validated session state with the active task node.
        semantic_subgoal: The orchestrator-selected semantic objective for this
            dispatch. Omission reads a persisted dispatch subgoal when present
            and otherwise retains the active node objective.
        coding_objective_mode: Validated coding mode from a persisted dispatch.
            Omission uses that dispatch mode or ``none`` for a direct
            checkpoint projection.

    Returns:
        One canonical specialist request whose objective is the selected
        semantic subgoal rather than a worker, tool, or routing instruction.
    """

    validated = normalize_started_dispatch_ledger(checkpoint)
    active_node = _active_node(validated)
    objective = active_node["objective"]
    mode = "none"
    pending_dispatch = validated["pending_dispatch"]
    if pending_dispatch is not None:
        if pending_dispatch["task_node_id"] != active_node["node_id"]:
            raise TaskResolutionContractError(
                "pending_dispatch.task_node_id: expected active task node"
            )
        objective = pending_dispatch["subgoal"]
        mode = pending_dispatch["coding_objective_mode"]
    if semantic_subgoal is not None:
        requested_subgoal = _require_semantic_subgoal(semantic_subgoal)
        if pending_dispatch is not None and requested_subgoal != objective:
            raise TaskResolutionContractError(
                "semantic_subgoal: does not match pending dispatch"
            )
        objective = requested_subgoal
    if coding_objective_mode is not None:
        requested_mode = _require_coding_objective_mode(coding_objective_mode)
        if pending_dispatch is not None and requested_mode != mode:
            raise TaskResolutionContractError(
                "coding_objective_mode: does not match pending dispatch"
            )
        mode = requested_mode
    request = {
        "schema_version": "task_specialist_request.v1",
        "task_node_id": active_node["node_id"],
        "objective": objective,
        "available_evidence": list(validated["evidence"]),
        "remaining_needs": list(validated["remaining_needs"]),
        "trusted_scope": dict(validated["source_scope"]),
        "coding_objective_mode": mode,
    }
    return validate_task_specialist_request(request)


def record_orchestrator_call(
    checkpoint: TaskResolutionCheckpointV1,
) -> TaskResolutionCheckpointV1:
    """Persist one raw task-orchestrator LLM call before parsing its output."""

    validated = normalize_started_dispatch_ledger(checkpoint)
    if validated["terminal_status"]:
        raise TaskResolutionContractError(
            "checkpoint: cannot call the orchestrator after terminal status"
        )
    if validated["pending_dispatch"] is not None:
        raise TaskResolutionContractError(
            "pending_dispatch: cannot select while a dispatch is pending"
        )
    if (
        validated["orchestrator_call_count"]
        >= MAX_TASK_RESOLUTION_ORCHESTRATOR_CALLS
    ):
        raise TaskResolutionContractError(
            "orchestrator_call_count: task budget exhausted"
        )
    updated = deepcopy(validated)
    updated["orchestrator_call_count"] += 1
    return validate_task_resolution_checkpoint(updated)


def select_pending_dispatch(
    checkpoint: TaskResolutionCheckpointV1,
    *,
    specialist: str,
    subgoal: str,
    coding_objective_mode: str,
) -> TaskResolutionCheckpointV1:
    """Store one validated specialist selection before durable execution."""

    validated = normalize_started_dispatch_ledger(checkpoint)
    if validated["terminal_status"]:
        raise TaskResolutionContractError(
            "checkpoint: cannot select after terminal status"
        )
    if validated["pending_dispatch"] is not None:
        raise TaskResolutionContractError(
            "pending_dispatch: existing dispatch must finish first"
        )
    if validated["dispatch_count"] >= MAX_TASK_RESOLUTION_DISPATCHES:
        raise TaskResolutionContractError("dispatch_count: task budget exhausted")
    active_node = _active_node(validated)
    normalized_specialist = _require_specialist(specialist)
    normalized_mode = _require_coding_objective_mode(coding_objective_mode)
    _validate_dispatch_mode(
        specialist=normalized_specialist,
        coding_objective_mode=normalized_mode,
    )
    if has_attempted_specialist(
        validated,
        task_node_id=active_node["node_id"],
        specialist=normalized_specialist,
    ):
        raise TaskResolutionContractError(
            "attempted_specialists: incompatible pair already attempted"
        )
    if specialist_invocation_count(
        validated,
        task_node_id=active_node["node_id"],
        specialist=normalized_specialist,
    ) >= MAX_TASK_RESOLUTION_SPECIALIST_INVOCATIONS:
        raise TaskResolutionContractError(
            "specialist_invocation_counts: exceeds specialist cap"
        )
    updated = deepcopy(validated)
    updated_active_node = _active_node(updated)
    updated_active_node["status"] = "pending"
    updated["pending_dispatch"] = {
        "schema_version": TASK_PENDING_DISPATCH_VERSION,
        "task_node_id": updated_active_node["node_id"],
        "specialist": normalized_specialist,
        "subgoal": _require_semantic_subgoal(subgoal),
        "coding_objective_mode": normalized_mode,
        "phase": "selected",
    }
    return validate_task_resolution_checkpoint(updated)


def mark_pending_dispatch_started(
    checkpoint: TaskResolutionCheckpointV1,
) -> TaskResolutionCheckpointV1:
    """Consume one dispatch ledger entry before invoking a specialist."""

    validated = normalize_started_dispatch_ledger(checkpoint)
    pending_dispatch = validated["pending_dispatch"]
    if pending_dispatch is None or pending_dispatch["phase"] != "selected":
        raise TaskResolutionContractError(
            "pending_dispatch: expected selected dispatch before start"
        )
    if validated["dispatch_count"] >= MAX_TASK_RESOLUTION_DISPATCHES:
        raise TaskResolutionContractError("dispatch_count: task budget exhausted")
    active_node = _active_node(validated)
    if pending_dispatch["task_node_id"] != active_node["node_id"]:
        raise TaskResolutionContractError(
            "pending_dispatch.task_node_id: expected active task node"
        )
    specialist = pending_dispatch["specialist"]
    if specialist_invocation_count(
        validated,
        task_node_id=active_node["node_id"],
        specialist=specialist,
    ) >= MAX_TASK_RESOLUTION_SPECIALIST_INVOCATIONS:
        raise TaskResolutionContractError(
            "specialist_invocation_counts: exceeds specialist cap"
        )
    updated = deepcopy(validated)
    _increment_invocation_count(updated, active_node["node_id"], specialist)
    updated["dispatch_count"] += 1
    updated_active_node = _active_node(updated)
    updated_active_node["status"] = "resolving"
    updated_pending = updated["pending_dispatch"]
    if updated_pending is None:
        raise TaskResolutionContractError("pending_dispatch: missing after start")
    updated_pending["phase"] = "started"
    return validate_task_resolution_checkpoint(updated)


def consume_started_dispatch_as_unavailable(
    checkpoint: TaskResolutionCheckpointV1,
) -> TaskResolutionCheckpointV1:
    """Settle a lease-recovered started dispatch without relaunching it."""

    validated = normalize_started_dispatch_ledger(checkpoint)
    pending_dispatch = validated["pending_dispatch"]
    if pending_dispatch is None or pending_dispatch["phase"] != "started":
        raise TaskResolutionContractError(
            "pending_dispatch: expected started dispatch for recovery"
        )
    updated = deepcopy(validated)
    active_node = _active_node(updated)
    if pending_dispatch["task_node_id"] != active_node["node_id"]:
        raise TaskResolutionContractError(
            "pending_dispatch.task_node_id: expected active task node"
        )
    updated["pending_dispatch"] = None
    active_node["status"] = "blocked"
    updated["trace_summary"].append({
        "dispatch_index": updated["dispatch_count"],
        "task_node_id": active_node["node_id"],
        "specialist": pending_dispatch["specialist"],
        "result_status": "temporarily_unavailable",
        "reason": "A previously started specialist dispatch cannot be repeated.",
    })
    updated["terminal_status"] = "unavailable"
    return validate_task_resolution_checkpoint(updated)


def record_specialist_result(
    checkpoint: TaskResolutionCheckpointV1,
    result: TaskSpecialistResultV1,
) -> TaskResolutionCheckpointV1:
    """Merge one validated specialist outcome into the durable session state.

    The helper owns counters, incompatible-pair bookkeeping, and prompt-safe
    trace rows.  Semantic specialist selection remains outside this function.
    """

    validated_checkpoint = normalize_started_dispatch_ledger(checkpoint)
    validated_result = validate_task_specialist_result(result)
    if validated_checkpoint["terminal_status"]:
        raise TaskResolutionContractError(
            "checkpoint: cannot record a result after terminal status"
        )
    if (
        validated_checkpoint["pending_dispatch"] is None
        and validated_checkpoint["dispatch_count"] >= MAX_TASK_RESOLUTION_DISPATCHES
    ):
        raise TaskResolutionContractError("dispatch_count: task budget exhausted")

    updated = deepcopy(validated_checkpoint)
    active_node = _active_node(updated)
    specialist = validated_result["specialist"]
    _validate_result_evidence_scope(
        validated_result,
        task_node_id=active_node["node_id"],
    )
    pending_dispatch = validated_checkpoint["pending_dispatch"]
    dispatch_was_started = pending_dispatch is not None
    if pending_dispatch is not None:
        if pending_dispatch["phase"] != "started":
            raise TaskResolutionContractError(
                "pending_dispatch: selected dispatch cannot record a result"
            )
        if pending_dispatch["task_node_id"] != active_node["node_id"]:
            raise TaskResolutionContractError(
                "pending_dispatch.task_node_id: expected active task node"
            )
        if pending_dispatch["specialist"] != specialist:
            raise TaskResolutionContractError(
                "pending_dispatch.specialist: expected started specialist"
            )
        updated["pending_dispatch"] = None
    else:
        invocation_count = specialist_invocation_count(
            validated_checkpoint,
            task_node_id=active_node["node_id"],
            specialist=specialist,
        )
        if invocation_count >= MAX_TASK_RESOLUTION_SPECIALIST_INVOCATIONS:
            raise TaskResolutionContractError(
                "specialist_invocation_counts: exceeds specialist cap"
            )
        _increment_invocation_count(updated, active_node["node_id"], specialist)
        updated["dispatch_count"] += 1
    if not dispatch_was_started and (
        updated["dispatch_count"] > MAX_TASK_RESOLUTION_DISPATCHES
    ):
        raise TaskResolutionContractError("dispatch_count: task budget exhausted")
    trace_entry = {
        "dispatch_index": updated["dispatch_count"],
        "task_node_id": active_node["node_id"],
        "specialist": specialist,
        "result_status": validated_result["status"],
        "reason": validated_result["reason"],
    }
    updated["trace_summary"].append(trace_entry)

    status = validated_result["status"]
    if status == "incompatible":
        _record_incompatible_pair(updated, active_node["node_id"], specialist)
        active_node["status"] = "incompatible"
        updated["remaining_needs"] = list(validated_result["remaining_needs"])
        if not updated["terminal_status"] and not _can_continue(updated):
            updated["terminal_status"] = _partial_or_unavailable_status(updated)
    elif status in {"resolved", "partial"}:
        _merge_evidence(updated, validated_result["evidence"])
        active_node["status"] = "resolved"
        updated["remaining_needs"] = list(validated_result["remaining_needs"])
        _materialize_remaining_need_nodes(
            updated,
            completed_node_id=active_node["node_id"],
            remaining_needs=updated["remaining_needs"],
        )
        if _activate_first_eligible_pending_node(updated) and _can_continue(updated):
            updated["terminal_status"] = ""
        elif status == "partial" or _has_unresolved_nodes(updated):
            updated["terminal_status"] = _partial_or_unavailable_status(updated)
        else:
            updated["terminal_status"] = "resolved"
    elif status == "needs_user_input":
        active_node["status"] = "blocked"
        updated["remaining_needs"] = list(validated_result["remaining_needs"])
        updated["terminal_status"] = "needs_user_input"
    elif status == "approval_required":
        active_node["status"] = "blocked"
        updated["remaining_needs"] = list(validated_result["remaining_needs"])
        updated["terminal_status"] = "approval_required"
    elif status == "failed":
        active_node["status"] = "blocked"
        updated["remaining_needs"] = list(validated_result["remaining_needs"])
        updated["terminal_status"] = "failed"
    else:
        active_node["status"] = "pending"
        _merge_evidence(updated, validated_result["evidence"])
        updated["remaining_needs"] = list(validated_result["remaining_needs"])
        if not _can_continue(updated):
            updated["terminal_status"] = _partial_or_unavailable_status(updated)

    normalized = validate_task_resolution_checkpoint(updated)
    return normalized


def result_from_checkpoint(
    checkpoint: TaskResolutionCheckpointV1,
    *,
    status: str,
    prompt_safe_summary: str,
    completed_subgoals: list[str],
    coding_run_context: dict[str, object] | None = None,
) -> TaskResolutionResultV1:
    """Build one validated public result from a checkpoint transition."""

    validated = normalize_started_dispatch_ledger(checkpoint)
    checkpoint_projection: dict[str, object] = {}
    if status == "deferred":
        checkpoint_projection = dict(validated)
    evidence_state, evidence_excerpts, evidence_handles = (
        _result_evidence_projection(validated, status=status)
    )
    result: TaskResolutionResultV1 = {
        "schema_version": TASK_RESOLUTION_RESULT_VERSION,
        "semantic_objective": validated["semantic_objective"],
        "status": status,
        "scene_context": validated["scene_context"],
        "goal_continuation_ref": validated["goal_continuation_ref"],
        "evidence_state": evidence_state,
        "evidence_excerpts": evidence_excerpts,
        "evidence_handles": evidence_handles,
        "prompt_safe_summary": prompt_safe_summary,
        "evidence": list(validated["evidence"]),
        "completed_subgoals": list(completed_subgoals),
        "remaining_needs": list(validated["remaining_needs"]),
        "checkpoint": checkpoint_projection,
        "coding_run_context": dict(coding_run_context or {}),
    }
    normalized = validate_task_resolution_result(result)
    return normalized


def remaining_dispatch_budget(checkpoint: TaskResolutionCheckpointV1) -> int:
    """Return the immutable semantic-dispatch budget left for this session."""

    validated = normalize_started_dispatch_ledger(checkpoint)
    remaining = MAX_TASK_RESOLUTION_DISPATCHES - validated["dispatch_count"]
    return remaining


def has_attempted_specialist(
    checkpoint: TaskResolutionCheckpointV1,
    *,
    task_node_id: str,
    specialist: str,
) -> bool:
    """Return whether one node/specialist pair was already incompatible."""

    validated = normalize_started_dispatch_ledger(checkpoint)
    for attempt in validated["attempted_specialists"]:
        if (
            attempt["task_node_id"] == task_node_id
            and attempt["specialist"] == specialist
        ):
            return True
    return False


def specialist_invocation_count(
    checkpoint: TaskResolutionCheckpointV1,
    *,
    task_node_id: str,
    specialist: str,
) -> int:
    """Return the persisted invocation count for one specialist/node pair."""

    validated = normalize_started_dispatch_ledger(checkpoint)
    for row in validated["specialist_invocation_counts"]:
        if (
            row["task_node_id"] == task_node_id
            and row["specialist"] == specialist
        ):
            return row["count"]
    return 0


def normalize_started_dispatch_ledger(
    checkpoint: TaskResolutionCheckpointV1,
) -> TaskResolutionCheckpointV1:
    """Normalize an in-memory started transition before strict validation.

    Durable writers always persist the ledger-consuming ``started`` shape.
    This narrow normalization also lets a caller hand a just-started in-memory
    dispatch directly to completion or recovery without double-counting it.
    """

    if not isinstance(checkpoint, Mapping):
        return validate_task_resolution_checkpoint(checkpoint)
    candidate = deepcopy(dict(checkpoint))
    pending_dispatch = candidate.get("pending_dispatch")
    if not _started_dispatch_needs_ledger(candidate, pending_dispatch):
        return validate_task_resolution_checkpoint(candidate)
    if not isinstance(pending_dispatch, Mapping):
        return validate_task_resolution_checkpoint(candidate)
    task_node_id = pending_dispatch.get("task_node_id")
    specialist = pending_dispatch.get("specialist")
    if not isinstance(task_node_id, str) or not isinstance(specialist, str):
        return validate_task_resolution_checkpoint(candidate)
    candidate["dispatch_count"] += 1
    _increment_invocation_count(candidate, task_node_id, specialist)
    return validate_task_resolution_checkpoint(candidate)


def _started_dispatch_needs_ledger(
    checkpoint: Mapping[str, object],
    pending_dispatch: object,
) -> bool:
    """Recognize only an unpersisted, otherwise ordinary started transition."""

    if not isinstance(pending_dispatch, Mapping):
        return False
    if pending_dispatch.get("phase") != "started":
        return False
    dispatch_count = checkpoint.get("dispatch_count")
    trace_summary = checkpoint.get("trace_summary")
    invocation_counts = checkpoint.get("specialist_invocation_counts")
    if (
        not isinstance(dispatch_count, int)
        or isinstance(dispatch_count, bool)
        or not isinstance(trace_summary, list)
        or not isinstance(invocation_counts, list)
        or dispatch_count != len(trace_summary)
    ):
        return False
    invocation_total = 0
    for row in invocation_counts:
        if not isinstance(row, Mapping):
            return False
        count = row.get("count")
        if not isinstance(count, int) or isinstance(count, bool):
            return False
        invocation_total += count
    return invocation_total == dispatch_count


def _require_specialist(value: object) -> str:
    """Require one registered task-resolution specialist name."""

    if not isinstance(value, str) or value not in TASK_SPECIALISTS:
        raise TaskResolutionContractError("specialist: unsupported value")
    return value


def _require_coding_objective_mode(value: object) -> str:
    """Require one closed coding objective mode without semantic rewriting."""

    if not isinstance(value, str) or value not in TASK_CODING_OBJECTIVE_MODES:
        raise TaskResolutionContractError(
            "coding_objective_mode: unsupported value"
        )
    return value


def _validate_dispatch_mode(
    *,
    specialist: str,
    coding_objective_mode: str,
) -> None:
    """Keep code-objective modes exclusive to the coding specialist."""

    if specialist == "coding":
        if coding_objective_mode not in TASK_CODING_SPECIALIST_OBJECTIVE_MODES:
            raise TaskResolutionContractError(
                "coding_objective_mode: coding requires read_only or propose_patch"
            )
        return
    if coding_objective_mode != "none":
        raise TaskResolutionContractError(
            "coding_objective_mode: non-coding specialists require none"
        )


def _materialize_remaining_need_nodes(
    checkpoint: dict[str, Any],
    *,
    completed_node_id: str,
    remaining_needs: list[str],
) -> None:
    """Create deterministic dependent nodes for newly returned needs."""

    known_objectives = {
        _normalized_objective(node["objective"])
        for node in checkpoint["nodes"]
    }
    for remaining_need in remaining_needs:
        normalized_objective = _normalized_objective(remaining_need)
        if normalized_objective in known_objectives:
            continue
        if len(checkpoint["nodes"]) >= MAX_TASK_RESOLUTION_NODES:
            return
        node_id = _next_task_node_id(checkpoint)
        checkpoint["nodes"].append({
            "schema_version": TASK_RESOLUTION_NODE_VERSION,
            "node_id": node_id,
            "objective": remaining_need,
            "status": "pending",
            "depends_on": [completed_node_id],
        })
        known_objectives.add(normalized_objective)


def _normalized_objective(value: object) -> str:
    """Normalize exact semantic objectives for deterministic node deduplication."""

    normalized = _require_semantic_subgoal(value)
    return " ".join(normalized.split()).casefold()


def _next_task_node_id(checkpoint: Mapping[str, object]) -> str:
    """Return the next deterministic node identifier within the fixed cap."""

    raw_nodes = checkpoint.get("nodes")
    if not isinstance(raw_nodes, list):
        raise TaskResolutionContractError("nodes: expected list")
    node_ids = {
        row.get("node_id")
        for row in raw_nodes
        if isinstance(row, Mapping) and isinstance(row.get("node_id"), str)
    }
    index = len(raw_nodes) + 1
    while f"node-{index}" in node_ids:
        index += 1
    return f"node-{index}"


def _activate_first_eligible_pending_node(checkpoint: dict[str, Any]) -> bool:
    """Activate the first dependency-ready pending node in stable node order."""

    nodes_by_id = {
        node["node_id"]: node
        for node in checkpoint["nodes"]
    }
    for node in checkpoint["nodes"]:
        if node["status"] != "pending":
            continue
        if all(
            nodes_by_id[dependency_id]["status"] == "resolved"
            for dependency_id in node["depends_on"]
        ):
            checkpoint["active_node_id"] = node["node_id"]
            return True
    return False


def _has_unresolved_nodes(checkpoint: Mapping[str, object]) -> bool:
    """Return whether any node still needs work after a successful result."""

    nodes = checkpoint.get("nodes")
    if not isinstance(nodes, list):
        raise TaskResolutionContractError("nodes: expected list")
    return any(
        isinstance(node, Mapping) and node.get("status") != "resolved"
        for node in nodes
    )


def _can_continue(checkpoint: Mapping[str, object]) -> bool:
    """Require both semantic dispatch and selection-call budget for another node."""

    dispatch_count = checkpoint.get("dispatch_count")
    orchestrator_call_count = checkpoint.get("orchestrator_call_count")
    if not isinstance(dispatch_count, int) or isinstance(dispatch_count, bool):
        raise TaskResolutionContractError("dispatch_count: expected integer")
    if (
        not isinstance(orchestrator_call_count, int)
        or isinstance(orchestrator_call_count, bool)
    ):
        raise TaskResolutionContractError(
            "orchestrator_call_count: expected integer"
        )
    return (
        dispatch_count < MAX_TASK_RESOLUTION_DISPATCHES
        and orchestrator_call_count < MAX_TASK_RESOLUTION_ORCHESTRATOR_CALLS
    )


def _partial_or_unavailable_status(checkpoint: Mapping[str, object]) -> str:
    """Terminalize exhausted continuation as grounded partial or unavailable."""

    evidence = checkpoint.get("evidence")
    if isinstance(evidence, list) and evidence:
        return "partial"
    return "unavailable"


def _active_node(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    """Return the active node object held by the validated checkpoint."""

    active_node_id = checkpoint["active_node_id"]
    nodes = checkpoint["nodes"]
    for node in nodes:
        if node["node_id"] == active_node_id:
            return node
    raise TaskResolutionContractError("active_node_id: missing from nodes")


def _increment_invocation_count(
    checkpoint: dict[str, Any],
    task_node_id: str,
    specialist: str,
) -> None:
    """Increment one persisted same-specialist counter within its fixed cap."""

    if specialist not in TASK_SPECIALISTS:
        raise TaskResolutionContractError("specialist: unsupported value")
    counts = checkpoint["specialist_invocation_counts"]
    for row in counts:
        if (
            row["task_node_id"] == task_node_id
            and row["specialist"] == specialist
        ):
            row["count"] += 1
            return
    counts.append({
        "task_node_id": task_node_id,
        "specialist": specialist,
        "count": 1,
    })


def _record_incompatible_pair(
    checkpoint: dict[str, Any],
    task_node_id: str,
    specialist: str,
) -> None:
    """Record one non-repeat incompatible route and increment corrections."""

    for attempt in checkpoint["attempted_specialists"]:
        if (
            attempt["task_node_id"] == task_node_id
            and attempt["specialist"] == specialist
        ):
            raise TaskResolutionContractError(
                "attempted_specialists: incompatible pair already attempted"
            )
    if checkpoint["route_correction_count"] >= MAX_TASK_RESOLUTION_ROUTE_CORRECTIONS:
        checkpoint["terminal_status"] = _partial_or_unavailable_status(checkpoint)
        return
    checkpoint["attempted_specialists"].append({
        "task_node_id": task_node_id,
        "specialist": specialist,
    })
    checkpoint["route_correction_count"] += 1


def _merge_evidence(
    checkpoint: dict[str, Any],
    evidence: list[dict[str, object]],
) -> None:
    """Append new evidence while preserving its unique durable identifiers."""

    existing_ids = {
        item["evidence_id"]
        for item in checkpoint["evidence"]
    }
    for item in evidence:
        evidence_id = item["evidence_id"]
        if evidence_id not in existing_ids:
            checkpoint["evidence"].append(item)
            existing_ids.add(evidence_id)


def _validate_result_evidence_scope(
    result: TaskSpecialistResultV1,
    *,
    task_node_id: str,
) -> None:
    """Keep a specialist result's evidence attached to its active node."""

    specialist = result["specialist"]
    for evidence in result["evidence"]:
        if evidence["specialist"] != specialist:
            raise TaskResolutionContractError(
                "evidence.specialist: expected result specialist"
            )
        if evidence["task_node_id"] != task_node_id:
            raise TaskResolutionContractError(
                "evidence.task_node_id: expected active task node"
            )


def _result_evidence_projection(
    checkpoint: TaskResolutionCheckpointV1,
    *,
    status: str,
) -> tuple[str, list[str], list[str]]:
    """Return result-visible evidence only for factual terminal states."""

    state_by_status = {
        "resolved": "complete",
        "partial": "partial",
        "deferred": "pending",
        "needs_user_input": "blocked",
        "approval_required": "blocked",
        "unavailable": "blocked",
        "failed": "blocked",
    }
    evidence_state = state_by_status.get(status)
    if evidence_state is None:
        raise TaskResolutionContractError(
            "status: unsupported task-resolution result status"
        )
    if evidence_state not in {"complete", "partial"}:
        return evidence_state, [], []
    evidence_excerpts = [row["summary"] for row in checkpoint["evidence"]]
    evidence_handles = [row["evidence_id"] for row in checkpoint["evidence"]]
    return evidence_state, evidence_excerpts, evidence_handles


def _request_goal_continuation_ref(
    request: Mapping[str, object],
) -> GoalContinuationRefV1:
    """Require the upstream deterministic continuation reference."""

    raw_ref = request.get("goal_continuation_ref")
    if raw_ref is None:
        raise TaskResolutionContractError(
            "goal_continuation_ref: required for task-resolution request"
        )
    try:
        continuation_ref = validate_goal_continuation_ref(raw_ref)
    except CognitiveEpisodeValidationError as exc:
        raise TaskResolutionContractError(
            f"goal_continuation_ref: invalid reference: {exc}"
        ) from exc
    return continuation_ref


def _request_text(request: Mapping[str, object], field_name: str) -> str:
    """Read one required resolver request semantic field."""

    value = request.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise TaskResolutionContractError(
            f"{field_name}: expected non-empty resolver request text"
        )
    normalized = value.strip()
    return normalized


def _require_semantic_subgoal(value: object) -> str:
    """Validate the bounded semantic objective selected for one dispatch."""

    if not isinstance(value, str):
        raise TaskResolutionContractError("semantic_subgoal: expected string")
    normalized = value.strip()
    if not normalized or len(normalized) > 1200:
        raise TaskResolutionContractError("semantic_subgoal: invalid text")
    return normalized
