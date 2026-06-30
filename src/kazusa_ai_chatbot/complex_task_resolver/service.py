"""Standalone complex-task resolver orchestration service."""

from __future__ import annotations

import ast
import logging
import re

from .contracts import (
    ALLOWED_NODE_ATTEMPT_ACTIONS,
    ALLOWED_NODE_ATTEMPT_STATUSES,
    COMPLEX_TASK_FOLLOWUP_TASK_VERSION,
    COMPLEX_TASK_GRAPH_VERSION,
    COMPLEX_TASK_NODE_ATTEMPT_VERSION,
    COMPLEX_TASK_NODE_VERSION,
    COMPLEX_TASK_RESOLUTION_PACKET_VERSION,
    COMPLEX_TASK_RESOLVER_OPTIONS_VERSION,
    COMPLEX_TASK_SUBAGENT_REQUEST_VERSION,
    COMPLEX_TASK_SUBAGENT_RESULT_VERSION,
    ComplexTaskFollowupTaskV1,
    ComplexTaskGraphV1,
    ComplexTaskNodeV1,
    ComplexTaskResolutionPacketV1,
    ComplexTaskResolverContextV1,
    ComplexTaskResolverOptionsV1,
    ComplexTaskResolverRequestV1,
    ComplexTaskSubagentResultV1,
    ComplexTaskValidationError,
    validate_complex_task_graph,
    validate_complex_task_followup_task,
    validate_complex_task_resolution_packet,
    validate_complex_task_resolver_context,
    validate_complex_task_resolver_options,
    validate_complex_task_resolver_request,
    validate_complex_task_subagent_request,
    validate_complex_task_subagent_result,
)
from .graph import find_next_active_node
from .stages import (
    plan_complex_task_graph as _plan_stage_handler,
    resolve_complex_task_node as _node_stage_handler,
    review_complex_task_collapse as _collapse_stage_handler,
    synthesize_complex_task_packet as _synthesizer_stage_handler,
)
from .subagent import (
    _SUBAGENT_DEFAULT_ACTIONS,
    _SUBAGENT_NAMES,
    _SUBAGENT_SUPPORTED_ACTIONS,
    create_subagents,
    owned_subagent_for_node_kind,
)

logger = logging.getLogger(__name__)

_PRODUCTION_NODE_STAGE_HANDLER = _node_stage_handler
_PRODUCTION_COLLAPSE_STAGE_HANDLER = _collapse_stage_handler
_PRODUCTION_SYNTHESIZER_STAGE_HANDLER = _synthesizer_stage_handler

_NODE_UPDATE_FIELDS = frozenset((
    "status",
    "investigation_summary",
    "knowledge_we_know_so_far",
    "knowledge_still_lacking",
    "recommended_next_iteration",
    "evidence_boundary_notes",
    "evidence_refs",
    "source_observation_ids",
    "collapsed_into",
))

_PLANNER_TASK_KINDS = frozenset((
    "subtask",
    "evidence_need",
    "algorithmic_task",
    "synthesis",
))
_SEMANTIC_WORK_TYPE_TO_KIND = {
    "subtask": "subtask",
    "public_evidence": "evidence_need",
    "calculation": "algorithmic_task",
    "arithmetic": "algorithmic_task",
    "analysis": "synthesis",
    "synthesis": "synthesis",
}
_SEMANTIC_COMPLETION_TO_NODE_STATUS = {
    "completed": "resolved",
    "blocked": "blocked",
    "not_answerable": "cannot_answer",
}
_FORBIDDEN_SEMANTIC_OUTPUT_KEYS = frozenset((
    "schema_version",
    "node_id",
    "parent_id",
    "source_node_id",
    "target_node_id",
    "attempt_index",
    "status",
    "trace",
    "cache",
))

_FOLLOWUP_TASKS_PER_SOURCE_LIMIT = 2
_TRACE_TEXT_LIMIT = 4000
_TRACE_LIST_LIMIT = 12
_TRACE_DICT_LIMIT = 24
_NUMERIC_TEXT_PATTERN = re.compile(
    r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$"
)


async def resolve_complex_task(
    request: ComplexTaskResolverRequestV1,
    context: ComplexTaskResolverContextV1,
    options: ComplexTaskResolverOptionsV1 | None = None,
) -> ComplexTaskResolutionPacketV1:
    """Resolve a bounded task graph through production-owned stages."""

    try:
        validated_request = validate_complex_task_resolver_request(request)
        validated_context = validate_complex_task_resolver_context(context)
        raw_options = options
        if raw_options is None:
            raw_options = {
                "schema_version": COMPLEX_TASK_RESOLVER_OPTIONS_VERSION,
                "limits": {},
            }
        validated_options = validate_complex_task_resolver_options(raw_options)
    except ComplexTaskValidationError as exc:
        packet = _failed_packet_from_exception(
            root_question=_root_question_from_request(request),
            reason=f"invalid resolver input: {exc}",
            failure_stage="input_validation",
        )
        return packet
    try:
        packet = await _resolve_complex_task_validated(
            validated_request,
            validated_context,
            validated_options,
        )
    except Exception as exc:
        logger.exception(f"Complex task resolver failed: {exc}")
        packet = _failed_packet_from_exception(
            root_question=validated_request["objective"],
            reason=(
                "complex task resolver failed: "
                f"{type(exc).__name__}: {_safe_failure_text(str(exc))}"
            ),
            failure_stage="internal_resolution",
        )
    return packet


async def _resolve_complex_task_validated(
    validated_request: ComplexTaskResolverRequestV1,
    validated_context: ComplexTaskResolverContextV1,
    validated_options: ComplexTaskResolverOptionsV1,
) -> ComplexTaskResolutionPacketV1:
    """Run the resolver after public inputs have been validated."""

    trace_summary: dict[str, object] = {
        "iterations": 0,
        "collapse_count": 0,
        "node_attempt_count": 0,
        "node_attempt_log": [],
        "subagent_calls": 0,
        "subagent_call_log": [],
        "followup_created_count": 0,
        "followup_rejected_count": 0,
        "followup_event_log": [],
        "stage_io_log": [],
    }
    graph = await _plan_graph(
        validated_request,
        validated_context,
        validated_options,
        trace_summary,
    )
    max_nodes = _option_limit(validated_options, "max_nodes", 8)
    max_iterations = _option_limit(
        validated_options,
        "max_iterations",
        max_nodes,
    )
    await _run_graph_traversal(
        validated_request=validated_request,
        validated_context=validated_context,
        validated_options=validated_options,
        graph=graph,
        trace_summary=trace_summary,
        max_iterations=max_iterations,
    )
    packet = await _synthesize_packet(
        validated_request,
        validated_context,
        validated_options,
        graph,
        trace_summary,
    )
    return packet


async def _run_graph_traversal(
    *,
    validated_request: ComplexTaskResolverRequestV1,
    validated_context: ComplexTaskResolverContextV1,
    validated_options: ComplexTaskResolverOptionsV1,
    graph: ComplexTaskGraphV1,
    trace_summary: dict[str, object],
    max_iterations: int,
) -> int:
    """Resolve pending graph nodes through the normal bounded traversal path."""

    iterations_run = 0
    for _iteration_index in range(max_iterations):
        active_node_id = find_next_active_node(graph)
        if active_node_id is None:
            break
        graph["active_node_id"] = active_node_id
        graph = validate_complex_task_graph(graph)
        node_response = await _resolve_active_node(
            validated_request,
            validated_context,
            validated_options,
            graph,
            active_node_id,
            trace_summary,
        )
        remaining_followup_iterations = max_iterations - _iteration_index - 1
        _apply_active_node_response(
            graph=graph,
            active_node_id=active_node_id,
            response=node_response,
            trace_summary=trace_summary,
            remaining_followup_iterations=remaining_followup_iterations,
        )
        _record_traversal(graph, active_node_id)
        graph = validate_complex_task_graph(graph)
        collapse_response = await _review_collapse(
            validated_request,
            validated_context,
            validated_options,
            graph,
            active_node_id,
        )
        _apply_collapse_response(
            graph,
            active_node_id,
            collapse_response,
            trace_summary,
        )
        graph = validate_complex_task_graph(graph)
        trace_summary["iterations"] = int(trace_summary["iterations"]) + 1
        iterations_run += 1
    return iterations_run


def _failed_packet_from_exception(
    *,
    root_question: str,
    reason: str,
    failure_stage: str,
) -> ComplexTaskResolutionPacketV1:
    """Build a validated failed packet for public-boundary failures."""

    safe_reason = _safe_failure_text(reason)
    root_node = _make_graph_node(
        node_id="root",
        parent_id=None,
        depth=0,
        objective=root_question,
        node_kind="root",
        status="cannot_answer",
        children=[],
    )
    _set_node_semantic_projection(
        root_node,
        investigation_summary=safe_reason,
        knowledge_we_know_so_far=[],
        knowledge_still_lacking=[safe_reason],
        recommended_next_iteration=[
            "Treat this investigation as unavailable unless another evidence "
            "path exists."
        ],
        evidence_boundary_notes=[
            "The resolver did not complete its structural investigation."
        ],
    )
    graph = {
        "schema_version": COMPLEX_TASK_GRAPH_VERSION,
        "root_node_id": "root",
        "active_node_id": "root",
        "nodes": {"root": root_node},
        "traversal_order": ["root"],
        "collapse_events": [],
        "max_nodes": 1,
        "max_depth": 1,
    }
    packet = {
        "schema_version": COMPLEX_TASK_RESOLUTION_PACKET_VERSION,
        "root_question": root_question,
        "investigation_summary": (
            "The resolver failed before completing public answer research: "
            f"{safe_reason}"
        ),
        "knowledge_we_know_so_far": [],
        "knowledge_still_lacking": [safe_reason],
        "recommended_next_iteration": [
            "Treat this investigation as unavailable unless another evidence "
            "path exists."
        ],
        "evidence_boundary_notes": [
            "The resolver did not complete its structural investigation."
        ],
        "graph": graph,
        "trace_summary": {
            "iterations": 0,
            "collapse_count": 0,
            "node_attempt_count": 0,
            "subagent_calls": 0,
            "failure_stage": failure_stage,
            "failure_reason": safe_reason,
        },
    }
    validated_packet = validate_complex_task_resolution_packet(packet)
    return validated_packet


def _root_question_from_request(request: object) -> str:
    """Read a prompt-safe root question from an unvalidated request."""

    if isinstance(request, dict):
        objective = request.get("objective")
        if isinstance(objective, str) and objective.strip():
            root_question = _safe_failure_text(objective)
            return root_question
    root_question = "invalid complex-task request"
    return root_question


def _safe_failure_text(value: str) -> str:
    """Return a compact single-line failure string for public packets."""

    collapsed = " ".join(value.strip().split())
    if not collapsed:
        return "unknown resolver failure"
    max_length = 300
    if len(collapsed) > max_length:
        collapsed = collapsed[: max_length - 3].rstrip() + "..."
    return collapsed


async def _plan_graph(
    request: ComplexTaskResolverRequestV1,
    context: ComplexTaskResolverContextV1,
    options: ComplexTaskResolverOptionsV1,
    trace_summary: dict[str, object],
) -> ComplexTaskGraphV1:
    """Call the planner and map semantic decomposition into graph state."""

    payload = {
        "stage": "graph_planner",
        "objective": request["objective"],
        "reason": request["reason"],
        "context": _compact_context(context),
    }
    response = await _plan_stage_handler(payload)
    _record_stage_io(
        trace_summary=trace_summary,
        stage="graph_planner",
        prompt_payload=payload,
        parsed_output=response,
    )
    graph = _graph_from_semantic_decomposition(
        request=request,
        response=response,
        options=options,
    )
    return graph


def _graph_from_semantic_decomposition(
    *,
    request: ComplexTaskResolverRequestV1,
    response: dict[str, object],
    options: ComplexTaskResolverOptionsV1,
) -> ComplexTaskGraphV1:
    """Create strict graph nodes from explicit semantic planner tasks."""

    tasks = _semantic_tasks(response)
    max_nodes = _option_limit(options, "max_nodes", 8)
    max_depth = _option_limit(options, "max_depth", 3)
    if _should_group_mixed_prerequisites(tasks, max_nodes=max_nodes):
        graph = _grouped_prerequisite_graph_from_tasks(
            request=request,
            tasks=tasks,
            max_nodes=max_nodes,
            max_depth=max_depth,
        )
        return graph
    if len(tasks) > max_nodes - 1:
        raise ComplexTaskValidationError("planner tasks: exceeds max_nodes")
    root_children = [
        _task_node_id(index)
        for index, _ in enumerate(tasks, start=1)
    ]
    nodes: dict[str, ComplexTaskNodeV1] = {
        "root": _make_graph_node(
            node_id="root",
            parent_id=None,
            depth=0,
            objective=request["objective"],
            node_kind="root",
            status="expanded",
            children=root_children,
        )
    }
    for index, task in enumerate(tasks, start=1):
        node_id = _task_node_id(index)
        nodes[node_id] = _make_graph_node(
            node_id=node_id,
            parent_id="root",
            depth=1,
            objective=task["objective"],
            node_kind=_graph_node_kind_from_semantic_kind(task["kind"]),
            status="pending",
            children=[],
        )
    graph = {
        "schema_version": COMPLEX_TASK_GRAPH_VERSION,
        "root_node_id": "root",
        "active_node_id": root_children[0],
        "nodes": nodes,
        "traversal_order": ["root"],
        "collapse_events": [],
        "max_nodes": max_nodes,
        "max_depth": max_depth,
    }
    validated_graph = validate_complex_task_graph(graph)
    return validated_graph


def _should_group_mixed_prerequisites(
    tasks: list[dict[str, str]],
    *,
    max_nodes: int,
) -> bool:
    """Return whether flat prerequisite tasks should form a trunk."""

    task_kinds = {task["kind"] for task in tasks}
    if "subtask" in task_kinds:
        return False
    if "synthesis" not in task_kinds:
        return False
    if not {"evidence_need", "algorithmic_task"}.issubset(task_kinds):
        return False
    needed_nodes = len(tasks) + 2
    return needed_nodes <= max_nodes


def _grouped_prerequisite_graph_from_tasks(
    *,
    request: ComplexTaskResolverRequestV1,
    tasks: list[dict[str, str]],
    max_nodes: int,
    max_depth: int,
) -> ComplexTaskGraphV1:
    """Build a graph with one prerequisite trunk for mixed task kinds."""

    if max_depth < 2:
        raise ComplexTaskValidationError(
            "planner tasks: mixed prerequisites require max_depth >= 2"
        )
    prerequisite_tasks = [
        task
        for task in tasks
        if task["kind"] != "synthesis"
    ]
    synthesis_tasks = [
        task
        for task in tasks
        if task["kind"] == "synthesis"
    ]
    prerequisite_child_ids = [
        f"task_1_{index}"
        for index, _ in enumerate(prerequisite_tasks, start=1)
    ]
    root_children = ["task_1"] + [
        _task_node_id(index)
        for index, _ in enumerate(synthesis_tasks, start=2)
    ]
    nodes: dict[str, ComplexTaskNodeV1] = {
        "root": _make_graph_node(
            node_id="root",
            parent_id=None,
            depth=0,
            objective=request["objective"],
            node_kind="root",
            status="expanded",
            children=root_children,
        ),
        "task_1": _make_graph_node(
            node_id="task_1",
            parent_id="root",
            depth=1,
            objective="Resolve prerequisite evidence and calculation branches.",
            node_kind="subtask",
            status="expanded",
            children=prerequisite_child_ids,
        ),
    }
    for index, task in enumerate(prerequisite_tasks, start=1):
        child_id = f"task_1_{index}"
        nodes[child_id] = _make_graph_node(
            node_id=child_id,
            parent_id="task_1",
            depth=2,
            objective=task["objective"],
            node_kind=_graph_node_kind_from_semantic_kind(task["kind"]),
            status="pending",
            children=[],
        )
    for index, task in enumerate(synthesis_tasks, start=2):
        node_id = _task_node_id(index)
        nodes[node_id] = _make_graph_node(
            node_id=node_id,
            parent_id="root",
            depth=1,
            objective=task["objective"],
            node_kind=_graph_node_kind_from_semantic_kind(task["kind"]),
            status="pending",
            children=[],
        )
    graph = {
        "schema_version": COMPLEX_TASK_GRAPH_VERSION,
        "root_node_id": "root",
        "active_node_id": prerequisite_child_ids[0],
        "nodes": nodes,
        "traversal_order": ["root"],
        "collapse_events": [],
        "max_nodes": max_nodes,
        "max_depth": max_depth,
    }
    validated_graph = validate_complex_task_graph(graph)
    return validated_graph


def _semantic_tasks(response: dict[str, object]) -> list[dict[str, str]]:
    """Validate local-LLM-friendly planner tasks."""

    if "tasks" not in response:
        raise ComplexTaskValidationError("planner response: missing tasks")
    raw_tasks = response["tasks"]
    if not isinstance(raw_tasks, list) or not raw_tasks:
        raise ComplexTaskValidationError("planner tasks: expected non-empty list")
    tasks: list[dict[str, str]] = []
    for raw_task in raw_tasks:
        if not isinstance(raw_task, dict):
            raise ComplexTaskValidationError("planner task: expected object")
        objective = raw_task.get("objective")
        if not isinstance(objective, str) or not objective.strip():
            raise ComplexTaskValidationError(
                "planner task objective: expected non-empty string"
            )
        raw_task_kind = raw_task.get("kind")
        if not isinstance(raw_task_kind, str):
            raise ComplexTaskValidationError(
                "planner task kind: expected known task kind"
            )
        task_kind = _SEMANTIC_WORK_TYPE_TO_KIND.get(
            raw_task_kind,
            raw_task_kind,
        )
        if task_kind not in _PLANNER_TASK_KINDS:
            raise ComplexTaskValidationError(
                "planner task kind: expected known task kind"
            )
        tasks.append({
            "objective": objective.strip(),
            "kind": task_kind,
        })
    return tasks


def _make_graph_node(
    *,
    node_id: str,
    parent_id: str | None,
    depth: int,
    objective: str,
    node_kind: str,
    status: str,
    children: list[str],
) -> ComplexTaskNodeV1:
    """Build one strict graph node from validated service data."""

    node = {
        "schema_version": COMPLEX_TASK_NODE_VERSION,
        "node_id": node_id,
        "parent_id": parent_id,
        "depth": depth,
        "objective": objective,
        "node_kind": node_kind,
        "status": status,
        "children": children,
        "investigation_summary": "",
        "knowledge_we_know_so_far": [],
        "knowledge_still_lacking": [],
        "recommended_next_iteration": [],
        "evidence_boundary_notes": [],
        "evidence_refs": [],
        "source_observation_ids": [],
        "collapsed_into": None,
        "attempts": [],
    }
    return node


def _graph_node_kind_from_semantic_kind(task_kind: str) -> str:
    """Map local-LLM planner hints into graph-safe node kinds."""

    return_value = task_kind
    return return_value


def _task_node_id(index: int) -> str:
    """Return a deterministic node id for a planner task position."""

    node_id = f"task_{index}"
    return node_id


async def _resolve_active_node(
    request: ComplexTaskResolverRequestV1,
    context: ComplexTaskResolverContextV1,
    options: ComplexTaskResolverOptionsV1,
    graph: ComplexTaskGraphV1,
    active_node_id: str,
    trace_summary: dict[str, object],
) -> dict[str, object]:
    """Resolve one active node through a bounded local attempt loop."""

    active_node = graph["nodes"][active_node_id]
    max_attempts = _option_limit(options, "max_node_attempts", 3)
    last_response: dict[str, object] = {}
    for _attempt_number in range(max_attempts):
        attempt_index = len(active_node["attempts"]) + 1
        response = await _resolve_active_node_once(
            request=request,
            context=context,
            options=options,
            graph=graph,
            active_node_id=active_node_id,
            trace_summary=trace_summary,
        )
        last_response = response
        attempt = _node_attempt_from_response(
            node=active_node,
            response=response,
            attempt_index=attempt_index,
        )
        _append_node_attempt(active_node, attempt, trace_summary)
        if _is_terminal_node_response(response):
            return response
    blocked_response = _node_attempts_exhausted_response(active_node, last_response)
    return blocked_response


async def _resolve_active_node_once(
    *,
    request: ComplexTaskResolverRequestV1,
    context: ComplexTaskResolverContextV1,
    options: ComplexTaskResolverOptionsV1,
    graph: ComplexTaskGraphV1,
    active_node_id: str,
    trace_summary: dict[str, object],
) -> dict[str, object]:
    """Run one active-node resolver pass and optional subagent dispatch."""

    active_node = graph["nodes"][active_node_id]
    owned_subagent_name = _node_owned_subagent(active_node)
    payload = {
        "stage": "active_node_resolver",
        "root_question": request["objective"],
        "active_node": _compact_node(active_node),
        "parent_chain_summary": _parent_chain_summary(graph, active_node),
        "sibling_summaries": _sibling_summaries(graph, active_node),
        "context": _compact_context(context),
    }
    raw_response = await _node_stage_handler(payload)
    _record_stage_io(
        trace_summary=trace_summary,
        stage="active_node_resolver",
        prompt_payload=payload,
        parsed_output=raw_response,
    )
    response = _normalize_node_stage_response(
        raw_response,
        graph=graph,
        active_node=active_node,
        allow_internal_envelope=not _using_production_node_stage_handler(),
    )
    if "subagent_request" in response:
        subagent_result = await _run_requested_subagent(
            options=options,
            context=context,
            graph=graph,
            active_node=active_node,
            request_payload=response["subagent_request"],
            trace_summary=trace_summary,
        )
        response["subagent_result"] = subagent_result
        trace_summary["subagent_calls"] = int(trace_summary["subagent_calls"]) + 1
        return response
    if (
        owned_subagent_name == "algorithmic"
        and "node_update" in response
        and "followup_tasks" not in response
    ):
        response = await _repair_subagent_owned_node(
            request=request,
            context=context,
            options=options,
            graph=graph,
            active_node=active_node,
            owned_subagent_name=owned_subagent_name,
            invalid_response=response,
            trace_summary=trace_summary,
        )
        if "subagent_request" in response:
            subagent_result = await _run_requested_subagent(
                options=options,
                context=context,
                graph=graph,
                active_node=active_node,
                request_payload=response["subagent_request"],
                trace_summary=trace_summary,
            )
            response["subagent_result"] = subagent_result
            trace_summary["subagent_calls"] = int(
                trace_summary["subagent_calls"]
            ) + 1
            return response
    if (
        owned_subagent_name == "algorithmic"
        and "node_expansion" in response
        and _is_unproductive_owned_subagent_expansion(active_node, response)
    ):
        response = await _repair_subagent_owned_node(
            request=request,
            context=context,
            options=options,
            graph=graph,
            active_node=active_node,
            owned_subagent_name=owned_subagent_name,
            invalid_response=response,
            trace_summary=trace_summary,
        )
        if "subagent_request" in response:
            subagent_result = await _run_requested_subagent(
                options=options,
                context=context,
                graph=graph,
                active_node=active_node,
                request_payload=response["subagent_request"],
                trace_summary=trace_summary,
            )
            response["subagent_result"] = subagent_result
            trace_summary["subagent_calls"] = int(
                trace_summary["subagent_calls"]
            ) + 1
            return response
    if (
        owned_subagent_name is not None
        and "node_expansion" in response
        and _node_expansion_would_exceed_graph_limits(
            graph,
            active_node,
            response["node_expansion"],
        )
    ):
        subagent_result = await _run_subagent(
            options=options,
            context=context,
            graph=graph,
            request_payload=_fallback_subagent_request(
                active_node,
                owned_subagent_name,
            ),
            trace_summary=trace_summary,
        )
        response = {"subagent_result": subagent_result}
        trace_summary["subagent_calls"] = int(trace_summary["subagent_calls"]) + 1
    if (
        owned_subagent_name is not None
        and "node_expansion" in response
        and _is_unproductive_owned_subagent_expansion(active_node, response)
    ):
        subagent_result = await _run_subagent(
            options=options,
            context=context,
            graph=graph,
            request_payload=_fallback_subagent_request(
                active_node,
                owned_subagent_name,
            ),
            trace_summary=trace_summary,
        )
        response = {"subagent_result": subagent_result}
        trace_summary["subagent_calls"] = int(trace_summary["subagent_calls"]) + 1
    if (
        owned_subagent_name is not None
        and "node_update" in response
        and "followup_tasks" not in response
    ):
        subagent_result = await _run_subagent(
            options=options,
            context=context,
            graph=graph,
            request_payload=_fallback_subagent_request(
                active_node,
                owned_subagent_name,
            ),
            trace_summary=trace_summary,
        )
        response = {"subagent_result": subagent_result}
        trace_summary["subagent_calls"] = int(trace_summary["subagent_calls"]) + 1
    return response


def _node_expansion_would_exceed_graph_limits(
    graph: ComplexTaskGraphV1,
    parent: ComplexTaskNodeV1,
    expansion: object,
) -> bool:
    """Return whether a requested expansion cannot fit in the graph."""

    if parent["children"]:
        return False
    if not isinstance(expansion, dict):
        return False
    children = expansion.get("children")
    if not isinstance(children, list):
        return False
    child_depth = parent["depth"] + 1
    if child_depth > graph["max_depth"]:
        return True
    if len(graph["nodes"]) + len(children) > graph["max_nodes"]:
        return True
    return False


def _is_unproductive_owned_subagent_expansion(
    active_node: ComplexTaskNodeV1,
    response: dict[str, object],
) -> bool:
    """Return whether an owned node tried to expand into the same task."""

    expansion = response["node_expansion"]
    if not isinstance(expansion, dict):
        return False
    children = expansion.get("children")
    if not isinstance(children, list) or len(children) != 1:
        return False
    child = children[0]
    if not isinstance(child, dict):
        return False
    child_kind = child.get("kind")
    if child_kind != active_node["node_kind"]:
        return False
    child_objective = child.get("objective")
    if not isinstance(child_objective, str):
        return False
    return_value = _same_structural_text(
        active_node["objective"],
        child_objective,
    )
    return return_value


def _normalize_node_stage_response(
    response: dict[str, object],
    *,
    graph: ComplexTaskGraphV1,
    active_node: ComplexTaskNodeV1,
    allow_internal_envelope: bool,
) -> dict[str, object]:
    """Map semantic LLM decisions into internal graph response envelopes."""

    if _looks_like_internal_node_response(response):
        if not allow_internal_envelope:
            raise ComplexTaskValidationError(
                "node resolver semantic output must not use internal envelopes"
            )
        return response
    decision = response.get("decision")
    if not isinstance(decision, str):
        return response
    _reject_forbidden_semantic_output_keys(response, "node_decision")
    if decision == "expand":
        children = response.get("children")
        normalized = {
            "node_expansion": {
                "children": _semantic_continuation_tasks(children),
            },
        }
        return normalized
    if decision == "record_knowledge":
        normalized = {
            "node_update": _semantic_node_update(response),
        }
        followups = _semantic_continuation_tasks(
            response.get("continuation_tasks")
        )
        if followups:
            normalized["followup_tasks"] = followups
        return normalized
    if decision == "use_subagent":
        request = _semantic_subagent_request(response, graph, active_node)
        return {"subagent_request": request}
    if decision == "continue_locally":
        normalized = {
            "node_attempt": {
                "action": _semantic_attempt_action(response.get("action")),
                "result_summary": _semantic_text(
                    response.get("result_summary")
                ),
                "blockers": _semantic_text_list(response.get("blockers")),
                "next_action": _semantic_text(response.get("next_action")),
            },
        }
        followups = _semantic_continuation_tasks(
            response.get("continuation_tasks")
        )
        if followups:
            normalized["followup_tasks"] = followups
        return normalized
    return response


def _using_production_node_stage_handler() -> bool:
    """Return whether active-node calls use the production LLM handler."""

    return_value = _node_stage_handler is _PRODUCTION_NODE_STAGE_HANDLER
    return return_value


def _using_production_collapse_stage_handler() -> bool:
    """Return whether collapse calls use the production LLM handler."""

    return_value = _collapse_stage_handler is _PRODUCTION_COLLAPSE_STAGE_HANDLER
    return return_value


def _using_production_synthesizer_stage_handler() -> bool:
    """Return whether synthesis calls use the production LLM handler."""

    return_value = (
        _synthesizer_stage_handler is _PRODUCTION_SYNTHESIZER_STAGE_HANDLER
    )
    return return_value


def _reject_forbidden_semantic_output_keys(value: object, path: str) -> None:
    """Reject deterministic transport keys in semantic LLM output."""

    if isinstance(value, dict):
        for key, nested_value in value.items():
            if isinstance(key, str) and key in _FORBIDDEN_SEMANTIC_OUTPUT_KEYS:
                raise ComplexTaskValidationError(
                    f"{path}.{key}: deterministic field is not semantic output"
                )
            _reject_forbidden_semantic_output_keys(
                nested_value,
                f"{path}.{key}",
            )
        return
    if isinstance(value, list):
        for index, nested_value in enumerate(value):
            _reject_forbidden_semantic_output_keys(
                nested_value,
                f"{path}[{index}]",
            )


def _looks_like_internal_node_response(response: dict[str, object]) -> bool:
    """Return whether deterministic tests supplied an internal envelope."""

    internal_keys = (
        "subagent_request",
        "subagent_result",
        "node_expansion",
        "node_update",
        "node_attempt",
        "followup_tasks",
    )
    return_value = any(key in response for key in internal_keys)
    return return_value


def _semantic_node_update(response: dict[str, object]) -> dict[str, object]:
    """Create an internal node update from semantic LLM fields."""

    completion = response.get("completion")
    if isinstance(completion, str):
        node_status = _SEMANTIC_COMPLETION_TO_NODE_STATUS.get(
            completion,
            "resolved",
        )
    else:
        lacking = _semantic_text_list(response.get("knowledge_still_lacking"))
        node_status = "blocked" if lacking else "resolved"
    update = {
        "status": node_status,
        "investigation_summary": _semantic_text(
            response.get("investigation_summary")
        ),
        "knowledge_we_know_so_far": _semantic_text_list(
            response.get("knowledge_we_know_so_far")
        ),
        "knowledge_still_lacking": _semantic_text_list(
            response.get("knowledge_still_lacking")
        ),
        "recommended_next_iteration": _semantic_text_list(
            response.get("recommended_next_iteration")
        ),
        "evidence_boundary_notes": _semantic_text_list(
            response.get("evidence_boundary_notes")
        ),
    }
    return update


def _semantic_subagent_request(
    response: dict[str, object],
    graph: ComplexTaskGraphV1,
    active_node: ComplexTaskNodeV1,
) -> dict[str, object]:
    """Create typed subagent IO from a semantic capability request."""

    capability = response.get("capability")
    if not isinstance(capability, str) or capability not in _SUBAGENT_NAMES:
        owned_capability = _node_owned_subagent(active_node)
        if owned_capability is None:
            raise ComplexTaskValidationError(
                "semantic subagent request: unknown capability"
            )
        capability = owned_capability
    action = response.get("action")
    if not isinstance(action, str) or not action.strip():
        action = _SUBAGENT_DEFAULT_ACTIONS[capability]
    objective = _semantic_text(response.get("objective"))
    if not objective:
        objective = active_node["objective"]
    payload = response.get("request")
    if not isinstance(payload, dict):
        payload = {}
    constraints = response.get("requirements")
    if not isinstance(constraints, dict):
        constraints = {}
    if capability == "algorithmic":
        payload = _attach_algorithmic_source_node_ids(
            payload=payload,
            graph=graph,
            active_node_id=active_node["node_id"],
        )
    request = {
        "schema_version": COMPLEX_TASK_SUBAGENT_REQUEST_VERSION,
        "node_id": active_node["node_id"],
        "subagent": capability,
        "action": action.strip(),
        "objective": objective,
        "payload": payload,
        "constraints": constraints,
    }
    return request


def _semantic_continuation_tasks(value: object) -> list[dict[str, object]]:
    """Map semantic executable child work into internal follow-up tasks."""

    if not isinstance(value, list):
        return []
    tasks: list[dict[str, object]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        objective = _semantic_text(item.get("objective"))
        reason = _semantic_text(item.get("reason"))
        work_type = item.get("work_type")
        if not isinstance(work_type, str):
            work_type = item.get("kind")
        if not objective or not reason:
            continue
        if not isinstance(work_type, str):
            continue
        task_kind = _SEMANTIC_WORK_TYPE_TO_KIND.get(work_type, work_type)
        if not isinstance(task_kind, str):
            continue
        tasks.append({
            "schema_version": COMPLEX_TASK_FOLLOWUP_TASK_VERSION,
            "objective": objective,
            "kind": task_kind,
            "reason": reason,
        })
    return tasks


def _semantic_attempt_action(value: object) -> str:
    """Map semantic local continuation labels into known attempt actions."""

    if not isinstance(value, str):
        return "refine_search"
    normalized = value.strip().lower().replace(" ", "_")
    if normalized in ALLOWED_NODE_ATTEMPT_ACTIONS:
        return normalized
    if "expand" in normalized or "decompos" in normalized:
        return "expand_node"
    if "subagent" in normalized or "evidence" in normalized:
        return "call_subagent"
    if "calcul" in normalized:
        return "revise_calculation_request"
    if "conflict" in normalized:
        return "review_source_conflict"
    if "block" in normalized:
        return "block"
    return "refine_search"


def _attach_algorithmic_source_node_ids(
    *,
    payload: dict[str, object],
    graph: ComplexTaskGraphV1,
    active_node_id: str,
) -> dict[str, object]:
    """Attach graph provenance to semantic arithmetic operand rows."""

    normalized_payload = dict(payload)
    input_values = normalized_payload.get("input_values")
    if not isinstance(input_values, list):
        return normalized_payload
    normalized_inputs: list[object] = []
    for raw_entry in input_values:
        if not isinstance(raw_entry, dict):
            normalized_inputs.append(raw_entry)
            continue
        entry = dict(raw_entry)
        source_node_id = entry.get("source_node_id")
        if not isinstance(source_node_id, str) or not source_node_id.strip():
            source_text = entry.get("source_text")
            inferred_node_id = _infer_operand_source_node_id(
                source_text,
                graph=graph,
                active_node_id=active_node_id,
            )
            if inferred_node_id is not None:
                entry["source_node_id"] = inferred_node_id
        normalized_inputs.append(entry)
    normalized_payload["input_values"] = normalized_inputs
    return normalized_payload


def _infer_operand_source_node_id(
    source_text: object,
    *,
    graph: ComplexTaskGraphV1,
    active_node_id: str,
) -> str | None:
    """Find the graph node whose semantic projection contains source text."""

    if not isinstance(source_text, str) or not source_text.strip():
        return None
    candidate_ids = [active_node_id]
    for node_id in graph["nodes"]:
        if node_id not in candidate_ids:
            candidate_ids.append(node_id)
    for node_id in candidate_ids:
        node = graph["nodes"][node_id]
        if _contains_structural_text(
            _algorithmic_source_node_text(node),
            source_text,
        ):
            return node_id
    return None


def _is_terminal_node_response(response: dict[str, object]) -> bool:
    """Return whether a node resolver response can be applied to the graph."""

    terminal = any(
        field_name in response
        for field_name in ("subagent_result", "node_expansion", "node_update")
    )
    if not terminal and "node_attempt" in response and "followup_tasks" in response:
        terminal = True
    return terminal


def _node_attempt_from_response(
    *,
    node: ComplexTaskNodeV1,
    response: dict[str, object],
    attempt_index: int,
) -> dict[str, object]:
    """Build one compact active-node attempt observation."""

    raw_attempt = response.get("node_attempt")
    if isinstance(raw_attempt, dict):
        action = _safe_attempt_enum(
            raw_attempt.get("action"),
            ALLOWED_NODE_ATTEMPT_ACTIONS,
            _infer_attempt_action(response),
        )
        status = _safe_attempt_enum(
            raw_attempt.get("status"),
            ALLOWED_NODE_ATTEMPT_STATUSES,
            _infer_attempt_status(response),
        )
        result_summary = _safe_attempt_string(
            raw_attempt.get("result_summary"),
            _infer_attempt_result_summary(response),
        )
        blockers = _safe_attempt_string_list(raw_attempt.get("blockers"))
        next_action = _safe_attempt_string(raw_attempt.get("next_action"), "")
    else:
        action = _infer_attempt_action(response)
        status = _infer_attempt_status(response)
        result_summary = _infer_attempt_result_summary(response)
        blockers = []
        next_action = ""
    if status in ("blocked", "cannot_answer", "invalid") and not blockers:
        blockers = [_fallback_attempt_blocker(response)]
    attempt = {
        "schema_version": COMPLEX_TASK_NODE_ATTEMPT_VERSION,
        "attempt_index": attempt_index,
        "action": action,
        "status": status,
        "input_summary": f"{node['node_kind']}: {node['objective']}",
        "result_summary": result_summary,
        "blockers": blockers,
        "next_action": next_action,
    }
    return attempt


def _safe_attempt_enum(
    value: object,
    allowed_values: frozenset[str],
    fallback: str,
) -> str:
    """Return a known attempt enum value or a deterministic fallback."""

    if isinstance(value, str) and value in allowed_values:
        return_value = value
        return return_value
    return fallback


def _safe_attempt_string(value: object, fallback: str) -> str:
    """Return a prompt-safe string from an attempt field."""

    if isinstance(value, str):
        return_value = value.strip()
        return return_value
    return fallback


def _safe_attempt_string_list(value: object) -> list[str]:
    """Return non-empty string blockers from a raw attempt field."""

    if not isinstance(value, list):
        return []
    blockers: list[str] = []
    for item in value:
        if isinstance(item, str) and item.strip():
            blockers.append(item.strip())
    return blockers


def _infer_attempt_action(response: dict[str, object]) -> str:
    """Infer the local action represented by a node resolver response."""

    if "subagent_result" in response or "subagent_request" in response:
        return "call_subagent"
    if "node_expansion" in response:
        return "expand_node"
    if "node_update" in response:
        update = response["node_update"]
        if isinstance(update, dict) and update.get("status") == "cannot_answer":
            return "block"
        return "resolve_direct"
    if "node_attempt" in response:
        return "refine_search"
    return "block"


def _infer_attempt_status(response: dict[str, object]) -> str:
    """Infer attempt status from terminal or intermediate response shape."""

    if "subagent_result" in response:
        subagent_result = validate_complex_task_subagent_result(
            response["subagent_result"]
        )
        if subagent_result["status"] == "resolved":
            return "resolved"
        if subagent_result["status"] == "partial":
            return "partial"
        if subagent_result["status"] == "invalid":
            return "invalid"
        if subagent_result["status"] == "unavailable":
            return "blocked"
        return "blocked"
    if "node_expansion" in response:
        return "resolved"
    if "node_update" in response:
        update = response["node_update"]
        if isinstance(update, dict):
            status = update.get("status")
            if status == "resolved":
                return "resolved"
            if status == "cannot_answer":
                return "cannot_answer"
            if status == "blocked":
                return "blocked"
        return "partial"
    if "node_attempt" in response:
        return "partial"
    return "invalid"


def _infer_attempt_result_summary(response: dict[str, object]) -> str:
    """Return a compact result summary for an attempt observation."""

    if "subagent_result" in response:
        subagent_result = validate_complex_task_subagent_result(
            response["subagent_result"]
        )
        if subagent_result["resolved"]:
            return _summarize_result(subagent_result["result"])
        return "; ".join(subagent_result["unresolved_items"])
    if "node_expansion" in response:
        return "node expansion requested"
    if "node_update" in response:
        update = response["node_update"]
        if isinstance(update, dict):
            summary = update.get("investigation_summary")
            if isinstance(summary, str) and summary.strip():
                return summary.strip()
            for field_name in (
                "knowledge_we_know_so_far",
                "knowledge_still_lacking",
                "recommended_next_iteration",
                "evidence_boundary_notes",
            ):
                rows = update.get(field_name)
                if not isinstance(rows, list):
                    continue
                for row in rows:
                    if isinstance(row, str) and row.strip():
                        return row.strip()
        return "node update requested"
    if "node_attempt" in response:
        return "non-terminal node attempt recorded"
    return "node resolver did not return an applicable result"


def _fallback_attempt_blocker(response: dict[str, object]) -> str:
    """Return a deterministic blocker for an unsuccessful attempt."""

    if "subagent_result" in response:
        subagent_result = validate_complex_task_subagent_result(
            response["subagent_result"]
        )
        if subagent_result["unresolved_items"]:
            return_value = subagent_result["unresolved_items"][0]
            return return_value
    return "node resolver did not produce a resolved result"


def _append_node_attempt(
    node: ComplexTaskNodeV1,
    attempt: dict[str, object],
    trace_summary: dict[str, object],
) -> None:
    """Append one node attempt and mirror a compact entry into trace."""

    node["attempts"].append(attempt)
    trace_summary["node_attempt_count"] = (
        int(trace_summary["node_attempt_count"]) + 1
    )
    attempt_log = trace_summary["node_attempt_log"]
    if isinstance(attempt_log, list):
        attempt_log.append({
            "node_id": node["node_id"],
            "attempt_index": attempt["attempt_index"],
            "action": attempt["action"],
            "status": attempt["status"],
            "result_summary": attempt["result_summary"],
            "blockers": attempt["blockers"],
            "next_action": attempt["next_action"],
        })


def _node_attempts_exhausted_response(
    node: ComplexTaskNodeV1,
    last_response: dict[str, object],
) -> dict[str, object]:
    """Build a fail-closed node update after exhausting local attempts."""

    blockers: list[str] = []
    for attempt in node["attempts"]:
        for blocker in attempt["blockers"]:
            if blocker not in blockers:
                blockers.append(blocker)
    if not blockers:
        blockers.append(_fallback_attempt_blocker(last_response))
    reason = "node-resolution loop exhausted: " + "; ".join(blockers)
    response = {
        "resolver_loop_exhausted": True,
        "node_update": {
            "status": "blocked",
            "investigation_summary": (
                "The node did not produce structured terminal semantic output "
                "within its bounded attempts."
            ),
            "knowledge_we_know_so_far": _node_known_so_far(node),
            "knowledge_still_lacking": [
                "structured terminal output or usable evidence for this node",
                reason,
            ],
            "recommended_next_iteration": [
                "Use a narrower node objective or a different evidence path "
                "before reattempting this branch."
            ],
            "evidence_boundary_notes": [
                "Node resolution stopped at the configured attempt limit."
            ],
        },
    }
    return response


async def _repair_subagent_owned_node(
    *,
    request: ComplexTaskResolverRequestV1,
    context: ComplexTaskResolverContextV1,
    options: ComplexTaskResolverOptionsV1,
    graph: ComplexTaskGraphV1,
    active_node: ComplexTaskNodeV1,
    owned_subagent_name: str,
    invalid_response: dict[str, object],
    trace_summary: dict[str, object],
) -> dict[str, object]:
    """Ask the resolver stage for typed subagent IO after prose resolution."""

    payload = {
        "stage": "subagent_request_repair",
        "root_question": request["objective"],
        "active_node": _compact_node(active_node),
        "parent_chain_summary": _parent_chain_summary(graph, active_node),
        "sibling_summaries": _sibling_summaries(graph, active_node),
        "required_subagent": owned_subagent_name,
        "previous_attempt_summary": _semantic_repair_context(
            invalid_response,
        ),
        "context": _compact_context(context),
    }
    del options

    raw_response = await _node_stage_handler(payload)
    _record_stage_io(
        trace_summary=trace_summary,
        stage="subagent_request_repair",
        prompt_payload=payload,
        parsed_output=raw_response,
    )
    response = _normalize_node_stage_response(
        raw_response,
        graph=graph,
        active_node=active_node,
        allow_internal_envelope=not _using_production_node_stage_handler(),
    )
    return response


def _semantic_repair_context(response: dict[str, object]) -> dict[str, object]:
    """Return semantic repair context without internal response envelopes."""

    summary = _infer_attempt_result_summary(response)
    blockers: list[str] = []
    if "node_update" in response:
        update = response["node_update"]
        if isinstance(update, dict):
            blockers = _semantic_text_list(update.get("knowledge_still_lacking"))
    if not blockers:
        blockers = [_fallback_attempt_blocker(response)]
    context = {
        "problem": (
            "The previous attempt did not provide the resolver-local "
            "subagent request needed by this node."
        ),
        "observed_result": summary,
        "blockers": blockers,
    }
    return context


async def _run_requested_subagent(
    *,
    options: ComplexTaskResolverOptionsV1,
    context: ComplexTaskResolverContextV1,
    graph: ComplexTaskGraphV1,
    active_node: ComplexTaskNodeV1,
    request_payload: object,
    trace_summary: dict[str, object],
) -> ComplexTaskSubagentResultV1:
    """Run a requested subagent or fail closed through node ownership."""

    try:
        result = await _run_subagent(
            options=options,
            context=context,
            graph=graph,
            request_payload=request_payload,
            trace_summary=trace_summary,
        )
        return result
    except ComplexTaskValidationError as exc:
        fallback_name = _fallback_subagent_name(request_payload, active_node)
        if fallback_name is None:
            raise
        result = await _run_subagent(
            options=options,
            context=context,
            graph=graph,
            request_payload=_fallback_subagent_request(
                active_node,
                fallback_name,
                reason=f"invalid subagent request: {exc}",
            ),
            trace_summary=trace_summary,
        )
        return result


async def _run_subagent(
    *,
    options: ComplexTaskResolverOptionsV1,
    context: ComplexTaskResolverContextV1,
    graph: ComplexTaskGraphV1,
    request_payload: object,
    trace_summary: dict[str, object],
) -> ComplexTaskSubagentResultV1:
    """Run one typed resolver-local subagent from the module-owned registry."""

    subagent_request = validate_complex_task_subagent_request(request_payload)
    subagent_name = subagent_request["subagent"]
    subagents = _internal_subagents()
    if subagent_name not in subagents:
        raise ComplexTaskValidationError("subagents: missing requested subagent")
    supported_actions = _SUBAGENT_SUPPORTED_ACTIONS.get(subagent_name)
    if (
        supported_actions is not None
        and subagent_request["action"] not in supported_actions
    ):
        raise ComplexTaskValidationError(
            "subagent request.action: unsupported for requested subagent"
        )
    node = graph["nodes"].get(subagent_request["node_id"])
    if node is None:
        node = graph["nodes"][graph["root_node_id"]]
    subagent_context = {
        "root_question": graph["nodes"][graph["root_node_id"]]["objective"],
        "parent_chain_summary": _parent_chain_summary(graph, node),
        "sibling_summaries": _sibling_summaries(graph, node),
        "available_evidence": context["available_evidence"],
        "time_context": context["time_context"],
    }
    max_attempts = _option_limit(options, "max_subagent_attempts", 1)
    validation_error = _subagent_request_validation_error(
        request=subagent_request,
        graph=graph,
    )
    if validation_error is not None:
        subagent_result = _invalid_subagent_result(
            request=subagent_request,
            max_attempts=max_attempts,
            reason=validation_error,
        )
        _record_subagent_call(
            trace_summary=trace_summary,
            request=subagent_request,
            result=subagent_result,
        )
        return subagent_result
    raw_result = await subagents[subagent_name].run(
        subagent_request,
        subagent_context,
        max_attempts=max_attempts,
    )
    subagent_result = validate_complex_task_subagent_result(raw_result)
    _record_subagent_call(
        trace_summary=trace_summary,
        request=subagent_request,
        result=subagent_result,
    )
    return subagent_result


def _subagent_request_validation_error(
    *,
    request: dict[str, object],
    graph: ComplexTaskGraphV1,
) -> str | None:
    """Return a structural validation error for resolver-owned subagent IO."""

    node_id = request["node_id"]
    if not isinstance(node_id, str):
        return "subagent request.node_id: expected string"
    node = graph["nodes"].get(node_id)
    if node is not None:
        owned_subagent = _node_owned_subagent(node)
        if owned_subagent is not None and request["subagent"] != owned_subagent:
            return (
                f"subagent request: {node['node_kind']} nodes require "
                f"{owned_subagent} subagent"
            )
    if request["subagent"] != "algorithmic":
        return None
    if request["action"] != "evaluate_expression":
        return None
    return _algorithmic_operand_provenance_error(request, graph)


def _algorithmic_operand_provenance_error(
    request: dict[str, object],
    graph: ComplexTaskGraphV1,
) -> str | None:
    """Validate that arithmetic operands are declared from graph evidence."""

    payload = request["payload"]
    if not isinstance(payload, dict):
        return "algorithmic payload: expected object"
    expression = payload.get("expression")
    if not isinstance(expression, str) or not expression.strip():
        return "algorithmic payload.expression: expected non-empty string"
    expression_numbers = _expression_numeric_literals(expression)
    if not expression_numbers:
        return None

    input_values = payload.get("input_values")
    if not isinstance(input_values, list):
        return "algorithmic payload.input_values: expected list"
    formula_constants = payload.get("formula_constants", [])
    if not isinstance(formula_constants, list):
        return "algorithmic payload.formula_constants: expected list"

    declared_numbers: set[str] = set()
    for index, raw_entry in enumerate(input_values):
        if not isinstance(raw_entry, dict):
            return f"algorithmic input_values[{index}]: expected object"
        error = _validate_algorithmic_input_value(raw_entry, index, graph)
        if error is not None:
            return error
        declared_numbers.add(_numeric_text(raw_entry["value"]))

    for index, raw_entry in enumerate(formula_constants):
        if not isinstance(raw_entry, dict):
            return f"algorithmic formula_constants[{index}]: expected object"
        value = raw_entry.get("value")
        purpose = raw_entry.get("purpose")
        if not _is_numeric_value(value):
            return f"algorithmic formula_constants[{index}].value: expected number text"
        if not isinstance(purpose, str) or not purpose.strip():
            return f"algorithmic formula_constants[{index}].purpose: expected text"
        declared_numbers.add(_numeric_text(value))

    missing_numbers = [
        number for number in expression_numbers
        if number not in declared_numbers
    ]
    if missing_numbers:
        return (
            "algorithmic payload: numeric literals missing operand provenance: "
            + ", ".join(missing_numbers)
        )
    return None


def _validate_algorithmic_input_value(
    entry: dict[object, object],
    index: int,
    graph: ComplexTaskGraphV1,
) -> str | None:
    """Validate one declared arithmetic input against existing graph text."""

    value = entry.get("value")
    source_node_id = entry.get("source_node_id")
    source_text = entry.get("source_text")
    if not _is_numeric_value(value):
        return f"algorithmic input_values[{index}].value: expected number text"
    if not isinstance(source_node_id, str) or not source_node_id.strip():
        return f"algorithmic input_values[{index}].source_node_id: expected node id"
    source_node = graph["nodes"].get(source_node_id)
    if source_node is None:
        return (
            f"algorithmic input_values[{index}].source_node_id: "
            "unknown graph node"
        )
    if not isinstance(source_text, str) or not source_text.strip():
        return f"algorithmic input_values[{index}].source_text: expected text"
    value_text = _numeric_text(value)
    if value_text not in source_text:
        return (
            f"algorithmic input_values[{index}].source_text: "
            "must include the declared value"
        )
    if not _contains_structural_text(
        _algorithmic_source_node_text(source_node),
        source_text,
    ):
        return (
            f"algorithmic input_values[{index}].source_text: "
            "not found in source node projection"
        )
    return None


def _expression_numeric_literals(expression: str) -> list[str]:
    """Extract numeric constants from the calculator expression AST."""

    try:
        parsed = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ComplexTaskValidationError(
            f"algorithmic payload.expression: invalid syntax: {exc.msg}"
        ) from exc
    values: list[str] = []
    seen_values: set[str] = set()
    for child in ast.walk(parsed):
        if not isinstance(child, ast.Constant):
            continue
        value = child.value
        if isinstance(value, bool):
            continue
        if _is_numeric_value(value):
            value_text = _numeric_text(value)
            if value_text not in seen_values:
                values.append(value_text)
                seen_values.add(value_text)
    return values


def _is_numeric_value(value: object) -> bool:
    """Return whether a JSON-like value can declare one numeric operand."""

    if isinstance(value, bool):
        return False
    if isinstance(value, int | float):
        return True
    if isinstance(value, str):
        return bool(_NUMERIC_TEXT_PATTERN.fullmatch(value.strip()))
    return False


def _numeric_text(value: object) -> str:
    """Return the canonical text used for operand declaration matching."""

    if not _is_numeric_value(value):
        raise ComplexTaskValidationError("numeric operand: expected number text")
    if isinstance(value, str):
        return value.strip()
    return str(value)


def _algorithmic_source_node_text(node: ComplexTaskNodeV1) -> str:
    """Return graph text that may structurally support an arithmetic operand."""

    rows = [
        node["objective"],
        node["investigation_summary"],
        *node["knowledge_we_know_so_far"],
        *node["knowledge_still_lacking"],
        *node["evidence_boundary_notes"],
    ]
    return "\n".join(row for row in rows if row)


def _contains_structural_text(haystack: str, needle: str) -> bool:
    """Compare prompt text after whitespace folding."""

    folded_haystack = " ".join(haystack.split())
    folded_needle = " ".join(needle.split())
    return folded_needle in folded_haystack


def _same_structural_text(left: str, right: str) -> bool:
    """Return whether two model-authored task strings are the same task."""

    folded_left = " ".join(left.casefold().split())
    folded_right = " ".join(right.casefold().split())
    return_value = folded_left == folded_right
    return return_value


def _invalid_subagent_result(
    *,
    request: dict[str, object],
    max_attempts: int,
    reason: str,
) -> ComplexTaskSubagentResultV1:
    """Build a fail-closed subagent result for invalid internal IO."""

    result = {
        "schema_version": COMPLEX_TASK_SUBAGENT_RESULT_VERSION,
        "resolved": False,
        "status": "invalid",
        "result": {},
        "attempts": 1,
        "cache": {
            "enabled": False,
            "hit": False,
            "cache_name": str(request["subagent"]),
            "reason": "invalid subagent request",
        },
        "trace": {
            "node_id": request["node_id"],
            "action": request["action"],
            "attempt_limit": max_attempts,
        },
        "unresolved_items": [reason],
    }
    return validate_complex_task_subagent_result(result)


def _internal_subagents() -> dict[str, object]:
    """Return resolver-owned subagents, not caller-provided dependencies."""

    subagents = create_subagents()
    return subagents


def _node_owned_subagent(node: ComplexTaskNodeV1) -> str | None:
    """Return the internal subagent that owns a node kind, when any."""

    return_value = owned_subagent_for_node_kind(node["node_kind"])
    return return_value


def _fallback_subagent_request(
    node: ComplexTaskNodeV1,
    subagent_name: str,
    *,
    reason: str = "subagent-owned node cannot be resolved by direct prose",
) -> dict[str, object]:
    """Build a fail-closed request when a subagent-owned node got prose."""

    action = _SUBAGENT_DEFAULT_ACTIONS.get(subagent_name, "collect_evidence")
    request = {
        "schema_version": "complex_task_subagent_request.v1",
        "node_id": node["node_id"],
        "subagent": subagent_name,
        "action": action,
        "objective": node["objective"],
        "payload": {},
        "constraints": {
            "reason": reason,
        },
    }
    return request


def _fallback_subagent_name(
    request_payload: object,
    active_node: ComplexTaskNodeV1,
) -> str | None:
    """Choose a fail-closed internal subagent for a malformed request."""

    owned_name = _node_owned_subagent(active_node)
    if owned_name is not None:
        return owned_name
    if isinstance(request_payload, dict):
        requested_name = request_payload.get("subagent")
        if requested_name in _SUBAGENT_NAMES:
            return requested_name
    return None


def _record_subagent_call(
    *,
    trace_summary: dict[str, object],
    request: dict[str, object],
    result: ComplexTaskSubagentResultV1,
) -> None:
    """Append a compact, read-only subagent call record to packet trace."""

    call_log = trace_summary["subagent_call_log"]
    if not isinstance(call_log, list):
        raise ComplexTaskValidationError(
            "trace_summary.subagent_call_log: expected list"
        )
    call_log.append({
        "subagent": request["subagent"],
        "node_id": request["node_id"],
        "action": request["action"],
        "objective": request["objective"],
        "payload": _trace_safe_value(request["payload"]),
        "constraints": _trace_safe_value(request["constraints"]),
        "resolved": result["resolved"],
        "status": result["status"],
        "result": _trace_safe_value(result["result"]),
        "attempts": result["attempts"],
        "trace": _trace_safe_value(result["trace"]),
        "unresolved_items": list(result["unresolved_items"]),
    })


def _record_stage_io(
    *,
    trace_summary: dict[str, object],
    stage: str,
    prompt_payload: dict[str, object],
    parsed_output: dict[str, object],
) -> None:
    """Append bounded LLM stage input and parsed output for review artifacts."""

    stage_log = trace_summary["stage_io_log"]
    if not isinstance(stage_log, list):
        raise ComplexTaskValidationError("trace_summary.stage_io_log: expected list")
    stage_log.append({
        "stage": stage,
        "prompt_payload": _trace_safe_value(prompt_payload),
        "parsed_output": _trace_safe_value(parsed_output),
    })


def _trace_safe_value(value: object) -> object:
    """Return a bounded JSON-like value suitable for packet trace output."""

    if isinstance(value, str):
        if len(value) <= _TRACE_TEXT_LIMIT:
            return value
        return value[: _TRACE_TEXT_LIMIT - 3].rstrip() + "..."
    if isinstance(value, bool | int | float) or value is None:
        return value
    if isinstance(value, list):
        return [
            _trace_safe_value(item)
            for item in value[:_TRACE_LIST_LIMIT]
        ]
    if isinstance(value, tuple):
        return [
            _trace_safe_value(item)
            for item in value[:_TRACE_LIST_LIMIT]
        ]
    if isinstance(value, dict):
        safe_dict: dict[str, object] = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= _TRACE_DICT_LIMIT:
                safe_dict["__truncated__"] = True
                break
            safe_dict[str(key)] = _trace_safe_value(item)
        return safe_dict
    return str(value)


def _apply_active_node_response(
    *,
    graph: ComplexTaskGraphV1,
    active_node_id: str,
    response: dict[str, object],
    trace_summary: dict[str, object],
    remaining_followup_iterations: int,
) -> None:
    """Merge one active-node result into the graph."""

    node = graph["nodes"][active_node_id]
    if node["node_kind"] == "synthesis":
        dependency_blockers = _synthesis_dependency_blockers(
            graph,
            active_node_id,
        )
        if dependency_blockers:
            _apply_active_followup_tasks(
                graph=graph,
                source_node_id=active_node_id,
                response=response,
                trace_summary=trace_summary,
                remaining_followup_iterations=remaining_followup_iterations,
            )
            if node["status"] == "expanded":
                return
            _block_synthesis_node_with_dependencies(node, dependency_blockers)
            return
    if "subagent_result" in response:
        subagent_result = validate_complex_task_subagent_result(
            response["subagent_result"]
        )
        _apply_subagent_result(node, subagent_result)
        _reject_misplaced_followup_tasks(
            graph=graph,
            source_node_id=active_node_id,
            response=response,
            trace_summary=trace_summary,
            reason="active-node followup_tasks require node_update or node_attempt",
        )
        return
    if "node_expansion" in response:
        _apply_node_expansion(graph, active_node_id, response["node_expansion"])
        _reject_misplaced_followup_tasks(
            graph=graph,
            source_node_id=active_node_id,
            response=response,
            trace_summary=trace_summary,
            reason="active-node followup_tasks require node_update or node_attempt",
        )
        return
    if "node_update" in response:
        loop_exhausted = response.get("resolver_loop_exhausted") is True
        has_followup_tasks = "followup_tasks" in response
        if (
            node["node_kind"] == "algorithmic_task"
            and not loop_exhausted
            and not has_followup_tasks
        ):
            _block_algorithmic_node_without_subagent(node)
            return
        if (
            node["node_kind"] == "evidence_need"
            and not loop_exhausted
            and not has_followup_tasks
        ):
            _block_evidence_node_without_subagent(node)
            return
        _merge_node_update(node, response["node_update"])
        _apply_active_followup_tasks(
            graph=graph,
            source_node_id=active_node_id,
            response=response,
            trace_summary=trace_summary,
            remaining_followup_iterations=remaining_followup_iterations,
        )
        return
    if "node_attempt" in response and "followup_tasks" in response:
        _apply_attempt_followup_response(
            graph=graph,
            source_node_id=active_node_id,
            response=response,
            trace_summary=trace_summary,
            remaining_followup_iterations=remaining_followup_iterations,
        )
        return
    raise ComplexTaskValidationError("node resolver response: missing node result")


def _apply_node_expansion(
    graph: ComplexTaskGraphV1,
    active_node_id: str,
    expansion: object,
) -> None:
    """Create bounded child nodes under a complicated active node."""

    parent = graph["nodes"][active_node_id]
    if parent["children"]:
        raise ComplexTaskValidationError("node_expansion: parent already has children")
    if not isinstance(expansion, dict):
        raise ComplexTaskValidationError("node_expansion: expected object")
    if "children" not in expansion:
        raise ComplexTaskValidationError("node_expansion.children: missing")
    children = expansion["children"]
    tasks = _semantic_tasks({"tasks": children})
    child_depth = parent["depth"] + 1
    if child_depth > graph["max_depth"]:
        _block_node_expansion_rejected(
            parent,
            reason="node_expansion: exceeds max_depth",
        )
        return
    if len(graph["nodes"]) + len(tasks) > graph["max_nodes"]:
        _block_node_expansion_rejected(
            parent,
            reason="node_expansion: exceeds max_nodes",
        )
        return
    child_ids: list[str] = []
    for task in tasks:
        child_id = _next_child_node_id(graph, active_node_id)
        child_ids.append(child_id)
        graph["nodes"][child_id] = _make_graph_node(
            node_id=child_id,
            parent_id=active_node_id,
            depth=child_depth,
            objective=task["objective"],
            node_kind=_graph_node_kind_from_semantic_kind(task["kind"]),
            status="pending",
            children=[],
        )
    parent["status"] = "expanded"
    parent["children"] = child_ids
    _set_node_semantic_projection(
        parent,
        investigation_summary=f"Expanded into {len(child_ids)} child tasks.",
        knowledge_we_know_so_far=[],
        knowledge_still_lacking=[],
        recommended_next_iteration=[],
        evidence_boundary_notes=[],
    )


def _apply_active_followup_tasks(
    *,
    graph: ComplexTaskGraphV1,
    source_node_id: str,
    response: dict[str, object],
    trace_summary: dict[str, object],
    remaining_followup_iterations: int,
) -> None:
    """Create bounded follow-up child nodes from an active-node update."""

    if "followup_tasks" not in response:
        return
    source_node = graph["nodes"][source_node_id]
    created_count = _create_followup_nodes(
        graph=graph,
        parent_node_id=source_node_id,
        source_key=source_node_id,
        source_stage="active_node_resolver",
        raw_tasks=response["followup_tasks"],
        trace_summary=trace_summary,
        rejection_target=source_node,
        allow_creation=_has_followup_iteration_budget(
            response["followup_tasks"],
            remaining_followup_iterations,
        ),
        rejection_reason="active-node follow-up rejected at max_iterations",
    )
    if created_count < 1:
        return
    source_node["status"] = "expanded"
    _append_missing_items(
        source_node["evidence_boundary_notes"],
        ["Resolver created follow-up child nodes from structured tasks."],
    )


def _apply_attempt_followup_response(
    *,
    graph: ComplexTaskGraphV1,
    source_node_id: str,
    response: dict[str, object],
    trace_summary: dict[str, object],
    remaining_followup_iterations: int,
) -> None:
    """Apply a node_attempt that delegates progress to follow-up child nodes."""

    source_node = graph["nodes"][source_node_id]
    created_count = _create_followup_nodes(
        graph=graph,
        parent_node_id=source_node_id,
        source_key=source_node_id,
        source_stage="active_node_resolver",
        raw_tasks=response["followup_tasks"],
        trace_summary=trace_summary,
        rejection_target=source_node,
        allow_creation=_has_followup_iteration_budget(
            response["followup_tasks"],
            remaining_followup_iterations,
        ),
        rejection_reason="active-node follow-up rejected at max_iterations",
    )
    attempt_summary = "Structured follow-up tasks were requested."
    next_action = ""
    if source_node["attempts"]:
        latest_attempt = source_node["attempts"][-1]
        attempt_summary = latest_attempt["result_summary"]
        next_action = latest_attempt["next_action"]
    if created_count > 0:
        source_node["status"] = "expanded"
        recommended_next_iteration: list[str] = []
        if next_action:
            recommended_next_iteration.append(next_action)
        lacking = list(source_node["knowledge_still_lacking"])
        boundary_notes = list(source_node["evidence_boundary_notes"])
        _append_missing_items(
            boundary_notes,
            ["Resolver created follow-up child nodes from structured tasks."],
        )
        _set_node_semantic_projection(
            source_node,
            investigation_summary=attempt_summary,
            knowledge_we_know_so_far=_node_known_so_far(source_node),
            knowledge_still_lacking=lacking,
            recommended_next_iteration=recommended_next_iteration,
            evidence_boundary_notes=boundary_notes,
        )
        return
    source_node["status"] = "blocked"
    lacking = list(source_node["knowledge_still_lacking"])
    if not lacking:
        lacking.append("valid follow-up task within resolver graph limits")
    boundary_notes = list(source_node["evidence_boundary_notes"])
    if not boundary_notes:
        boundary_notes.append("Resolver follow-up tasks were not created.")
    _set_node_semantic_projection(
        source_node,
        investigation_summary=attempt_summary,
        knowledge_we_know_so_far=_node_known_so_far(source_node),
        knowledge_still_lacking=lacking,
        recommended_next_iteration=[],
        evidence_boundary_notes=boundary_notes,
    )


def _reject_misplaced_followup_tasks(
    *,
    graph: ComplexTaskGraphV1,
    source_node_id: str,
    response: dict[str, object],
    trace_summary: dict[str, object],
    reason: str,
) -> None:
    """Reject follow-up tasks emitted beside an unsupported response shape."""

    if "followup_tasks" not in response:
        return
    source_node = graph["nodes"][source_node_id]
    _create_followup_nodes(
        graph=graph,
        parent_node_id=source_node_id,
        source_key=source_node_id,
        source_stage="active_node_resolver",
        raw_tasks=response["followup_tasks"],
        trace_summary=trace_summary,
        rejection_target=source_node,
        allow_creation=False,
        rejection_reason=reason,
    )


def _has_followup_iteration_budget(
    raw_tasks: object,
    remaining_followup_iterations: int,
) -> bool:
    """Return whether structured follow-ups can be traversed this run."""

    if not isinstance(raw_tasks, list):
        return_value = False
        return return_value
    task_count = len(raw_tasks)
    return_value = task_count <= remaining_followup_iterations
    return return_value


def _create_followup_nodes(
    *,
    graph: ComplexTaskGraphV1,
    parent_node_id: str,
    source_key: str,
    source_stage: str,
    raw_tasks: object,
    trace_summary: dict[str, object],
    rejection_target: ComplexTaskNodeV1 | dict[str, object],
    allow_creation: bool,
    rejection_reason: str | None,
) -> int:
    """Validate follow-up tasks and append bounded pending graph nodes."""

    if not isinstance(raw_tasks, list):
        _record_followup_rejection(
            trace_summary=trace_summary,
            source_key=source_key,
            source_stage=source_stage,
            parent_node_id=parent_node_id,
            objective=_raw_followup_objective(raw_tasks),
            kind=_raw_followup_kind(raw_tasks),
            reason="followup_tasks: expected list",
            rejection_target=rejection_target,
        )
        return 0
    created_count = 0
    for raw_task in raw_tasks:
        try:
            task = validate_complex_task_followup_task(raw_task)
        except ComplexTaskValidationError as exc:
            _record_followup_rejection(
                trace_summary=trace_summary,
                source_key=source_key,
                source_stage=source_stage,
                parent_node_id=parent_node_id,
                objective=_raw_followup_objective(raw_task),
                kind=_raw_followup_kind(raw_task),
                reason=f"followup task invalid: {exc}",
                rejection_target=rejection_target,
            )
            continue
        if not allow_creation:
            reason = rejection_reason
            if reason is None:
                reason = "follow-up creation is not allowed in this pass"
            _record_followup_rejection(
                trace_summary=trace_summary,
                source_key=source_key,
                source_stage=source_stage,
                parent_node_id=parent_node_id,
                objective=task["objective"],
                kind=task["kind"],
                reason=reason,
                rejection_target=rejection_target,
            )
            continue
        source_created_count = _followup_created_count(
            trace_summary,
            source_key,
        )
        if source_created_count >= _FOLLOWUP_TASKS_PER_SOURCE_LIMIT:
            _record_followup_rejection(
                trace_summary=trace_summary,
                source_key=source_key,
                source_stage=source_stage,
                parent_node_id=parent_node_id,
                objective=task["objective"],
                kind=task["kind"],
                reason="followup_tasks: exceeds per-source cap",
                rejection_target=rejection_target,
            )
            continue
        child_depth = graph["nodes"][parent_node_id]["depth"] + 1
        if child_depth > graph["max_depth"]:
            _record_followup_rejection(
                trace_summary=trace_summary,
                source_key=source_key,
                source_stage=source_stage,
                parent_node_id=parent_node_id,
                objective=task["objective"],
                kind=task["kind"],
                reason="followup_tasks: exceeds max_depth",
                rejection_target=rejection_target,
            )
            continue
        if len(graph["nodes"]) + 1 > graph["max_nodes"]:
            _record_followup_rejection(
                trace_summary=trace_summary,
                source_key=source_key,
                source_stage=source_stage,
                parent_node_id=parent_node_id,
                objective=task["objective"],
                kind=task["kind"],
                reason="followup_tasks: exceeds max_nodes",
                rejection_target=rejection_target,
            )
            continue
        child_id = _next_child_node_id(graph, parent_node_id)
        graph["nodes"][child_id] = _make_followup_node(
            node_id=child_id,
            parent_id=parent_node_id,
            depth=child_depth,
            task=task,
        )
        graph["nodes"][parent_node_id]["children"].append(child_id)
        created_count += 1
        _record_followup_creation(
            trace_summary=trace_summary,
            source_key=source_key,
            source_stage=source_stage,
            parent_node_id=parent_node_id,
            child_node_id=child_id,
            task=task,
        )
    return created_count


def _make_followup_node(
    *,
    node_id: str,
    parent_id: str,
    depth: int,
    task: ComplexTaskFollowupTaskV1,
) -> ComplexTaskNodeV1:
    """Create one pending graph node from a validated follow-up task."""

    node = _make_graph_node(
        node_id=node_id,
        parent_id=parent_id,
        depth=depth,
        objective=task["objective"],
        node_kind=task["kind"],
        status="pending",
        children=[],
    )
    _set_node_semantic_projection(
        node,
        investigation_summary=f"Follow-up task requested: {task['reason']}",
        knowledge_we_know_so_far=[],
        knowledge_still_lacking=[],
        recommended_next_iteration=[],
        evidence_boundary_notes=[],
    )
    return node


def _followup_created_count(
    trace_summary: dict[str, object],
    source_key: str,
) -> int:
    """Return how many follow-up nodes this source has already created."""

    event_log = trace_summary["followup_event_log"]
    if not isinstance(event_log, list):
        raise ComplexTaskValidationError(
            "trace_summary.followup_event_log: expected list"
        )
    created_count = 0
    for event in event_log:
        if not isinstance(event, dict):
            continue
        if event.get("event") != "created":
            continue
        if event.get("source_key") == source_key:
            created_count += 1
    return created_count


def _record_followup_creation(
    *,
    trace_summary: dict[str, object],
    source_key: str,
    source_stage: str,
    parent_node_id: str,
    child_node_id: str,
    task: ComplexTaskFollowupTaskV1,
) -> None:
    """Record a compact read-only trace entry for follow-up creation."""

    trace_summary["followup_created_count"] = (
        int(trace_summary["followup_created_count"]) + 1
    )
    event_log = trace_summary["followup_event_log"]
    if not isinstance(event_log, list):
        raise ComplexTaskValidationError(
            "trace_summary.followup_event_log: expected list"
        )
    event_log.append({
        "event": "created",
        "source_key": source_key,
        "source_stage": source_stage,
        "parent_node_id": parent_node_id,
        "child_node_id": child_node_id,
        "objective": task["objective"],
        "kind": task["kind"],
        "reason": task["reason"],
    })


def _record_followup_rejection(
    *,
    trace_summary: dict[str, object],
    source_key: str,
    source_stage: str,
    parent_node_id: str,
    objective: str,
    kind: str,
    reason: str,
    rejection_target: ComplexTaskNodeV1 | dict[str, object],
) -> None:
    """Record a compact trace entry and semantic gap for a rejected follow-up."""

    trace_summary["followup_rejected_count"] = (
        int(trace_summary["followup_rejected_count"]) + 1
    )
    event_log = trace_summary["followup_event_log"]
    if not isinstance(event_log, list):
        raise ComplexTaskValidationError(
            "trace_summary.followup_event_log: expected list"
        )
    safe_objective = objective
    if not safe_objective:
        safe_objective = "unreadable follow-up task"
    event_log.append({
        "event": "rejected",
        "source_key": source_key,
        "source_stage": source_stage,
        "parent_node_id": parent_node_id,
        "objective": safe_objective,
        "kind": kind,
        "reason": reason,
    })
    _append_followup_rejection(
        rejection_target,
        objective=safe_objective,
        reason=reason,
    )


def _append_followup_rejection(
    target: ComplexTaskNodeV1 | dict[str, object],
    *,
    objective: str,
    reason: str,
) -> None:
    """Preserve a rejected follow-up as semantic lacking knowledge."""

    lacking = _mutable_semantic_rows(target, "knowledge_still_lacking")
    boundary_notes = _mutable_semantic_rows(target, "evidence_boundary_notes")
    _append_missing_items(
        lacking,
        [f"follow-up task not created: {objective}"],
    )
    _append_missing_items(
        boundary_notes,
        [f"Resolver follow-up task rejected: {reason}"],
    )


def _mutable_semantic_rows(
    target: dict[str, object],
    field_name: str,
) -> list[str]:
    """Return a mutable semantic list field on a node or synthesis response."""

    rows = target.get(field_name)
    if not isinstance(rows, list):
        rows = []
        target[field_name] = rows
    semantic_rows: list[str] = []
    for row in rows:
        if isinstance(row, str):
            semantic_rows.append(row)
    target[field_name] = semantic_rows
    return semantic_rows


def _raw_followup_objective(raw_task: object) -> str:
    """Extract a compact objective from an invalid follow-up row."""

    if isinstance(raw_task, dict):
        objective = raw_task.get("objective")
        if isinstance(objective, str) and objective.strip():
            return objective.strip()
    return ""


def _raw_followup_kind(raw_task: object) -> str:
    """Extract a compact task kind from an invalid follow-up row."""

    if isinstance(raw_task, dict):
        task_kind = raw_task.get("kind")
        if isinstance(task_kind, str) and task_kind.strip():
            return task_kind.strip()
    return ""


def _block_node_expansion_rejected(
    node: ComplexTaskNodeV1,
    *,
    reason: str,
) -> None:
    """Block a node when the requested expansion violates graph limits."""

    node["status"] = "blocked"
    _set_node_semantic_projection(
        node,
        investigation_summary=(
            "The node requested further decomposition, but the resolver "
            "rejected that expansion at the configured graph boundary."
        ),
        knowledge_we_know_so_far=_node_known_so_far(node),
        knowledge_still_lacking=[reason],
        recommended_next_iteration=[
            "Resolve this node within the current graph depth, synthesize from "
            "available sibling knowledge, or narrow this branch before retry."
        ],
        evidence_boundary_notes=[
            "Resolver graph limits rejected the requested node_expansion."
        ],
    )


def _next_child_node_id(
    graph: ComplexTaskGraphV1,
    parent_node_id: str,
) -> str:
    """Allocate a stable child id below the active parent node."""

    index = 1
    while True:
        node_id = f"{parent_node_id}_{index}"
        if node_id not in graph["nodes"]:
            return_value = node_id
            return return_value
        index += 1


def _block_algorithmic_node_without_subagent(node: ComplexTaskNodeV1) -> None:
    """Fail closed when deterministic arithmetic is answered by prose."""

    node["status"] = "blocked"
    _set_node_semantic_projection(
        node,
        investigation_summary=(
            "The algorithmic node was blocked because it did not return a "
            "deterministic subagent result."
        ),
        knowledge_we_know_so_far=_node_known_so_far(node),
        knowledge_still_lacking=[
            "structured deterministic calculation result for this node"
        ],
        recommended_next_iteration=[
            "Prepare a typed evaluate_expression request with normalized "
            "numeric operands."
        ],
        evidence_boundary_notes=[
            "Algorithmic task nodes require the resolver-local algorithmic "
            "subagent."
        ],
    )


def _block_evidence_node_without_subagent(node: ComplexTaskNodeV1) -> None:
    """Fail closed when external evidence is answered by prose."""

    node["status"] = "blocked"
    _set_node_semantic_projection(
        node,
        investigation_summary=(
            "The evidence node was blocked because it did not return an "
            "evidence subagent result."
        ),
        knowledge_we_know_so_far=_node_known_so_far(node),
        knowledge_still_lacking=[
            "structured evidence result for this node"
        ],
        recommended_next_iteration=[
            "Issue a resolver-local evidence subagent request with a narrower "
            "public evidence objective."
        ],
        evidence_boundary_notes=[
            "Evidence-need nodes require the resolver-local evidence subagent."
        ],
    )


def _synthesis_dependency_blockers(
    graph: ComplexTaskGraphV1,
    active_node_id: str,
) -> list[str]:
    """Return unresolved prerequisite branches before a synthesis node."""

    active_node = graph["nodes"][active_node_id]
    parent_id = active_node["parent_id"]
    if parent_id is None:
        return []
    parent = graph["nodes"][parent_id]
    blockers: list[str] = []
    for sibling_id in parent["children"]:
        if sibling_id == active_node_id:
            break
        sibling_blockers = _subtree_unresolved_objectives(graph, sibling_id)
        blockers.extend(sibling_blockers)
    return blockers


def _subtree_unresolved_objectives(
    graph: ComplexTaskGraphV1,
    node_id: str,
) -> list[str]:
    """Collect unresolved objectives from a graph subtree."""

    node = graph["nodes"][node_id]
    blockers: list[str] = []
    if node["status"] in ("pending", "blocked", "cannot_answer"):
        blockers.append(node["objective"])
    for child_id in node["children"]:
        child_blockers = _subtree_unresolved_objectives(graph, child_id)
        blockers.extend(child_blockers)
    return blockers


def _block_synthesis_node_with_dependencies(
    node: ComplexTaskNodeV1,
    blockers: list[str],
) -> None:
    """Prevent synthesis from completing before prerequisites resolve."""

    node["status"] = "blocked"
    reason = "synthesis requires resolved prerequisite nodes: " + "; ".join(blockers)
    _set_node_semantic_projection(
        node,
        investigation_summary="The synthesis node is blocked by prerequisites.",
        knowledge_we_know_so_far=_node_known_so_far(node),
        knowledge_still_lacking=[reason],
        recommended_next_iteration=[
            "Resolve or explicitly bound the listed prerequisite branches "
            "before synthesizing this node."
        ],
        evidence_boundary_notes=[
            "Synthesis could not proceed until prerequisite branches produce "
            "usable semantic knowledge."
        ],
    )


def _apply_subagent_result(
    node: ComplexTaskNodeV1,
    subagent_result: ComplexTaskSubagentResultV1,
) -> None:
    """Translate a subagent envelope into a graph-node result."""

    if subagent_result["resolved"]:
        node["status"] = "resolved"
        summary = _summarize_result(subagent_result["result"])
        known_rows = _subagent_resolved_known_rows(node, subagent_result)
        lacking_rows = _subagent_result_semantic_rows(
            subagent_result,
            "knowledge_still_lacking",
        )
        next_rows = _subagent_result_semantic_rows(
            subagent_result,
            "recommended_next_iteration",
        )
        boundary_notes = _subagent_boundary_notes(subagent_result)
        _append_missing_items(
            boundary_notes,
            _subagent_result_semantic_rows(
                subagent_result,
                "evidence_boundary_notes",
            ),
        )
        if node["node_kind"] == "evidence_need":
            _append_missing_items(
                boundary_notes,
                [
                    (
                        "Evidence subagent returned prose summary; preserve "
                        "embedded caveats instead of treating every sentence "
                        "as confirmed knowledge."
                    ),
                ],
            )
        _set_node_semantic_projection(
            node,
            investigation_summary=summary,
            knowledge_we_know_so_far=known_rows,
            knowledge_still_lacking=lacking_rows,
            recommended_next_iteration=next_rows,
            evidence_boundary_notes=boundary_notes,
        )
        return
    node["status"] = "blocked"
    _set_node_semantic_projection(
        node,
        investigation_summary="The subagent did not complete this node.",
        knowledge_we_know_so_far=_subagent_known_so_far(subagent_result),
        knowledge_still_lacking=_subagent_lacking_items(subagent_result),
        recommended_next_iteration=[
            "Retry this node only with a narrower subagent request or a "
            "different evidence direction."
        ],
        evidence_boundary_notes=_subagent_boundary_notes(subagent_result),
    )


def _merge_node_update(node: ComplexTaskNodeV1, update: object) -> None:
    """Merge whitelisted node fields from an LLM-owned update."""

    if not isinstance(update, dict):
        raise ComplexTaskValidationError("node_update: expected object")
    supported_field_seen = False
    for field_name, value in update.items():
        if field_name not in _NODE_UPDATE_FIELDS:
            continue
        supported_field_seen = True
        node[field_name] = value
    if not supported_field_seen:
        _block_node_with_invalid_update(node)
        return
    _normalize_node_projection(node)


def _normalize_node_projection(node: ComplexTaskNodeV1) -> None:
    """Normalize node semantic rows without judging their correctness."""

    _set_node_semantic_projection(
        node,
        investigation_summary=node["investigation_summary"],
        knowledge_we_know_so_far=node["knowledge_we_know_so_far"],
        knowledge_still_lacking=node["knowledge_still_lacking"],
        recommended_next_iteration=node["recommended_next_iteration"],
        evidence_boundary_notes=node["evidence_boundary_notes"],
    )


def _block_node_with_invalid_update(node: ComplexTaskNodeV1) -> None:
    """Block an active node when the LLM update has no usable graph fields."""

    node["status"] = "blocked"
    _set_node_semantic_projection(
        node,
        investigation_summary="The node update did not contain supported fields.",
        knowledge_we_know_so_far=_node_known_so_far(node),
        knowledge_still_lacking=[
            "node-level semantic projection fields"
        ],
        recommended_next_iteration=[
            "Return a node_update with the stable semantic projection fields."
        ],
        evidence_boundary_notes=[
            "The resolver accepted no unsupported node_update fields."
        ],
    )


async def _review_collapse(
    request: ComplexTaskResolverRequestV1,
    context: ComplexTaskResolverContextV1,
    options: ComplexTaskResolverOptionsV1,
    graph: ComplexTaskGraphV1,
    active_node_id: str,
) -> dict[str, object]:
    """Ask the injected collapse reviewer about bounded graph candidates."""

    candidates = _collapse_candidates(graph, active_node_id)
    if not candidates:
        response = {
            "collapse_decision": {
                "should_collapse": False,
                "matching_candidate": "",
                "reason": "no bounded candidates",
            },
        }
        return response
    payload = {
        "stage": "collapse_review",
        "root_question": request["objective"],
        "active_node": _compact_node(graph["nodes"][active_node_id]),
        "candidates": candidates,
        "context": _compact_context(context),
    }
    del options

    response = await _collapse_stage_handler(payload)
    return response


def _apply_collapse_response(
    graph: ComplexTaskGraphV1,
    active_node_id: str,
    response: dict[str, object],
    trace_summary: dict[str, object],
) -> None:
    """Apply validated bounded collapse updates to the graph."""

    if "collapse_decision" in response:
        if _looks_like_internal_collapse_response(response):
            if _using_production_collapse_stage_handler():
                raise ComplexTaskValidationError(
                    "collapse semantic output must not use graph targets"
                )
        else:
            _reject_forbidden_semantic_output_keys(response, "collapse")
        _apply_semantic_collapse_decision(
            graph,
            active_node_id,
            response["collapse_decision"],
            trace_summary,
        )
        return
    raise ComplexTaskValidationError("collapse response: missing collapse_decision")


def _looks_like_internal_collapse_response(response: dict[str, object]) -> bool:
    """Return whether a deterministic test supplied internal collapse IO."""

    decision = response.get("collapse_decision")
    return_value = isinstance(decision, dict) and "target_node_id" in decision
    return return_value


def _apply_semantic_collapse_decision(
    graph: ComplexTaskGraphV1,
    active_node_id: str,
    decision: object,
    trace_summary: dict[str, object],
) -> None:
    """Build graph collapse state from a bounded semantic decision."""

    if not isinstance(decision, dict):
        raise ComplexTaskValidationError("collapse_decision: expected object")
    should_collapse = decision["should_collapse"]
    if not isinstance(should_collapse, bool):
        raise ComplexTaskValidationError(
            "collapse_decision.should_collapse: expected boolean"
        )
    if not should_collapse:
        return
    target_node_id = _collapse_target_node_id(
        graph=graph,
        active_node_id=active_node_id,
        decision=decision,
    )
    if target_node_id is None:
        raise ComplexTaskValidationError(
            "collapse_decision.matching_candidate: expected existing candidate"
        )
    if target_node_id == active_node_id:
        raise ComplexTaskValidationError(
            "collapse_decision.matching_candidate: cannot target active node"
        )
    active_node = graph["nodes"][active_node_id]
    target_node = graph["nodes"][target_node_id]
    if target_node["node_kind"] != active_node["node_kind"]:
        raise ComplexTaskValidationError(
            "collapse_decision.target_node_id: expected same node kind"
        )
    reason = decision["reason"]
    if not isinstance(reason, str) or not reason.strip():
        raise ComplexTaskValidationError(
            "collapse_decision.reason: expected non-empty string"
        )
    active_node["status"] = "collapsed"
    active_node["collapsed_into"] = target_node_id
    _set_node_semantic_projection(
        active_node,
        investigation_summary=f"Collapsed into {target_node_id}: {reason.strip()}",
        knowledge_we_know_so_far=list(target_node["knowledge_we_know_so_far"]),
        knowledge_still_lacking=list(target_node["knowledge_still_lacking"]),
        recommended_next_iteration=list(
            target_node["recommended_next_iteration"]
        ),
        evidence_boundary_notes=list(target_node["evidence_boundary_notes"]),
    )
    graph["collapse_events"].append({
        "from_node_id": active_node_id,
        "to_node_id": target_node_id,
        "reason": reason.strip(),
    })
    trace_summary["collapse_count"] = int(trace_summary["collapse_count"]) + 1


def _collapse_target_node_id(
    *,
    graph: ComplexTaskGraphV1,
    active_node_id: str,
    decision: dict[str, object],
) -> str | None:
    """Map semantic collapse text to an existing graph candidate."""

    target_node_id = decision.get("target_node_id")
    if isinstance(target_node_id, str) and target_node_id in graph["nodes"]:
        return target_node_id
    matching_candidate = decision.get("matching_candidate")
    if not isinstance(matching_candidate, str) or not matching_candidate.strip():
        return None
    active_node = graph["nodes"][active_node_id]
    for node_id, node in graph["nodes"].items():
        if node_id == active_node_id:
            continue
        if node["node_kind"] != active_node["node_kind"]:
            continue
        if node["status"] != "resolved":
            continue
        candidate_text = _collapse_candidate_text(node)
        if _contains_structural_text(candidate_text, matching_candidate):
            return node_id
    return None


def _collapse_candidate_text(node: ComplexTaskNodeV1) -> str:
    """Return semantic text that can identify a collapse candidate."""

    rows = [
        node["objective"],
        node["investigation_summary"],
        *node["knowledge_we_know_so_far"],
        *node["knowledge_still_lacking"],
        *node["evidence_boundary_notes"],
    ]
    return "\n".join(row for row in rows if row)


async def _synthesize_packet(
    request: ComplexTaskResolverRequestV1,
    context: ComplexTaskResolverContextV1,
    options: ComplexTaskResolverOptionsV1,
    graph: ComplexTaskGraphV1,
    trace_summary: dict[str, object],
) -> ComplexTaskResolutionPacketV1:
    """Call the injected synthesizer and assemble a validated packet."""

    response, unresolved_nodes, node_boundary_notes = await _call_synthesis_stage(
        request=request,
        context=context,
        graph=graph,
        trace_summary=trace_summary,
    )
    max_nodes = _option_limit(options, "max_nodes", 8)
    max_iterations = _option_limit(options, "max_iterations", max_nodes)
    remaining_iterations = max_iterations - int(trace_summary["iterations"])
    created_count = _apply_synthesis_followup_tasks(
        graph=graph,
        response=response,
        trace_summary=trace_summary,
        allow_creation=remaining_iterations > 0,
        rejection_reason="synthesis follow-up rejected at max_iterations",
    )
    if created_count > 0:
        graph = validate_complex_task_graph(graph)
        remaining_iterations = max_iterations - int(trace_summary["iterations"])
        await _run_graph_traversal(
            validated_request=request,
            validated_context=context,
            validated_options=options,
            graph=graph,
            trace_summary=trace_summary,
            max_iterations=remaining_iterations,
        )
        response, unresolved_nodes, node_boundary_notes = await _call_synthesis_stage(
            request=request,
            context=context,
            graph=graph,
            trace_summary=trace_summary,
        )
        _apply_synthesis_followup_tasks(
            graph=graph,
            response=response,
            trace_summary=trace_summary,
            allow_creation=False,
            rejection_reason="synthesis follow-up pass already used",
        )
    packet = _packet_from_synthesis_response(
        request=request,
        graph=graph,
        trace_summary=trace_summary,
        response=response,
        unresolved_nodes=unresolved_nodes,
        node_boundary_notes=node_boundary_notes,
    )
    return packet


async def _call_synthesis_stage(
    *,
    request: ComplexTaskResolverRequestV1,
    context: ComplexTaskResolverContextV1,
    graph: ComplexTaskGraphV1,
    trace_summary: dict[str, object],
) -> tuple[dict[str, object], list[dict[str, object]], list[str]]:
    """Call bottom-up synthesis and return parsed semantic source material."""

    unresolved_nodes = _unresolved_node_summaries(graph)
    node_boundary_notes = _node_evidence_boundary_notes(graph)
    payload = {
        "stage": "bottom_up_synthesis",
        "root_question": request["objective"],
        "resolved_nodes": _resolved_node_summaries(graph),
        "unresolved_nodes": unresolved_nodes,
        "node_evidence_boundary_notes": node_boundary_notes,
        "context": _compact_context(context),
    }

    raw_response = await _synthesizer_stage_handler(payload)
    _record_stage_io(
        trace_summary=trace_summary,
        stage="bottom_up_synthesis",
        prompt_payload=payload,
        parsed_output=raw_response,
    )
    response = _normalize_synthesis_stage_response(
        raw_response,
        allow_internal_envelope=not _using_production_synthesizer_stage_handler(),
    )
    result = (response, unresolved_nodes, node_boundary_notes)
    return result


def _normalize_synthesis_stage_response(
    response: dict[str, object],
    *,
    allow_internal_envelope: bool,
) -> dict[str, object]:
    """Map semantic synthesis continuation tasks into internal tasks."""

    if "followup_tasks" in response:
        if not allow_internal_envelope:
            raise ComplexTaskValidationError(
                "synthesis semantic output must not use internal follow-up tasks"
            )
        return response
    _reject_forbidden_semantic_output_keys(response, "synthesis")
    followups = _semantic_continuation_tasks(response.get("continuation_tasks"))
    if not followups:
        return response
    normalized = dict(response)
    normalized["followup_tasks"] = followups
    return normalized


def _packet_from_synthesis_response(
    *,
    request: ComplexTaskResolverRequestV1,
    graph: ComplexTaskGraphV1,
    trace_summary: dict[str, object],
    response: dict[str, object],
    unresolved_nodes: list[dict[str, object]],
    node_boundary_notes: list[str],
) -> ComplexTaskResolutionPacketV1:
    """Build the final public packet from semantic synthesis fields."""

    synthesis = _semantic_synthesis_response(
        response=response,
        unresolved_nodes=unresolved_nodes,
        evidence_boundary_notes=node_boundary_notes,
    )
    _append_missing_items(
        synthesis["knowledge_still_lacking"],
        _followup_rejection_lacking_rows(trace_summary),
    )
    _append_missing_items(
        synthesis["knowledge_we_know_so_far"],
        _resolved_node_known_rows(graph),
    )
    _append_missing_items(
        synthesis["evidence_boundary_notes"],
        _followup_rejection_boundary_notes(trace_summary),
    )
    packet = {
        "schema_version": COMPLEX_TASK_RESOLUTION_PACKET_VERSION,
        "root_question": request["objective"],
        "investigation_summary": synthesis["investigation_summary"],
        "knowledge_we_know_so_far": synthesis["knowledge_we_know_so_far"],
        "knowledge_still_lacking": synthesis["knowledge_still_lacking"],
        "recommended_next_iteration": synthesis["recommended_next_iteration"],
        "evidence_boundary_notes": _packet_evidence_boundary_notes(
            synthesis["evidence_boundary_notes"],
            node_boundary_notes,
        ),
        "graph": graph,
        "trace_summary": trace_summary,
    }
    validated_packet = validate_complex_task_resolution_packet(packet)
    return validated_packet


def _apply_synthesis_followup_tasks(
    *,
    graph: ComplexTaskGraphV1,
    response: dict[str, object],
    trace_summary: dict[str, object],
    allow_creation: bool,
    rejection_reason: str,
) -> int:
    """Create bounded root-level follow-up nodes from synthesis output."""

    if "followup_tasks" not in response:
        return 0
    root_node_id = graph["root_node_id"]
    created_count = _create_followup_nodes(
        graph=graph,
        parent_node_id=root_node_id,
        source_key="bottom_up_synthesis",
        source_stage="bottom_up_synthesis",
        raw_tasks=response["followup_tasks"],
        trace_summary=trace_summary,
        rejection_target=response,
        allow_creation=allow_creation,
        rejection_reason=rejection_reason,
    )
    if created_count > 0:
        graph["nodes"][root_node_id]["status"] = "expanded"
    return created_count


def _followup_rejection_lacking_rows(
    trace_summary: dict[str, object],
) -> list[str]:
    """Project rejected follow-up objectives into final semantic gaps."""

    rows: list[str] = []
    for event in _followup_events(trace_summary):
        if event.get("event") != "rejected":
            continue
        objective = event.get("objective")
        if not isinstance(objective, str) or not objective.strip():
            objective = "unreadable follow-up task"
        rows.append(f"follow-up task not created: {objective}")
    return rows


def _resolved_node_known_rows(graph: ComplexTaskGraphV1) -> list[str]:
    """Collect resolved node semantic knowledge for packet preservation."""

    rows: list[str] = []
    for node in graph["nodes"].values():
        if node["status"] not in ("resolved", "collapsed"):
            continue
        _append_missing_items(rows, node["knowledge_we_know_so_far"])
    return rows


def _followup_rejection_boundary_notes(
    trace_summary: dict[str, object],
) -> list[str]:
    """Project rejected follow-up reasons into final boundary notes."""

    rows: list[str] = []
    for event in _followup_events(trace_summary):
        if event.get("event") != "rejected":
            continue
        reason = event.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            reason = "unknown follow-up rejection"
        rows.append(f"Resolver follow-up task rejected: {reason}")
    return rows


def _followup_events(trace_summary: dict[str, object]) -> list[dict[str, object]]:
    """Return compact follow-up trace events with invalid rows filtered out."""

    event_log = trace_summary["followup_event_log"]
    if not isinstance(event_log, list):
        raise ComplexTaskValidationError(
            "trace_summary.followup_event_log: expected list"
        )
    events: list[dict[str, object]] = []
    for event in event_log:
        if isinstance(event, dict):
            events.append(event)
    return events


def _semantic_synthesis_response(
    *,
    response: dict[str, object],
    unresolved_nodes: list[dict[str, object]],
    evidence_boundary_notes: list[str],
) -> dict[str, object]:
    """Normalize synthesis output into semantic investigation sections."""

    summary = _semantic_text(response.get("investigation_summary"))
    if not summary:
        summary = "The investigation produced semantic knowledge sections."
    recommended_next_iteration = _semantic_text_list(
        response.get("recommended_next_iteration")
    )
    knowledge_still_lacking = _semantic_text_list(
        response.get("knowledge_still_lacking")
    )
    _append_unresolved_node_knowledge(knowledge_still_lacking, unresolved_nodes)
    if not recommended_next_iteration and knowledge_still_lacking:
        recommended_next_iteration.append(
            "Cognition can use the listed knowledge gaps to decide whether a "
            "narrower evidence request, user clarification, or bounded answer "
            "is appropriate."
        )
    synthesis = {
        "investigation_summary": summary,
        "knowledge_we_know_so_far": _semantic_text_list(
            response.get("knowledge_we_know_so_far")
        ),
        "knowledge_still_lacking": knowledge_still_lacking,
        "recommended_next_iteration": recommended_next_iteration,
        "evidence_boundary_notes": _semantic_text_list(
            response.get("evidence_boundary_notes")
        ),
    }
    _append_missing_items(
        synthesis["evidence_boundary_notes"],
        evidence_boundary_notes,
    )
    return synthesis


def _packet_evidence_boundary_notes(
    llm_notes: list[str],
    deterministic_notes: list[str],
) -> list[str]:
    """Merge LLM-noted and deterministic evidence boundaries."""

    notes = list(llm_notes)
    _append_missing_items(notes, deterministic_notes)
    return notes


def _semantic_text(value: object) -> str:
    """Return a compact semantic text field from LLM output."""

    if isinstance(value, str):
        text = value.strip()
        return text
    return ""


def _semantic_text_list(value: object) -> list[str]:
    """Return non-empty semantic text rows from LLM output."""

    if not isinstance(value, list):
        return []
    rows: list[str] = []
    for item in value:
        if not isinstance(item, str):
            continue
        text = item.strip()
        if text:
            rows.append(text)
    return rows


def _set_node_semantic_projection(
    node: ComplexTaskNodeV1,
    *,
    investigation_summary: str,
    knowledge_we_know_so_far: list[str],
    knowledge_still_lacking: list[str],
    recommended_next_iteration: list[str],
    evidence_boundary_notes: list[str],
) -> None:
    """Set the node-local semantic packet fields after structural cleanup."""

    if not isinstance(investigation_summary, str):
        raise ComplexTaskValidationError(
            "node investigation_summary: expected string"
        )
    node["investigation_summary"] = investigation_summary.strip()
    node["knowledge_we_know_so_far"] = _clean_semantic_rows(
        knowledge_we_know_so_far
    )
    node["knowledge_still_lacking"] = _clean_semantic_rows(
        knowledge_still_lacking
    )
    node["recommended_next_iteration"] = _clean_semantic_rows(
        recommended_next_iteration
    )
    node["evidence_boundary_notes"] = _clean_semantic_rows(
        evidence_boundary_notes
    )


def _clean_semantic_rows(values: object) -> list[str]:
    """Return non-empty semantic rows while preserving first-seen order."""

    if not isinstance(values, list):
        raise ComplexTaskValidationError("node semantic projection: expected list")
    rows: list[str] = []
    for value in values:
        if not isinstance(value, str):
            continue
        row = value.strip()
        if not row or row in rows:
            continue
        rows.append(row)
    return rows


def _node_known_so_far(node: ComplexTaskNodeV1) -> list[str]:
    """Return node knowledge that can survive a later blocked projection."""

    rows = list(node["knowledge_we_know_so_far"])
    if node["investigation_summary"]:
        _append_missing_items(rows, [node["investigation_summary"]])
    return rows


def _subagent_known_so_far(
    subagent_result: ComplexTaskSubagentResultV1,
) -> list[str]:
    """Project any structured subagent result text into node knowledge."""

    summary = _summarize_result(subagent_result["result"])
    if summary == "subagent result resolved":
        return []
    rows = [summary]
    return rows


def _subagent_resolved_known_rows(
    node: ComplexTaskNodeV1,
    subagent_result: ComplexTaskSubagentResultV1,
) -> list[str]:
    """Return resolved subagent knowledge rows for bottom-up synthesis."""

    structured_rows = _subagent_result_semantic_rows(
        subagent_result,
        "knowledge_we_know_so_far",
    )
    if structured_rows:
        return structured_rows
    if node["node_kind"] in ("algorithmic_task", "evidence_need"):
        known_rows = _subagent_known_so_far(subagent_result)
        return known_rows
    return []


def _subagent_result_semantic_rows(
    subagent_result: ComplexTaskSubagentResultV1,
    field_name: str,
) -> list[str]:
    """Read optional structured semantic rows from a subagent result dict."""

    result = subagent_result["result"]
    rows = result.get(field_name)
    if not isinstance(rows, list):
        return []
    semantic_rows = _clean_semantic_rows(rows)
    return semantic_rows


def _subagent_lacking_items(
    subagent_result: ComplexTaskSubagentResultV1,
) -> list[str]:
    """Project subagent unresolved items into node semantic gaps."""

    lacking = list(subagent_result["unresolved_items"])
    if not lacking:
        lacking.append("structured subagent output or evidence for this node")
    return lacking


def _subagent_boundary_notes(
    subagent_result: ComplexTaskSubagentResultV1,
) -> list[str]:
    """Project structural subagent provenance into semantic boundary notes."""

    if subagent_result["resolved"]:
        notes = ["Resolver-local subagent produced bounded output."]
        return notes
    notes = ["Resolver-local subagent could not complete this node."]
    return notes


def _append_unresolved_node_knowledge(
    rows: list[str],
    unresolved_nodes: list[dict[str, object]],
) -> None:
    """Add unresolved graph nodes as semantic knowledge gaps."""

    node_items: list[str] = []
    for node in unresolved_nodes:
        objective = node["objective"]
        node_items.append(f"unresolved branch: {objective}")
    _append_missing_items(rows, node_items)


def _append_missing_items(rows: list[str], new_items: list[str]) -> None:
    """Append non-empty items that are not already present."""

    existing_items = set(rows)
    for item in new_items:
        if not item:
            continue
        if item in existing_items:
            continue
        rows.append(item)
        existing_items.add(item)


def _node_evidence_boundary_notes(graph: ComplexTaskGraphV1) -> list[str]:
    """Collect node-authored evidence boundary notes for synthesis."""

    notes: list[str] = []
    for node in graph["nodes"].values():
        _append_missing_items(notes, node["evidence_boundary_notes"])
    return notes


def _option_limit(
    options: ComplexTaskResolverOptionsV1,
    field_name: str,
    default_value: int,
) -> int:
    """Read a positive integer structural option limit."""

    limits = options["limits"]
    if field_name not in limits:
        return default_value
    value = limits[field_name]
    if not isinstance(value, int) or value < 1:
        raise ComplexTaskValidationError(f"limits.{field_name}: expected positive int")
    return value


def _compact_context(context: ComplexTaskResolverContextV1) -> dict[str, object]:
    """Return compact context safe for stage prompts and review artifacts."""

    compact = {
        "conversation_summary": context["conversation_summary"],
        "persona_context_summary": context["persona_context_summary"],
        "time_context": context["time_context"],
        "available_evidence": _compact_available_evidence(
            context["available_evidence"]
        ),
    }
    return compact


def _compact_available_evidence(values: list[object]) -> list[dict[str, object]]:
    """Project caller evidence into semantic prompt-facing rows."""

    compact_rows: list[dict[str, object]] = []
    for value in values:
        if not isinstance(value, dict):
            continue
        row: dict[str, object] = {}
        evidence_kind = value.get("evidence_kind")
        if isinstance(evidence_kind, str) and evidence_kind.strip():
            row["kind"] = evidence_kind.strip()
        evidence_id = value.get("evidence_id")
        if isinstance(evidence_id, str) and evidence_id.strip():
            row["source"] = evidence_id.strip()
        excerpt = value.get("excerpt")
        if isinstance(excerpt, str) and excerpt.strip():
            row["excerpt"] = excerpt.strip()
        observed_at = value.get("observed_at")
        if isinstance(observed_at, str) and observed_at.strip():
            row["observed_at"] = observed_at.strip()
        if row:
            compact_rows.append(row)
    return compact_rows


def _compact_node(node: ComplexTaskNodeV1) -> dict[str, object]:
    """Return node fields needed by semantic review stages."""

    compact = {
        "objective": node["objective"],
        "work_type": _semantic_work_type_from_node_kind(node["node_kind"]),
        "investigation_summary": node["investigation_summary"],
        "knowledge_we_know_so_far": node["knowledge_we_know_so_far"],
        "knowledge_still_lacking": node["knowledge_still_lacking"],
        "recommended_next_iteration": node["recommended_next_iteration"],
        "evidence_boundary_notes": node["evidence_boundary_notes"],
        "recent_attempts": _compact_node_attempts(node),
    }
    return compact


def _semantic_work_type_from_node_kind(node_kind: str) -> str:
    """Return prompt-facing work-type language for one graph node kind."""

    if node_kind == "evidence_need":
        return "public_evidence"
    if node_kind == "algorithmic_task":
        return "calculation"
    return_value = node_kind
    return return_value


def _compact_node_attempts(node: ComplexTaskNodeV1) -> list[dict[str, object]]:
    """Return bounded prompt-facing attempt observations for one node."""

    attempts: list[dict[str, object]] = []
    for attempt in node["attempts"][-3:]:
        attempts.append({
            "action": attempt["action"],
            "result_summary": attempt["result_summary"],
            "blockers": list(attempt["blockers"]),
            "next_action": attempt["next_action"],
        })
    return attempts


def _parent_chain_summary(
    graph: ComplexTaskGraphV1,
    node: ComplexTaskNodeV1,
) -> str:
    """Summarize the active node's parent chain."""

    summaries: list[str] = []
    parent_id = node["parent_id"]
    while parent_id is not None:
        parent = graph["nodes"][parent_id]
        summaries.append(parent["objective"])
        parent_id = parent["parent_id"]
    summary = " > ".join(reversed(summaries))
    return summary


def _sibling_summaries(
    graph: ComplexTaskGraphV1,
    node: ComplexTaskNodeV1,
) -> list[str]:
    """Collect concise resolved sibling summaries."""

    parent_id = node["parent_id"]
    if parent_id is None:
        return []
    parent = graph["nodes"][parent_id]
    summaries: list[str] = []
    for sibling_id in parent["children"]:
        if sibling_id == node["node_id"]:
            continue
        sibling = graph["nodes"][sibling_id]
        summary = sibling["investigation_summary"]
        if summary:
            summaries.append(summary)
    return summaries


def _collapse_candidates(
    graph: ComplexTaskGraphV1,
    active_node_id: str,
) -> list[dict[str, object]]:
    """Return existing resolved nodes as bounded collapse candidates."""

    active_node = graph["nodes"][active_node_id]
    if active_node["node_kind"] == "synthesis":
        return []
    candidates: list[dict[str, object]] = []
    for node_id, node in graph["nodes"].items():
        if node_id == active_node_id:
            continue
        if node["node_kind"] != active_node["node_kind"]:
            continue
        if node["status"] != "resolved":
            continue
        candidates.append(_compact_node(node))
    return candidates


def _resolved_node_summaries(
    graph: ComplexTaskGraphV1,
) -> list[dict[str, object]]:
    """Collect answer-bearing node summaries for bottom-up synthesis."""

    summaries: list[dict[str, object]] = []
    for node in graph["nodes"].values():
        if node["status"] not in ("resolved", "collapsed"):
            continue
        summaries.append(_compact_node(node))
    return summaries


def _unresolved_node_summaries(
    graph: ComplexTaskGraphV1,
) -> list[dict[str, object]]:
    """Collect blocked or unanswerable node summaries for synthesis."""

    summaries: list[dict[str, object]] = []
    for node in graph["nodes"].values():
        if node["status"] not in ("pending", "blocked", "cannot_answer"):
            continue
        summaries.append(_compact_node(node))
    return summaries


def _record_traversal(graph: ComplexTaskGraphV1, active_node_id: str) -> None:
    """Append a node id to traversal order if it has not been recorded."""

    if active_node_id not in graph["traversal_order"]:
        graph["traversal_order"].append(active_node_id)


def _summarize_result(result: dict[str, object]) -> str:
    """Return a compact deterministic summary for a subagent result."""

    if "display" in result and isinstance(result["display"], str):
        return_value = result["display"]
        return return_value
    if "summary" in result and isinstance(result["summary"], str):
        return_value = result["summary"]
        return return_value
    if "formula" in result and isinstance(result["formula"], str):
        return_value = result["formula"]
        return return_value
    return_value = "subagent result resolved"
    return return_value
