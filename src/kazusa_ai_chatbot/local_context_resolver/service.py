"""Standalone local-context resolver orchestration service."""

from __future__ import annotations

import json
import logging
import re

from .constants import (
    DEFAULT_OPTION_LIMITS,
    ROOT_CHILD_DEPTH,
    ROOT_NODE_DEPTH,
    ROOT_NODE_ID,
    SAFE_FAILURE_TEXT_LIMIT,
    TEXT_ELLIPSIS,
)
from .cache import (
    RAG3_ACTIVE_NODE_CACHE_NAME,
    RAG3_PLANNER_CACHE_NAME,
    active_node_cache_ttl_seconds,
    build_active_node_cache_dependencies,
    build_active_node_cache_key,
    build_planner_cache_key,
)
from .contracts import (
    ALLOWED_NODE_KINDS,
    ALLOWED_NODE_STATUSES,
    LOCAL_CONTEXT_ARTIFACT_VERSION,
    LOCAL_CONTEXT_GRAPH_VERSION,
    LOCAL_CONTEXT_NODE_VERSION,
    LOCAL_CONTEXT_RESOLUTION_PACKET_VERSION,
    LOCAL_CONTEXT_RESOLVER_OPTIONS_VERSION,
    LocalContextArtifactV1,
    LocalContextGraphV1,
    LocalContextNodeV1,
    LocalContextResolutionPacketV1,
    LocalContextResolverContextV1,
    LocalContextResolverOptionsV1,
    LocalContextResolverRequestV1,
    LocalContextValidationError,
    validate_local_context_artifact,
    validate_local_context_graph,
    validate_local_context_node,
    validate_local_context_resolution_packet,
    validate_local_context_resolver_context,
    validate_local_context_resolver_options,
    validate_local_context_resolver_request,
)
from .graph import find_next_active_node
from .subagent import dispatch_subagent_for_node
from .stages import (
    active_node_stage_cache_identity,
    plan_local_context_graph as _planner_stage_handler,
    planner_stage_cache_identity,
    resolve_local_context_node as _node_stage_handler,
    review_local_context_collapse as _collapse_stage_handler,
    synthesize_local_context_packet as _synthesizer_stage_handler,
)
from kazusa_ai_chatbot.rag.cache2_runtime import get_rag_cache2_runtime
from kazusa_ai_chatbot.rag.user_memory_unit_retrieval import (
    empty_user_memory_context,
)

logger = logging.getLogger(__name__)

_PROMPT_VISIBLE_RAG_LIST_FIELDS = (
    "third_party_profiles",
    "memory_evidence",
    "recall_evidence",
    "conversation_evidence",
    "external_evidence",
    "media_evidence",
    "user_memory_unit_candidates",
)
_TRACE_SOURCE_REF_LIMIT = 8
_NODE_UPDATE_FIELDS = frozenset((
    "status",
    "investigation_summary",
    "knowledge_we_know_so_far",
    "knowledge_still_lacking",
    "recommended_next_iteration",
    "evidence_boundary_notes",
    "attempts",
    "produces",
    "collapsed_into",
))
_FORBIDDEN_PROMPT_PAYLOAD_KEYS = frozenset((
    "_id",
    "adapter_message_id",
    "adapter_user_id",
    "cache_key",
    "cache_ref",
    "base64_data",
    "content_hash",
    "channel_id",
    "conversation_row_id",
    "created_at",
    "database_id",
    "embedding",
    "global_user_id",
    "llm_trace_id",
    "local_time",
    "local_timestamp",
    "message_id",
    "message_time",
    "message_timestamp",
    "platform_message_id",
    "platform_channel_id",
    "platform_user_id",
    "raw_timestamp",
    "raw_id",
    "scope_global_user_id",
    "source_message_id",
    "timestamp",
    "trace_id",
    "updated_at",
    "utc_timestamp",
))
_PROMPT_METADATA_REDACTION = "[metadata removed]"
_PROMPT_TIMESTAMP_REDACTION = "[time removed]"
_UNSAFE_METADATA_ASSIGNMENT_RE = re.compile(
    r"\b(?:"
    + "|".join(re.escape(key) for key in sorted(_FORBIDDEN_PROMPT_PAYLOAD_KEYS))
    + r")\b\s*[:=]\s*[\"']?[^,;)\]\s}]+[\"']?",
    re.IGNORECASE,
)
_UNSAFE_PROMPT_ID_RE = re.compile(
    r"\b(?:adapter|channel|character|database|message|platform|trace|user)-"
    r"[A-Za-z0-9_-]+\b",
    re.IGNORECASE,
)
_UNSAFE_UTC_TIMESTAMP_RE = re.compile(
    r"\b\d{4}-\d{2}-\d{2}T\d{2}:\d{2}"
    r"(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:\d{2})\b"
)


async def resolve_local_context(
    request: LocalContextResolverRequestV1,
    context: LocalContextResolverContextV1,
    options: LocalContextResolverOptionsV1 | None = None,
) -> LocalContextResolutionPacketV1:
    """Resolve a standalone local-context graph through the stable public IO."""

    try:
        validated_request = validate_local_context_resolver_request(request)
        validated_context = validate_local_context_resolver_context(context)
        raw_options = options
        if raw_options is None:
            raw_options = _default_options()
        validated_options = validate_local_context_resolver_options(raw_options)
    except LocalContextValidationError as exc:
        packet = _blocked_packet_from_reason(
            objective=_objective_from_request(request),
            reason=f"invalid local-context resolver input: {exc}",
            failure_stage="input_validation",
        )
        return packet

    try:
        packet = await _resolve_local_context_validated(
            validated_request,
            validated_context,
            validated_options,
        )
    except (LocalContextValidationError, ValueError) as exc:
        logger.warning(f"Local-context resolver produced a blocked packet: {exc}")
        packet = _blocked_packet_from_reason(
            objective=validated_request["objective"],
            reason=(
                "local-context resolver could not complete: "
                f"{type(exc).__name__}: {_safe_failure_text(str(exc))}"
            ),
            failure_stage="local_resolution",
        )
    return packet


async def _resolve_local_context_validated(
    request: LocalContextResolverRequestV1,
    context: LocalContextResolverContextV1,
    options: LocalContextResolverOptionsV1,
) -> LocalContextResolutionPacketV1:
    """Run the local-context resolver after public inputs validate."""

    trace_summary = _initial_trace_summary()
    artifacts: list[LocalContextArtifactV1] = []
    graph = await _plan_graph(request, context, options, trace_summary)
    await _run_graph_traversal(
        request=request,
        context=context,
        options=options,
        graph=graph,
        artifacts=artifacts,
        trace_summary=trace_summary,
    )
    packet = await _synthesize_packet(
        request=request,
        context=context,
        options=options,
        graph=graph,
        artifacts=artifacts,
        trace_summary=trace_summary,
    )
    return packet


async def _plan_graph(
    request: LocalContextResolverRequestV1,
    context: LocalContextResolverContextV1,
    options: LocalContextResolverOptionsV1,
    trace_summary: dict[str, object],
) -> LocalContextGraphV1:
    """Call the planner and map semantic tasks into deterministic graph nodes."""

    payload = {
        "stage": "graph_planner",
        "request": request,
        "context": _compact_context(context),
        "limits": _option_limits(options),
    }
    cache_key = build_planner_cache_key(
        request=request,
        context=context,
        options=options,
        stage_identity=planner_stage_cache_identity(),
    )
    runtime = get_rag_cache2_runtime()
    cached_response = await runtime.get(
        cache_key,
        cache_name=RAG3_PLANNER_CACHE_NAME,
        agent_name=RAG3_PLANNER_CACHE_NAME,
    )
    if isinstance(cached_response, dict):
        trace_summary["planner_cache_hits"] = (
            int(trace_summary["planner_cache_hits"]) + 1
        )
        trace_summary["cache_hits"] = int(trace_summary["cache_hits"]) + 1
        graph = _graph_from_planner_response(request, cached_response, options)
        _refresh_trace_counts(graph, trace_summary)
        return graph

    response = await _planner_stage_handler(payload)
    trace_summary["planner_calls"] = int(trace_summary["planner_calls"]) + 1
    graph = _graph_from_planner_response(request, response, options)
    await runtime.store(
        cache_key=cache_key,
        cache_name=RAG3_PLANNER_CACHE_NAME,
        result=response,
        dependencies=[],
        metadata={"agent_name": RAG3_PLANNER_CACHE_NAME},
    )
    _refresh_trace_counts(graph, trace_summary)
    return graph


def _graph_from_planner_response(
    request: LocalContextResolverRequestV1,
    response: dict[str, object],
    options: LocalContextResolverOptionsV1,
) -> LocalContextGraphV1:
    """Create strict graph nodes from the planner's semantic tasks."""

    tasks = _planner_tasks(response)
    max_nodes = options["max_nodes"]
    max_depth = options["max_depth"]
    if len(tasks) > max_nodes - 1:
        raise LocalContextValidationError("planner tasks: exceeds max_nodes")
    root_children = [
        _task_node_id(index)
        for index, _ in enumerate(tasks, start=1)
    ]
    nodes: dict[str, LocalContextNodeV1] = {
        ROOT_NODE_ID: _make_graph_node(
            node_id=ROOT_NODE_ID,
            node_kind="synthesis",
            objective=request["objective"],
            parent_id=None,
            children=root_children,
            status="resolved",
        )
    }
    for index, task in enumerate(tasks, start=1):
        node_id = _task_node_id(index)
        nodes[node_id] = _make_graph_node(
            node_id=node_id,
            node_kind=task["node_kind"],
            objective=task["objective"],
            parent_id=ROOT_NODE_ID,
            children=[],
            status="pending",
        )
    graph = {
        "schema_version": LOCAL_CONTEXT_GRAPH_VERSION,
        "root_node_id": ROOT_NODE_ID,
        "active_node_id": root_children[0],
        "nodes": nodes,
        "traversal_order": [ROOT_NODE_ID],
        "collapse_events": [],
        "max_nodes": max_nodes,
        "max_depth": max_depth,
    }
    validated_graph = validate_local_context_graph(graph)
    return validated_graph


def _planner_tasks(response: dict[str, object]) -> list[dict[str, str]]:
    """Validate local-LLM-friendly planner task rows."""

    raw_tasks = response.get("tasks")
    if not isinstance(raw_tasks, list) or not raw_tasks:
        raise LocalContextValidationError("planner tasks: expected non-empty list")
    tasks: list[dict[str, str]] = []
    for raw_task in raw_tasks:
        if not isinstance(raw_task, dict):
            raise LocalContextValidationError("planner task: expected object")
        objective = raw_task.get("objective")
        if not isinstance(objective, str) or not objective.strip():
            raise LocalContextValidationError(
                "planner task objective: expected non-empty string"
            )
        raw_node_kind = raw_task.get("node_kind")
        if not isinstance(raw_node_kind, str):
            raw_node_kind = raw_task.get("kind")
        if not isinstance(raw_node_kind, str):
            raise LocalContextValidationError(
                "planner task node_kind: expected string"
            )
        node_kind = _node_kind_from_semantic_text(raw_node_kind)
        if node_kind not in ALLOWED_NODE_KINDS:
            raise LocalContextValidationError(
                "planner task node_kind: expected known node kind"
            )
        if node_kind == "synthesis":
            continue
        tasks.append({
            "objective": objective.strip(),
            "node_kind": node_kind,
        })
    if not tasks:
        raise LocalContextValidationError("planner tasks: no evidence tasks")
    return tasks


async def _run_graph_traversal(
    *,
    request: LocalContextResolverRequestV1,
    context: LocalContextResolverContextV1,
    options: LocalContextResolverOptionsV1,
    graph: LocalContextGraphV1,
    artifacts: list[LocalContextArtifactV1],
    trace_summary: dict[str, object],
) -> None:
    """Resolve pending graph nodes through a bounded traversal loop."""

    for _ in range(options["max_iterations"]):
        active_node_id = find_next_active_node(graph)
        if active_node_id is None:
            break
        graph["active_node_id"] = active_node_id
        validate_local_context_graph(graph)
        response = await _resolve_active_node(
            request=request,
            context=context,
            options=options,
            graph=graph,
            active_node_id=active_node_id,
            trace_summary=trace_summary,
        )
        _apply_active_node_response(
            graph=graph,
            active_node_id=active_node_id,
            response=response,
            artifacts=artifacts,
            trace_summary=trace_summary,
        )
        _record_traversal(graph, active_node_id)
        validate_local_context_graph(graph)
        collapse_response = await _review_collapse(
            request=request,
            context=context,
            options=options,
            graph=graph,
            active_node_id=active_node_id,
            trace_summary=trace_summary,
        )
        _apply_collapse_response(
            graph=graph,
            active_node_id=active_node_id,
            response=collapse_response,
            trace_summary=trace_summary,
        )
        trace_summary["iterations"] = int(trace_summary["iterations"]) + 1
        _refresh_trace_counts(graph, trace_summary)
        validate_local_context_graph(graph)


async def _resolve_active_node(
    *,
    request: LocalContextResolverRequestV1,
    context: LocalContextResolverContextV1,
    options: LocalContextResolverOptionsV1,
    graph: LocalContextGraphV1,
    active_node_id: str,
    trace_summary: dict[str, object],
) -> dict[str, object]:
    """Call the active-node stage for one graph node."""

    active_node = graph["nodes"][active_node_id]
    compact_context = _compact_context(context)
    dependency_context = _dependency_context(graph, active_node)
    use_active_node_cache = active_node["node_kind"] not in (
        "current_turn_media",
        "recent_media",
    )
    cache_key = build_active_node_cache_key(
        request=request,
        context=context,
        compact_context=compact_context,
        active_node=active_node,
        dependency_context=dependency_context,
        options=options,
        stage_identity=active_node_stage_cache_identity(),
    )
    runtime = get_rag_cache2_runtime()
    cached_response = None
    if use_active_node_cache:
        cached_response = await runtime.get(
            cache_key,
            cache_name=RAG3_ACTIVE_NODE_CACHE_NAME,
            agent_name=RAG3_ACTIVE_NODE_CACHE_NAME,
        )
    if isinstance(cached_response, dict):
        trace_summary["active_node_cache_hits"] = (
            int(trace_summary["active_node_cache_hits"]) + 1
        )
        trace_summary["cache_hits"] = int(trace_summary["cache_hits"]) + 1
        return cached_response

    subagent_result = _empty_subagent_result()
    if request["source"] in ("l2d", "live_llm_review"):
        subagent_result = await dispatch_subagent_for_node(
            active_node=active_node,
            context=context,
            dependency_context=dependency_context,
            max_attempts=options["max_subagent_attempts"],
        )
        trace_summary["subagent_calls"] = int(trace_summary["subagent_calls"]) + 1
    prompt_context = dict(compact_context)
    subagent_payload = subagent_result["result"]
    source_records = subagent_payload.get("source_records")
    if isinstance(source_records, list) and source_records:
        prompt_context["source_context"] = _sanitized_prompt_payload(
            source_records
        )
    payload = {
        "stage": "active_node_resolver",
        "request": request,
        "context": prompt_context,
        "active_node": _compact_node(active_node),
        "dependency_context": dependency_context,
        "limits": _option_limits(options),
    }
    response = await _node_stage_handler(payload)
    response = _merge_subagent_response(
        response,
        subagent_payload,
    )
    cacheable_response = _cacheable_active_node_response(
        response,
        active_node_id,
    )
    trace_summary["active_node_calls"] = int(
        trace_summary["active_node_calls"]
    ) + 1
    if use_active_node_cache:
        await runtime.store(
            cache_key=cache_key,
            cache_name=RAG3_ACTIVE_NODE_CACHE_NAME,
            result=cacheable_response,
            dependencies=build_active_node_cache_dependencies(
                node_kind=active_node["node_kind"],
                context=context,
            ),
            metadata={
                "agent_name": RAG3_ACTIVE_NODE_CACHE_NAME,
                "node_kind": active_node["node_kind"],
            },
            ttl_seconds=active_node_cache_ttl_seconds(active_node["node_kind"]),
        )
    return cacheable_response


def _merge_subagent_response(
    response: dict[str, object],
    subagent_payload: dict[str, object],
) -> dict[str, object]:
    """Merge deterministic source evidence into an active-node response."""

    source_update = subagent_payload.get("node_update")
    source_artifacts = subagent_payload.get("artifacts")
    if not isinstance(source_update, dict):
        source_update = {}
    if not isinstance(source_artifacts, list):
        source_artifacts = []
    if not source_update and not source_artifacts:
        return response

    merged_response = dict(response)
    raw_node_update = response.get("node_update")
    if isinstance(raw_node_update, dict):
        node_update = dict(raw_node_update)
    else:
        node_update = {}
    _merge_source_node_update(node_update, source_update)
    merged_response["node_update"] = node_update

    raw_artifacts = response.get("artifacts")
    if isinstance(raw_artifacts, list):
        artifacts = list(raw_artifacts)
    else:
        artifacts = []
    artifacts.extend(source_artifacts)
    merged_response["artifacts"] = artifacts
    return merged_response


def _empty_subagent_result() -> dict[str, object]:
    """Return the canonical no-dispatch result for standalone graph review."""

    result = {
        "result": {
            "source_records": [],
            "artifacts": [],
            "node_update": {},
        },
    }
    return result


def _merge_source_node_update(
    node_update: dict[str, object],
    source_update: dict[str, object],
) -> None:
    """Merge source-owned node rows while preserving model-authored rows."""

    if source_update.get("status") == "resolved":
        node_update["status"] = "resolved"
        node_update["knowledge_still_lacking"] = []
    for field_name in (
        "investigation_summary",
        "knowledge_we_know_so_far",
        "recommended_next_iteration",
        "evidence_boundary_notes",
        "produces",
    ):
        source_rows = source_update.get(field_name)
        if not isinstance(source_rows, list):
            continue
        target_rows = node_update.get(field_name)
        if not isinstance(target_rows, list):
            target_rows = []
        for row in source_rows:
            if not isinstance(row, str) or not row.strip():
                continue
            if row in target_rows:
                continue
            target_rows.append(row)
        node_update[field_name] = target_rows


def _apply_active_node_response(
    *,
    graph: LocalContextGraphV1,
    active_node_id: str,
    response: dict[str, object],
    artifacts: list[LocalContextArtifactV1],
    trace_summary: dict[str, object],
) -> None:
    """Apply semantic node updates and prompt-visible artifacts."""

    node = graph["nodes"][active_node_id]
    node_update = response.get("node_update")
    if not isinstance(node_update, dict):
        raise LocalContextValidationError("node_update: expected object")
    for field_name in _NODE_UPDATE_FIELDS:
        if field_name not in node_update:
            continue
        node[field_name] = _node_update_value(field_name, node_update[field_name])
    if node["status"] not in ALLOWED_NODE_STATUSES:
        raise LocalContextValidationError("node_update.status: unknown status")
    raw_artifacts = response.get("artifacts")
    if raw_artifacts is None:
        raw_artifacts = []
    if not isinstance(raw_artifacts, list):
        raise LocalContextValidationError("artifacts: expected list")
    for raw_artifact in raw_artifacts:
        artifact = _validated_artifact_for_node(raw_artifact, active_node_id)
        artifacts.append(artifact)
    _refresh_trace_counts(graph, trace_summary)


def _cacheable_active_node_response(
    response: dict[str, object],
    active_node_id: str,
) -> dict[str, object]:
    """Return the normalized active-node response safe to store in Cache2."""

    node_update = response.get("node_update")
    if not isinstance(node_update, dict):
        raise LocalContextValidationError("node_update: expected object")
    normalized_update: dict[str, object] = {}
    for field_name, value in node_update.items():
        if field_name not in _NODE_UPDATE_FIELDS:
            continue
        normalized_update[field_name] = _node_update_value(field_name, value)
    _validate_cacheable_node_update(normalized_update)

    raw_artifacts = response.get("artifacts")
    if raw_artifacts is None:
        raw_artifacts = []
    if not isinstance(raw_artifacts, list):
        raise LocalContextValidationError("artifacts: expected list")
    normalized_artifacts: list[LocalContextArtifactV1] = []
    for raw_artifact in raw_artifacts:
        artifact = _validated_artifact_for_node(raw_artifact, active_node_id)
        normalized_artifacts.append(artifact)

    cacheable_response = {
        "node_update": normalized_update,
        "artifacts": normalized_artifacts,
    }
    return cacheable_response


def _validate_cacheable_node_update(node_update: dict[str, object]) -> None:
    """Validate normalized active-node rows before Cache2 storage."""

    status = node_update.get("status")
    if status is not None and status not in ALLOWED_NODE_STATUSES:
        raise LocalContextValidationError("node_update.status: unknown status")
    for field_name in (
        "investigation_summary",
        "knowledge_we_know_so_far",
        "knowledge_still_lacking",
        "recommended_next_iteration",
        "evidence_boundary_notes",
        "produces",
    ):
        value = node_update.get(field_name)
        if value is None:
            continue
        if not isinstance(value, list):
            raise LocalContextValidationError(f"{field_name}: expected list")
        for item in value:
            if not isinstance(item, str) or not item.strip():
                raise LocalContextValidationError(
                    f"{field_name}: expected non-empty strings"
                )
    attempts = node_update.get("attempts")
    if attempts is not None and not isinstance(attempts, list):
        raise LocalContextValidationError("attempts: expected list")
    collapsed_into = node_update.get("collapsed_into")
    if collapsed_into is not None:
        if not isinstance(collapsed_into, str) or not collapsed_into.strip():
            raise LocalContextValidationError(
                "collapsed_into: expected non-empty string"
            )


def _node_update_value(field_name: str, value: object) -> object:
    """Validate one stage-owned node update value before assignment."""

    if field_name in {
        "investigation_summary",
        "knowledge_we_know_so_far",
        "knowledge_still_lacking",
        "recommended_next_iteration",
        "evidence_boundary_notes",
        "produces",
    }:
        if isinstance(value, str) and value.strip():
            return_value = [value.strip()]
            return return_value
        if not isinstance(value, list):
            raise LocalContextValidationError(f"{field_name}: expected list")
    if field_name == "attempts":
        if not isinstance(value, list):
            raise LocalContextValidationError(f"{field_name}: expected list")
    if field_name == "status":
        if not isinstance(value, str):
            raise LocalContextValidationError("status: expected string")
    if field_name == "collapsed_into":
        if value is not None and not isinstance(value, str):
            raise LocalContextValidationError("collapsed_into: expected string")
    return_value = value
    return return_value


def _validated_artifact_for_node(
    raw_artifact: object,
    active_node_id: str,
) -> LocalContextArtifactV1:
    """Validate a node artifact and bind active-node aliases deterministically."""

    if not isinstance(raw_artifact, dict):
        raise LocalContextValidationError("artifact: expected object")
    artifact_data = dict(raw_artifact)
    artifact_data["producer_node_id"] = active_node_id
    if "schema_version" not in artifact_data:
        artifact_data["schema_version"] = LOCAL_CONTEXT_ARTIFACT_VERSION
    artifact_type = artifact_data.get("artifact_type")
    if isinstance(artifact_type, str):
        artifact_data["artifact_type"] = _artifact_type_from_semantic_text(
            artifact_type
        )
    artifact = validate_local_context_artifact(artifact_data)
    return artifact


async def _review_collapse(
    *,
    request: LocalContextResolverRequestV1,
    context: LocalContextResolverContextV1,
    options: LocalContextResolverOptionsV1,
    graph: LocalContextGraphV1,
    active_node_id: str,
    trace_summary: dict[str, object],
) -> dict[str, object]:
    """Call the collapse-review stage for one resolved active node."""

    active_node = graph["nodes"][active_node_id]
    candidates = _collapse_candidates(graph, active_node_id)
    if not candidates:
        return_value: dict[str, object] = {
            "collapse_decision": {
                "should_collapse": False,
                "target_candidate_ref": "",
                "reason": "no resolved same-kind collapse candidates",
            },
        }
        return return_value
    payload = {
        "stage": "collapse_review",
        "request": request,
        "context": _compact_context(context),
        "active_node": _compact_node(active_node),
        "candidates": candidates,
        "limits": _option_limits(options),
    }
    response = await _collapse_stage_handler(payload)
    trace_summary["collapse_calls"] = int(trace_summary["collapse_calls"]) + 1
    return response


def _apply_collapse_response(
    *,
    graph: LocalContextGraphV1,
    active_node_id: str,
    response: dict[str, object],
    trace_summary: dict[str, object],
) -> None:
    """Apply a bounded collapse decision when it targets a valid node."""

    decision = response.get("collapse_decision")
    if not isinstance(decision, dict):
        return
    should_collapse = decision.get("should_collapse")
    if should_collapse is not True:
        return
    target_candidate_ref = decision.get("target_candidate_ref")
    if not isinstance(target_candidate_ref, str):
        return
    candidate_refs = _collapse_candidate_ref_map(graph, active_node_id)
    target_internal_id = candidate_refs.get(target_candidate_ref)
    if target_internal_id is None:
        return
    if target_internal_id == active_node_id:
        return
    active_node = graph["nodes"][active_node_id]
    if active_node["status"] != "resolved":
        return
    target_node = graph["nodes"][target_internal_id]
    if target_node["status"] not in ("resolved", "collapsed"):
        return
    active_node["status"] = "collapsed"
    active_node["collapsed_into"] = target_internal_id
    reason = decision.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        reason = "semantically duplicate local-context node"
    graph["collapse_events"].append({
        "from_node_id": active_node_id,
        "to_node_id": target_internal_id,
        "reason": reason,
    })
    trace_summary["collapse_count"] = int(trace_summary["collapse_count"]) + 1


async def _synthesize_packet(
    *,
    request: LocalContextResolverRequestV1,
    context: LocalContextResolverContextV1,
    options: LocalContextResolverOptionsV1,
    graph: LocalContextGraphV1,
    artifacts: list[LocalContextArtifactV1],
    trace_summary: dict[str, object],
) -> LocalContextResolutionPacketV1:
    """Synthesize bottom-up semantic sections and retained RAG result."""

    if _can_use_deterministic_synthesis(graph, artifacts):
        synthesis = _deterministic_synthesis_response(graph)
    else:
        payload = {
            "stage": "bottom_up_synthesis",
            "request": request,
            "context": _compact_context(context),
            "resolved_nodes": _resolved_node_summaries(graph),
            "unresolved_nodes": _unresolved_node_summaries(graph),
            "limits": _option_limits(options),
        }
        response = await _synthesizer_stage_handler(payload)
        trace_summary["synthesis_calls"] = int(trace_summary["synthesis_calls"]) + 1
        synthesis = _semantic_synthesis_response(response, graph)
    _refresh_trace_counts(graph, trace_summary)
    rag_result = _rag_result_from_artifacts(
        artifacts=artifacts,
        synthesis=synthesis,
        context=context,
        graph=graph,
        trace_summary=trace_summary,
    )
    packet = {
        "schema_version": LOCAL_CONTEXT_RESOLUTION_PACKET_VERSION,
        "investigation_summary": synthesis["investigation_summary"],
        "knowledge_we_know_so_far": synthesis["knowledge_we_know_so_far"],
        "knowledge_still_lacking": synthesis["knowledge_still_lacking"],
        "recommended_next_iteration": synthesis["recommended_next_iteration"],
        "evidence_boundary_notes": synthesis["evidence_boundary_notes"],
        "rag_result": rag_result,
        "graph": graph,
        "trace_summary": trace_summary,
    }
    validated_packet = validate_local_context_resolution_packet(packet)
    return validated_packet


def _can_use_deterministic_synthesis(
    graph: LocalContextGraphV1,
    artifacts: list[LocalContextArtifactV1],
) -> bool:
    """Return whether node-owned rows are sufficient for final packet fields."""

    if _unresolved_node_summaries(graph):
        return False
    if artifacts:
        return True
    if _resolved_node_known_rows(graph):
        return True
    return False


def _deterministic_synthesis_response(
    graph: LocalContextGraphV1,
) -> dict[str, list[str]]:
    """Aggregate bottom-up packet rows without adding new semantics."""

    synthesis = {
        "investigation_summary": _unique_semantic_rows(
            _resolved_node_summary_rows(graph)
        ),
        "knowledge_we_know_so_far": _unique_semantic_rows(
            _resolved_node_known_rows(graph)
        ),
        "knowledge_still_lacking": [],
        "recommended_next_iteration": [],
        "evidence_boundary_notes": _unique_semantic_rows(
            _node_boundary_note_rows(graph)
        ),
    }
    return synthesis


def _semantic_synthesis_response(
    response: dict[str, object],
    graph: LocalContextGraphV1,
) -> dict[str, list[str]]:
    """Normalize synthesis output into final semantic packet rows."""

    synthesis = {
        "investigation_summary": _semantic_text_list(
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
    if not synthesis["investigation_summary"]:
        synthesis["investigation_summary"] = _resolved_node_summary_rows(graph)
    if not synthesis["knowledge_we_know_so_far"]:
        synthesis["knowledge_we_know_so_far"] = _resolved_node_known_rows(graph)
    if not synthesis["knowledge_still_lacking"]:
        synthesis["knowledge_still_lacking"] = _unresolved_node_lacking_rows(graph)
    if not synthesis["evidence_boundary_notes"]:
        synthesis["evidence_boundary_notes"] = _node_boundary_note_rows(graph)
    return synthesis


def _rag_result_from_artifacts(
    *,
    artifacts: list[LocalContextArtifactV1],
    synthesis: dict[str, list[str]],
    context: LocalContextResolverContextV1 | None = None,
    graph: LocalContextGraphV1 | None = None,
    trace_summary: dict[str, object],
) -> dict[str, object]:
    """Build the retained prompt-facing RAG result from visible artifacts."""

    del synthesis
    rag_result = _empty_rag_result(trace_summary)
    if context is not None and graph is not None and _has_resolved_live_context(graph):
        _merge_structured_live_context(
            rag_result,
            context["local_time_context"],
        )
    for artifact in artifacts:
        if not artifact["prompt_visible"]:
            continue
        payload = artifact["projection_payload"]
        _merge_projection_payload(
            rag_result,
            payload,
            artifact_type=artifact["artifact_type"],
            source_policy=artifact["source_policy"],
        )
    return rag_result


def _merge_projection_payload(
    rag_result: dict[str, object],
    payload: dict[str, object],
    *,
    artifact_type: str = "",
    source_policy: str = "",
) -> None:
    """Merge one prompt-visible artifact payload into the RAG result shape."""

    _append_conversation_trace_refs(
        rag_result,
        payload,
        artifact_type=artifact_type,
    )
    payload = _normalized_projection_payload(
        payload,
        artifact_type=artifact_type,
        source_policy=source_policy,
    )
    answer = payload.get("answer")
    if isinstance(answer, str) and answer.strip():
        sanitized_answer = _sanitized_prompt_payload(answer)
        if isinstance(sanitized_answer, str):
            rag_result["answer"] = sanitized_answer
    for field_name in _PROMPT_VISIBLE_RAG_LIST_FIELDS:
        value = payload.get(field_name)
        if not isinstance(value, list):
            continue
        target = rag_result[field_name]
        if not isinstance(target, list):
            raise LocalContextValidationError(f"rag_result.{field_name}: expected list")
        for item in value:
            sanitized_item = _sanitized_prompt_payload(item)
            if sanitized_item is None or sanitized_item == {}:
                continue
            if sanitized_item in target:
                continue
            target.append(sanitized_item)
    for field_name in ("user_image", "character_image"):
        value = payload.get(field_name)
        if isinstance(value, dict):
            sanitized_value = _sanitized_prompt_payload(value)
            if isinstance(sanitized_value, dict):
                if field_name == "user_image":
                    sanitized_value = _normalized_user_image(sanitized_value)
                rag_result[field_name] = sanitized_value


def _append_conversation_trace_refs(
    rag_result: dict[str, object],
    payload: dict[str, object],
    *,
    artifact_type: str,
) -> None:
    """Preserve private row refs for past-dialog cognition trace consumers."""

    if artifact_type != "conversation_ref":
        return
    source_refs = _conversation_source_refs_from_value(payload)
    if not source_refs:
        return
    supervisor_trace = rag_result.get("supervisor_trace")
    if not isinstance(supervisor_trace, dict):
        return
    dispatched = supervisor_trace.get("dispatched")
    if not isinstance(dispatched, list):
        dispatched = []
        supervisor_trace["dispatched"] = dispatched
    dispatched.append({
        "agent": "conversation_evidence_agent",
        "source_refs": source_refs,
    })


def _conversation_source_refs_from_value(value: object) -> list[dict[str, str]]:
    """Extract trace-only conversation row refs from raw projection payloads."""

    source_refs: list[dict[str, str]] = []
    seen_refs: set[tuple[str, str]] = set()

    def visit(candidate: object) -> None:
        if len(source_refs) >= _TRACE_SOURCE_REF_LIMIT:
            return
        if isinstance(candidate, dict):
            row_id = _source_ref_text(candidate.get("conversation_row_id"))
            if row_id:
                key = ("conversation_row_id", row_id)
                if key not in seen_refs:
                    source_refs.append({"conversation_row_id": row_id})
                    seen_refs.add(key)
            else:
                object_id = _source_ref_text(candidate.get("_id"))
                if object_id:
                    key = ("_id", object_id)
                    if key not in seen_refs:
                        source_refs.append({"_id": object_id})
                        seen_refs.add(key)
            for item in candidate.values():
                visit(item)
            return
        if isinstance(candidate, list):
            for item in candidate:
                visit(item)

    visit(value)
    return source_refs


def _source_ref_text(value: object) -> str:
    """Return a compact trace source ref string, or empty string."""

    if value is None:
        return ""
    if not isinstance(value, str):
        if value.__class__.__name__ != "ObjectId":
            return ""
        value = str(value)
    stripped_value = value.strip()
    return stripped_value


def _normalized_user_image(user_image: dict[str, object]) -> dict[str, object]:
    """Return a user image with the retained memory-context substructure."""

    normalized_user_image = dict(user_image)
    memory_context = normalized_user_image.get("user_memory_context")
    if not isinstance(memory_context, dict):
        normalized_user_image["user_memory_context"] = empty_user_memory_context()
    return normalized_user_image


def _normalized_projection_payload(
    payload: dict[str, object],
    *,
    artifact_type: str,
    source_policy: str,
) -> dict[str, object]:
    """Normalize source-owned evidence fields before prompt-facing merge."""

    normalized_payload: dict[str, object] = dict(payload)
    _move_live_evidence_to_conversation(normalized_payload)
    _move_user_memory_units_to_candidates(normalized_payload, source_policy)
    if artifact_type == "conversation_ref":
        _move_list_field(
            normalized_payload,
            source_field="recall_evidence",
            target_field="conversation_evidence",
        )
        _move_chat_memory_rows_to_conversation(normalized_payload)
    if artifact_type == "external_ref":
        _move_list_field(
            normalized_payload,
            source_field="conversation_evidence",
            target_field="external_evidence",
        )
    if artifact_type == "recall_ref" and normalized_payload.get("recall_evidence"):
        normalized_payload["conversation_evidence"] = []
    return normalized_payload


def _move_live_evidence_to_conversation(payload: dict[str, object]) -> None:
    """Map model-emitted live evidence into the retained RAG surface."""

    _move_list_field(
        payload,
        source_field="live_evidence",
        target_field="conversation_evidence",
    )


def _move_user_memory_units_to_candidates(
    payload: dict[str, object],
    source_policy: str,
) -> None:
    """Put current-user memory-unit rows in the scoped candidate field."""

    memory_items = payload.get("memory_evidence")
    if not isinstance(memory_items, list):
        return
    retained_memory_items: list[object] = []
    scoped_items: list[object] = []
    for item in memory_items:
        if _is_user_memory_unit_item(item) or _is_user_memory_source(source_policy):
            scoped_items.append(item)
        else:
            retained_memory_items.append(item)
    payload["memory_evidence"] = retained_memory_items
    if not scoped_items:
        return
    candidates = payload.get("user_memory_unit_candidates")
    if not isinstance(candidates, list):
        candidates = []
    for item in scoped_items:
        if item in candidates:
            continue
        candidates.append(item)
    payload["user_memory_unit_candidates"] = candidates


def _move_chat_memory_rows_to_conversation(payload: dict[str, object]) -> None:
    """Move chat-source rows emitted in memory_evidence to conversation evidence."""

    memory_items = payload.get("memory_evidence")
    if not isinstance(memory_items, list):
        return
    retained_memory_items: list[object] = []
    conversation_items = payload.get("conversation_evidence")
    if not isinstance(conversation_items, list):
        conversation_items = []
    for item in memory_items:
        if _is_chat_source_item(item):
            if item not in conversation_items:
                conversation_items.append(item)
        else:
            retained_memory_items.append(item)
    payload["memory_evidence"] = retained_memory_items
    payload["conversation_evidence"] = conversation_items


def _move_list_field(
    payload: dict[str, object],
    *,
    source_field: str,
    target_field: str,
) -> None:
    """Move list payload rows from one prompt-facing field to another."""

    source = payload.get(source_field)
    if not isinstance(source, list) or not source:
        return
    target = payload.get(target_field)
    if not isinstance(target, list):
        target = []
    for item in source:
        if item in target:
            continue
        target.append(item)
    payload[target_field] = target
    payload[source_field] = []


def _is_user_memory_unit_item(item: object) -> bool:
    """Return whether one evidence row is explicitly from user memory units."""

    return _item_text_contains(
        item,
        (
            "user_memory_units",
            "user-scoped",
            "user scoped",
            "current user's continuity",
            "private continuity",
        ),
    )


def _is_user_memory_source(source_policy: str) -> bool:
    """Return whether a source policy describes scoped user memory."""

    return _item_text_contains(
        source_policy,
        (
            "user_memory_units",
            "user-scoped",
            "user scoped",
            "user memory",
            "current-user memory",
        ),
    )


def _is_chat_source_item(item: object) -> bool:
    """Return whether one evidence row clearly names chat history as source."""

    return _item_text_contains(
        item,
        (
            "chat_history_recent",
            "chat history",
            "conversation",
        ),
    )


def _item_text_contains(item: object, fragments: tuple[str, ...]) -> bool:
    """Search one simple JSON-like item for source-identifying fragments."""

    item_text = json.dumps(item, ensure_ascii=False).lower()
    return any(fragment in item_text for fragment in fragments)


def _has_resolved_live_context(graph: LocalContextGraphV1) -> bool:
    """Return whether traversal resolved a live-context node."""

    for node in graph["nodes"].values():
        if node["node_kind"] == "live_context" and node["status"] == "resolved":
            return True
    return False


def _merge_structured_live_context(
    rag_result: dict[str, object],
    local_time_context: dict[str, object],
) -> None:
    """Project structured live context into the retained prompt surface."""

    labels = (
        ("local_date", "date"),
        ("local_weekday", "weekday"),
        ("local_time", "time"),
        ("timezone", "timezone"),
    )
    parts: list[str] = []
    for key, label in labels:
        value = local_time_context.get(key)
        if not isinstance(value, str) or not value.strip():
            continue
        parts.append(f"{label}: {value.strip()}")
    if not parts:
        return
    target = rag_result["conversation_evidence"]
    if not isinstance(target, list):
        raise LocalContextValidationError(
            "rag_result.conversation_evidence: expected list"
        )
    item = {
        "source": "live_context",
        "content": "Live local context; " + "; ".join(parts),
    }
    if item not in target:
        target.append(item)


def _sanitized_prompt_payload(value: object) -> object | None:
    """Strip prompt-unsafe metadata from projection payload values."""

    if isinstance(value, dict):
        sanitized_dict: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                continue
            if key in _FORBIDDEN_PROMPT_PAYLOAD_KEYS:
                continue
            sanitized_item = _sanitized_prompt_payload(item)
            if sanitized_item is None:
                continue
            sanitized_dict[key] = sanitized_item
        return_value = sanitized_dict
        return return_value
    if isinstance(value, list):
        sanitized_list: list[object] = []
        for item in value:
            sanitized_item = _sanitized_prompt_payload(item)
            if sanitized_item is None or sanitized_item == {}:
                continue
            if sanitized_item in sanitized_list:
                continue
            sanitized_list.append(sanitized_item)
        return_value = sanitized_list
        return return_value
    if isinstance(value, str):
        return_value = _sanitized_prompt_string(value)
        return return_value
    if isinstance(value, (int, float, bool)) or value is None:
        return_value = value
        return return_value
    return_value = None
    return return_value


def _sanitized_prompt_string(value: str) -> str | None:
    """Remove prompt-unsafe embedded metadata from one string value."""

    stripped_value = value.strip()
    if not stripped_value:
        return None
    safe_value = _UNSAFE_METADATA_ASSIGNMENT_RE.sub(
        _PROMPT_METADATA_REDACTION,
        stripped_value,
    )
    safe_value = _UNSAFE_UTC_TIMESTAMP_RE.sub(
        _PROMPT_TIMESTAMP_REDACTION,
        safe_value,
    )
    safe_value = _UNSAFE_PROMPT_ID_RE.sub(
        _PROMPT_METADATA_REDACTION,
        safe_value,
    )
    safe_value = " ".join(safe_value.split())
    if not safe_value or safe_value in {
        _PROMPT_METADATA_REDACTION,
        _PROMPT_TIMESTAMP_REDACTION,
    }:
        return None
    return_value = safe_value
    return return_value


def _empty_rag_result(trace_summary: dict[str, object]) -> dict[str, object]:
    """Return the retained RAG2-compatible top-level evidence surface."""

    supervisor_trace = {
        "resolver": "local_context_resolver",
        "iterations": trace_summary["iterations"],
        "node_count": trace_summary["node_count"],
        "resolved_node_count": trace_summary["resolved_node_count"],
        "blocked_node_count": trace_summary["blocked_node_count"],
        "dispatched": [],
    }
    rag_result = {
        "answer": "",
        "user_image": {
            "user_memory_context": empty_user_memory_context(),
        },
        "user_memory_unit_candidates": [],
        "character_image": {},
        "third_party_profiles": [],
        "memory_evidence": [],
        "recall_evidence": [],
        "conversation_evidence": [],
        "external_evidence": [],
        "media_evidence": [],
        "supervisor_trace": supervisor_trace,
    }
    return rag_result


def _blocked_packet_from_reason(
    *,
    objective: str,
    reason: str,
    failure_stage: str,
) -> LocalContextResolutionPacketV1:
    """Build a bounded blocked packet for malformed input or stage output."""

    safe_reason = _safe_failure_text(reason)
    root_node = _make_graph_node(
        node_id=ROOT_NODE_ID,
        node_kind="synthesis",
        objective=objective,
        parent_id=None,
        children=[],
        status="blocked",
    )
    root_node["knowledge_still_lacking"] = [safe_reason]
    graph = {
        "schema_version": LOCAL_CONTEXT_GRAPH_VERSION,
        "root_node_id": ROOT_NODE_ID,
        "active_node_id": ROOT_NODE_ID,
        "nodes": {ROOT_NODE_ID: root_node},
        "traversal_order": [ROOT_NODE_ID],
        "collapse_events": [],
        "max_nodes": 1,
        "max_depth": 1,
    }
    trace_summary = _initial_trace_summary()
    trace_summary["failure_stage"] = failure_stage
    trace_summary["failure_reason"] = safe_reason
    _refresh_trace_counts(graph, trace_summary)
    rag_result = _empty_rag_result(trace_summary)
    packet = {
        "schema_version": LOCAL_CONTEXT_RESOLUTION_PACKET_VERSION,
        "investigation_summary": [
            "The local-context resolver did not complete its investigation."
        ],
        "knowledge_we_know_so_far": [],
        "knowledge_still_lacking": [safe_reason],
        "recommended_next_iteration": [
            "Cognition may continue without this local-context evidence."
        ],
        "evidence_boundary_notes": [
            "The resolver returned a bounded blocked packet."
        ],
        "rag_result": rag_result,
        "graph": graph,
        "trace_summary": trace_summary,
    }
    validated_packet = validate_local_context_resolution_packet(packet)
    return validated_packet


def _initial_trace_summary() -> dict[str, object]:
    """Return raw counters used by standalone efficiency review."""

    trace_summary: dict[str, object] = {
        "iterations": 0,
        "node_count": 0,
        "max_depth_observed": 0,
        "resolved_node_count": 0,
        "blocked_node_count": 0,
        "planner_calls": 0,
        "planner_cache_hits": 0,
        "active_node_calls": 0,
        "active_node_cache_hits": 0,
        "collapse_calls": 0,
        "synthesis_calls": 0,
        "subagent_calls": 0,
        "collapse_count": 0,
        "cache_hits": 0,
    }
    return trace_summary


def _refresh_trace_counts(
    graph: LocalContextGraphV1,
    trace_summary: dict[str, object],
) -> None:
    """Refresh graph-derived raw counters after a state mutation."""

    trace_summary["node_count"] = len(graph["nodes"])
    trace_summary["max_depth_observed"] = _max_depth_observed(graph)
    trace_summary["resolved_node_count"] = sum(
        1
        for node in graph["nodes"].values()
        if node["status"] in ("resolved", "collapsed")
    )
    trace_summary["blocked_node_count"] = sum(
        1
        for node in graph["nodes"].values()
        if node["status"] in ("blocked", "cannot_answer")
    )


def _make_graph_node(
    *,
    node_id: str,
    node_kind: str,
    objective: str,
    parent_id: str | None,
    children: list[str],
    status: str,
) -> LocalContextNodeV1:
    """Create one graph node with empty semantic projection rows."""

    node = {
        "schema_version": LOCAL_CONTEXT_NODE_VERSION,
        "node_id": node_id,
        "node_kind": node_kind,
        "objective": objective,
        "parent_id": parent_id,
        "children": children,
        "depends_on": [],
        "consumes": {},
        "produces": [],
        "status": status,
        "investigation_summary": [],
        "knowledge_we_know_so_far": [],
        "knowledge_still_lacking": [],
        "recommended_next_iteration": [],
        "evidence_boundary_notes": [],
        "attempts": [],
        "collapsed_into": None,
    }
    validated_node = validate_local_context_node(node)
    return validated_node


def _task_node_id(index: int) -> str:
    """Return the deterministic node id for one planner task."""

    node_id = f"task_{index}"
    return node_id


def _node_kind_from_semantic_text(value: str) -> str:
    """Map planner vocabulary into local-context node kinds."""

    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "conversation": "conversation_evidence",
        "conversation_evidence": "conversation_evidence",
        "current_media": "current_turn_media",
        "current_turn_media": "current_turn_media",
        "external": "external_evidence",
        "external_evidence": "external_evidence",
        "live": "live_context",
        "live_context": "live_context",
        "memory": "memory_evidence",
        "memory_evidence": "memory_evidence",
        "person": "person_context",
        "person_context": "person_context",
        "profile": "person_context",
        "recall": "recall_evidence",
        "recall_evidence": "recall_evidence",
        "recent_media": "recent_media",
        "scoped_memory": "scoped_memory",
        "subtask": "subtask",
        "synthesis": "synthesis",
    }
    node_kind = aliases.get(normalized, normalized)
    return node_kind


def _artifact_type_from_semantic_text(value: str) -> str:
    """Map stage vocabulary into source-owned artifact types."""

    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "conversation": "conversation_ref",
        "conversation_evidence": "conversation_ref",
        "conversation_ref": "conversation_ref",
        "external": "external_ref",
        "external_evidence": "external_ref",
        "external_ref": "external_ref",
        "live": "live_context_ref",
        "live_context": "live_context_ref",
        "live_context_ref": "live_context_ref",
        "memory": "memory_ref",
        "memory_evidence": "memory_ref",
        "memory_ref": "memory_ref",
        "scoped_memory": "memory_ref",
        "scoped_memory_ref": "memory_ref",
        "user_memory": "memory_ref",
        "user_memory_ref": "memory_ref",
        "user_memory_unit": "memory_ref",
        "user_memory_unit_candidate": "memory_ref",
        "user_memory_units": "memory_ref",
        "user_memory_unit_recall": "memory_ref",
        "user_memory_unit_candidates": "memory_ref",
        "person": "person_ref",
        "person_context": "person_ref",
        "person_ref": "person_ref",
        "profile": "person_ref",
        "third_party_profile": "person_ref",
        "third_party_profiles": "person_ref",
        "user_image": "person_ref",
        "character_image": "person_ref",
        "recall": "recall_ref",
        "recall_evidence": "recall_ref",
        "recall_ref": "recall_ref",
        "semantic_packet": "semantic_packet",
        "media": "media_ref",
        "media_ref": "media_ref",
    }
    artifact_type = aliases.get(normalized, normalized)
    return artifact_type


def _default_options() -> LocalContextResolverOptionsV1:
    """Return centrally defined default structural options."""

    options = {
        "schema_version": LOCAL_CONTEXT_RESOLVER_OPTIONS_VERSION,
        **DEFAULT_OPTION_LIMITS,
    }
    return options


def _option_limits(
    options: LocalContextResolverOptionsV1,
) -> dict[str, int]:
    """Return flat structural limits without schema metadata."""

    limits = {
        "max_iterations": options["max_iterations"],
        "max_nodes": options["max_nodes"],
        "max_depth": options["max_depth"],
        "max_node_attempts": options["max_node_attempts"],
        "max_subagent_attempts": options["max_subagent_attempts"],
    }
    return limits


def _compact_context(
    context: LocalContextResolverContextV1,
) -> dict[str, object]:
    """Return compact prompt-safe caller context for LLM stages."""

    compact = {
        "platform": context["platform"],
        "character_role": "active character",
        "current_user": context["user_name"],
        "local_time_context": _sanitized_prompt_payload(
            context["local_time_context"]
        ),
        "prompt_message_context": _sanitized_prompt_payload(
            context["prompt_message_context"]
        ),
        "chat_history_recent": _sanitized_prompt_payload(
            context["chat_history_recent"]
        ),
        "chat_history_wide": _sanitized_prompt_payload(
            context["chat_history_wide"]
        ),
        "conversation_progress": _sanitized_prompt_payload(
            context["conversation_progress"]
        ),
    }
    original_user_request = context.get("original_user_request")
    if isinstance(original_user_request, str) and original_user_request.strip():
        compact["original_user_request"] = _sanitized_prompt_payload(
            original_user_request
        )
    compact["session_media_aliases"] = _session_media_aliases(context)
    return compact


def _session_media_aliases(
    context: LocalContextResolverContextV1,
) -> list[dict[str, str]]:
    """Project trusted session cache rows into selector-only prompt aliases."""

    raw_refs = context.get("session_media_refs")
    if not isinstance(raw_refs, list):
        return []
    counters = {"current": 0, "recent": 0}
    aliases: list[dict[str, str]] = []
    for raw_ref in reversed(raw_refs):
        if not isinstance(raw_ref, dict):
            continue
        relation = raw_ref.get("turn_relation")
        content_type = raw_ref.get("content_type")
        source_summary = raw_ref.get("source_summary")
        if relation not in counters or not isinstance(content_type, str):
            continue
        counters[relation] += 1
        aliases.append({
            "alias": f"{relation}_media_{counters[relation]}",
            "media_kind": "image",
            "content_type": content_type,
            "turn_relation": relation,
            "source_summary": source_summary if isinstance(source_summary, str) else "",
        })
    return aliases


def _compact_node(node: LocalContextNodeV1) -> dict[str, object]:
    """Return semantic node fields needed by resolver stages."""

    compact = {
        "objective": node["objective"],
        "node_kind": node["node_kind"],
        "status": node["status"],
        "investigation_summary": node["investigation_summary"],
        "knowledge_we_know_so_far": node["knowledge_we_know_so_far"],
        "knowledge_still_lacking": node["knowledge_still_lacking"],
        "recommended_next_iteration": node["recommended_next_iteration"],
        "evidence_boundary_notes": node["evidence_boundary_notes"],
        "attempts": node["attempts"],
    }
    return compact


def _dependency_context(
    graph: LocalContextGraphV1,
    node: LocalContextNodeV1,
) -> list[dict[str, object]]:
    """Return prompt-safe summaries for resolved dependency nodes."""

    rows: list[dict[str, object]] = []
    for dependency_id in node["depends_on"]:
        dependency = graph["nodes"][dependency_id]
        rows.append(_compact_node(dependency))
    return rows


def _collapse_candidates(
    graph: LocalContextGraphV1,
    active_node_id: str,
) -> list[dict[str, object]]:
    """Return resolved same-kind candidate summaries for collapse review."""

    active_node = graph["nodes"][active_node_id]
    candidates: list[dict[str, object]] = []
    for candidate_index, node_id in enumerate(
        _collapse_candidate_node_ids(graph, active_node),
        start=1,
    ):
        node = graph["nodes"][node_id]
        candidate = {
            "candidate_ref": f"candidate_{candidate_index}",
            **_compact_node(node),
        }
        candidates.append(candidate)
    return candidates


def _collapse_candidate_ref_map(
    graph: LocalContextGraphV1,
    active_node_id: str,
) -> dict[str, str]:
    """Return per-call collapse candidate refs mapped to internal node ids."""

    active_node = graph["nodes"][active_node_id]
    candidate_refs: dict[str, str] = {}
    for candidate_index, node_id in enumerate(
        _collapse_candidate_node_ids(graph, active_node),
        start=1,
    ):
        candidate_refs[f"candidate_{candidate_index}"] = node_id
    return candidate_refs


def _collapse_candidate_node_ids(
    graph: LocalContextGraphV1,
    active_node: LocalContextNodeV1,
) -> list[str]:
    """Return resolved same-kind candidate node ids in graph order."""

    node_ids: list[str] = []
    for node_id, node in graph["nodes"].items():
        if node_id == ROOT_NODE_ID:
            continue
        if node_id == active_node["node_id"]:
            continue
        if node["node_kind"] != active_node["node_kind"]:
            continue
        if node["status"] != "resolved":
            continue
        node_ids.append(node_id)
    return node_ids


def _resolved_node_summaries(
    graph: LocalContextGraphV1,
) -> list[dict[str, object]]:
    """Collect resolved node summaries for bottom-up synthesis."""

    rows: list[dict[str, object]] = []
    for node in graph["nodes"].values():
        if node["status"] not in ("resolved", "collapsed"):
            continue
        rows.append(_compact_node(node))
    return rows


def _unresolved_node_summaries(
    graph: LocalContextGraphV1,
) -> list[dict[str, object]]:
    """Collect unresolved node summaries for bottom-up synthesis."""

    rows: list[dict[str, object]] = []
    for node in graph["nodes"].values():
        if node["status"] not in ("pending", "blocked", "cannot_answer"):
            continue
        rows.append(_compact_node(node))
    return rows


def _record_traversal(graph: LocalContextGraphV1, active_node_id: str) -> None:
    """Append one active node id to traversal order once."""

    if active_node_id not in graph["traversal_order"]:
        graph["traversal_order"].append(active_node_id)


def _semantic_text_list(value: object) -> list[str]:
    """Return non-empty semantic text rows from LLM output."""

    if isinstance(value, str):
        text = value.strip()
        return_value = [text] if text else []
        return return_value
    if not isinstance(value, list):
        return []
    rows: list[str] = []
    for item in value:
        if not isinstance(item, str):
            continue
        text = item.strip()
        if not text or text in rows:
            continue
        rows.append(text)
    return rows


def _unique_semantic_rows(rows: list[str]) -> list[str]:
    """Return non-empty semantic rows without duplicates."""

    unique_rows: list[str] = []
    for row in rows:
        if not isinstance(row, str):
            continue
        text = row.strip()
        if not text or text in unique_rows:
            continue
        unique_rows.append(text)
    return unique_rows


def _resolved_node_summary_rows(graph: LocalContextGraphV1) -> list[str]:
    """Collect resolved node investigation summaries."""

    rows: list[str] = []
    for node in graph["nodes"].values():
        if node["status"] not in ("resolved", "collapsed"):
            continue
        rows.extend(node["investigation_summary"])
    return rows


def _resolved_node_known_rows(graph: LocalContextGraphV1) -> list[str]:
    """Collect resolved node knowledge rows."""

    rows: list[str] = []
    for node in graph["nodes"].values():
        if node["status"] not in ("resolved", "collapsed"):
            continue
        rows.extend(node["knowledge_we_know_so_far"])
    return rows


def _unresolved_node_lacking_rows(graph: LocalContextGraphV1) -> list[str]:
    """Collect unresolved node knowledge gaps."""

    rows: list[str] = []
    for node in graph["nodes"].values():
        if node["status"] not in ("pending", "blocked", "cannot_answer"):
            continue
        rows.extend(node["knowledge_still_lacking"])
        if not node["knowledge_still_lacking"]:
            rows.append(f"unresolved local-context node: {node['objective']}")
    return rows


def _node_boundary_note_rows(graph: LocalContextGraphV1) -> list[str]:
    """Collect node-authored boundary notes."""

    rows: list[str] = []
    for node in graph["nodes"].values():
        rows.extend(node["evidence_boundary_notes"])
    return rows


def _max_depth_observed(graph: LocalContextGraphV1) -> int:
    """Return maximum depth observed by walking parent links."""

    max_depth = ROOT_NODE_DEPTH
    for node_id in graph["nodes"]:
        depth = _node_depth(graph, node_id)
        if depth > max_depth:
            max_depth = depth
    return max_depth


def _node_depth(graph: LocalContextGraphV1, node_id: str) -> int:
    """Return one node's depth by following parents to the root."""

    depth = ROOT_NODE_DEPTH
    current_node = graph["nodes"][node_id]
    parent_id = current_node["parent_id"]
    while parent_id is not None:
        depth += ROOT_CHILD_DEPTH
        current_node = graph["nodes"][parent_id]
        parent_id = current_node["parent_id"]
    return depth


def _objective_from_request(request: object) -> str:
    """Read a safe objective from an unvalidated request."""

    if isinstance(request, dict):
        objective = request.get("objective")
        if isinstance(objective, str) and objective.strip():
            return_value = _safe_failure_text(objective)
            return return_value
    return_value = "invalid local-context request"
    return return_value


def _safe_failure_text(value: str) -> str:
    """Return a compact single-line failure string for public packets."""

    collapsed = " ".join(value.strip().split())
    if not collapsed:
        return "unknown local-context resolver failure"
    if len(collapsed) > SAFE_FAILURE_TEXT_LIMIT:
        collapsed = (
            collapsed[: SAFE_FAILURE_TEXT_LIMIT - len(TEXT_ELLIPSIS)].rstrip()
            + TEXT_ELLIPSIS
        )
    return collapsed
