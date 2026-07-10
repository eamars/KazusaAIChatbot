"""Standalone local-context resolver contracts and validators."""

from __future__ import annotations

import copy
from typing import Literal, NotRequired, Protocol, TypedDict

from .constants import (
    DEFAULT_SUBAGENT_MAX_ATTEMPTS,
    OPTION_LIMIT_CAPS,
)

LOCAL_CONTEXT_ARTIFACT_VERSION = "local_context_artifact.v1"
LOCAL_CONTEXT_GRAPH_VERSION = "local_context_graph.v1"
LOCAL_CONTEXT_NODE_VERSION = "local_context_node.v1"
LOCAL_CONTEXT_RESOLUTION_PACKET_VERSION = "local_context_resolution_packet.v1"
LOCAL_CONTEXT_RESOLVER_CONTEXT_VERSION = "local_context_resolver_context.v1"
LOCAL_CONTEXT_RESOLVER_OPTIONS_VERSION = "local_context_resolver_options.v1"
LOCAL_CONTEXT_RESOLVER_REQUEST_VERSION = "local_context_resolver_request.v1"
LOCAL_CONTEXT_SUBAGENT_REQUEST_VERSION = "local_context_subagent_request.v1"
LOCAL_CONTEXT_SUBAGENT_RESULT_VERSION = "local_context_subagent_result.v1"

ALLOWED_ARTIFACT_TYPES = frozenset((
    "conversation_ref",
    "external_ref",
    "live_context_ref",
    "media_ref",
    "memory_ref",
    "person_ref",
    "recall_ref",
    "semantic_packet",
))
ALLOWED_NODE_KINDS = frozenset((
    "conversation_evidence",
    "current_turn_media",
    "external_evidence",
    "live_context",
    "memory_evidence",
    "person_context",
    "recall_evidence",
    "recent_media",
    "scoped_memory",
    "subtask",
    "synthesis",
))
ALLOWED_NODE_STATUSES = frozenset((
    "pending",
    "resolving",
    "resolved",
    "blocked",
    "cannot_answer",
    "collapsed",
))
ALLOWED_REQUEST_PRIORITIES = frozenset(("normal", "high"))
ALLOWED_REQUEST_SOURCES = frozenset((
    "l2d",
    "live_llm_review",
    "prewarm",
    "standalone_eval",
    "test",
))
ALLOWED_SUBAGENT_STATUSES = frozenset((
    "resolved",
    "partial",
    "invalid",
    "unavailable",
    "failed",
))
ALLOWED_OPTION_LIMITS = frozenset((
    "max_iterations",
    "max_nodes",
    "max_depth",
    "max_node_attempts",
    "max_subagent_attempts",
))
FORBIDDEN_OPTION_FIELDS = frozenset((
    "planner_llm",
    "node_resolver_llm",
    "collapse_llm",
    "synthesizer_llm",
    "subagents",
    "clock",
))


class LocalContextValidationError(ValueError):
    """Raised when a local-context resolver contract is structurally invalid."""


class LocalContextResolverRequestV1(TypedDict):
    """Stable public request for standalone and production resolver calls."""

    schema_version: Literal["local_context_resolver_request.v1"]
    objective: str
    source: Literal[
        "standalone_eval",
        "l2d",
        "prewarm",
        "test",
        "live_llm_review",
    ]
    reason: str
    priority: Literal["normal", "high"]


class LocalContextResolverContextV1(TypedDict):
    """Prompt-safe local context supplied by the caller."""

    schema_version: Literal["local_context_resolver_context.v1"]
    character_name: str
    platform: str
    platform_channel_id: str
    global_user_id: str
    user_name: str
    local_time_context: dict[str, object]
    prompt_message_context: dict[str, object]
    chat_history_recent: list[dict[str, object]]
    chat_history_wide: list[dict[str, object]]
    conversation_progress: dict[str, object]
    original_user_request: NotRequired[str]
    current_timestamp_utc: NotRequired[str]
    current_platform_message_id: NotRequired[str]
    active_turn_platform_message_ids: NotRequired[list[str]]
    active_turn_conversation_row_ids: NotRequired[list[str]]
    session_media_refs: NotRequired[list[dict[str, object]]]


class LocalContextResolverOptionsV1(TypedDict):
    """Flat public structural limits for resolver execution."""

    schema_version: Literal["local_context_resolver_options.v1"]
    max_iterations: int
    max_nodes: int
    max_depth: int
    max_node_attempts: int
    max_subagent_attempts: int


class LocalContextNodeV1(TypedDict):
    """One deterministic node in a bounded local-context graph."""

    schema_version: Literal["local_context_node.v1"]
    node_id: str
    node_kind: Literal[
        "conversation_evidence",
        "current_turn_media",
        "external_evidence",
        "live_context",
        "memory_evidence",
        "person_context",
        "recall_evidence",
        "recent_media",
        "scoped_memory",
        "subtask",
        "synthesis",
    ]
    objective: str
    parent_id: str | None
    children: list[str]
    depends_on: list[str]
    consumes: dict[str, str]
    produces: list[str]
    status: Literal[
        "pending",
        "resolving",
        "resolved",
        "blocked",
        "cannot_answer",
        "collapsed",
    ]
    investigation_summary: list[str]
    knowledge_we_know_so_far: list[str]
    knowledge_still_lacking: list[str]
    recommended_next_iteration: list[str]
    evidence_boundary_notes: list[str]
    attempts: list[dict[str, object]]
    collapsed_into: str | None


class LocalContextGraphV1(TypedDict):
    """Bounded local-context graph with deterministic traversal metadata."""

    schema_version: Literal["local_context_graph.v1"]
    root_node_id: str
    active_node_id: str
    nodes: dict[str, LocalContextNodeV1]
    traversal_order: list[str]
    collapse_events: list[dict[str, str]]
    max_nodes: int
    max_depth: int


class LocalContextArtifactV1(TypedDict):
    """Prompt-visible or trace-only source-owned evidence artifact."""

    schema_version: Literal["local_context_artifact.v1"]
    artifact_id: str
    artifact_type: Literal[
        "conversation_ref",
        "external_ref",
        "live_context_ref",
        "memory_ref",
        "person_ref",
        "recall_ref",
        "semantic_packet",
    ]
    producer_node_id: str
    summary: str
    projection_payload: dict[str, object]
    source_policy: str
    prompt_visible: bool


class LocalContextResolutionPacketV1(TypedDict):
    """Prompt-safe local evidence packet returned by the resolver."""

    schema_version: Literal["local_context_resolution_packet.v1"]
    investigation_summary: list[str]
    knowledge_we_know_so_far: list[str]
    knowledge_still_lacking: list[str]
    recommended_next_iteration: list[str]
    evidence_boundary_notes: list[str]
    rag_result: dict[str, object]
    graph: LocalContextGraphV1
    trace_summary: dict[str, object]


class LocalContextSubagentRequestV1(TypedDict):
    """Bounded request sent to one resolver-local source handler."""

    schema_version: Literal["local_context_subagent_request.v1"]
    node_id: str
    subagent: str
    action: str
    objective: str
    payload: dict[str, object]
    constraints: dict[str, object]


class LocalContextSubagentResultV1(TypedDict):
    """Small source-owned result envelope returned by local-context handlers."""

    schema_version: Literal["local_context_subagent_result.v1"]
    resolved: bool
    status: Literal["resolved", "partial", "invalid", "unavailable", "failed"]
    result: dict[str, object]
    attempts: int
    cache: dict[str, object]
    trace: dict[str, object] | list[str]
    unresolved_items: list[str]


class LocalContextSubagentV1(Protocol):
    """Resolver-local subagent protocol for future source-owned handlers."""

    async def run(
        self,
        task: LocalContextSubagentRequestV1,
        context: dict[str, object],
        max_attempts: int = DEFAULT_SUBAGENT_MAX_ATTEMPTS,
    ) -> LocalContextSubagentResultV1:
        """Return one bounded source result envelope."""


def validate_local_context_resolver_request(
    value: object,
) -> LocalContextResolverRequestV1:
    """Validate the public resolver request shape."""

    data = _require_mapping(value, "local_context_resolver_request")
    _require_version(data, LOCAL_CONTEXT_RESOLVER_REQUEST_VERSION)
    _require_non_empty_string(data, "objective")
    _require_enum(data, "source", ALLOWED_REQUEST_SOURCES)
    _require_non_empty_string(data, "reason")
    _require_enum(data, "priority", ALLOWED_REQUEST_PRIORITIES)
    return_value = data
    return return_value


def validate_local_context_resolver_context(
    value: object,
) -> LocalContextResolverContextV1:
    """Validate prompt-safe caller context for a resolver run."""

    data = _require_mapping(value, "local_context_resolver_context")
    _require_version(data, LOCAL_CONTEXT_RESOLVER_CONTEXT_VERSION)
    for field_name in (
        "character_name",
        "platform",
        "platform_channel_id",
        "global_user_id",
        "user_name",
    ):
        _require_string(data, field_name)
    for field_name in (
        "local_time_context",
        "prompt_message_context",
        "conversation_progress",
    ):
        _require_dict(data, field_name)
    for field_name in ("chat_history_recent", "chat_history_wide"):
        _require_list(data, field_name)
    for field_name in (
        "current_timestamp_utc",
        "current_platform_message_id",
        "original_user_request",
    ):
        if field_name in data:
            _require_string(data, field_name)
    for field_name in (
        "active_turn_platform_message_ids",
        "active_turn_conversation_row_ids",
    ):
        if field_name in data:
            _validate_string_list(
                _require_list(data, field_name),
                field_name,
                allow_empty_strings=False,
            )
    if "session_media_refs" in data:
        _require_list(data, "session_media_refs")
    return_value = data
    return return_value


def validate_local_context_resolver_options(
    value: object,
) -> LocalContextResolverOptionsV1:
    """Validate flat structural resolver options and reject behavior hooks."""

    data = _require_mapping(value, "local_context_resolver_options")
    _require_version(data, LOCAL_CONTEXT_RESOLVER_OPTIONS_VERSION)
    for field_name in data:
        if field_name in FORBIDDEN_OPTION_FIELDS:
            raise LocalContextValidationError(
                f"{field_name}: behavior injection is not accepted"
            )
        if field_name != "schema_version" and field_name not in ALLOWED_OPTION_LIMITS:
            raise LocalContextValidationError(
                f"{field_name}: unsupported resolver option"
            )
    for limit_name in ALLOWED_OPTION_LIMITS:
        limit_value = _require_positive_integer(data, limit_name)
        if limit_value > OPTION_LIMIT_CAPS[limit_name]:
            raise LocalContextValidationError(
                f"{limit_name}: exceeds cap {OPTION_LIMIT_CAPS[limit_name]}"
            )
    return_value = data
    return return_value


def validate_local_context_node(value: object) -> LocalContextNodeV1:
    """Validate one local-context graph node without graph-wide checks."""

    data = _require_mapping(value, "local_context_node")
    _require_version(data, LOCAL_CONTEXT_NODE_VERSION)
    _require_non_empty_string(data, "node_id")
    _require_enum(data, "node_kind", ALLOWED_NODE_KINDS)
    _require_non_empty_string(data, "objective")
    _require_nullable_non_empty_string(data, "parent_id")
    _validate_string_list(
        _require_list(data, "children"),
        "children",
        allow_empty_strings=False,
    )
    _validate_string_list(
        _require_list(data, "depends_on"),
        "depends_on",
        allow_empty_strings=False,
    )
    _validate_string_dict(_require_dict(data, "consumes"), "consumes")
    _validate_string_list(
        _require_list(data, "produces"),
        "produces",
        allow_empty_strings=False,
    )
    status = _require_enum(data, "status", ALLOWED_NODE_STATUSES)
    for field_name in (
        "investigation_summary",
        "knowledge_we_know_so_far",
        "knowledge_still_lacking",
        "recommended_next_iteration",
        "evidence_boundary_notes",
    ):
        _validate_string_list(
            _require_list(data, field_name),
            field_name,
            allow_empty_strings=False,
        )
    _require_list(data, "attempts")
    collapsed_into = _require_nullable_non_empty_string(data, "collapsed_into")
    if status == "collapsed" and collapsed_into is None:
        raise LocalContextValidationError("collapsed_into: required")
    if status != "collapsed" and collapsed_into is not None:
        raise LocalContextValidationError("collapsed_into: unexpected")
    return_value = data
    return return_value


def validate_local_context_artifact(value: object) -> LocalContextArtifactV1:
    """Validate one prompt-visible or trace-only evidence artifact."""

    data = _require_mapping(value, "local_context_artifact")
    _require_version(data, LOCAL_CONTEXT_ARTIFACT_VERSION)
    _require_non_empty_string(data, "artifact_id")
    _require_enum(data, "artifact_type", ALLOWED_ARTIFACT_TYPES)
    _require_non_empty_string(data, "producer_node_id")
    _require_string(data, "summary")
    _require_dict(data, "projection_payload")
    _require_non_empty_string(data, "source_policy")
    prompt_visible = data.get("prompt_visible")
    if not isinstance(prompt_visible, bool):
        raise LocalContextValidationError("prompt_visible: expected boolean")
    return_value = data
    return return_value


def validate_local_context_graph(value: object) -> LocalContextGraphV1:
    """Validate graph-wide references, dependencies, cycles, and depth."""

    data = _require_mapping(value, "local_context_graph")
    _require_version(data, LOCAL_CONTEXT_GRAPH_VERSION)
    root_node_id = _require_non_empty_string(data, "root_node_id")
    active_node_id = _require_non_empty_string(data, "active_node_id")
    nodes = _require_dict(data, "nodes")
    max_nodes = _require_positive_integer(data, "max_nodes")
    max_depth = _require_positive_integer(data, "max_depth")
    if len(nodes) > max_nodes:
        raise LocalContextValidationError("nodes: exceeds max_nodes")
    if root_node_id not in nodes:
        raise LocalContextValidationError("root_node_id: missing node")
    if active_node_id not in nodes:
        raise LocalContextValidationError("active_node_id: missing node")
    for map_node_id, raw_node in nodes.items():
        node = validate_local_context_node(raw_node)
        if node["node_id"] != map_node_id:
            raise LocalContextValidationError("nodes: key must match node_id")
    traversal_order = _require_list(data, "traversal_order")
    _validate_string_list(
        traversal_order,
        "traversal_order",
        allow_empty_strings=False,
    )
    _validate_graph_references(nodes, root_node_id, traversal_order)
    _validate_graph_is_acyclic(nodes, root_node_id)
    _validate_graph_dependencies(nodes)
    _validate_graph_depth(nodes, root_node_id, max_depth)
    _validate_collapse_events(_require_list(data, "collapse_events"), nodes)
    return_value = data
    return return_value


def validate_local_context_resolution_packet(
    value: object,
) -> LocalContextResolutionPacketV1:
    """Validate the public resolver packet returned to callers."""

    data = _require_mapping(value, "local_context_resolution_packet")
    _require_version(data, LOCAL_CONTEXT_RESOLUTION_PACKET_VERSION)
    for field_name in (
        "investigation_summary",
        "knowledge_we_know_so_far",
        "knowledge_still_lacking",
        "recommended_next_iteration",
        "evidence_boundary_notes",
    ):
        _validate_string_list(
            _require_list(data, field_name),
            field_name,
            allow_empty_strings=False,
        )
    _validate_rag_result(_require_dict(data, "rag_result"))
    validate_local_context_graph(data.get("graph"))
    _require_dict(data, "trace_summary")
    return_value = data
    return return_value


def validate_local_context_subagent_request(
    value: object,
) -> LocalContextSubagentRequestV1:
    """Validate a bounded local-context subagent request."""

    data = _require_mapping(value, "local_context_subagent_request")
    _require_version(data, LOCAL_CONTEXT_SUBAGENT_REQUEST_VERSION)
    _require_non_empty_string(data, "node_id")
    _require_non_empty_string(data, "subagent")
    _require_non_empty_string(data, "action")
    _require_non_empty_string(data, "objective")
    _require_dict(data, "payload")
    _require_dict(data, "constraints")
    return_value = data
    return return_value


def validate_local_context_subagent_result(
    value: object,
) -> LocalContextSubagentResultV1:
    """Validate a bounded local-context subagent result."""

    data = _require_mapping(value, "local_context_subagent_result")
    _require_version(data, LOCAL_CONTEXT_SUBAGENT_RESULT_VERSION)
    resolved = data.get("resolved")
    if not isinstance(resolved, bool):
        raise LocalContextValidationError("resolved: expected boolean")
    _require_enum(data, "status", ALLOWED_SUBAGENT_STATUSES)
    _require_dict(data, "result")
    attempts = _require_integer(data, "attempts")
    if attempts < 0:
        raise LocalContextValidationError("attempts: expected non-negative integer")
    _require_dict(data, "cache")
    trace = data.get("trace")
    if not isinstance(trace, (dict, list)):
        raise LocalContextValidationError("trace: expected object or list")
    _validate_string_list(
        _require_list(data, "unresolved_items"),
        "unresolved_items",
        allow_empty_strings=False,
    )
    status = data["status"]
    if status == "resolved" and resolved is not True:
        raise LocalContextValidationError("resolved: required for resolved status")
    if status in ("invalid", "unavailable", "failed") and resolved is True:
        raise LocalContextValidationError("resolved: forbidden for terminal failure")
    return_value = data
    return return_value


def project_local_context_packet(packet: object) -> dict[str, object]:
    """Return only the prompt-facing ``rag_result`` from a resolver packet."""

    validated_packet = validate_local_context_resolution_packet(packet)
    rag_result = copy.deepcopy(validated_packet["rag_result"])
    rag_result.pop("graph", None)
    rag_result.pop("trace_summary", None)
    return_value = rag_result
    return return_value


def _require_mapping(value: object, label: str) -> dict:
    """Return a dictionary payload or raise a contract error."""

    if not isinstance(value, dict):
        raise LocalContextValidationError(f"{label}: expected object")
    return_value = value
    return return_value


def _require_version(data: dict, expected: str) -> None:
    """Require a specific schema version on one contract object."""

    actual = data.get("schema_version")
    if actual != expected:
        raise LocalContextValidationError(f"schema_version: expected {expected}")


def _require_string(data: dict, field_name: str) -> str:
    """Require one string field, allowing the empty string."""

    value = data.get(field_name)
    if not isinstance(value, str):
        raise LocalContextValidationError(f"{field_name}: expected string")
    return_value = value
    return return_value


def _require_non_empty_string(data: dict, field_name: str) -> str:
    """Require one non-empty string field."""

    value = data.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise LocalContextValidationError(
            f"{field_name}: expected non-empty string"
        )
    return_value = value
    return return_value


def _require_nullable_non_empty_string(
    data: dict,
    field_name: str,
) -> str | None:
    """Require one field to be null or a non-empty string."""

    value = data.get(field_name)
    if value is None:
        return_value = None
        return return_value
    if not isinstance(value, str) or not value.strip():
        raise LocalContextValidationError(
            f"{field_name}: expected non-empty string"
        )
    return_value = value
    return return_value


def _require_enum(data: dict, field_name: str, allowed: frozenset[str]) -> str:
    """Require one field to belong to a closed vocabulary."""

    value = data.get(field_name)
    if not isinstance(value, str) or value not in allowed:
        expected = sorted(allowed)
        raise LocalContextValidationError(f"{field_name}: expected one of {expected}")
    return_value = value
    return return_value


def _require_list(data: dict, field_name: str) -> list:
    """Require one list field."""

    value = data.get(field_name)
    if not isinstance(value, list):
        raise LocalContextValidationError(f"{field_name}: expected list")
    return_value = value
    return return_value


def _require_dict(data: dict, field_name: str) -> dict:
    """Require one object field."""

    value = data.get(field_name)
    if not isinstance(value, dict):
        raise LocalContextValidationError(f"{field_name}: expected object")
    return_value = value
    return return_value


def _require_integer(data: dict, field_name: str) -> int:
    """Require one integer field."""

    value = data.get(field_name)
    if not isinstance(value, int) or isinstance(value, bool):
        raise LocalContextValidationError(f"{field_name}: expected integer")
    return_value = value
    return return_value


def _require_positive_integer(data: dict, field_name: str) -> int:
    """Require one positive integer field."""

    value = _require_integer(data, field_name)
    if value < 1:
        raise LocalContextValidationError(
            f"{field_name}: expected positive integer"
        )
    return_value = value
    return return_value


def _validate_string_list(
    values: list,
    field_name: str,
    *,
    allow_empty_strings: bool,
) -> None:
    """Validate that a list contains only strings."""

    for item in values:
        if not isinstance(item, str):
            raise LocalContextValidationError(f"{field_name}: expected strings")
        if not allow_empty_strings and not item.strip():
            raise LocalContextValidationError(
                f"{field_name}: expected non-empty strings"
            )


def _validate_string_dict(values: dict, field_name: str) -> None:
    """Validate that a dictionary contains string keys and values."""

    for key, value in values.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise LocalContextValidationError(
                f"{field_name}: expected string key/value pairs"
            )


def _validate_rag_result(rag_result: dict) -> None:
    """Validate the retained prompt-facing RAG result top-level shape."""

    _require_string(rag_result, "answer")
    for field_name in ("user_image", "character_image", "supervisor_trace"):
        _require_dict(rag_result, field_name)
    for field_name in (
        "user_memory_unit_candidates",
        "third_party_profiles",
        "memory_evidence",
        "recall_evidence",
        "conversation_evidence",
        "external_evidence",
    ):
        _require_list(rag_result, field_name)
    if "media_evidence" in rag_result:
        _require_list(rag_result, "media_evidence")


def _validate_graph_references(
    nodes: dict[str, LocalContextNodeV1],
    root_node_id: str,
    traversal_order: list[str],
) -> None:
    """Validate parent, child, collapse, and traversal references."""

    root_node = nodes[root_node_id]
    if root_node["parent_id"] is not None:
        raise LocalContextValidationError("root_node_id: root parent must be null")
    for node_id, node in nodes.items():
        parent_id = node["parent_id"]
        if parent_id is not None and parent_id not in nodes:
            raise LocalContextValidationError("parent_id: missing node")
        for child_id in node["children"]:
            if child_id not in nodes:
                raise LocalContextValidationError("children: missing node")
            child = nodes[child_id]
            if child["parent_id"] != node_id:
                raise LocalContextValidationError("children: child parent mismatch")
        collapsed_into = node["collapsed_into"]
        if collapsed_into is not None:
            if collapsed_into not in nodes:
                raise LocalContextValidationError("collapsed_into: missing node")
            if collapsed_into == node_id:
                raise LocalContextValidationError("collapsed_into: cannot target self")
    for traversal_node_id in traversal_order:
        if traversal_node_id not in nodes:
            raise LocalContextValidationError("traversal_order: missing node")


def _validate_graph_is_acyclic(
    nodes: dict[str, LocalContextNodeV1],
    root_node_id: str,
) -> None:
    """Reject cycles in child links reachable from the root."""

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node_id: str) -> None:
        if node_id in visiting:
            raise LocalContextValidationError("nodes: cycle detected")
        if node_id in visited:
            return
        visiting.add(node_id)
        node = nodes[node_id]
        for child_id in node["children"]:
            visit(child_id)
        visiting.remove(node_id)
        visited.add(node_id)

    visit(root_node_id)


def _validate_graph_dependencies(
    nodes: dict[str, LocalContextNodeV1],
) -> None:
    """Validate that dependency references point to existing nodes."""

    for node in nodes.values():
        for dependency_id in node["depends_on"]:
            if dependency_id not in nodes:
                raise LocalContextValidationError("depends_on: missing node")


def _validate_graph_depth(
    nodes: dict[str, LocalContextNodeV1],
    root_node_id: str,
    max_depth: int,
) -> None:
    """Validate depth by walking child links from the root."""

    def visit(node_id: str, depth: int) -> None:
        if depth > max_depth:
            raise LocalContextValidationError("depth: exceeds max_depth")
        node = nodes[node_id]
        for child_id in node["children"]:
            visit(child_id, depth + 1)

    visit(root_node_id, 0)


def _validate_collapse_events(
    collapse_events: list,
    nodes: dict[str, LocalContextNodeV1],
) -> None:
    """Validate collapse event references and concise reasons."""

    for event in collapse_events:
        if not isinstance(event, dict):
            raise LocalContextValidationError("collapse_events: expected objects")
        from_node_id = _require_non_empty_string(event, "from_node_id")
        to_node_id = _require_non_empty_string(event, "to_node_id")
        _require_non_empty_string(event, "reason")
        if from_node_id not in nodes or to_node_id not in nodes:
            raise LocalContextValidationError("collapse_events: missing node")
