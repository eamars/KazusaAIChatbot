"""Standalone complex-task resolver contracts and validators."""

from __future__ import annotations

from typing import Literal, Protocol, TypedDict

from kazusa_ai_chatbot.action_spec.models import (
    ActionValidationError,
    EvidenceRefV1,
    validate_evidence_ref,
)

COMPLEX_TASK_GRAPH_VERSION = "complex_task_graph.v1"
COMPLEX_TASK_FOLLOWUP_TASK_VERSION = "complex_task_followup_task.v1"
COMPLEX_TASK_NODE_ATTEMPT_VERSION = "complex_task_node_attempt.v1"
COMPLEX_TASK_NODE_VERSION = "complex_task_node.v1"
COMPLEX_TASK_RESOLUTION_PACKET_VERSION = "complex_task_resolution_packet.v1"
COMPLEX_TASK_RESOLVER_CONTEXT_VERSION = "complex_task_resolver_context.v1"
COMPLEX_TASK_RESOLVER_OPTIONS_VERSION = "complex_task_resolver_options.v1"
COMPLEX_TASK_RESOLVER_REQUEST_VERSION = "complex_task_resolver_request.v1"
COMPLEX_TASK_SUBAGENT_REQUEST_VERSION = "complex_task_subagent_request.v1"
COMPLEX_TASK_SUBAGENT_RESULT_VERSION = "complex_task_subagent_result.v1"

ALLOWED_REQUEST_SOURCES = frozenset((
    "test",
    "review_harness",
    "live_llm_review",
    "l2d",
))
ALLOWED_REQUEST_PRIORITIES = frozenset(("normal", "review"))
ALLOWED_NODE_KINDS = frozenset(
    ("root", "subtask", "evidence_need", "algorithmic_task", "synthesis")
)
ALLOWED_FOLLOWUP_TASK_KINDS = frozenset(
    ("subtask", "evidence_need", "algorithmic_task", "synthesis")
)
ALLOWED_NODE_STATUSES = frozenset(
    (
        "pending",
        "expanded",
        "resolving",
        "resolved",
        "blocked",
        "cannot_answer",
        "collapsed",
    )
)
ALLOWED_NODE_ATTEMPT_ACTIONS = frozenset(
    (
        "resolve_direct",
        "expand_node",
        "call_subagent",
        "refine_search",
        "disambiguate_entity",
        "repair_subagent_request",
        "revise_calculation_request",
        "review_source_conflict",
        "synthesize_partial",
        "ask_user_input",
        "block",
    )
)
ALLOWED_NODE_ATTEMPT_STATUSES = frozenset(
    ("planned", "resolved", "partial", "blocked", "cannot_answer", "invalid")
)
ALLOWED_SUBAGENT_STATUSES = frozenset(
    ("resolved", "partial", "invalid", "unavailable", "failed")
)
ALLOWED_OPTION_LIMITS = frozenset((
    "max_iterations",
    "max_nodes",
    "max_depth",
    "max_node_attempts",
    "max_subagent_attempts",
))
OPTION_LIMIT_CAPS = {
    "max_iterations": 8,
    "max_nodes": 8,
    "max_depth": 3,
    "max_node_attempts": 3,
    "max_subagent_attempts": 1,
}
FORBIDDEN_OPTION_FIELDS = frozenset((
    "planner_llm",
    "node_resolver_llm",
    "collapse_llm",
    "synthesizer_llm",
    "subagents",
    "clock",
))

_FORBIDDEN_HINT_FRAGMENT_SETS = (
    ("case", "id"),
    ("expected", "final", "answer"),
    ("minimum", "viable", "answer"),
    ("expected", "graph", "trace"),
    ("expected", "status"),
    ("performance", "reference", "summary"),
    ("forbidden", "failure", "modes"),
)


class ComplexTaskValidationError(ValueError):
    """Raised when a complex-task resolver contract is structurally invalid."""


class ComplexTaskResolverRequestV1(TypedDict):
    """Standalone resolver request supplied by tests or review harnesses."""

    schema_version: Literal["complex_task_resolver_request.v1"]
    objective: str
    reason: str
    source: Literal["test", "review_harness", "live_llm_review", "l2d"]
    priority: Literal["normal", "review"]


class ComplexTaskResolverContextV1(TypedDict):
    """Compact standalone context for resolver review runs."""

    schema_version: Literal["complex_task_resolver_context.v1"]
    conversation_summary: str
    persona_context_summary: str
    time_context: dict[str, object]
    available_evidence: list[EvidenceRefV1]


class ComplexTaskNodeAttemptV1(TypedDict):
    """Prompt-safe observation for one active-node resolution attempt."""

    schema_version: Literal["complex_task_node_attempt.v1"]
    attempt_index: int
    action: Literal[
        "resolve_direct",
        "expand_node",
        "call_subagent",
        "refine_search",
        "disambiguate_entity",
        "repair_subagent_request",
        "revise_calculation_request",
        "review_source_conflict",
        "synthesize_partial",
        "ask_user_input",
        "block",
    ]
    status: Literal[
        "planned",
        "resolved",
        "partial",
        "blocked",
        "cannot_answer",
        "invalid",
    ]
    input_summary: str
    result_summary: str
    blockers: list[str]
    next_action: str


class ComplexTaskFollowupTaskV1(TypedDict):
    """Resolver-internal executable follow-up task emitted by local stages."""

    schema_version: Literal["complex_task_followup_task.v1"]
    objective: str
    kind: Literal["subtask", "evidence_need", "algorithmic_task", "synthesis"]
    reason: str


class ComplexTaskNodeV1(TypedDict):
    """One stable node in a bounded complex-task graph."""

    schema_version: Literal["complex_task_node.v1"]
    node_id: str
    parent_id: str | None
    depth: int
    objective: str
    node_kind: Literal[
        "root",
        "subtask",
        "evidence_need",
        "algorithmic_task",
        "synthesis",
    ]
    status: Literal[
        "pending",
        "expanded",
        "resolving",
        "resolved",
        "blocked",
        "cannot_answer",
        "collapsed",
    ]
    children: list[str]
    investigation_summary: str
    knowledge_we_know_so_far: list[str]
    knowledge_still_lacking: list[str]
    recommended_next_iteration: list[str]
    evidence_boundary_notes: list[str]
    evidence_refs: list[EvidenceRefV1]
    source_observation_ids: list[str]
    collapsed_into: str | None
    attempts: list[ComplexTaskNodeAttemptV1]


class ComplexTaskGraphV1(TypedDict):
    """Bounded graph of task nodes with deterministic traversal metadata."""

    schema_version: Literal["complex_task_graph.v1"]
    root_node_id: str
    active_node_id: str
    nodes: dict[str, ComplexTaskNodeV1]
    traversal_order: list[str]
    collapse_events: list[dict[str, str]]
    max_nodes: int
    max_depth: int


class ComplexTaskResolutionPacketV1(TypedDict):
    """Prompt-safe investigation packet, not final character dialog."""

    schema_version: Literal["complex_task_resolution_packet.v1"]
    root_question: str
    investigation_summary: str
    knowledge_we_know_so_far: list[str]
    knowledge_still_lacking: list[str]
    recommended_next_iteration: list[str]
    evidence_boundary_notes: list[str]
    graph: ComplexTaskGraphV1
    trace_summary: dict[str, object]


class ComplexTaskSubagentRequestV1(TypedDict):
    """Bounded request sent to one resolver-local subagent."""

    schema_version: Literal["complex_task_subagent_request.v1"]
    node_id: str
    subagent: str
    action: str
    objective: str
    payload: dict[str, object]
    constraints: dict[str, object]


class ComplexTaskSubagentResultV1(TypedDict):
    """Small resolved/result envelope returned by resolver subagents."""

    schema_version: Literal["complex_task_subagent_result.v1"]
    resolved: bool
    status: Literal["resolved", "partial", "invalid", "unavailable", "failed"]
    result: dict[str, object]
    attempts: int
    cache: dict[str, object]
    trace: dict[str, object] | list[str]
    unresolved_items: list[str]


class ComplexTaskSubagentV1(Protocol):
    """Resolver-local subagent interface used by active task nodes."""

    async def run(
        self,
        task: ComplexTaskSubagentRequestV1,
        context: dict[str, object],
        max_attempts: int = 1,
    ) -> ComplexTaskSubagentResultV1:
        """Return one bounded subagent result envelope."""


class ComplexTaskResolverOptionsV1(TypedDict):
    """Public structural resolver limits for standalone calls."""

    schema_version: Literal["complex_task_resolver_options.v1"]
    limits: dict[str, int]


def validate_complex_task_resolver_request(
    value: object,
) -> ComplexTaskResolverRequestV1:
    """Validate a standalone resolver request."""

    data = _require_mapping(value, "complex_task_resolver_request")
    _require_version(data, COMPLEX_TASK_RESOLVER_REQUEST_VERSION)
    _require_non_empty_string(data, "objective")
    _require_non_empty_string(data, "reason")
    _require_enum(data, "source", ALLOWED_REQUEST_SOURCES)
    _require_enum(data, "priority", ALLOWED_REQUEST_PRIORITIES)
    return_value = data
    return return_value


def validate_complex_task_resolver_context(
    value: object,
) -> ComplexTaskResolverContextV1:
    """Validate compact context supplied to standalone resolver runs."""

    data = _require_mapping(value, "complex_task_resolver_context")
    _require_version(data, COMPLEX_TASK_RESOLVER_CONTEXT_VERSION)
    _require_string(data, "conversation_summary")
    _require_string(data, "persona_context_summary")
    time_context = data.get("time_context")
    if not isinstance(time_context, dict):
        raise ComplexTaskValidationError("time_context: expected object")
    evidence_refs = _require_list(data, "available_evidence")
    _validate_evidence_refs(evidence_refs, "available_evidence")
    return_value = data
    return return_value


def validate_complex_task_node(value: object) -> ComplexTaskNodeV1:
    """Validate one complex-task graph node without graph-wide checks."""

    data = _require_mapping(value, "complex_task_node")
    _require_version(data, COMPLEX_TASK_NODE_VERSION)
    _require_non_empty_string(data, "node_id")
    _require_nullable_non_empty_string(data, "parent_id")
    depth = _require_integer(data, "depth")
    if depth < 0:
        raise ComplexTaskValidationError("depth: expected non-negative integer")
    _require_non_empty_string(data, "objective")
    _require_enum(data, "node_kind", ALLOWED_NODE_KINDS)
    status = _require_enum(data, "status", ALLOWED_NODE_STATUSES)
    children = _require_list(data, "children")
    _validate_string_list(children, "children", allow_empty_strings=False)
    _require_string(data, "investigation_summary")
    for field_name in (
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
    evidence_refs = _require_list(data, "evidence_refs")
    _validate_evidence_refs(evidence_refs, "evidence_refs")
    _validate_string_list(
        _require_list(data, "source_observation_ids"),
        "source_observation_ids",
        allow_empty_strings=False,
    )
    collapsed_into = _require_nullable_non_empty_string(data, "collapsed_into")
    if status == "collapsed" and collapsed_into is None:
        raise ComplexTaskValidationError("collapsed_into: required for collapsed node")
    if status != "collapsed" and collapsed_into is not None:
        raise ComplexTaskValidationError(
            "collapsed_into: only valid for collapsed node"
        )
    attempts = _require_list(data, "attempts")
    for index, raw_attempt in enumerate(attempts):
        attempt = validate_complex_task_node_attempt(raw_attempt)
        expected_index = index + 1
        if attempt["attempt_index"] != expected_index:
            raise ComplexTaskValidationError(
                "attempts.attempt_index: expected contiguous one-based indexes"
            )
    return_value = data
    return return_value


def validate_complex_task_node_attempt(value: object) -> ComplexTaskNodeAttemptV1:
    """Validate one prompt-safe active-node attempt observation."""

    data = _require_mapping(value, "complex_task_node_attempt")
    _require_version(data, COMPLEX_TASK_NODE_ATTEMPT_VERSION)
    attempt_index = _require_positive_integer(data, "attempt_index")
    if attempt_index < 1:
        raise ComplexTaskValidationError(
            "attempt_index: expected positive integer"
        )
    _require_enum(data, "action", ALLOWED_NODE_ATTEMPT_ACTIONS)
    _require_enum(data, "status", ALLOWED_NODE_ATTEMPT_STATUSES)
    _require_string(data, "input_summary")
    _require_string(data, "result_summary")
    _validate_string_list(
        _require_list(data, "blockers"),
        "blockers",
        allow_empty_strings=False,
    )
    _require_string(data, "next_action")
    return_value = data
    return return_value


def validate_complex_task_followup_task(
    value: object,
) -> ComplexTaskFollowupTaskV1:
    """Validate one resolver-internal executable follow-up task."""

    _reject_hidden_review_hints(value, "followup_task")
    data = _require_mapping(value, "complex_task_followup_task")
    _require_version(data, COMPLEX_TASK_FOLLOWUP_TASK_VERSION)
    _require_non_empty_string(data, "objective")
    _require_enum(data, "kind", ALLOWED_FOLLOWUP_TASK_KINDS)
    _require_non_empty_string(data, "reason")
    return_value = data
    return return_value


def validate_complex_task_graph(value: object) -> ComplexTaskGraphV1:
    """Validate graph-wide node references, bounds, and traversal metadata."""

    data = _require_mapping(value, "complex_task_graph")
    _require_version(data, COMPLEX_TASK_GRAPH_VERSION)
    root_node_id = _require_non_empty_string(data, "root_node_id")
    active_node_id = _require_non_empty_string(data, "active_node_id")
    nodes = data.get("nodes")
    if not isinstance(nodes, dict):
        raise ComplexTaskValidationError("nodes: expected object")
    max_nodes = _require_positive_integer(data, "max_nodes")
    max_depth = _require_positive_integer(data, "max_depth")
    if len(nodes) > max_nodes:
        raise ComplexTaskValidationError("nodes: exceeds max_nodes")
    if root_node_id not in nodes:
        raise ComplexTaskValidationError("root_node_id: missing node")
    if active_node_id not in nodes:
        raise ComplexTaskValidationError("active_node_id: missing node")
    for map_node_id, raw_node in nodes.items():
        node = validate_complex_task_node(raw_node)
        if node["node_id"] != map_node_id:
            raise ComplexTaskValidationError("nodes: key must match node_id")
        if node["depth"] > max_depth:
            raise ComplexTaskValidationError("depth: exceeds max_depth")
    traversal_order = _require_list(data, "traversal_order")
    _validate_string_list(
        traversal_order,
        "traversal_order",
        allow_empty_strings=False,
    )
    _validate_graph_references(nodes, root_node_id, traversal_order)
    _validate_graph_is_acyclic(nodes, root_node_id)
    _validate_collapse_events(_require_list(data, "collapse_events"), nodes)
    return_value = data
    return return_value


def validate_complex_task_resolution_packet(
    value: object,
) -> ComplexTaskResolutionPacketV1:
    """Validate the standalone resolver packet returned to review callers."""

    data = _require_mapping(value, "complex_task_resolution_packet")
    _require_version(data, COMPLEX_TASK_RESOLUTION_PACKET_VERSION)
    _require_non_empty_string(data, "root_question")
    _require_string(data, "investigation_summary")
    for field_name in (
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
    validate_complex_task_graph(data.get("graph"))
    trace_summary = data.get("trace_summary")
    if not isinstance(trace_summary, dict):
        raise ComplexTaskValidationError("trace_summary: expected object")
    return_value = data
    return return_value


def validate_complex_task_subagent_request(
    value: object,
) -> ComplexTaskSubagentRequestV1:
    """Validate a bounded request for a resolver-local subagent."""

    data = _require_mapping(value, "complex_task_subagent_request")
    _require_version(data, COMPLEX_TASK_SUBAGENT_REQUEST_VERSION)
    _require_non_empty_string(data, "node_id")
    _require_non_empty_string(data, "subagent")
    _require_non_empty_string(data, "action")
    _require_non_empty_string(data, "objective")
    if not isinstance(data.get("payload"), dict):
        raise ComplexTaskValidationError("payload: expected object")
    if not isinstance(data.get("constraints"), dict):
        raise ComplexTaskValidationError("constraints: expected object")
    return_value = data
    return return_value


def validate_complex_task_subagent_result(
    value: object,
) -> ComplexTaskSubagentResultV1:
    """Validate resolver-local subagent result semantic content."""

    data = _require_mapping(value, "complex_task_subagent_result")
    _reject_subagent_result_review_hints(data)
    _require_version(data, COMPLEX_TASK_SUBAGENT_RESULT_VERSION)
    resolved = data.get("resolved")
    if not isinstance(resolved, bool):
        raise ComplexTaskValidationError("resolved: expected boolean")
    _require_enum(data, "status", ALLOWED_SUBAGENT_STATUSES)
    if not isinstance(data.get("result"), dict):
        raise ComplexTaskValidationError("result: expected object")
    attempts = _require_integer(data, "attempts")
    if attempts < 0:
        raise ComplexTaskValidationError("attempts: expected non-negative integer")
    cache = data.get("cache")
    if not isinstance(cache, dict):
        raise ComplexTaskValidationError("cache: expected object")
    cache_enabled = cache.get("enabled")
    if not isinstance(cache_enabled, bool):
        raise ComplexTaskValidationError("cache.enabled: expected boolean")
    trace = data.get("trace")
    if not isinstance(trace, (dict, list)):
        raise ComplexTaskValidationError("trace: expected object or list")
    _validate_string_list(
        _require_list(data, "unresolved_items"),
        "unresolved_items",
        allow_empty_strings=False,
    )
    return_value = data
    return return_value


def validate_complex_task_resolver_options(
    value: object,
) -> ComplexTaskResolverOptionsV1:
    """Validate public resolver options and reject behavior injection."""

    data = _require_mapping(value, "complex_task_resolver_options")
    _require_version(data, COMPLEX_TASK_RESOLVER_OPTIONS_VERSION)
    for field_name in data:
        if field_name in FORBIDDEN_OPTION_FIELDS:
            raise ComplexTaskValidationError(
                f"{field_name}: behavior injection is not accepted"
            )
        if field_name not in ("schema_version", "limits"):
            raise ComplexTaskValidationError(
                f"{field_name}: unsupported resolver option"
            )
    if "limits" not in data:
        raise ComplexTaskValidationError("limits: required")
    limits = data["limits"]
    if not isinstance(limits, dict):
        raise ComplexTaskValidationError("limits: expected object")
    for limit_name, limit_value in limits.items():
        if limit_name in FORBIDDEN_OPTION_FIELDS:
            raise ComplexTaskValidationError(
                f"limits.{limit_name}: behavior injection is not accepted"
            )
        if limit_name not in ALLOWED_OPTION_LIMITS:
            raise ComplexTaskValidationError(
                f"limits.{limit_name}: unsupported structural limit"
            )
        if not isinstance(limit_value, int) or limit_value < 1:
            raise ComplexTaskValidationError(
                f"limits.{limit_name}: expected positive int"
            )
        if limit_value > OPTION_LIMIT_CAPS[limit_name]:
            raise ComplexTaskValidationError(
                f"limits.{limit_name}: exceeds cap {OPTION_LIMIT_CAPS[limit_name]}"
            )
    return_value = data
    return return_value


def project_complex_task_packet(
    packet: object,
) -> dict[str, object]:
    """Return a compact prompt-safe packet projection for review artifacts."""

    validated_packet = validate_complex_task_resolution_packet(packet)
    trace_summary = validated_packet["trace_summary"]
    compact_trace: dict[str, object] = {}
    for field_name in (
        "iterations",
        "collapse_count",
        "node_attempt_count",
        "subagent_calls",
        "followup_created_count",
        "followup_rejected_count",
    ):
        if field_name in trace_summary:
            compact_trace[field_name] = trace_summary[field_name]
    projection = {
        "root_question": validated_packet["root_question"],
        "investigation_summary": validated_packet["investigation_summary"],
        "knowledge_we_know_so_far": list(
            validated_packet["knowledge_we_know_so_far"]
        ),
        "knowledge_still_lacking": list(
            validated_packet["knowledge_still_lacking"]
        ),
        "recommended_next_iteration": list(
            validated_packet["recommended_next_iteration"]
        ),
        "evidence_boundary_notes": list(
            validated_packet["evidence_boundary_notes"]
        ),
        "trace_summary": compact_trace,
    }
    return projection


def _require_mapping(value: object, label: str) -> dict:
    """Return a dictionary payload or raise a contract error."""

    if not isinstance(value, dict):
        raise ComplexTaskValidationError(f"{label}: expected object")
    return_value = value
    return return_value


def _require_version(data: dict, expected: str) -> None:
    """Require a specific schema version on one contract object."""

    actual = data.get("schema_version")
    if actual != expected:
        raise ComplexTaskValidationError(f"schema_version: expected {expected}")


def _require_string(data: dict, field_name: str) -> str:
    """Require one string field, allowing the empty string."""

    value = data.get(field_name)
    if not isinstance(value, str):
        raise ComplexTaskValidationError(f"{field_name}: expected string")
    return_value = value
    return return_value


def _require_non_empty_string(data: dict, field_name: str) -> str:
    """Require one non-empty string field."""

    value = data.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise ComplexTaskValidationError(f"{field_name}: expected non-empty string")
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
        raise ComplexTaskValidationError(f"{field_name}: expected non-empty string")
    return_value = value
    return return_value


def _require_enum(data: dict, field_name: str, allowed: frozenset[str]) -> str:
    """Require one field to belong to a closed vocabulary."""

    value = data.get(field_name)
    if not isinstance(value, str) or value not in allowed:
        expected = sorted(allowed)
        raise ComplexTaskValidationError(f"{field_name}: expected one of {expected}")
    return_value = value
    return return_value


def _require_list(data: dict, field_name: str) -> list:
    """Require one list field."""

    value = data.get(field_name)
    if not isinstance(value, list):
        raise ComplexTaskValidationError(f"{field_name}: expected list")
    return_value = value
    return return_value


def _require_integer(data: dict, field_name: str) -> int:
    """Require one integer field."""

    value = data.get(field_name)
    if not isinstance(value, int):
        raise ComplexTaskValidationError(f"{field_name}: expected integer")
    return_value = value
    return return_value


def _require_positive_integer(data: dict, field_name: str) -> int:
    """Require one positive integer field."""

    value = _require_integer(data, field_name)
    if value < 1:
        raise ComplexTaskValidationError(f"{field_name}: expected positive integer")
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
            raise ComplexTaskValidationError(f"{field_name}: expected strings")
        if not allow_empty_strings and not item.strip():
            raise ComplexTaskValidationError(
                f"{field_name}: expected non-empty strings"
            )


def _validate_evidence_refs(values: list, field_name: str) -> None:
    """Validate action-spec evidence references inside resolver contracts."""

    for evidence_ref in values:
        try:
            validate_evidence_ref(evidence_ref)
        except ActionValidationError as exc:
            raise ComplexTaskValidationError(
                f"{field_name}: invalid evidence ref: {exc}"
            ) from exc


def _validate_graph_references(
    nodes: dict[str, ComplexTaskNodeV1],
    root_node_id: str,
    traversal_order: list[str],
) -> None:
    """Validate node parent, child, collapse, and traversal references."""

    root_node = nodes[root_node_id]
    if root_node["parent_id"] is not None:
        raise ComplexTaskValidationError("root_node_id: root parent must be null")
    for node_id, node in nodes.items():
        parent_id = node["parent_id"]
        if parent_id is not None and parent_id not in nodes:
            raise ComplexTaskValidationError("parent_id: missing node")
        for child_id in node["children"]:
            if child_id not in nodes:
                raise ComplexTaskValidationError("children: missing node")
            child = nodes[child_id]
            if child["parent_id"] != node_id:
                raise ComplexTaskValidationError("children: child parent mismatch")
        collapsed_into = node["collapsed_into"]
        if collapsed_into is not None:
            if collapsed_into not in nodes:
                raise ComplexTaskValidationError("collapsed_into: missing node")
            if collapsed_into == node_id:
                raise ComplexTaskValidationError("collapsed_into: cannot target self")
            target = nodes[collapsed_into]
            if target["status"] == "collapsed":
                raise ComplexTaskValidationError(
                    "collapsed_into: target must not be collapsed"
                )
    for traversal_node_id in traversal_order:
        if traversal_node_id not in nodes:
            raise ComplexTaskValidationError("traversal_order: missing node")


def _validate_graph_is_acyclic(
    nodes: dict[str, ComplexTaskNodeV1],
    root_node_id: str,
) -> None:
    """Reject cycles in child links reachable from the root."""

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node_id: str) -> None:
        if node_id in visiting:
            raise ComplexTaskValidationError("nodes: cycle detected")
        if node_id in visited:
            return
        visiting.add(node_id)
        node = nodes[node_id]
        for child_id in node["children"]:
            visit(child_id)
        visiting.remove(node_id)
        visited.add(node_id)

    visit(root_node_id)


def _validate_collapse_events(
    collapse_events: list,
    nodes: dict[str, ComplexTaskNodeV1],
) -> None:
    """Validate collapse event references and concise reasons."""

    for event in collapse_events:
        if not isinstance(event, dict):
            raise ComplexTaskValidationError("collapse_events: expected objects")
        from_node_id = _require_non_empty_string(event, "from_node_id")
        to_node_id = _require_non_empty_string(event, "to_node_id")
        _require_non_empty_string(event, "reason")
        if from_node_id not in nodes or to_node_id not in nodes:
            raise ComplexTaskValidationError("collapse_events: missing node")


def _reject_hidden_review_hints(value: object, path: str) -> None:
    """Reject fixture metadata if it appears in subagent output."""

    if isinstance(value, dict):
        for key, nested_value in value.items():
            if isinstance(key, str):
                _reject_hint_string(key, f"{path}.key")
            _reject_hidden_review_hints(nested_value, f"{path}.{key}")
        return
    if isinstance(value, list):
        for index, nested_value in enumerate(value):
            _reject_hidden_review_hints(nested_value, f"{path}[{index}]")
        return
    if isinstance(value, str):
        _reject_hint_string(value, path)


def _reject_subagent_result_review_hints(data: dict[str, object]) -> None:
    """Reject fixture metadata from semantic subagent result fields."""

    for key, nested_value in data.items():
        if isinstance(key, str):
            _reject_hint_string(key, "subagent_result.key")
        if key == "trace":
            continue
        _reject_hidden_review_hints(nested_value, f"subagent_result.{key}")


def _reject_hint_string(value: str, path: str) -> None:
    """Reject strings that look like review fixture metadata markers."""

    normalized = _normalized_hint_text(value)
    for fragments in _FORBIDDEN_HINT_FRAGMENT_SETS:
        if all(fragment in normalized for fragment in fragments):
            raise ComplexTaskValidationError(f"{path}: hidden review hint rejected")
    if _looks_like_review_case_identifier(normalized):
        raise ComplexTaskValidationError(f"{path}: hidden review hint rejected")


def _normalized_hint_text(value: str) -> str:
    """Normalize possible metadata labels into simple search text."""

    chars: list[str] = []
    for char in value.lower():
        if char.isalnum():
            chars.append(char)
            continue
        chars.append(" ")
    normalized = " ".join("".join(chars).split())
    return_value = normalized
    return return_value


def _looks_like_review_case_identifier(value: str) -> bool:
    """Return whether text looks like the review-case identifier format."""

    compact = value.replace(" ", "")
    prefix = "ctr" + "_"
    return_value = compact.startswith(prefix)
    return return_value
