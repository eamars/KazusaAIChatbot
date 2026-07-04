"""Graph validation and traversal tests for the local-context resolver."""

from __future__ import annotations

import pytest

from kazusa_ai_chatbot.local_context_resolver import (
    LOCAL_CONTEXT_GRAPH_VERSION,
    LOCAL_CONTEXT_NODE_VERSION,
    LocalContextValidationError,
    validate_local_context_graph,
)
from kazusa_ai_chatbot.local_context_resolver.graph import find_next_active_node


def test_graph_selects_pending_node_after_dependencies_resolve() -> None:
    """Select the next node whose graph dependencies are already resolved."""

    graph = validate_local_context_graph({
        "schema_version": LOCAL_CONTEXT_GRAPH_VERSION,
        "root_node_id": "root",
        "active_node_id": "conversation_1",
        "nodes": {
            "root": _node(
                "root",
                parent_id=None,
                status="resolved",
                children=["conversation_1", "person_1"],
            ),
            "conversation_1": _node(
                "conversation_1",
                parent_id="root",
                node_kind="conversation_evidence",
                status="resolved",
                produces=["speaker_ref"],
            ),
            "person_1": _node(
                "person_1",
                parent_id="root",
                node_kind="person_context",
                status="pending",
                depends_on=["conversation_1"],
                consumes={"speaker": "speaker_ref"},
            ),
        },
        "traversal_order": ["root", "conversation_1"],
        "collapse_events": [],
        "max_nodes": 8,
        "max_depth": 3,
    })

    active_node_id = find_next_active_node(graph)

    assert active_node_id == "person_1"


def test_graph_skips_pending_node_with_blocked_dependency() -> None:
    """Do not activate a node whose dependency is still blocked."""

    graph = validate_local_context_graph({
        "schema_version": LOCAL_CONTEXT_GRAPH_VERSION,
        "root_node_id": "root",
        "active_node_id": "conversation_1",
        "nodes": {
            "root": _node(
                "root",
                parent_id=None,
                status="resolved",
                children=["conversation_1", "memory_1"],
            ),
            "conversation_1": _node(
                "conversation_1",
                parent_id="root",
                node_kind="conversation_evidence",
                status="blocked",
            ),
            "memory_1": _node(
                "memory_1",
                parent_id="root",
                node_kind="memory_evidence",
                status="pending",
                depends_on=["conversation_1"],
            ),
        },
        "traversal_order": ["root", "conversation_1"],
        "collapse_events": [],
        "max_nodes": 8,
        "max_depth": 3,
    })

    active_node_id = find_next_active_node(graph)

    assert active_node_id is None


def test_graph_validation_rejects_cycles_missing_dependencies_and_depth() -> None:
    """Reject graph shapes that would make traversal ambiguous or unsafe."""

    with pytest.raises(LocalContextValidationError):
        validate_local_context_graph({
            "schema_version": LOCAL_CONTEXT_GRAPH_VERSION,
            "root_node_id": "root",
            "active_node_id": "root",
            "nodes": {
                "root": _node(
                    "root",
                    parent_id=None,
                    status="resolved",
                    children=["child"],
                ),
                "child": _node(
                    "child",
                    parent_id="root",
                    status="resolved",
                    children=["root"],
                ),
            },
            "traversal_order": [],
            "collapse_events": [],
            "max_nodes": 8,
            "max_depth": 3,
        })

    with pytest.raises(LocalContextValidationError):
        validate_local_context_graph({
            "schema_version": LOCAL_CONTEXT_GRAPH_VERSION,
            "root_node_id": "root",
            "active_node_id": "memory_1",
            "nodes": {
                "root": _node(
                    "root",
                    parent_id=None,
                    status="resolved",
                    children=["memory_1"],
                ),
                "memory_1": _node(
                    "memory_1",
                    parent_id="root",
                    status="pending",
                    depends_on=["missing"],
                ),
            },
            "traversal_order": ["root"],
            "collapse_events": [],
            "max_nodes": 8,
            "max_depth": 3,
        })

    with pytest.raises(LocalContextValidationError):
        validate_local_context_graph({
            "schema_version": LOCAL_CONTEXT_GRAPH_VERSION,
            "root_node_id": "root",
            "active_node_id": "deep",
            "nodes": {
                "root": _node(
                    "root",
                    parent_id=None,
                    status="resolved",
                    children=["a"],
                ),
                "a": _node(
                    "a",
                    parent_id="root",
                    status="resolved",
                    children=["b"],
                ),
                "b": _node(
                    "b",
                    parent_id="a",
                    status="resolved",
                    children=["deep"],
                ),
                "deep": _node(
                    "deep",
                    parent_id="b",
                    status="pending",
                ),
            },
            "traversal_order": ["root", "a", "b"],
            "collapse_events": [],
            "max_nodes": 8,
            "max_depth": 2,
        })


def _node(
    node_id: str,
    *,
    parent_id: str | None,
    status: str,
    node_kind: str = "memory_evidence",
    children: list[str] | None = None,
    depends_on: list[str] | None = None,
    consumes: dict[str, str] | None = None,
    produces: list[str] | None = None,
) -> dict[str, object]:
    """Build a valid graph node for traversal tests."""

    selected_kind = "synthesis" if parent_id is None else node_kind
    node = {
        "schema_version": LOCAL_CONTEXT_NODE_VERSION,
        "node_id": node_id,
        "node_kind": selected_kind,
        "objective": f"Objective for {node_id}",
        "parent_id": parent_id,
        "children": children or [],
        "depends_on": depends_on or [],
        "consumes": consumes or {},
        "produces": produces or [],
        "status": status,
        "investigation_summary": [],
        "knowledge_we_know_so_far": [],
        "knowledge_still_lacking": [],
        "recommended_next_iteration": [],
        "evidence_boundary_notes": [],
        "attempts": [],
        "collapsed_into": None,
    }
    return node
