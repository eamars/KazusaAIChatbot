"""Graph validation and traversal tests for the complex-task resolver."""

from __future__ import annotations

import pytest

from kazusa_ai_chatbot.complex_task_resolver import (
    COMPLEX_TASK_GRAPH_VERSION,
    COMPLEX_TASK_NODE_VERSION,
    validate_complex_task_graph,
)
from kazusa_ai_chatbot.complex_task_resolver.contracts import (
    ComplexTaskValidationError,
)
from kazusa_ai_chatbot.complex_task_resolver.graph import find_next_active_node


def _node(
    node_id: str,
    *,
    parent_id: str | None,
    depth: int,
    status: str,
    children: list[str] | None = None,
    collapsed_into: str | None = None,
) -> dict:
    node = {
        "schema_version": COMPLEX_TASK_NODE_VERSION,
        "node_id": node_id,
        "parent_id": parent_id,
        "depth": depth,
        "objective": f"Objective for {node_id}",
        "node_kind": "subtask" if parent_id else "root",
        "status": status,
        "children": children or [],
        "investigation_summary": "",
        "knowledge_we_know_so_far": [],
        "knowledge_still_lacking": [],
        "recommended_next_iteration": [],
        "evidence_boundary_notes": [],
        "evidence_refs": [],
        "source_observation_ids": [],
        "collapsed_into": collapsed_into,
        "attempts": [],
    }
    return node


def test_graph_validation_accepts_bounded_tree_and_selects_one_active_node() -> None:
    """Validate a bounded top-down graph and deterministic active-node choice."""

    graph = validate_complex_task_graph({
        "schema_version": COMPLEX_TASK_GRAPH_VERSION,
        "root_node_id": "root",
        "active_node_id": "task_a",
        "nodes": {
            "root": _node(
                "root",
                parent_id=None,
                depth=0,
                status="expanded",
                children=["task_a", "task_b"],
            ),
            "task_a": _node(
                "task_a",
                parent_id="root",
                depth=1,
                status="pending",
            ),
            "task_b": _node(
                "task_b",
                parent_id="root",
                depth=1,
                status="resolved",
            ),
        },
        "traversal_order": ["root"],
        "collapse_events": [],
        "max_nodes": 8,
        "max_depth": 3,
    })

    active_node_id = find_next_active_node(graph)

    assert active_node_id == "task_a"


def test_graph_validation_rejects_cycles_and_depth_overflow() -> None:
    """Reject recursive graph shapes that exceed deterministic safety caps."""

    with pytest.raises(ComplexTaskValidationError):
        validate_complex_task_graph({
            "schema_version": COMPLEX_TASK_GRAPH_VERSION,
            "root_node_id": "root",
            "active_node_id": "root",
            "nodes": {
                "root": _node(
                    "root",
                    parent_id=None,
                    depth=0,
                    status="expanded",
                    children=["child"],
                ),
                "child": _node(
                    "child",
                    parent_id="root",
                    depth=1,
                    status="expanded",
                    children=["root"],
                ),
            },
            "traversal_order": [],
            "collapse_events": [],
            "max_nodes": 8,
            "max_depth": 3,
        })

    with pytest.raises(ComplexTaskValidationError):
        validate_complex_task_graph({
            "schema_version": COMPLEX_TASK_GRAPH_VERSION,
            "root_node_id": "root",
            "active_node_id": "deep",
            "nodes": {
                "root": _node(
                    "root",
                    parent_id=None,
                    depth=0,
                    status="expanded",
                    children=["deep"],
                ),
                "deep": _node(
                    "deep",
                    parent_id="root",
                    depth=4,
                    status="pending",
                ),
            },
            "traversal_order": ["root"],
            "collapse_events": [],
            "max_nodes": 8,
            "max_depth": 3,
        })


def test_collapsed_nodes_must_point_to_non_collapsed_existing_nodes() -> None:
    """Reject collapse chains and missing collapse targets."""

    with pytest.raises(ComplexTaskValidationError):
        validate_complex_task_graph({
            "schema_version": COMPLEX_TASK_GRAPH_VERSION,
            "root_node_id": "root",
            "active_node_id": "root",
            "nodes": {
                "root": _node(
                    "root",
                    parent_id=None,
                    depth=0,
                    status="expanded",
                    children=["a", "b"],
                ),
                "a": _node(
                    "a",
                    parent_id="root",
                    depth=1,
                    status="collapsed",
                    collapsed_into="b",
                ),
                "b": _node(
                    "b",
                    parent_id="root",
                    depth=1,
                    status="collapsed",
                    collapsed_into="a",
                ),
            },
            "traversal_order": ["root", "a", "b"],
            "collapse_events": [{
                "from_node_id": "a",
                "to_node_id": "b",
                "reason": "same answer",
            }],
            "max_nodes": 8,
            "max_depth": 3,
        })
