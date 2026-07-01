"""Deterministic traversal helpers for complex-task graphs."""

from __future__ import annotations

from .contracts import ComplexTaskGraphV1, validate_complex_task_graph


def find_next_active_node(graph: ComplexTaskGraphV1) -> str | None:
    """Return the next pending node in bounded depth-first graph order."""

    validated_graph = validate_complex_task_graph(graph)
    node_ids: list[str] = []

    def visit(node_id: str) -> None:
        if node_id not in node_ids:
            node_ids.append(node_id)
        node = validated_graph["nodes"][node_id]
        for child_id in node["children"]:
            visit(child_id)

    visit(validated_graph["root_node_id"])

    for node_id in validated_graph["nodes"]:
        if node_id not in node_ids:
            node_ids.append(node_id)

    for node_id in node_ids:
        node = validated_graph["nodes"][node_id]
        if node["status"] == "pending":
            return_value = node_id
            return return_value
    return_value = None
    return return_value
