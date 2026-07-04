"""Deterministic traversal helpers for local-context graphs."""

from __future__ import annotations

from .contracts import LocalContextGraphV1, validate_local_context_graph


def find_next_active_node(graph: LocalContextGraphV1) -> str | None:
    """Return the next pending node whose dependencies are already resolved."""

    validated_graph = validate_local_context_graph(graph)
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
        if node["status"] != "pending":
            continue
        if not _dependencies_resolved(validated_graph, node["depends_on"]):
            continue
        return_value = node_id
        return return_value
    return_value = None
    return return_value


def _dependencies_resolved(
    graph: LocalContextGraphV1,
    dependency_ids: list[str],
) -> bool:
    """Return whether every dependency is resolved or collapsed."""

    for dependency_id in dependency_ids:
        dependency_node = graph["nodes"][dependency_id]
        if dependency_node["status"] not in ("resolved", "collapsed"):
            return_value = False
            return return_value
    return_value = True
    return return_value
