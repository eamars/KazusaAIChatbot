"""Explicit dependency-DAG validation for V2 goal cognition branches."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable, Mapping

from kazusa_ai_chatbot.cognition_core_v2.contracts import BranchDefinition


class DependencyGraphError(ValueError):
    """Raise when declared branch dependencies cannot be scheduled safely."""


@dataclass(frozen=True)
class DependencyGraph:
    """Hold a validated immutable branch dependency mapping."""

    definitions: Mapping[str, BranchDefinition]

    def ready_branch_ids(
        self,
        completed_branch_ids: set[str],
        failed_branch_ids: set[str],
        started_branch_ids: set[str],
    ) -> list[str]:
        """Return branches whose declared dependencies completed successfully."""

        ready = []
        for branch_id, definition in self.definitions.items():
            dependencies = set(definition.dependencies)
            if branch_id in started_branch_ids:
                continue
            if dependencies.intersection(failed_branch_ids):
                continue
            if dependencies.issubset(completed_branch_ids):
                ready.append(branch_id)
        ready.sort()
        return ready


def build_dependency_graph(
    definitions: Iterable[BranchDefinition],
) -> DependencyGraph:
    """Validate branch ids and dependencies before bounded concurrent execution.

    Args:
        definitions: Activated branch definitions for one V2 invocation.

    Returns:
        An immutable graph used by the parallel executor.

    Raises:
        DependencyGraphError: A branch id duplicates, references an absent
            dependency, or participates in a dependency cycle.
    """

    definitions_by_id: dict[str, BranchDefinition] = {}
    for definition in definitions:
        if definition.branch_id in definitions_by_id:
            raise DependencyGraphError(f"duplicate branch id: {definition.branch_id}")
        definitions_by_id[definition.branch_id] = definition
    for definition in definitions_by_id.values():
        unknown_dependencies = set(definition.dependencies).difference(
            definitions_by_id,
        )
        if unknown_dependencies:
            raise DependencyGraphError(
                f"{definition.branch_id} depends on unknown branches: "
                f"{sorted(unknown_dependencies)}"
            )
    _require_acyclic(definitions_by_id)
    dependency_graph = DependencyGraph(definitions=definitions_by_id)
    return dependency_graph


def build_dependency_levels(
    definitions: Iterable[BranchDefinition],
) -> tuple[tuple[str, ...], ...]:
    """Return stable dependency-ready branch levels for deterministic tests.

    Args:
        definitions: Activated branch definitions with declared dependencies.

    Returns:
        Ordered groups whose members have no unmet dependency in prior levels.

    Raises:
        DependencyGraphError: A definition references an absent dependency or
            the graph contains a cycle.
    """

    graph = build_dependency_graph(definitions)
    completed: set[str] = set()
    started: set[str] = set()
    levels: list[tuple[str, ...]] = []
    while len(completed) < len(graph.definitions):
        ready_ids = graph.ready_branch_ids(completed, set(), started)
        if not ready_ids:
            raise DependencyGraphError("dependency graph has no ready branches")
        level = tuple(ready_ids)
        levels.append(level)
        completed.update(level)
        started.update(level)
    result = tuple(levels)
    return result


def _require_acyclic(definitions: Mapping[str, BranchDefinition]) -> None:
    """Reject circular branch dependencies before any branch starts."""

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(branch_id: str) -> None:
        """Depth-first validate one branch and its declared dependencies."""

        if branch_id in visiting:
            raise DependencyGraphError(f"dependency cycle includes {branch_id}")
        if branch_id in visited:
            return
        visiting.add(branch_id)
        for dependency_id in definitions[branch_id].dependencies:
            visit(dependency_id)
        visiting.remove(branch_id)
        visited.add(branch_id)

    for branch_id in definitions:
        visit(branch_id)
