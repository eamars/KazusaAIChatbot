"""Explicit dependency-DAG validation for appraisal and goal cognition."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from kazusa_ai_chatbot.cognition_core_v2.contracts import BranchDefinition


class DependencyGraphError(ValueError):
    """Raise when declared dependencies cannot be scheduled safely."""


@dataclass(frozen=True)
class DependencyGraph:
    """Hold a validated branch graph and its external question dependencies."""

    definitions: Mapping[str, BranchDefinition]
    external_dependencies: frozenset[str] = frozenset()

    def ready_branch_ids(
        self,
        completed_branch_ids: set[str],
        failed_branch_ids: set[str],
        started_branch_ids: set[str],
        completed_external_dependencies: set[str] | None = None,
    ) -> list[str]:
        """Return branches whose internal and external dependencies are ready."""

        external_completed = completed_external_dependencies or set()
        ready: list[str] = []
        for branch_id, definition in self.definitions.items():
            if branch_id in started_branch_ids:
                continue
            internal = set(definition.dependencies).intersection(self.definitions)
            external = set(definition.dependencies).intersection(
                self.external_dependencies
            )
            if internal.intersection(failed_branch_ids):
                continue
            if not internal.issubset(completed_branch_ids):
                continue
            if not external.issubset(external_completed):
                continue
            ready.append(branch_id)
        ready.sort()
        return ready


def build_dependency_graph(
    definitions: Iterable[BranchDefinition],
    external_dependencies: Iterable[str] = (),
) -> DependencyGraph:
    """Validate branch ids, external refs, and cycles before any execution."""

    definitions_by_id: dict[str, BranchDefinition] = {}
    for definition in definitions:
        if definition.branch_id in definitions_by_id:
            raise DependencyGraphError(f"duplicate branch id: {definition.branch_id}")
        definitions_by_id[definition.branch_id] = definition
    external = frozenset(external_dependencies)
    for definition in definitions_by_id.values():
        unknown = set(definition.dependencies).difference(
            definitions_by_id,
            external,
        )
        if unknown:
            raise DependencyGraphError(
                f"{definition.branch_id} depends on unknown refs: {sorted(unknown)}"
            )
    _require_acyclic(definitions_by_id)
    return DependencyGraph(definitions_by_id, external)


def build_dependency_levels(
    definitions: Iterable[BranchDefinition],
    external_dependencies: Iterable[str] = (),
) -> tuple[tuple[str, ...], ...]:
    """Return stable dependency levels for a graph with ready external refs."""

    external = frozenset(external_dependencies)
    graph = build_dependency_graph(definitions, external)
    completed: set[str] = set()
    started: set[str] = set()
    levels: list[tuple[str, ...]] = []
    while len(completed) < len(graph.definitions):
        ready = graph.ready_branch_ids(completed, set(), started, set(external))
        if not ready:
            raise DependencyGraphError("dependency graph has no ready branches")
        level = tuple(ready)
        levels.append(level)
        completed.update(level)
        started.update(level)
    return tuple(levels)


def _require_acyclic(definitions: Mapping[str, BranchDefinition]) -> None:
    """Reject circular internal branch dependencies before execution."""

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(branch_id: str) -> None:
        if branch_id in visiting:
            raise DependencyGraphError(f"dependency cycle includes {branch_id}")
        if branch_id in visited:
            return
        visiting.add(branch_id)
        for dependency_id in definitions[branch_id].dependencies:
            if dependency_id in definitions:
                visit(dependency_id)
        visiting.remove(branch_id)
        visited.add(branch_id)

    for branch_id in definitions:
        visit(branch_id)
