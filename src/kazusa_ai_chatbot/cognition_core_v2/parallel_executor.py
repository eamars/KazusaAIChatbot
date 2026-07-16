"""Dependency-ready concurrent execution with isolated result slots."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from kazusa_ai_chatbot.cognition_core_v2.contracts import BranchDefinition
from kazusa_ai_chatbot.cognition_core_v2.dependency_graph import DependencyGraph


logger = logging.getLogger(__name__)
BranchHandler = Callable[[BranchDefinition], Awaitable[Any]]


@dataclass
class ParallelExecutionResult:
    """Preserve branch results, timings, failures, and overlap evidence."""

    results: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    started_at: dict[str, float] = field(default_factory=dict)
    ended_at: dict[str, float] = field(default_factory=dict)
    maximum_concurrency: int = 0
    failed_branch_ids: set[str] = field(default_factory=set)
    call_count: int = 0
    overlap_ms: int = 0
    dependency_wait_ms: int = 0
    critical_path_ms: int = 0


async def execute_dependency_graph(
    graph: DependencyGraph,
    handler: BranchHandler,
    *,
    completed_external_dependencies: set[str] | None = None,
) -> ParallelExecutionResult:
    """Run every currently ready branch and release dependents after success."""

    execution = ParallelExecutionResult()
    execution_started_at = time.perf_counter()
    completed: set[str] = set()
    failed: set[str] = set()
    started: set[str] = set()
    active: dict[asyncio.Task[Any], str] = {}
    external_completed = completed_external_dependencies or set()
    dependency_ready_at: dict[str, float] = {}

    async def run_one(definition: BranchDefinition) -> Any:
        execution.started_at[definition.branch_id] = time.perf_counter()
        return await handler(definition)

    while len(completed) + len(failed) < len(graph.definitions):
        ready_ids = graph.ready_branch_ids(
            completed,
            failed,
            started,
            external_completed,
        )
        for branch_id in ready_ids:
            internal_dependencies = graph.definitions[branch_id].dependencies
            dependency_ready_at[branch_id] = max(
                (
                    execution.ended_at[dependency_id]
                    for dependency_id in internal_dependencies
                    if dependency_id in execution.ended_at
                ),
                default=execution_started_at,
            )
            task = asyncio.create_task(run_one(graph.definitions[branch_id]))
            active[task] = branch_id
            started.add(branch_id)
            execution.call_count += 1
        execution.maximum_concurrency = max(
            execution.maximum_concurrency,
            len(active),
        )
        if not active:
            skipped = set(graph.definitions).difference(completed, failed)
            for branch_id in sorted(skipped):
                failed.add(branch_id)
                execution.failed_branch_ids.add(branch_id)
                execution.ended_at[branch_id] = time.perf_counter()
                execution.warnings.append(
                    f"{branch_id} skipped because a dependency failed "
                    "or was unavailable"
                )
            continue
        done, _ = await asyncio.wait(
            active,
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in done:
            branch_id = active.pop(task)
            execution.ended_at[branch_id] = time.perf_counter()
            execution.dependency_wait_ms += max(
                0,
                int(
                    (
                        execution.started_at[branch_id]
                        - dependency_ready_at.get(
                            branch_id,
                            execution_started_at,
                        )
                    )
                    * 1000
                ),
            )
            try:
                execution.results[branch_id] = task.result()
            except Exception as exc:
                logger.exception(f"V2 branch {branch_id} failed: {exc}")
                failed.add(branch_id)
                execution.failed_branch_ids.add(branch_id)
                execution.warnings.append(f"{branch_id} failed: {exc}")
            else:
                completed.add(branch_id)
    durations = [
        execution.ended_at[branch_id] - execution.started_at[branch_id]
        for branch_id in execution.started_at
        if branch_id in execution.ended_at
    ]
    if durations:
        execution.critical_path_ms = max(
            0,
            int((max(execution.ended_at.values()) - execution_started_at) * 1000),
        )
        execution.overlap_ms = max(
            0,
            int((sum(durations) - execution.critical_path_ms / 1000) * 1000),
        )
    return execution
