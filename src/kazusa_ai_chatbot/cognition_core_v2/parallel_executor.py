"""Bounded concurrent execution for dependency-ready cognition branches."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from kazusa_ai_chatbot.cognition_core_v2.contracts import (
    BranchDefinition,
    BranchResult,
)
from kazusa_ai_chatbot.cognition_core_v2.dependency_graph import DependencyGraph


logger = logging.getLogger(__name__)
DEFAULT_LLM_CONCURRENCY_CAP = 4

BranchHandler = Callable[[BranchDefinition], Awaitable[BranchResult]]


@dataclass
class ParallelExecutionResult:
    """Preserve branch results, timing, failures, and concurrency evidence."""

    results: dict[str, BranchResult] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    started_at: dict[str, float] = field(default_factory=dict)
    ended_at: dict[str, float] = field(default_factory=dict)
    maximum_concurrency: int = 0


async def execute_dependency_graph(
    graph: DependencyGraph,
    handler: BranchHandler,
    *,
    concurrency_cap: int = DEFAULT_LLM_CONCURRENCY_CAP,
) -> ParallelExecutionResult:
    """Run only dependency-ready branches while respecting a fixed LLM cap.

    Args:
        graph: Validated branch dependency graph for one cognition invocation.
        handler: Branch-owned async model handler.
        concurrency_cap: Maximum concurrently active branch handlers.

    Returns:
        Successful branch results plus failed/skipped branch warnings and timing.
    """

    if concurrency_cap < 1:
        raise ValueError("concurrency cap must be positive")
    execution = ParallelExecutionResult()
    semaphore = asyncio.Semaphore(concurrency_cap)
    completed: set[str] = set()
    failed: set[str] = set()
    started: set[str] = set()
    active: dict[asyncio.Task[BranchResult], str] = {}

    async def run_one(definition: BranchDefinition) -> BranchResult:
        """Time one externally backed branch call inside the shared semaphore."""

        async with semaphore:
            execution.started_at[definition.branch_id] = time.perf_counter()
            active_count = len(execution.started_at) - len(execution.ended_at)
            execution.maximum_concurrency = max(
                execution.maximum_concurrency,
                active_count,
            )
            result = await handler(definition)
            execution.ended_at[definition.branch_id] = time.perf_counter()
            return result

    while len(completed) + len(failed) < len(graph.definitions):
        ready_ids = graph.ready_branch_ids(completed, failed, started)
        for branch_id in ready_ids:
            definition = graph.definitions[branch_id]
            task = asyncio.create_task(run_one(definition))
            active[task] = branch_id
            started.add(branch_id)
        if not active:
            skipped_ids = set(graph.definitions).difference(completed, failed)
            for branch_id in sorted(skipped_ids):
                failed.add(branch_id)
                execution.warnings.append(
                    f"{branch_id} skipped because a dependency failed"
                )
            continue
        done, _ = await asyncio.wait(
            active,
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in done:
            branch_id = active.pop(task)
            try:
                result = task.result()
            except Exception as exc:
                logger.exception(f"V2 branch {branch_id} failed: {exc}")
                failed.add(branch_id)
                execution.ended_at.setdefault(branch_id, time.perf_counter())
                execution.warnings.append(f"{branch_id} failed: {exc}")
                continue
            execution.results[branch_id] = result
            completed.add(branch_id)
    return execution
