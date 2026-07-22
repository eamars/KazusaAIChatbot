"""Dependency-ready concurrent execution with isolated result slots."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from kazusa_ai_chatbot.cognition_core_v2.contracts import (
    BranchDefinition,
    CognitionContextLimitError,
    CognitionExecutionError,
    classify_cognition_failure,
)
from kazusa_ai_chatbot.cognition_core_v2.dependency_graph import DependencyGraph


logger = logging.getLogger(__name__)
BranchHandler = Callable[[BranchDefinition], Awaitable[Any]]


@dataclass(frozen=True)
class BranchFailure:
    """Preserve the typed cause and safe-retry boundary of one branch failure."""

    branch_id: str
    error_code: str
    stage: str
    attempt_count: int
    safe_checkpoint: str
    retryable: bool
    exception_class: str
    exception: BaseException | None = field(default=None, repr=False)

    @classmethod
    def from_exception(
        cls,
        branch_id: str,
        exception: BaseException,
    ) -> "BranchFailure":
        """Project a branch exception into bounded operational metadata.

        Args:
            branch_id: Identifier of the dependency-graph branch that failed.
            exception: Original exception raised by the branch handler.

        Returns:
            Typed failure record retaining retry safety and original cause.
        """

        if isinstance(exception, CognitionExecutionError):
            error_code = exception.error_code
            stage = exception.stage or "cognition_branch"
            attempt_count = exception.attempt_count
            safe_checkpoint = exception.safe_checkpoint
            retryable = exception.retryable
        elif isinstance(exception, CognitionContextLimitError):
            error_code = classify_cognition_failure(exception)
            stage = "cognition_branch"
            attempt_count = 1
            safe_checkpoint = "unknown"
            retryable = False
        elif isinstance(exception, (TimeoutError, ConnectionError)):
            error_code = classify_cognition_failure(exception)
            stage = "cognition_branch"
            attempt_count = 1
            safe_checkpoint = "pre_state_commit"
            retryable = True
        elif isinstance(exception, ValueError):
            error_code = classify_cognition_failure(exception)
            stage = "cognition_branch"
            attempt_count = 1
            safe_checkpoint = "unknown"
            retryable = False
        else:
            error_code = classify_cognition_failure(exception)
            stage = "cognition_branch"
            attempt_count = 1
            safe_checkpoint = "unknown"
            retryable = False
        return cls(
            branch_id=branch_id,
            error_code=error_code,
            stage=stage,
            attempt_count=max(1, attempt_count),
            safe_checkpoint=safe_checkpoint,
            retryable=retryable,
            exception_class=exception.__class__.__name__,
            exception=exception,
        )


@dataclass
class ParallelExecutionResult:
    """Preserve branch results, timings, failures, and overlap evidence."""

    results: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    started_at: dict[str, float] = field(default_factory=dict)
    ended_at: dict[str, float] = field(default_factory=dict)
    maximum_concurrency: int = 0
    failed_branch_ids: set[str] = field(default_factory=set)
    failure_records: dict[str, BranchFailure] = field(default_factory=dict)
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
                execution.failure_records[branch_id] = BranchFailure(
                    branch_id=branch_id,
                    error_code="internal_invariant",
                    stage="dependency_graph",
                    attempt_count=1,
                    safe_checkpoint="unknown",
                    retryable=False,
                    exception_class="DependencyUnavailable",
                )
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
                execution.failure_records[branch_id] = (
                    BranchFailure.from_exception(branch_id, exc)
                )
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
