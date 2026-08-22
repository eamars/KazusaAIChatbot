"""Typed V3 branch execution records without parallel orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from kazusa_ai_chatbot.cognition_shared.contracts import (
    CognitionContextLimitError,
    CognitionExecutionError,
    classify_cognition_failure,
)

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
