"""Contracts for bounded managed-workspace code execution."""

from typing import Literal, TypedDict


CodeExecutionStatus = Literal["succeeded", "failed", "rejected", "timed_out"]
CodeExecutionTool = Literal["python_compileall", "pytest"]


class CodeExecutionSpec(TypedDict, total=False):
    """Trusted structured execution spec selected by deterministic callers."""

    tool: str
    paths: list[str]
    pytest_selectors: list[str]
    timeout_seconds: int


class CodeExecutionRequest(TypedDict, total=False):
    """Trusted direct request to run a bounded verification command."""

    workspace_root: str
    apply_package_id: str
    apply_workspace_ref: dict[str, object]
    execution: CodeExecutionSpec
    max_stdout_chars: int
    max_stderr_chars: int


class CodeExecutionResponse(TypedDict):
    """Public-safe result for one bounded execution request."""

    status: CodeExecutionStatus
    tool: str
    exit_code: int | None
    timed_out: bool
    duration_ms: int
    stdout_excerpt: str
    stderr_excerpt: str
    output_truncated: bool
    executed_paths: list[str]
    limitations: list[str]
    trace_summary: list[str]
