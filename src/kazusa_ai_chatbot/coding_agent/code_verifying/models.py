"""Contracts for direct verify-and-repair orchestration."""

from typing import Literal, NotRequired, TypedDict

from kazusa_ai_chatbot.coding_agent.code_executing.models import (
    CodeExecutionResponse,
    CodeExecutionSpec,
)
from kazusa_ai_chatbot.coding_agent.code_patching.models import (
    ApplyWorkspaceRef,
    ChangedFileSummary,
    CodingPatchApplyResponse,
    PatchApplyApproval,
    PatchArtifact,
    PatchSourceIdentity,
)
from kazusa_ai_chatbot.coding_agent.models import (
    CodingAgentRepositorySummary,
    CodingAgentSourceScope,
    SourceKind,
)

VerifyRepairStatus = Literal[
    "succeeded",
    "failed",
    "rejected",
    "timed_out",
    "blocked",
]


class ExecutionRepairFeedback(TypedDict):
    """Structured execution failure evidence for source repair."""

    feedback_source: str
    attempt_index: int
    overall_status: str
    failed_tools: list[str]
    failed_paths: list[str]
    exit_codes: list[dict[str, object]]
    failure_summaries: list[str]
    stdout_excerpt: str
    stderr_excerpt: str
    output_truncated: bool
    instruction: str


class VerifyRepairAttempt(TypedDict):
    """Public-safe summary for one apply and execution attempt."""

    attempt_index: int
    proposal_status: str
    apply_status: str
    execution_statuses: list[str]
    patch_artifact_count: int
    changed_files: list[ChangedFileSummary]
    apply_package_id: str | None
    limitations: list[str]
    trace_summary: list[str]


class CodingVerifyRepairRequest(TypedDict, total=False):
    """Trusted direct request for bounded verify-and-repair work."""

    question: str
    source_url: str
    repo_url: str
    repo_hint: str
    local_root_hint: str
    local_path_hint: str
    requested_ref: str
    source_scope_hint: SourceKind
    workspace_root: str
    preferred_language: str
    max_answer_chars: int
    max_artifact_chars: int
    approval: PatchApplyApproval
    execution_specs: list[CodeExecutionSpec]
    repair_attempt_limit: int
    max_repair_feedback_chars: int
    initial_patch_artifacts: list[PatchArtifact]
    expected_source_identity: PatchSourceIdentity
    session_id: str


class CodingVerifyRepairResponse(TypedDict):
    """Public-safe terminal response for verify-and-repair work."""

    status: VerifyRepairStatus
    answer_text: str
    repository: CodingAgentRepositorySummary | None
    source_scope: CodingAgentSourceScope | None
    attempts: list[VerifyRepairAttempt]
    final_patch_artifacts: list[PatchArtifact]
    final_changed_files: list[ChangedFileSummary]
    final_apply: CodingPatchApplyResponse | None
    final_execution: list[CodeExecutionResponse]
    blockers: list[dict[str, object]]
    limitations: list[str]
    trace_summary: list[str]
    trace: NotRequired[dict[str, object]]


class RepairProposalValidation(TypedDict):
    """Validation result for repaired proposal scope."""

    errors: list[str]
    changed_paths: list[str]
    apply_workspace_ref: ApplyWorkspaceRef | None
