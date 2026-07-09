"""Public contracts for durable coding-agent runs."""

from typing import Literal, TypedDict

from kazusa_ai_chatbot.coding_agent.code_executing.models import (
    CodeExecutionSpec,
)
from kazusa_ai_chatbot.coding_agent.code_patching.models import (
    ChangedFileSummary,
    PatchApplyApproval,
    PatchArtifact,
    PatchSourceIdentity,
)
from kazusa_ai_chatbot.coding_agent.models import (
    CodeEvidenceReference,
    CodingAgentRepositorySummary,
    CodingAgentSourceScope,
    InlineSourceInput,
    SourceKind,
)


CodingRunObjectiveType = Literal["read_only", "propose_patch", "verify_repair"]
CodingRunAction = Literal[
    "revise_proposal",
    "summarize",
    "approve_and_verify",
    "cancel",
]
CodingRunStatus = Literal[
    "created",
    "source_resolved",
    "evidence_collected",
    "proposal_ready",
    "awaiting_approval",
    "applying",
    "verifying",
    "repairing",
    "completed",
    "blocked",
    "rejected",
    "failed",
    "cancelled",
]


class CodingRunStartRequest(TypedDict, total=False):
    """Request to start one durable coding-agent run."""

    question: str
    objective_type: str
    source_url: str
    repo_url: str
    repo_hint: str
    local_root_hint: str
    local_path_hint: str
    requested_ref: str
    source_scope_hint: SourceKind
    inline_sources: list[InlineSourceInput]
    workspace_root: str
    preferred_language: str
    max_answer_chars: int
    max_artifact_chars: int
    session_id: str
    approval: PatchApplyApproval
    execution_specs: list[CodeExecutionSpec]
    repair_attempt_limit: int
    initial_patch_artifacts: list[PatchArtifact]
    expected_source_identity: PatchSourceIdentity


class CodingRunContinueRequest(TypedDict, total=False):
    """Request to continue an existing durable coding-agent run."""

    workspace_root: str
    run_id: str
    action: str
    revision_instruction: str
    approval: PatchApplyApproval
    execution_specs: list[CodeExecutionSpec]
    repair_attempt_limit: int
    reason: str


class CodingRunGetRequest(TypedDict, total=False):
    """Request to inspect an existing durable coding-agent run."""

    workspace_root: str
    run_id: str


class CodingRunAttempt(TypedDict, total=False):
    """Public-safe summary for one apply and verification attempt."""

    attempt_index: int
    proposal_status: str
    apply_status: str
    execution_statuses: list[str]
    patch_artifact_count: int
    changed_files: list[ChangedFileSummary]
    apply_package_id: str | None
    limitations: list[str]
    trace_summary: list[str]


class CodingRunBlocker(TypedDict, total=False):
    """Public blocker that explains why a run cannot continue automatically."""

    code: str
    message: str
    details: dict[str, object]


class CodingRunEvent(TypedDict):
    """Public lifecycle event for a durable coding-agent run."""

    event_id: str
    run_id: str
    sequence: int
    event_type: str
    status: str
    summary: str
    public_payload: dict[str, object]


class CodingRunLedger(TypedDict):
    """Workspace-local durable state for one coding-agent run."""

    schema_version: int
    run_id: str
    status: str
    goal: str
    objective_type: str
    created_at: str
    updated_at: str
    source_request: dict[str, object]
    repository: CodingAgentRepositorySummary | None
    source_scope: CodingAgentSourceScope | None
    answer_text: str
    evidence: list[CodeEvidenceReference]
    patch_artifacts: list[PatchArtifact]
    changed_files: list[ChangedFileSummary]
    approvals: list[dict[str, object]]
    apply_attempts: list[dict[str, object]]
    execution_attempts: list[dict[str, object]]
    repair_attempts: list[CodingRunAttempt]
    attempts: list[CodingRunAttempt]
    blockers: list[CodingRunBlocker]
    limitations: list[str]
    trace_summary: list[str]


class CodingRunResponse(TypedDict):
    """Public projection returned by start, continue, and get calls."""

    status: str
    run_id: str
    goal: str
    objective_type: str
    answer_text: str
    repository: CodingAgentRepositorySummary | None
    source_scope: CodingAgentSourceScope | None
    evidence: list[CodeEvidenceReference]
    patch_artifacts: list[PatchArtifact]
    changed_files: list[ChangedFileSummary]
    apply_attempts: list[dict[str, object]]
    execution_attempts: list[dict[str, object]]
    repair_attempts: list[CodingRunAttempt]
    attempts: list[CodingRunAttempt]
    blockers: list[CodingRunBlocker]
    events: list[CodingRunEvent]
    allowed_next_actions: list[str]
    limitations: list[str]
    trace_summary: list[str]
