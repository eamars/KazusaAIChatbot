"""Contracts for inert patch proposal assembly and review materialization."""

from typing import Literal, TypedDict

PatchValidationStatus = Literal["succeeded", "failed", "rejected"]
PatchApplyStatus = Literal["succeeded", "failed", "rejected"]
PatchOperationKind = Literal[
    "create_file",
    "insert_before",
    "insert_after",
    "replace",
    "replace_file_small",
]
PatchPackageStatus = Literal["succeeded", "failed", "rejected"]


class PatchArtifact(TypedDict):
    """Unified-diff patch proposal for review-only materialization."""

    artifact_id: str
    base: str
    diff_text: str
    files: list[str]
    summary: str


class PatchOperation(TypedDict, total=False):
    """Structured inert edit operation compiled into a unified diff."""

    operation_id: str
    kind: PatchOperationKind
    path: str
    content: str
    anchor: str
    summary: str
    evidence_ids: list[str]
    full_file_rationale: str


class CreatedFileSummary(TypedDict):
    """Public summary for a file introduced by a proposal."""

    path: str
    role: str


class ChangedFileSummary(TypedDict):
    """Public summary for one file touched by proposed patch artifacts."""

    path: str
    change_type: str
    summary: str


class PatchValidationSummary(TypedDict):
    """Public-safe patch validation result."""

    status: PatchValidationStatus
    parsed: bool
    sandbox_applied: bool
    errors: list[str]
    warnings: list[str]
    files: list[str]


class PatchSourceIdentity(TypedDict, total=False):
    """Public source identity used to reject stale apply requests."""

    provider: str
    owner: str | None
    repo: str | None
    current_commit: str
    dirty_state: str


class PatchApplyApproval(TypedDict):
    """Trusted structured approval required for patch application."""

    approved: bool
    approved_by: str
    approved_at: str
    approval_reason: str


class ApplyWorkspaceRef(TypedDict):
    """Opaque public reference to a managed patch apply workspace."""

    kind: str
    apply_package_id: str
    source_identity: dict[str, object]
    applied_files: list[str]


class PatchApplyValidation(TypedDict):
    """Public-safe patch apply validation result."""

    status: PatchApplyStatus
    errors: list[str]
    warnings: list[str]


class CodingPatchApplyRequest(TypedDict, total=False):
    """Trusted direct request to apply approved patch artifacts."""

    workspace_root: str
    source_root: str
    source_identity: PatchSourceIdentity
    expected_source_identity: PatchSourceIdentity
    patch_artifacts: list[PatchArtifact]
    approval: PatchApplyApproval
    max_files: int
    max_diff_chars: int


class CodingPatchApplyResponse(TypedDict):
    """Public-safe response for approved patch application."""

    status: PatchApplyStatus
    apply_package_id: str
    source_identity: dict[str, object]
    apply_workspace_ref: ApplyWorkspaceRef
    applied_files: list[str]
    changed_files: list[ChangedFileSummary]
    validation: PatchApplyValidation
    limitations: list[str]
    trace_summary: list[str]


class PatchProposalInput(TypedDict):
    """Selected artifacts handed to the canonical patcher boundary."""

    artifact_package_id: str
    artifacts: list[dict[str, object]]
    reserved_paths: list[dict[str, object]]
    max_artifact_chars: int


class PatchProposalReport(TypedDict):
    """Patch materialization result returned to callers."""

    status: PatchPackageStatus
    artifact_package: dict[str, object]
    patch_artifacts: list[PatchArtifact]
    created_files: list[CreatedFileSummary]
    changed_files: list[ChangedFileSummary]
    diagnostics: list[str]
