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
    "delete_file",
    "rename_file",
]
PatchPackageStatus = Literal["succeeded", "failed", "rejected"]
CandidateBaseline = Literal["resolved_source", "empty_source_free"]
CandidateAuthorizationPurpose = Literal[
    "preapproval_preflight",
    "approved_verification",
]


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
    target_path: str
    content: str
    anchor: str
    summary: str
    evidence_ids: list[str]
    full_file_rationale: str
    expected_source_sha256: str
    expected_candidate_revision: int


class CanonicalPatchOperationRecord(TypedDict):
    """One candidate-bound operation retained through review and apply."""

    operation_id: str
    kind: PatchOperationKind
    source_path: str | None
    target_path: str | None
    expected_source_sha256: str | None
    expected_candidate_revision: int
    result_sha256: str | None
    content_sha256: str | None


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


class PatchApprovalBinding(TypedDict):
    """Internal action-loop binding between approval and reviewed candidate."""

    schema_version: Literal["coding_action_loop_approval_binding.v1"]
    proposal_digest: str
    candidate_revision: int
    candidate_tree_digest: str
    approval_evidence_digest: str
    source_message_id: str


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
    candidate_baseline: CandidateBaseline
    authorization_purpose: CandidateAuthorizationPurpose
    canonical_operation_records: list[CanonicalPatchOperationRecord]
    proposal_digest: str
    candidate_revision: int
    candidate_tree_digest: str
    approval_binding: PatchApprovalBinding
    apply_package_id: str


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
    canonical_operation_records: list[CanonicalPatchOperationRecord]
    proposal_digest: str
    candidate_tree_digest: str


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
