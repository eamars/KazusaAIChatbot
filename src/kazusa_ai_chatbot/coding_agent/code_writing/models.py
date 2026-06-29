"""Public and internal data contracts for new-artifact code writing."""

from typing import Literal, NotRequired, TypedDict

from kazusa_ai_chatbot.coding_agent.code_reading.models import (
    CodeReadingResult,
)

WritingMode = Literal["create_new_project", "edit_existing_repository"]
WritingPMStatus = Literal[
    "request_information",
    "create_child_pm",
    "create_programmer_task",
    "repair_child",
    "complete",
    "blocked",
]
WritingDomain = Literal["reading", "writing", "modifying"]
WritingResultStatus = Literal[
    "succeeded",
    "failed",
    "needs_user_input",
    "rejected",
    "need_external_evidence",
    "need_reading",
]
WritingFileKind = Literal["source", "test", "docs", "config", "data"]
WritingContentFormat = Literal["python", "markdown", "text", "json", "csv"]
WritingProgrammerStatus = Literal["succeeded", "blocked"]
WritingPatcherStatus = Literal["succeeded", "failed", "rejected"]
PatchValidationStatus = Literal["succeeded", "failed", "rejected"]
PatchOperationKind = Literal["create_file"]
WritingAlignmentStatus = Literal["pass", "fail"]


class PatchArtifact(TypedDict):
    """Unified-diff patch proposal for generated artifacts."""

    artifact_id: str
    base: str
    diff_text: str
    files: list[str]
    summary: str


class PatchOperation(TypedDict, total=False):
    """Structured new-file operation compiled into a unified diff."""

    operation_id: str
    kind: PatchOperationKind
    path: str
    content: str
    summary: str


class CreatedFileSummary(TypedDict):
    """Public summary for a file introduced by a proposal."""

    path: str
    role: str


class ChangedFileSummary(TypedDict):
    """Public summary for one file touched by proposed patch artifacts."""

    path: str
    change_type: str
    summary: str


class ExternalEvidenceSummary(TypedDict):
    """Captured external evidence requested by the writing PM."""

    request_id: str
    task: str
    resolved: bool
    result: str
    limitation: NotRequired[str]


class SupervisorFactSummary(TypedDict):
    """Compact facts resolved by the top-level coding supervisor."""

    request_id: str
    kind: str
    task: str
    resolved: bool
    result: str
    limitation: NotRequired[str]


class WritingExternalEvidenceRequest(TypedDict):
    """One supervisor-mediated fact request from the writing PM."""

    request_id: str
    task: str
    reason: str


class WritingReadingRequest(TypedDict):
    """One supervisor-mediated read-only source question from the writing PM."""

    request_id: str
    task: str
    reason: str
    target_artifacts: list[str]


class PatchValidationSummary(TypedDict):
    """Public-safe patch validation result."""

    status: PatchValidationStatus
    parsed: bool
    sandbox_applied: bool
    errors: list[str]
    warnings: list[str]
    files: list[str]


class WritingAcceptanceCriterion(TypedDict):
    """One preserved user-visible requirement for artifact alignment."""

    criterion_id: str
    requirement: str
    evidence_needed: str


class WritingAcceptanceResult(TypedDict):
    """Requirement preservation result before artifact decomposition."""

    status: WritingAlignmentStatus
    acceptance_criteria: list[WritingAcceptanceCriterion]
    limitations: list[str]


class WritingAlignmentResult(TypedDict):
    """Semantic artifact alignment result after artifact generation."""

    status: WritingAlignmentStatus
    confidence: int
    request_satisfied: bool
    reasons: list[str]
    blockers: list[str]
    feedback_for_pm: str


class WritingSessionSummary(TypedDict):
    """Public-safe persistent writing session handle."""

    session_id: str
    public_handle: str
    invalidated_previous: bool


class CodeWritingRequest(TypedDict, total=False):
    """Request accepted by `code_writing.run` after orchestration selection."""

    question: str
    mode_hint: WritingMode
    reading_result: CodeReadingResult | None
    supervisor_evidence_state: dict[str, object]
    workspace_root: str
    preferred_language: str
    session_id: str
    max_answer_chars: int
    max_artifact_chars: int
    external_evidence: list[ExternalEvidenceSummary]
    supervisor_facts: list[SupervisorFactSummary]


class WritingWorkItem(TypedDict):
    """One PM-owned unit of writing work."""

    goal: str
    scope: str
    constraints: list[str]
    expected_result: str


class WritingPMInput(TypedDict):
    """Compact model-facing input for one PM lifecycle decision."""

    pm_id: str
    domain: WritingDomain
    work_item: WritingWorkItem
    available_facts: list[dict[str, object]]
    direct_child_reports: list[dict[str, object]]
    child_feedback: list[dict[str, object]]
    context_limits: dict[str, object]


class WritingInformationRequest(TypedDict):
    """Facts requested before the PM can issue the next child instruction."""

    request_id: str
    needed_facts: list[str]
    target_artifacts: list[str]
    reason_for_next_instruction: str


class WritingChildPMTask(TypedDict):
    """Direct child PM task emitted by a parent PM."""

    child_pm_id: str
    domain: WritingDomain
    goal: str
    scope: str
    constraints: list[str]
    expected_report: list[str]


class WritingProgrammerTask(TypedDict):
    """Direct programmer task emitted by a PM."""

    task_id: str
    artifact_purpose: str
    required_behavior: list[str]
    provided_interfaces: list[str]
    consumed_interfaces: list[str]
    imports: list[str]
    output_format: str


class WritingRepairInstruction(TypedDict):
    """Repair request for one direct child."""

    child_id: str
    feedback: str
    expected_correction: str


class WritingCompletionReport(TypedDict):
    """Compact report from a PM to its parent or supervisor."""

    pm_id: str
    status: Literal["complete"]
    provided_facts: list[str]
    created_artifacts: list[dict[str, object]]
    consumed_facts: list[str]
    open_risks: list[str]
    next_dependency_needs: list[str]


class WritingBlocker(TypedDict):
    """Terminal blocker emitted by a PM."""

    summary: str
    missing_facts: list[str]
    why_information_request_is_not_enough: str


class WritingPMDecision(TypedDict):
    """The lifecycle action returned by a writing PM."""

    status: WritingPMStatus
    reason: str
    information_request: WritingInformationRequest | None
    child_pm_task: WritingChildPMTask | None
    programmer_task: WritingProgrammerTask | None
    repair_instruction: WritingRepairInstruction | None
    completion_report: WritingCompletionReport | None
    blocker: WritingBlocker | None


class WritingArtifactContract(TypedDict, total=False):
    """Internal one-artifact contract accepted for file and programmer work."""

    artifact_id: str
    file_label: str
    file_kind: WritingFileKind
    content_format: WritingContentFormat
    purpose: str
    imports: list[str]
    provided_interfaces: list[str]
    consumed_interfaces: list[str]
    required_behavior: list[str]
    preferred_name: str


class ReservedArtifactPath(TypedDict):
    """File Agent reservation for one new artifact."""

    artifact_id: str
    file_label: str
    path: str
    file_kind: WritingFileKind
    content_format: WritingContentFormat
    purpose: str


class ArtifactReservationResult(TypedDict):
    """File Agent result for accepted artifact contracts."""

    status: Literal["accepted", "repair_required"]
    reserved_paths: list[ReservedArtifactPath]
    errors: list[str]
    repair_feedback: list[str]


class WritingProgrammerContract(TypedDict):
    """One new-artifact contract for one programmer call."""

    artifact_id: str
    file_label: str
    file_kind: WritingFileKind
    content_format: WritingContentFormat
    purpose: str
    imports: list[str]
    provided_interfaces: list[str]
    consumed_interfaces: list[str]
    required_behavior: list[str]


class WritingProgrammerResult(TypedDict):
    """Source content returned by one writing programmer worker."""

    artifact_id: str
    status: WritingProgrammerStatus
    content_format: WritingContentFormat
    code_artifact: str
    diagnostics: list[str]


class GeneratedArtifact(TypedDict):
    """Generated artifact selected for patch materialization."""

    artifact_id: str
    file_label: str
    file_kind: WritingFileKind
    content_format: WritingContentFormat
    path: str
    content: str
    purpose: str


class WritingPatcherInput(TypedDict):
    """PM-selected generated artifacts handed to the materializer."""

    artifact_package_id: str
    artifacts: list[GeneratedArtifact]
    reserved_paths: list[ReservedArtifactPath]
    max_artifact_chars: int


class WritingPatcherReport(TypedDict):
    """Patch materialization result from the dedicated patcher boundary."""

    status: WritingPatcherStatus
    artifact_package: dict[str, object]
    patch_artifacts: list[PatchArtifact]
    created_files: list[CreatedFileSummary]
    changed_files: list[ChangedFileSummary]
    diagnostics: list[str]


class CodeWritingResult(TypedDict):
    """Patch proposal result from the code-writing subagent."""

    status: WritingResultStatus
    mode: WritingMode
    answer_text: str
    patch_artifacts: list[PatchArtifact]
    created_files: list[CreatedFileSummary]
    changed_files: list[ChangedFileSummary]
    reading_requests: NotRequired[list[WritingReadingRequest]]
    external_evidence_requests: list[WritingExternalEvidenceRequest]
    external_evidence: list[ExternalEvidenceSummary]
    reading_source: NotRequired[dict[str, object]]
    validation: PatchValidationSummary
    alignment: NotRequired[WritingAlignmentResult]
    session: WritingSessionSummary | None
    limitations: list[str]
    trace_summary: list[str]
    trace: NotRequired[dict[str, object]]
