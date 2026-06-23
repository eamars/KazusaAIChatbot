"""Public and internal data contracts for code writing."""

from typing import Literal, NotRequired, TypedDict

from kazusa_ai_chatbot.coding_agent.code_fetching.models import (
    CodeRepositoryRef,
    CodeSourceScope,
)
from kazusa_ai_chatbot.coding_agent.code_reading.models import (
    CodeEvidenceRow,
    CodeReadingResult,
)

WritingMode = Literal["edit_existing_repository", "create_new_project"]
WritingPMStatus = Literal[
    "need_reading",
    "need_module_pms",
    "ready_to_write",
    "needs_user_input",
    "overloaded",
    "rejected",
]
WritingResultStatus = Literal[
    "succeeded",
    "failed",
    "needs_user_input",
    "rejected",
    "need_reading",
    "need_external_evidence",
]
AssignmentScopeKind = Literal["repository", "file", "directory"]
WritingProgrammerReportStatus = Literal["succeeded", "blocked", "no_patch"]
WritingPatcherStatus = Literal["succeeded", "blocked"]
PatchValidationStatus = Literal["succeeded", "failed", "rejected"]
WritingPlanEvaluationStatus = Literal["accepted", "repair_required"]
ModuleContractEvaluationStatus = Literal["accepted", "repair_required"]
WritingFileKind = Literal["existing", "new", "test", "docs", "config", "support"]
WritingFileResolutionStatus = Literal["accepted", "repair_required"]
SourceOwnershipDecisionStatus = Literal[
    "accepted",
    "need_reading",
    "needs_pm_repair",
]
SourceOwnershipResolutionStatus = Literal[
    "accepted",
    "need_reading",
    "repair_required",
]
ModuleProgrammerEditMode = Literal["complete_file", "symbol_bundle"]
ModuleProgrammerContentFormat = Literal["python", "text"]
PatchOperationKind = Literal[
    "create_file",
    "insert_after",
    "insert_before",
    "replace",
]


class WritingAssignmentScope(TypedDict):
    """Internal bounded write scope used by supervisor and patcher helpers."""

    kind: AssignmentScopeKind
    values: list[str]


class PatchArtifact(TypedDict):
    """Unified-diff patch proposal."""

    artifact_id: str
    base: str
    diff_text: str
    files: list[str]
    summary: str


class PatchOperation(TypedDict, total=False):
    """LLM-selected structured edit compiled into a unified diff."""

    operation_id: str
    kind: PatchOperationKind
    path: str
    anchor: str
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


class PatchValidationSummary(TypedDict):
    """Public-safe patch validation result."""

    status: PatchValidationStatus
    parsed: bool
    sandbox_applied: bool
    errors: list[str]
    warnings: list[str]
    files: list[str]


class WritingFilePlanEvaluation(TypedDict):
    """Structural evaluation of resolved file/module contracts."""

    status: WritingPlanEvaluationStatus
    errors: list[str]
    repair_feedback: list[str]


class ModuleContractEvaluation(TypedDict):
    """Structural evaluation of one Module-PM programmer contract."""

    status: ModuleContractEvaluationStatus
    file_contract_id: str
    file_label: str
    errors: list[str]
    repair_feedback: list[str]


class SourceOwnerCandidate(TypedDict):
    """Structural source file hint derived from limited evidence files."""

    path: str
    role: str
    line_start: int
    line_end: int
    symbols: list[str]
    exception_types: list[str]
    feature_markers: list[str]
    reasons: list[str]
    evidence_refs: list[str]


class WritingFileDemand(TypedDict, total=False):
    """Semantic file need selected by the writing product manager."""

    demand_id: str
    role: str
    purpose: str
    file_kind: WritingFileKind
    preferred_path: str
    preferred_name: str
    placement_hint: str
    related_paths: list[str]
    read_only_paths: list[str]
    interface_contract: dict[str, object]
    integration_contract: dict[str, object]
    change_goal: str
    work_instructions: list[str]
    required_slots: list[str]
    validation_expectations: list[str]
    forbidden_paths: list[str]


class WritingFileResolution(TypedDict):
    """Resolved file plan returned before Module PM dispatch."""

    status: WritingFileResolutionStatus
    file_contracts: list["WritingFileModuleContract"]
    owned_path_map: dict[str, str]
    read_only_path_map: dict[str, list[str]]
    errors: list[str]
    repair_feedback: list[str]


class SourceOwnershipDecision(TypedDict):
    """One LLM-selected existing-source owner decision."""

    demand_id: str
    status: SourceOwnershipDecisionStatus
    owned_path: str
    read_only_paths: list[str]
    reason: str
    evidence_refs: list[str]
    required_slots: list[str]


class SourceOwnershipResolution(TypedDict):
    """Source-owner decisions returned before file mechanics run."""

    status: SourceOwnershipResolutionStatus
    decisions: list[SourceOwnershipDecision]
    errors: list[str]
    repair_feedback: list[str]
    reading_requests: list["WritingReadingEvidenceRequest"]


class WritingSessionSummary(TypedDict):
    """Public-safe persistent writing session handle."""

    session_id: str
    public_handle: str
    invalidated_previous: bool


class CodeWritingRequest(TypedDict, total=False):
    """Request accepted by `code_writing.run` after orchestration selection."""

    question: str
    mode_hint: WritingMode
    repository: CodeRepositoryRef | None
    source_scope: CodeSourceScope | None
    reading_result: CodeReadingResult | None
    supervisor_evidence_state: dict[str, object]
    workspace_root: str
    preferred_language: str
    session_id: str
    max_answer_chars: int
    max_artifact_chars: int
    external_evidence: list[ExternalEvidenceSummary]


class WritingExternalEvidenceRequest(TypedDict):
    """One PM-requested external evidence task."""

    request_id: str
    task: str
    reason: str


class WritingReadingEvidenceRequest(TypedDict):
    """One PM-requested source-reading evidence task."""

    request_id: str
    task: str
    reason: str
    required_slots: list[str]


class CodeWritingResult(TypedDict):
    """Patch proposal result from the code-writing subagent."""

    status: WritingResultStatus
    mode: WritingMode
    answer_text: str
    patch_artifacts: list[PatchArtifact]
    created_files: list[CreatedFileSummary]
    changed_files: list[ChangedFileSummary]
    reading_requests: NotRequired[list[WritingReadingEvidenceRequest]]
    external_evidence_requests: list[WritingExternalEvidenceRequest]
    external_evidence: list[ExternalEvidenceSummary]
    validation: PatchValidationSummary
    session: WritingSessionSummary | None
    limitations: list[str]
    trace_summary: list[str]
    trace: NotRequired[dict[str, object]]


class WritingPMInput(TypedDict):
    """Compact model-facing input for the writing product manager."""

    question: str
    mode: WritingMode
    repository_summary: dict[str, object] | None
    reading_reports: list[dict[str, object]]
    supervisor_evidence_state: NotRequired[dict[str, object]]
    owner_candidates: NotRequired[list[SourceOwnerCandidate]]
    previous_writing_reports: list["WritingProgrammerReport"]
    validation_feedback: NotRequired[PatchValidationSummary]
    file_resolution_feedback: NotRequired[WritingFileResolution]
    file_plan_feedback: NotRequired[WritingFilePlanEvaluation]
    external_evidence: NotRequired[list[ExternalEvidenceSummary]]


class WritingProgrammerAssignment(TypedDict, total=False):
    """Internal bounded write scope derived from a file contract."""

    assignment_id: str
    role: str
    scope: WritingAssignmentScope
    owned_paths: list[str]
    read_only_paths: list[str]
    interface_contract: dict[str, object]
    integration_contract: dict[str, object]
    change_goal: str
    work_instructions: list[str]
    required_slots: list[str]
    validation_expectations: list[str]
    forbidden_paths: list[str]


class WritingFileModuleContract(TypedDict, total=False):
    """One resolved file/module responsibility owned by a Module PM."""

    file_contract_id: str
    role: str
    purpose: str
    file_kind: WritingFileKind
    owned_path: str
    owned_paths: list[str]
    read_only_paths: list[str]
    interface_contract: dict[str, object]
    integration_contract: dict[str, object]
    change_goal: str
    cross_file_imports: list[str]
    work_instructions: list[str]
    required_slots: list[str]
    validation_expectations: list[str]
    forbidden_paths: list[str]


class ModuleProgrammerSymbol(TypedDict, total=False):
    """One top-level symbol required from a module programmer."""

    name: str
    kind: str
    signature: str
    body_contract: str
    children: list[object]


class ModuleProgrammerContract(TypedDict):
    """One module-level implementation contract for a programmer worker."""

    file_label: str
    edit_mode: ModuleProgrammerEditMode
    content_format: ModuleProgrammerContentFormat
    module_purpose: str
    lifecycle_owner: str
    provided_interfaces: list[dict[str, object]]
    consumed_interfaces: list[dict[str, object]]
    existing_source_anchors: list[dict[str, object]]
    imports: list[str]
    current_file_context: str
    symbols_to_define: list[ModuleProgrammerSymbol]
    symbols_to_modify: list[ModuleProgrammerSymbol]
    required_behavior: list[str]


class ModuleProgrammerResult(TypedDict):
    """Source content returned by a module programmer worker."""

    code_artifact: str


class CrossSliceInterfaceSummary(TypedDict):
    """Compact summary of one provided interface from another module slice."""

    provider_slice_id: str
    name: str
    contract: str


class ModulePMInput(TypedDict):
    """Model-facing input for one Module PM module contract."""

    file_label: str
    edit_mode: ModuleProgrammerEditMode
    content_format: ModuleProgrammerContentFormat
    module_purpose: str
    lifecycle_owner: str
    provided_interfaces: list[dict[str, object]]
    consumed_interfaces: list[dict[str, object]]
    existing_source_anchors: list[dict[str, object]]
    integration_behaviors: list[str]
    imports: list[str]
    current_file_context: str
    source_file_chars: int
    selected_evidence: list[dict[str, object]]
    required_behavior: list[str]
    cross_slice_interfaces: list[CrossSliceInterfaceSummary]
    module_contract_feedback: NotRequired[ModuleContractEvaluation]


class WritingPMDecision(TypedDict):
    """The decision shape returned by the writing product manager."""

    status: WritingPMStatus
    mode: WritingMode
    intent: str
    file_demands: list[WritingFileDemand]
    file_contracts: list[WritingFileModuleContract]
    cross_module_imports: dict[str, list[str]]
    missing_slots: list[str]
    reading_requests: list[WritingReadingEvidenceRequest]
    external_evidence_requests: list[WritingExternalEvidenceRequest]


class WritingProgrammerFact(TypedDict):
    """A compact programmer observation used by synthesis."""

    kind: str
    summary: str
    evidence_refs: list[str]


class WritingProgrammerReport(TypedDict):
    """Compressed memory returned by one programmer worker."""

    assignment_id: str
    file_contract_id: str
    file_label: str
    edit_mode: ModuleProgrammerEditMode
    status: WritingProgrammerReportStatus
    files_considered: list[str]
    facts: list[WritingProgrammerFact]
    code_artifact: str
    open_questions: list[str]
    created_files: list[CreatedFileSummary]
    changed_files: list[ChangedFileSummary]
    evidence: list[CodeEvidenceRow]


class WritingPatcherInput(TypedDict):
    """PM-selected implementation packet handed to the patch materializer."""

    question: str
    mode: WritingMode
    base_identity: str
    owned_path_map: dict[str, str]
    base_file_summaries: list[dict[str, object]]
    selected_programmer_reports: list[WritingProgrammerReport]
    pm_integration_notes: list[str]
    artifact_limits: dict[str, int]


class WritingPatcherReport(TypedDict):
    """Patch materialization result from the dedicated patcher role."""

    status: WritingPatcherStatus
    patch_artifacts: list[PatchArtifact]
    created_files: list[CreatedFileSummary]
    changed_files: list[ChangedFileSummary]
    edit_diagnostics: list[str]
    unmaterialized_reports: list[str]
