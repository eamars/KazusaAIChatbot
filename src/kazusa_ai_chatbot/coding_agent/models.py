"""Public contracts and shared constants for standalone coding-agent modules."""

from typing import Literal, NotRequired, TypedDict

DEFAULT_WORKSPACE_DIR_NAME = "kazusa_coding_agent"
GIT_COMMAND_TIMEOUT_SECONDS = 60
MANAGED_METADATA_SCHEMA_VERSION = 1
TRACE_ITEM_LIMIT = 12

SourceKind = Literal["repository", "directory", "file"]
ResultStatus = Literal["succeeded", "failed", "needs_user_input", "rejected"]
CodingAgentBackgroundOperation = Literal[
    "code_reading",
    "code_writing",
    "unsupported",
]


class CodingAgentSourceScope(TypedDict):
    """Public resolved repository-relative source scope."""

    kind: SourceKind
    repo_relative_path: str | None
    source_url: str
    requested_ref: str | None
    interpretation: str


class CodingAgentRequest(TypedDict, total=False):
    """Top-level direct coding-agent request.

    The fetching fields are passed through unchanged to source fetching.
    Reading fields are consumed only after fetching returns a successful
    repository contract.
    """

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


class CodingAgentWriteRequest(TypedDict, total=False):
    """Top-level direct coding-agent request for patch proposal work.

    The source fields are explicit structure used to decide whether source
    fetching and reading are required. Source-free requests are handled as
    new-project writing tasks.
    """

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
    session_id: str
    max_artifact_chars: int


class CodingAgentBackgroundRequest(TypedDict, total=False):
    """Accepted background coding task handled by the coding-agent supervisor."""

    question: str
    source_summary: str
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
    session_id: str


class CodingAgentRepositorySummary(TypedDict):
    """Public-safe repository metadata for direct callers and future workers."""

    provider: Literal["github"]
    owner: str
    repo: str
    source_url: str
    requested_ref: str | None
    resolved_ref: str
    current_commit: str
    default_branch: str
    storage_kind: str
    managed_checkout: bool
    dirty_state: str


class CodeEvidenceReference(TypedDict):
    """Public evidence row returned by the top-level coding agent."""

    path: str
    line_start: int
    line_end: int
    symbol_or_topic: str
    excerpt: str
    reason: str


class CodingAgentResponse(TypedDict):
    """Top-level direct coding-agent response."""

    status: ResultStatus
    answer_text: str
    repository: CodingAgentRepositorySummary | None
    source_scope: CodingAgentSourceScope | None
    evidence: list[CodeEvidenceReference]
    limitations: list[str]
    trace_summary: list[str]


class CodingPatchProposalResponse(TypedDict):
    """Top-level direct coding-agent response for patch proposal work."""

    status: ResultStatus
    mode: str
    answer_text: str
    repository: CodingAgentRepositorySummary | None
    source_scope: CodingAgentSourceScope | None
    evidence: list[CodeEvidenceReference]
    patch_artifacts: list[dict[str, object]]
    created_files: list[dict[str, str]]
    changed_files: list[dict[str, str]]
    validation: dict[str, object]
    alignment: NotRequired[dict[str, object]]
    external_evidence: list[dict[str, object]]
    session: dict[str, object] | None
    limitations: list[str]
    trace_summary: list[str]
    trace: NotRequired[dict[str, object]]


class CodingAgentBackgroundResponse(TypedDict):
    """Unified public response for background coding work."""

    status: ResultStatus
    operation: CodingAgentBackgroundOperation
    answer_text: str
    repository: CodingAgentRepositorySummary | None
    source_scope: CodingAgentSourceScope | None
    evidence: list[CodeEvidenceReference]
    patch_artifacts: list[dict[str, object]]
    created_files: list[dict[str, str]]
    changed_files: list[dict[str, str]]
    validation: dict[str, object] | None
    limitations: list[str]
    trace_summary: list[str]
