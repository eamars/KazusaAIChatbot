"""Public data contracts for code fetching."""

from typing import Literal, TypedDict

SourceKind = Literal["repository", "directory", "file"]
ResultStatus = Literal["succeeded", "failed", "needs_user_input", "rejected"]
StorageKind = Literal["existing_local_checkout", "managed_clone"]
DirtyState = Literal["clean", "dirty", "unknown"]


class CodeFetchingRequest(TypedDict, total=False):
    """Typed request accepted by `code_fetching.run`.

    Args:
        question: User-visible task text that may contain GitHub links.
        source_url: Explicit code source URL.
        repo_url: Explicit repository URL.
        repo_hint: GitHub shorthand such as `owner/repo`.
        local_root_hint: Existing local Git checkout root.
        local_path_hint: Existing local file or directory inside a checkout.
        requested_ref: Optional branch, tag, or commit supplied by the caller.
        source_scope_hint: Optional desired scope when the source is otherwise
            repository-level.
        workspace_root: Managed checkout workspace root.

    Returns:
        A source-resolution request. At least one source field or embedded
        GitHub URL is required.
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


class CodeSourceScope(TypedDict):
    """Resolved repository-relative source scope."""

    kind: SourceKind
    repo_relative_path: str | None
    source_url: str
    requested_ref: str | None
    interpretation: str


class CodeRepositoryRef(TypedDict):
    """Resolved repository checkout contract."""

    provider: Literal["github"]
    owner: str
    repo: str
    source_url: str
    requested_ref: str | None
    resolved_ref: str
    current_commit: str
    default_branch: str
    local_root: str
    storage_kind: StorageKind
    managed_checkout: bool
    workspace_root: str | None
    cache_key: str | None
    dirty_state: DirtyState


class CodeFetchingResult(TypedDict):
    """Public code-fetching result returned to upstream callers."""

    status: ResultStatus
    message: str
    repository: CodeRepositoryRef | None
    source_scope: CodeSourceScope | None
    limitations: list[str]
    trace_summary: list[str]
