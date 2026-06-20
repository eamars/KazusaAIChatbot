"""Orchestration for the code-fetching subagent."""

from dataclasses import dataclass
from pathlib import Path

from kazusa_ai_chatbot.coding_agent.code_fetching import github
from kazusa_ai_chatbot.coding_agent.code_fetching import local_checkout
from kazusa_ai_chatbot.coding_agent.code_fetching import managed_clone
from kazusa_ai_chatbot.coding_agent.code_fetching import source_scope
from kazusa_ai_chatbot.coding_agent.code_fetching.github import GitHubSource
from kazusa_ai_chatbot.coding_agent.code_fetching.models import (
    CodeFetchingRequest,
    CodeFetchingResult,
    CodeRepositoryRef,
    CodeSourceScope,
    ResultStatus,
)
from kazusa_ai_chatbot.coding_agent.tools.paths import (
    PathSafetyError,
    ensure_path_inside,
)


@dataclass(frozen=True)
class _SourceSelection:
    status: ResultStatus
    message: str
    source: GitHubSource | None
    limitations: list[str]


async def run(request: CodeFetchingRequest) -> CodeFetchingResult:
    """Resolve one source-code location into a local repository contract.

    Args:
        request: Public code-fetching request.

    Returns:
        Structured result with repository and source scope on success.
    """

    trace_summary: list[str] = []

    local_path_hint = request.get("local_path_hint")
    if local_path_hint:
        result = _resolve_local_path(local_path_hint, trace_summary)
        return result

    local_root_hint = request.get("local_root_hint")
    if local_root_hint:
        result = _resolve_local_root(local_root_hint, trace_summary)
        return result

    source_selection = _select_github_source(request, trace_summary)
    if source_selection.status != "succeeded":
        result = _result(
            status=source_selection.status,
            message=source_selection.message,
            repository=None,
            source_scope=None,
            limitations=source_selection.limitations,
            trace_summary=trace_summary,
        )
        return result

    source = source_selection.source
    if source is None:
        result = _result(
            status="failed",
            message="Source selection failed unexpectedly.",
            repository=None,
            source_scope=None,
            limitations=["Source selection returned no source."],
            trace_summary=trace_summary,
        )
        return result

    workspace_root = request.get("workspace_root")
    if not workspace_root:
        workspace_root = managed_clone.default_workspace_root()
        trace_summary.append("Using standalone temp coding workspace.")

    try:
        repository = managed_clone.ensure_managed_checkout(
            source,
            workspace_root,
        )
    except managed_clone.ManagedCloneError as exc:
        result = _result(
            status="failed",
            message=f"Unable to prepare managed checkout: {exc}",
            repository=None,
            source_scope=None,
            limitations=["Managed checkout preparation failed."],
            trace_summary=trace_summary,
        )
        return result

    source_scope_error = _source_scope_validation_error(repository, source)
    if source_scope_error:
        limitations = [source_scope_error, *source_selection.limitations]
        result = _result(
            status="rejected",
            message=source_scope_error,
            repository=None,
            source_scope=None,
            limitations=limitations,
            trace_summary=trace_summary,
        )
        return result

    source_scope_result = _source_scope_from_github_source(source)
    trace_summary.append("Resolved public GitHub source to managed checkout.")
    result = _result(
        status="succeeded",
        message="Code source resolved.",
        repository=repository,
        source_scope=source_scope_result,
        limitations=source_selection.limitations,
        trace_summary=trace_summary,
    )
    return result


def _resolve_local_path(
    local_path_hint: str,
    trace_summary: list[str],
) -> CodeFetchingResult:
    try:
        repository, scope = local_checkout.resolve_local_path_hint(local_path_hint)
    except (
        local_checkout.LocalCheckoutError,
        local_checkout.PathSafetyError,
    ) as exc:
        result = _result(
            status="rejected",
            message=f"Local path source is unsupported: {exc}",
            repository=None,
            source_scope=None,
            limitations=["Local path must be inside a GitHub-backed checkout."],
            trace_summary=trace_summary,
        )
        return result

    trace_summary.append("Resolved explicit local path without checkout mutation.")
    result = _result(
        status="succeeded",
        message="Local code source resolved.",
        repository=repository,
        source_scope=scope,
        limitations=[],
        trace_summary=trace_summary,
    )
    return result


def _resolve_local_root(
    local_root_hint: str,
    trace_summary: list[str],
) -> CodeFetchingResult:
    try:
        repository = local_checkout.resolve_existing_checkout(local_root_hint)
    except (
        local_checkout.LocalCheckoutError,
        local_checkout.PathSafetyError,
    ) as exc:
        result = _result(
            status="rejected",
            message=f"Local checkout source is unsupported: {exc}",
            repository=None,
            source_scope=None,
            limitations=["Local root must be a GitHub-backed checkout."],
            trace_summary=trace_summary,
        )
        return result

    scope = local_checkout.source_scope_for_repository(repository)
    trace_summary.append("Resolved explicit local checkout without mutation.")
    result = _result(
        status="succeeded",
        message="Local checkout resolved.",
        repository=repository,
        source_scope=scope,
        limitations=[],
        trace_summary=trace_summary,
    )
    return result


def _select_github_source(
    request: CodeFetchingRequest,
    trace_summary: list[str],
) -> _SourceSelection:
    candidates: list[GitHubSource] = []
    unsupported_reasons: list[str] = []

    explicit_values = _explicit_source_values(request)
    for source_text in explicit_values:
        _collect_source_text(
            source_text,
            candidates=candidates,
            unsupported_reasons=unsupported_reasons,
        )

    question = request.get("question")
    if question:
        for url in github.extract_http_urls(question):
            _collect_source_text(
                url,
                candidates=candidates,
                unsupported_reasons=unsupported_reasons,
            )

    repo_hint = request.get("repo_hint")
    if repo_hint:
        source = github.parse_repo_hint(repo_hint)
        if source is None:
            unsupported_reasons.append("repo_hint must use owner/repo format.")
        else:
            candidates.append(source)

    requested_ref = request.get("requested_ref")
    if requested_ref:
        candidates = [
            github.with_requested_ref(candidate, requested_ref)
            for candidate in candidates
        ]

    if not candidates and unsupported_reasons:
        return _SourceSelection(
            status="rejected",
            message=unsupported_reasons[0],
            source=None,
            limitations=unsupported_reasons,
        )

    selection = source_scope.choose_source(candidates)
    if selection.status == "needs_user_input":
        return _SourceSelection(
            status="needs_user_input",
            message=selection.message,
            source=None,
            limitations=unsupported_reasons,
        )

    if selection.source is None:
        return _SourceSelection(
            status="failed",
            message="Source selection failed unexpectedly.",
            source=None,
            limitations=["Source selection returned no source."],
        )

    source_scope_hint = request.get("source_scope_hint")
    if source_scope_hint and selection.source.source_kind != source_scope_hint:
        return _SourceSelection(
            status="needs_user_input",
            message=(
                "source_scope_hint conflicts with the resolved source shape; "
                "provide a matching source URL or local path."
            ),
            source=None,
            limitations=unsupported_reasons,
        )

    trace_summary.append("Selected one supported GitHub source.")
    return _SourceSelection(
        status="succeeded",
        message="Source candidate selected.",
        source=selection.source,
        limitations=unsupported_reasons,
    )


def _explicit_source_values(request: CodeFetchingRequest) -> list[str]:
    values: list[str] = []
    source_url = request.get("source_url")
    if source_url:
        values.append(source_url)

    repo_url = request.get("repo_url")
    if repo_url:
        values.append(repo_url)

    return values


def _collect_source_text(
    source_text: str,
    *,
    candidates: list[GitHubSource],
    unsupported_reasons: list[str],
) -> None:
    source = github.parse_github_source(source_text)
    if source is not None:
        candidates.append(source)
        return

    reason = github.unsupported_source_reason(source_text)
    if reason:
        redacted_text = github.redact_source_text(source_text)
        unsupported_reasons.append(f"{reason} Source: {redacted_text}")


def _source_scope_from_github_source(source: GitHubSource) -> CodeSourceScope:
    interpretation = f"public GitHub {source.source_kind} source"
    scope: CodeSourceScope = {
        "kind": source.source_kind,
        "repo_relative_path": source.repo_relative_path,
        "source_url": source.source_url,
        "requested_ref": source.requested_ref,
        "interpretation": interpretation,
    }
    return scope


def _source_scope_validation_error(
    repository: CodeRepositoryRef,
    source: GitHubSource,
) -> str | None:
    """Validate that a parsed scoped source exists inside the checkout.

    Args:
        repository: Resolved checkout metadata containing the local root.
        source: Parsed source scope requested by the caller.

    Returns:
        A public-safe rejection message when the scope is invalid, otherwise
        `None`.
    """

    if source.repo_relative_path is None:
        return None

    repo_root = Path(repository["local_root"]).resolve(strict=False)
    candidate_path = repo_root / source.repo_relative_path

    try:
        scoped_path = ensure_path_inside(candidate_path, repo_root)
    except PathSafetyError as exc:
        message = f"Resolved {source.source_kind} scope is unsafe: {exc}"
        return message

    if source.source_kind == "file" and not scoped_path.is_file():
        message = (
            "Resolved file scope does not exist in checkout: "
            f"{source.repo_relative_path}"
        )
        return message

    if source.source_kind == "directory" and not scoped_path.is_dir():
        message = (
            "Resolved directory scope does not exist in checkout: "
            f"{source.repo_relative_path}"
        )
        return message

    return None


def _result(
    *,
    status: ResultStatus,
    message: str,
    repository: CodeRepositoryRef | None,
    source_scope: CodeSourceScope | None,
    limitations: list[str],
    trace_summary: list[str],
) -> CodeFetchingResult:
    bounded_trace = trace_summary[:]
    result: CodeFetchingResult = {
        "status": status,
        "message": message,
        "repository": repository,
        "source_scope": source_scope,
        "limitations": limitations,
        "trace_summary": bounded_trace,
    }
    return result
