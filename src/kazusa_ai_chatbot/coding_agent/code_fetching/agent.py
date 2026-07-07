"""Orchestration for the code-fetching subagent."""

from pathlib import Path

from kazusa_ai_chatbot.coding_agent.code_fetching import github
from kazusa_ai_chatbot.coding_agent.code_fetching import local_checkout
from kazusa_ai_chatbot.coding_agent.code_fetching import managed_inline
from kazusa_ai_chatbot.coding_agent.code_fetching import managed_clone
from kazusa_ai_chatbot.coding_agent.code_fetching import managed_download
from kazusa_ai_chatbot.coding_agent.code_fetching import source_resolver
from kazusa_ai_chatbot.coding_agent.code_fetching.github import GitHubSource
from kazusa_ai_chatbot.coding_agent.code_fetching.managed_inline import (
    InlineSourceBundle,
)
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

    source_selection = await source_resolver.select_source_for_request(
        request,
        trace_summary,
    )
    if source_selection.status != "succeeded":
        result = _result(
            status=source_selection.status,
            message=source_selection.message,
            repository=None,
            source_scope=None,
            limitations=list(source_selection.limitations),
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

    if isinstance(source, InlineSourceBundle):
        try:
            repository, source_scope_result = (
                managed_inline.materialize_inline_source_bundle(
                    source,
                    workspace_root,
                )
            )
        except managed_inline.ManagedInlineSourceError:
            result = _managed_inline_failure_result(trace_summary)
            return result
        trace_summary.append(
            f"managed_inline:materialized:{len(source.fragments)}"
        )
        result = _result(
            status="succeeded",
            message="Inline code source resolved.",
            repository=repository,
            source_scope=source_scope_result,
            limitations=list(source_selection.limitations),
            trace_summary=trace_summary,
        )
        return result

    if not isinstance(source, GitHubSource):
        result = _result(
            status="failed",
            message="Source selection returned an unsupported source type.",
            repository=None,
            source_scope=None,
            limitations=["Source selection returned an unsupported source type."],
            trace_summary=trace_summary,
        )
        return result

    if github.is_raw_github_source(source):
        try:
            repository = managed_download.ensure_managed_raw_file_download(
                source,
                workspace_root,
            )
        except managed_download.ManagedDownloadError as exc:
            result = _managed_download_failure_result(exc, trace_summary)
            return result
    else:
        try:
            repository = managed_clone.ensure_managed_checkout(
                source,
                workspace_root,
            )
        except managed_clone.ManagedCloneError as exc:
            result = _managed_clone_failure_result(exc, trace_summary)
            return result

    source_scope_error = _source_scope_validation_error(repository, source)
    if source_scope_error:
        limitations = [source_scope_error, *source_selection.limitations]
        result = _result(
            status="needs_user_input",
            message=source_scope_error,
            repository=None,
            source_scope=None,
            limitations=limitations,
            trace_summary=trace_summary,
        )
        return result

    source_scope_result = _source_scope_from_github_source(source)
    if repository["storage_kind"] == "managed_download":
        trace_summary.append("Resolved raw GitHub file to managed download.")
    else:
        trace_summary.append("Resolved public GitHub source to managed checkout.")
    result = _result(
        status="succeeded",
        message="Code source resolved.",
        repository=repository,
        source_scope=source_scope_result,
        limitations=list(source_selection.limitations),
        trace_summary=trace_summary,
    )
    return result


def _managed_clone_failure_result(
    exc: managed_clone.ManagedCloneError,
    trace_summary: list[str],
) -> CodeFetchingResult:
    if _managed_clone_error_is_source_access_failure(exc):
        result = _result(
            status="needs_user_input",
            message="Unable to access requested GitHub source.",
            repository=None,
            source_scope=None,
            limitations=[
                "GitHub source is unavailable or the requested ref is invalid.",
            ],
            trace_summary=trace_summary,
        )
        return result

    result = _result(
        status="failed",
        message="Unable to prepare managed checkout.",
        repository=None,
        source_scope=None,
        limitations=["Managed checkout preparation failed."],
        trace_summary=trace_summary,
    )
    return result


def _managed_download_failure_result(
    exc: managed_download.ManagedDownloadError,
    trace_summary: list[str],
) -> CodeFetchingResult:
    if _managed_download_error_is_source_access_failure(exc):
        result = _result(
            status="needs_user_input",
            message="Unable to access requested raw GitHub file.",
            repository=None,
            source_scope=None,
            limitations=[
                "Raw GitHub file is unavailable or exceeds the download limit.",
            ],
            trace_summary=trace_summary,
        )
        return result

    result = _result(
        status="failed",
        message="Unable to prepare managed raw file download.",
        repository=None,
        source_scope=None,
        limitations=["Managed raw file download failed."],
        trace_summary=trace_summary,
    )
    return result


def _managed_inline_failure_result(
    trace_summary: list[str],
) -> CodeFetchingResult:
    result = _result(
        status="failed",
        message="Unable to prepare managed inline source bundle.",
        repository=None,
        source_scope=None,
        limitations=["Managed inline source preparation failed."],
        trace_summary=trace_summary,
    )
    return result


def _managed_clone_error_is_source_access_failure(
    exc: managed_clone.ManagedCloneError,
) -> bool:
    message = str(exc)
    prefixes = (
        "managed git clone failed:",
        "managed requested ref fetch failed:",
        "managed requested ref checkout failed:",
    )
    return message.startswith(prefixes)


def _managed_download_error_is_source_access_failure(
    exc: managed_download.ManagedDownloadError,
) -> bool:
    message = str(exc)
    prefixes = (
        "raw GitHub file download failed:",
        "raw GitHub file exceeds managed download size limit.",
    )
    return message.startswith(prefixes)


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
