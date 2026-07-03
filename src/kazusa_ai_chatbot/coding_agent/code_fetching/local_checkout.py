"""Read-only metadata extraction for existing local checkouts."""

from pathlib import Path

from kazusa_ai_chatbot.coding_agent.code_fetching.github import (
    is_safe_repo_relative_path,
    parse_github_source,
)
from kazusa_ai_chatbot.coding_agent.code_fetching.models import (
    CodeRepositoryRef,
    CodeSourceScope,
)
from kazusa_ai_chatbot.coding_agent.tools.git import (
    GitCommandError,
    run_git_command,
)
from kazusa_ai_chatbot.coding_agent.tools.paths import (
    PathSafetyError,
    resolve_existing_path,
)


class LocalCheckoutError(RuntimeError):
    """Raised when a local path cannot be resolved as a GitHub checkout."""


def resolve_existing_checkout(local_root_hint: str) -> CodeRepositoryRef:
    """Resolve a GitHub-backed local checkout without mutating it.

    Args:
        local_root_hint: Existing checkout path.

    Returns:
        Repository metadata for the local checkout.
    """

    local_path = resolve_existing_path(local_root_hint)
    if not local_path.is_dir():
        message = "local_root_hint must point to a directory."
        raise LocalCheckoutError(message)

    repo_root = _git_toplevel(local_path)
    _reject_unsafe_local_scope(local_path, repo_root)
    repository = _repository_from_root(repo_root)
    return repository


def resolve_local_path_hint(
    local_path_hint: str,
) -> tuple[CodeRepositoryRef, CodeSourceScope]:
    """Resolve a file or directory inside a GitHub-backed local checkout.

    Args:
        local_path_hint: Existing file or directory path.

    Returns:
        Repository metadata and a repository-relative source scope.
    """

    local_path = resolve_existing_path(local_path_hint)
    git_cwd = local_path if local_path.is_dir() else local_path.parent
    repo_root = _git_toplevel(git_cwd)
    if not local_path.is_relative_to(repo_root):
        message = "local_path_hint is not inside the resolved Git checkout."
        raise LocalCheckoutError(message)

    _reject_unsafe_local_scope(local_path, repo_root)
    repository = _repository_from_root(repo_root)
    relative_path = local_path.relative_to(repo_root).as_posix()
    scope_kind = "directory" if local_path.is_dir() else "file"
    source_scope: CodeSourceScope = {
        "kind": scope_kind,
        "repo_relative_path": relative_path,
        "source_url": _public_local_source_url(repository, relative_path),
        "requested_ref": repository["resolved_ref"],
        "interpretation": f"explicit local {scope_kind} path",
    }
    return repository, source_scope


def source_scope_for_repository(
    repository: CodeRepositoryRef,
) -> CodeSourceScope:
    """Build a repository-level source scope for a local checkout.

    Args:
        repository: Resolved local repository metadata.

    Returns:
        Repository-level source scope.
    """

    scope: CodeSourceScope = {
        "kind": "repository",
        "repo_relative_path": None,
        "source_url": _public_local_source_url(repository, None),
        "requested_ref": repository["resolved_ref"],
        "interpretation": "explicit local repository checkout",
    }
    return scope


def _repository_from_root(repo_root: Path) -> CodeRepositoryRef:
    remote_url = _git_output(["config", "--get", "remote.origin.url"], repo_root)
    source = parse_github_source(remote_url)
    if source is None:
        message = "local checkout must have a public GitHub origin remote."
        raise LocalCheckoutError(message)

    current_commit = _git_output(["rev-parse", "HEAD"], repo_root)
    resolved_ref = _current_branch(repo_root)
    default_branch = _default_branch(repo_root)
    dirty_output = _git_output(["status", "--porcelain"], repo_root)
    dirty_state = "dirty" if dirty_output else "clean"

    repository: CodeRepositoryRef = {
        "provider": "github",
        "owner": source.owner,
        "repo": source.repo,
        "source_url": f"https://github.com/{source.owner}/{source.repo}",
        "requested_ref": None,
        "resolved_ref": resolved_ref,
        "current_commit": current_commit,
        "default_branch": default_branch,
        "local_root": str(repo_root),
        "storage_kind": "existing_local_checkout",
        "managed_checkout": False,
        "workspace_root": None,
        "cache_key": None,
        "dirty_state": dirty_state,
    }
    return repository


def _reject_unsafe_local_scope(local_path: Path, repo_root: Path) -> None:
    if local_path == repo_root or not local_path.is_relative_to(repo_root):
        return

    relative_path = local_path.relative_to(repo_root).as_posix()
    if is_safe_repo_relative_path(relative_path):
        return

    message = "local path cannot target repository-internal .git or traversal paths."
    raise LocalCheckoutError(message)


def _public_local_source_url(
    repository: CodeRepositoryRef,
    repo_relative_path: str | None,
) -> str:
    """Build a public-safe source label for local checkout scopes.

    Args:
        repository: Resolved local checkout metadata.
        repo_relative_path: Optional path inside the checkout.

    Returns:
        Stable local source label that does not expose an absolute path.
    """

    base_url = f"local://github/{repository['owner']}/{repository['repo']}"
    if repo_relative_path is None:
        source_url = base_url
        return source_url

    source_url = f"{base_url}/{repo_relative_path}"
    return source_url


def _git_toplevel(path: Path) -> Path:
    try:
        result = run_git_command(["rev-parse", "--show-toplevel"], cwd=str(path))
    except GitCommandError as exc:
        message = f"path is not inside a Git checkout: {exc}"
        raise LocalCheckoutError(message) from exc

    try:
        repo_root = Path(result.stdout).resolve(strict=True)
    except OSError as exc:
        message = f"resolved Git checkout root is unavailable: {exc}"
        raise LocalCheckoutError(message) from exc

    return repo_root


def _git_output(args: list[str], cwd: Path) -> str:
    try:
        result = run_git_command(args, cwd=str(cwd))
    except GitCommandError as exc:
        message = f"Git metadata command failed: {exc}"
        raise LocalCheckoutError(message) from exc

    return result.stdout


def _git_output_optional(args: list[str], cwd: Path) -> str | None:
    try:
        result = run_git_command(args, cwd=str(cwd))
    except GitCommandError:
        return None

    return result.stdout


def _current_branch(repo_root: Path) -> str:
    branch = _git_output_optional(["branch", "--show-current"], repo_root)
    if branch:
        return branch

    commit = _git_output(["rev-parse", "--short", "HEAD"], repo_root)
    detached_ref = f"detached:{commit}"
    return detached_ref


def _default_branch(repo_root: Path) -> str:
    origin_head = _git_output_optional(
        ["symbolic-ref", "--short", "refs/remotes/origin/HEAD"],
        repo_root,
    )
    if origin_head and origin_head.startswith("origin/"):
        default_branch = origin_head.removeprefix("origin/")
        return default_branch
    if origin_head:
        return origin_head

    current_branch = _current_branch(repo_root)
    return current_branch


__all__ = [
    "LocalCheckoutError",
    "PathSafetyError",
    "resolve_existing_checkout",
    "resolve_local_path_hint",
    "source_scope_for_repository",
]
