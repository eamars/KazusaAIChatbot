"""Managed public GitHub clone storage."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile
from typing import TypedDict

from kazusa_ai_chatbot.coding_agent.models import (
    DEFAULT_WORKSPACE_DIR_NAME,
    MANAGED_METADATA_SCHEMA_VERSION,
)
from kazusa_ai_chatbot.coding_agent.code_fetching.github import GitHubSource
from kazusa_ai_chatbot.coding_agent.code_fetching.models import CodeRepositoryRef
from kazusa_ai_chatbot.coding_agent.tools.git import (
    GitCommandError,
    run_git_command,
)
from kazusa_ai_chatbot.coding_agent.tools.paths import ensure_path_inside

_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")


class ManagedCheckoutPaths(TypedDict):
    """Managed checkout path bundle."""

    workspace_root: str
    ref_key: str
    checkout_root: str
    metadata_path: str
    temporary_root: str
    lock_path: str
    cache_key: str


class ManagedCloneError(RuntimeError):
    """Raised when a managed checkout cannot be created or reused."""


def default_workspace_root() -> str:
    """Return the standalone fallback workspace root.

    Returns:
        OS-temp workspace path used when direct callers do not pass a root.
    """

    workspace_root = str(Path(tempfile.gettempdir()) / DEFAULT_WORKSPACE_DIR_NAME)
    return workspace_root


def build_managed_checkout_paths(
    *,
    workspace_root: str,
    provider: str,
    owner: str,
    repo: str,
    requested_ref: str | None,
) -> ManagedCheckoutPaths:
    """Build contained managed-checkout paths for one repository ref.

    Args:
        workspace_root: Configured coding-agent workspace root.
        provider: Source provider label.
        owner: Repository owner.
        repo: Repository name.
        requested_ref: Requested branch, tag, or commit.

    Returns:
        Path bundle for checkout, metadata, lock, and temporary clone location.
    """

    root = Path(workspace_root).expanduser().resolve(strict=False)
    ref_value = requested_ref or "default"
    ref_slug = _safe_slug(ref_value)
    ref_hash = hashlib.sha1(ref_value.encode("utf-8")).hexdigest()[:12]
    ref_key = f"{ref_slug}-{ref_hash}"
    owner_slug = _safe_slug(owner)
    repo_slug = _safe_slug(repo)
    provider_slug = _safe_slug(provider)
    cache_key = f"{provider_slug}-{owner_slug}-{repo_slug}-{ref_key}"

    checkout_root = (
        root
        / "repos"
        / provider_slug
        / owner_slug
        / repo_slug
        / "refs"
        / ref_key
        / "checkout"
    )
    metadata_path = checkout_root.parent / "metadata.json"
    lock_path = root / "locks" / f"{cache_key}.lock"
    temporary_root = root / "tmp" / f"{cache_key}.tmp"

    ensure_path_inside(checkout_root, root)
    ensure_path_inside(metadata_path, root)
    ensure_path_inside(lock_path, root)
    ensure_path_inside(temporary_root, root)

    paths: ManagedCheckoutPaths = {
        "workspace_root": str(root),
        "ref_key": ref_key,
        "checkout_root": str(checkout_root),
        "metadata_path": str(metadata_path),
        "temporary_root": str(temporary_root),
        "lock_path": str(lock_path),
        "cache_key": cache_key,
    }
    return paths


def write_metadata(metadata_path: str, metadata: dict[str, object]) -> None:
    """Write managed checkout metadata as UTF-8 JSON.

    Args:
        metadata_path: Metadata file path.
        metadata: JSON-serializable metadata.
    """

    path = Path(metadata_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def can_reuse_managed_checkout(
    metadata_path: str,
    expected_metadata: dict[str, object],
) -> bool:
    """Check whether existing metadata matches a requested managed source.

    Args:
        metadata_path: Existing metadata path.
        expected_metadata: Expected source identity fields.

    Returns:
        `True` only when all expected keys match the metadata file.
    """

    path = Path(metadata_path)
    if not path.exists():
        return False

    try:
        raw_text = path.read_text(encoding="utf-8")
        stored_metadata = json.loads(raw_text)
    except (OSError, json.JSONDecodeError):
        return False

    for key, expected_value in expected_metadata.items():
        stored_value = stored_metadata.get(key)
        if stored_value != expected_value:
            return False

    return True


def ensure_managed_checkout(
    source: GitHubSource,
    workspace_root: str,
) -> CodeRepositoryRef:
    """Create or reuse a managed public GitHub checkout.

    Args:
        source: Parsed GitHub source.
        workspace_root: Configured workspace root.

    Returns:
        Repository metadata for the managed checkout.
    """

    paths = build_managed_checkout_paths(
        workspace_root=workspace_root,
        provider="github",
        owner=source.owner,
        repo=source.repo,
        requested_ref=source.requested_ref,
    )
    checkout_root = Path(paths["checkout_root"])
    metadata_path = Path(paths["metadata_path"])
    expected_identity = _expected_identity(source)

    if metadata_path.exists() and not can_reuse_managed_checkout(
        str(metadata_path),
        expected_identity,
    ):
        message = "managed checkout metadata does not match requested source."
        raise ManagedCloneError(message)

    if not checkout_root.exists():
        _clone_into_managed_checkout(source, paths)

    repository = _repository_from_managed_checkout(source, paths)
    metadata = _metadata_for_repository(source, repository)
    write_metadata(str(metadata_path), metadata)
    return repository


def _clone_into_managed_checkout(
    source: GitHubSource,
    paths: ManagedCheckoutPaths,
) -> None:
    checkout_root = Path(paths["checkout_root"])
    temporary_root = Path(paths["temporary_root"])
    workspace_root = Path(paths["workspace_root"])

    ensure_path_inside(temporary_root, workspace_root)
    ensure_path_inside(checkout_root, workspace_root)

    if temporary_root.exists():
        shutil.rmtree(temporary_root)

    checkout_root.parent.mkdir(parents=True, exist_ok=True)
    temporary_root.parent.mkdir(parents=True, exist_ok=True)

    clone_url = f"https://github.com/{source.owner}/{source.repo}.git"
    clone_args = ["clone", "--depth", "1", "--no-tags"]
    if source.requested_ref:
        clone_args.extend(["--branch", source.requested_ref])
    clone_args.extend([clone_url, str(temporary_root)])

    try:
        run_git_command(clone_args)
    except GitCommandError as exc:
        if temporary_root.exists():
            shutil.rmtree(temporary_root)
        message = f"managed git clone failed: {exc}"
        raise ManagedCloneError(message) from exc

    try:
        temporary_root.replace(checkout_root)
    except OSError as exc:
        if temporary_root.exists():
            shutil.rmtree(temporary_root)
        message = f"managed checkout move failed: {exc}"
        raise ManagedCloneError(message) from exc


def _repository_from_managed_checkout(
    source: GitHubSource,
    paths: ManagedCheckoutPaths,
) -> CodeRepositoryRef:
    checkout_root = Path(paths["checkout_root"])
    current_commit = _git_output(["rev-parse", "HEAD"], checkout_root)
    resolved_ref = source.requested_ref or _default_branch(checkout_root)
    default_branch = _default_branch(checkout_root)

    repository: CodeRepositoryRef = {
        "provider": "github",
        "owner": source.owner,
        "repo": source.repo,
        "source_url": f"https://github.com/{source.owner}/{source.repo}",
        "requested_ref": source.requested_ref,
        "resolved_ref": resolved_ref,
        "current_commit": current_commit,
        "default_branch": default_branch,
        "local_root": str(checkout_root),
        "storage_kind": "managed_clone",
        "managed_checkout": True,
        "workspace_root": paths["workspace_root"],
        "cache_key": paths["cache_key"],
        "dirty_state": "clean",
    }
    return repository


def _metadata_for_repository(
    source: GitHubSource,
    repository: CodeRepositoryRef,
) -> dict[str, object]:
    metadata: dict[str, object] = {
        "schema_version": MANAGED_METADATA_SCHEMA_VERSION,
        "provider": "github",
        "owner": source.owner,
        "repo": source.repo,
        "source_url": f"https://github.com/{source.owner}/{source.repo}",
        "requested_ref": source.requested_ref,
        "resolved_ref": repository["resolved_ref"],
        "current_commit": repository["current_commit"],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    return metadata


def _expected_identity(source: GitHubSource) -> dict[str, object]:
    identity: dict[str, object] = {
        "schema_version": MANAGED_METADATA_SCHEMA_VERSION,
        "provider": "github",
        "owner": source.owner,
        "repo": source.repo,
        "source_url": f"https://github.com/{source.owner}/{source.repo}",
        "requested_ref": source.requested_ref,
    }
    return identity


def _git_output(args: list[str], cwd: Path) -> str:
    try:
        result = run_git_command(args, cwd=str(cwd))
    except GitCommandError as exc:
        message = f"managed checkout metadata command failed: {exc}"
        raise ManagedCloneError(message) from exc

    return result.stdout


def _git_output_optional(args: list[str], cwd: Path) -> str | None:
    try:
        result = run_git_command(args, cwd=str(cwd))
    except GitCommandError:
        return None

    return result.stdout


def _default_branch(checkout_root: Path) -> str:
    origin_head = _git_output_optional(
        ["symbolic-ref", "--short", "refs/remotes/origin/HEAD"],
        checkout_root,
    )
    if origin_head and origin_head.startswith("origin/"):
        default_branch = origin_head.removeprefix("origin/")
        return default_branch
    if origin_head:
        return origin_head

    current_branch = _git_output_optional(["branch", "--show-current"], checkout_root)
    if current_branch:
        return current_branch

    fallback_branch = "HEAD"
    return fallback_branch


def _safe_slug(value: str) -> str:
    replaced_value = _SAFE_NAME_RE.sub("_", value).strip("._-")
    if replaced_value:
        return replaced_value[:80]

    fallback_slug = "default"
    return fallback_slug
