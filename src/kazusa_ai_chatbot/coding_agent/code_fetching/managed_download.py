"""Managed raw GitHub file download storage."""

from datetime import datetime, timezone
import hashlib
from http.client import HTTPException
from pathlib import Path
import shutil
import urllib.error
import urllib.request

from kazusa_ai_chatbot.coding_agent.models import (
    MANAGED_METADATA_SCHEMA_VERSION,
)
from kazusa_ai_chatbot.coding_agent.code_fetching.github import GitHubSource
from kazusa_ai_chatbot.coding_agent.code_fetching.managed_clone import (
    ManagedCheckoutPaths,
    build_managed_checkout_paths,
    can_reuse_managed_checkout,
    write_metadata,
)
from kazusa_ai_chatbot.coding_agent.code_fetching.models import CodeRepositoryRef
from kazusa_ai_chatbot.coding_agent.tools.paths import ensure_path_inside

RAW_FILE_DOWNLOAD_TIMEOUT_SECONDS = 60
MAX_RAW_FILE_BYTES = 2_000_000


class ManagedDownloadError(RuntimeError):
    """Raised when a managed raw file download cannot be created or reused."""


def ensure_managed_raw_file_download(
    source: GitHubSource,
    workspace_root: str,
) -> CodeRepositoryRef:
    """Create or reuse a managed workspace containing one raw GitHub file.

    Args:
        source: Parsed raw GitHub file source.
        workspace_root: Configured coding-agent workspace root.

    Returns:
        Repository metadata for the managed single-file workspace.
    """

    if source.source_kind != "file" or source.repo_relative_path is None:
        message = "managed raw download requires a file-scoped source."
        raise ManagedDownloadError(message)

    paths = build_managed_checkout_paths(
        workspace_root=workspace_root,
        provider="github_raw_file",
        owner=source.owner,
        repo=source.repo,
        requested_ref=source.requested_ref,
        require_checkout_path_budget=False,
    )
    checkout_root = Path(paths["checkout_root"])
    metadata_path = Path(paths["metadata_path"])
    expected_identity = _expected_identity(source)

    if checkout_root.exists() and not metadata_path.exists():
        message = "managed raw download exists without metadata."
        raise ManagedDownloadError(message)

    if metadata_path.exists() and not can_reuse_managed_checkout(
        str(metadata_path),
        expected_identity,
    ):
        message = "managed raw download metadata does not match requested source."
        raise ManagedDownloadError(message)

    downloaded_path = _downloaded_file_path(checkout_root, source)
    if metadata_path.exists() and downloaded_path.exists():
        try:
            data = downloaded_path.read_bytes()
        except OSError as exc:
            message = f"managed raw download read failed: {exc}"
            raise ManagedDownloadError(message) from exc
        content_identity = _content_identity(data)
        repository = _repository_from_download(source, paths, content_identity)
        return repository

    data = _download_raw_file(source.source_url)
    if len(data) > MAX_RAW_FILE_BYTES:
        message = "raw GitHub file exceeds managed download size limit."
        raise ManagedDownloadError(message)

    _write_managed_file(source, paths, data)
    content_identity = _content_identity(data)
    repository = _repository_from_download(source, paths, content_identity)
    metadata = _metadata_for_repository(source, repository)
    write_metadata(str(metadata_path), metadata)
    return repository


def _download_raw_file(source_url: str) -> bytes:
    request = urllib.request.Request(
        source_url,
        headers={"User-Agent": "KazusaCodingAgent/1.0"},
    )

    try:
        with urllib.request.urlopen(
            request,
            timeout=RAW_FILE_DOWNLOAD_TIMEOUT_SECONDS,
        ) as response:
            data = response.read()
    except (urllib.error.URLError, TimeoutError, OSError, HTTPException) as exc:
        message = f"raw GitHub file download failed: {exc}"
        raise ManagedDownloadError(message) from exc

    if not isinstance(data, bytes):
        message = "raw GitHub file download returned non-bytes content."
        raise ManagedDownloadError(message)

    return data


def _write_managed_file(
    source: GitHubSource,
    paths: ManagedCheckoutPaths,
    data: bytes,
) -> None:
    checkout_root = Path(paths["checkout_root"])
    temporary_root = Path(paths["temporary_root"])
    workspace_root = Path(paths["workspace_root"])

    ensure_path_inside(temporary_root, workspace_root)
    ensure_path_inside(checkout_root, workspace_root)

    try:
        if temporary_root.exists():
            shutil.rmtree(temporary_root)
        if checkout_root.exists():
            shutil.rmtree(checkout_root)
    except OSError as exc:
        message = f"managed raw download cleanup failed: {exc}"
        raise ManagedDownloadError(message) from exc

    target_path = _downloaded_file_path(temporary_root, source)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        target_path.write_bytes(data)
    except OSError as exc:
        if temporary_root.exists():
            try:
                shutil.rmtree(temporary_root)
            except OSError:
                pass
        message = f"managed raw download write failed: {exc}"
        raise ManagedDownloadError(message) from exc

    checkout_root.parent.mkdir(parents=True, exist_ok=True)
    try:
        temporary_root.replace(checkout_root)
    except OSError as exc:
        if temporary_root.exists():
            try:
                shutil.rmtree(temporary_root)
            except OSError:
                pass
        message = f"managed raw download move failed: {exc}"
        raise ManagedDownloadError(message) from exc


def _downloaded_file_path(
    root: Path,
    source: GitHubSource,
) -> Path:
    if source.repo_relative_path is None:
        message = "managed raw download requires a repository-relative path."
        raise ManagedDownloadError(message)

    target_path = ensure_path_inside(root / source.repo_relative_path, root)
    return target_path


def _repository_from_download(
    source: GitHubSource,
    paths: ManagedCheckoutPaths,
    content_identity: str,
) -> CodeRepositoryRef:
    resolved_ref = source.requested_ref or "raw-url"
    repository: CodeRepositoryRef = {
        "provider": "github",
        "owner": source.owner,
        "repo": source.repo,
        "source_url": f"https://github.com/{source.owner}/{source.repo}",
        "requested_ref": source.requested_ref,
        "resolved_ref": resolved_ref,
        "current_commit": content_identity,
        "default_branch": resolved_ref,
        "local_root": paths["checkout_root"],
        "storage_kind": "managed_download",
        "managed_checkout": True,
        "workspace_root": paths["workspace_root"],
        "cache_key": paths["cache_key"],
        "dirty_state": "unknown",
    }
    return repository


def _metadata_for_repository(
    source: GitHubSource,
    repository: CodeRepositoryRef,
) -> dict[str, object]:
    metadata: dict[str, object] = {
        "schema_version": MANAGED_METADATA_SCHEMA_VERSION,
        "storage_kind": "managed_download",
        "provider": "github",
        "owner": source.owner,
        "repo": source.repo,
        "source_url": f"https://github.com/{source.owner}/{source.repo}",
        "raw_source_url": source.source_url,
        "requested_ref": source.requested_ref,
        "repo_relative_path": source.repo_relative_path,
        "resolved_ref": repository["resolved_ref"],
        "current_commit": repository["current_commit"],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    return metadata


def _expected_identity(source: GitHubSource) -> dict[str, object]:
    identity: dict[str, object] = {
        "schema_version": MANAGED_METADATA_SCHEMA_VERSION,
        "storage_kind": "managed_download",
        "provider": "github",
        "owner": source.owner,
        "repo": source.repo,
        "source_url": f"https://github.com/{source.owner}/{source.repo}",
        "raw_source_url": source.source_url,
        "requested_ref": source.requested_ref,
        "repo_relative_path": source.repo_relative_path,
    }
    return identity


def _content_identity(data: bytes) -> str:
    digest = hashlib.sha256(data).hexdigest()
    identity = f"raw-sha256:{digest}"
    return identity
