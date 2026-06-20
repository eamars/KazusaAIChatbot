"""Deterministic GitHub source parsing."""

from dataclasses import dataclass
import re
from urllib.parse import urlparse

from kazusa_ai_chatbot.coding_agent.code_fetching.models import SourceKind

_GITHUB_HOSTS = {"github.com", "www.github.com"}
_RAW_GITHUB_HOST = "raw.githubusercontent.com"
_OWNER_REPO_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(?:\.git)?$")
_HTTPS_URL_RE = re.compile(r"https?://[^\s\])>]+")
_VALID_SEGMENT_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
_ARCHIVE_SUFFIXES = (".zip", ".tar.gz", ".tgz")
_PACKAGE_REGISTRY_SCHEMES = {"npm", "pypi", "cargo", "crate", "go", "gomod"}
_UNSAFE_REPO_PATH_SEGMENTS = {".git", ".."}
_UNSUPPORTED_GITHUB_KINDS = {
    "issues",
    "pull",
    "pulls",
    "discussions",
    "releases",
    "archive",
}


@dataclass(frozen=True)
class GitHubSource:
    """Parsed public GitHub source.

    Args:
        owner: GitHub owner or organization.
        repo: GitHub repository name without `.git`.
        source_url: Source URL or canonical shorthand URL.
        source_kind: Repository, directory, or file scope.
        requested_ref: Optional branch, tag, or commit.
        repo_relative_path: Optional path inside the repository.

    Returns:
        Immutable source descriptor used before checkout resolution.
    """

    owner: str
    repo: str
    source_url: str
    source_kind: SourceKind
    requested_ref: str | None
    repo_relative_path: str | None


def extract_http_urls(text: str) -> list[str]:
    """Extract HTTP URLs from caller text.

    Args:
        text: Caller text that may contain markdown or raw URLs.

    Returns:
        URLs in appearance order with common trailing punctuation removed.
    """

    urls: list[str] = []
    for match in _HTTPS_URL_RE.finditer(text):
        url = match.group(0).rstrip(".,;:")
        urls.append(url)

    return urls


def parse_repo_hint(repo_hint: str) -> GitHubSource | None:
    """Parse an `owner/repo` GitHub shorthand.

    Args:
        repo_hint: Repository shorthand supplied by a trusted caller field.

    Returns:
        Parsed source, or `None` when the shorthand is unsupported.
    """

    stripped_hint = repo_hint.strip()
    if not _OWNER_REPO_RE.fullmatch(stripped_hint):
        return None

    owner, repo = stripped_hint.split("/", 1)
    normalized_repo = _strip_git_suffix(repo)
    source = GitHubSource(
        owner=owner,
        repo=normalized_repo,
        source_url=f"https://github.com/{owner}/{normalized_repo}",
        source_kind="repository",
        requested_ref=None,
        repo_relative_path=None,
    )
    return source


def parse_github_source(source_url: str) -> GitHubSource | None:
    """Parse supported public GitHub source URLs.

    Args:
        source_url: Public GitHub repository, tree, blob, or raw URL.

    Returns:
        Parsed source, or `None` when the URL is not a supported source.
    """

    stripped_url = source_url.strip()
    parsed_url = urlparse(stripped_url)
    if parsed_url.scheme not in {"http", "https"}:
        return None
    if parsed_url.username or parsed_url.password:
        return None

    host = parsed_url.netloc.lower()
    if host == _RAW_GITHUB_HOST:
        source = _parse_raw_github_url(stripped_url, parsed_url.path)
        return source
    if host not in _GITHUB_HOSTS:
        return None

    source = _parse_github_web_url(stripped_url, parsed_url.path)
    return source


def unsupported_source_reason(source_text: str) -> str | None:
    """Return a stable unsupported-source reason for known unsafe inputs.

    Args:
        source_text: URL or source hint supplied by the caller.

    Returns:
        Human-readable reason, or `None` when the text is merely unrecognized.
    """

    stripped_text = source_text.strip()
    lowered_text = stripped_text.lower()
    if lowered_text.startswith(("git@", "ssh://", "git://")):
        return "SSH and git-protocol sources are not supported."

    parsed_url = urlparse(stripped_text)
    if parsed_url.scheme.lower() in _PACKAGE_REGISTRY_SCHEMES:
        return "Package registry sources are not supported."

    if parsed_url.username or parsed_url.password:
        return "Credential-bearing source URLs are rejected."

    host = parsed_url.netloc.lower()
    if host == "gist.github.com":
        return "Gist sources are not supported."
    if parsed_url.scheme in {"http", "https"} and host not in (
        *_GITHUB_HOSTS,
        _RAW_GITHUB_HOST,
    ):
        return "Only public GitHub repository sources are supported."

    unsafe_path_reason = _unsafe_repo_path_reason(host, parsed_url.path)
    if unsafe_path_reason:
        return unsafe_path_reason

    path_lower = parsed_url.path.lower()
    if path_lower.endswith(_ARCHIVE_SUFFIXES):
        return "Archive and release asset sources are not supported."

    parts = [part for part in parsed_url.path.split("/") if part]
    if host in _GITHUB_HOSTS and len(parts) >= 3:
        source_kind = parts[2].lower()
        if source_kind in _UNSUPPORTED_GITHUB_KINDS:
            return f"GitHub {source_kind} sources are not supported."

    return None


def redact_source_text(source_text: str) -> str:
    """Redact credentials from source text before public reporting.

    Args:
        source_text: Caller-provided URL or hint.

    Returns:
        Redacted source string safe for trace summaries.
    """

    parsed_url = urlparse(source_text)
    if not parsed_url.username and not parsed_url.password:
        return source_text

    host = parsed_url.hostname or ""
    redacted = parsed_url._replace(netloc=f"<redacted>@{host}").geturl()
    return redacted


def with_requested_ref(
    source: GitHubSource,
    requested_ref: str | None,
) -> GitHubSource:
    """Apply an explicit ref to a source when the source has no ref.

    Args:
        source: Parsed GitHub source.
        requested_ref: Caller-supplied ref.

    Returns:
        Source with the requested ref applied when possible.
    """

    if not requested_ref or source.requested_ref == requested_ref:
        return source
    if source.requested_ref is not None:
        return source

    updated_source = GitHubSource(
        owner=source.owner,
        repo=source.repo,
        source_url=source.source_url,
        source_kind=source.source_kind,
        requested_ref=requested_ref,
        repo_relative_path=source.repo_relative_path,
    )
    return updated_source


def is_raw_github_source(source: GitHubSource) -> bool:
    """Return whether a parsed source came from raw.githubusercontent.com.

    Args:
        source: Parsed GitHub source descriptor.

    Returns:
        `True` when the source URL is a raw GitHub file URL.
    """

    parsed_url = urlparse(source.source_url)
    is_raw_source = parsed_url.netloc.lower() == _RAW_GITHUB_HOST
    return is_raw_source


def is_safe_repo_relative_path(repo_relative_path: str | None) -> bool:
    """Return whether a repository-relative path is safe to expose.

    Args:
        repo_relative_path: Path inside a repository, or `None` for repo scope.

    Returns:
        `True` when the path does not target Git internals or traversal.
    """

    if repo_relative_path is None or repo_relative_path == ".":
        return True

    parts = [part for part in repo_relative_path.replace("\\", "/").split("/") if part]
    for part in parts:
        if part.lower() in _UNSAFE_REPO_PATH_SEGMENTS:
            return False

    return True


def _parse_raw_github_url(
    source_url: str,
    path: str,
) -> GitHubSource | None:
    parts = [part for part in path.split("/") if part]
    if len(parts) < 4:
        return None
    owner, repo = parts[0], parts[1]
    if not _valid_owner_repo(owner, repo):
        return None

    requested_ref, repo_relative_path = _raw_ref_and_path(parts[2:])
    if not repo_relative_path:
        return None
    if not is_safe_repo_relative_path(repo_relative_path):
        return None

    source = GitHubSource(
        owner=owner,
        repo=_strip_git_suffix(repo),
        source_url=source_url,
        source_kind="file",
        requested_ref=requested_ref,
        repo_relative_path=repo_relative_path,
    )
    return source


def _raw_ref_and_path(parts: list[str]) -> tuple[str, str | None]:
    """Split raw GitHub URL path segments into ref and repository path.

    Args:
        parts: URL path segments after owner and repository.

    Returns:
        Requested ref plus the repository-relative file path when available.
    """

    if len(parts) >= 4 and parts[0] == "refs" and parts[1] in {"heads", "tags"}:
        requested_ref = "/".join(parts[:3])
        repo_relative_path = "/".join(parts[3:])
        split_result = (requested_ref, repo_relative_path)
        return split_result

    requested_ref = parts[0]
    repo_relative_path = "/".join(parts[1:])
    split_result = (requested_ref, repo_relative_path or None)
    return split_result


def _parse_github_web_url(
    source_url: str,
    path: str,
) -> GitHubSource | None:
    parts = [part for part in path.split("/") if part]
    if len(parts) < 2:
        return None

    owner, repo = parts[0], _strip_git_suffix(parts[1])
    if not _valid_owner_repo(owner, repo):
        return None

    if len(parts) == 2:
        source = GitHubSource(
            owner=owner,
            repo=repo,
            source_url=source_url,
            source_kind="repository",
            requested_ref=None,
            repo_relative_path=None,
        )
        return source

    url_kind = parts[2].lower()
    if url_kind == "tree":
        source = _parse_tree_or_blob_url(
            source_url=source_url,
            owner=owner,
            repo=repo,
            parts=parts,
            source_kind="directory",
        )
        return source
    if url_kind == "blob":
        source = _parse_tree_or_blob_url(
            source_url=source_url,
            owner=owner,
            repo=repo,
            parts=parts,
            source_kind="file",
        )
        return source

    return None


def _parse_tree_or_blob_url(
    *,
    source_url: str,
    owner: str,
    repo: str,
    parts: list[str],
    source_kind: SourceKind,
) -> GitHubSource | None:
    if len(parts) < 4:
        return None

    requested_ref = parts[3]
    repo_relative_path = "/".join(parts[4:]) if len(parts) > 4 else None
    if source_kind in {"directory", "file"} and not repo_relative_path:
        return None
    if not is_safe_repo_relative_path(repo_relative_path):
        return None

    source = GitHubSource(
        owner=owner,
        repo=repo,
        source_url=source_url,
        source_kind=source_kind,
        requested_ref=requested_ref,
        repo_relative_path=repo_relative_path,
    )
    return source


def _valid_owner_repo(owner: str, repo: str) -> bool:
    owner_valid = _VALID_SEGMENT_RE.fullmatch(owner) is not None
    repo_valid = _VALID_SEGMENT_RE.fullmatch(repo) is not None
    is_valid = owner_valid and repo_valid
    return is_valid


def _strip_git_suffix(repo: str) -> str:
    if repo.endswith(".git"):
        stripped_repo = repo[:-4]
        return stripped_repo

    return repo


def _unsafe_repo_path_reason(host: str, path: str) -> str | None:
    parts = [part for part in path.split("/") if part]
    repo_path_parts: list[str] = []

    if host == _RAW_GITHUB_HOST and len(parts) >= 4:
        repo_path_parts = parts[3:]
    elif host in _GITHUB_HOSTS and len(parts) >= 5:
        url_kind = parts[2].lower()
        if url_kind in {"tree", "blob"}:
            repo_path_parts = parts[4:]
    elif host in _GITHUB_HOSTS and len(parts) > 2:
        repo_path_parts = parts[2:]

    if _contains_unsafe_repo_path_segment(repo_path_parts):
        return "Repository-internal .git and traversal paths are rejected."

    return None


def _contains_unsafe_repo_path_segment(parts: list[str]) -> bool:
    for part in parts:
        if part.lower() in _UNSAFE_REPO_PATH_SEGMENTS:
            return True

    return False
