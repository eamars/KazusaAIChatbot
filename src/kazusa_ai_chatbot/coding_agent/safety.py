"""Shared deterministic safety predicates for coding-agent repository paths."""

import hashlib
import shutil
from collections.abc import Iterable
from pathlib import Path
from pathlib import PurePosixPath


_SECRET_PATH_FRAGMENTS = (
    "secret",
    "credential",
    "private_key",
    "id_rsa",
    "token",
    "password",
)
_SECRET_SUFFIXES = (".pem", ".key", ".p12", ".pfx")
_BINARY_SUFFIXES = (
    ".7z",
    ".avif",
    ".bmp",
    ".class",
    ".dll",
    ".exe",
    ".gif",
    ".ico",
    ".jpeg",
    ".jpg",
    ".pdf",
    ".png",
    ".pyc",
    ".so",
    ".webp",
    ".zip",
)
_MANAGED_COPY_EXCLUDED_NAMES = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "node_modules",
    "patch_apply",
    "candidate_exec",
    "coding_runs",
    "repository_indexes",
    "test_artifacts",
    "venv",
    ".venv",
}


def is_secret_like_path(path: str) -> bool:
    """Return whether a repository-relative path is likely secret material."""

    lowered_path = path.casefold()
    has_fragment = any(
        fragment in lowered_path for fragment in _SECRET_PATH_FRAGMENTS
    )
    has_suffix = lowered_path.endswith(_SECRET_SUFFIXES)
    return has_fragment or has_suffix


def is_binary_like_path(path: str) -> bool:
    """Return whether a path suffix normally denotes binary content."""

    binary_like = path.casefold().endswith(_BINARY_SUFFIXES)
    return binary_like


def normalize_safe_repo_relative_path(path_text: str) -> str | None:
    """Normalize one safe text path or reject traversal and protected material."""

    stripped_path = path_text.strip().replace("\\", "/")
    if not stripped_path:
        return None
    path = PurePosixPath(stripped_path)
    if path.is_absolute() or ".." in path.parts:
        return None
    if any(
        part in ("", ".")
        or part.casefold() in _MANAGED_COPY_EXCLUDED_NAMES
        for part in path.parts
    ):
        return None
    lowered_name = path.name.casefold()
    if lowered_name == ".env" or lowered_name.startswith(".env."):
        return None
    normalized_path = path.as_posix()
    if is_secret_like_path(normalized_path):
        return None
    if is_binary_like_path(normalized_path):
        return None
    return normalized_path


def confined_managed_repo_path(root: Path, repo_path: str) -> Path:
    """Return a non-symlink repository path confined to one managed root."""

    safe_path = normalize_safe_repo_relative_path(repo_path)
    if safe_path is None or safe_path != repo_path:
        raise ValueError("managed repository path is unsafe")
    if root.is_symlink():
        raise ValueError("managed repository root is a symlink")
    resolved_root = root.resolve(strict=True)
    candidate_path = resolved_root.joinpath(*PurePosixPath(safe_path).parts)
    current_path = candidate_path
    while current_path != resolved_root:
        if current_path.is_symlink():
            raise ValueError("managed repository path contains a symlink")
        current_path = current_path.parent
    resolved_candidate = candidate_path.resolve(strict=False)
    if not resolved_candidate.is_relative_to(resolved_root):
        raise ValueError("managed repository path escapes its root")
    return candidate_path


def copy_managed_source_tree(
    source_root: Path,
    destination_root: Path,
    *,
    dirs_exist_ok: bool = False,
    extra_excluded_names: Iterable[str] = (),
) -> None:
    """Copy a repository without symlinks, credentials, or generated workspaces."""

    resolved_source = source_root.resolve(strict=True)
    excluded_names = {
        *(_MANAGED_COPY_EXCLUDED_NAMES),
        *(name.casefold() for name in extra_excluded_names),
    }

    def ignore(directory: str, names: list[str]) -> set[str]:
        directory_path = Path(directory)
        ignored: set[str] = set()
        for name in names:
            path = directory_path / name
            try:
                relative_path = path.relative_to(resolved_source).as_posix()
            except ValueError:
                ignored.add(name)
                continue
            lowered_name = name.casefold()
            if (
                path.is_symlink()
                or lowered_name in excluded_names
                or lowered_name == ".env"
                or lowered_name.startswith(".env.")
                or is_secret_like_path(relative_path)
            ):
                ignored.add(name)
        return ignored

    shutil.copytree(
        resolved_source,
        destination_root,
        dirs_exist_ok=dirs_exist_ok,
        ignore=ignore,
    )


def managed_source_tree_digest(
    source_root: Path,
    *,
    extra_excluded_names: Iterable[str] = (),
) -> str:
    """Hash exactly the regular files admitted by the managed-copy policy."""

    resolved_source = source_root.resolve(strict=True)
    excluded_names = {
        *(_MANAGED_COPY_EXCLUDED_NAMES),
        *(name.casefold() for name in extra_excluded_names),
    }
    digest = hashlib.sha256()
    for path in sorted(resolved_source.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        relative_path = path.relative_to(resolved_source)
        lowered_parts = {part.casefold() for part in relative_path.parts}
        lowered_name = relative_path.name.casefold()
        if (
            lowered_parts & excluded_names
            or lowered_name == ".env"
            or lowered_name.startswith(".env.")
            or is_secret_like_path(relative_path.as_posix())
        ):
            continue
        digest.update(relative_path.as_posix().encode("utf-8"))
        digest.update(b"\0")
        content = path.read_bytes()
        if not is_binary_like_path(relative_path.as_posix()):
            content = content.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        digest.update(hashlib.sha256(content).digest())
    tree_digest = digest.hexdigest()
    return tree_digest
