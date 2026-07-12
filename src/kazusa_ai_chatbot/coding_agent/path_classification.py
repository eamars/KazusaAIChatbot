"""Shared repository-path role classification for coding-agent evidence."""

from __future__ import annotations

from pathlib import PurePosixPath


def is_test_path(path: str) -> bool:
    """Return whether a repository-relative path names a test artifact.

    Args:
        path: Repository-relative path using either Windows or POSIX separators.

    Returns:
        ``True`` for conventional test filenames or paths beneath a test
        directory, otherwise ``False``.
    """

    normalized = path.replace("\\", "/").lstrip("./")
    path_value = PurePosixPath(normalized)
    name = path_value.name.casefold()
    directory_parts = {part.casefold() for part in path_value.parts[:-1]}
    return (
        name.startswith("test_")
        or name.endswith("_test.py")
        or bool(directory_parts & {"test", "tests"})
    )
