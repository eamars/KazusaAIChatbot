"""Filesystem safety helpers for coding-agent modules."""

from pathlib import Path


class PathSafetyError(ValueError):
    """Raised when a path violates the coding-agent containment rules."""


def resolve_existing_path(path_value: str) -> Path:
    """Resolve a caller-provided path that must already exist.

    Args:
        path_value: Caller-provided filesystem path.

    Returns:
        Absolute resolved path.
    """

    try:
        resolved_path = Path(path_value).expanduser().resolve(strict=True)
    except OSError as exc:
        message = f"path does not exist or cannot be resolved: {exc}"
        raise PathSafetyError(message) from exc

    return resolved_path


def ensure_path_inside(path_value: str | Path, root_value: str | Path) -> Path:
    """Validate that a path is contained by the supplied root.

    Args:
        path_value: Path that must remain inside the root.
        root_value: Containing root path.

    Returns:
        Absolute resolved path when it is contained by the root.
    """

    path = Path(path_value).expanduser().resolve(strict=False)
    root = Path(root_value).expanduser().resolve(strict=False)
    if not path.is_relative_to(root):
        message = f"path is outside the allowed root: {path}"
        raise PathSafetyError(message)

    return path
