"""Application-level database exceptions."""

from __future__ import annotations


class DatabaseOperationError(RuntimeError):
    """Raised when a public DB helper cannot complete its operation."""
