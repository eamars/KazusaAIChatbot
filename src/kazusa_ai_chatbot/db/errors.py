"""Application-level database exceptions."""

from __future__ import annotations

from pymongo.errors import PyMongoError


class DatabaseOperationError(RuntimeError):
    """Raised when a public DB helper cannot complete its operation."""


DatabaseBackendError = PyMongoError
