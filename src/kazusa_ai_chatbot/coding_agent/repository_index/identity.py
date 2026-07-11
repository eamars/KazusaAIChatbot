"""Repository-index identity and exclusion policy."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from kazusa_ai_chatbot.coding_agent.safety import (
    is_binary_like_path,
    is_secret_like_path,
)


EXCLUSION_POLICY_VERSION = "repository_index_exclusion.v1"
_EXCLUDED_SEGMENTS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "coding_runs",
    "repository_indexes",
    "test_artifacts",
    "venv",
    "__pycache__",
}
_SECRET_MARKERS = (
    "-----begin private key-----",
    "-----begin rsa private key-----",
    "-----begin openssh private key-----",
)
_CREDENTIAL_ASSIGNMENT = re.compile(
    r"(?i)(api[_-]?key|access[_-]?token|password|secret)\s*[:=]\s*[^\s]+",
)


def source_identity_hash(source_identity: dict[str, object]) -> str:
    """Return a stable filesystem-safe identifier for one resolved source."""

    canonical = json.dumps(
        source_identity,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    identity_hash = hashlib.sha256(canonical).hexdigest()
    return identity_hash


def excluded_path_reason(path: Path) -> str | None:
    """Return the policy reason when a relative path must stay out of an index."""

    parts = [part.casefold() for part in path.parts]
    if any(part == ".env" or part.startswith(".env.") for part in parts):
        return "environment_path"
    if any(part in _EXCLUDED_SEGMENTS for part in parts):
        return "excluded_path"
    normalized_path = path.as_posix()
    if is_secret_like_path(normalized_path):
        return "credential_path"
    if is_binary_like_path(normalized_path):
        return "binary_path"
    return None


def has_secret_marker(content: str) -> bool:
    """Detect credential-like content before a safe-text file is indexed."""

    lowered_content = content.casefold()
    has_marker = (
        any(marker in lowered_content for marker in _SECRET_MARKERS)
        or _CREDENTIAL_ASSIGNMENT.search(content) is not None
    )
    return has_marker
