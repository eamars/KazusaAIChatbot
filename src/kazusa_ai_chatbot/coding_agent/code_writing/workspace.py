"""Persistent public-safe session storage for code-writing proposals."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from kazusa_ai_chatbot.coding_agent.code_writing.models import (
    WritingMode,
    WritingSessionSummary,
)
from kazusa_ai_chatbot.coding_agent.tools.paths import (
    PathSafetyError,
    ensure_path_inside,
)

SESSION_ROOT_NAME = "writing_sessions"
SESSION_METADATA_NAME = "session.json"
SESSION_ID_PREFIX = "session-"
SESSION_ID_HASH_CHARS = 16
_SESSION_SAFE_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def prepare_writing_workspace(
    *,
    workspace_root: str | Path,
    session_id: str | None,
    base_identity: str,
    mode: WritingMode,
) -> WritingSessionSummary:
    """Prepare persistent session metadata and invalidate stale base state.

    Args:
        workspace_root: Caller-configured storage root for writing sessions.
        session_id: Optional stable public session id supplied by the caller.
        base_identity: Public-safe identity for the repository or new-project
            base being proposed against.
        mode: Writing mode for the current proposal.

    Returns:
        Public-safe session handle with no filesystem paths.
    """

    root = _prepare_root(workspace_root)
    public_session_id = _safe_session_id(session_id, base_identity)
    session_dir = ensure_path_inside(root / SESSION_ROOT_NAME / public_session_id, root)
    session_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = ensure_path_inside(session_dir / SESSION_METADATA_NAME, root)

    previous_metadata = _read_metadata(metadata_path)
    invalidated_previous = False
    if previous_metadata is not None:
        previous_base = previous_metadata["base_identity"]
        invalidated_previous = previous_base != base_identity

    metadata = {
        "session_id": public_session_id,
        "public_handle": _public_handle(public_session_id),
        "base_identity": base_identity,
        "mode": mode,
        "invalidated_previous": invalidated_previous,
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary: WritingSessionSummary = {
        "session_id": public_session_id,
        "public_handle": metadata["public_handle"],
        "invalidated_previous": invalidated_previous,
    }
    return summary


def _prepare_root(workspace_root: str | Path) -> Path:
    root = Path(workspace_root).expanduser().resolve(strict=False)
    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        message = f"writing storage cannot be prepared: {exc}"
        raise PathSafetyError(message) from exc
    if not root.is_dir():
        raise PathSafetyError("writing storage root is not a directory.")
    return root


def _safe_session_id(session_id: str | None, base_identity: str) -> str:
    if session_id is None or not session_id.strip():
        digest = hashlib.sha256(base_identity.encode("utf-8")).hexdigest()
        generated = f"{SESSION_ID_PREFIX}{digest[:SESSION_ID_HASH_CHARS]}"
        return generated

    compact = _SESSION_SAFE_RE.sub("-", session_id.strip())
    compact = compact.strip(".-")
    if not compact:
        digest = hashlib.sha256(base_identity.encode("utf-8")).hexdigest()
        compact = f"{SESSION_ID_PREFIX}{digest[:SESSION_ID_HASH_CHARS]}"
    return compact[:80]


def _public_handle(session_id: str) -> str:
    handle = f"writing-{session_id}"
    return handle


def _read_metadata(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    try:
        raw_text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    if not isinstance(parsed.get("base_identity"), str):
        return None
    return parsed
