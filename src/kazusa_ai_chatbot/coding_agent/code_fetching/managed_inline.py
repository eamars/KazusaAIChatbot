"""Managed read-only storage for inline source fragments."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

from kazusa_ai_chatbot.coding_agent.models import (
    MANAGED_METADATA_SCHEMA_VERSION,
)
from kazusa_ai_chatbot.coding_agent.code_fetching.models import (
    CodeRepositoryRef,
    CodeSourceScope,
)
from kazusa_ai_chatbot.coding_agent.tools.paths import ensure_path_inside

INLINE_SOURCE_DIR_NAME = "inline_sources"
INLINE_PROVIDER = "inline"
INLINE_OWNER = "inline"
INLINE_RESOLVED_REF = "inline"
INLINE_BUNDLE_HASH_CHARS = 32
MAX_INLINE_FILENAME_CHARS = 80
MAX_INLINE_LABEL_CHARS = 120

_SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")
_RESERVED_FILENAMES = {
    ".env",
    ".git",
    "con",
    "prn",
    "aux",
    "nul",
    "com1",
    "com2",
    "com3",
    "com4",
    "com5",
    "com6",
    "com7",
    "com8",
    "com9",
    "lpt1",
    "lpt2",
    "lpt3",
    "lpt4",
    "lpt5",
    "lpt6",
    "lpt7",
    "lpt8",
    "lpt9",
}
_LANGUAGE_EXTENSIONS = {
    "bash": ".sh",
    "c": ".c",
    "cpp": ".cpp",
    "diff": ".diff",
    "go": ".go",
    "java": ".java",
    "javascript": ".js",
    "json": ".json",
    "markdown": ".md",
    "python": ".py",
    "ruby": ".rb",
    "rust": ".rs",
    "shell": ".sh",
    "sh": ".sh",
    "text": ".txt",
    "typescript": ".ts",
    "yaml": ".yaml",
}


@dataclass(frozen=True)
class InlineSourceFragment:
    """Exact inline source text selected from a user-visible request."""

    content: str
    filename_hint: str
    language_hint: str
    source_label: str


@dataclass(frozen=True)
class InlineSourceBundle:
    """One managed inline source bundle for a single code-reading task."""

    fragments: tuple[InlineSourceFragment, ...]


class ManagedInlineSourceError(RuntimeError):
    """Raised when inline source material cannot be prepared."""


def materialize_inline_source_bundle(
    bundle: InlineSourceBundle,
    workspace_root: str,
) -> tuple[CodeRepositoryRef, CodeSourceScope]:
    """Write inline fragments under the configured coding workspace.

    Args:
        bundle: Exact inline fragments selected by source resolution.
        workspace_root: Configured coding-agent workspace root.

    Returns:
        Repository metadata and source scope for code reading.
    """

    if not bundle.fragments:
        message = "managed inline source requires at least one fragment."
        raise ManagedInlineSourceError(message)

    bundle_identity = _bundle_identity(bundle)
    bundle_id = f"bundle-{bundle_identity[:INLINE_BUNDLE_HASH_CHARS]}"
    root = Path(workspace_root).expanduser().resolve(strict=False)
    bundle_root = root / INLINE_SOURCE_DIR_NAME / bundle_id
    ensure_path_inside(bundle_root, root)
    filenames = _fragment_filenames(bundle)

    try:
        bundle_root.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        message = f"managed inline source directory creation failed: {exc}"
        raise ManagedInlineSourceError(message) from exc

    for fragment, filename in zip(bundle.fragments, filenames, strict=True):
        _write_fragment(bundle_root, filename, fragment.content)

    manifest = _manifest_for_bundle(
        bundle=bundle,
        bundle_id=bundle_id,
        filenames=filenames,
        bundle_identity=bundle_identity,
    )
    _write_manifest(bundle_root, manifest)

    repository = _repository_for_bundle(
        bundle_id=bundle_id,
        bundle_root=bundle_root,
        workspace_root=root,
        bundle_identity=bundle_identity,
    )
    source_scope = _source_scope_for_bundle(bundle_id, filenames)
    result = (repository, source_scope)
    return result


def inline_source_bundle_to_dict(
    bundle: InlineSourceBundle,
) -> dict[str, object]:
    """Serialize an inline bundle without exposing full source text."""

    fragments: list[dict[str, object]] = []
    for fragment in bundle.fragments:
        fragments.append({
            "content_sha256": _sha256_text(fragment.content),
            "content_chars": len(fragment.content),
            "filename_hint": fragment.filename_hint,
            "language_hint": fragment.language_hint,
            "source_label": fragment.source_label,
        })
    result: dict[str, object] = {
        "fragment_count": len(bundle.fragments),
        "fragments": fragments,
    }
    return result


def _bundle_identity(bundle: InlineSourceBundle) -> str:
    payload = [
        {
            "content": fragment.content,
            "filename_hint": fragment.filename_hint,
            "language_hint": fragment.language_hint,
            "source_label": fragment.source_label,
        }
        for fragment in bundle.fragments
    ]
    payload_text = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    identity = _sha256_text(payload_text)
    return identity


def _fragment_filenames(bundle: InlineSourceBundle) -> list[str]:
    filenames: list[str] = []
    for index, fragment in enumerate(bundle.fragments, start=1):
        filename = _filename_for_fragment(fragment, index, filenames)
        filenames.append(filename)
    return filenames


def _filename_for_fragment(
    fragment: InlineSourceFragment,
    index: int,
    existing_filenames: list[str],
) -> str:
    hint = _safe_filename_hint(fragment.filename_hint)
    if hint:
        candidate = hint
    else:
        extension = _extension_for_language(fragment.language_hint)
        candidate = f"fragment_{index:03d}{extension}"

    while candidate in existing_filenames:
        stem = Path(candidate).stem
        suffix = Path(candidate).suffix
        candidate = f"{stem}_{len(existing_filenames) + 1:03d}{suffix}"

    return candidate


def _safe_filename_hint(filename_hint: str) -> str:
    hint = Path(filename_hint.strip()).name
    if not hint:
        return ""
    safe = _SAFE_FILENAME_RE.sub("_", hint).strip("._-")
    if not safe:
        return ""
    safe = safe[:MAX_INLINE_FILENAME_CHARS]
    lowered = safe.lower()
    if lowered in _RESERVED_FILENAMES:
        return ""
    if lowered.endswith((".env", ".git")):
        return ""
    return safe


def _extension_for_language(language_hint: str) -> str:
    language_key = language_hint.strip().lower()
    extension = _LANGUAGE_EXTENSIONS.get(language_key)
    if extension is None:
        extension = ".txt"
    return extension


def _write_fragment(
    bundle_root: Path,
    filename: str,
    content: str,
) -> None:
    target_path = ensure_path_inside(bundle_root / filename, bundle_root)
    try:
        target_path.write_text(content, encoding="utf-8", newline="")
    except OSError as exc:
        message = f"managed inline source write failed: {exc}"
        raise ManagedInlineSourceError(message) from exc


def _write_manifest(
    bundle_root: Path,
    manifest: dict[str, object],
) -> None:
    manifest_path = ensure_path_inside(bundle_root / "manifest.json", bundle_root)
    manifest_text = json.dumps(
        manifest,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    try:
        manifest_path.write_text(manifest_text + "\n", encoding="utf-8")
    except OSError as exc:
        message = f"managed inline source manifest write failed: {exc}"
        raise ManagedInlineSourceError(message) from exc


def _manifest_for_bundle(
    *,
    bundle: InlineSourceBundle,
    bundle_id: str,
    filenames: list[str],
    bundle_identity: str,
) -> dict[str, object]:
    fragments: list[dict[str, object]] = []
    for fragment, filename in zip(bundle.fragments, filenames, strict=True):
        fragments.append({
            "filename": filename,
            "content_sha256": _sha256_text(fragment.content),
            "content_chars": len(fragment.content),
            "filename_hint": fragment.filename_hint,
            "language_hint": fragment.language_hint,
            "source_label": fragment.source_label[:MAX_INLINE_LABEL_CHARS],
        })

    manifest: dict[str, object] = {
        "schema_version": MANAGED_METADATA_SCHEMA_VERSION,
        "storage_kind": "managed_inline_bundle",
        "provider": INLINE_PROVIDER,
        "bundle_id": bundle_id,
        "current_commit": f"inline-sha256:{bundle_identity}",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "fragments": fragments,
    }
    return manifest


def _repository_for_bundle(
    *,
    bundle_id: str,
    bundle_root: Path,
    workspace_root: Path,
    bundle_identity: str,
) -> CodeRepositoryRef:
    source_url = f"inline://accepted-task/{bundle_id}"
    repository: CodeRepositoryRef = {
        "provider": INLINE_PROVIDER,
        "owner": INLINE_OWNER,
        "repo": bundle_id,
        "source_url": source_url,
        "requested_ref": None,
        "resolved_ref": INLINE_RESOLVED_REF,
        "current_commit": f"inline-sha256:{bundle_identity}",
        "default_branch": "",
        "local_root": str(bundle_root),
        "storage_kind": "managed_inline_bundle",
        "managed_checkout": True,
        "workspace_root": str(workspace_root),
        "cache_key": None,
        "dirty_state": "clean",
    }
    return repository


def _source_scope_for_bundle(
    bundle_id: str,
    filenames: list[str],
) -> CodeSourceScope:
    source_url = f"inline://accepted-task/{bundle_id}"
    if len(filenames) == 1:
        filename = filenames[0]
        scope: CodeSourceScope = {
            "kind": "file",
            "repo_relative_path": filename,
            "source_url": f"{source_url}/{filename}",
            "requested_ref": None,
            "interpretation": "managed inline source fragment",
        }
        return scope

    scope = {
        "kind": "directory",
        "repo_relative_path": None,
        "source_url": source_url,
        "requested_ref": None,
        "interpretation": "managed inline source bundle",
    }
    return scope


def _sha256_text(value: str) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return digest
