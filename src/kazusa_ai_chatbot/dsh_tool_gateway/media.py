"""Scoped attached-media inspection for the semantic gateway."""

from __future__ import annotations

import json
import os
from collections.abc import Awaitable, Callable, Mapping
from copy import copy
from pathlib import Path
from time import time
from typing import Any

from kazusa_ai_chatbot.config import MEDIA_SESSION_CACHE_TTL_SECONDS
from kazusa_ai_chatbot.dsh_tool_gateway.contracts import (
    KazusaSemanticCapabilityResultV1,
    OpaqueReferenceCodec,
    content_digest,
    new_evidence_receipt,
)
from kazusa_ai_chatbot.media_inspection.service import inspect_media
from kazusa_ai_chatbot.media_inspection.session_cache import get_session_media


class MediaSemanticService:
    """Inspect only media issued for the exact Brain conversation scope."""

    def __init__(
        self,
        *,
        scope: tuple[str, str, str],
        codec: OpaqueReferenceCodec,
        get_media: Callable[
            [tuple[str, str, str], str],
            dict[str, object] | None,
        ]
        | None = None,
        inspect: Callable[[object], Awaitable[dict[str, object]]] = inspect_media,
    ) -> None:
        self._scope = scope
        self._codec = codec
        self._get_media = get_media or get_attached_media
        self._inspect = inspect

    def with_authority(self, authority: Mapping[str, Any] | object) -> "MediaSemanticService":
        """Return a call-local service bound to the signed authority."""

        bound = copy(self)
        bound._codec = self._codec.with_authority(authority)
        return bound

    async def inspect_attached_media(
        self,
        *,
        attached_media_ref: str,
        question: str,
    ) -> KazusaSemanticCapabilityResultV1:
        """Answer one bounded question from one exact scoped image ref."""

        if not isinstance(question, str) or not question.strip():
            return KazusaSemanticCapabilityResultV1.failure(
                "invalid", "QUESTION_REQUIRED", "A visual question is required."
            )
        try:
            payload = self._codec.resolve(attached_media_ref, "attached-media")
        except ValueError:
            return KazusaSemanticCapabilityResultV1.failure(
                "invalid", "MEDIA_REFERENCE_INVALID", "The attached media reference is invalid."
            )
        cache_ref = payload.get("cache_ref")
        encoded_scope = payload.get("scope")
        expected_scope = list(self._scope)
        if cache_ref is None or encoded_scope != expected_scope:
            return KazusaSemanticCapabilityResultV1.failure(
                "denied", "MEDIA_SCOPE_MISMATCH", "The attached media is outside this conversation scope."
            )
        if not isinstance(cache_ref, str):
            return KazusaSemanticCapabilityResultV1.failure(
                "invalid", "MEDIA_REFERENCE_INVALID", "The attached media reference is invalid."
            )
        media = self._get_media(self._scope, cache_ref)
        if media is None:
            return KazusaSemanticCapabilityResultV1.failure(
                "empty", "MEDIA_NOT_FOUND", "The attached media is no longer available."
            )
        content_type = media.get("content_type")
        base64_data = media.get("base64_data")
        if not isinstance(content_type, str) or not isinstance(base64_data, str):
            return KazusaSemanticCapabilityResultV1.failure(
                "invalid", "MEDIA_PAYLOAD_INVALID", "The attached media payload is invalid."
            )
        inspection = await self._inspect({
            "schema_version": "media_inspection_request.v1",
            "source": "test",
            "media_kind": "image",
            "content_type": content_type,
            "base64_data": base64_data,
            "question": question.strip(),
            "existing_descriptor": str(media.get("existing_descriptor") or ""),
        })
        answer = str(inspection.get("answer") or "")
        status = str(inspection.get("status") or "failed")
        entity = {
            "attached_media_ref": attached_media_ref,
            "status": status,
            "answer": answer,
            "evidence_boundary_notes": inspection.get("evidence_boundary_notes", []),
        }
        return KazusaSemanticCapabilityResultV1.success(
            entities=[entity],
            evidence=[new_evidence_receipt(
                receipt_id=f"receipt-attached-media-{content_digest(attached_media_ref)}",
                source_kind="attached_media",
                semantic_ref=attached_media_ref,
                value=entity,
            )],
        )


def issue_attached_media_reference(
    *,
    codec: OpaqueReferenceCodec,
    scope: tuple[str, str, str],
    cache_ref: str,
) -> str:
    """Issue a Brain-scoped opaque media reference."""

    if len(scope) != 3 or any(not isinstance(value, str) or not value for value in scope):
        raise ValueError("media scope must contain platform, channel, and user")
    if not isinstance(cache_ref, str) or not cache_ref:
        raise ValueError("cache_ref is required")
    return codec.issue(
        "attached-media",
        {"scope": list(scope), "cache_ref": cache_ref},
    )


def persist_attached_media(
    scope: tuple[str, str, str],
    references: list[dict[str, object]],
) -> None:
    """Mirror current session media into the shared DSH data boundary."""

    _validate_scope(scope)
    root = _shared_media_root()
    if root is None:
        return
    root.mkdir(parents=True, exist_ok=True)
    for reference in references:
        cache_ref = reference.get("cache_ref")
        if not isinstance(cache_ref, str) or not cache_ref:
            raise ValueError("attached media cache_ref is required")
        payload = get_session_media(scope, cache_ref)
        if payload is None:
            continue
        document = {
            "scope": list(scope),
            "cache_ref": cache_ref,
            "expires_at_epoch": time() + MEDIA_SESSION_CACHE_TTL_SECONDS,
            "payload": payload,
        }
        target = _shared_media_path(root, scope, cache_ref)
        temporary = target.with_suffix(f".{os.getpid()}.tmp")
        temporary.write_text(
            json.dumps(
                document,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )
        os.replace(temporary, target)


def get_attached_media(
    scope: tuple[str, str, str],
    cache_ref: str,
) -> dict[str, object] | None:
    """Load attached media from this process or the shared DSH data root."""

    _validate_scope(scope)
    if not isinstance(cache_ref, str) or not cache_ref:
        return None
    local = get_session_media(scope, cache_ref)
    if local is not None:
        return local
    root = _shared_media_root()
    if root is None:
        return None
    path = _shared_media_path(root, scope, cache_ref)
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(document, Mapping):
        return None
    if document.get("scope") != list(scope):
        return None
    if document.get("cache_ref") != cache_ref:
        return None
    expires_at = document.get("expires_at_epoch")
    if not isinstance(expires_at, (int, float)) or isinstance(expires_at, bool):
        return None
    if expires_at <= time():
        path.unlink(missing_ok=True)
        return None
    payload = document.get("payload")
    if not isinstance(payload, Mapping):
        return None
    return dict(payload)


def _shared_media_root() -> Path | None:
    """Return the configured cross-process attached-media directory."""

    data_root = os.environ.get("KAZUSA_DSH_DATA_ROOT", "").strip()
    if not data_root:
        return None
    root = Path(data_root)
    if not root.is_absolute():
        raise ValueError("KAZUSA_DSH_DATA_ROOT must be absolute")
    return root / "attached-media-v1"


def _shared_media_path(
    root: Path,
    scope: tuple[str, str, str],
    cache_ref: str,
) -> Path:
    """Derive one traversal-safe path from scoped opaque cache identity."""

    digest = content_digest({"scope": list(scope), "cache_ref": cache_ref})
    return root / f"{digest.removeprefix('sha256:')}.json"


def _validate_scope(scope: tuple[str, str, str]) -> None:
    """Require the exact platform, channel, and user media scope."""

    if (
        not isinstance(scope, tuple)
        or len(scope) != 3
        or any(not isinstance(value, str) or not value for value in scope)
    ):
        raise ValueError("media scope must contain platform, channel, and user")
