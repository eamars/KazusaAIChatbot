"""Scoped attached and public-media inspection for the semantic gateway."""

from __future__ import annotations

import base64
import hashlib
import ipaddress
import json
import os
import socket
from collections.abc import Awaitable, Callable, Mapping
from copy import copy
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from time import time
from typing import Any
from urllib.parse import urljoin, urlsplit, urlunsplit

import httpx
from PIL import Image, UnidentifiedImageError

from kazusa_ai_chatbot.config import MEDIA_SESSION_CACHE_TTL_SECONDS
from kazusa_ai_chatbot.dsh_tool_gateway.contracts import (
    KazusaSemanticCapabilityResultV1,
    OpaqueReferenceCodec,
    content_digest,
    new_evidence_receipt,
)
from kazusa_ai_chatbot.media_inspection.service import inspect_media
from kazusa_ai_chatbot.media_inspection.session_cache import get_session_media

_MAX_PUBLIC_MEDIA_REDIRECTS = 3
_MAX_PUBLIC_MEDIA_BYTES = 6 * 1024 * 1024
_MAX_PUBLIC_MEDIA_DIMENSION = 8192
_PUBLIC_MEDIA_FETCH_TIMEOUT_SECONDS = 15.0


class _PublicMediaBoundaryError(ValueError):
    """Carry the closed semantic result for a deterministic media boundary."""

    def __init__(self, status: str, code: str, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


@dataclass(frozen=True)
class _PublicMediaTarget:
    """Canonical URL plus the already-vetted address for one fetch hop."""

    canonical_url: str
    hostname: str
    host_header: str
    resolved_ip: str
    scheme: str
    port: int | None
    path: str
    query: str

    @property
    def transport_url(self) -> str:
        """Build an IP-pinned request URL while retaining the URL authority."""

        address = self.resolved_ip
        if ":" in address and not address.startswith("["):
            address = f"[{address}]"
        netloc = address
        if self.port is not None:
            netloc += f":{self.port}"
        return urlunsplit((self.scheme, netloc, self.path, self.query, ""))


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

    def with_authority(self, authority: Mapping[str, Any] | object) -> MediaSemanticService:
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

    async def inspect_public_media(
        self,
        *,
        public_media_url: str,
        question: str,
    ) -> KazusaSemanticCapabilityResultV1:
        """Fetch one public image under bounded SSRF and payload checks."""

        if not isinstance(question, str) or not question.strip():
            return KazusaSemanticCapabilityResultV1.failure(
                "invalid", "QUESTION_REQUIRED", "A visual question is required."
            )
        if not isinstance(public_media_url, str):
            return KazusaSemanticCapabilityResultV1.failure(
                "invalid",
                "PUBLIC_MEDIA_URL_INVALID",
                "The public media URL is invalid.",
            )
        try:
            content_type, image_bytes, final_url = await _fetch_public_image(
                public_media_url,
            )
        except _PublicMediaBoundaryError as exc:
            return KazusaSemanticCapabilityResultV1.failure(
                exc.status,
                exc.code,
                exc.message,
            )
        image_sha256 = hashlib.sha256(image_bytes).hexdigest()
        media_identity = {
            "final_url": final_url,
            "image_sha256": image_sha256,
        }
        media_digest = content_digest(media_identity)
        semantic_ref = f"public-media:{media_digest.removeprefix('sha256:')}"
        inspection = await self._inspect({
            "schema_version": "media_inspection_request.v1",
            "source": "dsh_public_media",
            "media_kind": "image",
            "content_type": content_type,
            "base64_data": base64.b64encode(image_bytes).decode("ascii"),
            "question": question.strip(),
            "existing_descriptor": "",
        })
        entity = {
            "status": str(inspection.get("status") or "failed"),
            "answer": str(inspection.get("answer") or ""),
            "source_url": final_url,
            "content_type": content_type,
            "byte_count": len(image_bytes),
            "evidence_boundary_notes": inspection.get(
                "evidence_boundary_notes",
                [],
            ),
        }
        return KazusaSemanticCapabilityResultV1.success(
            entities=[entity],
            evidence=[new_evidence_receipt(
                receipt_id=(
                    f"receipt-public-media-"
                    f"{media_digest.removeprefix('sha256:')}"
                ),
                source_kind="public_media",
                semantic_ref=semantic_ref,
                value=media_identity,
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


async def _fetch_public_image(url: str) -> tuple[str, bytes, str]:
    """Fetch one bounded image while checking every redirect target."""

    current_target = _validated_public_target(url)
    redirect_count = 0
    while True:
        try:
            async with httpx.AsyncClient(
                follow_redirects=False,
                timeout=_PUBLIC_MEDIA_FETCH_TIMEOUT_SECONDS,
                trust_env=False,
            ) as client:
                response = await client.get(
                    current_target.transport_url,
                    headers={"host": current_target.host_header},
                    extensions={"sni_hostname": current_target.hostname},
                )
        except httpx.TimeoutException as exc:
            raise _PublicMediaBoundaryError(
                "timeout",
                "PUBLIC_MEDIA_FETCH_TIMEOUT",
                "The public media fetch timed out.",
            ) from exc
        except httpx.HTTPError as exc:
            raise _PublicMediaBoundaryError(
                "unavailable",
                "PUBLIC_MEDIA_FETCH_UNAVAILABLE",
                "The public media fetch was unavailable.",
            ) from exc

        if response.is_redirect:
            if redirect_count >= _MAX_PUBLIC_MEDIA_REDIRECTS:
                raise _PublicMediaBoundaryError(
                    "invalid",
                    "PUBLIC_MEDIA_REDIRECT_INVALID",
                    "The public media redirect limit was exceeded.",
                )
            location = response.headers.get("location")
            if not isinstance(location, str) or not location.strip():
                raise _PublicMediaBoundaryError(
                    "invalid",
                    "PUBLIC_MEDIA_REDIRECT_INVALID",
                    "The public media redirect destination is invalid.",
                )
            current_target = _validated_public_target(
                urljoin(current_target.canonical_url, location),
            )
            redirect_count += 1
            continue

        try:
            response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise _PublicMediaBoundaryError(
                "timeout",
                "PUBLIC_MEDIA_FETCH_TIMEOUT",
                "The public media fetch timed out.",
            ) from exc
        except httpx.HTTPError as exc:
            raise _PublicMediaBoundaryError(
                "unavailable",
                "PUBLIC_MEDIA_FETCH_UNAVAILABLE",
                "The public media fetch was unavailable.",
            ) from exc

        content_length = response.headers.get("content-length")
        if content_length is not None:
            try:
                declared_length = int(content_length)
            except (TypeError, ValueError) as exc:
                raise _PublicMediaBoundaryError(
                    "invalid",
                    "PUBLIC_MEDIA_TOO_LARGE",
                    "The public media size declaration is invalid.",
                ) from exc
            if declared_length < 0 or declared_length > _MAX_PUBLIC_MEDIA_BYTES:
                raise _PublicMediaBoundaryError(
                    "invalid",
                    "PUBLIC_MEDIA_TOO_LARGE",
                    "The public media exceeds the byte limit.",
                )

        body = bytearray()
        try:
            async for chunk in response.aiter_bytes():
                if not isinstance(chunk, bytes):
                    raise _PublicMediaBoundaryError(
                        "unavailable",
                        "PUBLIC_MEDIA_FETCH_UNAVAILABLE",
                        "The public media response was invalid.",
                    )
                body.extend(chunk)
                if len(body) > _MAX_PUBLIC_MEDIA_BYTES:
                    raise _PublicMediaBoundaryError(
                        "invalid",
                        "PUBLIC_MEDIA_TOO_LARGE",
                        "The public media exceeds the byte limit.",
                    )
        except _PublicMediaBoundaryError:
            raise
        except httpx.TimeoutException as exc:
            raise _PublicMediaBoundaryError(
                "timeout",
                "PUBLIC_MEDIA_FETCH_TIMEOUT",
                "The public media fetch timed out.",
            ) from exc
        except httpx.HTTPError as exc:
            raise _PublicMediaBoundaryError(
                "unavailable",
                "PUBLIC_MEDIA_FETCH_UNAVAILABLE",
                "The public media fetch was unavailable.",
            ) from exc

        content_type = response.headers.get("content-type", "").split(
            ";",
            1,
        )[0].strip().lower()
        image_bytes = bytes(body)
        if not content_type.startswith("image/") or not _image_magic_matches(
            content_type,
            image_bytes,
        ):
            raise _PublicMediaBoundaryError(
                "invalid",
                "PUBLIC_MEDIA_TYPE_INVALID",
                "The public media response is not a supported image.",
            )
        _validate_public_image_decode(image_bytes)
        return content_type, image_bytes, current_target.canonical_url


def _validated_public_url(value: str) -> str:
    """Accept only public HTTP(S) URL targets with no credentials or fragment."""

    return _validated_public_target(value).canonical_url


def _validated_public_target(value: str) -> _PublicMediaTarget:
    """Validate one URL and retain its vetted address for the transport hop."""

    if value != value.strip() or any(char in value for char in "\r\n\t"):
        raise _PublicMediaBoundaryError(
            "invalid",
            "PUBLIC_MEDIA_URL_INVALID",
            "The public media URL is invalid.",
        )
    try:
        parsed = urlsplit(value)
        hostname = parsed.hostname
        port = parsed.port
    except ValueError as exc:
        raise _PublicMediaBoundaryError(
            "invalid",
            "PUBLIC_MEDIA_URL_INVALID",
            "The public media URL is invalid.",
        ) from exc
    if (
        parsed.scheme not in ("http", "https")
        or not hostname
        or parsed.username
        or parsed.password
        or parsed.fragment
        or (port is not None and not 1 <= port <= 65535)
    ):
        raise _PublicMediaBoundaryError(
            "invalid",
            "PUBLIC_MEDIA_URL_INVALID",
            "The public media URL is invalid.",
        )
    try:
        addresses = socket.getaddrinfo(
            hostname,
            parsed.port,
            type=socket.SOCK_STREAM,
        )
    except socket.gaierror as exc:
        raise _PublicMediaBoundaryError(
            "unavailable",
            "PUBLIC_MEDIA_FETCH_UNAVAILABLE",
            "The public media hostname could not be resolved.",
        ) from exc
    resolved_ips: list[str] = []
    for address in addresses:
        try:
            ip_address = ipaddress.ip_address(address[4][0])
        except (IndexError, ValueError) as exc:
            raise _PublicMediaBoundaryError(
                "unavailable",
                "PUBLIC_MEDIA_FETCH_UNAVAILABLE",
                "The public media hostname returned an invalid address.",
            ) from exc
        if (
            ip_address.is_private
            or ip_address.is_loopback
            or ip_address.is_link_local
            or ip_address.is_multicast
            or ip_address.is_reserved
            or ip_address.is_unspecified
        ):
            raise _PublicMediaBoundaryError(
                "denied",
                "PUBLIC_MEDIA_URL_DENIED",
                "The public media URL resolves to a denied address.",
            )
        resolved_ips.append(str(ip_address))
    if not resolved_ips:
        raise _PublicMediaBoundaryError(
            "unavailable",
            "PUBLIC_MEDIA_FETCH_UNAVAILABLE",
            "The public media hostname returned no addresses.",
        )
    return _PublicMediaTarget(
        canonical_url=value,
        hostname=hostname,
        host_header=parsed.netloc,
        resolved_ip=resolved_ips[0],
        scheme=parsed.scheme,
        port=parsed.port,
        path=parsed.path,
        query=parsed.query,
    )


def _image_magic_matches(content_type: str, image_bytes: bytes) -> bool:
    """Require a supported image MIME declaration and matching magic bytes."""

    signatures = {
        "image/png": b"\x89PNG\r\n\x1a\n",
        "image/jpeg": b"\xff\xd8\xff",
        "image/gif": b"GIF8",
        "image/webp": b"RIFF",
    }
    signature = signatures.get(content_type)
    if signature is None or not image_bytes.startswith(signature):
        return False
    return content_type != "image/webp" or image_bytes[8:12] == b"WEBP"


def _validate_public_image_decode(image_bytes: bytes) -> None:
    """Decode-check an image and reject dimensions outside the safe bounds."""

    try:
        with Image.open(BytesIO(image_bytes)) as image:
            image.verify()
        with Image.open(BytesIO(image_bytes)) as image:
            width, height = image.size
    except (Image.DecompressionBombError, UnidentifiedImageError, OSError, ValueError) as exc:
        raise _PublicMediaBoundaryError(
            "invalid",
            "PUBLIC_MEDIA_DECODE_INVALID",
            "The public media image failed decoding.",
        ) from exc
    if (
        width < 1
        or height < 1
        or width > _MAX_PUBLIC_MEDIA_DIMENSION
        or height > _MAX_PUBLIC_MEDIA_DIMENSION
    ):
        raise _PublicMediaBoundaryError(
            "invalid",
            "PUBLIC_MEDIA_DECODE_INVALID",
            "The public media image dimensions are invalid.",
        )
