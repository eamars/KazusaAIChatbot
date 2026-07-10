"""Bounded external image-intake subagent for complex task resolution."""

from __future__ import annotations

import base64
import ipaddress
import socket
from io import BytesIO
from urllib.parse import urlsplit

import httpx
from PIL import Image, UnidentifiedImageError

from kazusa_ai_chatbot.complex_task_resolver.contracts import (
    COMPLEX_TASK_SUBAGENT_RESULT_VERSION,
    ComplexTaskSubagentRequestV1,
    ComplexTaskSubagentResultV1,
    validate_complex_task_subagent_request,
    validate_complex_task_subagent_result,
)
from kazusa_ai_chatbot.media_inspection.service import inspect_media

SUBAGENT = "media"
DESCRIPTION = "bounded external image inspection after deterministic URL safety checks"
SUPPORTED_ACTIONS = ("inspect_media",)
OWNED_NODE_KINDS = ("media_inspection_task",)
DEFAULT_ACTION = "inspect_media"
_MAX_REDIRECTS = 3
_MAX_IMAGE_BYTES = 6 * 1024 * 1024
_MAX_IMAGE_DIMENSION = 8192
_FETCH_TIMEOUT_SECONDS = 15.0


class MediaSubagent:
    """Fetch a safe external image and call the shared visual inspector."""

    def __init__(self, inspect_func=inspect_media) -> None:
        """Create the media intake subagent with the production inspector."""

        self._inspect_func = inspect_func

    async def run(
        self,
        task: ComplexTaskSubagentRequestV1,
        context: dict[str, object],
        max_attempts: int = 1,
    ) -> ComplexTaskSubagentResultV1:
        """Return external image observations or a bounded fetch failure."""

        del context, max_attempts
        request = validate_complex_task_subagent_request(task)
        url = request["payload"].get("url")
        question = request["payload"].get("question")
        if (
            not isinstance(url, str)
            or not isinstance(question, str)
            or not question.strip()
        ):
            return _failure(request, "missing external image URL or question")
        try:
            content_type, image_bytes, final_url = await _fetch_image(url)
        except ValueError as exc:
            return _failure(request, str(exc))
        except httpx.HTTPError:
            return _failure(request, "external image fetch failed")
        inspection = await self._inspect_func({
            "schema_version": "media_inspection_request.v1",
            "source": "complex_external_media",
            "media_kind": "image",
            "content_type": content_type,
            "base64_data": base64.b64encode(image_bytes).decode("ascii"),
            "question": question.strip(),
            "existing_descriptor": "",
        })
        answer = inspection["answer"]
        status = inspection["status"]
        resolved = status in ("answered", "uncertain")
        result: ComplexTaskSubagentResultV1 = {
            "schema_version": COMPLEX_TASK_SUBAGENT_RESULT_VERSION,
            "resolved": resolved,
            "status": "resolved" if status == "answered" else "partial",
            "result": {
                "summary": answer,
                "evidence_boundary_notes": (
                    inspection["evidence_boundary_notes"]
                ),
                "source_url": final_url,
                "content_type": content_type,
                "byte_count": len(image_bytes),
            },
            "attempts": 1,
            "cache": {"enabled": False},
            "trace": {"media_inspection_called": True},
            "unresolved_items": (
                [] if resolved else ["image inspection was uncertain"]
            ),
        }
        validated = validate_complex_task_subagent_result(result)
        return validated


def create() -> MediaSubagent:
    """Create the production external-media subagent."""

    return MediaSubagent()


async def _fetch_image(url: str) -> tuple[str, bytes, str]:
    """Fetch one bounded image while checking every redirect target."""

    current_url = _validated_public_url(url)
    for _ in range(_MAX_REDIRECTS + 1):
        async with httpx.AsyncClient(
            follow_redirects=False,
            timeout=_FETCH_TIMEOUT_SECONDS,
        ) as client:
            response = await client.get(current_url)
        if response.is_redirect:
            location = response.headers.get("location")
            if not location:
                raise ValueError("external image redirect had no destination")
            current_url = _validated_public_url(
                str(response.url.join(location))
            )
            continue
        response.raise_for_status()
        content_type = response.headers.get("content-type", "").split(
            ";",
            1,
        )[0].lower()
        image_bytes = response.content
        if len(image_bytes) > _MAX_IMAGE_BYTES:
            raise ValueError("external image exceeds byte limit")
        if (
            not content_type.startswith("image/")
            or not _image_magic_matches(content_type, image_bytes)
        ):
            raise ValueError("external response is not a valid image")
        _validate_image_decode(image_bytes)
        return content_type, image_bytes, current_url
    raise ValueError("external image exceeded redirect limit")


def _validated_public_url(value: str) -> str:
    """Accept only public HTTP(S) URL targets with no credential component."""

    parsed = urlsplit(value)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError("external image URL must use http or https with a host")
    if parsed.username or parsed.password or parsed.fragment:
        raise ValueError("external image URL contains unsupported credentials or fragment")
    try:
        addresses = socket.getaddrinfo(
            parsed.hostname,
            None,
            type=socket.SOCK_STREAM,
        )
    except socket.gaierror as exc:
        raise ValueError("external image hostname could not be resolved") from exc
    for address in addresses:
        ip_address = ipaddress.ip_address(address[4][0])
        if (
            ip_address.is_private
            or ip_address.is_loopback
            or ip_address.is_link_local
            or ip_address.is_multicast
            or ip_address.is_reserved
            or ip_address.is_unspecified
        ):
            raise ValueError(
                "external image URL resolves to a private or reserved address"
            )
    result = value
    return result


def _image_magic_matches(content_type: str, image_bytes: bytes) -> bool:
    """Require known image bytes as well as the remote MIME declaration."""

    signatures = {
        "image/png": b"\x89PNG\r\n\x1a\n",
        "image/jpeg": b"\xff\xd8\xff",
        "image/gif": b"GIF8",
        "image/webp": b"RIFF",
    }
    signature = signatures.get(content_type)
    if signature is None or not image_bytes.startswith(signature):
        return False
    if content_type == "image/webp" and image_bytes[8:12] != b"WEBP":
        return False
    return True


def _validate_image_decode(image_bytes: bytes) -> None:
    """Decode-check a bounded image and reject excessive dimensions."""

    try:
        with Image.open(BytesIO(image_bytes)) as image:
            image.verify()
        with Image.open(BytesIO(image_bytes)) as image:
            width, height = image.size
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError("external response failed image decoding") from exc
    if (
        width < 1
        or height < 1
        or width > _MAX_IMAGE_DIMENSION
        or height > _MAX_IMAGE_DIMENSION
    ):
        raise ValueError("external image dimensions exceed the safety limit")


def _failure(
    request: ComplexTaskSubagentRequestV1,
    reason: str,
) -> ComplexTaskSubagentResultV1:
    """Return one sanitized media-fetch boundary result."""

    result: ComplexTaskSubagentResultV1 = {
        "schema_version": COMPLEX_TASK_SUBAGENT_RESULT_VERSION,
        "resolved": False,
        "status": "failed",
        "result": {"evidence_boundary_notes": [reason]},
        "attempts": 1,
        "cache": {"enabled": False},
        "trace": {"media_inspection_called": False},
        "unresolved_items": [reason],
    }
    validated = validate_complex_task_subagent_result(result)
    return validated
