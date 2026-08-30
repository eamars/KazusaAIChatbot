"""Attached and public-media semantic capability tests."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import socket as stdlib_socket
import subprocess
import sys
from types import SimpleNamespace
from typing import ClassVar, Self
from urllib.parse import urljoin, urlsplit, urlunsplit

import pytest


def _png_bytes(*, width: int = 1, height: int = 1) -> bytes:
    """Create a deterministic small PNG for the decode boundary."""

    from io import BytesIO

    from PIL import Image

    buffer = BytesIO()
    Image.new("RGBA", (width, height), (20, 40, 60, 255)).save(
        buffer, format="PNG"
    )
    return buffer.getvalue()


class _Url(str):
    """HTTP URL value supporting both httpx-style and stdlib joins."""

    def join(self, location: str) -> _Url:
        return _Url(urljoin(self, location))


class _PublicResponse:
    """Minimal streamed response used by the public-media boundary."""

    def __init__(
        self,
        url: str,
        *,
        body: bytes = b"",
        content_type: str = "image/png",
        status_code: int = 200,
        location: str | None = None,
        content_length: int | None = None,
        chunks: tuple[bytes, ...] | None = None,
    ) -> None:
        self.url = _Url(url)
        self.status_code = status_code
        self.headers: dict[str, str] = {}
        if content_type:
            self.headers["content-type"] = content_type
        if location is not None:
            self.headers["location"] = location
        if content_length is not None:
            self.headers["content-length"] = str(content_length)
        self.content = body
        self._chunks = chunks or (body,)

    @property
    def is_redirect(self) -> bool:
        return self.status_code in {301, 302, 303, 307, 308}

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise _FakeHttpError(f"HTTP {self.status_code}")

    async def aiter_bytes(self):
        for chunk in self._chunks:
            yield chunk


class _FakeHttpError(Exception):
    """Transport exception exposed through the patched httpx boundary."""


class _FakeTimeout(_FakeHttpError):
    """Deterministic timeout exception for failure-map coverage."""


class _PublicHttpClient:
    """Route fake requests to predeclared response streams."""

    routes: ClassVar[dict[str, _PublicResponse | BaseException]] = {}
    requests: ClassVar[list[str]] = []
    request_details: ClassVar[list[dict[str, object]]] = []
    client_options: ClassVar[list[dict[str, object]]] = []

    def __init__(self, **options: object) -> None:
        self.client_options.append(dict(options))

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *_args: object) -> None:
        return None

    async def get(
        self,
        url: object,
        *,
        headers: dict[str, str] | None = None,
        extensions: dict[str, object] | None = None,
    ) -> _PublicResponse:
        transport_url = str(url)
        parsed = urlsplit(transport_url)
        host_header = (headers or {}).get("host", parsed.netloc)
        key = urlunsplit((parsed.scheme, host_header, parsed.path, parsed.query, ""))
        self.requests.append(transport_url)
        self.request_details.append({
            "transport_url": transport_url,
            "headers": dict(headers or {}),
            "extensions": dict(extensions or {}),
        })
        response = self.routes[key]
        if isinstance(response, BaseException):
            raise response
        return response


def _install_public_boundaries(
    monkeypatch: pytest.MonkeyPatch,
    media_module: object,
    *,
    routes: dict[str, _PublicResponse | BaseException],
    addresses: dict[str, str | BaseException],
) -> None:
    """Patch DNS and HTTP only, leaving the media service implementation real."""

    class _Socket:
        AF_INET = stdlib_socket.AF_INET
        SOCK_STREAM = stdlib_socket.SOCK_STREAM
        gaierror = stdlib_socket.gaierror

        @staticmethod
        def getaddrinfo(host: str, *_args: object, **_kwargs: object):
            value = addresses.get(host, "93.184.216.34")
            if isinstance(value, BaseException):
                raise value
            return [(
                stdlib_socket.AF_INET,
                stdlib_socket.SOCK_STREAM,
                6,
                "",
                (value, 0),
            )]

    _PublicHttpClient.routes = routes
    _PublicHttpClient.requests = []
    _PublicHttpClient.request_details = []
    _PublicHttpClient.client_options = []
    fake_httpx = SimpleNamespace(
        AsyncClient=_PublicHttpClient,
        HTTPError=_FakeHttpError,
        TimeoutException=_FakeTimeout,
        ConnectError=_FakeHttpError,
        NetworkError=_FakeHttpError,
    )
    monkeypatch.setattr(media_module, "socket", _Socket, raising=False)
    monkeypatch.setattr(media_module, "httpx", fake_httpx, raising=False)


def _public_service(media_module: object, inspector):
    """Build the real service with a deterministic vision collaborator."""

    from kazusa_ai_chatbot.dsh_tool_gateway.contracts import OpaqueReferenceCodec

    return media_module.MediaSemanticService(
        scope=("debug", "channel-1", "user-1"),
        codec=OpaqueReferenceCodec(b"public-media-test"),
        inspect=inspector,
    )


@pytest.mark.asyncio
async def test_attached_media_inspection_accepts_only_brain_issued_semantic_refs() -> None:
    from kazusa_ai_chatbot.dsh_tool_gateway.contracts import OpaqueReferenceCodec
    from kazusa_ai_chatbot.dsh_tool_gateway.media import (
        MediaSemanticService,
        issue_attached_media_reference,
    )

    scope = ("debug", "channel-1", "user-1")
    codec = OpaqueReferenceCodec(b"secret")
    reference = issue_attached_media_reference(
        codec=codec, scope=scope, cache_ref="cache-1"
    )

    async def inspect(request):
        assert request["source"] == "test"
        return {
            "status": "answered",
            "answer": "a cup",
            "evidence_boundary_notes": ["visible"],
        }

    def get_media(received_scope, cache_ref):
        assert received_scope == scope
        assert cache_ref == "cache-1"
        return {
            "content_type": "image/png",
            "base64_data": "aGVsbG8=",
            "existing_descriptor": "",
        }

    service = MediaSemanticService(
        scope=scope, codec=codec, get_media=get_media, inspect=inspect
    )
    result = await service.inspect_attached_media(
        attached_media_ref=reference, question="What is visible?"
    )
    assert result.entities[0]["answer"] == "a cup"
    foreign = issue_attached_media_reference(
        codec=codec,
        scope=("debug", "other", "user-1"),
        cache_ref="cache-1",
    )
    denied = await service.inspect_attached_media(
        attached_media_ref=foreign, question="What is visible?"
    )
    assert denied.status == "denied"


def test_attached_media_cache_is_available_to_a_separate_worker_process(
    tmp_path,
    monkeypatch,
) -> None:
    """The Brain cache mirror crosses the sidecar worker process boundary."""

    from kazusa_ai_chatbot.dsh_tool_gateway.media import persist_attached_media
    from kazusa_ai_chatbot.media_inspection.session_cache import (
        clear_session_media,
        put_session_media,
    )

    scope = ("debug", "channel-1", "user-1")
    monkeypatch.setenv("KAZUSA_DSH_DATA_ROOT", str(tmp_path.resolve()))
    references = put_session_media(scope, [{
        "media_kind": "image",
        "content_type": "image/png",
        "base64_data": "AA==",
        "source_summary": "one pixel",
    }])
    persist_attached_media(scope, references)
    clear_session_media(scope)
    cache_ref = str(references[0]["cache_ref"])
    script = (
        "import json,sys; "
        "from kazusa_ai_chatbot.dsh_tool_gateway.media import get_attached_media; "
        "value=get_attached_media(('debug','channel-1','user-1'),sys.argv[1]); "
        "print(json.dumps(value))"
    )
    environment = os.environ.copy()
    completed = subprocess.run(
        [sys.executable, "-c", script, cache_ref],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["content_type"] == "image/png"
    assert payload["base64_data"] == "AA=="


@pytest.mark.asyncio
async def test_public_media_dispatch_accepts_only_url_and_question() -> None:
    """The real dispatcher forwards exactly the public URL and visual question."""

    from kazusa_ai_chatbot.dsh_tool_gateway.authority import SignedSemanticCallV1
    from kazusa_ai_chatbot.dsh_tool_gateway.contracts import (
        KazusaSemanticCapabilityResultV1,
    )
    from kazusa_ai_chatbot.dsh_tool_gateway.dispatch import (
        SemanticCapabilityDispatcher,
    )

    received: list[dict[str, object]] = []

    class Media:
        async def inspect_public_media(self, **arguments):
            received.append(dict(arguments))
            return KazusaSemanticCapabilityResultV1.success(
                entities=[{"status": "answered"}]
            )

    authority = SimpleNamespace(
        catalog_digest="sha256:test-catalog",
        service_scope={
            "platform": "debug",
            "platform_channel_id": "channel-1",
            "global_user_id": "user-1",
        },
    )

    def call(arguments: dict[str, object]) -> SignedSemanticCallV1:
        return SignedSemanticCallV1(
            call_id="call-1",
            operation="kazusa_inspect_public_media",
            arguments=arguments,
            authority=authority,
            arguments_digest="sha256:arguments",
            issued_reference_digest="sha256:issued",
            idempotency_key=None,
            signature="signature",
        )

    dispatcher = SemanticCapabilityDispatcher(
        conversation=SimpleNamespace(),
        memory=SimpleNamespace(),
        people=SimpleNamespace(),
        recall_calendar=SimpleNamespace(),
        media=Media(),
        expected_catalog_digest="sha256:test-catalog",
    )
    result = await dispatcher.dispatch(call({
        "public_media_url": "https://example.test/picture.png",
        "question": "What is visible?",
    }))
    assert result.status == "ok"
    assert received == [{
        "public_media_url": "https://example.test/picture.png",
        "question": "What is visible?",
    }]
    invalid = await dispatcher.dispatch(call({
        "public_media_url": "https://example.test/picture.png",
        "question": "What is visible?",
        "capability_token": "must-not-forward",
    }))
    assert invalid.status == "invalid"
    assert invalid.error is not None
    assert invalid.error.code == "SEMANTIC_ARGUMENTS_INVALID"
    assert received == [{
        "public_media_url": "https://example.test/picture.png",
        "question": "What is visible?",
    }]


@pytest.mark.asyncio
async def test_public_media_inspection_preserves_bounded_safe_fetch_and_visual_result(
    monkeypatch,
) -> None:
    """A valid public response yields an exact prompt-safe entity and receipt."""

    from kazusa_ai_chatbot.dsh_tool_gateway import media
    from kazusa_ai_chatbot.dsh_tool_gateway.contracts import content_digest

    image_bytes = _png_bytes()
    inspections: list[dict[str, object]] = []

    async def inspector(request: dict[str, object]) -> dict[str, object]:
        inspections.append(request)
        return {
            "schema_version": "media_inspection_result.v1",
            "status": "answered",
            "answer": "a blue square",
            "evidence_boundary_notes": ["visible pixels only"],
        }

    url = "https://public.test/picture.png"
    _install_public_boundaries(
        monkeypatch,
        media,
        routes={url: _PublicResponse(
            url,
            body=image_bytes,
            content_type="image/png",
            chunks=(image_bytes[:8], image_bytes[8:]),
        )},
        addresses={"public.test": "93.184.216.34"},
    )
    result = await _public_service(media, inspector).inspect_public_media(
        public_media_url=url,
        question="What color is the square?",
    )

    digest = content_digest({
        "final_url": url,
        "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
    })
    semantic_ref = f"public-media:{digest.removeprefix('sha256:')}"
    assert result.status == "ok"
    assert result.entities == ({
        "status": "answered",
        "answer": "a blue square",
        "source_url": url,
        "content_type": "image/png",
        "byte_count": len(image_bytes),
        "evidence_boundary_notes": ["visible pixels only"],
    },)
    assert len(result.evidence) == 1
    assert result.evidence[0].semantic_ref == semantic_ref
    assert result.evidence[0].receipt_id == f"receipt-public-media-{digest.removeprefix('sha256:')}"
    assert inspections[0]["source"] == "dsh_public_media"
    assert inspections[0]["question"] == "What color is the square?"
    serialized = json.dumps(result.to_dict(), sort_keys=True)
    assert base64.b64encode(image_bytes).decode("ascii") not in serialized
    assert "image_bytes" not in serialized
    assert "base64_data" not in serialized


@pytest.mark.asyncio
async def test_public_media_pins_vetted_ip_across_dns_rebinding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A later private DNS answer cannot redirect the inspected request."""

    from kazusa_ai_chatbot.dsh_tool_gateway import media

    image_bytes = _png_bytes()
    inspections: list[dict[str, object]] = []

    async def inspector(request: dict[str, object]) -> dict[str, object]:
        inspections.append(request)
        return {
            "schema_version": "media_inspection_result.v1",
            "status": "answered",
            "answer": "a public image",
            "evidence_boundary_notes": [],
        }

    url = "https://public.test/rebinding.png"
    _install_public_boundaries(
        monkeypatch,
        media,
        routes={url: _PublicResponse(url, body=image_bytes)},
        addresses={"public.test": "93.184.216.34"},
    )

    class _RebindingSocket:
        AF_INET = stdlib_socket.AF_INET
        SOCK_STREAM = stdlib_socket.SOCK_STREAM
        gaierror = stdlib_socket.gaierror
        calls: ClassVar[list[str]] = []

        @classmethod
        def getaddrinfo(cls, host: str, *_args: object, **_kwargs: object):
            cls.calls.append(host)
            address = "93.184.216.34" if len(cls.calls) == 1 else "10.0.0.4"
            return [(
                stdlib_socket.AF_INET,
                stdlib_socket.SOCK_STREAM,
                6,
                "",
                (address, 0),
            )]

    monkeypatch.setattr(media, "socket", _RebindingSocket)
    result = await _public_service(media, inspector).inspect_public_media(
        public_media_url=url,
        question="What is visible?",
    )

    assert result.status == "ok"
    assert len(inspections) == 1
    assert _RebindingSocket.calls == ["public.test"]
    request = _PublicHttpClient.request_details[0]
    assert urlsplit(str(request["transport_url"])).hostname == "93.184.216.34"
    assert request["headers"] == {"host": "public.test"}
    assert request["extensions"] == {"sni_hostname": "public.test"}
    assert _PublicHttpClient.client_options[0]["trust_env"] is False


@pytest.mark.asyncio
async def test_public_media_rejects_private_redirect_oversize_or_invalid_image_before_inspection(
    monkeypatch,
) -> None:
    """All deterministic fetch, URL, and decode faults map before vision runs."""

    from kazusa_ai_chatbot.dsh_tool_gateway import media

    max_bytes = 6 * 1024 * 1024
    valid_png = _png_bytes()
    cases = [
        {
            "name": "question-required",
            "url": "https://public.test/image.png",
            "routes": {},
            "addresses": {},
            "question": " ",
            "status": "invalid",
            "code": "QUESTION_REQUIRED",
        },
        {
            "name": "malformed-url",
            "url": "file:///private/image.png",
            "routes": {},
            "addresses": {},
            "question": "Describe it",
            "status": "invalid",
            "code": "PUBLIC_MEDIA_URL_INVALID",
        },
        {
            "name": "credentialed-url",
            "url": "https://user:secret@public.test/image.png",
            "routes": {},
            "addresses": {},
            "question": "Describe it",
            "status": "invalid",
            "code": "PUBLIC_MEDIA_URL_INVALID",
        },
        {
            "name": "fragment-url",
            "url": "https://public.test/image.png#fragment",
            "routes": {},
            "addresses": {},
            "question": "Describe it",
            "status": "invalid",
            "code": "PUBLIC_MEDIA_URL_INVALID",
        },
        {
            "name": "initial-private",
            "url": "https://private.test/image.png",
            "routes": {},
            "addresses": {"private.test": "127.0.0.1"},
            "question": "Describe it",
            "status": "denied",
            "code": "PUBLIC_MEDIA_URL_DENIED",
        },
        {
            "name": "redirected-private",
            "url": "https://public.test/start",
            "routes": {
                "https://public.test/start": _PublicResponse(
                    "https://public.test/start",
                    status_code=302,
                    location="https://private.test/image.png",
                ),
            },
            "addresses": {
                "public.test": "93.184.216.34",
                "private.test": "10.0.0.4",
            },
            "question": "Describe it",
            "status": "denied",
            "code": "PUBLIC_MEDIA_URL_DENIED",
        },
        {
            "name": "four-redirects",
            "url": "https://public.test/0",
            "routes": {
                f"https://public.test/{index}": _PublicResponse(
                    f"https://public.test/{index}",
                    status_code=302,
                    location=f"https://public.test/{index + 1}",
                )
                for index in range(4)
            },
            "addresses": {"public.test": "93.184.216.34"},
            "question": "Describe it",
            "status": "invalid",
            "code": "PUBLIC_MEDIA_REDIRECT_INVALID",
        },
        {
            "name": "declared-too-large",
            "url": "https://public.test/declared",
            "routes": {
                "https://public.test/declared": _PublicResponse(
                    "https://public.test/declared",
                    body=valid_png,
                    content_length=max_bytes + 1,
                ),
            },
            "addresses": {"public.test": "93.184.216.34"},
            "question": "Describe it",
            "status": "invalid",
            "code": "PUBLIC_MEDIA_TOO_LARGE",
        },
        {
            "name": "streamed-too-large",
            "url": "https://public.test/streamed",
            "routes": {
                "https://public.test/streamed": _PublicResponse(
                    "https://public.test/streamed",
                    chunks=(b"x" * max_bytes, b"x"),
                ),
            },
            "addresses": {"public.test": "93.184.216.34"},
            "question": "Describe it",
            "status": "invalid",
            "code": "PUBLIC_MEDIA_TOO_LARGE",
        },
        {
            "name": "mime-magic-mismatch",
            "url": "https://public.test/mismatch",
            "routes": {
                "https://public.test/mismatch": _PublicResponse(
                    "https://public.test/mismatch",
                    body=valid_png,
                    content_type="image/jpeg",
                ),
            },
            "addresses": {"public.test": "93.184.216.34"},
            "question": "Describe it",
            "status": "invalid",
            "code": "PUBLIC_MEDIA_TYPE_INVALID",
        },
        {
            "name": "decode-invalid",
            "url": "https://public.test/invalid",
            "routes": {
                "https://public.test/invalid": _PublicResponse(
                    "https://public.test/invalid",
                    body=b"\x89PNG\r\n\x1a\nnot-an-image",
                ),
            },
            "addresses": {"public.test": "93.184.216.34"},
            "question": "Describe it",
            "status": "invalid",
            "code": "PUBLIC_MEDIA_DECODE_INVALID",
        },
        {
            "name": "dimension-invalid",
            "url": "https://public.test/dimensions",
            "routes": {
                "https://public.test/dimensions": _PublicResponse(
                    "https://public.test/dimensions",
                    body=_png_bytes(width=8193),
                ),
            },
            "addresses": {"public.test": "93.184.216.34"},
            "question": "Describe it",
            "status": "invalid",
            "code": "PUBLIC_MEDIA_DECODE_INVALID",
        },
        {
            "name": "dns-failure",
            "url": "https://unknown.test/image.png",
            "routes": {},
            "addresses": {"unknown.test": stdlib_socket.gaierror("missing")},
            "question": "Describe it",
            "status": "unavailable",
            "code": "PUBLIC_MEDIA_FETCH_UNAVAILABLE",
        },
        {
            "name": "timeout",
            "url": "https://public.test/timeout",
            "routes": {"https://public.test/timeout": _FakeTimeout("slow")},
            "addresses": {"public.test": "93.184.216.34"},
            "question": "Describe it",
            "status": "timeout",
            "code": "PUBLIC_MEDIA_FETCH_TIMEOUT",
        },
        {
            "name": "http-failure",
            "url": "https://public.test/error",
            "routes": {
                "https://public.test/error": _PublicResponse(
                    "https://public.test/error",
                    status_code=503,
                ),
            },
            "addresses": {"public.test": "93.184.216.34"},
            "question": "Describe it",
            "status": "unavailable",
            "code": "PUBLIC_MEDIA_FETCH_UNAVAILABLE",
        },
    ]
    for case in cases:
        inspections: list[dict[str, object]] = []

        async def inspector(
            request: dict[str, object],
            captured: list[dict[str, object]] = inspections,
        ) -> dict[str, object]:
            captured.append(request)
            return {
                "schema_version": "media_inspection_result.v1",
                "status": "answered",
                "answer": "should not run",
                "evidence_boundary_notes": ["unexpected"],
            }

        _install_public_boundaries(
            monkeypatch,
            media,
            routes=case["routes"],
            addresses=case["addresses"],
        )
        result = await _public_service(media, inspector).inspect_public_media(
            public_media_url=case["url"],
            question=case["question"],
        )
        assert result.status == case["status"], case["name"]
        assert result.error is not None, case["name"]
        assert result.error.code == case["code"], case["name"]
        assert inspections == [], case["name"]


def test_media_inspection_source_contract_accepts_dsh_public_media() -> None:
    """DSH public-media provenance crosses the shared inspector boundary."""

    from kazusa_ai_chatbot.media_inspection.contracts import (
        validate_media_inspection_request,
    )

    request = {
        "schema_version": "media_inspection_request.v1",
        "source": "dsh_public_media",
        "media_kind": "image",
        "content_type": "image/png",
        "base64_data": base64.b64encode(_png_bytes()).decode("ascii"),
        "question": "Describe it",
        "existing_descriptor": "",
    }
    assert validate_media_inspection_request(request) == request
