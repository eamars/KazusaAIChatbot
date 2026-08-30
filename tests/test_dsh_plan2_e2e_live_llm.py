"""Named real-local-model end-to-end sign-off cases for DSH V2."""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
from collections.abc import Mapping
from datetime import UTC, datetime
from hashlib import sha256
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock, Thread
from typing import Any
from uuid import uuid4

import pytest

from tests.test_dsh_standard_profile_live_llm import (
    _require_live_backend,
    _resolve_live,
)

pytestmark = pytest.mark.live_llm


class _LoopbackAdapterServer:
    """Capture one registered Brain adapter's capability and send calls."""

    def __init__(self, *, platform: str, shared_secret: str) -> None:
        if not platform.strip():
            raise ValueError("loopback adapter platform is required")
        if not shared_secret.strip():
            raise ValueError("loopback adapter shared secret is required")
        self._platform = platform
        self._shared_secret = shared_secret
        self._lock = Lock()
        self._capability_payloads: list[dict[str, Any]] = []
        self._delivery_payloads: list[dict[str, Any]] = []
        owner = self

        class Handler(BaseHTTPRequestHandler):
            """Serve the two typed callback endpoints for this test."""

            def do_POST(self) -> None:
                owner._handle_post(self)

            def log_message(self, format: str, *args: object) -> None:
                del format, args

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True
        self._thread = Thread(
            target=self._server.serve_forever,
            name="dsh-e2e-loopback-adapter",
            daemon=True,
        )

    @property
    def callback_url(self) -> str:
        """Return the bound callback base URL."""

        address = self._server.server_address
        return f"http://127.0.0.1:{address[1]}"

    def start(self) -> None:
        """Start serving callback requests."""

        self._thread.start()

    def stop(self) -> None:
        """Stop the callback server and join its bounded worker thread."""

        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)
        if self._thread.is_alive():
            raise RuntimeError("loopback adapter thread did not stop")

    def capability_payloads(self) -> list[dict[str, Any]]:
        """Return a thread-safe snapshot of capability probes."""

        with self._lock:
            return [dict(item) for item in self._capability_payloads]

    def delivery_payloads(self) -> list[dict[str, Any]]:
        """Return a thread-safe snapshot of durable send responses."""

        with self._lock:
            return [dict(item) for item in self._delivery_payloads]

    def _handle_post(self, request: BaseHTTPRequestHandler) -> None:
        """Handle one authenticated adapter callback request."""

        expected_authorization = f"Bearer {self._shared_secret}"
        if request.headers.get("Authorization") != expected_authorization:
            self._respond(request, 401, {"available": False})
            return
        raw_length = request.headers.get("Content-Length")
        try:
            length = int(raw_length or "")
        except ValueError:
            self._respond(request, 400, {"error": "invalid content length"})
            return
        try:
            payload_value = json.loads(
                request.rfile.read(length).decode("utf-8")
            )
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._respond(request, 400, {"error": "invalid JSON"})
            return
        if not isinstance(payload_value, dict):
            self._respond(request, 400, {"error": "JSON object required"})
            return

        if request.path == "/send_message/capability":
            with self._lock:
                self._capability_payloads.append(dict(payload_value))
            self._respond(request, 200, {"available": True})
            return
        if request.path != "/send_message":
            self._respond(request, 404, {"error": "unknown callback"})
            return

        message_id = f"dsh-e2e-adapter-message-{uuid4().hex}"
        sent_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
        delivery = {
            **payload_value,
            "message_id": message_id,
            "sent_at": sent_at,
        }
        with self._lock:
            self._delivery_payloads.append(delivery)
        self._respond(
            request,
            200,
            {
                "platform": self._platform,
                "channel_id": payload_value.get("channel_id", ""),
                "message_id": message_id,
                "sent_at": sent_at,
            },
        )

    @staticmethod
    def _respond(
        request: BaseHTTPRequestHandler,
        status_code: int,
        payload: Mapping[str, object],
    ) -> None:
        """Send one compact JSON callback response."""

        encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        request.send_response(status_code)
        request.send_header("Content-Type", "application/json")
        request.send_header("Content-Length", str(len(encoded)))
        request.send_header("Connection", "close")
        request.end_headers()
        request.wfile.write(encoded)


def _read_exact_dsh_events(
    *,
    resolution_thread_id: str,
    segment_id: str,
    event_types: tuple[str, ...] | None = None,
) -> list[tuple[int, str, dict[str, Any]]]:
    """Read exact-session events, optionally restricted to event types."""

    import zstandard

    from agentic_resolver.contracts import DSH_RELEASE

    if not resolution_thread_id.strip() or not segment_id.strip():
        pytest.fail("DSH session identity is incomplete for audit")
    session_identity = f"{resolution_thread_id}\0{segment_id}".encode()
    session_id = (
        "kazusa-resolution-"
        f"{sha256(session_identity).hexdigest()[:32]}"
    )
    data_root = os.environ.get("KAZUSA_DSH_DATA_ROOT")
    if data_root is None or not data_root.strip():
        pytest.fail("KAZUSA_DSH_DATA_ROOT is unavailable for DSH audit")
    store_path = Path(data_root) / "dsh" / DSH_RELEASE / "sessions.sqlite"
    if not store_path.is_file():
        pytest.fail("DSH session store is unavailable for the live audit")

    uri = f"file:{store_path.as_posix()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        session_rows = connection.execute(
            "SELECT id FROM sessions WHERE id = ?",
            (session_id,),
        ).fetchall()
        if session_rows != [(session_id,)]:
            pytest.fail("the DSH identity did not resolve one root session")
        if event_types is None:
            event_rows = connection.execute(
                "SELECT seq, type, data FROM events "
                "WHERE session_id = ? ORDER BY seq",
                (session_id,),
            ).fetchall()
        else:
            placeholders = ", ".join("?" for _ in event_types)
            event_rows = connection.execute(
                "SELECT seq, type, data FROM events "
                f"WHERE session_id = ? AND type IN ({placeholders}) "
                "ORDER BY seq",
                (session_id, *event_types),
            ).fetchall()

    decoder = zstandard.ZstdDecompressor()
    events: list[tuple[int, str, dict[str, Any]]] = []
    for sequence, event_type, raw_data in event_rows:
        event_data: object = raw_data
        if isinstance(event_data, bytes):
            try:
                event_data = decoder.decompress(event_data).decode("utf-8")
            except (UnicodeDecodeError, zstandard.ZstdError) as exc:
                pytest.fail(
                    f"DSH {event_type} at sequence {sequence} cannot be decoded: "
                    f"{exc}"
                )
        if not isinstance(event_data, str):
            pytest.fail(
                f"DSH {event_type} at sequence {sequence} is not JSON text"
            )
        try:
            parsed = json.loads(event_data)
        except json.JSONDecodeError as exc:
            pytest.fail(
                f"DSH {event_type} at sequence {sequence} is malformed: {exc}"
            )
        if not isinstance(parsed, dict):
            pytest.fail(
                f"DSH {event_type} at sequence {sequence} is not an object"
            )
        events.append((sequence, event_type, parsed))
    return events


def _read_exact_dsh_tool_events(
    result: Mapping[str, object],
) -> list[tuple[int, str, dict[str, Any]]]:
    """Read tool events from the root session named by one live result."""

    identity = result.get("identity")
    if not isinstance(identity, Mapping):
        pytest.fail("live result identity is unavailable for DSH audit")
    resolution_thread_id = identity.get("resolution_thread_id")
    segment_id = identity.get("segment_id")
    if (
        not isinstance(resolution_thread_id, str)
        or not resolution_thread_id.strip()
        or not isinstance(segment_id, str)
        or not segment_id.strip()
    ):
        pytest.fail("live result identity cannot identify a DSH root session")
    return _read_exact_dsh_events(
        resolution_thread_id=resolution_thread_id,
        segment_id=segment_id,
        event_types=("tool/call", "tool/result"),
    )


def _matching_dsh_tool_result(
    events: list[tuple[int, str, dict[str, Any]]],
    *,
    tool_name: str,
) -> tuple[dict[str, Any], dict[str, Any], Mapping[str, object]]:
    """Return one exact tool call and its correlated stored result."""

    calls = [
        (sequence, event)
        for sequence, event_type, event in events
        if event_type == "tool/call" and event.get("name") == tool_name
    ]
    if not calls:
        pytest.fail(f"exact DSH session has no {tool_name} tool call")
    call_sequence, call = calls[0]
    call_id = call.get("callId")
    if not isinstance(call_id, str) or not call_id.strip():
        pytest.fail(f"{tool_name} call at sequence {call_sequence} has no call id")

    matches: list[tuple[int, dict[str, Any], Mapping[str, object]]] = []
    for sequence, event_type, event in events:
        if event_type != "tool/result":
            continue
        message = event.get("message")
        if not isinstance(message, Mapping):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, Mapping):
                continue
            if item.get("toolCallId") == call_id:
                matches.append((sequence, event, item))
    if len(matches) != 1:
        pytest.fail(
            f"{tool_name} call at sequence {call_sequence} has "
            f"{len(matches)} matching tool results"
        )
    return call, matches[0][1], matches[0][2]


@pytest.mark.asyncio
async def test_e2e_context_people_memory_recall_and_calendar(tmp_path: Path) -> None:
    """Sign off context, people, memory, recall, and calendar traversal."""

    _require_live_backend()
    result = await _resolve_live(
        tmp_path,
        "Inspect relevant people, memories, active recall, and calendar context; summarize what is available, explicitly state which requested categories are empty, and submit the grounded result.",
        "e2e-context",
    )
    rendered = json.dumps(result, ensure_ascii=False)
    terminal = result.get("terminal")
    assert isinstance(terminal, Mapping)
    assert terminal.get("status") in {"resolved", "partial"}
    assert "kazusa_" in rendered


@pytest.mark.asyncio
async def test_e2e_memory_create_revise_lifecycle_and_readback(tmp_path: Path) -> None:
    """Sign off durable semantic memory lifecycle and idempotent readback."""

    _require_live_backend()
    from kazusa_ai_chatbot.db._client import get_db

    suffix = uuid4().hex
    global_user_id = f"dsh-e2e-memory-user-{suffix}"
    database = await get_db()
    try:
        result = await _resolve_live(
            tmp_path,
            (
                f"Remember the exact current-user experience marker teal-{suffix} "
                "using current_task provenance, revise the returned memory to "
                f"violet-{suffix}, complete its lifecycle, read that final memory "
                "through the returned opaque reference, and submit the result."
            ),
            "e2e-memory",
            global_user_id=global_user_id,
        )
    finally:
        await database.memory.delete_many(
            {"source_global_user_id": global_user_id}
        )
    rendered = json.dumps(result, ensure_ascii=False)
    for name in (
        "kazusa_remember_information",
        "kazusa_revise_memory",
        "kazusa_change_memory_lifecycle",
        "kazusa_read_memories",
    ):
        assert name in rendered


@pytest.mark.asyncio
async def test_e2e_attached_media_native_web_and_semantic_evidence(tmp_path: Path) -> None:
    """Sign off native web, attached media, and evidence-bound resolution."""

    _require_live_backend()
    from kazusa_ai_chatbot.dsh_tool_gateway.authority import (
        verify_activation_token,
    )
    from kazusa_ai_chatbot.dsh_tool_gateway.contracts import OpaqueReferenceCodec
    from kazusa_ai_chatbot.dsh_tool_gateway.media import (
        issue_attached_media_reference,
        persist_attached_media,
    )
    from kazusa_ai_chatbot.media_inspection.session_cache import (
        clear_session_media,
        put_session_media,
    )

    global_user_id = f"dsh-e2e-media-user-{uuid4().hex}"
    scope = ("debug", "live", global_user_id)
    references = put_session_media(scope, [{
        "media_kind": "image",
        "content_type": "image/png",
        "base64_data": (
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNg"
            "YAAAAAMAASsJTYQAAAAASUVORK5CYII="
        ),
        "source_summary": "one-pixel E2E fixture",
    }])
    persist_attached_media(scope, references)

    def facts(authority: Mapping[str, object]) -> list[str]:
        secret = os.environ["KAZUSA_DSH_TOOL_GATEWAY_SECRET"].encode("utf-8")
        token = authority["semantic_tool_authority"]["token"]  # type: ignore[index]
        verified = verify_activation_token(str(token), secret=secret)
        codec = OpaqueReferenceCodec(secret).with_authority(verified)
        reference = issue_attached_media_reference(
            codec=codec,
            scope=scope,
            cache_ref=str(references[0]["cache_ref"]),
        )
        return [f"Exact attached_media_ref: {reference}"]

    try:
        result = await _resolve_live(
            tmp_path,
            (
                "Use native web_search to find the official Python homepage and "
                "report one returned source URL, then inspect the supplied "
                "attached_media_ref. If media inspection reports a typed "
                "unavailable or unsupported limitation, preserve it as a "
                "remaining need and submit a partial grounded result after "
                "completing the web task; otherwise submit a resolved or "
                "partial grounded result."
            ),
            "e2e-media-web",
            facts_factory=facts,
            global_user_id=global_user_id,
        )
    finally:
        clear_session_media(scope)

    events = _read_exact_dsh_tool_events(result)
    _, web_result, web_result_item = _matching_dsh_tool_result(
        events,
        tool_name="web_search",
    )
    if web_result_item.get("isError") is not False:
        pytest.fail("the correlated web_search result reported an error")
    web_meta = web_result.get("meta")
    if not isinstance(web_meta, Mapping):
        pytest.fail("the correlated web_search result has no metadata")
    web_sources = web_meta.get("sources")
    if not isinstance(web_sources, list) or not any(
        isinstance(source, Mapping)
        and isinstance(source.get("url"), str)
        and bool(source.get("url", "").strip())
        for source in web_sources
    ):
        pytest.fail("the successful web_search result has no source URL")

    _, media_result, media_result_item = _matching_dsh_tool_result(
        events,
        tool_name="kazusa_inspect_attached_media",
    )
    if media_result_item.get("isError") is not False:
        pytest.fail("the correlated attached-media result reported an error")
    media_meta = media_result.get("meta")
    if not isinstance(media_meta, Mapping) or media_meta.get("status") != "ok":
        pytest.fail("the attached-media semantic envelope was not successful")
    media_entities = media_meta.get("entities")
    if not isinstance(media_entities, list):
        pytest.fail("the attached-media result has no entity list")
    media_statuses: list[str] = []
    for entity in media_entities:
        if not isinstance(entity, Mapping):
            pytest.fail("the attached-media entity is not an object")
        status = entity.get("status")
        if isinstance(status, str):
            media_statuses.append(status)
    if not media_statuses:
        pytest.fail("the attached-media result has no entity status")

    evidence = result.get("evidence")
    if not isinstance(evidence, list):
        pytest.fail("the live result has no public evidence list")
    evidence_ids: list[str] = []
    attached_media_ids: list[str] = []
    for receipt in evidence:
        if not isinstance(receipt, Mapping):
            pytest.fail("the public evidence list contains a non-object")
        evidence_id = receipt.get("evidence_id")
        if not isinstance(evidence_id, str) or not evidence_id.strip():
            pytest.fail("the public evidence list contains an invalid receipt")
        evidence_ids.append(evidence_id)
        if receipt.get("source_kind") == "attached_media":
            attached_media_ids.append(evidence_id)
    if not attached_media_ids:
        pytest.fail("the public evidence list has no attached-media receipt")
    if len(evidence_ids) != len(set(evidence_ids)):
        pytest.fail("the public evidence list contains duplicate receipt ids")

    terminal = result.get("terminal")
    if not isinstance(terminal, Mapping):
        pytest.fail("the live result has no terminal projection")
    terminal_status = terminal.get("status")
    if terminal_status not in {"resolved", "partial"}:
        pytest.fail("the live terminal projection has an invalid status")
    if any(status in {"failed", "unsupported"} for status in media_statuses):
        remaining_needs = terminal.get("remaining_needs")
        warnings = terminal.get("warnings")
        has_remaining_needs = (
            isinstance(remaining_needs, list) and bool(remaining_needs)
        ) or (
            isinstance(remaining_needs, str) and bool(remaining_needs.strip())
        )
        has_warnings = (
            isinstance(warnings, list) and bool(warnings)
        ) or (isinstance(warnings, str) and bool(warnings.strip()))
        if terminal_status != "partial" or not (
            has_remaining_needs or has_warnings
        ):
            pytest.fail(
                "a typed attached-media limitation requires a partial "
                "terminal with a remaining need or warning"
            )
